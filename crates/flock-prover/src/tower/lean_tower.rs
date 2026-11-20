//! The Lean cone's recursion tower: uniform Lean leaves under one shared
//! bus, paired into first-level nodes, those folded by a base node and a
//! spine (each spine node prepends the next-earlier first-level node), and
//! the root checked against the cone's statement.
//!
//! The statement is the commitment root ([`lean_cone_root`] over each
//! leaf's [`lean_leaf_word`]): the shared challenges are drawn from it
//! ([`shared_challenges_from_root`]), every node carries the challenges'
//! seed, the commitment digest and the products of the open bus sides up
//! the tree, and the root checks all three. With the discharge of the
//! accumulated claims against the leaf, first-level and node tables, that
//! is the cone's whole verification.

use std::time::Instant;

use flock_core::{
    aggregate::ElementMatrices,
    pcs::{LOG_PACKING, jagged::JaggedParams},
};

use super::{
    ChainLane, F128, FlApp, FlFamily, LaneFamily, MixedInner, MixedProof, SpineIn, TowerConfig,
    build_first_level_core, build_node_outer_app,
    driver::discharge_outer,
    envelope::{lane_block_words, set_envelope_override},
    lean_leaf::{
        LEAN_DOMAIN, build_lean_leaf, challenge_seed, f128_digest, lean_cone_root, lean_leaf_word,
    },
    node::NodeOut,
};
use crate::lean_rows::{Piece, shared_challenges_from_root};

/// The envelope a Lean tower fits: `nu*`, `lanes*` and the dense floor
/// `m*` every outer shares.
#[derive(Clone, Copy, Debug)]
pub struct LeanEnvelope {
    pub nu: usize,
    pub lanes: usize,
    pub m: usize,
}

/// What a Lean tower run measured.
#[derive(Clone, Debug)]
pub struct LeanTowerReport {
    pub leaves: usize,
    /// Leaves, first-level nodes and nodes proved, in ms (the sums).
    pub leaf_prove_ms: f64,
    pub fl_prove_ms: f64,
    pub node_prove_ms: f64,
    /// The root proof's own verification, and the root-side discharge.
    pub root_verify_ms: f64,
    pub discharge_ms: f64,
    pub root_proof_bytes: usize,
    /// The cone's shared bus balances (Πsend = Πrecv at the root). Only a
    /// whole cone balances: a probe over some of its leaves does not.
    pub balanced: bool,
}

fn proof_bytes(p: &MixedProof) -> usize {
    match p {
        MixedProof::Rs(p) => bincode::serialize(p).map_or(0, |b| b.len()),
        MixedProof::Ag(p) => bincode::serialize(p).map_or(0, |b| b.len()),
    }
}

/// Prove the tower over `n` leaves built by `piece` (uniform leaves of
/// one profile, an even count of at least four; each is built twice, for
/// its word and for its proof) and verify its root: the statement checks and
/// the discharge. Panics on any failure; returns the timings.
pub fn prove_lean_tower(
    cfg: TowerConfig,
    n: usize,
    mut piece: impl FnMut(usize) -> Piece,
    statement: &LeanStatement,
    env: LeanEnvelope,
) -> LeanTowerReport {
    let words: Vec<[u8; 32]> = (0..n).map(|i| leaf_word(&piece(i))).collect();
    prove_lean_tower_streaming(cfg, &words, piece, statement, env, |line| {
        eprintln!("{line}")
    })
}

/// What the root's statement names besides the commitment root: every
/// leaf slot's chunk word, all distinct (so no two rows share an id), and
/// the keys of the declarations checked. The verifier receives each on
/// the shared bus ([`crate::lean_rows::statement_product`]).
#[derive(Clone, Debug, Default)]
pub struct LeanStatement {
    pub chunk_words: Vec<u64>,
    pub keys: Vec<[F128; 2]>,
    /// The constants the rules read from the verifier, per reading row:
    /// the number and its key ([`crate::lean_rows::prims_product`]). One
    /// key per number: the keys of `Nat` and of `String`.
    pub prims: Vec<(u64, [F128; 2])>,
    /// The steps of arithmetic on literals the rules read from the
    /// verifier, per reading row ([`crate::lean_rows::arith_product`]).
    pub arith: Vec<crate::lean_rows::Arith>,
    /// What the declarations are ([`crate::lean_rows::facts_product`]).
    pub facts: Vec<crate::lean_rows::Tuple>,
}

/// A leaf's word in the commitment tree: its publics digest chained with
/// its commitment cap (the first pass of a run: commit only).
pub fn leaf_word(p: &Piece) -> [u8; 32] {
    lean_leaf_word(&p.public(), &p.commit().cap)
}

/// [`prove_lean_tower`] holding one leaf pair at a time: `words` are the
/// leaves' words from a first pass ([`leaf_word`]) and `piece(i)` builds
/// leaf `i`: leaf 0 first (its shape keeps the lane's tables), then in
/// DECREASING order from `n - 1` down to 0 (the spine folds from the tail). `progress` hears a line per proof.
pub fn prove_lean_tower_streaming(
    cfg: TowerConfig,
    words: &[[u8; 32]],
    piece: impl FnMut(usize) -> Piece,
    statement: &LeanStatement,
    env: LeanEnvelope,
    mut progress: impl FnMut(&str),
) -> LeanTowerReport {
    let s = prove_lean_segment(cfg, words, 0, None, piece, env, &mut progress);
    if !s.key_claims.is_empty() {
        let b = prove_root_key_batch(s.key_tables.clone(), &s.key_claims);
        progress(&format!(
            "root key: {} tables, {} sigma claims; key {:.0} ms, prove {:.0} ms, proof {} KiB, verify {:.1} ms",
            b.tables,
            b.claims,
            b.key_ms,
            b.prove_ms,
            b.proof_bytes / 1024,
            b.verify_ms
        ));
    }
    let balanced = match segments_balanced(&s.root, &[s.claim()], statement) {
        Ok(b) => b,
        Err(e) => {
            progress(&format!("the statement is refused: {e}"));
            false
        }
    };
    LeanTowerReport {
        balanced,
        ..s.report
    }
}

/// What one segment's root proof states: the commitment root of its own
/// leaves' words, the seed of the challenges its bus used, and its two
/// bus products.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentClaim {
    pub root: [u8; 32],
    pub seed: [u8; 32],
    pub send: F128,
    pub recv: F128,
}

/// A proved segment: its measurements (`balanced` is not its to say) and
/// its claim.
#[derive(Clone, Debug)]
pub struct LeanSegmentOut {
    pub report: LeanTowerReport,
    pub root: [u8; 32],
    pub seed: [u8; 32],
    pub send: F128,
    pub recv: F128,
    /// With a root key: the sigma claims the segment left to it, and the
    /// structure tables (with sigma) of the circuits they name.
    pub key_claims: Vec<([u8; 32], flock_core::matrix_fold::MatrixClaim)>,
    pub key_tables: Vec<flock_core::circuit::StructureTable>,
    /// The segment's root proof and the VK it is checked against (a
    /// spine root: at least three leaf pairs), checked here once from
    /// bytes as a verifier would ([`super::lean_verify`]).
    pub saved: Option<(super::lean_verify::LeanVk, super::lean_verify::LeanSegmentProof)>,
}

impl LeanSegmentOut {
    pub fn claim(&self) -> SegmentClaim {
        SegmentClaim {
            root: self.root,
            seed: self.seed,
            send: self.send,
            recv: self.recv,
        }
    }
}

/// The root the challenges of a cone proved in segments are drawn from:
/// it commits to every segment's commitment root, so to every leaf's word.
pub fn lean_segments_root(roots: &[[u8; 32]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key("cas-lean cone segments v0");
    h.update(&(roots.len() as u64).to_le_bytes());
    for r in roots {
        h.update(r);
    }
    *h.finalize().as_bytes()
}

/// A segment's own commitment root, from its leaves' words.
pub fn lean_segment_root(words: &[[u8; 32]]) -> [u8; 32] {
    lean_cone_root(words)
}

/// The verifier's last step over a cone's segment proofs (each already
/// verified): every segment used the challenges of `challenge_root`
/// (which for more than one segment must be [`lean_segments_root`] of
/// their roots, in order), and over all segments the shared bus balances
/// against the statement: Πsend = Πrecv · statement.
pub fn segments_balanced(
    challenge_root: &[u8; 32],
    segments: &[SegmentClaim],
    statement: &LeanStatement,
) -> Result<bool, String> {
    if segments.len() > 1 {
        let roots: Vec<[u8; 32]> = segments.iter().map(|s| s.root).collect();
        if lean_segments_root(&roots) != *challenge_root {
            return Err("the challenge root is not the segments'".into());
        }
    } else if segments.iter().any(|s| s.root != *challenge_root) {
        return Err("the challenge root is not the segment's".into());
    }
    let ch = shared_challenges_from_root(challenge_root);
    let seed = challenge_seed(&ch);
    if segments.iter().any(|s| s.seed != seed) {
        return Err("a segment used other challenges".into());
    }
    let st = crate::lean_rows::statement_product(&ch, &statement.chunk_words, &statement.keys)?
        * crate::lean_rows::facts_product(&ch, &statement.facts);
    let prims = crate::lean_rows::prims_product(&ch, &statement.prims)?;
    let (arith, _) = crate::lean_rows::arith_product(&ch, &statement.arith)?;
    let send = segments.iter().fold(prims * arith, |a, s| a * s.send);
    let recv = segments.iter().fold(F128::ONE, |a, s| a * s.recv);
    Ok(send != F128::ZERO && send == recv * st)
}

/// One segment of a cone's tower: leaves `first .. first + words.len()`
/// (`words` are theirs), under the challenges of `challenge_root` (None:
/// the segment is the whole cone, and its own root gives them). A cone
/// too long for one run is proved a segment at a time, each a proof of
/// its own; [`segments_balanced`] joins them.
pub fn prove_lean_segment(
    cfg: TowerConfig,
    words: &[[u8; 32]],
    first: usize,
    challenge_root: Option<[u8; 32]>,
    piece: impl FnMut(usize) -> Piece,
    env: LeanEnvelope,
    progress: impl FnMut(&str),
) -> LeanSegmentOut {
    prove_lean_segment_keyed(cfg, words, first, challenge_root, piece, env, progress, None)
}

/// [`prove_lean_segment`], writing `params` (the leaves' profile and
/// widths; its depth class is set here) into the saved key, so the key can
/// be made again from them ([`regenerate_lean_vk`]).
#[allow(clippy::too_many_arguments)]
pub fn prove_lean_segment_keyed(
    cfg: TowerConfig,
    words: &[[u8; 32]],
    first: usize,
    challenge_root: Option<[u8; 32]>,
    mut piece: impl FnMut(usize) -> Piece,
    env: LeanEnvelope,
    mut progress: impl FnMut(&str),
    params: Option<super::lean_verify::LeanKeyParams>,
) -> LeanSegmentOut {
    let mut piece = |i: usize| piece(first + i);
    let tag = |i: usize| ((first + i) as u64).to_le_bytes();
    let n = words.len();
    assert!(n >= 4 && n % 2 == 0, "leaf pairs, at least two of them");
    set_envelope_override(Some((env.nu, env.lanes, env.m)));
    // Two leaves, a node and their proofs share 15 GB: cap what the
    // prover's buffer pool keeps between and within proofs.
    let pool_gb: usize = std::env::var("LEAN_POOL_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let pool_bytes = std::env::var("LEAN_POOL_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|m| m.max(1) << 20)
        .unwrap_or(pool_gb << 30);
    flock_core::scratch::set_byte_budget(Some(pool_bytes));

    // ---- THE STATEMENT: the commitment root, and its challenges ----
    let root_digest = lean_cone_root(words);
    let ch = shared_challenges_from_root(&challenge_root.unwrap_or(root_digest));
    let seed = challenge_seed(&ch);

    // ---- LEAF 0, the lane's table owner ----
    let t = Instant::now();
    let piece0 = piece(0);
    let leaf0 = build_lean_leaf(&piece0, Some(&ch), &tag(0));
    assert_eq!(
        lean_leaf_word(&leaf0.built.witness.public, &leaf0.commitment.cap),
        words[0],
        "leaf 0's word from the first pass"
    );
    let mut leaf_prove_ms = t.elapsed().as_secs_f64() * 1e3;
    progress(&format!("leaf 0: {leaf_prove_ms:.0} ms"));
    // Leaf 0's shape owns the lane's tables; its witness and proof drop
    // here, and first-level node 0 proves it again at the end.
    let shape0 = leaf0.built.shape.clone();
    let gate0 = piece0
        .tower_gate()
        .expect("a leaf in the tower's posture")
        .clone();
    let u = leaf0.union();
    let jp = JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        leaf0.commitment.params.m - LOG_PACKING,
    );
    drop(u);
    let leaf_full_height = leaf0.full_height;
    let leaf_m = leaf0.commitment.params.m;
    drop(leaf0);
    drop(piece0);
    progress("leaf 0's shape kept, its proof dropped");
    let registry = &shape0.registry;
    let family = FlFamily {
        domain: LEAN_DOMAIN,
        mats: registry
            .boolean_types()
            .iter()
            .map(|ty| (&ty.a_0, &ty.b_0))
            .collect(),
        circs: vec![gate0.r1cs().csc_lincheck_circuit()],
        lane_words: None,
        shared: None,
        env_nu: Some(env.nu),
    };
    let el_types: Vec<_> = registry
        .element_types()
        .iter()
        .map(|s| s.element_type().expect("an element slot's table"))
        .collect();
    let el_mats: Vec<ElementMatrices<'_>> = el_types.iter().map(|t| (t.a_0(), t.b_0())).collect();
    let fl_domain: &'static [u8] = b"cas-lean-fl-node-v0";
    let mut fl_prove_ms = 0.0;
    let mut node_prove_ms = 0.0;
    // First-level node j over leaves 2j, 2j+1 (leaf 0 is held).
    let mut fl = |j: usize, leaf_prove_ms: &mut f64, progress: &mut dyn FnMut(&str)| {
        let t = Instant::now();
        let mut build = |i: usize| {
            let p = piece(i);
            let l = build_lean_leaf(&p, Some(&ch), &tag(i));
            assert_eq!(
                lean_leaf_word(&l.built.witness.public, &l.commitment.cap),
                words[i],
                "leaf {i}'s word from the first pass"
            );
            l
        };
        let l1 = build(2 * j + 1);
        progress(&format!("leaf {}", first + 2 * j + 1));
        let l0: MixedInner = if j == 0 { build(0) } else { build(2 * j) };
        progress(&format!("leaf {}", first + 2 * j));
        *leaf_prove_ms += t.elapsed().as_secs_f64() * 1e3;
        let f = build_first_level_core(
            cfg,
            fl_domain,
            &[&l0, &l1],
            &family,
            &FlApp::LeanBus { seed },
        );
        drop((l0, l1));
        let mut f = f;
        f.lo.shape.drop_program();
        release_pool();
        progress(&format!(
            "first-level node {j}: prove {:.0} ms",
            f.t.prove_ms
        ));
        f
    };
    let k = n / 2;
    let fl_last = fl(k - 1, &mut leaf_prove_ms, &mut progress);
    let fl_prev = fl(k - 2, &mut leaf_prove_ms, &mut progress);
    fl_prove_ms += fl_last.t.prove_ms + fl_prev.t.prove_ms;
    let app_base = fl_last.stmt_base;
    let claims_base = fl_last.fold_pub_base;
    let block_words = lane_block_words(&fl_last.acc);
    macro_rules! lane {
        ($priors:expr) => {
            ChainLane {
                registry,
                mats: &family.mats,
                circs: &family.circs,
                circuit: &shape0.circuit,
                params: &jp,
                priors: $priors,
                claims_base,
                family: LaneFamily::Lean {
                    block_words,
                    el_mats: &el_mats,
                },
            }
        };
    }

    // ---- THE BASE over the last two first-level nodes, then the SPINE,
    // one first-level node at a time ----
    let base_priors = [&fl_prev.acc, &fl_last.acc];
    let mut base: NodeOut = build_node_outer_app(
        cfg,
        &[&fl_prev.lo, &fl_last.lo],
        Some(app_base),
        Some(lane!(&base_priors)),
        None,
    );
    release_pool();
    // Held to the root for the discharge; its program is spent.
    base.lo.shape.drop_program();
    node_prove_ms += base.online.prove_ms;
    progress(&format!("base node: prove {:.0} ms", base.online.prove_ms));
    let (fl0, base, root) = if k == 2 {
        drop(fl_last);
        (fl_prev, None, base)
    } else {
        drop((fl_last, fl_prev));
        let mut cur: Option<NodeOut> = None;
        let mut fl0 = None;
        for i in (0..k - 2).rev() {
            let f = fl(i, &mut leaf_prove_ms, &mut progress);
            fl_prove_ms += f.t.prove_ms;
            let prev = cur.as_ref().unwrap_or(&base);
            let prev_lane = prev.lane_acc.clone().expect("the lane rides every level");
            let priors = [&f.acc, &prev_lane];
            let nd = build_node_outer_app(
                cfg,
                &[&f.lo, &prev.lo],
                Some(app_base),
                Some(lane!(&priors)),
                Some(SpineIn {
                    node_child: 1,
                    prior: &prev.block,
                    forge: false,
                }),
            );
            release_pool();
            let mut nd = nd;
            nd.lo.shape.drop_program();
            node_prove_ms += nd.online.prove_ms;
            progress(&format!(
                "spine node {i}: prove {:.0} ms",
                nd.online.prove_ms
            ));
            if let Some(c) = &cur {
                assert_eq!(
                    nd.lo.shape.circuit.digest(),
                    c.lo.shape.circuit.digest(),
                    "the spine converged and must stay converged"
                );
            }
            cur = Some(nd);
            if i == 0 {
                fl0 = Some(f);
            }
        }
        (
            fl0.expect("first-level node 0"),
            Some(base),
            cur.expect("k > 2 built a spine node"),
        )
    };

    // ---- THE ROOT: the statement from its publics, then the discharge ----
    let t = Instant::now();
    let p = &root.lo.public;
    let a = app_base;
    assert_eq!(
        f128_digest([p[a], p[a + 1]]),
        root_digest,
        "the commitment root"
    );
    assert_eq!(
        f128_digest([p[a + 2], p[a + 3]]),
        seed,
        "the challenges' seed"
    );
    let (send, recv) = (p[a + 4], p[a + 5]);
    assert_ne!(p[a + 4], F128::ZERO, "a nonzero product");
    let lane_acc = root.lane_acc.as_ref().expect("the lane accumulator");
    let leaf_circuit = &shape0.circuit;
    // THE ROOT KEY (LEAN_ROOT_KEY=1): the structure-table (sigma) claims
    // are not read off the wiring here; the segment hands them out, with
    // its circuits' tables, and the join proves them all in one batch
    // against a committed key ([`prove_root_key_batch`]).
    let keyed = std::env::var_os("LEAN_ROOT_KEY").is_some();
    // Wake the thread pool first: an idle pool's first call costs ~20 ms,
    // which is not the check's.
    rayon::broadcast(|_| ());
    let tm = Instant::now();
    assert!(lane_acc.discharge(&family.mats), "the lane's boolean class");
    let t_bool = tm.elapsed().as_secs_f64() * 1e3;
    let tm = Instant::now();
    assert!(
        lane_acc.discharge_element(&el_mats),
        "the lane's element class"
    );
    let t_el = tm.elapsed().as_secs_f64() * 1e3;
    let tm = Instant::now();
    assert!(
        keyed || lane_acc.discharge_sigma(&[leaf_circuit]),
        "the lane's sigma"
    );
    let t_sigma = tm.elapsed().as_secs_f64() * 1e3;
    let tm = Instant::now();
    assert!(
        lane_acc.discharge_jagged(&[(leaf_circuit.digest(), &jp)]),
        "the lane's jagged entry"
    );
    let t_jagged = tm.elapsed().as_secs_f64() * 1e3;
    let tm = Instant::now();
    super::driver::discharge_outer_with(
        &fl0.lo,
        &root.lo,
        base.as_ref(),
        &root.acc,
        &root.block.passenger,
        k >= 4,
        !keyed,
    )
    .expect("the main fold and passenger discharge");
    let t_outer = tm.elapsed().as_secs_f64() * 1e3;
    let discharge_ms = t.elapsed().as_secs_f64() * 1e3;
    progress(&format!(
        "root-side checks{}: boolean {t_bool:.1} ms, element {t_el:.1} ms, sigma {t_sigma:.1} ms, jagged {t_jagged:.1} ms, main fold and passenger {t_outer:.1} ms",
        if keyed {
            " (sigma left to the key)"
        } else {
            ""
        }
    ));
    let mut key_claims = Vec::new();
    let mut key_tables = Vec::new();
    if keyed {
        use flock_core::circuit::{Circuit, StructureTable};
        let lane_digest = leaf_circuit.digest();
        let mut circs: Vec<&Circuit> =
            vec![leaf_circuit, &fl0.lo.shape.circuit, &root.lo.shape.circuit];
        if let Some(b) = base.as_ref() {
            circs.push(&b.lo.shape.circuit);
        }
        print!("{}", super::root_key::describe(&circs));
        // Every sigma claim must name one of these circuits: the lane's the
        // leaf's, the main fold's the FL's, the root's or the base's.
        let known = |d: &[u8; 32]| circs.iter().any(|c| c.digest() == *d);
        for (d, c) in &lane_acc.sigma {
            assert!(*d == lane_digest, "the lane's sigma names the leaf");
            key_claims.push((*d, c.clone()));
        }
        for (d, c) in &root.acc.sigma {
            assert!(known(d), "the main fold's sigma names a known circuit");
            key_claims.push((*d, c.clone()));
        }
        if let (Some(b), Some((_, c))) = (base.as_ref(), root.block.passenger.first()) {
            if super::node::entry_live(c) {
                key_claims.push((b.lo.shape.circuit.digest(), c.clone()));
            }
        }
        let tt = Instant::now();
        let mut seen: Vec<[u8; 32]> = Vec::new();
        for c in &circs {
            if !seen.contains(&c.digest()) {
                seen.push(c.digest());
                key_tables.push(StructureTable::from_circuit(c, true));
            }
        }
        progress(&format!(
            "root key part: {} sigma claims over {} tables ({:.0} ms to extract)",
            key_claims.len(),
            key_tables.len(),
            tt.elapsed().as_secs_f64() * 1e3
        ));
    }
    // ---- THE SAVED PROOF: the root and its VK, checked from bytes ----
    let saved = base.as_ref().map(|b| {
        use super::lean_verify::{LeanSegmentProof, LeanVk, OuterVk, check_segment};
        use flock_core::circuit::recipe::CircuitRecipe;
        let tv = Instant::now();
        let outer = |lo: &super::query::LeafOuter| OuterVk {
            circuit: CircuitRecipe::of_with(
                &lo.shape.circuit,
                &super::lean_verify::outer_namer(lo.shape.registry.nu()),
            ),
            pcs: lo.pcs.clone(),
            m: lo.commitment.params.m,
        };
        // The leaf's bus challenges come from this cone's commitment root:
        // they are the cone's, not the key's (the digest leaves the bus out,
        // and the first-level circuit checks the bus under the seed the
        // root publishes), so the key holds the bus without them and is
        // the same for every cone at its parameters.
        let mut leaf_recipe = CircuitRecipe::of(leaf_circuit);
        if let Some(bus) = leaf_recipe.bus.as_mut() {
            bus.shared = None;
        }
        let vk = LeanVk {
            envelope: (env.nu, env.lanes, env.m),
            leaf: leaf_recipe,
            leaf_full_height,
            leaf_m,
            fl: outer(&fl0.lo),
            base: outer(&b.lo),
            steady: outer(&root.lo),
            layout: super::verify::AccLayout::from_parts(
                &super::envelope::envelope_shape_with_lane(block_words),
                claims_base,
                &root,
            ),
            app_base: a,
            params: params.map(|p| super::lean_verify::LeanKeyParams {
                leaves: if n == 6 { 6 } else { 8 },
                body_extra: super::envelope::lean_body_extra(),
                ..p
            }),
        };
        let proof = LeanSegmentProof {
            vk: 0,
            first_leaf: first as u64,
            leaves: n as u64,
            public: root.lo.public.clone(),
            proof: root.lo.proof.clone(),
            commitment: root.lo.commitment.clone(),
        };
        for (what, r, c) in [
            ("leaf", &vk.leaf, leaf_circuit),
            ("first-level", &vk.fl.circuit, &fl0.lo.shape.circuit),
            ("base", &vk.base.circuit, &b.lo.shape.circuit),
            ("steady", &vk.steady.circuit, &root.lo.shape.circuit),
        ] {
            let d = r
                .rebuild_with(&super::lean_verify::resolve)
                .unwrap_or_else(|e| panic!("the {what} recipe does not rebuild: {e:?}"))
                .digest();
            assert_eq!(d, c.digest(), "the {what} recipe rebuilds another circuit");
        }
        let vk_bytes = bincode::serialize(&vk).expect("a VK serializes");
        let proof_bytes = bincode::serialize(&proof).expect("a proof serializes");
        let vk2: LeanVk = bincode::deserialize(&vk_bytes).expect("a VK decodes");
        let proof2: LeanSegmentProof = bincode::deserialize(&proof_bytes).expect("a proof decodes");
        let c = check_segment(&vk2, &proof2).expect("the saved segment verifies from bytes");
        assert!(
            c.root == root_digest && c.seed == seed && c.send == send && c.recv == recv,
            "the saved segment claims what the prover claimed"
        );
        progress(&format!(
            "saved: VK blake3:{} ({} KiB), proof {} KiB, checked from bytes in {:.0} ms",
            hex(&vk2.fingerprint()),
            vk_bytes.len() / 1024,
            proof_bytes.len() / 1024,
            tv.elapsed().as_secs_f64() * 1e3
        ));
        (vk2, proof2)
    });
    set_envelope_override(None);
    flock_core::scratch::set_byte_budget(None);
    LeanSegmentOut {
        report: LeanTowerReport {
            leaves: n,
            leaf_prove_ms,
            fl_prove_ms,
            node_prove_ms,
            root_verify_ms: root.online.verify_ms,
            discharge_ms,
            root_proof_bytes: proof_bytes(&root.lo.proof),
            balanced: false,
        },
        root: root_digest,
        seed,
        send,
        recv,
        key_claims,
        key_tables,
        saved,
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Probe: eight leaves, four first-level nodes, a base and two spine
/// nodes, on test data.
#[test]
#[ignore] // Heavier — run alone with `-- --ignored`.
#[cfg(not(target_arch = "aarch64"))]
fn lean_tower_spine() {
    use crate::lean_rows::{
        LeafSlot, LeanCircuit, Profile, Rows, leaf, leaf_prims, leaf_statement,
    };
    let small = include_str!("../../tests/data/lean_small.rows");
    let rows = Rows::merge((0..9000).map(|_| Rows::parse(small).unwrap()).collect());
    let c = LeanCircuit::build(&rows).unwrap();
    let mut per_table: Vec<Vec<usize>> =
        vec![Vec::new(); c.chunks().iter().map(|x| x.0 + 1).max().unwrap()];
    for (s, ch, _) in c.chunks() {
        per_table[s].push(ch);
    }
    let n: Vec<usize> = per_table.iter().map(|t| t.len()).collect();
    let profile = Profile::even(c.nu, &n, 8);
    let deal = profile.deal(&per_table, 8);
    let mut st = LeanStatement::default();
    for (i, d) in deal.iter().enumerate() {
        let slots: Vec<LeafSlot<'_>> = d
            .iter()
            .zip(&profile.slots)
            .map(|(x, &s)| x.map(|ch| c.slot(s, ch)))
            .collect();
        let (w, k) = leaf_statement(&c, &profile, &slots, i);
        st.chunk_words.extend(w);
        st.keys.extend(k);
        st.prims.extend(leaf_prims(&c, &profile, &slots));
    }
    let r = prove_lean_tower(
        super::test_config(),
        8,
        |i| {
            let slots: Vec<LeafSlot<'_>> = deal[i]
                .iter()
                .zip(&profile.slots)
                .map(|(x, &s)| x.map(|ch| c.slot(s, ch)))
                .collect();
            leaf(&c, &profile, &slots, i).unwrap()
        },
        &st,
        LeanEnvelope {
            nu: 15,
            lanes: 40,
            m: 30,
        },
    );
    eprintln!("{r:?}");
    assert!(r.balanced, "the shared bus balances over the whole cone");

    // The same cone as two segments of four leaves, each a proof of its
    // own: together they balance, and not with a claim changed or the
    // challenges of another root.
    let mut piece = |i: usize| {
        let slots: Vec<LeafSlot<'_>> = deal[i]
            .iter()
            .zip(&profile.slots)
            .map(|(x, &s)| x.map(|ch| c.slot(s, ch)))
            .collect();
        leaf(&c, &profile, &slots, i).unwrap()
    };
    let words: Vec<[u8; 32]> = (0..8).map(|i| leaf_word(&piece(i))).collect();
    let roots = [
        lean_segment_root(&words[..4]),
        lean_segment_root(&words[4..]),
    ];
    let all = lean_segments_root(&roots);
    let env = LeanEnvelope {
        nu: 15,
        lanes: 40,
        m: 30,
    };
    let claims: Vec<SegmentClaim> = (0..2)
        .map(|s| {
            let o = prove_lean_segment(
                super::test_config(),
                &words[4 * s..4 * s + 4],
                4 * s,
                Some(all),
                &mut piece,
                env,
                |line| eprintln!("segment {s}: {line}"),
            );
            assert_eq!(o.root, roots[s]);
            o.claim()
        })
        .collect();
    assert_eq!(segments_balanced(&all, &claims, &st), Ok(true));
    assert_eq!(
        segments_balanced(&all, &claims[..1], &st).ok(),
        None,
        "a segment missing"
    );
    let mut swapped = claims.clone();
    swapped.swap(0, 1);
    assert!(segments_balanced(&all, &swapped, &st).is_err());
    let mut bad = claims.clone();
    bad[1].send = bad[1].send + F128::ONE;
    assert_eq!(segments_balanced(&all, &bad, &st), Ok(false));
}

/// Release the prover's scratch pool after a node's proof.
///
/// The pool keeps a proof's big buffers for the next proof. Leaves reuse
/// each other's, and a first-level node reuses its leaves' (cleared before
/// it, the node's prove grows the pool past 15 GB: every buffer size it
/// takes once stays pooled). But a node's buffers, kept, sit under the next
/// two leaves' proves and push them out of memory, so the pool is released
/// after each node and not after a leaf.
fn release_pool() {
    if std::env::var_os("LEAN_KEEP_POOL").is_none() {
        flock_core::scratch::clear();
    }
}

/// What the join's root-key batch measured.
#[derive(Clone, Debug)]
pub struct RootKeyBatch {
    pub tables: usize,
    pub claims: usize,
    pub key_ms: f64,
    pub prove_ms: f64,
    pub proof_bytes: usize,
    pub verify_ms: f64,
}

/// THE JOIN'S ONE BATCH: every segment's sigma claims, proved together by
/// openings against one key over the circuits' structure tables, then
/// verified against the tables without sigma (the verification key) — the
/// verifier never reads a wiring. Tables are deduplicated by digest;
/// claims on one table are folded into one. Panics if a claim is false or
/// names no table, or if a tampered claim would pass.
pub fn prove_root_key_batch(
    tables: Vec<flock_core::circuit::StructureTable>,
    claims: &[([u8; 32], flock_core::matrix_fold::MatrixClaim)],
) -> RootKeyBatch {
    use super::root_key::RootKeyTables;
    let tk = Instant::now();
    let keys = RootKeyTables::from_tables(tables).expect("the root key");
    let key_ms = tk.elapsed().as_secs_f64() * 1e3;
    let indexed: Vec<(usize, flock_core::matrix_fold::MatrixClaim)> = claims
        .iter()
        .map(|(d, c)| (keys.index(d).expect("a keyed circuit"), c.clone()))
        .collect();
    let tp = Instant::now();
    let proof = keys.prove(&indexed).expect("the root key proves");
    let prove_ms = tp.elapsed().as_secs_f64() * 1e3;
    let proof_bytes = bincode::serialize(&proof)
        .expect("a proof serializes")
        .len();
    let vks = keys.vks();
    let tv = Instant::now();
    assert!(
        keys.key.verify(&vks, &indexed, &proof),
        "the root key verifies"
    );
    let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
    let mut bad = indexed.clone();
    bad[0].1.value += F128::ONE;
    assert!(
        !keys.key.verify(&vks, &bad, &proof),
        "a tampered claim fails"
    );
    RootKeyBatch {
        tables: keys.key.tables.len(),
        claims: indexed.len(),
        key_ms,
        prove_ms,
        proof_bytes,
        verify_ms,
    }
}

/// A segment's verification key made again from the prover's code and the
/// key's parameters alone: a segment of `params.leaves` empty leaves (every
/// slot padding) is proved at the envelope, and its saved key returned.
/// Every circuit of a Lean tower is the same whatever its leaves hold, so a
/// key equal to it (one fingerprint) holds the circuits this code makes.
/// It costs a short proof of that many leaves.
pub fn regenerate_lean_vk(
    envelope: (usize, usize, usize),
    params: &super::lean_verify::LeanKeyParams,
    mut progress: impl FnMut(&str),
) -> Result<super::lean_verify::LeanVk, String> {
    use crate::lean_rows::{LeanCircuit, Profile, Rows, Side, leaf, set_mult_bits};
    let p = params;
    if p.leaves != 6 && p.leaves != 8 {
        return Err(format!("a key's depth class is 6 or 8 leaves, not {}", p.leaves));
    }
    if p.slots.is_empty() {
        return Err("a key's leaf has no slots".into());
    }
    set_mult_bits(Some(p.mult_bits.clone()));
    super::envelope::set_lean_body_extra(Some(p.body_extra));
    let out = (|| {
        let (proto, _) = LeanCircuit::build_side(
            &Rows::default(),
            &Rows::default(),
            Side::Batch,
            1 << 20,
            &std::collections::HashMap::new(),
        )?;
        let n_tables = proto.mult_bits().len();
        if let Some(&s) = p.slots.iter().find(|&&s| s >= n_tables) {
            return Err(format!("a slot names table {s} of {n_tables}"));
        }
        let profile = Profile {
            nu: p.nu,
            slots: p.slots.clone(),
        };
        let empty = vec![None; p.slots.len()];
        let piece = |i: usize| leaf(&proto, &profile, &empty, i);
        let n = p.leaves as usize;
        let mut words = Vec::with_capacity(n);
        for i in 0..n {
            words.push(leaf_word(&piece(i)?));
            progress(&format!("empty leaf {i} committed"));
        }
        let o = prove_lean_segment_keyed(
            TowerConfig::Chain128,
            &words,
            0,
            None,
            |i| piece(i).expect("an empty leaf builds"),
            LeanEnvelope {
                nu: envelope.0,
                lanes: envelope.1,
                m: envelope.2,
            },
            &mut progress,
            Some(p.clone()),
        );
        o.saved
            .map(|s| s.0)
            .ok_or_else(|| "the empty segment saved no key".to_string())
    })();
    set_mult_bits(None);
    super::envelope::set_lean_body_extra(None);
    out
}
