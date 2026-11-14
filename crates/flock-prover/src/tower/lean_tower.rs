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
}

fn proof_bytes(p: &MixedProof) -> usize {
    match p {
        MixedProof::Rs(p) => bincode::serialize(p).map_or(0, |b| b.len()),
        MixedProof::Ag(p) => bincode::serialize(p).map_or(0, |b| b.len()),
    }
}

/// Prove the tower over `pieces` (uniform leaves of one profile, an even
/// count of at least four) and verify its root: the statement checks and
/// the discharge. Panics on any failure; returns the timings.
pub fn prove_lean_tower(cfg: TowerConfig, pieces: &[Piece], env: LeanEnvelope) -> LeanTowerReport {
    let n = pieces.len();
    assert!(n >= 4 && n % 2 == 0, "leaf pairs, at least two of them");
    set_envelope_override(Some((env.nu, env.lanes, env.m)));

    // ---- THE STATEMENT: the commitment root, and its challenges ----
    let words: Vec<[u8; 32]> = pieces
        .iter()
        .map(|p| lean_leaf_word(&p.public(), &p.commit().cap))
        .collect();
    let root_digest = lean_cone_root(&words);
    let ch = shared_challenges_from_root(&root_digest);
    let seed = challenge_seed(&ch);

    // ---- LEAVES + FIRST LEVEL, a pair at a time; leaf 0 stays as the
    // lane's table owner ----
    let leaf = |i: usize| build_lean_leaf(&pieces[i], Some(&ch), &(i as u64).to_le_bytes());
    let t = Instant::now();
    let leaf0 = leaf(0);
    let mut leaf_prove_ms = t.elapsed().as_secs_f64() * 1e3;
    let registry = &leaf0.built.shape.registry;
    let family = FlFamily {
        domain: LEAN_DOMAIN,
        mats: registry
            .boolean_types()
            .iter()
            .map(|ty| (&ty.a_0, &ty.b_0))
            .collect(),
        circs: pieces[0].lincheck_circuits(),
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
    let u = leaf0.union();
    let jp = JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        leaf0.commitment.params.m - LOG_PACKING,
    );
    drop(u);
    let fl_domain: &'static [u8] = b"cas-lean-fl-node-v0";
    let mut fl_prove_ms = 0.0;
    let mut fls = Vec::with_capacity(n / 2);
    for j in 0..n / 2 {
        let t = Instant::now();
        let a: MixedInner;
        let l0 = if j == 0 {
            &leaf0
        } else {
            a = leaf(2 * j);
            &a
        };
        let l1 = leaf(2 * j + 1);
        leaf_prove_ms += t.elapsed().as_secs_f64() * 1e3;
        let fl = build_first_level_core(
            cfg,
            fl_domain,
            &[l0, &l1],
            &family,
            &FlApp::LeanBus { seed },
        );
        fl_prove_ms += fl.t.prove_ms;
        fls.push(fl);
    }
    let app_base = fls[0].stmt_base;
    let claims_base = fls[0].fold_pub_base;
    let block_words = lane_block_words(&fls[0].acc);
    macro_rules! lane {
        ($priors:expr) => {
            ChainLane {
                registry,
                mats: &family.mats,
                circs: &family.circs,
                circuit: &leaf0.built.shape.circuit,
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

    // ---- THE BASE over the last two first-level nodes, then the SPINE ----
    let k = fls.len();
    let mut node_prove_ms = 0.0;
    let base_priors = [&fls[k - 2].acc, &fls[k - 1].acc];
    let base: NodeOut = build_node_outer_app(
        cfg,
        &[&fls[k - 2].lo, &fls[k - 1].lo],
        Some(app_base),
        Some(lane!(&base_priors)),
        None,
    );
    node_prove_ms += base.online.prove_ms;
    let (base, root) = if k == 2 {
        (None, base)
    } else {
        let mut cur: Option<NodeOut> = None;
        for i in (0..k - 2).rev() {
            let prev = cur.as_ref().unwrap_or(&base);
            let prev_lane = prev.lane_acc.clone().expect("the lane rides every level");
            let priors = [&fls[i].acc, &prev_lane];
            let nd = build_node_outer_app(
                cfg,
                &[&fls[i].lo, &prev.lo],
                Some(app_base),
                Some(lane!(&priors)),
                Some(SpineIn {
                    node_child: 1,
                    prior: &prev.block,
                    forge: false,
                }),
            );
            node_prove_ms += nd.online.prove_ms;
            if let Some(c) = &cur {
                assert_eq!(
                    nd.lo.shape.circuit.digest(),
                    c.lo.shape.circuit.digest(),
                    "the spine converged and must stay converged"
                );
            }
            cur = Some(nd);
        }
        (Some(base), cur.expect("k > 2 built a spine node"))
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
    assert_eq!(p[a + 4], p[a + 5], "the shared bus balances over the cone");
    assert_ne!(p[a + 4], F128::ZERO, "a nonzero product");
    let lane_acc = root.lane_acc.as_ref().expect("the lane accumulator");
    let leaf_circuit = &leaf0.built.shape.circuit;
    assert!(lane_acc.discharge(&family.mats), "the lane's boolean class");
    assert!(
        lane_acc.discharge_element(&el_mats),
        "the lane's element class"
    );
    assert!(
        lane_acc.discharge_sigma(&[leaf_circuit]),
        "the lane's sigma"
    );
    assert!(
        lane_acc.discharge_jagged(&[(leaf_circuit.digest(), &jp)]),
        "the lane's jagged entry"
    );
    discharge_outer(
        &fls[0].lo,
        &root.lo,
        base.as_ref(),
        &root.acc,
        &root.block.passenger,
        k >= 4,
    )
    .expect("the main fold and passenger discharge");
    let discharge_ms = t.elapsed().as_secs_f64() * 1e3;
    set_envelope_override(None);
    LeanTowerReport {
        leaves: n,
        leaf_prove_ms,
        fl_prove_ms,
        node_prove_ms,
        root_verify_ms: root.online.verify_ms,
        discharge_ms,
        root_proof_bytes: proof_bytes(&root.lo.proof),
    }
}

/// Probe: eight leaves, four first-level nodes, a base and two spine
/// nodes, on test data.
#[test]
#[ignore] // Heavier — run alone with `-- --ignored`.
#[cfg(not(target_arch = "aarch64"))]
fn lean_tower_spine() {
    use crate::lean_rows::{LeafSlot, LeanCircuit, Profile, Rows, leaf};
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
    let pieces: Vec<Piece> = profile
        .deal(&per_table, 8)
        .iter()
        .map(|d| {
            let slots: Vec<LeafSlot<'_>> = d.iter().map(|x| x.map(|ch| (&c, ch))).collect();
            leaf(&c, &profile, &slots).unwrap()
        })
        .collect();
    for (i, p) in pieces.iter().enumerate() {
        eprintln!("leaf {i}: {} rows", p.rows());
    }
    let r = prove_lean_tower(
        super::test_config(),
        &pieces,
        LeanEnvelope {
            nu: 15,
            lanes: 40,
            m: 30,
        },
    );
    eprintln!("{r:?}");
}
