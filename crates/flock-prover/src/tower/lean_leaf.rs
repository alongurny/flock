//! Lean pieces (`crate::lean_rows`) as recursion-tower leaves: a piece in
//! the tower's posture (one boolean slot, the chained transcript),
//! deferred-verified, so [`super::ChildTape`] can replay its verifier
//! in-circuit.

#[cfg(test)]
use {
    crate::{
        lean_rows::{LeanCircuit, Rows},
        tower::{
            ChildSlots, ChildTape, SLOT_WORDS, ShapeBuilder, Wire, check_child_region,
            emit_child_region,
        },
    },
    std::any::Any,
};

use flock_core::{circuit::bus::SharedChallenges, union::publics_digest};
use flock_merkle::{HashKind, hash_pair};

#[cfg(not(target_arch = "aarch64"))]
use crate::verifier::verify_ligerito_union_circuit_deferred;
use crate::{
    lean_rows::{Piece, TOWER_DOMAIN},
    tower::{F128, MixedInner, MixedProof},
};

/// The Lean leaf's transcript domain.
pub(super) const LEAN_DOMAIN: &[u8] = TOWER_DOMAIN;

/// A field word pair back to the 32 bytes it packs (the inverse of the
/// tower's `digest_f128`).
pub(super) fn f128_digest(w: [F128; 2]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (k, x) in w.iter().enumerate() {
        d[16 * k..16 * k + 8].copy_from_slice(&x.lo.to_le_bytes());
        d[16 * k + 8..16 * k + 16].copy_from_slice(&x.hi.to_le_bytes());
    }
    d
}

/// A Lean leaf's word in the commitment tree: its publics digest chained
/// with each digest of its commitment cap, as BLAKE3 parents.
pub(super) fn lean_leaf_word(public: &[F128], cap: &[[u8; 32]]) -> [u8; 32] {
    let mut acc = publics_digest(public);
    for c in cap {
        acc = hash_pair(&acc, c, HashKind::Blake3);
    }
    acc
}

/// The commitment root in the tower's shape: leaf pairs hash into the
/// first-level digests `f_j`, the base node pairs the last two, and each
/// spine node prepends the next-earlier one,
/// `H(f_0, H(f_1, … H(f_{k-2}, f_{k-1})))`.
pub(super) fn lean_cone_root(words: &[[u8; 32]]) -> [u8; 32] {
    assert!(
        words.len() >= 4 && words.len() % 2 == 0,
        "leaf pairs, at least two"
    );
    let pair = |l: &[u8; 32], r: &[u8; 32]| hash_pair(l, r, HashKind::Blake3);
    let fls: Vec<[u8; 32]> = words.chunks(2).map(|p| pair(&p[0], &p[1])).collect();
    let k = fls.len();
    let mut acc = pair(&fls[k - 2], &fls[k - 1]);
    for f in fls[..k - 2].iter().rev() {
        acc = pair(f, &acc);
    }
    acc
}

/// The challenges' seed every node carries: β and the γs, paired into
/// digests and chained as BLAKE3 parents.
pub(super) fn challenge_seed(ch: &SharedChallenges) -> [u8; 32] {
    let words: Vec<F128> = std::iter::once(ch.beta)
        .chain(ch.gamma.iter().copied())
        .collect();
    assert!(
        words.len() % 2 == 0 && words.len() >= 4,
        "β and the γs pair up"
    );
    let d = |k: usize| f128_digest([words[k], words[k + 1]]);
    let mut acc = hash_pair(&d(0), &d(2), HashKind::Blake3);
    for k in (4..words.len()).step_by(2) {
        acc = hash_pair(&acc, &d(k), HashKind::Blake3);
    }
    acc
}

/// Prove `piece` (in the tower's posture) and deferred-verify the proof.
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn build_lean_leaf(
    piece: &Piece,
    shared: Option<&SharedChallenges>,
    label: &[u8],
) -> MixedInner {
    let (built, proof, commitment) = piece.tower_parts(shared, label);
    let union = piece.union_instance();
    let pcs = piece.pcs_params();
    let lcs = piece.lincheck_circuits();
    let mut ch = piece.challenger(label);
    let (result, _) = (
        verify_ligerito_union_circuit_deferred(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &lcs,
            &commitment,
            &proof,
            &pcs,
            &mut ch,
        ),
        (),
    );
    let (_claims, work, sigma) = result.expect("the deferred verify accepts an honest piece");
    work.boolean
        .as_ref()
        .expect("boolean matrix work")
        .check(&union, &lcs)
        .expect("the boolean matrix work discharges");
    assert!(sigma.check(&built.shape.circuit), "sigma discharges");
    drop(union);
    let _: Option<F128> = None;
    MixedInner {
        nu: piece.nu,
        built,
        proof: MixedProof::Rs(proof),
        commitment,
        pcs,
        work,
        sigma,
        full_height: false,
    }
}

/// Probe: one Lean piece's tape, then its child region alone, emitted, run
/// and checked. Today this stops where the tower refuses what a Lean
/// piece needs (several element tables, buses with multiplicities).
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
#[cfg(not(target_arch = "aarch64"))]
fn lean_child_region_emits_alone() {
    let small = include_str!("../../tests/data/lean_small.rows");
    let rows = Rows::merge((0..4000).map(|_| Rows::parse(small).unwrap()).collect());
    let c = LeanCircuit::build(&rows).unwrap();
    let piece = c.piece_with(&c.pieces(usize::MAX)[0], true).unwrap();
    let inner = build_lean_leaf(&piece, None, b"");
    let lcs = piece.lincheck_circuits();
    let ct = ChildTape::new(&inner, LEAN_DOMAIN, &lcs);
    let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(16);
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let mut vals: Vec<F128> = Vec::new();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let b3_slot = cs.q.b3;
    let region = emit_child_region(
        &mut sb,
        &mut cs,
        b3_slot,
        &ct,
        &mut vals,
        &mut hints,
        &mut consts,
    );
    let shape2 = sb.finish().expect("the Lean child circuit builds");
    let hint_refs: Vec<&(dyn Any + Sync)> = hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
    let built2 = shape2.run(&vals, &hint_refs);
    let consumed = check_child_region(&built2.public, &ct, &region);
    assert_eq!(region.pub_base + consumed, built2.public.len());
}

/// The multiplicity sumcheck binds in-circuit: with one row-tag value at a
/// `σ_i`, or one bus column at `ρ_row`, moved by one, the region's witness
/// cannot be completed. Only tags whose bit is nonzero at `r` are read (the
/// verifier multiplies a tag by its bit), so the test moves those.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
#[cfg(not(target_arch = "aarch64"))]
fn lean_child_region_rejects_moved_counts() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let small = include_str!("../../tests/data/lean_small.rows");
    let rows = Rows::merge((0..4000).map(|_| Rows::parse(small).unwrap()).collect());
    let c = LeanCircuit::build(&rows).unwrap();
    let piece = c.piece_with(&c.pieces(usize::MAX)[0], true).unwrap();
    let inner = build_lean_leaf(&piece, None, b"");
    let lcs = piece.lincheck_circuits();
    let mut ct = ChildTape::new(&inner, LEAN_DOMAIN, &lcs);
    let runs = |ct: &ChildTape<'_>| -> bool {
        let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(16);
        let mut sb = ShapeBuilder::new(nu2);
        let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
        let (mut vals, mut hints, mut consts): (
            Vec<F128>,
            Vec<[u32; SLOT_WORDS]>,
            Vec<(F128, Wire)>,
        ) = (Vec::new(), Vec::new(), Vec::new());
        let b3_slot = cs.q.b3;
        let _ = emit_child_region(
            &mut sb,
            &mut cs,
            b3_slot,
            ct,
            &mut vals,
            &mut hints,
            &mut consts,
        );
        let shape2 = sb.finish().expect("the Lean child circuit builds");
        let hint_refs: Vec<&(dyn Any + Sync)> =
            hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
        catch_unwind(AssertUnwindSafe(|| shape2.run(&vals, &hint_refs))).is_ok()
    };
    assert!(runs(&ct), "the honest region runs");
    let honest = ct.sigma_native.bus.clone();
    ct.sigma_native.bus.as_mut().unwrap().values[0] += F128::ONE;
    assert!(!runs(&ct), "a moved bus column at ρ_row");
    ct.sigma_native.bus = honest;
    let m = ct
        .sigma_native
        .bus_mult
        .clone()
        .expect("Lean counts its lookups");
    eprintln!(
        "{} count bits, {} tag columns at the σ_i",
        m.sigma.len(),
        m.values.len()
    );
    assert!(!m.values.is_empty(), "Lean's counted ports carry row tags");
    let bus = inner.built.shape.circuit.bus().expect("a bus");
    let mp = inner
        .proof
        .wiring()
        .bus
        .as_ref()
        .unwrap()
        .mult
        .as_ref()
        .unwrap();
    let cols = bus.table_columns();
    let bit_slots = bus.mult_bits();
    let read: Vec<usize> = (0..m.columns.len())
        .filter(|&k| {
            let (i, c) = m.columns[k];
            bus.ports.iter().filter(|p| p.mult.len() > i).any(|p| {
                let bit = mp.bits[bit_slots.binary_search(&p.mult[i]).unwrap()];
                bit != F128::ZERO
                    && p.terms.iter().filter_map(|t| t.row).any(|tag| {
                        cols[c] == flock_core::circuit::bus::BusColumn::Tag { ty: p.ty, tag }
                    })
            })
        })
        .collect();
    eprintln!("{} of the tags are read", read.len());
    assert!(read.len() >= 2, "tags to move");
    let (first, last) = (read[0], read[read.len() - 1]);
    for i in [first, last] {
        let mut bad = m.clone();
        bad.values[i] += F128::ONE;
        ct.sigma_native.bus_mult = Some(bad);
        assert!(!runs(&ct), "a moved tag value {i} at its σ");
    }
    ct.sigma_native.bus_mult = Some(m);
    let honest = ct.sigma_native.bus.clone();
    ct.sigma_native.bus.as_mut().unwrap().values[0] += F128::ONE;
    assert!(!runs(&ct), "a moved bus column at ρ_row");
    ct.sigma_native.bus = honest;
    assert!(runs(&ct), "restored, it runs again");
}

/// Two Lean pieces sharing one bus: committed first, challenges drawn from
/// both commitments, each proven as a tower leaf with its products left
/// open. Each child region emits, runs and checks, and the open products
/// balance across the two.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
#[cfg(not(target_arch = "aarch64"))]
fn lean_shared_pieces_emit() {
    use crate::lean_rows::shared_challenges;
    let small = include_str!("../../tests/data/lean_small.rows");
    let rows = Rows::merge((0..4000).map(|_| Rows::parse(small).unwrap()).collect());
    let c = LeanCircuit::build(&rows).unwrap();
    let total: usize = c.chunks().iter().map(|x| x.2).sum();
    let pieces: Vec<_> = c
        .pieces(total / 2 + 1)
        .iter()
        .map(|p| c.piece_with(p, true).unwrap())
        .collect();
    assert_eq!(pieces.len(), 2);
    let statement: Vec<_> = pieces.iter().map(|p| (p.digest(), p.commit())).collect();
    let ch = shared_challenges(&statement);
    let (mut lhs, mut rhs) = (F128::ONE, F128::ONE);
    for (i, p) in pieces.iter().enumerate() {
        let label = (i as u64).to_le_bytes();
        let inner = build_lean_leaf(p, Some(&ch), &label);
        let gkr = &inner.proof.wiring().bus.as_ref().unwrap().gkr;
        lhs *= gkr.top_lhs;
        rhs *= gkr.top_rhs;
        assert_ne!(gkr.top_lhs, gkr.top_rhs, "one piece alone does not balance");
        let lcs = p.lincheck_circuits();
        let ct = ChildTape::new(&inner, LEAN_DOMAIN, &lcs);
        let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(16);
        let mut sb = ShapeBuilder::new(nu2);
        let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let b3_slot = cs.q.b3;
        let region = emit_child_region(
            &mut sb,
            &mut cs,
            b3_slot,
            &ct,
            &mut vals,
            &mut hints,
            &mut consts,
        );
        let shape2 = sb.finish().expect("the Lean child circuit builds");
        let hint_refs: Vec<&(dyn Any + Sync)> =
            hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
        let built2 = shape2.run(&vals, &hint_refs);
        let consumed = check_child_region(&built2.public, &ct, &region);
        assert_eq!(region.pub_base + consumed, built2.public.len());
    }
    assert_eq!(lhs, rhs, "the open products balance across the pieces");
}

/// Probe: a first-level node over two Lean leaves of one profile, sharing
/// one bus.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
#[cfg(not(target_arch = "aarch64"))]
fn lean_first_level_node() {
    use crate::{
        lean_rows::{LeafSlot, Profile, leaf, shared_challenges},
        tower::{FlApp, FlFamily, build_first_level_core, test_config},
    };
    let small = include_str!("../../tests/data/lean_small.rows");
    let rows = Rows::merge((0..4000).map(|_| Rows::parse(small).unwrap()).collect());
    let c = LeanCircuit::build(&rows).unwrap();
    let mut per_table: Vec<Vec<usize>> =
        vec![Vec::new(); c.chunks().iter().map(|x| x.0 + 1).max().unwrap()];
    for (s, ch, _) in c.chunks() {
        per_table[s].push(ch);
    }
    let n: Vec<usize> = per_table.iter().map(|t| t.len()).collect();
    let profile = Profile::even(c.nu, &n, 2);
    let leaves: Vec<_> = profile
        .deal(&per_table, 2)
        .iter()
        .map(|d| {
            let slots: Vec<LeafSlot<'_>> = d.iter().map(|x| x.map(|ch| (&c, ch))).collect();
            leaf(&c, &profile, &slots).unwrap()
        })
        .collect();
    let statement: Vec<_> = leaves.iter().map(|p| (p.digest(), p.commit())).collect();
    let ch = shared_challenges(&statement);
    let inners: Vec<MixedInner> = leaves
        .iter()
        .enumerate()
        .map(|(i, p)| build_lean_leaf(p, Some(&ch), &(i as u64).to_le_bytes()))
        .collect();
    let registry = &inners[0].built.shape.registry;
    let family = FlFamily {
        domain: LEAN_DOMAIN,
        mats: registry
            .boolean_types()
            .iter()
            .map(|ty| (&ty.a_0, &ty.b_0))
            .collect(),
        circs: leaves[0].lincheck_circuits(),
        lane_words: None,
        shared: None,
        env_nu: Some(15),
    };
    crate::tower::envelope::set_envelope_override(Some((15, 40, 30)));
    let t = std::time::Instant::now();
    let core = build_first_level_core(
        test_config(),
        b"cas-lean-fl-node-v0",
        &[&inners[0], &inners[1]],
        &family,
        &FlApp::Batches { per_leaf: 1 },
    );
    let bytes = match &core.lo.proof {
        MixedProof::Rs(p) => bincode::serialize(p).unwrap().len(),
        MixedProof::Ag(p) => bincode::serialize(p).unwrap().len(),
    };
    eprintln!(
        "first-level node over two Lean leaves: built in {:?}; prove {:.0} ms, verify {:.1} ms, proof {} KiB",
        t.elapsed(),
        core.t.prove_ms,
        core.t.verify_ms,
        bytes / 1024
    );
    crate::tower::envelope::set_envelope_override(None);
}

/// Probe: an internal node over two first-level nodes, each over two
/// Lean leaves of one profile, all four sharing one bus.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
#[cfg(not(target_arch = "aarch64"))]
fn lean_internal_node() {
    use crate::{
        lean_rows::{LeafSlot, Profile, leaf},
        tower::{FlApp, FlFamily, build_first_level_core, build_node_outer_app, test_config},
    };
    let small = include_str!("../../tests/data/lean_small.rows");
    let rows = Rows::merge((0..4000).map(|_| Rows::parse(small).unwrap()).collect());
    let c = LeanCircuit::build(&rows).unwrap();
    let mut per_table: Vec<Vec<usize>> =
        vec![Vec::new(); c.chunks().iter().map(|x| x.0 + 1).max().unwrap()];
    for (s, ch, _) in c.chunks() {
        per_table[s].push(ch);
    }
    let n: Vec<usize> = per_table.iter().map(|t| t.len()).collect();
    let profile = Profile::even(c.nu, &n, 4);
    let leaves: Vec<_> = profile
        .deal(&per_table, 4)
        .iter()
        .map(|d| {
            let slots: Vec<LeafSlot<'_>> = d.iter().map(|x| x.map(|ch| (&c, ch))).collect();
            leaf(&c, &profile, &slots).unwrap()
        })
        .collect();
    // The cone's statement: the commitment tree over the leaves' words,
    // and the challenges drawn from its root.
    let words: Vec<[u8; 32]> = leaves
        .iter()
        .map(|p| lean_leaf_word(&p.public(), &p.commit().cap))
        .collect();
    let root = lean_cone_root(&words);
    let ch = crate::lean_rows::shared_challenges_from_root(&root);
    let seed = challenge_seed(&ch);
    let inners: Vec<MixedInner> = leaves
        .iter()
        .enumerate()
        .map(|(i, p)| build_lean_leaf(p, Some(&ch), &(i as u64).to_le_bytes()))
        .collect();
    let registry = &inners[0].built.shape.registry;
    let family = FlFamily {
        domain: LEAN_DOMAIN,
        mats: registry
            .boolean_types()
            .iter()
            .map(|ty| (&ty.a_0, &ty.b_0))
            .collect(),
        circs: leaves[0].lincheck_circuits(),
        lane_words: None,
        shared: None,
        env_nu: Some(15),
    };
    crate::tower::envelope::set_envelope_override(Some((15, 40, 30)));
    let fls: Vec<_> = (0..2)
        .map(|j| {
            build_first_level_core(
                test_config(),
                b"cas-lean-fl-node-v0",
                &[&inners[2 * j], &inners[2 * j + 1]],
                &family,
                &FlApp::LeanBus { seed },
            )
        })
        .collect();
    assert_eq!(
        fls[0].lo.shape.circuit.digest(),
        fls[1].lo.shape.circuit.digest(),
        "one first-level circuit for every pair of Lean leaves"
    );
    let t = std::time::Instant::now();
    // The lane: the leaves' own accumulators, folded priors-only.
    let el_types: Vec<_> = registry
        .element_types()
        .iter()
        .map(|s| s.element_type().expect("an element slot's table"))
        .collect();
    let el_mats: Vec<_> = el_types.iter().map(|t| (t.a_0(), t.b_0())).collect();
    let u = inners[0].union();
    let jp = flock_core::pcs::jagged::JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        inners[0].commitment.params.m - flock_core::pcs::LOG_PACKING,
    );
    let lane = crate::tower::ChainLane {
        registry,
        mats: &family.mats,
        circs: &family.circs,
        circuit: &inners[0].built.shape.circuit,
        params: &jp,
        priors: &[&fls[0].acc, &fls[1].acc],
        claims_base: fls[0].fold_pub_base,
        family: crate::tower::LaneFamily::Lean {
            block_words: crate::tower::envelope::lane_block_words(&fls[0].acc),
            el_mats: &el_mats,
        },
    };
    let out = build_node_outer_app(
        test_config(),
        &[&fls[0].lo, &fls[1].lo],
        Some(fls[0].stmt_base),
        Some(lane),
        None,
    );
    // THE ROOT'S CHECKS, from the node's publics alone: the commitment
    // digest is the tree's root, the seed the challenges that root draws,
    // and the open products balance.
    let app = out.app_base.expect("the node carries the app block");
    let p = &out.lo.public;
    assert_eq!(
        f128_digest([p[app], p[app + 1]]),
        root,
        "the commitment root"
    );
    assert_eq!(
        f128_digest([p[app + 2], p[app + 3]]),
        seed,
        "the challenges' seed"
    );
    assert_eq!(
        p[app + 4],
        p[app + 5],
        "the shared bus balances over the cone"
    );
    assert_ne!(p[app + 4], F128::ZERO, "a nonzero product");
    // THE ROOT-SIDE DISCHARGE: the lane against the Lean leaf's tables, the
    // main fold against the first-level and node circuits. With the root
    // proof's own verify, this is the cone's whole verification.
    let lane_acc = out.lane_acc.as_ref().expect("the lane accumulator");
    assert_eq!(
        lane_acc.per_element.len(),
        el_mats.len(),
        "the lane folds the element class"
    );
    let t_d = std::time::Instant::now();
    let leaf_circuit = &inners[0].built.shape.circuit;
    assert!(
        lane_acc.discharge(&family.mats),
        "the lane's boolean class discharges"
    );
    assert!(
        lane_acc.discharge_element(&el_mats),
        "the lane's element class discharges"
    );
    assert!(
        lane_acc.discharge_sigma(&[leaf_circuit]),
        "the lane's sigma discharges"
    );
    assert!(
        lane_acc.discharge_jagged(&[(leaf_circuit.digest(), &jp)]),
        "the lane's jagged entry discharges"
    );
    let lane_ms = t_d.elapsed().as_secs_f64() * 1e3;
    {
        let fl_lo = &fls[0].lo;
        let t = std::time::Instant::now();
        let _ = out
            .acc
            .discharge_sigma(&[&fl_lo.shape.circuit, &out.lo.shape.circuit]);
        let sig = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let fj = crate::tower::node::node_jagged_params(fl_lo);
        let rj = crate::tower::node::node_jagged_params(&out.lo);
        let _ = out.acc.discharge_jagged(&[
            (fl_lo.shape.circuit.digest(), &fj),
            (out.lo.shape.circuit.digest(), &rj),
        ]);
        let jag = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let _ = lane_acc.discharge_sigma(&[leaf_circuit]);
        let lsig = t.elapsed().as_secs_f64() * 1e3;
        {
            let t = std::time::Instant::now();
            let _ = out
                .acc
                .discharge(&crate::tower::query::leaf_boolean_mats(fl_lo));
            let b = t.elapsed().as_secs_f64() * 1e3;
            let em: Vec<_> = fl_lo
                .shape
                .registry
                .element_types()
                .iter()
                .map(|t| {
                    let e = t.element_type().unwrap();
                    (e.a_0(), e.b_0())
                })
                .collect();
            let t = std::time::Instant::now();
            let _ = out.acc.discharge_element(&em);
            let e = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "main boolean {b:.1} ms ({} types), main element {e:.1} ms ({} types)",
                out.acc.per_type.len(),
                em.len()
            );
        }
        eprintln!(
            "main sigma {sig:.1} ms, main jagged {jag:.1} ms, lane sigma {lsig:.1} ms; sigma entries {}",
            out.acc.sigma.len()
        );
    }
    let t_m = std::time::Instant::now();
    crate::tower::driver::discharge_outer(&fls[0].lo, &out.lo, None, &out.acc, &[], false)
        .expect("the main fold discharges");
    let main_ms = t_m.elapsed().as_secs_f64() * 1e3;
    eprintln!("root-side discharge: lane {lane_ms:.1} ms, main {main_ms:.1} ms");
    eprintln!(
        "the cone's verification: root proof {:.1} ms + discharge {:.1} ms",
        out.online.verify_ms,
        lane_ms + main_ms
    );
    let lane_acc = out.lane_acc.as_ref().expect("the lane accumulator");
    assert_eq!(
        lane_acc.per_element.len(),
        el_mats.len(),
        "the lane folds the element class"
    );
    let bytes = match &out.lo.proof {
        MixedProof::Rs(p) => bincode::serialize(p).unwrap().len(),
        MixedProof::Ag(p) => bincode::serialize(p).unwrap().len(),
    };
    eprintln!(
        "internal node over two Lean first-level nodes: built in {:?}; prove {:.0} ms, verify {:.1} ms, proof {} KiB",
        t.elapsed(),
        out.online.prove_ms,
        out.online.verify_ms,
        bytes / 1024
    );
    crate::tower::envelope::set_envelope_override(None);
}
