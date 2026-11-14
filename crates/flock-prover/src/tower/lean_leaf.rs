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

use flock_core::circuit::bus::SharedChallenges;

#[cfg(not(target_arch = "aarch64"))]
use crate::verifier::verify_ligerito_union_circuit_deferred;
use crate::{
    lean_rows::{Piece, TOWER_DOMAIN},
    tower::{F128, MixedInner, MixedProof},
};

/// The Lean leaf's transcript domain.
pub(super) const LEAN_DOMAIN: &[u8] = TOWER_DOMAIN;

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
    let region = emit_child_region(&mut sb, &mut cs, b3_slot, &ct, &mut vals, &mut hints, &mut consts);
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
        let (mut vals, mut hints, mut consts): (Vec<F128>, Vec<[u32; SLOT_WORDS]>, Vec<(F128, Wire)>) =
            (Vec::new(), Vec::new(), Vec::new());
        let b3_slot = cs.q.b3;
        let _ = emit_child_region(&mut sb, &mut cs, b3_slot, ct, &mut vals, &mut hints, &mut consts);
        let shape2 = sb.finish().expect("the Lean child circuit builds");
        let hint_refs: Vec<&(dyn Any + Sync)> = hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
        catch_unwind(AssertUnwindSafe(|| shape2.run(&vals, &hint_refs))).is_ok()
    };
    assert!(runs(&ct), "the honest region runs");
    let honest = ct.sigma_native.bus.clone();
    ct.sigma_native.bus.as_mut().unwrap().values[0] += F128::ONE;
    assert!(!runs(&ct), "a moved bus column at ρ_row");
    ct.sigma_native.bus = honest;
    let m = ct.sigma_native.bus_mult.clone().expect("Lean counts its lookups");
    eprintln!(
        "{} count bits, {} tag columns at the σ_i",
        m.sigma.len(),
        m.values.len()
    );
    assert!(!m.values.is_empty(), "Lean's counted ports carry row tags");
    let bus = inner.built.shape.circuit.bus().expect("a bus");
    let mp = inner.proof.wiring().bus.as_ref().unwrap().mult.as_ref().unwrap();
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
        let region = emit_child_region(&mut sb, &mut cs, b3_slot, &ct, &mut vals, &mut hints, &mut consts);
        let shape2 = sb.finish().expect("the Lean child circuit builds");
        let hint_refs: Vec<&(dyn Any + Sync)> = hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
        let built2 = shape2.run(&vals, &hint_refs);
        let consumed = check_child_region(&built2.public, &ct, &region);
        assert_eq!(region.pub_base + consumed, built2.public.len());
    }
    assert_eq!(lhs, rhs, "the open products balance across the pieces");
}
