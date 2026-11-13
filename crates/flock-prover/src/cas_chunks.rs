//! Chunks of a blob proven against its key without the rest of its bytes
//! (alon/cas issue #8).
//!
//! BLAKE3 hashes a blob as a tree of 1024-byte chunks. A proof here opens
//! some chunks of a blob of a given length: their bytes are hashed with
//! their chunk index, and every subtree that holds no opened chunk enters
//! as its chaining value, a private input of two words. The parent rows
//! join them up to the blob's key, which is public. So the cost is the
//! opened chunks' blocks and one parent per level, not the whole blob.
//!
//! With two sides, the proof is of an edit: two blobs of the same length,
//! each with a public key, whose chunks are the same except the opened ones.
//! The closed subtrees are one set of inputs, shared by both sides, so the
//! verifier learns that everything outside the opened chunks is equal
//! without hashing it again.

use flock_core::{
    circuit::{
        boolean::BooleanGate,
        builder::{CircuitShape, CircuitWitness, FillPlan, ShapeBuilder, SlotId, Wire},
    },
    lincheck::LincheckCircuit,
    union::UnionInstance,
    verifier::FlockVerifyError,
};
use flock_field::F128;
use std::sync::Arc;

use crate::{
    cas_typed::{
        CHUNK_END, CHUNK_START, Consts, PARENT, ROOT, ROWS, TypedProof, block_ctl, compression, iv, key_words, words,
    },
    leanxmss::{
        circuit::XmssCircuit,
        compress::{self, CompressPacker, ctl},
    },
    prover::UnionSlotProverInput,
};

/// One piece of the tree in input order: a closed subtree (chunks
/// `lo..hi`) or an opened chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Piece {
    Closed(usize, usize),
    Open(usize),
}

/// The pieces of the tree of `n` chunks over `lo..hi`, left to right, as
/// BLAKE3 splits it (the left side takes the largest power of two below the
/// count).
fn pieces(lo: usize, hi: usize, opened: &[usize], out: &mut Vec<Piece>) {
    if !opened.iter().any(|&c| (lo..hi).contains(&c)) {
        out.push(Piece::Closed(lo, hi));
    } else if hi - lo == 1 {
        out.push(Piece::Open(lo));
    } else {
        let left = (hi - lo).next_power_of_two() / 2;
        pieces(lo, lo + left, opened, out);
        pieces(lo + left, hi, opened, out);
    }
}

/// Chunks of one blob, or of an edit of it, proven against the keys.
pub struct ChunkCircuit {
    len: usize,
    pieces: Vec<Piece>,
    sides: usize,
    consts: Vec<F128>,
    shape: CircuitShape,
    plan: FillPlan,
    gadget: (SlotId, BooleanGate),
    packer: Arc<CompressPacker>,
    /// Compression rows, without padding.
    pub compressions: usize,
}

impl ChunkCircuit {
    /// Open the chunks `opened` (indices, ascending, no repeats) of a blob of
    /// `len` bytes; with `sides` 2, of two blobs that differ only there.
    pub fn new(len: usize, opened: &[usize], sides: usize) -> Self {
        let chunks = len.div_ceil(1024).max(1);
        assert!((1..=2).contains(&sides));
        assert!(!opened.is_empty() && opened.windows(2).all(|w| w[0] < w[1]) && opened.iter().all(|&c| c < chunks));
        assert!(len > 0, "the empty blob has no chunk to open");
        let mut ps = Vec::new();
        pieces(0, chunks, opened, &mut ps);
        let blocks_of = |c: usize| (len.div_ceil(64) - 16 * c).min(16);
        let compressions = sides * opened.iter().map(|&c| blocks_of(c)).sum::<usize>()
            + sides * (ps.len() - 1);

        let mut consts = Consts::default();
        let (zero, empty) = (F128::new(0, 0), ctl(CHUNK_START | CHUNK_END | ROOT, 0));
        for v in [iv()[0], iv()[1], zero, empty, ctl(PARENT, 64), ctl(PARENT | ROOT, 64)] {
            consts.add(v);
        }
        for &c in opened {
            let n = blocks_of(c);
            for j in 0..n {
                consts.add(block_ctl(len, c, j, n));
            }
        }
        let nu = compressions.next_multiple_of(ROWS).next_power_of_two().trailing_zeros().max(8) as usize;
        let mut sb = ShapeBuilder::new(nu);
        sb.prune_unwired();
        sb.open_public_columns();
        sb.pack_live_slots();
        let (cg, packer) = compression(nu);
        let cs = sb.slot(cg.clone());
        let wires: Vec<Wire> = consts.values.iter().map(|&v| sb.fixed_public_input(v)).collect();
        let wire = |v: F128| wires[consts.index[&(v.lo, v.hi)]];
        let (ivw, zw) = ([wire(iv()[0]), wire(iv()[1])], wire(zero));
        for _ in compressions..compressions.next_multiple_of(ROWS) {
            sb.gate(cs, &[ivw[0], ivw[1], zw, zw, zw, zw, wire(empty)]);
        }
        let keys: Vec<[Wire; 2]> = (0..sides).map(|_| [sb.public_input(), sb.public_input()]).collect();
        // Inputs in piece order: a closed subtree's two words, or each
        // side's blocks of an opened chunk.
        let mut leaves: Vec<Vec<[Wire; 2]>> = vec![Vec::new(); sides];
        for p in &ps {
            match *p {
                Piece::Closed(..) => {
                    let cv = [sb.input(), sb.input()];
                    for l in leaves.iter_mut() {
                        l.push(cv);
                    }
                }
                Piece::Open(c) => {
                    let n = blocks_of(c);
                    for l in leaves.iter_mut() {
                        let m: Vec<[Wire; 4]> = (0..n).map(|_| std::array::from_fn(|_| sb.input())).collect();
                        let mut cv = ivw;
                        for (j, b) in m.iter().enumerate() {
                            let out = sb.gate(cs, &[cv[0], cv[1], b[0], b[1], b[2], b[3], wire(block_ctl(len, c, j, n))]);
                            cv = [out[0], out[1]];
                        }
                        l.push(cv);
                    }
                }
            }
        }
        // Join the pieces as the tree joins them.
        let parents = [wire(ctl(PARENT, 64)), wire(ctl(PARENT | ROOT, 64))];
        for (side, l) in leaves.iter().enumerate() {
            let mut at = 0;
            let root = join(&mut sb, cs, ivw, parents, 0, chunks, opened, l, &mut at, true);
            assert_eq!(at, l.len());
            sb.connect(root[0], keys[side][0]);
            sb.connect(root[1], keys[side][1]);
        }
        let shape = sb.finish().expect("well formed");
        let plan = shape.fill_plan();
        ChunkCircuit {
            len,
            pieces: ps,
            sides,
            consts: consts.values,
            shape,
            plan,
            gadget: (cs, cg),
            packer,
            compressions,
        }
    }

    /// Input values in declaration order; without `blobs`, the public ones.
    fn values(&self, keys: &[[u8; 32]], blobs: Option<&[Vec<u8>]>) -> Vec<F128> {
        let mut v = self.consts.clone();
        for k in &keys[..self.sides] {
            v.extend(key_words(k));
        }
        let Some(blobs) = blobs else { return v };
        for p in &self.pieces {
            match *p {
                Piece::Closed(lo, hi) => {
                    let cv = subtree_cv(&blobs[0], lo, hi);
                    if self.sides == 2 {
                        assert_eq!(cv, subtree_cv(&blobs[1], lo, hi), "the blobs differ outside the opened chunks");
                    }
                    v.extend(key_words(&cv));
                }
                Piece::Open(c) => {
                    for b in &blobs[..self.sides] {
                        let end = (1024 * (c + 1)).min(self.len);
                        let mut chunk = b[1024 * c..end].to_vec();
                        chunk.resize(chunk.len().next_multiple_of(64), 0);
                        for block in chunk.chunks_exact(64) {
                            v.extend(words(block));
                        }
                    }
                }
            }
        }
        v
    }

    /// The witness from the whole blobs (one for each side), which the
    /// prover has.
    pub fn witness(&self, keys: &[[u8; 32]], blobs: &[Vec<u8>]) -> CircuitWitness {
        assert!(blobs.iter().all(|b| b.len() == self.len));
        self.shape.run_filled_deferred(&self.plan, &self.values(keys, Some(blobs)), &[])
    }

    fn union(&self) -> UnionInstance<'_> {
        let mut union = UnionInstance::new(&self.shape.registry, self.shape.counts.clone());
        union.set_full_height_commit();
        union
    }

    fn challenger() -> flock_core::challenger::FsChallenger {
        flock_core::challenger::FsChallenger::new(b"cas-chunks-blake3-v0")
    }

    pub fn prove(&self, witness: &CircuitWitness) -> TypedProof {
        let union = self.union();
        let pcs = XmssCircuit::pcs_params(&union);
        let (id, gate) = &self.gadget;
        let rows = witness.rows::<BooleanGate>(*id);
        assert_eq!(gate.name(), compress::NAME);
        let packer = &*self.packer;
        let input = UnionSlotProverInput::in_place(move |dst| packer.pack_into(rows, dst), gate.r1cs().csc_lincheck_circuit());
        let mut challenger = Self::challenger();
        let (proof, commitment, _) = crate::leanxmss::circuit::prove_circuit(
            &union,
            &self.shape.circuit,
            &witness.public,
            &pcs,
            vec![input],
            Vec::new(),
            &mut challenger,
        );
        TypedProof { proof, commitment }
    }

    pub fn verify(&self, keys: &[[u8; 32]], proof: &TypedProof) -> Result<(), FlockVerifyError> {
        let union = self.union();
        let pcs = XmssCircuit::pcs_params(&union);
        let circuits: Vec<&dyn LincheckCircuit> = vec![self.gadget.1.r1cs().csc_lincheck_circuit() as _];
        let mut challenger = Self::challenger();
        crate::leanxmss::circuit::verify_circuit(
            &union,
            &self.shape.circuit,
            &self.values(keys, None),
            &circuits,
            &proof.commitment,
            &proof.proof,
            &pcs,
            &mut challenger,
        )
        .map(|_| ())
    }
}

/// Join the leaves of chunks `lo..hi`, taking pieces from `l` at `at`.
#[allow(clippy::too_many_arguments)]
fn join(
    sb: &mut ShapeBuilder,
    cs: SlotId,
    iv: [Wire; 2],
    parents: [Wire; 2],
    lo: usize,
    hi: usize,
    opened: &[usize],
    l: &[[Wire; 2]],
    at: &mut usize,
    root: bool,
) -> [Wire; 2] {
    if !opened.iter().any(|&c| (lo..hi).contains(&c)) || hi - lo == 1 {
        *at += 1;
        return l[*at - 1];
    }
    let left = (hi - lo).next_power_of_two() / 2;
    let a = join(sb, cs, iv, parents, lo, lo + left, opened, l, at, false);
    let b = join(sb, cs, iv, parents, lo + left, hi, opened, l, at, false);
    let out = sb.gate(cs, &[iv[0], iv[1], a[0], a[1], b[0], b[1], parents[usize::from(root)]]);
    [out[0], out[1]]
}

/// The chaining value of chunks `lo..hi` of `blob`, as a subtree that is not
/// the root.
fn subtree_cv(blob: &[u8], lo: usize, hi: usize) -> [u8; 32] {
    use blake3::hazmat::HasherExt;
    let end = (1024 * hi).min(blob.len());
    let mut h = blake3::Hasher::new();
    h.set_input_offset(1024 * lo as u64);
    h.update(&blob[1024 * lo..end]);
    h.finalize_non_root()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
    }

    #[test]
    fn one_chunk_is_proven_against_the_key() {
        // 9.5 chunks: an uneven tree, a short last chunk.
        let len = 9 * 1024 + 512;
        let b = blob(len, 1);
        let key = *blake3::hash(&b).as_bytes();
        for opened in [vec![0], vec![4], vec![9], vec![2, 7]] {
            let c = ChunkCircuit::new(len, &opened, 1);
            let whole = crate::cas_typed::compressions(len.div_ceil(64));
            assert!(c.compressions < whole / 2, "{} of {whole}", c.compressions);
            let proof = c.prove(&c.witness(&[key], std::slice::from_ref(&b)));
            c.verify(&[key], &proof).expect("verifies");
            let mut other = key;
            other[0] ^= 1;
            assert!(c.verify(&[other], &proof).is_err(), "a proof is of its key only");
        }
    }

    #[test]
    fn a_blob_of_one_chunk_opens_at_the_root() {
        let b = blob(700, 3);
        let key = *blake3::hash(&b).as_bytes();
        let c = ChunkCircuit::new(700, &[0], 1);
        c.verify(&[key], &c.prove(&c.witness(&[key], &[b]))).expect("verifies");
    }

    #[test]
    fn an_edit_reuses_the_old_tree() {
        let len = 20 * 1024;
        let old = blob(len, 5);
        let mut new = old.clone();
        new[5 * 1024 + 17] ^= 0xff;
        new[13 * 1024] ^= 1;
        let keys = [*blake3::hash(&old).as_bytes(), *blake3::hash(&new).as_bytes()];
        let c = ChunkCircuit::new(len, &[5, 13], 2);
        let proof = c.prove(&c.witness(&keys, &[old.clone(), new.clone()]));
        c.verify(&keys, &proof).expect("verifies");
        // The same proof does not say the edit is of another blob.
        let mut third = old.clone();
        third[0] ^= 1;
        let k3 = *blake3::hash(&third).as_bytes();
        assert!(c.verify(&[k3, keys[1]], &proof).is_err());
    }

    #[test]
    #[should_panic(expected = "differ outside")]
    fn an_edit_outside_the_opened_chunks_has_no_witness() {
        let len = 4 * 1024;
        let old = blob(len, 7);
        let mut new = old.clone();
        new[3000] ^= 1;
        let keys = [*blake3::hash(&old).as_bytes(), *blake3::hash(&new).as_bytes()];
        let c = ChunkCircuit::new(len, &[0], 2);
        c.witness(&keys, &[old, new]);
    }
}
