//! A cas format circuit wired to BLAKE3 rows: `n` blobs of the `runs` format,
//! each `k ≤ 16` blocks of 64 bytes (one BLAKE3 chunk), proven to hash to
//! their public keys and to satisfy the format.
//!
//! A `runs` block is one record: a 32-byte key, a 64-bit count ≥ 1 and 24
//! zero bytes. Adjacent records have different keys. The format row takes
//! the same four block words the compression row hashes, so the checked
//! bytes are the hashed bytes.

use flock_core::{
    challenger::FsChallenger,
    circuit::{
        boolean::{
            BooleanCircuit, BooleanGate, BooleanGateError, CircuitBuilder, ColumnRole,
            ColumnSchema, ColumnVisitor, LinearExpr as Expr,
        },
        builder::{CircuitShape, CircuitWitness, FillPlan, ShapeBuilder, SlotId, Wire},
    },
    lincheck::LincheckCircuit,
    pcs::Commitment,
    union::UnionInstance,
    verifier::FlockVerifyError,
};
use flock_field::F128;

use crate::{
    leanxmss::{
        circuit::XmssCircuit,
        compress::{self, CompressPacker, compress_gate, ctl},
    },
    prover::UnionSlotProverInput,
    r1cs_hashes::blake3::BLAKE3_IV,
};

const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const ROOT: u32 = 8;

pub struct RunsCols<T> {
    pub m0: [T; 128],
    pub m1: [T; 128],
    pub m2: [T; 128],
    pub m3: [T; 128],
    pub p0: [T; 128],
    pub p1: [T; 128],
    /// Bit 0 set on a blob's first block: no previous key.
    pub first: [T; 128],
    /// `same[i]`: the first `i + 1` key bits equal the previous key's.
    pub same: [T; 256],
    /// `zero[i]`: the first `i + 1` count bits are zero.
    pub zero: [T; 64],
}

struct RunsSchema;

impl ColumnSchema for RunsSchema {
    type Cols<T> = RunsCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> RunsCols<V::Value> {
        let mut w = |name: &str| v.word_aligned(name, ColumnRole::Input, 128);
        let (m0, m1, m2, m3) = (w("m0"), w("m1"), w("m2"), w("m3"));
        let (p0, p1, first) = (w("p0"), w("p1"), w("first"));
        RunsCols {
            m0,
            m1,
            m2,
            m3,
            p0,
            p1,
            first,
            same: v.word("same", ColumnRole::Witness),
            zero: v.word("zero", ColumnRole::Witness),
        }
    }
}

pub fn runs_circuit() -> BooleanCircuit {
    CircuitBuilder::compile(RunsSchema, |b, c| {
        let one: Expr = b.one().into();
        // Key differs from the previous key, unless this is the first block.
        let mut acc = one;
        for i in 0..256 {
            let (k, p) = if i < 128 { (c.m0[i], c.p0[i]) } else { (c.m1[i - 128], c.p1[i - 128]) };
            let diff = b.xor2(k, p);
            let eq = b.xor2(diff, one);
            b.define_and(c.same[i], acc, eq);
            acc = c.same[i].into();
        }
        let not_first = b.xor2(c.first[0], one);
        b.assert_zero_product(acc, not_first);
        // Count ≥ 1.
        let mut acc = one;
        for i in 0..64 {
            let z = b.xor2(c.m2[i], one);
            b.define_and(c.zero[i], acc, z);
            acc = c.zero[i].into();
        }
        b.assert_zero(acc);
        // Padding is zero.
        for i in 64..128 {
            b.assert_zero(c.m2[i]);
        }
        for i in 0..128 {
            b.assert_zero(c.m3[i]);
        }
    })
    .circuit()
    .clone()
}

/// Inputs `[m0, m1, m2, m3, p0, p1, first]`, no outputs.
pub fn runs_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(
        "cas.runs",
        &runs_circuit(),
        &["m0", "m1", "m2", "m3", "p0", "p1", "first"],
        &[],
        nu,
    )
    .map(|g| {
        g.with_native(|i, _| {
            let first = i[6].lo & 1 == 1;
            assert!(first || (i[0], i[1]) != (i[4], i[5]), "cas.runs: repeated key");
            assert!(i[2].lo != 0, "cas.runs: zero count");
            assert!(i[2].hi == 0 && i[3] == F128::new(0, 0), "cas.runs: padding");
        })
    })
}

fn words(block: &[u8]) -> [F128; 4] {
    std::array::from_fn(|i| {
        let w = |o: usize| u64::from_le_bytes(block[16 * i + o..16 * i + o + 8].try_into().unwrap());
        F128::new(w(0), w(8))
    })
}

fn key_words(key: &[u8; 32]) -> [F128; 2] {
    let w = |o: usize| u64::from_le_bytes(key[o..o + 8].try_into().unwrap());
    [F128::new(w(0), w(8)), F128::new(w(16), w(24))]
}

/// `n` blobs of `k` blocks each.
pub struct RunsCircuit {
    n: usize,
    k: usize,
    shape: CircuitShape,
    plan: FillPlan,
    gadgets: Vec<(SlotId, BooleanGate)>,
    packer: CompressPacker,
}

pub struct RunsProof {
    pub proof: crate::leanxmss::circuit::CircuitProof,
    pub commitment: Commitment,
}

impl RunsCircuit {
    pub fn new(n: usize, k: usize) -> Self {
        assert!(n >= 1 && (1..=16).contains(&k));
        let nu = (n * k).next_power_of_two().trailing_zeros().max(3) as usize;
        let mut sb = ShapeBuilder::new(nu);
        sb.prune_unwired();
        sb.open_public_columns();
        sb.pack_live_slots();
        let cg = compress_gate(nu).unwrap().with_zeroed_linear_words();
        let rg = runs_gate(nu).unwrap().with_zeroed_linear_words();
        let cs = sb.slot(cg.clone());
        let rs = sb.slot(rg.clone());
        let iv = Self::constants()[..2].iter().map(|&v| sb.fixed_public_input(v)).collect::<Vec<_>>();
        let consts: Vec<Wire> =
            Self::constants()[2..].iter().map(|&v| sb.fixed_public_input(v)).collect();
        let (zero, one) = (consts[0], consts[1]);
        let flag = |j: usize| {
            consts[2 + usize::from(j == 0) + 2 * usize::from(j == k - 1)]
        };
        for _ in 0..n {
            let island = sb.begin_island();
            let key = [sb.public_input(), sb.public_input()];
            let mut cv = [iv[0], iv[1]];
            let mut prev = [zero, zero];
            for j in 0..k {
                let m: [Wire; 4] = std::array::from_fn(|_| sb.input());
                let out = sb.gate(cs, &[cv[0], cv[1], m[0], m[1], m[2], m[3], flag(j)]);
                cv = [out[0], out[1]];
                sb.gate(rs, &[m[0], m[1], m[2], m[3], prev[0], prev[1], if j == 0 { one } else { zero }]);
                prev = [m[0], m[1]];
            }
            sb.connect(cv[0], key[0]);
            sb.connect(cv[1], key[1]);
            sb.end_island(island);
        }
        let shape = sb.finish().expect("well formed");
        let plan = shape.fill_plan();
        let packer = CompressPacker::new(&cg);
        RunsCircuit { n, k, shape, plan, gadgets: vec![(cs, cg), (rs, rg)], packer }
    }

    /// IV (2), zero, one, then the control words indexed `first + 2·last`.
    fn constants() -> [F128; 8] {
        let iv = |i: usize| {
            F128::new(
                BLAKE3_IV[i] as u64 | (BLAKE3_IV[i + 1] as u64) << 32,
                BLAKE3_IV[i + 2] as u64 | (BLAKE3_IV[i + 3] as u64) << 32,
            )
        };
        [
            iv(0),
            iv(4),
            F128::new(0, 0),
            F128::new(1, 0),
            ctl(0, 64),
            ctl(CHUNK_START, 64),
            ctl(CHUNK_END | ROOT, 64),
            ctl(CHUNK_START | CHUNK_END | ROOT, 64),
        ]
    }

    /// Input values in declaration order; `publics` selects the public ones.
    fn values(&self, keys: &[[u8; 32]], blobs: Option<&[Vec<u8>]>) -> Vec<F128> {
        let mut v = Self::constants().to_vec();
        for s in 0..self.n {
            v.extend(key_words(&keys[s]));
            if let Some(blobs) = blobs {
                assert_eq!(blobs[s].len(), 64 * self.k);
                for block in blobs[s].chunks_exact(64) {
                    v.extend(words(block));
                }
            }
        }
        v
    }

    pub fn witness(&self, keys: &[[u8; 32]], blobs: &[Vec<u8>]) -> CircuitWitness {
        self.shape.run_filled_deferred(&self.plan, &self.values(keys, Some(blobs)), &[])
    }

    fn union(&self) -> UnionInstance<'_> {
        let mut union = UnionInstance::new(&self.shape.registry, self.shape.counts.clone());
        union.set_full_height_commit();
        union
    }

    fn challenger() -> FsChallenger {
        FsChallenger::new(b"cas-runs-blake3-v0")
    }

    pub fn prove(&self, witness: &CircuitWitness) -> RunsProof {
        let union = self.union();
        let pcs = XmssCircuit::pcs_params(&union);
        let mut by_slot: Vec<(usize, UnionSlotProverInput<'_>)> = self
            .gadgets
            .iter()
            .map(|(id, gate)| {
                let rows = witness.rows::<BooleanGate>(*id);
                let packer = (gate.name() == compress::NAME).then_some(&self.packer);
                (
                    self.shape.registry_slot(*id),
                    UnionSlotProverInput::in_place(
                        move |dst| match packer {
                            Some(p) => p.pack_into(rows, dst),
                            None => gate.pack_into(rows, dst),
                        },
                        gate.r1cs().csc_lincheck_circuit(),
                    ),
                )
            })
            .collect();
        by_slot.sort_by_key(|&(slot, _)| slot);
        let mut challenger = Self::challenger();
        let (proof, commitment, _) = crate::leanxmss::circuit::prove_circuit(
            &union,
            &self.shape.circuit,
            &witness.public,
            &pcs,
            by_slot.into_iter().map(|(_, i)| i).collect(),
            Vec::new(),
            &mut challenger,
        );
        RunsProof { proof, commitment }
    }

    pub fn verify(&self, keys: &[[u8; 32]], proof: &RunsProof) -> Result<(), FlockVerifyError> {
        let union = self.union();
        let pcs = XmssCircuit::pcs_params(&union);
        let mut by_slot: Vec<(usize, &dyn LincheckCircuit)> = self
            .gadgets
            .iter()
            .map(|(id, g)| (self.shape.registry_slot(*id), g.r1cs().csc_lincheck_circuit() as _))
            .collect();
        by_slot.sort_by_key(|&(slot, _)| slot);
        let circuits: Vec<&dyn LincheckCircuit> = by_slot.into_iter().map(|(_, c)| c).collect();
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
