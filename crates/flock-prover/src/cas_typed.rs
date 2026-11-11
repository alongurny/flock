//! Blobs of a cas type proven valid: each hashes (BLAKE3) to its public key
//! and passes the rows that [`crate::cas_ir`] derives from the type. A
//! type's row takes the same four block words the compression row hashes,
//! so the checked bytes are the hashed bytes.
//!
//! A blob of any length is hashed as BLAKE3 does: chunks of 16 blocks, each
//! with its index as the counter, joined by parent rows.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use flock_core::{
    challenger::FsChallenger,
    circuit::{
        boolean::{
            BooleanCircuit, BooleanGate, CircuitBuilder, ColumnRole, ColumnSchema, ColumnVisitor,
            LinearExpr as Expr,
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
    cas_ir::{self, Circuit, Kind, Node, STATE, Type},
    leanxmss::{
        circuit::XmssCircuit,
        compress::{self, CompressPacker, compress_gate, ctl, ctl_at},
    },
    prover::UnionSlotProverInput,
    r1cs_hashes::blake3::BLAKE3_IV,
};

const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;

/// Compressions to hash a blob of `blocks` blocks: its blocks, and a parent
/// for each chunk after the first.
pub fn compressions(blocks: usize) -> usize {
    blocks.max(1) + blocks.div_ceil(16).max(1) - 1
}

struct IrSchema {
    inputs: usize,
    outputs: usize,
    ands: usize,
}

struct IrCols<T> {
    i: Vec<[T; 128]>,
    o: Vec<[T; 128]>,
    w: Vec<T>,
}

impl ColumnSchema for IrSchema {
    type Cols<T> = IrCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> IrCols<V::Value> {
        IrCols {
            i: (0..self.inputs).map(|n| v.word_aligned(&format!("i{n}"), ColumnRole::Input, 128)).collect(),
            o: (0..self.outputs).map(|n| v.word_aligned(&format!("o{n}"), ColumnRole::Output, 128)).collect(),
            w: if self.ands > 0 { v.bits("w", ColumnRole::Witness, self.ands, 1) } else { Vec::new() },
        }
    }
}

/// The circuit as Flock constraints: one product per AND node, XORs free.
pub fn lower(c: &Circuit) -> BooleanCircuit {
    CircuitBuilder::compile(IrSchema { inputs: c.inputs, outputs: c.outputs, ands: c.ands }, |b, cols| {
        let mut e: Vec<Expr> = Vec::with_capacity(c.nodes.len());
        let mut w = cols.w.iter();
        for (n, node) in c.nodes.iter().enumerate() {
            let x = match *node {
                Node::Const if n == 0 => b.zero(),
                Node::Const => b.one().into(),
                Node::In(j) => cols.i[j as usize / 128][j as usize % 128].into(),
                Node::Xor(x, y) => b.xor2(e[x.0 as usize], e[y.0 as usize]),
                Node::And(x, y) => {
                    let out = *w.next().unwrap();
                    b.define_and(out, e[x.0 as usize], e[y.0 as usize]);
                    out.into()
                }
            };
            e.push(x);
        }
        for z in &c.zero {
            if z.0 != 0 {
                b.assert_zero(e[z.0 as usize]);
            }
        }
        for (j, o) in c.out.iter().enumerate() {
            b.define_linear(cols.o[j / 128][j % 128], e[o.0 as usize]);
        }
    })
    .circuit()
    .clone()
}

/// The compression gate and its packer, built once for each table size.
fn compression(nu: usize) -> (BooleanGate, Arc<CompressPacker>) {
    static GATES: Mutex<Option<HashMap<usize, (BooleanGate, Arc<CompressPacker>)>>> = Mutex::new(None);
    if let Some(g) = GATES.lock().unwrap().get_or_insert_default().get(&nu) {
        return g.clone();
    }
    let g = compress_gate(nu).unwrap().with_zeroed_linear_words();
    let made = (g.clone(), Arc::new(CompressPacker::new(&g)));
    GATES.lock().unwrap().get_or_insert_default().insert(nu, made.clone());
    made
}

/// The gate of one kind of row of a type, built once for each table size.
fn gate(t: &Type, kind: Kind, nu: usize, native: bool) -> (BooleanGate, usize, usize) {
    type Key = (String, usize, bool);
    static GATES: Mutex<Option<HashMap<Key, (BooleanGate, usize, usize)>>> = Mutex::new(None);
    let key = (format!("{}/{kind:?}", t.alias), nu, native);
    if let Some(g) = GATES.lock().unwrap().get_or_insert_default().get(&key) {
        return g.clone();
    }
    let c = cas_ir::circuit(t, kind);
    let ins: Vec<String> = (0..c.inputs).map(|n| format!("i{n}")).collect();
    let outs: Vec<String> = (0..c.outputs).map(|n| format!("o{n}")).collect();
    let (ins, outs): (Vec<&str>, Vec<&str>) =
        (ins.iter().map(String::as_str).collect(), outs.iter().map(String::as_str).collect());
    let mut g = BooleanGate::new(&c.name, &lower(&c), &ins, &outs, nu)
        .unwrap_or_else(|e| panic!("{}: {e:?}", c.name))
        .with_zeroed_linear_words();
    let (inputs, ands) = (c.inputs, c.ands);
    if native {
        let c = Arc::new(c);
        g = g.with_native(move |i, o| {
            let words: Vec<u128> = i.iter().map(|w| u128::from(w.lo) | u128::from(w.hi) << 64).collect();
            let out = c.eval(&words).unwrap_or_else(|| panic!("{}: not valid", c.name));
            o.extend(out.iter().map(|&w| F128::new(w as u64, (w >> 64) as u64)));
        });
    }
    GATES.lock().unwrap().get_or_insert_default().insert(key, (g.clone(), inputs, ands));
    (g, inputs, ands)
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

/// Blobs, each of a type and a number of bytes, in one proof.
pub struct TypedCircuit {
    /// The bytes of each blob.
    lens: Vec<usize>,
    /// The words the circuit fixes, in the order it declares them.
    consts: Vec<F128>,
    shape: CircuitShape,
    plan: FillPlan,
    gadgets: Vec<(SlotId, BooleanGate)>,
    packer: Arc<CompressPacker>,
    /// AND gates of the types' rows over all blobs.
    pub ands: usize,
    /// Rows: compressions, then the types' rows, then padding rows.
    pub rows: (usize, usize, usize),
    /// The rows of each table, as a power of two.
    pub nu: usize,
}

pub struct TypedProof {
    pub proof: crate::leanxmss::circuit::CircuitProof,
    pub commitment: Commitment,
}

/// A table's rows must be a multiple of this: this Flock revision fails its
/// own check on other counts.
const ROWS: usize = 4;

/// The words a circuit fixes, each once.
#[derive(Default)]
struct Consts {
    values: Vec<F128>,
    index: HashMap<(u64, u64), usize>,
}

impl Consts {
    fn add(&mut self, v: F128) {
        let n = self.values.len();
        if let std::collections::hash_map::Entry::Vacant(e) = self.index.entry((v.lo, v.hi)) {
            e.insert(n);
            self.values.push(v);
        }
    }
}

/// The control word of block `j` of the `n` blocks of chunk `chunk` of a
/// blob of `len` bytes.
fn block_ctl(len: usize, chunk: usize, j: usize, n: usize) -> F128 {
    let last = j + 1 == n;
    let end = (64 * (16 * chunk + j + 1)).min(len);
    let bytes = end - 64 * (16 * chunk + j);
    let mut flags = 0;
    if j == 0 {
        flags |= CHUNK_START;
    }
    if last {
        flags |= CHUNK_END;
        // Only a blob of one chunk ends at the root in a block.
        if len <= 1024 {
            flags |= ROOT;
        }
    }
    ctl_at(flags, bytes as u32, chunk as u64)
}

fn iv() -> [F128; 2] {
    let iv = |i: usize| {
        F128::new(
            BLAKE3_IV[i] as u64 | (BLAKE3_IV[i + 1] as u64) << 32,
            BLAKE3_IV[i + 2] as u64 | (BLAKE3_IV[i + 3] as u64) << 32,
        )
    };
    [iv(0), iv(4)]
}

fn word(w: u128) -> F128 {
    F128::new(w as u64, (w >> 64) as u64)
}

impl TypedCircuit {
    /// With `native`, the witness comes from evaluating the rows directly,
    /// which panics on a blob that is not valid; without it, from walking
    /// the constraints, which gives a witness (a bad one) for any blob.
    pub fn new(blobs: &[(&Type, usize)], native: bool) -> Self {
        let plans: Vec<Vec<cas_ir::Row>> = blobs
            .iter()
            .map(|&(t, len)| cas_ir::plan(t, len).unwrap_or_else(|| panic!("no {} has {len} bytes", t.alias)))
            .collect();
        let lens: Vec<usize> = blobs.iter().map(|b| b.1).collect();
        let compressions: usize = lens.iter().map(|&len| compressions(len.div_ceil(64))).sum();
        // The tables: one for each kind of row of each type that has rows.
        let mut tables: Vec<(&Type, Kind, usize)> = Vec::new();
        for (&(t, _), plan) in blobs.iter().zip(&plans) {
            for r in plan {
                match tables.iter_mut().find(|x| x.0.alias == t.alias && x.1 == r.kind) {
                    Some(x) => x.2 += 1,
                    None => tables.push((t, r.kind, 1)),
                }
            }
        }
        // Every word the circuit fixes, found before anything is built.
        let (zero, one) = (F128::new(0, 0), F128::new(1, 0));
        let empty = ctl(CHUNK_START | CHUNK_END | ROOT, 0);
        let mut consts = Consts::default();
        for v in [iv()[0], iv()[1], zero, one, empty, ctl(PARENT, 64), ctl(PARENT | ROOT, 64)] {
            consts.add(v);
        }
        for (&len, plan) in lens.iter().zip(&plans) {
            let blocks = len.div_ceil(64);
            for chunk in 0..blocks.div_ceil(16) {
                let n = (blocks - 16 * chunk).min(16);
                for j in 0..n {
                    consts.add(block_ctl(len, chunk, j, n));
                }
            }
            for r in plan.iter().filter(|r| r.kind == Kind::Text) {
                consts.add(word(cas_ir::inside(r.bytes)));
            }
        }

        let most = tables.iter().map(|x| x.2).chain([compressions]).max().unwrap().next_multiple_of(ROWS);
        let nu = most.next_power_of_two().trailing_zeros().max(8) as usize;
        let mut sb = ShapeBuilder::new(nu);
        sb.prune_unwired();
        sb.open_public_columns();
        sb.pack_live_slots();
        let (cg, packer) = compression(nu);
        let cs = sb.slot(cg.clone());
        let mut gadgets = vec![(cs, cg)];
        let mut ands = 0;
        // Each table's slot and the words its row takes.
        let slots: Vec<(SlotId, usize)> = tables
            .iter()
            .map(|&(t, kind, count)| {
                let (g, inputs, gates) = gate(t, kind, nu, native);
                ands += gates * count;
                let s = sb.slot(g.clone());
                gadgets.push((s, g));
                (s, inputs)
            })
            .collect();
        let slot = |alias: &str, kind: Kind| slots[tables.iter().position(|x| x.0.alias == alias && x.1 == kind).unwrap()].0;
        let wires: Vec<Wire> = consts.values.iter().map(|&v| sb.fixed_public_input(v)).collect();
        let wire = |v: F128| wires[consts.index[&(v.lo, v.hi)]];
        let (iv, zero, one) = ([wire(iv()[0]), wire(iv()[1])], wire(zero), wire(one));
        let bit = |b: bool| if b { one } else { zero };
        // Padding rows: a compression of nothing, and rows turned off.
        let mut padding = 0;
        for _ in compressions..compressions.next_multiple_of(ROWS) {
            sb.gate(cs, &[iv[0], iv[1], zero, zero, zero, zero, wire(empty)]);
            padding += 1;
        }
        for (&(_, _, count), &(s, inputs)) in tables.iter().zip(&slots) {
            let mut row = vec![zero; inputs];
            row[inputs - 1] = one;
            for _ in count..count.next_multiple_of(ROWS) {
                sb.gate(s, &row);
                padding += 1;
            }
        }
        let mut rows = 0;
        for (&(t, len), plan) in blobs.iter().zip(&plans) {
            let island = sb.begin_island();
            let key = [sb.public_input(), sb.public_input()];
            let blocks = len.div_ceil(64);
            let m: Vec<[Wire; 4]> = (0..blocks).map(|_| std::array::from_fn(|_| sb.input())).collect();
            // The chaining value of each chunk, then the chunks joined as
            // BLAKE3 joins them: the left side takes the largest power of
            // two below the count.
            let mut leaves: Vec<[Wire; 2]> = Vec::new();
            for chunk in 0..blocks.div_ceil(16).max(1) {
                let mut cv = iv;
                if blocks == 0 {
                    let out = sb.gate(cs, &[cv[0], cv[1], zero, zero, zero, zero, wire(empty)]);
                    cv = [out[0], out[1]];
                }
                let n = (blocks - 16 * chunk).min(16);
                for j in 0..n {
                    let b = m[16 * chunk + j];
                    let out = sb.gate(cs, &[cv[0], cv[1], b[0], b[1], b[2], b[3], wire(block_ctl(len, chunk, j, n))]);
                    cv = [out[0], out[1]];
                }
                leaves.push(cv);
            }
            fn join(sb: &mut ShapeBuilder, cs: SlotId, iv: [Wire; 2], ctl: [Wire; 2], l: &[[Wire; 2]], root: bool) -> [Wire; 2] {
                if l.len() == 1 {
                    return l[0];
                }
                let left = l.len().next_power_of_two() / 2;
                let a = join(sb, cs, iv, ctl, &l[..left], false);
                let b = join(sb, cs, iv, ctl, &l[left..], false);
                let out = sb.gate(cs, &[iv[0], iv[1], a[0], a[1], b[0], b[1], ctl[usize::from(root)]]);
                [out[0], out[1]]
            }
            let parents = [wire(ctl(PARENT, 64)), wire(ctl(PARENT | ROOT, 64))];
            let cv = join(&mut sb, cs, iv, parents, &leaves, true);
            sb.connect(cv[0], key[0]);
            sb.connect(cv[1], key[1]);
            let mut state: Vec<Wire> = vec![zero; STATE];
            state[STATE - 1] = one;
            let mut text = one;
            let mut prev = [zero, zero];
            for r in plan {
                rows += 1;
                let mut inputs: Vec<Wire> = m[r.block..r.block + r.blocks].iter().flatten().copied().collect();
                let s = slot(t.alias, r.kind);
                // The last word, zero, leaves the row on.
                match r.kind {
                    Kind::Head => {
                        inputs.push(zero);
                        sb.gate(s, &inputs);
                    }
                    Kind::Rec => {
                        inputs.extend([prev[0], prev[1], bit(r.first), zero]);
                        sb.gate(s, &inputs);
                        prev = [m[r.block][0], m[r.block][1]];
                    }
                    Kind::Step => {
                        inputs.extend(&state);
                        inputs.extend([bit(r.last), zero]);
                        state = sb.gate(s, &inputs);
                    }
                    Kind::Text => {
                        inputs.extend([text, wire(word(cas_ir::inside(r.bytes))), bit(r.last), zero]);
                        text = sb.gate(s, &inputs)[0];
                    }
                }
            }
            sb.end_island(island);
        }
        let shape = sb.finish().expect("well formed");
        let plan = shape.fill_plan();
        let consts = consts.values;
        TypedCircuit { lens, consts, shape, plan, gadgets, packer, ands, rows: (compressions, rows, padding), nu }
    }

    /// Input values in declaration order; without `blobs`, the public ones.
    fn values(&self, keys: &[[u8; 32]], blobs: Option<&[Vec<u8>]>) -> Vec<F128> {
        let mut v = self.consts.clone();
        for (s, &len) in self.lens.iter().enumerate() {
            v.extend(key_words(&keys[s]));
            if let Some(blobs) = blobs {
                assert_eq!(blobs[s].len(), len);
                let mut b = blobs[s].clone();
                b.resize(len.next_multiple_of(64), 0);
                for block in b.chunks_exact(64) {
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
        FsChallenger::new(b"cas-typed-blake3-v0")
    }

    pub fn prove(&self, witness: &CircuitWitness) -> TypedProof {
        let union = self.union();
        let pcs = XmssCircuit::pcs_params(&union);
        let mut by_slot: Vec<(usize, UnionSlotProverInput<'_>)> = self
            .gadgets
            .iter()
            .map(|(id, gate)| {
                let rows = witness.rows::<BooleanGate>(*id);
                let packer = (gate.name() == compress::NAME).then_some(&*self.packer);
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
        TypedProof { proof, commitment }
    }

    pub fn verify(&self, keys: &[[u8; 32]], proof: &TypedProof) -> Result<(), FlockVerifyError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_blobs_prove() {
        let types = cas_ir::types();
        let t = types.iter().find(|t| t.alias == "runs").unwrap();
        let blobs: Vec<Vec<u8>> = (0..117u8)
            .map(|i| {
                let k = 1 + i % 16;
                let mut b = vec![0u8; 64 * k as usize];
                for (j, r) in b.chunks_exact_mut(64).enumerate() {
                    r[..32].fill(j as u8 + 1);
                    r[39] = k;
                }
                b
            })
            .collect();
        let c = cas_ir::circuits(t);
        assert!(blobs.iter().all(|b| cas_ir::check(t, &c, b)));
        let keys: Vec<[u8; 32]> = blobs.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        let blocks: Vec<usize> = blobs.iter().map(|b| b.len() / 64).collect();
        for native in [true, false] {
            let typed: Vec<(&Type, usize)> = blocks.iter().map(|&k| (t, 64 * k)).collect();
            let circuit = TypedCircuit::new(&typed, native);
            let w = circuit.witness(&keys, &blobs);
            let proof = circuit.prove(&w);
            circuit.verify(&keys, &proof).expect("verifies");
        }
    }
}
