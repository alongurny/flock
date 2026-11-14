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
        bus::{BusDirection, BusTerm},
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

pub(crate) const CHUNK_START: u32 = 1;
pub(crate) const CHUNK_END: u32 = 2;
pub(crate) const PARENT: u32 = 4;
pub(crate) const ROOT: u32 = 8;

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
            i: (0..self.inputs)
                .map(|n| v.word_aligned(&format!("i{n}"), ColumnRole::Input, 128))
                .collect(),
            o: (0..self.outputs)
                .map(|n| v.word_aligned(&format!("o{n}"), ColumnRole::Output, 128))
                .collect(),
            w: if self.ands > 0 {
                v.bits("w", ColumnRole::Witness, self.ands, 1)
            } else {
                Vec::new()
            },
        }
    }
}

/// The circuit as Flock constraints: one product per AND node, XORs free.
pub fn lower(c: &Circuit) -> BooleanCircuit {
    CircuitBuilder::compile(
        IrSchema {
            inputs: c.inputs,
            outputs: c.outputs,
            ands: c.ands,
        },
        |b, cols| {
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
        },
    )
    .circuit()
    .clone()
}

/// The compression gate and its packer, built once for each table size.
pub(crate) fn compression(nu: usize) -> (BooleanGate, Arc<CompressPacker>) {
    static GATES: Mutex<Option<HashMap<usize, (BooleanGate, Arc<CompressPacker>)>>> =
        Mutex::new(None);
    if let Some(g) = GATES.lock().unwrap().get_or_insert_default().get(&nu) {
        return g.clone();
    }
    let g = compress_gate(nu).unwrap().with_zeroed_linear_words();
    let made = (g.clone(), Arc::new(CompressPacker::new(&g)));
    GATES
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(nu, made.clone());
    made
}

/// The gate of one kind of row of a type, built once for each table size.
fn gate(t: &Type, kind: Kind, nu: usize, native: bool) -> (BooleanGate, usize, usize) {
    gate_of(format!("{}/{kind:?}", t.alias), nu, native, || {
        cas_ir::circuit(t, kind)
    })
}

/// The key rows' gate ([`cas_ir::key_circuit`]).
fn key_gate(bits: usize, nu: usize, native: bool) -> (BooleanGate, usize, usize) {
    gate_of(format!("key/{bits}"), nu, native, || {
        cas_ir::key_circuit(bits)
    })
}

/// A row circuit's gate, built once for each name and table size: the gate,
/// its input words and its AND gates.
fn gate_of(
    name: String,
    nu: usize,
    native: bool,
    make: impl FnOnce() -> Circuit,
) -> (BooleanGate, usize, usize) {
    type Key = (String, usize, bool);
    static GATES: Mutex<Option<HashMap<Key, (BooleanGate, usize, usize)>>> = Mutex::new(None);
    let key = (name, nu, native);
    if let Some(g) = GATES.lock().unwrap().get_or_insert_default().get(&key) {
        return g.clone();
    }
    let c = make();
    let ins: Vec<String> = (0..c.inputs).map(|n| format!("i{n}")).collect();
    let outs: Vec<String> = (0..c.outputs).map(|n| format!("o{n}")).collect();
    let (ins, outs): (Vec<&str>, Vec<&str>) = (
        ins.iter().map(String::as_str).collect(),
        outs.iter().map(String::as_str).collect(),
    );
    let mut g = BooleanGate::new(&c.name, &lower(&c), &ins, &outs, nu)
        .unwrap_or_else(|e| panic!("{}: {e:?}", c.name))
        .with_zeroed_linear_words();
    let (inputs, ands) = (c.inputs, c.ands);
    if native {
        let c = Arc::new(c);
        g = g.with_native(move |i, o| {
            let words: Vec<u128> = i
                .iter()
                .map(|w| u128::from(w.lo) | u128::from(w.hi) << 64)
                .collect();
            let out = c
                .eval(&words)
                .unwrap_or_else(|| panic!("{}: not valid", c.name));
            o.extend(out.iter().map(|&w| F128::new(w as u64, (w >> 64) as u64)));
        });
    }
    GATES
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(key, (g.clone(), inputs, ands));
    (g, inputs, ands)
}

pub(crate) fn words(block: &[u8]) -> [F128; 4] {
    std::array::from_fn(|i| {
        let w =
            |o: usize| u64::from_le_bytes(block[16 * i + o..16 * i + o + 8].try_into().unwrap());
        F128::new(w(0), w(8))
    })
}

pub(crate) fn key_words(key: &[u8; 32]) -> [F128; 2] {
    let w = |o: usize| u64::from_le_bytes(key[o..o + 8].try_into().unwrap());
    [F128::new(w(0), w(8)), F128::new(w(16), w(24))]
}

/// What a blob is to the key lookup ([`TypedCircuit::with_lookup`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Brought by the proof: each key it holds must be found, and its own
    /// key is found by any number of keys that name it.
    New,
    /// Opened from a parent: its own key must be found, among the keys that
    /// opened blobs and parents hold, and the keys it holds are found by
    /// any number of keys that name them.
    Opened,
    /// A parent node, valid by a proof of its own: the keys it holds are
    /// found as an opened blob's are; its own key is not looked up.
    Parent,
}

/// One blob of a proof with the key lookup: its type, bytes, role, and the
/// code of its type ([`cas_ir::code`], the manifest's tag byte).
#[derive(Clone, Copy)]
pub struct Blob<'a> {
    pub t: &'a Type,
    pub len: usize,
    pub role: Role,
    pub code: u8,
}

/// Where a key row's key comes from.
#[derive(Clone, Copy)]
enum Src {
    /// Blob `blob`'s own key.
    Own { blob: usize },
    /// Key `field` of row `row` of blob `blob`'s plan.
    Held {
        blob: usize,
        row: usize,
        field: usize,
    },
}

/// The key lookup: the count's bits, and each key row's source, whether its
/// key must be found, and whether it is found by others (its count bits are
/// inputs).
struct Lookup {
    bits: usize,
    rows: Vec<(Src, bool, bool)>,
    /// Each blob's type code.
    codes: Vec<u8>,
}

/// Blobs, each of a type and a number of bytes, in one proof.
pub struct TypedCircuit {
    /// The bytes of each blob.
    lens: Vec<usize>,
    /// Each blob's type, by alias.
    types: Vec<&'static str>,
    lookup: Option<Lookup>,
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
pub(crate) const ROWS: usize = 4;

/// The words a circuit fixes, each once.
#[derive(Default)]
pub(crate) struct Consts {
    pub(crate) values: Vec<F128>,
    pub(crate) index: HashMap<(u64, u64), usize>,
}

impl Consts {
    pub(crate) fn add(&mut self, v: F128) {
        let n = self.values.len();
        if let std::collections::hash_map::Entry::Vacant(e) = self.index.entry((v.lo, v.hi)) {
            e.insert(n);
            self.values.push(v);
        }
    }
}

/// The control word of block `j` of the `n` blocks of chunk `chunk` of a
/// blob of `len` bytes.
pub(crate) fn block_ctl(len: usize, chunk: usize, j: usize, n: usize) -> F128 {
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

pub(crate) fn iv() -> [F128; 2] {
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
    /// The keys the blobs hold are not looked up.
    pub fn new(blobs: &[(&Type, usize)], native: bool) -> Self {
        let blobs: Vec<Blob<'_>> = blobs
            .iter()
            .map(|&(t, len)| Blob {
                t,
                len,
                role: Role::New,
                code: 0,
            })
            .collect();
        Self::build(&blobs, false, native)
    }

    /// As [`Self::new`], and each key a [`Role::New`] blob holds that names a
    /// type a proof covers is found, with that type, as the own key of a new
    /// blob or among the keys that opened blobs and parents hold; so is each
    /// opened blob's own key. The match is the tuple bus with counts
    /// (multiplicities): each found key is sent once, and a key row that
    /// can be found sends its tuple as many times as it is asked for.
    pub fn with_lookup(blobs: &[Blob<'_>], native: bool) -> Self {
        Self::build(blobs, true, native)
    }

    fn build(blobs: &[Blob<'_>], lookup: bool, native: bool) -> Self {
        let plans: Vec<Vec<cas_ir::Row>> = blobs
            .iter()
            .map(|b| {
                cas_ir::plan(b.t, b.len)
                    .unwrap_or_else(|| panic!("no {} has {} bytes", b.t.alias, b.len))
            })
            .collect();
        let lens: Vec<usize> = blobs.iter().map(|b| b.len).collect();
        let compressions: usize = lens.iter().map(|&len| compressions(len.div_ceil(64))).sum();
        // The tables: one for each kind of row of each type that has rows.
        let mut tables: Vec<(&Type, Kind, usize)> = Vec::new();
        for (b, plan) in blobs.iter().zip(&plans) {
            for r in plan {
                match tables
                    .iter_mut()
                    .find(|x| x.0.alias == b.t.alias && x.1 == r.kind)
                {
                    Some(x) => x.2 += 1,
                    None => tables.push((b.t, r.kind, 1)),
                }
            }
        }
        // The key rows, and the bits a count needs: no key is asked for
        // more often than there are keys that must be found.
        let mut key_rows: Vec<(Src, bool, bool)> = Vec::new();
        if lookup {
            for (i, (b, plan)) in blobs.iter().zip(&plans).enumerate() {
                match b.role {
                    Role::New => key_rows.push((Src::Own { blob: i }, false, true)),
                    Role::Opened => key_rows.push((Src::Own { blob: i }, true, false)),
                    Role::Parent => {}
                }
                for (row, r) in plan.iter().enumerate() {
                    let fields = match r.kind {
                        Kind::Text => 0,
                        Kind::Step => 1,
                        Kind::Head => b.t.head.as_ref().unwrap().keys.len(),
                        Kind::Rec => b.t.list.as_ref().unwrap().0.keys.len(),
                    };
                    let new = b.role == Role::New;
                    key_rows.extend((0..fields).map(|field| {
                        (
                            Src::Held {
                                blob: i,
                                row,
                                field,
                            },
                            new,
                            !new,
                        )
                    }));
                }
            }
        }
        let asked = key_rows.iter().filter(|r| r.1).count();
        let bits = (usize::BITS - asked.leading_zeros()).max(1) as usize;
        // Every word the circuit fixes, found before anything is built.
        let (zero, one) = (F128::new(0, 0), F128::new(1, 0));
        let empty = ctl(CHUNK_START | CHUNK_END | ROOT, 0);
        let mut consts = Consts::default();
        for v in [
            iv()[0],
            iv()[1],
            zero,
            one,
            empty,
            ctl(PARENT, 64),
            ctl(PARENT | ROOT, 64),
        ] {
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
        if lookup {
            for b in blobs {
                consts.add(F128::new(u64::from(b.code), 0));
            }
        }

        let most = tables
            .iter()
            .map(|x| x.2)
            .chain([compressions, key_rows.len()])
            .max()
            .unwrap()
            .next_multiple_of(ROWS);
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
        // The key rows' slot and its ports: (key, code) asked for once when
        // the row's output says so, and given as often as its count says.
        let keys_slot = lookup.then(|| {
            let (g, inputs, gates) = key_gate(bits, nu, native);
            ands += gates * key_rows.len();
            let s = sb.slot(g.clone());
            gadgets.push((s, g));
            let tuple = || vec![BusTerm::word(0), BusTerm::word(1), BusTerm::word(2)];
            sb.bus_port_counted(s, BusDirection::Recv, tuple(), vec![inputs]);
            sb.bus_port_counted(s, BusDirection::Send, tuple(), (4..4 + bits).collect());
            (s, inputs)
        });
        let slot = |alias: &str, kind: Kind| {
            slots[tables
                .iter()
                .position(|x| x.0.alias == alias && x.1 == kind)
                .unwrap()]
            .0
        };
        let wires: Vec<Wire> = consts
            .values
            .iter()
            .map(|&v| sb.fixed_public_input(v))
            .collect();
        let wire = |v: F128| wires[consts.index[&(v.lo, v.hi)]];
        let (iv, zero, one) = ([wire(iv()[0]), wire(iv()[1])], wire(zero), wire(one));
        let bit = |b: bool| if b { one } else { zero };
        // Padding rows: a compression of nothing, and rows turned off.
        let mut padding = 0;
        for _ in compressions..compressions.next_multiple_of(ROWS) {
            sb.gate(cs, &[iv[0], iv[1], zero, zero, zero, zero, wire(empty)]);
            padding += 1;
        }
        let pads = tables
            .iter()
            .map(|x| x.2)
            .zip(&slots)
            .map(|(c, &s)| (c, s))
            .chain(keys_slot.map(|s| (key_rows.len(), s)));
        for (count, (s, inputs)) in pads {
            let mut row = vec![zero; inputs];
            row[inputs - 1] = one;
            for _ in count..count.next_multiple_of(ROWS) {
                sb.gate(s, &row);
                padding += 1;
            }
        }
        let mut rows = 0;
        // Each blob's own key, and each row's held keys with their codes.
        let mut own: Vec<[Wire; 2]> = Vec::new();
        let mut held: Vec<Vec<Vec<([Wire; 2], Wire)>>> = Vec::new();
        for (b, plan) in blobs.iter().zip(&plans) {
            let (t, len) = (b.t, b.len);
            let island = sb.begin_island();
            let key = [sb.public_input(), sb.public_input()];
            own.push(key);
            let blocks = len.div_ceil(64);
            let m: Vec<[Wire; 4]> = (0..blocks)
                .map(|_| std::array::from_fn(|_| sb.input()))
                .collect();
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
                    let out = sb.gate(
                        cs,
                        &[
                            cv[0],
                            cv[1],
                            b[0],
                            b[1],
                            b[2],
                            b[3],
                            wire(block_ctl(len, chunk, j, n)),
                        ],
                    );
                    cv = [out[0], out[1]];
                }
                leaves.push(cv);
            }
            fn join(
                sb: &mut ShapeBuilder,
                cs: SlotId,
                iv: [Wire; 2],
                ctl: [Wire; 2],
                l: &[[Wire; 2]],
                root: bool,
            ) -> [Wire; 2] {
                if l.len() == 1 {
                    return l[0];
                }
                let left = l.len().next_power_of_two() / 2;
                let a = join(sb, cs, iv, ctl, &l[..left], false);
                let b = join(sb, cs, iv, ctl, &l[left..], false);
                let out = sb.gate(
                    cs,
                    &[iv[0], iv[1], a[0], a[1], b[0], b[1], ctl[usize::from(root)]],
                );
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
            // Key `i` of a record that starts at block `at`.
            let field = |at: usize, i: usize| -> [Wire; 2] {
                let w = m[at + i / 2];
                [w[2 * (i % 2)], w[2 * (i % 2) + 1]]
            };
            let mut blob_held = Vec::new();
            for r in plan {
                rows += 1;
                let mut inputs: Vec<Wire> = m[r.block..r.block + r.blocks]
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                let s = slot(t.alias, r.kind);
                // The last word, zero, leaves the row on.
                blob_held.push(match r.kind {
                    Kind::Head => {
                        inputs.push(zero);
                        let codes = sb.gate(s, &inputs);
                        codes
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| (field(r.block, i), c))
                            .collect()
                    }
                    Kind::Rec => {
                        inputs.extend([prev[0], prev[1], bit(r.first), zero]);
                        let codes = sb.gate(s, &inputs);
                        prev = [m[r.block][0], m[r.block][1]];
                        codes
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| (field(r.block, i), c))
                            .collect()
                    }
                    Kind::Step => {
                        inputs.extend(&state);
                        inputs.extend([bit(r.last), zero]);
                        let out = sb.gate(s, &inputs);
                        state = out[..STATE].to_vec();
                        vec![(field(r.block, 0), out[STATE])]
                    }
                    Kind::Text => {
                        inputs.extend([
                            text,
                            wire(word(cas_ir::inside(r.bytes))),
                            bit(r.last),
                            zero,
                        ]);
                        text = sb.gate(s, &inputs)[0];
                        Vec::new()
                    }
                });
            }
            held.push(blob_held);
            sb.end_island(island);
        }
        // The key rows, their counts the last inputs declared.
        if let Some((s, _)) = keys_slot {
            for &(src, must, counts) in &key_rows {
                let (k, code) = match src {
                    Src::Own { blob } => {
                        (own[blob], wire(F128::new(u64::from(blobs[blob].code), 0)))
                    }
                    Src::Held { blob, row, field } => held[blob][row][field],
                };
                let mut inputs = vec![k[0], k[1], code, bit(must)];
                inputs.extend((0..bits).map(|_| if counts { sb.input() } else { zero }));
                inputs.push(zero);
                sb.gate(s, &inputs);
            }
        }
        let shape = sb.finish().expect("well formed");
        let plan = shape.fill_plan();
        let consts = consts.values;
        let lookup = lookup.then(|| Lookup {
            bits,
            rows: key_rows,
            codes: blobs.iter().map(|b| b.code).collect(),
        });
        TypedCircuit {
            lens,
            types: blobs.iter().map(|b| b.t.alias).collect(),
            lookup,
            consts,
            shape,
            plan,
            gadgets,
            packer,
            ands,
            rows: (compressions, rows, padding),
            nu,
        }
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
        if let (Some(blobs), Some(l)) = (blobs, &self.lookup) {
            for count in self.counts(keys, blobs, l) {
                v.extend((0..l.bits).map(|i| F128::new(count >> i & 1, 0)));
            }
        }
        v
    }

    /// How many times each key row that is found by others is asked for:
    /// all the asks for a (key, code) go to its first such row.
    fn counts(&self, keys: &[[u8; 32]], blobs: &[Vec<u8>], l: &Lookup) -> Vec<u64> {
        let types = cas_ir::types();
        let mut circuits: HashMap<&str, cas_ir::Circuits> = HashMap::new();
        let held: Vec<Vec<Vec<([u8; 32], u8)>>> = self
            .types
            .iter()
            .zip(blobs)
            .map(|(&alias, b)| {
                let t = types.iter().find(|t| t.alias == alias).unwrap();
                let c = circuits.entry(alias).or_insert_with(|| cas_ir::circuits(t));
                cas_ir::keys(t, c, b).unwrap_or_else(|| panic!("a {alias} that is not valid"))
            })
            .collect();
        let tuple = |src: Src| match src {
            Src::Own { blob } => (keys[blob], l.codes[blob]),
            Src::Held { blob, row, field } => held[blob][row][field],
        };
        let mut asked: HashMap<([u8; 32], u8), u64> = HashMap::new();
        for &(src, must, _) in &l.rows {
            let (k, c) = tuple(src);
            if must && c != 0 && k != [0; 32] {
                *asked.entry((k, c)).or_default() += 1;
            }
        }
        let counts = l
            .rows
            .iter()
            .filter(|r| r.2)
            .map(|&(src, _, _)| asked.remove(&tuple(src)).unwrap_or(0))
            .collect();
        if let Some(((k, c), _)) = asked.iter().next() {
            panic!(
                "the key {} of code {c} is held but not found",
                k.iter().map(|b| format!("{b:02x}")).collect::<String>()
            );
        }
        counts
    }

    pub fn witness(&self, keys: &[[u8; 32]], blobs: &[Vec<u8>]) -> CircuitWitness {
        self.shape
            .run_filled_deferred(&self.plan, &self.values(keys, Some(blobs)), &[])
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
            .map(|(id, g)| {
                (
                    self.shape.registry_slot(*id),
                    g.r1cs().csc_lincheck_circuit() as _,
                )
            })
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

    fn key(b: &[u8]) -> [u8; 32] {
        *blake3::hash(b).as_bytes()
    }

    /// A runs blob: one record per (note, count).
    fn runs(recs: &[([u8; 32], u8)]) -> Vec<u8> {
        let mut b = vec![0u8; 64 * recs.len()];
        for (r, (k, n)) in b.chunks_exact_mut(64).zip(recs) {
            r[..32].copy_from_slice(k);
            r[39] = *n;
        }
        b
    }

    /// A lines blob: note, old runs, new runs.
    fn lines(note: [u8; 32], old: [u8; 32], new: [u8; 32]) -> Vec<u8> {
        let mut b = vec![0u8; 128];
        b[..32].copy_from_slice(&note);
        b[32..64].copy_from_slice(&old);
        b[64..96].copy_from_slice(&new);
        b
    }

    fn prove_lookup(blobs: &[(&str, Role, u8, Vec<u8>)]) -> Result<(), FlockVerifyError> {
        let types = cas_ir::types();
        let t = |a: &str| types.iter().find(|t| t.alias == a).unwrap();
        let spec: Vec<Blob<'_>> = blobs
            .iter()
            .map(|(a, role, code, b)| Blob {
                t: t(a),
                len: b.len(),
                role: *role,
                code: *code,
            })
            .collect();
        let keys: Vec<[u8; 32]> = blobs.iter().map(|b| key(&b.3)).collect();
        let bytes: Vec<Vec<u8>> = blobs.iter().map(|b| b.3.clone()).collect();
        let c = TypedCircuit::with_lookup(&spec, true);
        let proof = c.prove(&c.witness(&keys, &bytes));
        c.verify(&keys, &proof)
    }

    /// New blobs find the keys they hold among themselves, and through an
    /// opened blob and a parent; a key nothing gives is not found.
    #[test]
    fn held_keys_are_found() {
        use cas_ir::code::{LINES, NOTE, RUNS};
        let note = b"a note".to_vec();
        let n = key(&note);
        let r1 = runs(&[(n, 3)]);
        let l = lines(n, key(&r1), key(&r1));
        prove_lookup(&[
            ("text", Role::New, NOTE, note.clone()),
            ("runs", Role::New, RUNS, r1.clone()),
            ("lines", Role::New, LINES, l),
        ])
        .expect("new blobs find each other");

        // A parent holds r0 and the note; r0 is opened from it; the new
        // lines and runs hold both.
        let r0 = runs(&[(n, 1), ([0; 32], 2)]);
        let parent = lines(n, key(&r0), [0; 32]);
        let r2 = runs(&[(n, 5)]);
        let l1 = lines(n, key(&r0), key(&r2));
        let opened = [
            ("lines", Role::Parent, LINES, parent.clone()),
            ("runs", Role::Opened, RUNS, r0.clone()),
            ("runs", Role::New, RUNS, r2.clone()),
            ("lines", Role::New, LINES, l1.clone()),
        ];
        prove_lookup(&opened).expect("found through the parent");

        // Without the parent, r0 and the note are found nowhere.
        let lost = std::panic::catch_unwind(|| prove_lookup(&opened[1..]));
        assert!(lost.is_err(), "the prover finds no one to give r0");
    }
}
