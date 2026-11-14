//! Lean derivations proven in Flock: the rows that cas-lean's `lean_dump`
//! writes (terms, cited declarations, judgement rows), one element table
//! per kind of row, tied together by the tuple bus.
//!
//! Every term, declaration and judgement row sends its tuple once, with a
//! count of how often it is asked for; a row that uses another (a premise,
//! a subterm) receives that tuple. Ids are row tags, so two rows never
//! share an id and equal ids mean equal terms. Columns shared between a
//! row's own tuple and the tuples it receives are the rule's equalities.
//!
//! Cycles are ruled out by ranks: each term and judgement row carries
//! `R = g^rank`, and a row's rank exceeds those of its premises and
//! subterms. `X ≥ c·Y` (exponents) is shown with two range lookups,
//! `X = c·Y·g^(d0)·h^(d1)` with `h = g^(2^L)` and `d0, d1 < 2^L`: the second
//! limb is found as `X·I` where `I = (c·Y·g^(d0))^(-1)` is pinned by the
//! range table's constant 1. A term's `L = g^b` bounds its loose bound
//! variables (`b ≥` the true range), so substitution into a closed part is
//! checked like Lean's `instantiate`. Bound variable indices and the `k` of
//! a substitution are `g^k`; `k + 1` is multiplication by `g`.
//!
//! Rules with constraints here: Refl, Symm, Trans, CongApp, App, Beta,
//! Delta, InstClosed, InstHit, InstApp, InstBind. Any other judgement row
//! goes to an *assumed* table that sends its fact unchecked, and the cited
//! declarations are sent unchecked (in cas they are bound to the store).
//! Levels, level lists and contexts are opaque ids. This is a cost
//! measurement, not yet a sound verifier for whole derivations.

use std::{collections::HashMap, sync::Arc, time::Instant};

use serde::{Deserialize, Serialize};

use flock_core::{
    challenger::FsChallenger,
    circuit::{
        boolean::BooleanGate,
        builder::{CircuitShape, CircuitWitness, GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
        Circuit,
        bus::{self, BusDirection, BusTerm, RowTag, SharedChallenges},
    },
    challenger::Challenger,
    pcs::Commitment,
    proof::R1csProofCircuitMerged,
    element_r1cs::{ElementTableBuilder, ElementTableType},
    pcs::PcsParams,
    schedule::{IoWord, TableType},
    union::UnionInstance,
    verifier::{FlockVerifyError, verify_ligerito_union_circuit},
};
use flock_field::F128;

use crate::prover::{
    UnionElementSlotInput, UnionSlotProverInput, commit_ligerito_union_circuit,
    prove_fast_ligerito_union_circuit,
};
use flock_core::lincheck::LincheckCircuit;

/// Tuple width: tag, id, kind, four fields, rank, and a judgement's owner
/// (the order of the declaration whose check it belongs to).
const W: usize = 9;

/// A tuple on the bus, as `(lo, hi)` words: how counts are keyed.
pub type Tuple = [(u64, u64); W];

/// The generator of the exponent encodings: `x`, primitive in this field
/// (see the test).
pub const G: F128 = F128::new(2, 0);

pub fn pow(mut b: F128, mut e: u128) -> F128 {
    let mut r = F128::ONE;
    while e > 0 {
        if e & 1 == 1 {
            r *= b;
        }
        b = b.square();
        e >>= 1;
    }
    r
}

/// `g^e` for a signed exponent.
fn gp(e: i64) -> F128 {
    if e >= 0 {
        pow(G, e as u128)
    } else {
        pow(G, e.unsigned_abs() as u128).inv()
    }
}

fn int(v: u64) -> F128 {
    F128::new(v, 0)
}

// Table tags (tuple position 0).
const TAG_TERM: u64 = 1;
const TAG_FACT: u64 = 2;
const TAG_DECL: u64 = 3;
const TAG_RANGE: u64 = 4;
// TAG_RANGE + 1 is the second limb's table.
const TAG_CHECKED: u64 = 6;
// Term kinds.
const K_BVAR: u64 = 1;
const K_LEAF: u64 = 2;
const K_APP: u64 = 3;
const K_LAM: u64 = 4;
const K_PI: u64 = 5;
const K_LET: u64 = 6;
const K_PROJ: u64 = 7;
// Leaf kinds.
const SUB_SORT: u64 = 1;
const SUB_CONST: u64 = 2;
const SUB_FVAR: u64 = 3;
const SUB_LIT: u64 = 4;
const SUB_STR: u64 = 5;
// Fact kinds.
const F_HAS: u64 = 1;
const F_INFER: u64 = 2;
const F_EQ: u64 = 3;
const F_INST: u64 = 4;
const F_LINST: u64 = 5;
const F_LINSTL: u64 = 6;
const F_LEQ: u64 = 7;

// ---------------------------------------------------------------- the data

/// A value and, for the exponent encodings, its exponent.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct V {
    f: F128,
    e: i64,
}

fn vi(v: u64) -> V {
    V { f: int(v), e: 0 }
}
fn ve(e: i64) -> V {
    V { f: gp(e), e }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TermRec {
    kind: u64,
    /// Parts: a bound variable's `g^j`; a leaf's kind and payload; children.
    x: [V; 3],
    /// Children (term indices) for kinds that have them, by part.
    child: [Option<usize>; 3],
    lbr: i64,
    rank: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DeclRec {
    unfold: u64,
    levels: u64,
    value: Option<usize>,
    /// The declared type, its kind (`DeclKind` from 1), the first 16 bytes
    /// of its key, and its order (position in the export).
    ty: Option<usize>,
    kind: u64,
    key: (u64, u64),
    order: i64,
    /// Whether a goal of this proof checks it.
    checked: bool,
}

/// A declaration's goal: its row and the judgements that establish it.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct GoalRec {
    decl: usize,
    /// `Infer([], ty, s)` and `Eq([], s, sort u)`.
    ty: (usize, usize),
    /// `Infer([], value, t)` and `Eq([], t, ty)`.
    value: Option<(usize, usize)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FactRec {
    rule: String,
    tag: u64,
    /// Raw fields from the dump.
    raw: [u64; 4],
    /// Which fields are terms.
    is_term: [bool; 4],
    premises: Vec<usize>,
    rank: i64,
    /// The order of the declaration whose check this judgement is part of.
    owner: i64,
}

/// A parsed derivation, or (in cone mode) the shared terms and
/// declarations of many, or one batch's judgements and goals over them.
#[derive(Default, Serialize, Deserialize)]
pub struct Rows {
    terms: Vec<TermRec>,
    decls: Vec<DeclRec>,
    facts: Vec<FactRec>,
    goals: Vec<GoalRec>,
    /// Limb width for every table, when fixed for a whole cone.
    limb: Option<usize>,
    /// The file's own level, list and context rows (local ids), kept only
    /// to put them in a cone's shared numbering.
    #[serde(skip)]
    local: Local,
}

#[derive(Default)]
struct Local {
    levels: Vec<(u8, u64, u64)>,
    lists: Vec<Option<(u64, u64)>>,
    ctxs: Vec<Option<(u64, u64)>>,
    /// The canonical number of each declaration row.
    decl_no: Vec<u64>,
}

impl Rows {
    pub fn parse(text: &str) -> Result<Rows, String> {
        let mut terms: Vec<TermRec> = Vec::new();
        let mut decl_at: HashMap<u64, usize> = HashMap::new();
        let mut decls = Vec::new();
        let mut facts = Vec::new();
        let mut local = Local::default();
        let mut goal_line: Option<((u64, u64), Vec<i64>)> = None;
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            let n = |i: usize| -> Result<u64, String> {
                w.get(i)
                    .ok_or(format!("short line: {line}"))?
                    .parse::<u64>()
                    .map_err(|e| format!("{e}: {line}"))
            };
            match w.first().copied() {
                Some("L") => {
                    let row = match w[2] {
                        "zero" => (0, 0, 0),
                        "succ" => (1, n(3)?, 0),
                        "max" => (2, n(3)?, n(4)?),
                        "imax" => (3, n(3)?, n(4)?),
                        _ => (4, n(3)?, 0),
                    };
                    local.levels.push(row);
                }
                Some("S") => local.lists.push((w[2] == "cons").then(|| (n(3).unwrap(), n(4).unwrap()))),
                Some("C") => local.ctxs.push((w[2] == "push").then(|| (n(3).unwrap(), n(4).unwrap()))),
                Some("G") => {
                    goal_line = Some((
                        (n(1)?, n(2)?),
                        w[3..]
                            .iter()
                            .map(|x| x.parse::<i64>().map_err(|e| format!("{e}: {line}")))
                            .collect::<Result<_, _>>()?,
                    ))
                }
                Some("T") => {
                    if n(1)? as usize != terms.len() {
                        return Err(format!("terms out of order: {line}"));
                    }
                    let ch = |i: usize| -> Result<usize, String> {
                        let c = n(i)? as usize;
                        if c >= terms.len() {
                            Err(format!("child after parent: {line}"))
                        } else {
                            Ok(c)
                        }
                    };
                    let (kind, x, child) = match w[2] {
                        "bvar" => (
                            K_BVAR,
                            [ve(n(3)? as i64), V::default(), V::default()],
                            [None; 3],
                        ),
                        "leaf" => {
                            let sub = match w[3] {
                                "sort" => SUB_SORT,
                                "const" => SUB_CONST,
                                "fvar" => SUB_FVAR,
                                "lit" => SUB_LIT,
                                "str" => SUB_STR,
                                s => return Err(format!("leaf {s}")),
                            };
                            // A constant's declaration is mapped to its id later.
                            (K_LEAF, [vi(sub), vi(n(4)?), vi(n(5)?)], [None; 3])
                        }
                        "app" | "lam" | "pi" => {
                            let k = match w[2] {
                                "app" => K_APP,
                                "lam" => K_LAM,
                                _ => K_PI,
                            };
                            (k, [V::default(); 3], [Some(ch(3)?), Some(ch(4)?), None])
                        }
                        "let" => (
                            K_LET,
                            [V::default(); 3],
                            [Some(ch(3)?), Some(ch(4)?), Some(ch(5)?)],
                        ),
                        "proj" => (
                            K_PROJ,
                            [vi(n(3)?), vi(n(4)?), V::default()],
                            [None, None, Some(ch(5)?)],
                        ),
                        k => return Err(format!("term kind {k}")),
                    };
                    let c = |i: usize| child[i].map(|c| &terms[c]);
                    let lbr = match kind {
                        K_BVAR => x[0].e + 1,
                        K_APP => c(0).unwrap().lbr.max(c(1).unwrap().lbr),
                        K_LAM | K_PI => c(0).unwrap().lbr.max(c(1).unwrap().lbr - 1),
                        K_LET => c(0)
                            .unwrap()
                            .lbr
                            .max(c(1).unwrap().lbr)
                            .max(c(2).unwrap().lbr - 1),
                        K_PROJ => c(2).unwrap().lbr,
                        _ => 0,
                    };
                    let rank = child
                        .iter()
                        .flatten()
                        .map(|&i| terms[i].rank + 1)
                        .max()
                        .unwrap_or(0);
                    terms.push(TermRec {
                        kind,
                        x,
                        child,
                        lbr,
                        rank,
                    });
                }
                Some("D") => {
                    let v: i64 = w[4].parse().map_err(|e| format!("{e}: {line}"))?;
                    decl_at.insert(n(1)?, decls.len());
                    local.decl_no.push(n(1)?);
                    let more = w.len() >= 10;
                    decls.push(DeclRec {
                        unfold: n(2)?,
                        levels: n(3)?,
                        value: (v >= 0).then_some(v as usize),
                        ty: if more { Some(n(8)? as usize) } else { None },
                        kind: if more { n(7)? } else { 0 },
                        key: if more { (n(5)?, n(6)?) } else { (0, 0) },
                        order: if more { w[9].parse().map_err(|e| format!("{e}: {line}"))? } else { 0 },
                        checked: false,
                    });
                }
                Some("F") => {
                    if n(1)? as usize != facts.len() {
                        return Err(format!("facts out of order: {line}"));
                    }
                    let tag = match w[3] {
                        "has" => F_HAS,
                        "infer" => F_INFER,
                        "eq" => F_EQ,
                        "inst" => F_INST,
                        "linst" => F_LINST,
                        "linstl" => F_LINSTL,
                        "leq" => F_LEQ,
                        f => return Err(format!("fact {f}")),
                    };
                    let is_term = match tag {
                        F_HAS => [false, false, true, false],
                        F_INFER | F_EQ => [false, true, true, false],
                        F_INST => [true, false, true, true],
                        F_LINST => [true, false, true, false],
                        _ => [false; 4],
                    };
                    let raw = [n(4)?, n(5)?, n(6)?, n(7)?];
                    let premises: Vec<usize> = w[9..]
                        .iter()
                        .map(|p| p.parse::<usize>().map_err(|e| format!("{e}: {line}")))
                        .collect::<Result<_, _>>()?;
                    if premises.iter().any(|&p| p >= facts.len()) {
                        return Err(format!("a premise after its use: {line}"));
                    }
                    let rank = premises
                        .iter()
                        .map(|&p: &usize| facts_rank(&facts, p) + 1)
                        .max()
                        .unwrap_or(0);
                    facts.push(FactRec {
                        rule: w[2].to_string(),
                        tag,
                        raw,
                        is_term,
                        premises,
                        rank,
                        owner: 0,
                    });
                }
                _ => {}
            }
        }
        // Constants name declarations by their canonical number: by row.
        for t in &mut terms {
            if t.kind == K_LEAF && t.x[0].f == int(SUB_CONST) {
                let c = t.x[1].f.lo;
                t.x[1] = vi(*decl_at.get(&c).ok_or(format!("no declaration {c}"))? as u64);
            }
            if t.kind == K_PROJ {
                let c = t.x[0].f.lo;
                t.x[0] = vi(*decl_at.get(&c).ok_or(format!("no declaration {c}"))? as u64);
            }
        }
        // The goal: the declaration checked, by key among the cited rows or
        // as a row of its own, and every judgement's owner.
        let mut goals = Vec::new();
        if let Some((key, g)) = goal_line {
            let [kind, levels, ty, value, ti, te, vi_, ve_, _prop, order, _empty] = g[..] else {
                return Err(format!("a goal row of {} fields", g.len() + 2));
            };
            let decl = match decls.iter().position(|d| d.key == key) {
                Some(i) => i,
                None => {
                    decls.push(DeclRec {
                        unfold: matches!(kind, 2 | 3) as u64,
                        levels: levels as u64,
                        value: (value >= 0).then_some(value as usize),
                        ty: Some(ty as usize),
                        kind: kind as u64,
                        key,
                        order,
                        checked: false,
                    });
                    local.decl_no.push(u64::MAX);
                    decls.len() - 1
                }
            };
            decls[decl].checked = true;
            for f in &mut facts {
                f.owner = order;
            }
            let fact = |i: i64| -> Result<usize, String> {
                (i >= 0 && (i as usize) < facts.len())
                    .then_some(i as usize)
                    .ok_or(format!("a goal names fact {i}"))
            };
            goals.push(GoalRec {
                decl,
                ty: (fact(ti)?, fact(te)?),
                value: if vi_ >= 0 { Some((fact(vi_)?, fact(ve_)?)) } else { None },
            });
        }
        Ok(Rows {
            terms,
            decls,
            facts,
            goals,
            limb: None,
            local,
        })
    }

    /// Several derivations as one: indices shifted, nothing shared (levels,
    /// lists and contexts stay opaque numbers).
    pub fn merge(parts: Vec<Rows>) -> Rows {
        let mut m = Rows::default();
        for p in parts {
            let (nt, nd, nf) = (m.terms.len(), m.decls.len(), m.facts.len());
            for mut t in p.terms {
                for c in t.child.iter_mut().flatten() {
                    *c += nt;
                }
                if t.kind == K_LEAF && t.x[0].f == int(SUB_CONST) {
                    t.x[1] = vi(t.x[1].f.lo + nd as u64);
                }
                if t.kind == K_PROJ {
                    t.x[0] = vi(t.x[0].f.lo + nd as u64);
                }
                m.terms.push(t);
            }
            for mut d in p.decls {
                d.value = d.value.map(|v| v + nt);
                d.ty = d.ty.map(|v| v + nt);
                m.decls.push(d);
            }
            for mut g in p.goals {
                g.decl += nd;
                g.ty = (g.ty.0 + nf, g.ty.1 + nf);
                g.value = g.value.map(|(a, b)| (a + nf, b + nf));
                m.goals.push(g);
            }
            for mut f in p.facts {
                for x in 0..4 {
                    if f.is_term[x] {
                        f.raw[x] += nt as u64;
                    }
                }
                for q in &mut f.premises {
                    *q += nf;
                }
                m.facts.push(f);
            }
        }
        m
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        (self.terms.len(), self.decls.len(), self.facts.len())
    }
}

fn facts_rank(f: &[FactRec], p: usize) -> i64 {
    f[p].rank
}

// ------------------------------------------------------------ the tables

/// Where an input column's value comes from, for the row being filled.
#[derive(Clone, Debug)]
enum TR {
    /// The term row itself.
    Own,
    /// The term in field `i` of the judgement.
    Fld(usize),
    /// The term in field `i` of premise `j`.
    PFld(usize, usize),
    /// Child `c` of a term.
    Child(Box<TR>, usize),
}

#[derive(Clone, Copy, Debug)]
enum Part {
    Id,
    Kind,
    X(usize),
    L,
    R,
    /// A judgement's owner, `g^order`.
    O,
}

#[derive(Clone, Debug)]
enum E {
    /// A part of the row's own record (a term, declaration or judgement).
    Own(Part),
    /// A part of premise `j`.
    Prem(usize, Part),
    /// A part of a term.
    Tm(TR, Part),
    /// A part of the declaration of a constant term.
    Dcl(TR, Part),
}

#[derive(Clone, Debug)]
enum Src {
    Val(E),
    /// Limb `l` of an exponent: `g^(e mod 2^L)` or `h^(e >> L)`.
    Limb(E, usize),
    /// The comparison `X ≥ c·Y`: its first limb, and the inverse.
    GeA(E, E, i64),
    GeI(E, E, i64),
    /// A count bit of a send port, filled after counting.
    Bit,
    /// The range table's value at this row.
    RangeV(usize),
    /// A value the verifier is given (a public word).
    Pub(E),
}

#[derive(Clone, Copy, Debug)]
enum Col {
    In,
    Mul(usize, usize),
    MulC(F128, usize, usize),
    Lin(F128, usize),
}

#[derive(Clone, Copy, Debug)]
enum T {
    C(usize),
    K(F128),
    Id,
}

fn k(v: u64) -> T {
    T::K(int(v))
}
const Z: T = T::K(F128::ZERO);

#[derive(Clone, Debug)]
struct Port {
    send: bool,
    t: [T; W],
    mult: Vec<usize>,
}

#[derive(Clone, Debug)]
struct Spec {
    name: String,
    cols: Vec<(Col, Option<Src>)>,
    ports: Vec<Port>,
    /// A judgement table's owner column.
    owner: Option<usize>,
}

fn pad<const N: usize>(t: [T; N]) -> [T; W] {
    std::array::from_fn(|i| if i < N { t[i] } else { Z })
}

/// An element table whose columns are inputs or products of earlier ones.
pub struct Gate {
    ty: Arc<ElementTableType>,
    cols: Arc<Vec<Col>>,
}

impl Gate {
    fn new(cols: Vec<Col>) -> Self {
        let kappa = cols.len().next_power_of_two().trailing_zeros().max(1) as usize;
        let mut b = ElementTableBuilder::new(kappa);
        for (y, c) in cols.iter().enumerate() {
            match *c {
                Col::In => {
                    b.free_wire(y);
                }
                Col::Mul(a, x) => {
                    b.mult(y, a, x);
                }
                Col::MulC(c, a, x) => {
                    b.mult_lin(y, &[(a, c)], &[(x, F128::ONE)]);
                }
                Col::Lin(c, a) => {
                    b.linear(y, &[(a, c)]);
                }
            }
        }
        Gate {
            ty: Arc::new(b.build().expect("a valid table")),
            cols: Arc::new(cols),
        }
    }

    fn complete(cols: &[Col], z: &mut [F128]) {
        for (y, c) in cols.iter().enumerate() {
            z[y] = match *c {
                Col::In => z[y],
                Col::Mul(a, x) => z[a] * z[x],
                Col::MulC(c, a, x) => c * z[a] * z[x],
                Col::Lin(c, a) => c * z[a],
            };
        }
    }
}

impl GateType for Gate {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        // Inputs first, then outputs, each in column order (see `io_index`).
        let ins = self
            .cols
            .iter()
            .enumerate()
            .filter(|c| matches!(c.1, Col::In))
            .map(|c| IoWord::input(c.0));
        let outs = self
            .cols
            .iter()
            .enumerate()
            .filter(|c| !matches!(c.1, Col::In))
            .map(|c| IoWord::output(c.0));
        let io = ins.chain(outs).collect();
        TableType::element(self.ty.clone()).with_io_schema(io)
    }

    fn eval(&self, inputs: &[F128], _: &(), outputs: &mut Vec<F128>) -> Vec<F128> {
        let mut z = vec![F128::ZERO; self.cols.len()];
        let mut i = 0;
        for (y, c) in self.cols.iter().enumerate() {
            if matches!(c, Col::In) {
                z[y] = inputs[i];
                i += 1;
            }
        }
        Self::complete(&self.cols, &mut z);
        for (y, c) in self.cols.iter().enumerate() {
            if !matches!(c, Col::In) {
                outputs.push(z[y]);
            }
        }
        z
    }

    fn witness(&self, rows: &[Vec<F128>], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// The io-schema position of each column: inputs first.
fn io_index(cols: &[Col]) -> Vec<usize> {
    let n_in = cols.iter().filter(|c| matches!(c, Col::In)).count();
    let (mut i, mut o) = (0, n_in);
    cols.iter()
        .map(|c| {
            let at = if matches!(c, Col::In) { &mut i } else { &mut o };
            *at += 1;
            *at - 1
        })
        .collect()
}

impl Spec {
    fn new(name: &str) -> Self {
        Spec {
            name: name.into(),
            cols: Vec::new(),
            ports: Vec::new(),
            owner: None,
        }
    }
    fn src(&mut self, s: Src) -> usize {
        self.cols.push((Col::In, Some(s)));
        self.cols.len() - 1
    }
    fn val(&mut self, e: E) -> usize {
        self.src(Src::Val(e))
    }
    fn out(&mut self, c: Col) -> usize {
        self.cols.push((c, None));
        self.cols.len() - 1
    }
    fn recv<const N: usize>(&mut self, t: [T; N]) {
        self.ports.push(Port {
            send: false,
            t: pad(t),
            mult: Vec::new(),
        });
    }
    fn send<const N: usize>(&mut self, t: [T; N]) {
        self.ports.push(Port {
            send: true,
            t: pad(t),
            mult: Vec::new(),
        });
    }
    /// `X ≥ c·Y` in exponents, `x` and `y` the columns and `ex`, `ey` how
    /// to compute them.
    fn ge(&mut self, x: usize, ex: E, y: usize, ey: E, c: i64) {
        let a0 = self.src(Src::GeA(ex.clone(), ey.clone(), c));
        let i = self.src(Src::GeI(ex, ey, c));
        let t = self.out(Col::Mul(y, a0));
        let u = self.out(Col::MulC(gp(c), t, i));
        let b = self.out(Col::Mul(x, i));
        self.recv([k(TAG_RANGE), T::C(a0), T::K(F128::ONE), Z, Z, Z, Z, Z]);
        self.recv([k(TAG_RANGE + 1), T::C(b), T::C(u), Z, Z, Z, Z, Z]);
    }
    /// Receives the term `tr` with id column `id`; the parts given are
    /// shared, the others free. Returns the L and R columns.
    fn term(
        &mut self,
        id: usize,
        tr: TR,
        kind: T,
        x: [Option<T>; 3],
        l: Option<T>,
    ) -> (usize, usize) {
        let kind = match kind {
            T::Id => T::C(self.val(E::Tm(tr.clone(), Part::Kind))),
            t => t,
        };
        let x: Vec<T> = (0..3)
            .map(|i| x[i].unwrap_or_else(|| T::C(self.val(E::Tm(tr.clone(), Part::X(i))))))
            .collect();
        let lc = self.val(E::Tm(tr.clone(), Part::L));
        let l = l.unwrap_or(T::C(lc));
        let r = self.val(E::Tm(tr, Part::R));
        self.recv([k(TAG_TERM), T::C(id), kind, x[0], x[1], x[2], l, T::C(r)]);
        (lc, r)
    }
    /// Receives premise `j` of kind `tag` with these fields (free where
    /// `None`), and checks its rank is below the row's (`r`).
    fn prem(&mut self, j: usize, tag: u64, x: [Option<T>; 4], r: usize) -> [T; 4] {
        let id = self.val(E::Prem(j, Part::Id));
        let x: Vec<T> = (0..4)
            .map(|i| x[i].unwrap_or_else(|| T::C(self.val(E::Prem(j, Part::X(i))))))
            .collect();
        let pr = self.val(E::Prem(j, Part::R));
        let o = self.owner.expect("a judgement table");
        self.recv([
            k(TAG_FACT),
            T::C(id),
            k(tag),
            x[0],
            x[1],
            x[2],
            x[3],
            T::C(pr),
            T::C(o),
        ]);
        if ablate("RANK") {
            return [x[0], x[1], x[2], x[3]];
        }
        self.ge(r, E::Own(Part::R), pr, E::Prem(j, Part::R), 1);
        [x[0], x[1], x[2], x[3]]
    }
}

fn c(i: usize) -> Option<T> {
    Some(T::C(i))
}
fn tc(i: usize) -> T {
    T::C(i)
}

/// What a table's rows are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Of {
    Range(usize),
    Term(u64),
    /// Cited declarations: checked by a goal of this proof, members of
    /// inductive blocks (no value, never unfolded), or free (assumed).
    DeclChecked,
    DeclBlock,
    DeclFree,
    Fact(&'static str),
    Assumed,
    /// A declaration's goal, with or without a value.
    Goal(bool),
}

impl Of {
    /// Whether the table belongs to the shared part of a cone (terms and
    /// declarations) rather than to a batch of judgements.
    fn shared(self) -> bool {
        matches!(self, Of::Term(_) | Of::DeclChecked | Of::DeclBlock | Of::DeclFree)
    }
}

fn spec(of: Of) -> Spec {
    match of {
        Of::Range(l) => {
            let mut s = Spec::new(&format!("range{l}"));
            let v = s.src(Src::RangeV(l));
            s.send([
                k(TAG_RANGE + l as u64),
                tc(v),
                T::K(F128::ONE),
                Z,
                Z,
                Z,
                Z,
                Z,
            ]);
            s
        }
        Of::DeclChecked | Of::DeclBlock | Of::DeclFree => {
            let mut s = Spec::new(match of {
                Of::DeclChecked => "decl_checked",
                Of::DeclBlock => "decl_block",
                _ => "decl_free",
            });
            let x: Vec<usize> = (0..4).map(|i| s.val(E::Own(Part::X(i)))).collect();
            let o = s.val(E::Own(Part::R));
            // The key, public: which declarations the proof checks or assumes.
            s.src(Src::Pub(E::Own(Part::X(4))));
            s.src(Src::Pub(E::Own(Part::X(5))));
            let (unfold, value) = match of {
                Of::DeclBlock => (Z, Z),
                _ => (tc(x[0]), tc(x[2])),
            };
            if of == Of::DeclChecked {
                s.recv([k(TAG_CHECKED), T::Id, tc(x[1]), tc(x[3]), tc(x[2]), tc(o)]);
            }
            s.send([k(TAG_DECL), T::Id, unfold, tc(x[1]), value, tc(x[3]), tc(o)]);
            s
        }
        Of::Goal(with_value) => {
            let mut s = Spec::new(if with_value { "goal" } else { "goal_ax" });
            let own = |p| E::Own(p);
            let (d, lv, ty) = (s.val(own(Part::X(0))), s.val(own(Part::X(1))), s.val(own(Part::X(3))));
            let o = s.val(own(Part::R));
            // A premise in the empty context (id 0), owned by this goal.
            let fact = |s: &mut Spec, j: usize, tag: u64, x: [T; 4]| {
                let id = s.val(E::Prem(j, Part::Id));
                let r = s.val(E::Prem(j, Part::R));
                s.recv([k(TAG_FACT), tc(id), k(tag), x[0], x[1], x[2], x[3], tc(r), tc(o)]);
            };
            let sv = s.val(E::Prem(0, Part::X(2)));
            let su = s.val(E::Prem(1, Part::X(2)));
            fact(&mut s, 0, F_INFER, [Z, tc(ty), tc(sv), Z]);
            fact(&mut s, 1, F_EQ, [Z, tc(sv), tc(su), Z]);
            let u = s.val(E::Tm(TR::PFld(1, 2), Part::X(1)));
            s.term(su, TR::PFld(1, 2), k(K_LEAF), [Some(k(SUB_SORT)), Some(tc(u)), Some(Z)], None);
            let v = if with_value {
                let v = s.val(own(Part::X(2)));
                let t = s.val(E::Prem(2, Part::X(2)));
                fact(&mut s, 2, F_INFER, [Z, tc(v), tc(t), Z]);
                fact(&mut s, 3, F_EQ, [Z, tc(t), tc(ty), Z]);
                tc(v)
            } else {
                Z
            };
            s.send([k(TAG_CHECKED), tc(d), tc(lv), tc(ty), v, tc(o)]);
            s
        }
        Of::Term(kind) => term_spec(kind),
        Of::Assumed => {
            let mut s = Spec::new("assumed");
            let tag = s.val(E::Own(Part::Kind));
            let x: Vec<usize> = (0..4).map(|i| s.val(E::Own(Part::X(i)))).collect();
            let r = s.val(E::Own(Part::R));
            let o = s.val(E::Own(Part::O));
            s.send([
                k(TAG_FACT),
                T::Id,
                tc(tag),
                tc(x[0]),
                tc(x[1]),
                tc(x[2]),
                tc(x[3]),
                tc(r),
                tc(o),
            ]);
            s
        }
        Of::Fact(rule) => fact_spec(rule),
    }
}

fn term_spec(kind: u64) -> Spec {
    let mut s = Spec::new(&format!("term{kind}"));
    let own = |p| E::Own(p);
    match kind {
        K_BVAR => {
            let a0 = s.src(Src::Limb(own(Part::X(0)), 0));
            let a1 = s.src(Src::Limb(own(Part::X(0)), 1));
            let j = s.out(Col::Mul(a0, a1));
            let l = s.out(Col::Lin(G, j));
            let r = s.val(own(Part::R));
            s.recv([k(TAG_RANGE), tc(a0), T::K(F128::ONE), Z, Z, Z, Z, Z]);
            s.recv([k(TAG_RANGE + 1), tc(a1), T::K(F128::ONE), Z, Z, Z, Z, Z]);
            s.send([k(TAG_TERM), T::Id, k(K_BVAR), tc(j), Z, Z, tc(l), tc(r)]);
        }
        K_LEAF => {
            let x: Vec<usize> = (0..3).map(|i| s.val(own(Part::X(i)))).collect();
            let (l, r) = (s.val(own(Part::L)), s.val(own(Part::R)));
            s.send([
                k(TAG_TERM),
                T::Id,
                k(K_LEAF),
                tc(x[0]),
                tc(x[1]),
                tc(x[2]),
                tc(l),
                tc(r),
            ]);
        }
        _ => {
            let (l, r) = (s.val(own(Part::L)), s.val(own(Part::R)));
            // Children by part, and how L bounds each: c·L_child ≤ L.
            let (parts, shift): (Vec<usize>, Vec<i64>) = match kind {
                K_APP => (vec![0, 1], vec![0, 0]),
                K_LAM | K_PI => (vec![0, 1], vec![0, -1]),
                K_LET => (vec![0, 1, 2], vec![0, 0, -1]),
                K_PROJ => (vec![2], vec![0]),
                _ => unreachable!(),
            };
            let mut x: Vec<T> = vec![Z; 3];
            for p in 0..3 {
                x[p] = tc(s.val(own(Part::X(p))));
            }
            for (&p, &sh) in parts.iter().zip(&shift) {
                let T::C(id) = x[p] else { unreachable!() };
                let ch = TR::Child(Box::new(TR::Own), p);
                let (lc, rc) = s.term(id, ch.clone(), T::Id, [None; 3], None);
                if !ablate("RANK") {
                    s.ge(r, own(Part::R), rc, E::Tm(ch.clone(), Part::R), 1);
                }
                if !ablate("LBR") {
                    s.ge(l, own(Part::L), lc, E::Tm(ch, Part::L), sh);
                }
            }
            s.send([k(TAG_TERM), T::Id, k(kind), x[0], x[1], x[2], tc(l), tc(r)]);
        }
    }
    s
}

fn fact_spec(rule: &str) -> Spec {
    let mut s = Spec::new(rule);
    let own = |p| E::Own(p);
    let r = s.val(own(Part::R));
    let o = s.val(own(Part::O));
    s.owner = Some(o);
    let fld = |s: &mut Spec, i: usize| s.val(own(Part::X(i)));
    let one = T::K(F128::ONE);
    let send = |s: &mut Spec, tag: u64, x: [T; 4]| {
        s.send([k(TAG_FACT), T::Id, k(tag), x[0], x[1], x[2], x[3], tc(r), tc(o)])
    };
    // A citation names an earlier declaration: owner ≥ order + 1.
    let earlier = |s: &mut Spec, ord: usize, tr: TR| {
        s.ge(o, E::Own(Part::O), ord, E::Dcl(tr, Part::R), 1);
    };
    // The term in judgement field i, and its children.
    let f = |i: usize| TR::Fld(i);
    let ch = |t: TR, c: usize| TR::Child(Box::new(t), c);
    // Lean's instantiate needs the substituted term closed: its L is g^0.
    let closed = |s: &mut Spec, a: usize| {
        s.term(a, f(2), T::Id, [None; 3], Some(one));
    };
    match rule {
        "Refl" => {
            let (g, a) = (fld(&mut s, 0), fld(&mut s, 1));
            send(&mut s, F_EQ, [tc(g), tc(a), tc(a), Z]);
        }
        "Symm" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            s.prem(0, F_EQ, [c(g), c(b), c(a), Some(Z)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "Trans" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let m = s.val(E::Prem(0, Part::X(2)));
            s.prem(0, F_EQ, [c(g), c(a), c(m), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(m), c(b), Some(Z)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "CongApp" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (fa, xa) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
            );
            let (fb, xb) = (
                s.val(E::Tm(ch(f(2), 0), Part::Id)),
                s.val(E::Tm(ch(f(2), 1), Part::Id)),
            );
            s.term(a, f(1), k(K_APP), [c(fa), c(xa), Some(Z)], None);
            s.term(b, f(2), k(K_APP), [c(fb), c(xb), Some(Z)], None);
            s.prem(0, F_EQ, [c(g), c(fa), c(fb), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(xa), c(xb), Some(Z)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "Beta" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (fun, x) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
            );
            let body = s.val(E::Tm(ch(ch(f(1), 0), 1), Part::Id));
            s.term(a, f(1), k(K_APP), [c(fun), c(x), Some(Z)], None);
            s.term(fun, ch(f(1), 0), k(K_LAM), [None, c(body), Some(Z)], None);
            s.prem(0, F_INST, [c(body), Some(one), c(x), c(b)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "Delta0" | "DeltaL" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (d, ls) = (
                s.val(E::Tm(f(1), Part::X(1))),
                s.val(E::Tm(f(1), Part::X(2))),
            );
            s.term(a, f(1), k(K_LEAF), [Some(k(SUB_CONST)), c(d), c(ls)], None);
            let ty = s.val(E::Dcl(f(1), Part::X(3)));
            let ord = s.val(E::Dcl(f(1), Part::R));
            if rule == "Delta0" {
                s.recv([k(TAG_DECL), tc(d), one, Z, tc(b), tc(ty), tc(ord)]);
            } else {
                let (lv, v) = (
                    s.val(E::Dcl(f(1), Part::X(1))),
                    s.val(E::Dcl(f(1), Part::X(2))),
                );
                s.recv([k(TAG_DECL), tc(d), one, tc(lv), tc(v), tc(ty), tc(ord)]);
                s.prem(0, F_LINST, [c(v), c(ls), c(b), Some(Z)], r);
            }
            earlier(&mut s, ord, f(1));
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "Const0" | "ConstL" => {
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (d, ls) = (
                s.val(E::Tm(f(1), Part::X(1))),
                s.val(E::Tm(f(1), Part::X(2))),
            );
            s.term(e, f(1), k(K_LEAF), [Some(k(SUB_CONST)), c(d), c(ls)], None);
            let unf = s.val(E::Dcl(f(1), Part::X(0)));
            let v = s.val(E::Dcl(f(1), Part::X(2)));
            let ord = s.val(E::Dcl(f(1), Part::R));
            if rule == "Const0" {
                // No level parameters: the type is the declared one.
                s.recv([k(TAG_DECL), tc(d), tc(unf), Z, tc(v), tc(t), tc(ord)]);
            } else {
                let lv = s.val(E::Dcl(f(1), Part::X(1)));
                let ty = s.val(E::Dcl(f(1), Part::X(3)));
                s.recv([k(TAG_DECL), tc(d), tc(unf), tc(lv), tc(v), tc(ty), tc(ord)]);
                s.prem(0, F_LINST, [c(ty), c(ls), c(t), Some(Z)], r);
            }
            earlier(&mut s, ord, f(1));
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "App" => {
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (fun, a) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
            );
            s.term(e, f(1), k(K_APP), [c(fun), c(a), Some(Z)], None);
            let ft = s.val(E::Prem(0, Part::X(2)));
            let pi = s.val(E::Prem(1, Part::X(2)));
            let at = s.val(E::Prem(2, Part::X(2)));
            let pit = TR::PFld(1, 2);
            let (dom, cod) = (
                s.val(E::Tm(ch(pit.clone(), 0), Part::Id)),
                s.val(E::Tm(ch(pit.clone(), 1), Part::Id)),
            );
            s.term(pi, pit, k(K_PI), [c(dom), c(cod), Some(Z)], None);
            s.prem(0, F_INFER, [c(g), c(fun), c(ft), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(ft), c(pi), Some(Z)], r);
            s.prem(2, F_INFER, [c(g), c(a), c(at), Some(Z)], r);
            s.prem(3, F_EQ, [c(g), c(at), c(dom), Some(Z)], r);
            s.prem(4, F_INST, [c(cod), Some(one), c(a), c(t)], r);
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "InstClosed" => {
            let (e, kk, a) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (le, _) = s.term(e, f(0), T::Id, [None; 3], None);
            s.ge(kk, own(Part::X(1)), le, E::Tm(f(0), Part::L), 0);
            closed(&mut s, a);
            send(&mut s, F_INST, [tc(e), tc(kk), tc(a), tc(e)]);
        }
        "InstHit" => {
            let (e, kk, a) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            s.term(e, f(0), k(K_BVAR), [c(kk), Some(Z), Some(Z)], None);
            closed(&mut s, a);
            send(&mut s, F_INST, [tc(e), tc(kk), tc(a), tc(a)]);
        }
        "InstApp" | "InstLam" | "InstPi" => {
            let (e, kk, a, res) = (
                fld(&mut s, 0),
                fld(&mut s, 1),
                fld(&mut s, 2),
                fld(&mut s, 3),
            );
            let kind = match rule {
                "InstApp" => K_APP,
                "InstLam" => K_LAM,
                _ => K_PI,
            };
            let p: Vec<usize> = (0..2)
                .map(|i| s.val(E::Tm(ch(f(0), i), Part::Id)))
                .collect();
            let q: Vec<usize> = (0..2)
                .map(|i| s.val(E::Tm(ch(f(3), i), Part::Id)))
                .collect();
            s.term(e, f(0), k(kind), [c(p[0]), c(p[1]), Some(Z)], None);
            s.term(res, f(3), k(kind), [c(q[0]), c(q[1]), Some(Z)], None);
            closed(&mut s, a);
            s.prem(0, F_INST, [c(p[0]), c(kk), c(a), c(q[0])], r);
            let k1 = if kind == K_APP {
                kk
            } else {
                s.out(Col::Lin(G, kk))
            };
            s.prem(1, F_INST, [c(p[1]), c(k1), c(a), c(q[1])], r);
            send(&mut s, F_INST, [tc(e), tc(kk), tc(a), tc(res)]);
        }
        r => panic!("no table for {r}"),
    }
    s
}

/// The table a judgement row goes to.
fn table_of(g: &Rows, rows: &Rows, i: usize) -> Of {
    let f = &rows.facts[i];
    let r = match f.rule.as_str() {
        "Refl" => "Refl",
        "Symm" => "Symm",
        "Trans" => "Trans",
        "CongApp" => "CongApp",
        "Beta" => "Beta",
        "App" => "App",
        "InstClosed" => "InstClosed",
        "InstHit" => "InstHit",
        "InstApp" => "InstApp",
        "Const" if f.premises.is_empty() => "Const0",
        "Const" => "ConstL",
        "Delta" if f.premises.is_empty() => "Delta0",
        "Delta" => "DeltaL",
        "InstBind" => {
            if g.terms[f.raw[0] as usize].kind == K_LAM {
                "InstLam"
            } else {
                "InstPi"
            }
        }
        _ => return Of::Assumed,
    };
    Of::Fact(r)
}

const FACT_TABLES: [&str; 16] = [
    "Const0",
    "ConstL",
    "Refl",
    "Symm",
    "Trans",
    "CongApp",
    "Beta",
    "App",
    "InstClosed",
    "InstHit",
    "InstApp",
    "InstLam",
    "InstPi",
    "Delta0",
    "DeltaL",
    "Assumed",
];

/// Which tables a build fills: all of a derivation's, or one side of a
/// cone (the shared terms and declarations, or a batch's judgements).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Whole,
    Shared,
    Batch,
}

/// The shared side of a cone starts at this chunk, so no term has id 0
/// (which stands for "no value").
pub const SHARED_BASE: usize = 1;

/// A derivation laid out as tables, ready to prove.
pub struct LeanCircuit {
    pub nu: usize,
    pub limb: usize,
    specs: Vec<Spec>,
    /// Every table's rows, with count bits.
    tables: Vec<Vec<Vec<F128>>>,
    /// Each table's first chunk number: chunk `c` of table `s` holds rows
    /// `[c·2^nu, (c+1)·2^nu)` and gives them the row tags
    /// `(chunk_base[s] + c)·2^32 + j`.
    chunk_base: Vec<usize>,
    pub build_ms: f64,
}

/// Some chunks of a [`LeanCircuit`] as one circuit: a piece of a proof
/// whose bus is shared by every piece ([`flock_core::circuit::bus::Bus::shared`]).
pub struct Piece {
    pub nu: usize,
    shape: CircuitShape,
    values: Vec<F128>,
    slots: Vec<(SlotId, String, usize, usize, usize)>,
    /// In the recursion tower's posture: one boolean slot (the tower's
    /// child walker expects a boolean class) and the chained transcript.
    tower: Option<(SlotId, BooleanGate)>,
}

/// A piece's shape and input values.
pub struct BuiltParts<'a> {
    pub shape: &'a CircuitShape,
    pub values: &'a [F128],
}

/// The transcript domain of a piece proven for the recursion tower.
pub const TOWER_DOMAIN: &[u8] = b"cas-lean-rows-tower-v0";

pub struct Report {
    pub nu: usize,
    pub tables: Vec<(String, usize, usize, usize)>,
    pub rows: usize,
    pub cells: usize,
    pub ports: usize,
    pub build_ms: f64,
    pub witness_ms: f64,
    pub prove_ms: f64,
    pub verify_ms: f64,
    pub proof_bytes: usize,
}

impl LeanCircuit {
    pub fn build(rows: &Rows) -> Result<LeanCircuit, String> {
        Self::build_with(rows, true)
    }

    /// With `strict` false, tuples asked for but never given do not stop
    /// the build: for tests that a false derivation is refused by the proof.
    pub fn build_with(rows: &Rows, strict: bool) -> Result<LeanCircuit, String> {
        Self::build_at(rows, strict, 0)
    }

    /// Chunk numbers start at `base`, so derivations built at disjoint
    /// bases have disjoint row tags and can share one bus.
    pub fn build_at(rows: &Rows, strict: bool, base: usize) -> Result<LeanCircuit, String> {
        let (c, left) = Self::build_side(rows, rows, Side::Whole, base, &HashMap::new())?;
        if let Some((t, n)) = left.iter().next().filter(|_| strict) {
            return Err(format!(
                "{} tuples asked for but never given, e.g. {t:?} ({n} times)",
                left.len()
            ));
        }
        Ok(c)
    }

    /// Lays out the tables of `shared` (terms, declarations) and `batch`
    /// (judgements, goals), and fills those of `side`: both for a whole
    /// derivation, or one side of a cone, where the shared side is built
    /// once at chunk base [`SHARED_BASE`] and each batch at its own base.
    /// `outside` counts tuples the other side asks for; returns the tuples
    /// this side asks for and does not give.
    pub fn build_side(
        g: &Rows,
        b: &Rows,
        side: Side,
        base: usize,
        outside: &HashMap<Tuple, u64>,
    ) -> Result<(LeanCircuit, HashMap<Tuple, u64>), String> {
        let t0 = Instant::now();
        // Tables and which rows go to each.
        let mut ofs: Vec<Of> = vec![Of::Range(0), Of::Range(1)];
        for kd in [K_BVAR, K_LEAF, K_APP, K_LAM, K_PI, K_LET, K_PROJ] {
            ofs.push(Of::Term(kd));
        }
        ofs.extend([Of::DeclChecked, Of::DeclBlock, Of::DeclFree]);
        for r in FACT_TABLES {
            ofs.push(if r == "Assumed" {
                Of::Assumed
            } else {
                Of::Fact(r)
            });
        }
        ofs.extend([Of::Goal(true), Of::Goal(false)]);
        let at = |of: Of| ofs.iter().position(|&o| o == of).unwrap();
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); ofs.len()];
        let mut term_at = vec![(0usize, 0usize); g.terms.len()];
        for (i, t) in g.terms.iter().enumerate() {
            let s = at(Of::Term(t.kind));
            term_at[i] = (s, members[s].len());
            members[s].push(i);
        }
        let mut decl_at = vec![(0usize, 0usize); g.decls.len()];
        for (i, d) in g.decls.iter().enumerate() {
            let s = at(if d.checked {
                Of::DeclChecked
            } else if d.kind >= 5 {
                Of::DeclBlock
            } else {
                Of::DeclFree
            });
            decl_at[i] = (s, members[s].len());
            members[s].push(i);
        }
        let mut fact_at = vec![(0usize, 0usize); b.facts.len()];
        for i in 0..b.facts.len() {
            let s = at(table_of(g, b, i));
            fact_at[i] = (s, members[s].len());
            members[s].push(i);
        }
        for (i, goal) in b.goals.iter().enumerate() {
            members[at(Of::Goal(goal.value.is_some()))].push(i);
        }
        // Range limbs wide enough for every rank, bound and order: 2^(2L) > span.
        let limb = g.limb.unwrap_or_else(|| {
            let span = b
                .facts
                .iter()
                .map(|f| f.rank.max(f.owner))
                .chain(g.terms.iter().map(|t| t.rank + t.lbr))
                .chain(g.decls.iter().map(|d| d.order + 1))
                .max()
                .unwrap_or(0);
            ((((span + 2) as f64).log2() / 2.0).ceil() as usize).max(4)
        });
        for l in 0..2 {
            members[at(Of::Range(l))] = (0..1usize << limb).collect();
        }
        let mine = |s: usize| match (side, ofs[s]) {
            (Side::Whole, _) | (_, Of::Range(_)) => true,
            (Side::Shared, o) => o.shared(),
            (Side::Batch, o) => !o.shared(),
        };
        // The capacity 2^nu of every slot: a table takes as many slots as it
        // needs, so pick the nu that wastes least (padded width times rows,
        // plus each port's group of 2^nu), counting each slot's fixed cost
        // in the proof and the verifier as 2^17 cells. Measured on a 250k
        // row derivation: nu 6, 8, 10, 12, 14 prove in 6.7, 5.1, 4.7, 4.4,
        // 5.3 s and verify in 1362, 289, 138, 95, 97 ms.
        let widths: Vec<(usize, usize)> = ofs
            .iter()
            .map(|&o| {
                let sp = spec(o);
                ((sp.cols.len() + 16).next_power_of_two(), sp.ports.len())
            })
            .collect();
        // A layout: the nu of some tables and each one's first chunk.
        let layout = |which: &dyn Fn(usize) -> bool, base: usize| -> (usize, Vec<usize>) {
            let cost = |nu: usize| -> usize {
                (0..ofs.len())
                    .filter(|&s| which(s))
                    .map(|s| {
                        let (w, p) = widths[s];
                        members[s].len().div_ceil(1 << nu) * (((w + 4 * p) << nu) + (1 << 17))
                    })
                    .sum()
            };
            let nu = std::env::var("LEAN_ROWS_NU")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| (8..=22).min_by_key(|&nu| cost(nu)).unwrap());
            let mut chunk_base = vec![0; ofs.len()];
            let mut n = base;
            for s in (0..ofs.len()).filter(|&s| which(s)) {
                chunk_base[s] = n;
                n += members[s].len().div_ceil(1 << nu);
            }
            (nu, chunk_base)
        };
        let is_shared = |s: usize| ofs[s].shared() || matches!(ofs[s], Of::Range(_));
        let is_batch = |s: usize| !ofs[s].shared();
        // Ids of the shared tables come from the shared layout; a batch's
        // own from its layout.
        let (whole, shared_l, batch_l) = match side {
            Side::Whole => {
                let l = layout(&|_| true, base);
                (Some(l), None, None)
            }
            Side::Shared => (None, Some(layout(&is_shared, base)), None),
            Side::Batch => (
                None,
                Some(layout(&is_shared, SHARED_BASE)),
                Some(layout(&is_batch, base)),
            ),
        };
        let lay_of = |s: usize| -> &(usize, Vec<usize>) {
            if let Some(l) = &whole {
                l
            } else if ofs[s].shared() {
                shared_l.as_ref().unwrap()
            } else {
                batch_l.as_ref().or(shared_l.as_ref()).unwrap()
            }
        };
        let id = |s: usize, j: usize| {
            let (nu, cb) = lay_of(s);
            int((((cb[s] + (j >> nu)) as u64) << 32) + (j & ((1usize << nu) - 1)) as u64)
        };
        let (nu, chunk_base) = match side {
            Side::Whole => whole.clone().unwrap(),
            Side::Shared => shared_l.clone().unwrap(),
            Side::Batch => batch_l.clone().unwrap(),
        };
        let term_id = |i: usize| id(term_at[i].0, term_at[i].1);
        let fact_id = |i: usize| id(fact_at[i].0, fact_at[i].1);
        let decl_id = |i: usize| id(decl_at[i].0, decl_at[i].1);
        let opt_term = |t: Option<usize>| t.map_or(F128::ZERO, term_id);
        // Fact fields as encoded values.
        let field = |i: usize, x: usize| -> V {
            let f = &b.facts[i];
            if f.is_term[x] {
                V {
                    f: term_id(f.raw[x] as usize),
                    e: 0,
                }
            } else if f.tag == F_INST && x == 1 {
                ve(f.raw[1] as i64)
            } else {
                vi(f.raw[x])
            }
        };
        let term_part = |i: usize, p: Part| -> V {
            let t = &g.terms[i];
            match p {
                Part::Id => V {
                    f: term_id(i),
                    e: 0,
                },
                Part::Kind => vi(t.kind),
                Part::X(x) => match t.child[x] {
                    Some(c) => V {
                        f: term_id(c),
                        e: 0,
                    },
                    None if t.kind == K_LEAF && t.x[0].f == int(SUB_CONST) && x == 1 => V {
                        f: decl_id(t.x[1].f.lo as usize),
                        e: 0,
                    },
                    None if t.kind == K_PROJ && x == 0 => V {
                        f: decl_id(t.x[0].f.lo as usize),
                        e: 0,
                    },
                    None => t.x[x],
                },
                Part::L => ve(t.lbr),
                Part::R => ve(t.rank),
                Part::O => unreachable!(),
            }
        };
        let decl_part = |i: usize, p: Part| -> V {
            let d = &g.decls[i];
            match p {
                Part::Id => V {
                    f: decl_id(i),
                    e: 0,
                },
                Part::X(0) => vi(d.unfold),
                Part::X(1) => vi(d.levels),
                Part::X(2) => V {
                    f: opt_term(d.value),
                    e: 0,
                },
                Part::X(3) => V {
                    f: opt_term(d.ty),
                    e: 0,
                },
                Part::X(4) => vi(d.key.0),
                Part::X(5) => vi(d.key.1),
                Part::R => ve(d.order),
                _ => unreachable!(),
            }
        };
        // Premise `j` of a row: a judgement's, or a goal's facts in order.
        let prem_of = |of: Of, own: usize, j: usize| -> usize {
            match of {
                Of::Goal(_) => {
                    let gl = &b.goals[own];
                    match j {
                        0 => gl.ty.0,
                        1 => gl.ty.1,
                        2 => gl.value.unwrap().0,
                        _ => gl.value.unwrap().1,
                    }
                }
                _ => b.facts[own].premises[j],
            }
        };
        let resolve = |tr: &TR, own: usize, of: Of| -> usize {
            fn go(
                tr: &TR,
                own: usize,
                of: Of,
                g: &Rows,
                b: &Rows,
                prem_of: &dyn Fn(Of, usize, usize) -> usize,
            ) -> usize {
                match tr {
                    TR::Own => own,
                    TR::Fld(x) => b.facts[own].raw[*x] as usize,
                    TR::PFld(j, x) => b.facts[prem_of(of, own, *j)].raw[*x] as usize,
                    TR::Child(t, c) => g.terms[go(t, own, of, g, b, prem_of)].child[*c].expect("a child"),
                }
            }
            go(tr, own, of, g, b, &prem_of)
        };
        let fact_part = |q: usize, p: Part| -> V {
            match p {
                Part::Id => V {
                    f: fact_id(q),
                    e: 0,
                },
                Part::Kind => vi(b.facts[q].tag),
                Part::X(x) => field(q, x),
                Part::R => ve(b.facts[q].rank),
                Part::O => ve(b.facts[q].owner),
                Part::L => unreachable!(),
            }
        };
        let eval = |e: &E, of: Of, own: usize| -> V {
            match e {
                E::Own(p) => match of {
                    Of::Term(_) => term_part(own, *p),
                    Of::DeclChecked | Of::DeclBlock | Of::DeclFree => decl_part(own, *p),
                    Of::Goal(_) => {
                        let d = b.goals[own].decl;
                        match p {
                            Part::X(0) => decl_part(d, Part::Id),
                            _ => decl_part(d, *p),
                        }
                    }
                    _ => fact_part(own, *p),
                },
                E::Prem(j, p) => fact_part(prem_of(of, own, *j), *p),
                E::Tm(tr, p) => term_part(resolve(tr, own, of), *p),
                E::Dcl(tr, p) => {
                    let t = &g.terms[resolve(tr, own, of)];
                    decl_part(t.x[1].f.lo as usize, *p)
                }
            }
        };
        let h = gp(1 << limb);
        let lmask = (1i64 << limb) - 1;
        // Rows, without count bits.
        let mut specs: Vec<Spec> = ofs.iter().map(|&o| spec(o)).collect();
        let mut tables: Vec<Vec<Vec<F128>>> = Vec::new();
        for (s, of) in ofs.iter().enumerate() {
            let sp = &specs[s];
            let cols: Vec<Col> = sp.cols.iter().map(|c| c.0).collect();
            let mut out = Vec::new();
            for &own in members[s].iter().filter(|_| mine(s)) {
                let mut z = vec![F128::ZERO; cols.len()];
                for (y, (_, src)) in sp.cols.iter().enumerate() {
                    let Some(src) = src else { continue };
                    z[y] = match src {
                        Src::Val(e) | Src::Pub(e) => eval(e, *of, own).f,
                        Src::Limb(e, l) => {
                            let x = eval(e, *of, own).e;
                            assert!(x >= 0 && x >> (2 * limb) == 0, "an index out of range");
                            if *l == 0 {
                                gp(x & lmask)
                            } else {
                                pow(h, (x >> limb) as u128)
                            }
                        }
                        Src::GeA(ex, ey, cc) | Src::GeI(ex, ey, cc) => {
                            let (x, y) = (eval(ex, *of, own), eval(ey, *of, own));
                            let d = x.e - y.e - cc;
                            if d < 0 || d >> (2 * limb) != 0 {
                                return Err(format!("{}: {} ≥ {} + {cc} fails", sp.name, x.e, y.e));
                            }
                            let a0 = gp(d & lmask);
                            if matches!(src, Src::GeA(..)) {
                                a0
                            } else {
                                (gp(*cc) * y.f * a0).inv()
                            }
                        }
                        Src::RangeV(l) => {
                            if *l == 0 {
                                gp(own as i64)
                            } else {
                                pow(h, own as u128)
                            }
                        }
                        Src::Bit => F128::ZERO,
                    };
                }
                Gate::complete(&cols, &mut z);
                out.push(z);
            }
            tables.push(out);
        }
        // Count every tuple received, here and outside.
        let tuple = |t: &[T; W], z: &[F128], s: usize, j: usize| -> Tuple {
            std::array::from_fn(|i| {
                let v = match t[i] {
                    T::C(c) => z[c],
                    T::K(v) => v,
                    T::Id => id(s, j),
                };
                (v.lo, v.hi)
            })
        };
        let mut asked: HashMap<Tuple, u64> = outside.clone();
        for (s, sp) in specs.iter().enumerate() {
            for p in sp.ports.iter().filter(|p| !p.send) {
                for (j, z) in tables[s].iter().enumerate() {
                    *asked.entry(tuple(&p.t, z, s, j)).or_default() += 1;
                }
            }
            // A batch's goals are received by the shared side, once each.
            if side == Side::Batch && matches!(ofs[s], Of::Goal(_)) {
                for p in sp.ports.iter().filter(|p| p.send) {
                    for (j, z) in tables[s].iter().enumerate() {
                        *asked.entry(tuple(&p.t, z, s, j)).or_default() += 1;
                    }
                }
            }
        }
        // Count bits for each send port.
        for s in 0..specs.len() {
            let sends: Vec<usize> = (0..specs[s].ports.len())
                .filter(|&p| specs[s].ports[p].send)
                .collect();
            for p in sends {
                let counts: Vec<u64> = tables[s]
                    .iter()
                    .enumerate()
                    .map(|(j, z)| {
                        asked
                            .remove(&tuple(&specs[s].ports[p].t, z, s, j))
                            .unwrap_or(0)
                    })
                    .collect();
                let bits = (64 - counts.iter().max().copied().unwrap_or(0).leading_zeros()).max(1)
                    as usize;
                let first = specs[s].cols.len();
                for _ in 0..bits {
                    specs[s].src(Src::Bit);
                }
                specs[s].ports[p].mult = (first..first + bits).collect();
                for (z, n) in tables[s].iter_mut().zip(counts) {
                    z.extend((0..bits).map(|b| int(n >> b & 1)));
                }
            }
        }
        // Tables of the other side are not part of this circuit.
        for s in 0..specs.len() {
            if !mine(s) {
                tables[s].clear();
            }
        }
        Ok((
            LeanCircuit {
                nu,
                limb,
                specs,
                tables,
                chunk_base,
                build_ms: t0.elapsed().as_secs_f64() * 1e3,
            },
            asked,
        ))
    }

    /// Every chunk `(table, chunk)` in order, with its row count.
    pub fn chunks(&self) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for (s, t) in self.tables.iter().enumerate() {
            for (c, ch) in t.chunks(1 << self.nu).enumerate() {
                out.push((s, c, ch.len()));
            }
        }
        out
    }

    /// The chunks in pieces of at most about `cap` rows, as even as whole
    /// chunks allow (so no piece is tiny: a commitment needs some size).
    pub fn pieces(&self, cap: usize) -> Vec<Vec<(usize, usize)>> {
        let chunks = self.chunks();
        let total: usize = chunks.iter().map(|c| c.2).sum();
        let k = total.div_ceil(cap.max(1)).max(1);
        let target = total.div_ceil(k);
        let mut out: Vec<Vec<(usize, usize)>> = vec![Vec::new()];
        let mut rows = 0;
        for (s, c, n) in chunks {
            if rows >= target && out.len() < k {
                out.push(Vec::new());
                rows = 0;
            }
            rows += n;
            out.last_mut().unwrap().push((s, c));
        }
        out
    }

    /// All chunks as one circuit.
    pub fn whole(&self) -> Result<Piece, String> {
        let all: Vec<(usize, usize)> = self.chunks().iter().map(|&(s, c, _)| (s, c)).collect();
        self.piece(&all)
    }

    /// The circuit of some chunks: one gate per row, inputs in order.
    pub fn piece(&self, chunks: &[(usize, usize)]) -> Result<Piece, String> {
        self.piece_with(chunks, false)
    }

    /// [`Self::piece`], in the recursion tower's posture when `tower`.
    pub fn piece_with(&self, chunks: &[(usize, usize)], tower: bool) -> Result<Piece, String> {
        let nu = self.nu;
        let mut sb = ShapeBuilder::new(nu);
        let mut values = Vec::new();
        let mut slots = Vec::new();
        let tower = if tower {
            // One swap row on zeros: the boolean class the tower needs.
            let gate = crate::leanxmss::gadgets::swap_gate(nu).map_err(|e| format!("{e:?}"))?;
            let slot = sb.slot(gate.clone());
            // Its first word is a fixed public zero: a tower child has a
            // public segment even when its tables publish nothing.
            let w: Vec<Wire> = (0..3)
                .map(|k| {
                    values.push(F128::ZERO);
                    if k == 0 { sb.fixed_public_input(F128::ZERO) } else { sb.input() }
                })
                .collect();
            sb.gate(slot, &w);
            Some((slot, gate))
        } else {
            None
        };
        for &(s, c) in chunks {
            let sp = &self.specs[s];
            let chunk = &self.tables[s][c << nu..((c + 1) << nu).min(self.tables[s].len())];
            let cols: Vec<Col> = sp.cols.iter().map(|c| c.0).collect();
            let io = io_index(&cols);
            let slot = sb.slot(Gate::new(cols.clone()));
            for p in &sp.ports {
                let terms =
                    p.t.iter()
                        .map(|t| match *t {
                            T::C(c) => BusTerm::word(io[c]),
                            T::K(v) => BusTerm {
                                word: None,
                                constant: v,
                                row: None,
                            },
                            T::Id => BusTerm::row(RowTag {
                                offset: ((self.chunk_base[s] + c) as u64) << 32,
                                div: 1,
                                modulo: 0,
                            }),
                        })
                        .collect();
                let dir = if p.send {
                    BusDirection::Send
                } else {
                    BusDirection::Recv
                };
                sb.bus_port_counted(slot, dir, terms, p.mult.iter().map(|&c| io[c]).collect());
            }
            for z in chunk {
                let wires: Vec<Wire> = sp
                    .cols
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| matches!(c.0, Col::In))
                    .map(|(y, c)| {
                        values.push(z[y]);
                        if matches!(c.1, Some(Src::RangeV(_)) | Some(Src::Pub(_))) {
                            sb.fixed_public_input(z[y])
                        } else {
                            sb.input()
                        }
                    })
                    .collect();
                sb.gate(slot, &wires);
            }
            slots.push((
                slot,
                sp.name.clone(),
                chunk.len(),
                sp.ports.len(),
                sp.cols.len(),
            ));
        }
        let shape = sb.finish().map_err(|e| format!("{e:?}"))?;
        Ok(Piece {
            nu,
            shape,
            values,
            slots,
            tower,
        })
    }

    /// Builds all chunks as one circuit, fills, proves and verifies;
    /// `tamper` changes one witness word first (the proof must then fail).
    pub fn prove_and_verify(&self, tamper: bool) -> Result<Report, FlockVerifyError> {
        let t = Instant::now();
        let piece = self.whole().expect("a valid shape");
        let mut r = piece.prove_and_verify(tamper)?;
        r.build_ms = self.build_ms + t.elapsed().as_secs_f64() * 1e3 - r.witness_ms - r.prove_ms - r.verify_ms;
        Ok(r)
    }
}

/// The shape every leaf of a recursive proof shares: the slot capacity
/// `2^nu` and each slot's table. Leaves of one profile are one circuit (one
/// digest), which the recursion tower needs; each slot holds one chunk of
/// its table, or none.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub nu: usize,
    pub slots: Vec<usize>,
}

/// One slot's content: a chunk `c` of table `s` of a built side.
pub type LeafSlot<'a> = Option<(&'a LeanCircuit, usize)>;

impl Profile {
    /// A profile for `leaves` leaves over chunk counts per table: table `s`
    /// takes `ceil(n_s / leaves)` slots in every leaf (tables in order).
    pub fn even(nu: usize, chunks: &[usize], leaves: usize) -> Profile {
        let mut slots = Vec::new();
        for (s, &n) in chunks.iter().enumerate() {
            slots.extend(std::iter::repeat_n(s, n.div_ceil(leaves.max(1))));
        }
        Profile { nu, slots }
    }

    /// Deal `chunks` (per table, in order: (side, chunk)) to `leaves`
    /// leaves: leaf `l` takes table `s`'s chunks `[l·k_s, (l+1)·k_s)`.
    /// Returns per leaf, per slot, the chunk or `None`.
    pub fn deal<T: Copy>(&self, per_table: &[Vec<T>], leaves: usize) -> Vec<Vec<Option<T>>> {
        let mut out = vec![Vec::with_capacity(self.slots.len()); leaves];
        let mut seen = vec![0usize; per_table.len()];
        for &s in &self.slots {
            let k = self.slots.iter().filter(|&&x| x == s).count();
            let i = seen[s];
            seen[s] += 1;
            for (l, leaf) in out.iter_mut().enumerate() {
                leaf.push(per_table[s].get(l * k + i).copied());
            }
        }
        out
    }
}

/// A leaf of `profile`: each slot full (`2^nu` rows), padded with rows that
/// count zero times. Every port is counted (a port without counts gets a
/// one-bit live column, 1 on a real row, 0 on padding), every row carries
/// its slot's chunk word (a public input: `chunk·2^32`, so a row's id is
/// that word plus its row tag, the same ids as [`LeanCircuit::piece`]),
/// and the verifier's words (`Pub`, range values) are public inputs, not
/// part of the circuit. Built in the tower's posture.
pub fn leaf(proto: &LeanCircuit, profile: &Profile, slots: &[LeafSlot<'_>]) -> Result<Piece, String> {
    let nu = profile.nu;
    if slots.len() != profile.slots.len() {
        return Err("one entry per profile slot".into());
    }
    let mut sb = ShapeBuilder::new(nu);
    let mut values = Vec::new();
    let mut out_slots = Vec::new();
    let gate = crate::leanxmss::gadgets::swap_gate(nu).map_err(|e| format!("{e:?}"))?;
    let bslot = sb.slot(gate.clone());
    let w: Vec<Wire> = (0..3)
        .map(|k| {
            values.push(F128::ZERO);
            if k == 0 { sb.public_input() } else { sb.input() }
        })
        .collect();
    sb.gate(bslot, &w);
    for (k, &s) in profile.slots.iter().enumerate() {
        let sp = &proto.specs[s];
        let (rows, chunk_no): (&[Vec<F128>], u64) = match slots[k] {
            Some((c, ch)) => {
                if c.nu != nu {
                    return Err(format!("a side at nu {} in a profile at nu {nu}", c.nu));
                }
                let t = &c.tables[s];
                (
                    &t[ch << nu..((ch + 1) << nu).min(t.len())],
                    (c.chunk_base[s] + ch) as u64,
                )
            }
            None => (&[], 0),
        };
        // The spec's columns, then the chunk word, then a live bit per port
        // without counts.
        let mut cols: Vec<Col> = sp.cols.iter().map(|c| c.0).collect();
        let word_col = cols.len();
        cols.push(Col::In);
        let mut live_col = vec![None; sp.ports.len()];
        for (pi, p) in sp.ports.iter().enumerate() {
            if p.mult.is_empty() {
                live_col[pi] = Some(cols.len());
                cols.push(Col::In);
            }
        }
        let io = io_index(&cols);
        let slot = sb.slot(Gate::new(cols.clone()));
        for (pi, p) in sp.ports.iter().enumerate() {
            let terms = p
                .t
                .iter()
                .map(|t| match *t {
                    T::C(c) => BusTerm::word(io[c]),
                    T::K(v) => BusTerm {
                        word: None,
                        constant: v,
                        row: None,
                    },
                    T::Id => BusTerm {
                        word: Some(io[word_col]),
                        constant: F128::ZERO,
                        row: Some(RowTag {
                            offset: 0,
                            div: 1,
                            modulo: 0,
                        }),
                    },
                })
                .collect();
            let dir = if p.send {
                BusDirection::Send
            } else {
                BusDirection::Recv
            };
            let mult = match live_col[pi] {
                Some(y) => vec![io[y]],
                None => p.mult.iter().map(|&c| io[c]).collect(),
            };
            sb.bus_port_counted(slot, dir, terms, mult);
        }
        values.push(int(chunk_no << 32));
        let word_w = sb.public_input();
        let n_live = live_col.iter().flatten().count();
        let zero_row = vec![F128::ZERO; sp.cols.len()];
        for j in 0..1usize << nu {
            let (z, live) = match rows.get(j) {
                Some(z) => (z, F128::ONE),
                None => (&zero_row, F128::ZERO),
            };
            let mut wires: Vec<Wire> = Vec::new();
            for (y, c) in sp.cols.iter().enumerate() {
                if !matches!(c.0, Col::In) {
                    continue;
                }
                values.push(z[y]);
                wires.push(if matches!(c.1, Some(Src::RangeV(_)) | Some(Src::Pub(_))) {
                    sb.public_input()
                } else {
                    sb.input()
                });
            }
            // The chunk word: every row the same public wire.
            wires.push(word_w);
            for _ in 0..n_live {
                values.push(live);
                wires.push(sb.input());
            }
            sb.gate(slot, &wires);
        }
        out_slots.push((slot, sp.name.clone(), 1usize << nu, sp.ports.len(), cols.len()));
    }
    let shape = sb.finish().map_err(|e| format!("{e:?}"))?;
    Ok(Piece {
        nu,
        shape,
        values,
        slots: out_slots,
        tower: Some((bslot, gate)),
    })
}

/// The bus challenges every piece shares: drawn after every piece's
/// circuit digest and commitment, in order.
pub fn shared_challenges(statement: &[([u8; 32], Commitment)]) -> SharedChallenges {
    let mut ch = FsChallenger::new(b"cas-lean-pieces-v0");
    ch.observe_label(&(statement.len() as u64).to_le_bytes());
    for (digest, c) in statement {
        ch.observe_label(digest);
        ch.observe_label(&bincode::serialize(c).expect("a commitment serializes"));
    }
    SharedChallenges {
        beta: ch.sample_f128(),
        gamma: (0..W).map(|_| ch.sample_f128()).collect(),
    }
}

/// The shared challenges of a recursive cone: drawn from the root of the
/// commitment tree the tower carries up ([`crate::tower`]'s Lean leaves:
/// each leaf's word chains its publics digest with its commitment cap, and
/// pairs hash as BLAKE3 parents). The tower checks the root against the
/// challenges every leaf absorbed.
pub fn shared_challenges_from_root(root: &[u8; 32]) -> SharedChallenges {
    let mut ch = FsChallenger::new(b"cas-lean-cone-v0");
    ch.observe_label(root);
    SharedChallenges {
        beta: ch.sample_f128(),
        gamma: (0..W).map(|_| ch.sample_f128()).collect(),
    }
}

/// A piece's proof under shared challenges.
pub struct PieceProof {
    pub proof: R1csProofCircuitMerged,
    pub commitment: Commitment,
    pub witness_ms: f64,
    pub prove_ms: f64,
}

impl Piece {
    fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.shape.registry, self.shape.counts.clone())
    }

    fn pcs(union: &UnionInstance<'_>) -> PcsParams {
        crate::leanxmss::circuit::XmssCircuit::pcs_params(union)
    }

    /// The circuit digest, which the shared challenges bind.
    pub fn digest(&self) -> [u8; 32] {
        self.shape.circuit.digest()
    }

    pub fn rows(&self) -> usize {
        self.slots.iter().map(|s| s.2).sum()
    }

    /// The witness, as the prover's element inputs, the public words, and
    /// the run (for the boolean slot's rows).
    fn witness(&self, tamper: bool) -> (Vec<Vec<F128>>, Vec<F128>, CircuitWitness) {
        let mut w = self.shape.run(&self.values, &[]);
        let mut el: Vec<(usize, Vec<F128>)> = self
            .slots
            .iter()
            .map(|s| {
                let r = self.shape.registry_slot(s.0);
                let SlotWitness::Element(z) =
                    std::mem::replace(&mut w.witnesses[r], SlotWitness::Element(Vec::new()))
                else {
                    unreachable!("element slots")
                };
                (r, z)
            })
            .collect();
        el.sort_by_key(|x| x.0);
        if tamper {
            // The first live cell: one word off.
            let (_, z) = el
                .iter_mut()
                .find(|(_, z)| z.iter().any(|v| !v.is_zero()))
                .unwrap();
            let i = z.iter().position(|v| !v.is_zero()).unwrap();
            z[i] += F128::ONE;
        }
        let public = w.public.clone();
        (el.into_iter().map(|x| x.1).collect(), public, w)
    }

    /// The boolean slot's prover input, in the tower's posture.
    fn boolean_inputs<'a>(&'a self, w: &'a CircuitWitness) -> Vec<UnionSlotProverInput<'a>> {
        self.tower
            .iter()
            .map(|(slot, gate)| {
                let rows = w.rows::<BooleanGate>(*slot);
                UnionSlotProverInput::in_place(
                    move |dst| gate.pack_into(rows, dst),
                    gate.r1cs().csc_lincheck_circuit(),
                )
            })
            .collect()
    }

    /// The lincheck circuits the verifier needs: the boolean slot's.
    pub fn lincheck_circuits(&self) -> Vec<&dyn LincheckCircuit> {
        self.tower
            .iter()
            .map(|(_, g)| g.r1cs().csc_lincheck_circuit() as &dyn LincheckCircuit)
            .collect()
    }

    /// The challenger a proof of this piece starts from.
    pub fn challenger(&self, label: &[u8]) -> FsChallenger {
        if self.tower.is_some() {
            FsChallenger::with_chained_blake3(TOWER_DOMAIN)
        } else {
            let mut ch = FsChallenger::new(b"cas-lean-rows-v0");
            ch.observe_label(label);
            ch
        }
    }

    /// The circuit, with the shared challenges when given.
    pub fn circuit_for(&self, shared: Option<&SharedChallenges>) -> Circuit {
        self.circuit(shared).into_owned()
    }

    /// The shape and its witness, for a recursion leaf.
    pub fn built(&self) -> crate::lean_rows::BuiltParts<'_> {
        BuiltParts { shape: &self.shape, values: &self.values }
    }

    pub fn union_instance(&self) -> UnionInstance<'_> {
        self.union()
    }

    pub fn pcs_params(&self) -> PcsParams {
        Self::pcs(&self.union())
    }

    fn inputs(z: &[Vec<F128>]) -> Vec<UnionElementSlotInput<'_>> {
        z.iter()
            .map(|z| UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(z)))
            .collect()
    }

    fn circuit(&self, shared: Option<&SharedChallenges>) -> std::borrow::Cow<'_, Circuit> {
        match shared {
            None => std::borrow::Cow::Borrowed(&self.shape.circuit),
            Some(s) => std::borrow::Cow::Owned(self.shape.circuit.clone().with_shared_challenges(s.clone())),
        }
    }

    /// A tower leaf's parts: the circuit (with the shared challenges when
    /// given) and its witness, the proof and its commitment. The piece
    /// must be in the tower's posture.
    pub(crate) fn tower_parts(
        &self,
        shared: Option<&SharedChallenges>,
        label: &[u8],
    ) -> (flock_core::circuit::builder::BuiltCircuit, R1csProofCircuitMerged, Commitment) {
        assert!(self.tower.is_some(), "a tower leaf is a piece in the tower's posture");
        let pp = self.prove(shared, label, false);
        let (_, _, witness) = self.witness(false);
        let mut shape = self.shape.clone();
        shape.circuit = self.circuit(shared).into_owned();
        (
            flock_core::circuit::builder::BuiltCircuit { shape, witness },
            pp.proof,
            pp.commitment,
        )
    }

    /// The first phase of a shared proof: the commitment only.
    pub fn commit(&self) -> Commitment {
        let (z, _, w) = self.witness(false);
        let union = self.union();
        let pcs = Self::pcs(&union);
        commit_ligerito_union_circuit(&union, &pcs, self.boolean_inputs(&w), Self::inputs(&z))
    }

    /// Proves this piece, with the bus challenges fixed from outside when
    /// `shared` is given. `label` separates the pieces' transcripts.
    pub fn prove(&self, shared: Option<&SharedChallenges>, label: &[u8], tamper: bool) -> PieceProof {
        let t = Instant::now();
        let (z, public, w) = self.witness(tamper);
        let witness_ms = t.elapsed().as_secs_f64() * 1e3;
        let union = self.union();
        let pcs = Self::pcs(&union);
        let circuit = self.circuit(shared);
        let t = Instant::now();
        let mut ch = self.challenger(label);
        let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
            &union,
            &circuit,
            &public,
            &pcs,
            self.boolean_inputs(&w),
            Self::inputs(&z),
            &mut ch,
        );
        PieceProof {
            proof,
            commitment,
            witness_ms,
            prove_ms: t.elapsed().as_secs_f64() * 1e3,
        }
    }

    /// The public segment: the circuit's fixed words, or (a leaf, whose
    /// chunk words and verifier words are inputs) the witness run's.
    pub fn public(&self) -> Vec<F128> {
        let c = &self.shape.circuit;
        let fixed: Option<Vec<F128>> = (0..c.num_public()).map(|i| c.fixed_public(i)).collect();
        fixed.unwrap_or_else(|| self.witness(false).1)
    }

    /// Verifies a piece's proof against its commitment; returns the time
    /// it took. Under shared challenges this does not check the bus's
    /// balance, which holds only over all pieces ([`shared_balance`]).
    pub fn verify(
        &self,
        shared: Option<&SharedChallenges>,
        label: &[u8],
        commitment: &Commitment,
        proof: &R1csProofCircuitMerged,
    ) -> Result<f64, FlockVerifyError> {
        let union = self.union();
        let pcs = Self::pcs(&union);
        let circuit = self.circuit(shared);
        let public = self.public();
        let t = Instant::now();
        let mut ch = self.challenger(label);
        let lcs = self.lincheck_circuits();
        verify_ligerito_union_circuit(
            &union, &circuit, &public, &lcs, commitment, proof, &pcs, &mut ch,
        )?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// Fills, proves and verifies alone; `tamper` changes one witness word
    /// first (the proof must then fail).
    pub fn prove_and_verify(&self, tamper: bool) -> Result<Report, FlockVerifyError> {
        let p = self.prove(None, b"", tamper);
        let proof_bytes = bincode::serialize(&p.proof).map(|b| b.len()).unwrap_or(0);
        let verify_ms = self.verify(None, b"", &p.commitment, &p.proof)?;
        let tables: Vec<(String, usize, usize, usize)> = self
            .slots
            .iter()
            .map(|s| (s.1.clone(), s.2, s.3, s.4))
            .collect();
        Ok(Report {
            nu: self.nu,
            rows: tables.iter().map(|t| t.1).sum(),
            cells: tables.iter().map(|t| t.1 * t.3).sum(),
            ports: tables.iter().map(|t| t.2).sum(),
            tables,
            build_ms: 0.0,
            witness_ms: p.witness_ms,
            prove_ms: p.prove_ms,
            verify_ms,
            proof_bytes,
        })
    }
}

/// Whether two commitments are the same root.
pub fn same_commitment(a: &Commitment, b: &Commitment) -> bool {
    bincode::serialize(a).ok() == bincode::serialize(b).ok()
}

/// Whether the pieces' shared buses balance together.
pub fn shared_balance<'a>(proofs: impl IntoIterator<Item = &'a R1csProofCircuitMerged>) -> bool {
    bus::shared_balance(proofs.into_iter().filter_map(|p| p.wiring.bus.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `x` generates the multiplicative group: `x^((2^128-1)/p) != 1` for
    /// every prime `p` of `2^128 - 1`.
    #[test]
    fn g_is_primitive() {
        let primes: [u128; 11] = [
            3,
            5,
            17,
            257,
            641,
            65537,
            274177,
            6700417,
            67280421310721,
            59649589127497217,
            5704689200685129054721,
        ];
        assert_eq!(primes.iter().product::<u128>(), u128::MAX);
        for p in primes {
            assert_ne!(pow(G, u128::MAX / p), F128::ONE, "{p}");
        }
    }

    /// `(\x. x) y = y` and the typing of an application, by hand.
    const SMALL: &str = include_str!("../tests/data/lean_small.rows");

    #[test]
    fn small_derivation_proves() {
        let rows = Rows::parse(SMALL).unwrap();
        let c = LeanCircuit::build(&rows).unwrap();
        let r = c.prove_and_verify(false).expect("verifies");
        assert!(r.verify_ms > 0.0);
        assert!(
            c.prove_and_verify(true).is_err(),
            "a changed word is refused"
        );
    }

    /// Beta claiming the wrong result: the build notices, and so does the proof.
    #[test]
    fn a_false_beta_is_refused() {
        let bad = SMALL
            .replace("F 2 Beta eq 0 5 4 0 : 1", "F 2 Beta eq 0 5 0 0 : 1")
            .replace("F 3 Trans eq 0 5 4 0 : 2 0", "F 3 Refl eq 0 5 5 0 :")
            .replace("F 4 Symm eq 0 4 5 0 : 3", "F 4 Refl eq 0 4 4 0 :");
        let rows = Rows::parse(&bad).unwrap();
        assert!(LeanCircuit::build(&rows).is_err());
        let c = LeanCircuit::build_with(&rows, false).unwrap();
        assert!(c.prove_and_verify(false).is_err());
    }

    /// Proves every piece under challenges drawn from all commitments;
    /// returns whether every piece verified and the buses balance.
    fn shared(c: &LeanCircuit, cap: usize, drop_last: bool) -> (usize, bool) {
        let pieces: Vec<Piece> = c.pieces(cap).iter().map(|p| c.piece(p).unwrap()).collect();
        let statement: Vec<([u8; 32], Commitment)> =
            pieces.iter().map(|p| (p.digest(), p.commit())).collect();
        let ch = shared_challenges(&statement);
        let mut proofs = Vec::new();
        let mut ok = true;
        for (i, p) in pieces.iter().enumerate() {
            let label = (i as u64).to_le_bytes();
            let pp = p.prove(Some(&ch), &label, false);
            assert!(same_commitment(&pp.commitment, &statement[i].1), "proving commits as the first phase did");
            ok &= p.verify(Some(&ch), &label, &statement[i].1, &pp.proof).is_ok();
            proofs.push(pp.proof);
        }
        if drop_last {
            proofs.pop();
        }
        (pieces.len(), ok && shared_balance(&proofs))
    }

    #[test]
    fn pieces_share_one_bus() {
        // Enough copies that each piece is big enough to commit to.
        let rows = Rows::merge((0..4000).map(|_| Rows::parse(SMALL).unwrap()).collect());
        let c = LeanCircuit::build(&rows).unwrap();
        let total: usize = c.chunks().iter().map(|x| x.2).sum();
        let (n, ok) = shared(&c, total / 2 + 1, false);
        assert_eq!(n, 2);
        assert!(ok, "the pieces verify and balance together");
        assert!(!shared(&c, total / 2 + 1, true).1, "a piece left out unbalances the bus");
    }

    /// A piece in the recursion tower's posture (a boolean slot, the
    /// chained transcript) proves, verifies and refuses a changed word.
    #[test]
    fn tower_piece_proves() {
        let rows = Rows::merge((0..4000).map(|_| Rows::parse(SMALL).unwrap()).collect());
        let c = LeanCircuit::build(&rows).unwrap();
        let p = c.piece_with(&c.pieces(usize::MAX)[0], true).unwrap();
        assert!(p.lincheck_circuits().len() == 1);
        let pp = p.prove(None, b"", false);
        p.verify(None, b"", &pp.commitment, &pp.proof).expect("the tower piece verifies");
        let bad = p.prove(None, b"", true);
        assert!(p.verify(None, b"", &bad.commitment, &bad.proof).is_err());
    }

    /// Two leaves of one profile: one circuit, the same ids as the pieces,
    /// and together they verify and balance under shared challenges.
    #[test]
    fn leaves_share_one_bus() {
        let rows = Rows::merge((0..4000).map(|_| Rows::parse(SMALL).unwrap()).collect());
        let c = LeanCircuit::build(&rows).unwrap();
        let mut per_table: Vec<Vec<usize>> = vec![Vec::new(); c.tables.len()];
        for (s, ch, _) in c.chunks() {
            per_table[s].push(ch);
        }
        let n: Vec<usize> = per_table.iter().map(|t| t.len()).collect();
        let profile = Profile::even(c.nu, &n, 2);
        let dealt = profile.deal(&per_table, 2);
        let leaves: Vec<Piece> = dealt
            .iter()
            .map(|d| {
                let slots: Vec<LeafSlot<'_>> = d.iter().map(|x| x.map(|ch| (&c, ch))).collect();
                leaf(&c, &profile, &slots).unwrap()
            })
            .collect();
        assert_eq!(leaves[0].digest(), leaves[1].digest(), "one circuit for every leaf");
        let statement: Vec<([u8; 32], Commitment)> =
            leaves.iter().map(|p| (p.digest(), p.commit())).collect();
        let ch = shared_challenges(&statement);
        let mut proofs = Vec::new();
        for (i, p) in leaves.iter().enumerate() {
            let label = (i as u64).to_le_bytes();
            let pp = p.prove(Some(&ch), &label, false);
            p.verify(Some(&ch), &label, &statement[i].1, &pp.proof).expect("the leaf verifies");
            proofs.push(pp.proof);
        }
        assert!(shared_balance(&proofs), "the leaves balance together");
        proofs.pop();
        assert!(!shared_balance(&proofs), "a leaf left out unbalances the bus");
    }

    #[test]
    fn a_false_beta_is_refused_in_pieces() {
        let bad = SMALL
            .replace("F 2 Beta eq 0 5 4 0 : 1", "F 2 Beta eq 0 5 0 0 : 1")
            .replace("F 3 Trans eq 0 5 4 0 : 2 0", "F 3 Refl eq 0 5 5 0 :")
            .replace("F 4 Symm eq 0 4 5 0 : 3", "F 4 Refl eq 0 4 4 0 :");
        let mut parts: Vec<Rows> = (0..3999).map(|_| Rows::parse(SMALL).unwrap()).collect();
        parts.push(Rows::parse(&bad).unwrap());
        let c = LeanCircuit::build_with(&Rows::merge(parts), false).unwrap();
        let total: usize = c.chunks().iter().map(|x| x.2).sum();
        assert!(!shared(&c, total / 2 + 1, false).1);
    }
}

/// Measurement switches (unsound): `LEAN_ROWS_ABLATE=RANK,LBR` drops those
/// checks to see what they cost.
fn ablate(what: &str) -> bool {
    std::env::var("LEAN_ROWS_ABLATE").is_ok_and(|v| v.split(',').any(|w| w == what))
}

// ---------------------------------------------------------------- cones

/// Many declarations' derivations in one numbering: equal terms, levels,
/// level lists and contexts share one id across all of them, and a
/// declaration is one row by key. A citation in one declaration's check
/// then names the very row whose goal another declaration's check proves.
/// The shared side (terms and declarations) is proven once; each batch of
/// judgements and goals refers to it over the shared bus.
pub struct Cone {
    pub shared: Rows,
    levels: HashMap<(u8, u64, u64), u64>,
    lists: HashMap<Option<(u64, u64)>, u64>,
    ctxs: HashMap<Option<(u64, u64)>, u64>,
    terms: HashMap<(u64, [(u64, u64); 3]), usize>,
    decls: HashMap<(u64, u64), usize>,
    span: i64,
}

impl Default for Cone {
    fn default() -> Self {
        Self::new()
    }
}

impl Cone {
    pub fn new() -> Cone {
        let mut ctxs = HashMap::new();
        // The empty context is 0, as goals expect.
        ctxs.insert(None, 0);
        Cone {
            shared: Rows::default(),
            levels: HashMap::new(),
            lists: HashMap::new(),
            ctxs,
            terms: HashMap::new(),
            decls: HashMap::new(),
            span: 0,
        }
    }

    /// Adds one declaration's derivation: its terms and cited
    /// declarations join the shared side, its judgements and goal are
    /// appended to `batch` in the shared numbering.
    pub fn add(&mut self, r: Rows, batch: &mut Rows) -> Result<(), String> {
        let lc = &r.local;
        // Declarations by key.
        let mut decl = Vec::with_capacity(r.decls.len());
        for d in &r.decls {
            if d.key == (0, 0) {
                return Err("a declaration without a key (an old dump)".into());
            }
            let i = match self.decls.get(&d.key) {
                Some(&i) => {
                    let e = &self.shared.decls[i];
                    if (e.unfold, e.levels, e.kind, e.order) != (d.unfold, d.levels, d.kind, d.order) {
                        return Err(format!("declaration {:?} differs between derivations", d.key));
                    }
                    i
                }
                None => {
                    let i = self.shared.decls.len();
                    self.shared.decls.push(DeclRec {
                        value: None,
                        ty: None,
                        checked: false,
                        ..d.clone()
                    });
                    self.decls.insert(d.key, i);
                    i
                }
            };
            self.shared.decls[i].checked |= d.checked;
            self.span = self.span.max(d.order + 1);
            decl.push(i);
        }
        // Levels and lists: opaque ids, shared by structure.
        let mut lv = Vec::with_capacity(lc.levels.len());
        for &(kind, a, b) in &lc.levels {
            let m = |x: u64| lv[x as usize];
            let key = match kind {
                0 => (0, 0, 0),
                1 => (1, m(a), 0),
                2 | 3 => (kind, m(a), m(b)),
                _ => (4, a, 0),
            };
            let n = self.levels.len() as u64;
            lv.push(*self.levels.entry(key).or_insert(n));
        }
        let mut ls: Vec<u64> = Vec::with_capacity(lc.lists.len());
        for &row in &lc.lists {
            let key = row.map(|(h, t)| (lv[h as usize], ls[t as usize]));
            let n = self.lists.len() as u64;
            ls.push(*self.lists.entry(key).or_insert(n));
        }
        // Terms.
        let mut tm = Vec::with_capacity(r.terms.len());
        for t in &r.terms {
            let mut t = t.clone();
            for c in t.child.iter_mut().flatten() {
                *c = tm[*c];
            }
            if t.kind == K_LEAF {
                let sub = t.x[0].f.lo;
                if sub == SUB_CONST {
                    t.x[1] = vi(decl[t.x[1].f.lo as usize] as u64);
                    t.x[2] = vi(*ls.get(t.x[2].f.lo as usize).ok_or("a list out of range")?);
                } else if sub == SUB_SORT {
                    t.x[1] = vi(*lv.get(t.x[1].f.lo as usize).ok_or("a level out of range")?);
                }
            }
            if t.kind == K_PROJ {
                t.x[0] = vi(decl[t.x[0].f.lo as usize] as u64);
            }
            let w = |i: usize| match t.child[i] {
                Some(c) => (c as u64, u64::MAX),
                None => (t.x[i].f.lo, t.x[i].f.hi),
            };
            let key = (t.kind, [w(0), w(1), w(2)]);
            let i = match self.terms.get(&key) {
                Some(&i) => i,
                None => {
                    let i = self.shared.terms.len();
                    self.span = self.span.max(t.rank + t.lbr);
                    self.shared.terms.push(t);
                    self.terms.insert(key, i);
                    i
                }
            };
            tm.push(i);
        }
        // Declared types and values, which must agree where cited twice.
        for (d, &i) in r.decls.iter().zip(&decl) {
            let e = &mut self.shared.decls[i];
            for (mine, theirs) in [(d.ty, &mut e.ty), (d.value, &mut e.value)] {
                if let Some(x) = mine {
                    let x = tm[x];
                    match theirs {
                        None => *theirs = Some(x),
                        Some(y) if *y == x => {}
                        Some(_) => return Err(format!("declaration {:?} has two types or values", d.key)),
                    }
                }
            }
        }
        // Contexts.
        let mut cx: Vec<u64> = Vec::with_capacity(lc.ctxs.len());
        for &row in &lc.ctxs {
            let key = row.map(|(q, a)| (cx[q as usize], tm[a as usize] as u64));
            let n = self.ctxs.len() as u64;
            cx.push(*self.ctxs.entry(key).or_insert(n));
        }
        // Judgements and the goal.
        let base = batch.facts.len();
        for mut f in r.facts {
            for x in 0..4 {
                if f.is_term[x] {
                    f.raw[x] = tm[f.raw[x] as usize] as u64;
                }
            }
            let ctx = |v: u64| cx.get(v as usize).copied().ok_or("a context out of range");
            let lvl = |v: u64| lv.get(v as usize).copied().ok_or("a level out of range");
            let lst = |v: u64| ls.get(v as usize).copied().ok_or("a list out of range");
            match f.tag {
                F_HAS | F_INFER | F_EQ => f.raw[0] = ctx(f.raw[0])?,
                F_LINST => f.raw[1] = lst(f.raw[1])?,
                F_LINSTL => {
                    f.raw[0] = lvl(f.raw[0])?;
                    f.raw[1] = lst(f.raw[1])?;
                    f.raw[2] = lvl(f.raw[2])?;
                }
                F_LEQ => {
                    f.raw[0] = lvl(f.raw[0])?;
                    f.raw[1] = lvl(f.raw[1])?;
                }
                _ => {}
            }
            for p in &mut f.premises {
                *p += base;
            }
            self.span = self.span.max(f.rank.max(f.owner));
            batch.facts.push(f);
        }
        for g in r.goals {
            batch.goals.push(GoalRec {
                decl: decl[g.decl],
                ty: (g.ty.0 + base, g.ty.1 + base),
                value: g.value.map(|(a, b)| (a + base, b + base)),
            });
        }
        Ok(())
    }

    /// The shared side, with one limb width for every build of the cone.
    pub fn finish(mut self) -> Rows {
        self.shared.limb = Some(((((self.span + 2) as f64).log2() / 2.0).ceil() as usize).max(4));
        self.shared
    }

    /// How many terms and declarations (checked, in blocks, free).
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        let d = &self.shared.decls;
        (
            self.shared.terms.len(),
            d.iter().filter(|d| d.checked).count(),
            d.iter().filter(|d| !d.checked && d.kind >= 5).count(),
            d.iter().filter(|d| !d.checked && d.kind < 5).count(),
        )
    }
}
