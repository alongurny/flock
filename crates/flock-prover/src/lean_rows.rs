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

use flock_core::{
    challenger::FsChallenger,
    circuit::{
        builder::{CircuitShape, GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
        bus::{BusDirection, BusTerm, RowTag},
    },
    element_r1cs::{ElementTableBuilder, ElementTableType},
    pcs::PcsParams,
    schedule::{IoWord, TableType},
    union::UnionInstance,
    verifier::{FlockVerifyError, verify_ligerito_union_circuit},
};
use flock_field::F128;

use crate::prover::{UnionElementSlotInput, prove_fast_ligerito_union_circuit};

/// Tuple width: tag, id, kind, four fields, rank.
const W: usize = 8;

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
#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Debug)]
struct TermRec {
    kind: u64,
    /// Parts: a bound variable's `g^j`; a leaf's kind and payload; children.
    x: [V; 3],
    /// Children (term indices) for kinds that have them, by part.
    child: [Option<usize>; 3],
    lbr: i64,
    rank: i64,
}

#[derive(Clone, Debug)]
struct DeclRec {
    unfold: u64,
    levels: u64,
    value: Option<usize>,
}

#[derive(Clone, Debug)]
struct FactRec {
    rule: String,
    tag: u64,
    /// Raw fields from the dump.
    raw: [u64; 4],
    /// Which fields are terms.
    is_term: [bool; 4],
    premises: Vec<usize>,
    rank: i64,
}

/// A parsed derivation.
pub struct Rows {
    terms: Vec<TermRec>,
    decls: Vec<DeclRec>,
    facts: Vec<FactRec>,
}

impl Rows {
    pub fn parse(text: &str) -> Result<Rows, String> {
        let mut terms: Vec<TermRec> = Vec::new();
        let mut decl_at: HashMap<u64, usize> = HashMap::new();
        let mut decls = Vec::new();
        let mut facts = Vec::new();
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            let n = |i: usize| -> Result<u64, String> {
                w.get(i)
                    .ok_or(format!("short line: {line}"))?
                    .parse::<u64>()
                    .map_err(|e| format!("{e}: {line}"))
            };
            match w.first().copied() {
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
                    decls.push(DeclRec {
                        unfold: n(2)?,
                        levels: n(3)?,
                        value: (v >= 0).then_some(v as usize),
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
        Ok(Rows {
            terms,
            decls,
            facts,
        })
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
    fn recv(&mut self, t: [T; W]) {
        self.ports.push(Port {
            send: false,
            t,
            mult: Vec::new(),
        });
    }
    fn send(&mut self, t: [T; W]) {
        self.ports.push(Port {
            send: true,
            t,
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
        self.recv([
            k(TAG_FACT),
            T::C(id),
            k(tag),
            x[0],
            x[1],
            x[2],
            x[3],
            T::C(pr),
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
    Decl,
    Fact(&'static str),
    Assumed,
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
        Of::Decl => {
            let mut s = Spec::new("decl");
            let x: Vec<usize> = (0..3).map(|i| s.val(E::Own(Part::X(i)))).collect();
            s.send([k(TAG_DECL), T::Id, tc(x[0]), tc(x[1]), tc(x[2]), Z, Z, Z]);
            s
        }
        Of::Term(kind) => term_spec(kind),
        Of::Assumed => {
            let mut s = Spec::new("assumed");
            let tag = s.val(E::Own(Part::Kind));
            let x: Vec<usize> = (0..4).map(|i| s.val(E::Own(Part::X(i)))).collect();
            let r = s.val(E::Own(Part::R));
            s.send([
                k(TAG_FACT),
                T::Id,
                tc(tag),
                tc(x[0]),
                tc(x[1]),
                tc(x[2]),
                tc(x[3]),
                tc(r),
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
    let fld = |s: &mut Spec, i: usize| s.val(own(Part::X(i)));
    let one = T::K(F128::ONE);
    let send = |s: &mut Spec, tag: u64, x: [T; 4]| {
        s.send([k(TAG_FACT), T::Id, k(tag), x[0], x[1], x[2], x[3], tc(r)])
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
            if rule == "Delta0" {
                s.recv([k(TAG_DECL), tc(d), one, Z, tc(b), Z, Z, Z]);
            } else {
                let (lv, v) = (
                    s.val(E::Dcl(f(1), Part::X(1))),
                    s.val(E::Dcl(f(1), Part::X(2))),
                );
                s.recv([k(TAG_DECL), tc(d), one, tc(lv), tc(v), Z, Z, Z]);
                s.prem(0, F_LINST, [c(v), c(ls), c(b), Some(Z)], r);
            }
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
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
fn table_of(rows: &Rows, i: usize) -> Of {
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
        "Delta" if f.premises.is_empty() => "Delta0",
        "Delta" => "DeltaL",
        "InstBind" => {
            if rows.terms[f.raw[0] as usize].kind == K_LAM {
                "InstLam"
            } else {
                "InstPi"
            }
        }
        _ => return Of::Assumed,
    };
    Of::Fact(r)
}

const FACT_TABLES: [&str; 14] = [
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

/// A derivation laid out as tables, ready to prove.
pub struct LeanCircuit {
    pub nu: usize,
    pub limb: usize,
    shape: CircuitShape,
    values: Vec<F128>,
    slots: Vec<(SlotId, String, usize, usize, usize)>,
    pub build_ms: f64,
}

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
        let t0 = Instant::now();
        // Tables and which rows go to each.
        let mut ofs: Vec<Of> = vec![Of::Range(0), Of::Range(1), Of::Decl];
        for kd in [K_BVAR, K_LEAF, K_APP, K_LAM, K_PI, K_LET, K_PROJ] {
            ofs.push(Of::Term(kd));
        }
        for r in FACT_TABLES {
            ofs.push(if r == "Assumed" {
                Of::Assumed
            } else {
                Of::Fact(r)
            });
        }
        let at = |of: Of| ofs.iter().position(|&o| o == of).unwrap();
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); ofs.len()];
        let mut term_at = vec![(0usize, 0usize); rows.terms.len()];
        for (i, t) in rows.terms.iter().enumerate() {
            let s = at(Of::Term(t.kind));
            term_at[i] = (s, members[s].len());
            members[s].push(i);
        }
        let mut fact_at = vec![(0usize, 0usize); rows.facts.len()];
        for i in 0..rows.facts.len() {
            let s = at(table_of(rows, i));
            fact_at[i] = (s, members[s].len());
            members[s].push(i);
        }
        let ds = at(Of::Decl);
        members[ds] = (0..rows.decls.len()).collect();
        // Range limbs wide enough for every rank and bound: 2^(2L) > span.
        let span = rows
            .facts
            .iter()
            .map(|f| f.rank)
            .chain(rows.terms.iter().map(|t| t.rank + t.lbr))
            .max()
            .unwrap_or(0);
        let limb = ((((span + 2) as f64).log2() / 2.0).ceil() as usize).max(4);
        for l in 0..2 {
            members[at(Of::Range(l))] = (0..1usize << limb).collect();
        }
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
        let cost = |nu: usize| -> usize {
            members
                .iter()
                .zip(&widths)
                .map(|(m, &(w, p))| m.len().div_ceil(1 << nu) * (((w + 4 * p) << nu) + (1 << 17)))
                .sum()
        };
        let nu = std::env::var("LEAN_ROWS_NU")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (8..=22).min_by_key(|&nu| cost(nu)).unwrap());
        let mask = (1usize << nu) - 1;
        let mut chunk_base = Vec::with_capacity(ofs.len());
        let mut n_slots = 0;
        for m in &members {
            chunk_base.push(n_slots);
            n_slots += m.len().div_ceil(1 << nu);
        }
        let chunk_base = &chunk_base;
        let id = move |s: usize, j: usize| {
            int((((chunk_base[s] + (j >> nu)) as u64) << 32) + (j & mask) as u64)
        };
        let term_id = |i: usize| id(term_at[i].0, term_at[i].1);
        let fact_id = |i: usize| id(fact_at[i].0, fact_at[i].1);
        let decl_id = |i: usize| id(ds, i);
        // Fact fields as encoded values.
        let field = |i: usize, x: usize| -> V {
            let f = &rows.facts[i];
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
            let t = &rows.terms[i];
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
            }
        };
        let resolve = |tr: &TR, own: usize, is_fact: bool| -> usize {
            fn go(tr: &TR, own: usize, is_fact: bool, rows: &Rows) -> usize {
                match tr {
                    TR::Own => own,
                    TR::Fld(x) => {
                        assert!(is_fact);
                        rows.facts[own].raw[*x] as usize
                    }
                    TR::PFld(j, x) => rows.facts[rows.facts[own].premises[*j]].raw[*x] as usize,
                    TR::Child(t, c) => {
                        rows.terms[go(t, own, is_fact, rows)].child[*c].expect("a child")
                    }
                }
            }
            go(tr, own, is_fact, rows)
        };
        let eval = |e: &E, of: Of, own: usize| -> V {
            let is_fact = matches!(of, Of::Fact(_) | Of::Assumed);
            match e {
                E::Own(p) => match of {
                    Of::Term(_) => term_part(own, *p),
                    Of::Decl => {
                        let d = &rows.decls[own];
                        match p {
                            Part::X(0) => vi(d.unfold),
                            Part::X(1) => vi(d.levels),
                            Part::X(2) => V {
                                f: d.value.map_or(F128::ZERO, term_id),
                                e: 0,
                            },
                            _ => unreachable!(),
                        }
                    }
                    _ => match p {
                        Part::Id => V {
                            f: fact_id(own),
                            e: 0,
                        },
                        Part::Kind => vi(rows.facts[own].tag),
                        Part::X(x) => field(own, *x),
                        Part::R => ve(rows.facts[own].rank),
                        Part::L => unreachable!(),
                    },
                },
                E::Prem(j, p) => {
                    let q = rows.facts[own].premises[*j];
                    match p {
                        Part::Id => V {
                            f: fact_id(q),
                            e: 0,
                        },
                        Part::Kind => vi(rows.facts[q].tag),
                        Part::X(x) => field(q, *x),
                        Part::R => ve(rows.facts[q].rank),
                        Part::L => unreachable!(),
                    }
                }
                E::Tm(tr, p) => term_part(resolve(tr, own, is_fact), *p),
                E::Dcl(tr, p) => {
                    let t = &rows.terms[resolve(tr, own, is_fact)];
                    let d = &rows.decls[t.x[1].f.lo as usize];
                    match p {
                        Part::X(1) => vi(d.levels),
                        Part::X(2) => V {
                            f: d.value.map_or(F128::ZERO, term_id),
                            e: 0,
                        },
                        _ => unreachable!(),
                    }
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
            let mut out = Vec::with_capacity(members[s].len());
            for &own in &members[s] {
                let mut z = vec![F128::ZERO; cols.len()];
                for (y, (_, src)) in sp.cols.iter().enumerate() {
                    let Some(src) = src else { continue };
                    z[y] = match src {
                        Src::Val(e) => eval(e, *of, own).f,
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
        // Count every tuple received.
        let tuple = |t: &[T; W], z: &[F128], s: usize, j: usize| -> [(u64, u64); W] {
            std::array::from_fn(|i| {
                let v = match t[i] {
                    T::C(c) => z[c],
                    T::K(v) => v,
                    T::Id => id(s, j),
                };
                (v.lo, v.hi)
            })
        };
        let mut asked: HashMap<[(u64, u64); W], u64> = HashMap::new();
        for (s, sp) in specs.iter().enumerate() {
            for p in sp.ports.iter().filter(|p| !p.send) {
                for (j, z) in tables[s].iter().enumerate() {
                    *asked.entry(tuple(&p.t, z, s, j)).or_default() += 1;
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
        if let Some((t, n)) = asked.iter().next().filter(|_| strict) {
            return Err(format!(
                "{} tuples asked for but never given, e.g. {t:?} ({n} times)",
                asked.len()
            ));
        }
        // The shape: one gate per row, inputs in order.
        let mut sb = ShapeBuilder::new(nu);
        let mut values = Vec::new();
        let mut slots = Vec::new();
        for (s, sp) in specs.iter().enumerate() {
            let cols: Vec<Col> = sp.cols.iter().map(|c| c.0).collect();
            let io = io_index(&cols);
            for (c, chunk) in tables[s].chunks(1 << nu).enumerate() {
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
                                    offset: ((chunk_base[s] + c) as u64) << 32,
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
                            if matches!(c.1, Some(Src::RangeV(_))) {
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
        }
        let shape = sb.finish().map_err(|e| format!("{e:?}"))?;
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(LeanCircuit {
            nu,
            limb,
            shape,
            values,
            slots,
            build_ms,
        })
    }

    fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.shape.registry, self.shape.counts.clone())
    }

    fn pcs(union: &UnionInstance<'_>) -> PcsParams {
        crate::leanxmss::circuit::XmssCircuit::pcs_params(union)
    }

    /// Fills, proves and verifies; `tamper` changes one witness word first
    /// (the proof must then fail).
    pub fn prove_and_verify(&self, tamper: bool) -> Result<Report, FlockVerifyError> {
        let t = Instant::now();
        let mut w = self.shape.run(&self.values, &[]);
        let witness_ms = t.elapsed().as_secs_f64() * 1e3;
        let union = self.union();
        let pcs = Self::pcs(&union);
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
            // A Trans row's own `a` (or any first live cell): one word off.
            let (_, z) = el
                .iter_mut()
                .find(|(_, z)| z.iter().any(|v| !v.is_zero()))
                .unwrap();
            let i = z.iter().position(|v| !v.is_zero()).unwrap();
            z[i] += F128::ONE;
        }
        let inputs: Vec<UnionElementSlotInput<'_>> = el
            .into_iter()
            .map(|(_, z)| {
                UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z))
            })
            .collect();
        let t = Instant::now();
        let mut ch = FsChallenger::new(b"cas-lean-rows-v0");
        let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
            &union,
            &self.shape.circuit,
            &w.public,
            &pcs,
            Vec::new(),
            inputs,
            &mut ch,
        );
        let prove_ms = t.elapsed().as_secs_f64() * 1e3;
        let proof_bytes = bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0);
        let t = Instant::now();
        let mut ch = FsChallenger::new(b"cas-lean-rows-v0");
        verify_ligerito_union_circuit(
            &union,
            &self.shape.circuit,
            &w.public,
            &[],
            &commitment,
            &proof,
            &pcs,
            &mut ch,
        )?;
        let verify_ms = t.elapsed().as_secs_f64() * 1e3;
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
            build_ms: self.build_ms,
            witness_ms,
            prove_ms,
            verify_ms,
            proof_bytes,
        })
    }
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
    const SMALL: &str = "\
T 0 leaf sort 1 0
T 1 bvar 0
T 2 pi 0 0
T 3 lam 0 1
T 4 leaf fvar 0 0
T 5 app 3 4
D 0 1 0 -1
F 0 Refl eq 0 4 4 0 :
F 1 InstHit inst 1 0 4 4 :
F 2 Beta eq 0 5 4 0 : 1
F 3 Trans eq 0 5 4 0 : 2 0
F 4 Symm eq 0 4 5 0 : 3
F 5 InstClosed inst 0 0 4 0 :
F 6 FVar infer 0 4 0 0 :
F 7 Lam infer 0 3 2 0 :
F 8 Refl eq 0 2 2 0 :
F 9 Refl eq 0 0 0 0 :
F 10 App infer 0 5 0 0 : 7 8 6 9 5
";

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
}

/// Measurement switches (unsound): `LEAN_ROWS_ABLATE=RANK,LBR` drops those
/// checks to see what they cost.
fn ablate(what: &str) -> bool {
    std::env::var("LEAN_ROWS_ABLATE").is_ok_and(|v| v.split(',').any(|w| w == what))
}
