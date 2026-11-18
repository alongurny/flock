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
//! Rules with constraints here: Refl, Symm, Trans, CongApp, CongProj, App,
//! Const, FVar, Beta, Delta, Zeta, InstClosed, InstHit, InstApp, InstBind,
//! InstLet, InstProj, level substitution through a node (LInstApp,
//! LInstBind, LInstLet, LInstProj), and the rules of contexts: HasLast,
//! HasUp, Lam, Let, CongLam, CongPi. A context is a row beside the terms
//! (its parent, the type added, `g^length`), so its id is a row tag too;
//! a term carries `F = g^fv`, a bound on its free variables, which a
//! context entry and a binder's closed body are held to. Any other judgement row
//! goes to an *assumed* table that sends its fact unchecked, and the cited
//! declarations are sent unchecked (in cas they are bound to the store).
//! Levels, level lists and contexts are rows among the terms: a sort
//! receives its level, a constant its level list, and every term carries a
//! mark that is zero only when no level parameter occurs in it. Not yet
//! checked: a pi's universe, level equivalence, and the rules of
//! inductive types, projections, literals and quotients.

use std::{collections::HashMap, sync::Arc, time::Instant};

use serde::{Deserialize, Serialize};

use flock_core::{
    challenger::Challenger,
    challenger::FsChallenger,
    circuit::{
        Circuit,
        boolean::BooleanGate,
        builder::{
            CircuitShape, CircuitWitness, GateType, ShapeBuilder, SlotId, SlotWitness, Wire,
        },
        bus::{self, BusDirection, BusTerm, RowTag, SharedChallenges},
    },
    element_r1cs::{ElementTableBuilder, ElementTableType},
    pcs::Commitment,
    pcs::PcsParams,
    proof::R1csProofCircuitMerged,
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
/// Odd, so that β and the tuple's γs pair up in the challenge seed.
const W: usize = 11;

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
/// A leaf slot's chunk word, sent once ([`leaf`]).
const TAG_CHUNK: u64 = 7;
/// A checked declaration's key, sent once to the statement.
const TAG_STMT: u64 = 8;
/// A context: its id, its parent, the type added, and `g^length`.
const TAG_CTX: u64 = 9;
/// Padding slots' chunk numbers start here, above every side's.
/// A level row: `[id, kind, a, b, g^rank]`.
const TAG_LVL: u64 = 10;
/// A level list row: `[id, head, tail, g^length]`.
const TAG_LIST: u64 = 11;
/// Inside a table of several rules: the fact a row's rule derives,
/// `[id, kind, fields, rank, owner]`, from the rule to the row itself.
const TAG_UFACT: u64 = 12;
/// What the reduction rules read of a declaration: `[id, what, inductive
/// type (of a constructor), g^a, g^b]` (see [`InfoRec`]).
const TAG_INFO: u64 = 13;
/// A declaration's key: `[id, key words]`.
const TAG_KEY: u64 = 14;
/// A recursor's rule: `[recursor, constructor, g^fields, right-hand side]`.
const TAG_RULE: u64 = 15;
/// The key of a constant the kernel computes with: `[number, key words]`,
/// from the verifier, to a rule that reads the constant ([`PRIM_NAT`]).
const TAG_PRIM: u64 = 16;
/// The type of number literals and the type of string literals.
pub const PRIM_NAT: u64 = 1;
pub const PRIM_STRING: u64 = 20;
const PAD_CHUNK: u64 = 1 << 31;
// Term kinds.
const K_BVAR: u64 = 1;
const K_LEAF: u64 = 2;
const K_APP: u64 = 3;
const K_LAM: u64 = 4;
const K_PI: u64 = 5;
const K_LET: u64 = 6;
const K_PROJ: u64 = 7;
/// Contexts are rows beside the terms: the empty one, and a parent with a type.
const K_CTX0: u64 = 8;
const K_CTX: u64 = 9;
/// A free variable, `g^n` its level.
const K_FVAR: u64 = 10;
/// Tables of the leaves (their rows keep the kind `K_LEAF`).
const K_SORT: u64 = 11;
const K_CONST: u64 = 12;
const K_LIT: u64 = 13;
const K_STR: u64 = 14;
/// Levels: zero or a parameter (`a` is 0, or the index plus one), a
/// successor, a maximum, an impredicative maximum; level lists.
const K_LLEAF: u64 = 15;
const K_LSUCC: u64 = 16;
const K_LMAX: u64 = 17;
const K_LIMAX: u64 = 18;
const K_NIL: u64 = 19;
const K_CONS: u64 = 20;
/// Every table of term rows, in order.
const TERM_TABLES: [u64; 19] = [
    K_BVAR, K_SORT, K_CONST, K_LIT, K_STR, K_APP, K_LAM, K_PI, K_LET, K_PROJ, K_CTX0, K_CTX, K_FVAR, K_LLEAF,
    K_LSUCC, K_LMAX, K_LIMAX, K_NIL, K_CONS,
];

/// The table a term row goes to.
fn table_kind(t: &TermRec) -> u64 {
    if t.kind != K_LEAF {
        return t.kind;
    }
    match t.x[0].f.lo {
        SUB_SORT => K_SORT,
        SUB_CONST => K_CONST,
        SUB_LIT => K_LIT,
        _ => K_STR,
    }
}
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
/// `LLe(l, m)`: l is at most m for every assignment of the parameters.
/// These rows are not in the derivation: [`certify`] adds them.
const F_LLE: u64 = 8;
/// Facts about application spines, added by [`certify_spines`] for the
/// rules that read one: `Spine(t, h, n)`, t is h applied to n arguments;
/// `Pre(t, p, n)`, p is t without its last n arguments; `Ext(u, v, w, n)`,
/// w is u applied to the last n arguments of v.
const F_SPINE: u64 = 9;
const F_PRE: u64 = 10;
const F_EXT: u64 = 11;
/// The walk of a structure's constructor type that types a projection,
/// added by [`certify_spines`]: `PTel(g, q, r)`, r is the constructor's
/// type at the levels and the arguments of q (the structure applied to
/// parameters); `FTel(g, proj S k x, r, prop)`, r is what is left of it
/// after the parameters of x's type and the first k fields, each replaced
/// by its projection. prop is 1 when every field passed that a later type
/// mentions was shown a proof, 0 when the structure is not a proposition.
const F_PTEL: u64 = 12;
const F_FTEL: u64 = 13;
/// `NZ(l)`: the level l is not zero under any assignment of the parameters
/// that the kernel would read as zero, so `sort l` is not the sort of
/// propositions: l is a successor or a parameter, a maximum with such a
/// side, or an impredicative maximum with such a second side. A level
/// without this is zero under every assignment.
const F_NZ: u64 = 14;

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
    /// One more than the largest free variable level (0 without any).
    fv: i64,
    /// Whether a level parameter occurs in it.
    #[serde(default)]
    lp: bool,
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
    /// What the reduction rules read of it.
    #[serde(default)]
    info: InfoRec,
}

/// A constructor (what 2): its inductive type's key, parameters `a` and
/// fields `b`. A recursor (what 3): `a` arguments before the indices and
/// `b` indices. An inductive type is what 1, a quotient constant what 4.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct InfoRec {
    what: u64,
    key: (u64, u64),
    a: u64,
    b: u64,
}

/// A recursor's rule for one constructor: the recursor's declaration row,
/// the constructor's key, its fields, and the right-hand side.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct RuleRec {
    rec: usize,
    ctor: (u64, u64),
    fields: u64,
    rhs: usize,
}

/// A declaration's goal: its row and the judgements that establish it.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct GoalRec {
    decl: usize,
    /// `Infer([], ty, s)` and `Eq([], s, sort u)`.
    ty: (usize, usize),
    /// `Infer([], value, t)` and `Eq([], t, ty)`.
    value: Option<(usize, usize)>,
    /// The empty context's row.
    ctx: usize,
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
    /// The recursor rules (of a cone's shared side).
    #[serde(default)]
    rules: Vec<RuleRec>,
    /// The file's own level, list and context rows (local ids), kept only
    /// to put them in a cone's shared numbering.
    #[serde(skip)]
    local: Local,
}

#[derive(Default)]
struct Local {
    /// The canonical number of each declaration row.
    decl_no: Vec<u64>,
}

impl Rows {
    pub fn parse(text: &str) -> Result<Rows, String> {
        Self::parse_with(text, None)
    }

    /// Parses a derivation; with the cone's declaration rows (`info`), the
    /// rules that read a constructor or a recursor get their certificates.
    pub fn parse_with(text: &str, info: Option<&Rows>) -> Result<Rows, String> {
        let mut rules: Vec<RuleRec> = Vec::new();
        let mut terms: Vec<TermRec> = Vec::new();
        let mut decl_at: HashMap<u64, usize> = HashMap::new();
        let mut decls = Vec::new();
        let mut facts = Vec::new();
        let mut local = Local::default();
        // Each context's row among the terms (contexts follow the terms).
        let mut ctx_at: Vec<usize> = Vec::new();
        let mut goal_line: Option<((u64, u64), Vec<i64>)> = None;
        // Levels and level lists first: they are rows among the terms,
        // before every term that names one.
        let mut lvl_at: Vec<usize> = Vec::new();
        let mut list_at: Vec<usize> = Vec::new();
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            let n = |i: usize| -> Result<u64, String> {
                w.get(i)
                    .ok_or(format!("short line: {line}"))?
                    .parse::<u64>()
                    .map_err(|e| format!("{e}: {line}"))
            };
            let lv = |i: usize| -> Result<usize, String> {
                lvl_at.get(n(i)? as usize).copied().ok_or(format!("a level after its use: {line}"))
            };
            let (kind, x, child) = match (w.first().copied(), w.get(2).copied()) {
                (Some("L"), Some("zero")) => (K_LLEAF, V::default(), [None; 3]),
                (Some("L"), Some("param")) => (K_LLEAF, vi(n(3)? + 1), [None; 3]),
                (Some("L"), Some("succ")) => (K_LSUCC, V::default(), [Some(lv(3)?), None, None]),
                (Some("L"), Some("max")) => (K_LMAX, V::default(), [Some(lv(3)?), Some(lv(4)?), None]),
                (Some("L"), Some("imax")) => (K_LIMAX, V::default(), [Some(lv(3)?), Some(lv(4)?), None]),
                (Some("S"), Some("nil")) => (K_NIL, V::default(), [None; 3]),
                (Some("S"), Some("cons")) => {
                    let t = *list_at.get(n(4)? as usize).ok_or(format!("a list after its use: {line}"))?;
                    (K_CONS, V::default(), [Some(lv(3)?), Some(t), None])
                }
                (Some("L" | "S"), _) => return Err(format!("a level row: {line}")),
                _ => continue,
            };
            let rank = child.iter().flatten().map(|&i| terms[i].rank + 1).max().unwrap_or(0);
            // A list keeps its length.
            let len = match kind {
                K_CONS => ve(terms[child[1].unwrap()].x[2].e + 1),
                K_NIL => ve(0),
                _ => V::default(),
            };
            if w[0] == "L" { lvl_at.push(terms.len()) } else { list_at.push(terms.len()) };
            terms.push(TermRec {
                kind,
                x: [x, V::default(), len],
                child,
                lbr: 0,
                rank,
                fv: 0,
                lp: x.f != F128::ZERO || child.iter().flatten().any(|&i| terms[i].lp),
            });
        }
        // Terms are numbered after them.
        let off = terms.len();
        let proper = |t: &TermRec| matches!(t.kind, K_BVAR..=K_PROJ | K_FVAR);
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            let n = |i: usize| -> Result<u64, String> {
                w.get(i)
                    .ok_or(format!("short line: {line}"))?
                    .parse::<u64>()
                    .map_err(|e| format!("{e}: {line}"))
            };
            match w.first().copied() {
                Some("C") => {
                    let (x, child) = if w[2] == "push" {
                        let q = *ctx_at.get(n(3)? as usize).ok_or(format!("a parent after its context: {line}"))?;
                        let a = n(4)? as usize + off;
                        if a >= terms.len() || !proper(&terms[a]) {
                            return Err(format!("a context's type is not a term: {line}"));
                        }
                        (ve(terms[q].x[2].e + 1), [Some(q), Some(a), None])
                    } else {
                        (ve(0), [None; 3])
                    };
                    ctx_at.push(terms.len());
                    terms.push(TermRec {
                        kind: if child[0].is_some() { K_CTX } else { K_CTX0 },
                        x: [V::default(), V::default(), x],
                        child,
                        lbr: 0,
                        rank: 0,
                        fv: 0,
                        lp: false,
                    });
                }
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
                    if n(1)? as usize + off != terms.len() || !ctx_at.is_empty() {
                        return Err(format!("terms out of order: {line}"));
                    }
                    let ch = |i: usize| -> Result<usize, String> {
                        let c = n(i)? as usize + off;
                        if c >= terms.len() || !proper(&terms[c]) {
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
                        "leaf" if w[3] == "fvar" => (
                            K_FVAR,
                            [ve(n(4)? as i64), V::default(), V::default()],
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
                            // A sort's level and a constant's level list are
                            // rows; a constant's declaration is mapped to
                            // its id later.
                            match sub {
                                SUB_SORT => {
                                    let l = *lvl_at.get(n(4)? as usize).ok_or(format!("no such level: {line}"))?;
                                    (K_LEAF, [vi(sub), V::default(), V::default()], [None, Some(l), None])
                                }
                                SUB_CONST => {
                                    let l = *list_at.get(n(5)? as usize).ok_or(format!("no such list: {line}"))?;
                                    (K_LEAF, [vi(sub), vi(n(4)?), V::default()], [None, None, Some(l)])
                                }
                                _ => (K_LEAF, [vi(sub), vi(n(4)?), vi(n(5)?)], [None; 3]),
                            }
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
                            [vi(n(3)?), ve(n(4)? as i64), V::default()],
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
                    let fv = match kind {
                        K_FVAR => x[0].e + 1,
                        _ => child.iter().flatten().map(|&i| terms[i].fv).max().unwrap_or(0),
                    };
                    terms.push(TermRec {
                        kind,
                        x,
                        child,
                        lbr,
                        rank,
                        fv,
                        lp: child.iter().flatten().any(|&i| terms[i].lp),
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
                        value: (v >= 0).then_some(v as usize + off),
                        ty: if more { Some(n(8)? as usize + off) } else { None },
                        kind: if more { n(7)? } else { 0 },
                        key: if more { (n(5)?, n(6)?) } else { (0, 0) },
                        order: if more {
                            w[9].parse().map_err(|e| format!("{e}: {line}"))?
                        } else {
                            0
                        },
                        checked: false,
                        info: InfoRec::default(),
                    });
                }
                Some("I") => {
                    let d = *decl_at.get(&n(1)?).ok_or(format!("no declaration: {line}"))?;
                    decls[d].info = match w[2] {
                        // The key is the constructor's when there is exactly one.
                        "induct" => InfoRec {
                            what: 1,
                            key: if n(5)? == 1 { (n(7)?, n(8)?) } else { (0, 0) },
                            a: n(3)?,
                            b: n(4)?,
                        },
                        "ctor" => InfoRec { what: 2, key: (n(3)?, n(4)?), a: n(5)?, b: n(6)? },
                        "rec" => InfoRec { what: 3, key: (0, 0), a: n(3)?, b: n(4)? },
                        "quot" => InfoRec { what: 4, key: (0, 0), a: n(3)?, b: 0 },
                        x => return Err(format!("declaration info {x}")),
                    };
                }
                Some("R") => {
                    let rec = *decl_at.get(&n(1)?).ok_or(format!("no declaration: {line}"))?;
                    rules.push(RuleRec { rec, ctor: (n(2)?, n(3)?), fields: n(4)?, rhs: n(5)? as usize + off });
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
                        F_HAS => [true, false, true, false],
                        F_INFER | F_EQ => [true, true, true, false],
                        F_INST => [true, false, true, true],
                        F_LINST | F_LINSTL => [true, true, true, false],
                        _ => [true, true, false, false],
                    };
                    let mut raw = [n(4)?, n(5)?, n(6)?, n(7)?];
                    // Fields name contexts, terms, levels and lists by
                    // their own numbers: here all are rows among the terms.
                    let (cx, tm, lv, ls) = (0, 1, 2, 3);
                    let what = match tag {
                        F_HAS => [Some(cx), None, Some(tm), None],
                        F_INFER | F_EQ => [Some(cx), Some(tm), Some(tm), None],
                        F_INST => [Some(tm), None, Some(tm), Some(tm)],
                        F_LINST => [Some(tm), Some(ls), Some(tm), None],
                        F_LINSTL => [Some(lv), Some(ls), Some(lv), None],
                        _ => [Some(lv), Some(lv), None, None],
                    };
                    for x in 0..4 {
                        let v = raw[x] as usize;
                        let at = match what[x] {
                            Some(0) => ctx_at.get(v).copied(),
                            Some(1) => (v + off < terms.len() && proper(&terms[v + off])).then_some(v + off),
                            Some(2) => lvl_at.get(v).copied(),
                            Some(_) => list_at.get(v).copied(),
                            None => continue,
                        };
                        raw[x] = at.ok_or(format!("field {x} names no row: {line}"))? as u64;
                    }
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
        certify(&mut terms, &mut facts);
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
        if let Some(info) = info {
            certify_spines(&terms, &decls, &mut facts, info);
        }
        // The goal: the declaration checked, by key among the cited rows or
        // as a row of its own, and every judgement's owner.
        let mut goals = Vec::new();
        if let Some((key, g)) = goal_line {
            let [
                kind,
                levels,
                ty,
                value,
                ti,
                te,
                vi_,
                ve_,
                _prop,
                order,
                empty,
            ] = g[..]
            else {
                return Err(format!("a goal row of {} fields", g.len() + 2));
            };
            let decl = match decls.iter().position(|d| d.key == key) {
                Some(i) => i,
                None => {
                    decls.push(DeclRec {
                        unfold: matches!(kind, 2 | 3) as u64,
                        levels: levels as u64,
                        value: (value >= 0).then_some(value as usize + off),
                        ty: Some(ty as usize + off),
                        kind: kind as u64,
                        key,
                        order,
                        checked: false,
                        info: InfoRec::default(),
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
                value: if vi_ >= 0 {
                    Some((fact(vi_)?, fact(ve_)?))
                } else {
                    None
                },
                ctx: *ctx_at
                    .get(empty as usize)
                    .filter(|&&c| terms[c].kind == K_CTX0)
                    .ok_or("the goal's context is not an empty one")?,
            });
        }
        Ok(Rows {
            terms,
            decls,
            facts,
            goals,
            limb: None,
            rules,
            local,
        })
    }

    /// Several derivations as one: indices shifted, nothing shared.
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
                g.ctx += nt;
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

/// Adds the level comparisons the universe rules need, which the
/// derivation leaves to its checker: a pi's universe is Lean's
/// simplification of `imax u v`, and two levels are equivalent by a
/// decision procedure. For each, rows `LLe(l, m)` built from a few sound
/// rules are added as premises, and the row is renamed to the rule that
/// uses them. A row for which they do not suffice keeps its name (and is
/// assumed).
fn certify(terms: &mut Vec<TermRec>, facts: &mut Vec<FactRec>) {
    struct Cx<'a> {
        terms: &'a [TermRec],
        base: usize,
        new: Vec<FactRec>,
        memo: HashMap<(usize, usize), Option<usize>>,
    }
    impl Cx<'_> {
        fn add(&mut self, rule: &str, l: usize, m: usize, premises: Vec<usize>) -> Option<usize> {
            self.new.push(FactRec {
                rule: rule.into(),
                tag: F_LLE,
                raw: [l as u64, m as u64, 0, 0],
                is_term: [true, true, false, false],
                premises,
                rank: 0,
                owner: 0,
            });
            Some(self.base + self.new.len() - 1)
        }
        fn le(&mut self, l: usize, m: usize) -> Option<usize> {
            if let Some(&r) = self.memo.get(&(l, m)) {
                return r;
            }
            let r = self.le_new(l, m);
            self.memo.insert((l, m), r);
            r
        }
        fn le_new(&mut self, l: usize, m: usize) -> Option<usize> {
            let (tl, tm) = (&self.terms[l], &self.terms[m]);
            let (lk, mk) = (tl.kind, tm.kind);
            let (lc, mc) = (tl.child, tm.child);
            if l == m {
                return self.add("LLeRefl", l, m, vec![]);
            }
            if lk == K_LLEAF && tl.x[0].f == F128::ZERO {
                return self.add("LLeZero", l, m, vec![]);
            }
            // A maximum on the left: both parts.
            if matches!(lk, K_LMAX | K_LIMAX) {
                if let (Some(a), Some(b)) = (self.le(lc[0].unwrap(), m), self.le(lc[1].unwrap(), m)) {
                    return self.add(if lk == K_LMAX { "LLeMaxL" } else { "LLeIMaxL" }, l, m, vec![a, b]);
                }
            }
            match mk {
                K_LSUCC => {
                    if lk == K_LSUCC {
                        if let Some(p) = self.le(lc[0].unwrap(), mc[0].unwrap()) {
                            return self.add("LLeSuccSucc", l, m, vec![p]);
                        }
                    }
                    let p = self.le(l, mc[0].unwrap())?;
                    self.add("LLeSucc", l, m, vec![p])
                }
                K_LMAX => {
                    if let Some(p) = self.le(l, mc[0].unwrap()) {
                        return self.add("LLeMaxR0", l, m, vec![p]);
                    }
                    let p = self.le(l, mc[1].unwrap())?;
                    self.add("LLeMaxR1", l, m, vec![p])
                }
                K_LIMAX => {
                    let p = self.le(l, mc[1].unwrap())?;
                    self.add("LLeIMaxR", l, m, vec![p])
                }
                _ => None,
            }
        }
    }
    // Zero and one, as rows.
    let mut find = |terms: &mut Vec<TermRec>, kind: u64, child: Option<usize>| -> usize {
        let hit = terms.iter().position(|t| t.kind == kind && t.child[0] == child && t.x[0].f == F128::ZERO);
        hit.unwrap_or_else(|| {
            terms.push(TermRec {
                kind,
                x: [V::default(); 3],
                child: [child, None, None],
                lbr: 0,
                rank: child.is_some() as i64,
                fv: 0,
                lp: false,
            });
            terms.len() - 1
        })
    };
    if !facts.iter().any(|f| matches!(f.rule.as_str(), "Pi" | "LEqNorm")) {
        return;
    }
    let zero = find(terms, K_LLEAF, None);
    let one = find(terms, K_LSUCC, Some(zero));
    let base = facts.len();
    let mut cx = Cx {
        terms,
        base,
        new: Vec::new(),
        memo: HashMap::new(),
    };
    let sort = |terms: &[TermRec], t: u64| -> Option<usize> {
        let t = &terms[t as usize];
        (t.kind == K_LEAF && t.x[0].f.lo == SUB_SORT).then(|| t.child[1].unwrap())
    };
    for i in 0..base {
        let f = &facts[i];
        if f.rule == "LEqNorm" && f.tag == F_LEQ {
            let (l, m) = (f.raw[0] as usize, f.raw[1] as usize);
            if let (Some(a), Some(b)) = (cx.le(l, m), cx.le(m, l)) {
                facts[i].rule = "LEqLe".into();
                facts[i].premises = vec![a, b];
            }
            continue;
        }
        if f.rule != "Pi" || f.premises.len() != 5 {
            continue;
        }
        let (su, sv) = (facts[f.premises[1]].raw[2], facts[f.premises[4]].raw[2]);
        let (Some(u), Some(v), Some(w)) = (sort(cx.terms, su), sort(cx.terms, sv), sort(cx.terms, f.raw[2])) else {
            continue;
        };
        // imax u v is: zero when v is; v when u ≤ v; max u v when 1 ≤ v,
        // which is u when also v ≤ u.
        let tw = &cx.terms[w];
        let is = |kind: u64| tw.kind == kind && tw.child[0] == Some(u) && tw.child[1] == Some(v);
        let (is_max, is_imax) = (is(K_LMAX), is(K_LIMAX));
        let v_zero = cx.terms[v].kind == K_LLEAF && cx.terms[v].x[0].f == F128::ZERO;
        let (rule, more) = if w == v && v_zero {
            ("PiZero", vec![])
        } else if is_imax {
            ("PiIMax", vec![])
        } else if let Some(p) = (w == v).then(|| cx.le(u, v)).flatten() {
            ("PiV", vec![p])
        } else if let Some(p) = is_max.then(|| cx.le(one, v)).flatten() {
            ("PiMax", vec![p])
        } else if let Some((p, q)) = (w == u).then(|| cx.le(v, u).zip(cx.le(one, v))).flatten() {
            ("PiU", vec![p, q])
        } else {
            continue;
        };
        facts[i].rule = rule.into();
        facts[i].premises.extend(more);
    }
    let new = std::mem::take(&mut cx.new);
    facts.extend(new);
    // Ranks again: the added rows first (their premises are added rows
    // before them), then the derivation's in order.
    for i in (base..facts.len()).chain(0..base) {
        facts[i].rank = facts[i].premises.iter().map(|&p| facts[p].rank + 1).max().unwrap_or(0);
    }
}

/// Adds what the rules that read an application spine need: a projection
/// of a constructor applied, and a recursor at a constructor applied. The
/// derivation leaves the spines to its checker; here each is spelled out
/// in `Spine`, `Pre` and `Ext` rows, and the row is renamed to the rule
/// that takes them as premises (`ProjCtorAt`, `Iota0`, `IotaL`).
fn certify_spines(terms: &[TermRec], decls: &[DeclRec], facts: &mut Vec<FactRec>, info: &Rows) {
    if !facts.iter().any(|f| matches!(f.rule.as_str(), "ProjCtor" | "Iota" | "Proj")) {
        return;
    }
    let by_key: HashMap<(u64, u64), InfoRec> = info.decls.iter().map(|d| (d.key, d.info)).collect();
    let rule_of: HashMap<((u64, u64), (u64, u64)), u64> =
        info.rules.iter().map(|r| ((info.decls[r.rec].key, r.ctor), r.fields)).collect();
    struct Cx<'a> {
        terms: &'a [TermRec],
        base: usize,
        new: Vec<FactRec>,
        memo: HashMap<(u64, [u64; 4]), usize>,
        clash: usize,
    }
    impl Cx<'_> {
        fn app(&self, t: usize) -> Option<(usize, usize)> {
            let t = &self.terms[t];
            (t.kind == K_APP).then(|| (t.child[0].unwrap(), t.child[1].unwrap()))
        }
        fn add(&mut self, rule: &str, tag: u64, raw: [u64; 4], premises: Vec<usize>) -> usize {
            if let Some(&i) = self.memo.get(&(tag, raw)) {
                // A walk met again must be the same walk.
                if matches!(tag, F_PTEL | F_FTEL) && self.new[i - self.base].premises != premises {
                    self.clash += 1;
                }
                return i;
            }
            self.new.push(FactRec {
                rule: rule.into(),
                tag,
                raw,
                is_term: [true, tag != F_NZ, matches!(tag, F_EXT | F_PTEL | F_FTEL), false],
                premises,
                rank: 0,
                owner: 0,
            });
            let i = self.base + self.new.len() - 1;
            self.memo.insert((tag, raw), i);
            i
        }
        /// The row saying the level l is never read as zero, if it is so.
        fn nz(&mut self, l: usize) -> Option<usize> {
            let t = &self.terms[l];
            let (kind, a, c0, c1) = (t.kind, t.x[0].f.lo, t.child[0], t.child[1]);
            let (rule, pr) = match kind {
                K_LSUCC => ("NZSucc", vec![]),
                K_LLEAF => (["NZPar0", "NZPar1", "NZPar2"].get((a as usize).wrapping_sub(1)).copied()?, vec![]),
                K_LMAX => match self.nz(c0?) {
                    Some(p) => ("NZMaxL", vec![p]),
                    None => ("NZMaxR", vec![self.nz(c1?)?]),
                },
                K_LIMAX => ("NZIMax", vec![self.nz(c1?)?]),
                _ => return None,
            };
            Some(self.add(rule, F_NZ, [l as u64, 0, 0, 0], pr))
        }
        /// The head and the number of arguments.
        fn head(&self, mut t: usize) -> (usize, u64) {
            let mut n = 0;
            while let Some((f, _)) = self.app(t) {
                (t, n) = (f, n + 1);
            }
            (t, n)
        }
        /// The head and the number of arguments, with the row saying so.
        fn spine(&mut self, t: usize) -> (usize, u64, usize) {
            match self.app(t) {
                None => (t, 0, self.add("SpineNil", F_SPINE, [t as u64, t as u64, 0, 0], vec![])),
                Some((f, _)) => {
                    let (h, n, p) = self.spine(f);
                    (h, n + 1, self.add("SpineApp", F_SPINE, [t as u64, h as u64, n + 1, 0], vec![p]))
                }
            }
        }
        /// t without its last n arguments, with the row saying so.
        fn pre(&mut self, t: usize, n: u64) -> Option<(usize, usize)> {
            if n == 0 {
                return Some((t, self.add("PreNil", F_PRE, [t as u64, t as u64, 0, 0], vec![])));
            }
            let (f, _) = self.app(t)?;
            let (p, row) = self.pre(f, n - 1)?;
            Some((p, self.add("PreApp", F_PRE, [t as u64, p as u64, n, 0], vec![row])))
        }
        /// The row saying w is u applied to the last n arguments of v.
        fn ext(&mut self, u: usize, v: usize, w: usize, n: u64) -> Option<usize> {
            if n == 0 {
                return (u == w).then(|| self.add("ExtNil", F_EXT, [u as u64, v as u64, w as u64, 0], vec![]));
            }
            let ((v1, x), (w1, y)) = (self.app(v)?, self.app(w)?);
            if x != y {
                return None;
            }
            let row = self.ext(u, v1, w1, n - 1)?;
            Some(self.add("ExtApp", F_EXT, [u as u64, v as u64, w as u64, n], vec![row]))
        }
        /// The rows that walk the constructor's type for the projection
        /// typed by fact i, and the rule and premises that fact gets. The
        /// derivation has every step among the fact's premises, in order.
        fn proj_walk(
            &mut self,
            facts: &[FactRec],
            i: usize,
            decls: &[DeclRec],
            decl_ix: &HashMap<(u64, u64), usize>,
            by_key: &HashMap<(u64, u64), InfoRec>,
        ) -> Option<(&'static str, Vec<usize>)> {
            let terms = self.terms;
            let f = &facts[i];
            let (g, e, t) = (f.raw[0], f.raw[1] as usize, f.raw[2] as usize);
            let te = &terms[e];
            if te.kind != K_PROJ {
                return None;
            }
            let (s, idx, x) = (te.x[0].f.lo as usize, te.x[1].e as u64, te.child[2]?);
            let p = &f.premises;
            // Premise j, a fact of this kind in the context.
            let fact = |j: usize, tag: u64| -> Option<&FactRec> {
                let q = facts.get(*p.get(j)?)?;
                (q.tag == tag && (tag == F_LINST || q.raw[0] == g)).then_some(q)
            };
            let level_of = |sort: u64| -> Option<&TermRec> {
                let sort = &terms[sort as usize];
                (sort.kind == K_LEAF && sort.x[0].f.lo == SUB_SORT).then(|| &terms[sort.child[1].unwrap()])
            };
            let zero = |l: &TermRec| l.kind == K_LLEAF && l.x[0].f.lo == 0;
            // `Infer(g, a, S)` and `Eq(g, S, sort zero)` at premises j, j + 1.
            let prop = |j: usize, a: usize| -> Option<()> {
                let (c1, c2) = (fact(j, F_INFER)?, fact(j + 1, F_EQ)?);
                (c1.raw[1] as usize == a && c2.raw[1] == c1.raw[2] && zero(level_of(c2.raw[2])?)).then_some(())
            };
            let (p0, p1, p2, p3, p4) =
                (fact(0, F_INFER)?, fact(1, F_EQ)?, fact(2, F_INFER)?, fact(3, F_EQ)?, fact(4, F_LINST)?);
            if p0.raw[1] as usize != x || p1.raw[1] != p0.raw[2] || p2.raw[1] != p0.raw[2] || p3.raw[1] != p2.raw[2] {
                return None;
            }
            let iapp = p1.raw[2] as usize;
            let lvl = level_of(p3.raw[2])?;
            let is_prop = zero(lvl);
            let lvl_ix = terms[p3.raw[2] as usize].child[1]?;
            let base = match (is_prop, lvl.kind) {
                (true, _) => "FTelNilP",
                (_, K_LSUCC) => "FTelNilS",
                (_, K_LMAX) => "FTelNilM",
                (_, K_LIMAX) => "FTelNilI",
                _ => return None,
            };
            // A maximum must be shown not to be zero in disguise.
            let nz = if matches!(base, "FTelNilM" | "FTelNilI") { Some(self.nz(lvl_ix)?) } else { None };
            let (h, n) = self.head(iapp);
            let th = &terms[h];
            if th.kind != K_LEAF || th.x[0].f.lo != SUB_CONST {
                return None;
            }
            let d = th.x[1].f.lo as usize;
            let info = by_key.get(&decls[d].key).copied().unwrap_or_default();
            if d != s || info.what != 1 || info.key == (0, 0) || n != info.a + info.b {
                return None;
            }
            let ctor = *decl_ix.get(&info.key)?;
            if decls[ctor].ty != Some(p4.raw[0] as usize) || Some(p4.raw[1] as usize) != th.child[2] {
                return None;
            }
            let (q, pre) = self.pre(iapp, info.b)?;
            let (_, _, spine) = self.spine(q);
            // q's prefixes, from the constant up.
            let mut pref = vec![q];
            while let Some((f0, _)) = self.app(*pref.last().unwrap()) {
                pref.push(f0);
            }
            pref.reverse();
            // `Eq(g, r, pi a body)` at premise j.
            let pi = |j: usize, r: u64| -> Option<(usize, usize)> {
                let q = fact(j, F_EQ)?;
                let w = &terms[q.raw[2] as usize];
                (q.raw[1] == r && w.kind == K_PI).then(|| (w.child[0].unwrap(), w.child[1].unwrap()))
            };
            let mut r = p4.raw[2];
            let mut row = self.add("PTelNil", F_PTEL, [g, h as u64, r, 0], vec![p[4]]);
            let mut j = 5;
            for &qk in &pref[1..] {
                let (_, arg) = self.app(qk)?;
                let (_, body) = pi(j, r)?;
                let q = facts.get(*p.get(j + 1)?)?;
                if q.tag != F_INST || q.raw[0] as usize != body || q.raw[1] != 0 || q.raw[2] as usize != arg {
                    return None;
                }
                r = q.raw[3];
                row = self.add("PTelApp", F_PTEL, [g, qk as u64, r, 0], vec![row, p[j], p[j + 1]]);
                j += 2;
            }
            let flag = is_prop as u64;
            // The row for `proj S k x` once that term is known: the base,
            // or the step from the projection before.
            let close = |cx: &mut Self, pt: usize, r: u64, pending: Option<(&'static str, Vec<usize>)>| -> usize {
                let (rule, pr) = pending.unwrap_or_else(|| {
                    let mut pr = vec![p[0], p[1], pre, spine, row];
                    if !is_prop {
                        pr.extend([p[2], p[3]]);
                        pr.extend(nz);
                    }
                    (base, pr)
                });
                cx.add(rule, F_FTEL, [g, pt as u64, r, flag], pr)
            };
            let mut pending = None;
            for k in 0..idx {
                let (a, body) = pi(j, r)?;
                let dep = is_prop && terms[body].lbr > 0;
                let ji = if dep { j + 3 } else { j + 1 };
                let q = facts.get(*p.get(ji)?)?;
                let pk = q.raw[2] as usize;
                let tp = &terms[pk];
                if q.tag != F_INST
                    || q.raw[0] as usize != body
                    || q.raw[1] != 0
                    || tp.kind != K_PROJ
                    || tp.x[0].f.lo as usize != s
                    || tp.x[1].e as u64 != k
                    || tp.child[2] != Some(x)
                {
                    return None;
                }
                if dep {
                    prop(j + 1, a)?;
                }
                let cur = close(self, pk, r, pending.take());
                let mut pr = vec![cur, p[j], p[ji]];
                let rule = if dep {
                    pr.extend([p[j + 1], p[j + 2]]);
                    "FTelStepP1"
                } else if is_prop {
                    "FTelStepP0"
                } else {
                    "FTelStep"
                };
                r = q.raw[3];
                pending = Some((rule, pr));
                j = ji + 1;
            }
            let (a, _) = pi(j, r)?;
            if a != t {
                return None;
            }
            if is_prop {
                prop(j + 1, a)?;
            }
            if p.len() != j + 1 + 2 * is_prop as usize {
                return None;
            }
            let cur = close(self, e, r, pending.take());
            let mut pr = vec![cur, p[j]];
            if is_prop {
                pr.extend([p[j + 1], p[j + 2]]);
            }
            Some((if is_prop { "ProjTyP" } else { "ProjTy" }, pr))
        }
    }
    let base = facts.len();
    let mut cx = Cx { terms, base, new: Vec::new(), memo: HashMap::new(), clash: 0 };
    let decl_ix: HashMap<(u64, u64), usize> = decls.iter().enumerate().map(|(i, d)| (d.key, i)).collect();
    let mut tel: Vec<(usize, &str, Vec<usize>)> = Vec::new();
    // The declaration a constant term names, with its information.
    let konst = |t: usize| -> Option<((u64, u64), InfoRec)> {
        let t = &terms[t];
        (t.kind == K_LEAF && t.x[0].f.lo == SUB_CONST).then(|| {
            let key = decls[t.x[1].f.lo as usize].key;
            (key, by_key.get(&key).copied().unwrap_or_default())
        })
    };
    for i in 0..base {
        let f = &facts[i];
        if f.rule == "Proj" && f.tag == F_INFER {
            if let Some((rule, premises)) = cx.proj_walk(facts, i, decls, &decl_ix, &by_key) {
                tel.push((i, rule, premises));
            }
            continue;
        }
        if f.tag != F_EQ {
            continue;
        }
        let (a, b) = (f.raw[1] as usize, f.raw[2] as usize);
        if f.rule == "ProjCtor" && terms[a].kind == K_PROJ {
            let ta = &terms[a];
            let (s, idx, x) = (ta.x[0].f.lo as usize, ta.x[1].e as u64, ta.child[2].unwrap());
            let (h, n) = cx.head(x);
            let Some((_, c)) = konst(h) else { continue };
            if c.what != 2 || c.key != decls[s].key || n != c.a + c.b || idx >= c.b {
                continue;
            }
            // x without the fields after idx ends in the field.
            let Some((p, pre)) = cx.pre(x, c.b - 1 - idx) else { continue };
            let Some((q, field)) = cx.app(p) else { continue };
            if field != b {
                continue;
            }
            let (_, _, spine) = cx.spine(q);
            facts[i].rule = "ProjCtorAt".into();
            facts[i].premises = vec![pre, spine];
            continue;
        }
        if f.rule == "Iota" && f.premises.len() <= 1 {
            let (h, n) = cx.head(a);
            let Some((rkey, r)) = konst(h) else { continue };
            if r.what != 3 || n < r.a + r.b + 1 {
                continue;
            }
            let e = n - r.a - r.b - 1;
            let Some((a1, p1)) = cx.pre(a, e) else { continue };
            let Some((a2, major)) = cx.app(a1) else { continue };
            let Some((a3, p2)) = cx.pre(a2, r.b) else { continue };
            let (_, _, p3) = cx.spine(a3);
            let (mh, mn, p4) = cx.spine(major);
            let Some((ckey, c)) = konst(mh) else { continue };
            let Some(&fields) = rule_of.get(&(rkey, ckey)) else { continue };
            if c.what != 2 || mn != c.a + c.b || fields != c.b {
                continue;
            }
            // b is the right-hand side at the arguments before the indices,
            // the fields, and the arguments after the major premise.
            let Some((b1, _)) = cx.pre(b, e) else { continue };
            let Some((b0, _)) = cx.pre(b1, fields) else { continue };
            let Some((rhs, _)) = cx.pre(b0, r.a) else { continue };
            let (Some(e1), Some(e2), Some(e3)) =
                (cx.ext(rhs, a3, b0, r.a), cx.ext(b0, major, b1, fields), cx.ext(b1, a, b, e))
            else {
                continue;
            };
            let with_levels = !facts[i].premises.is_empty();
            facts[i].rule = if with_levels { "IotaL" } else { "Iota0" }.into();
            facts[i].premises.extend([p1, p2, p3, p4, e1, e2, e3]);
        }
    }
    if cx.clash > 0 {
        eprintln!("{} projection walks met again with other premises", cx.clash);
    }
    for (i, rule, premises) in tel {
        facts[i].rule = rule.into();
        facts[i].premises = premises;
    }
    let new = std::mem::take(&mut cx.new);
    facts.extend(new);
    // The added rows and the derivation's read each other: until no rank moves.
    loop {
        let mut moved = false;
        for i in (base..facts.len()).chain(0..base) {
            let rank = facts[i].premises.iter().map(|&p| facts[p].rank + 1).max().unwrap_or(0);
            moved |= rank != facts[i].rank;
            facts[i].rank = rank;
        }
        if !moved {
            break;
        }
    }
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
    /// A term's bound on its free variables, `g^fv`.
    F,
    /// Whether a level parameter occurs in a term, level or list: zero
    /// when none does.
    P,
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
    /// A part of the declaration of the one constructor of the inductive
    /// type a constant term names.
    Ctor(TR, Part),
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
    /// Counted by a column the table sets itself (a rule's own live bit
    /// in a table of several rules), not by the number of requests.
    preset: bool,
}

/// One rule of a table of several: its kind of rows, where its inputs
/// go among the shared input columns, and its live bit.
#[derive(Clone, Debug)]
struct Group {
    of: Of,
    inputs: Vec<(usize, Src)>,
    live: usize,
}

#[derive(Clone, Debug)]
struct Spec {
    name: String,
    cols: Vec<(Col, Option<Src>)>,
    ports: Vec<Port>,
    /// A judgement table's owner column.
    owner: Option<usize>,
    /// The free-variable column of the term received last.
    last_f: usize,
    /// A table of terms: its own column that is zero when the row has no
    /// level parameter. A part received then has none either.
    taint: Option<usize>,
    /// The rules of a table of several rules (empty for a table of one).
    groups: Vec<Group>,
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
            last_f: 0,
            taint: None,
            groups: Vec::new(),
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
            preset: false,
        });
    }
    fn send<const N: usize>(&mut self, t: [T; N]) {
        self.ports.push(Port {
            send: true,
            t: pad(t),
            mult: Vec::new(),
            preset: false,
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
    /// The level-parameter entry of a part received: free, or in a table
    /// of terms the part's own value times the row's, so that a row marked
    /// free of level parameters receives only parts marked so.
    fn p_of(&mut self, tr: TR) -> T {
        let x = self.val(E::Tm(tr, Part::P));
        match self.taint {
            Some(own) => T::C(self.out(Col::Mul(x, own))),
            None => T::C(x),
        }
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
        self.term_p(id, tr, kind, x, l, None)
    }
    /// The same, with the level-parameter entry given.
    fn term_p(
        &mut self,
        id: usize,
        tr: TR,
        kind: T,
        x: [Option<T>; 3],
        l: Option<T>,
        p: Option<T>,
    ) -> (usize, usize) {
        let p = match p {
            Some(p) => p,
            None => self.p_of(tr.clone()),
        };
        let kind = match kind {
            T::Id => T::C(self.val(E::Tm(tr.clone(), Part::Kind))),
            t => t,
        };
        let x: Vec<T> = (0..3)
            .map(|i| x[i].unwrap_or_else(|| T::C(self.val(E::Tm(tr.clone(), Part::X(i))))))
            .collect();
        let lc = self.val(E::Tm(tr.clone(), Part::L));
        let l = l.unwrap_or(T::C(lc));
        let r = self.val(E::Tm(tr.clone(), Part::R));
        self.last_f = self.val(E::Tm(tr, Part::F));
        self.recv([k(TAG_TERM), T::C(id), kind, x[0], x[1], x[2], l, T::C(r), T::C(self.last_f), p]);
        (lc, r)
    }
    /// Receives the level `tr` with id column `id`; the parts given are
    /// shared, the others free. Returns the rank column.
    fn level(&mut self, id: usize, tr: TR, kind: Option<T>, a: Option<T>, b: Option<T>) -> usize {
        let kind = kind.unwrap_or_else(|| T::C(self.val(E::Tm(tr.clone(), Part::Kind))));
        let a = a.unwrap_or_else(|| T::C(self.val(E::Tm(tr.clone(), Part::X(0)))));
        let b = b.unwrap_or_else(|| T::C(self.val(E::Tm(tr.clone(), Part::X(1)))));
        let p = self.p_of(tr.clone());
        let r = self.val(E::Tm(tr, Part::R));
        self.recv([k(TAG_LVL), T::C(id), kind, a, b, T::C(r), p]);
        r
    }
    /// Receives the level list `tr` with id column `id`: its head, tail
    /// and `g^length`, shared where given.
    fn list(&mut self, id: usize, tr: TR, head: Option<T>, tail: Option<T>, n: Option<T>) {
        let head = head.unwrap_or_else(|| T::C(self.val(E::Tm(tr.clone(), Part::X(0)))));
        let tail = tail.unwrap_or_else(|| T::C(self.val(E::Tm(tr.clone(), Part::X(1)))));
        let p = self.p_of(tr.clone());
        let n = n.unwrap_or_else(|| T::C(self.val(E::Tm(tr, Part::X(2)))));
        self.recv([k(TAG_LIST), T::C(id), head, tail, n, p]);
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
    /// The recursors' rules.
    RecRule,
    Fact(&'static str),
    /// The judgements of the rules that are rare, in one table; the
    /// second such table holds the rules about application spines.
    Rare(usize),
    Assumed,
    /// A declaration's goal, with or without a value.
    Goal(bool),
}

impl Of {
    /// Whether the table belongs to the shared part of a cone (terms and
    /// declarations) rather than to a batch of judgements.
    fn shared(self) -> bool {
        matches!(
            self,
            Of::Term(_) | Of::DeclChecked | Of::DeclBlock | Of::DeclFree | Of::RecRule
        )
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
let k0 = s.src(Src::Pub(E::Own(Part::X(4))));
            let k1 = s.src(Src::Pub(E::Own(Part::X(5))));
            // What the reduction rules read, and the key, to whoever asks.
            let info: Vec<usize> = (6..10).map(|i| s.val(E::Own(Part::X(i)))).collect();
            s.send([k(TAG_INFO), T::Id, tc(info[0]), tc(info[3]), tc(info[1]), tc(info[2])]);
            s.send([k(TAG_KEY), T::Id, tc(k0), tc(k1)]);
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
        Of::RecRule => {
            // A recursor's rule for a constructor, both named by key (public).
            let mut s = Spec::new("rec_rule");
            let x: Vec<usize> = (0..4).map(|i| s.val(E::Own(Part::X(i)))).collect();
            let keys: Vec<usize> = (4..8).map(|i| s.src(Src::Pub(E::Own(Part::X(i))))).collect();
            s.recv([k(TAG_KEY), tc(x[0]), tc(keys[0]), tc(keys[1])]);
            s.recv([k(TAG_KEY), tc(x[1]), tc(keys[2]), tc(keys[3])]);
            s.send([k(TAG_RULE), tc(x[0]), tc(x[1]), tc(x[2]), tc(x[3])]);
            s
        }
        Of::Goal(with_value) => {
            let mut s = Spec::new(if with_value { "goal" } else { "goal_ax" });
            let own = |p| E::Own(p);
            let (d, lv, ty) = (
                s.val(own(Part::X(0))),
                s.val(own(Part::X(1))),
                s.val(own(Part::X(3))),
            );
            let o = s.val(own(Part::R));
            // The goal's judgements are in a context that is empty.
            let cx = s.val(own(Part::X(6)));
            s.recv([k(TAG_CTX), tc(cx), Z, Z, T::K(F128::ONE)]);
            // A premise in the empty context, owned by this goal.
            let fact = |s: &mut Spec, j: usize, tag: u64, x: [T; 4]| {
                let id = s.val(E::Prem(j, Part::Id));
                let r = s.val(E::Prem(j, Part::R));
                s.recv([
                    k(TAG_FACT),
                    tc(id),
                    k(tag),
                    x[0],
                    x[1],
                    x[2],
                    x[3],
                    tc(r),
                    tc(o),
                ]);
            };
            let sv = s.val(E::Prem(0, Part::X(2)));
            let su = s.val(E::Prem(1, Part::X(2)));
            fact(&mut s, 0, F_INFER, [tc(cx), tc(ty), tc(sv), Z]);
            fact(&mut s, 1, F_EQ, [tc(cx), tc(sv), tc(su), Z]);
            let u = s.val(E::Tm(TR::PFld(1, 2), Part::X(1)));
            s.term(
                su,
                TR::PFld(1, 2),
                k(K_LEAF),
                [Some(k(SUB_SORT)), Some(tc(u)), Some(Z)],
                None,
            );
            let v = if with_value {
                let v = s.val(own(Part::X(2)));
                let t = s.val(E::Prem(2, Part::X(2)));
                fact(&mut s, 2, F_INFER, [tc(cx), tc(v), tc(t), Z]);
                fact(&mut s, 3, F_EQ, [tc(cx), tc(t), tc(ty), Z]);
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
        Of::Rare(which) => rare_spec(which),
    }
}

/// The rules too rare for a table each (a table costs every leaf a slot):
/// their rows share one table.
const RARE: [&str; 38] = [
    "InstLet", "LInstLet", "LInstProj", "InstProj", "Zeta", "Let", "CongLam", "LInstSort", "CongSort", "LInstLZero",
    "LInstLSucc", "LInstLMax", "LInstLIMax", "LInstLParam0", "LInstLParam1", "LInstLParam2", "CongConst1",
    "CongConst2", "CongConst3", "LInstConst1", "LInstConst2", "LInstConst3", "LLeRefl", "LLeZero", "LLeSucc",
    "LLeSuccSucc", "LLeMaxR0", "LLeMaxR1", "LLeIMaxR", "LLeMaxL", "LLeIMaxL", "LEqLe", "PiZero", "PiIMax", "PiV",
    "PiMax", "PiU", "CongProj",
];

/// One table for the rare rules. A row is of one rule: it has that rule's
/// live bit set, and the rule's inputs in the input columns all rules
/// share. Every rule's ports are counted by its live bit, and where a
/// rule's own table would send its fact, it sends it to the row
/// (`TAG_UFACT`, once, on its live bit). The row receives that once and
/// sends the fact to whoever asks. The id in the tuple is the row's, so
/// the fact a row sends is one a rule live in that very row derived.
/// The rules that read an application spine, and the rows that spell a
/// spine out: one table, in leaves of their own like the rare rules'.
const SPINES: [&str; 8] =
    ["SpineNil", "SpineApp", "PreNil", "PreApp", "ExtNil", "ExtApp", "ProjCtorAt", "IotaL"];
/// The rules that type a projection, in a table of their own beside it.
const TELS: [&str; 18] = [
    "NZSucc", "NZPar0", "NZPar1", "NZPar2", "NZMaxL", "NZMaxR", "NZIMax",
    "PTelNil", "PTelApp", "FTelNilP", "FTelNilS", "FTelNilM", "FTelNilI", "FTelStep", "FTelStepP0", "FTelStepP1",
    "ProjTy", "ProjTyP",
];

fn rares(which: usize) -> &'static [&'static str] {
    match which {
        0 => &RARE,
        1 => &SPINES,
        _ => &TELS,
    }
}

/// The table of several rules a rule is in, and its place there.
fn rare_place(rule: &str) -> Option<(usize, usize)> {
    (0..3).find_map(|w| rares(w).iter().position(|&x| x == rule).map(|i| (w, i)))
}

fn rare_spec(which: usize) -> Spec {
    let rules = rares(which);
    let parts: Vec<Spec> = rules.iter().map(|r| fact_spec(r)).collect();
    let n_in = parts.iter().map(|sp| sp.cols.iter().filter(|c| matches!(c.0, Col::In)).count()).max().unwrap();
    let mut u = Spec::new(["rare", "spines", "tels"][which]);
    let own: Vec<T> = [Part::Kind, Part::X(0), Part::X(1), Part::X(2), Part::X(3), Part::R, Part::O]
        .into_iter()
        .map(|p| tc(u.val(E::Own(p))))
        .collect();
    let fact = |tag: u64| [k(tag), T::Id, own[0], own[1], own[2], own[3], own[4], own[5], own[6]];
    u.recv(fact(TAG_UFACT));
    u.send(fact(TAG_FACT));
    let first = u.cols.len();
    for _ in 0..n_in {
        u.cols.push((Col::In, None));
    }
    for (sp, &rule) in parts.iter().zip(rules) {
        let live = u.cols.len();
        u.cols.push((Col::In, None));
        let mut at = vec![0; sp.cols.len()];
        let mut inputs = Vec::new();
        for (y, (col, src)) in sp.cols.iter().enumerate() {
            at[y] = match *col {
                Col::In => {
                    inputs.push((first + inputs.len(), src.clone().expect("an input has a source")));
                    first + inputs.len() - 1
                }
                Col::Mul(a, b) => u.out(Col::Mul(at[a], at[b])),
                Col::MulC(c, a, b) => u.out(Col::MulC(c, at[a], at[b])),
                Col::Lin(c, a) => u.out(Col::Lin(c, at[a])),
            };
        }
        for p in &sp.ports {
            let mut t = p.t.map(|x| match x {
                T::C(c) => T::C(at[c]),
                x => x,
            });
            if p.send {
                assert!(matches!(t[0], T::K(v) if v == int(TAG_FACT)), "a rule sends its fact");
                t[0] = tc(u.out(Col::Lin(int(TAG_UFACT), live)));
            }
            u.ports.push(Port {
                send: p.send,
                t,
                mult: vec![live],
                preset: true,
            });
        }
        u.groups.push(Group {
            of: Of::Fact(rule),
            inputs,
            live,
        });
    }
    u
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
            s.send([k(TAG_TERM), T::Id, k(K_BVAR), tc(j), Z, Z, tc(l), tc(r), T::K(F128::ONE), Z]);
        }
        K_FVAR => {
            // Closed, and its bound on free variables is its level plus one.
            let n = s.val(own(Part::X(0)));
            let f = s.out(Col::Lin(G, n));
            let r = s.val(own(Part::R));
            s.send([k(TAG_TERM), T::Id, k(K_FVAR), tc(n), Z, Z, T::K(F128::ONE), tc(r), tc(f), Z]);
        }
        K_CTX0 => {
            s.send([k(TAG_CTX), T::Id, Z, Z, T::K(F128::ONE)]);
        }
        K_CTX => {
            // The parent's length plus one; the type added may mention only
            // the variables the parent has.
            let (p, a) = (s.val(own(Part::X(0))), s.val(own(Part::X(1))));
            let par = TR::Child(Box::new(TR::Own), 0);
            let (pp, pa, np) = (
                s.val(E::Tm(par.clone(), Part::X(0))),
                s.val(E::Tm(par.clone(), Part::X(1))),
                s.val(E::Tm(par.clone(), Part::X(2))),
            );
            s.recv([k(TAG_CTX), tc(p), tc(pp), tc(pa), tc(np)]);
            let ty = TR::Child(Box::new(TR::Own), 1);
            s.term(a, ty.clone(), T::Id, [None; 3], None);
            let fa = s.last_f;
            s.ge(np, E::Tm(par, Part::X(2)), fa, E::Tm(ty, Part::F), 0);
            let n = s.out(Col::Lin(G, np));
            s.send([k(TAG_CTX), T::Id, tc(p), tc(a), tc(n)]);
        }
        K_SORT | K_CONST | K_LIT | K_STR => {
            // A leaf is closed and has no variable. A sort's level and a
            // constant's level list are rows.
            let (x1, x2) = (s.val(own(Part::X(1))), s.val(own(Part::X(2))));
            let r = s.val(own(Part::R));
            let p = s.val(own(Part::P));
            s.taint = Some(p);
            let me = |c: usize| TR::Child(Box::new(TR::Own), c);
            let (sub, x2) = match kind {
                K_SORT => {
                    s.level(x1, me(1), None, None, None);
                    (SUB_SORT, Z)
                }
                K_CONST => {
                    s.list(x2, me(2), None, None, None);
                    (SUB_CONST, tc(x2))
                }
                K_LIT => (SUB_LIT, tc(x2)),
                _ => (SUB_STR, tc(x2)),
            };
            // A literal has no level.
            let p = if matches!(kind, K_SORT | K_CONST) { tc(p) } else { Z };
            let one = T::K(F128::ONE);
            s.send([k(TAG_TERM), T::Id, k(K_LEAF), k(sub), tc(x1), x2, one, tc(r), one, p]);
        }
        K_LLEAF => {
            // Zero (a = 0) or a parameter (a = its index plus one).
            let a = s.val(own(Part::X(0)));
            // Zero exactly when it is no parameter.
            s.send([k(TAG_LVL), T::Id, k(K_LLEAF), tc(a), Z, T::K(F128::ONE), tc(a)]);
        }
        K_LSUCC | K_LMAX | K_LIMAX => {
            let r = s.val(own(Part::R));
            let p = s.val(own(Part::P));
            s.taint = Some(p);
            let mut x = [Z; 2];
            for p in 0..if kind == K_LSUCC { 1 } else { 2 } {
                let id = s.val(own(Part::X(p)));
                let ch = TR::Child(Box::new(TR::Own), p);
                let rc = s.level(id, ch.clone(), None, None, None);
                s.ge(r, own(Part::R), rc, E::Tm(ch, Part::R), 1);
                x[p] = tc(id);
            }
            s.send([k(TAG_LVL), T::Id, k(kind), x[0], x[1], tc(r), tc(p)]);
        }
        K_NIL => {
            s.send([k(TAG_LIST), T::Id, Z, Z, T::K(F128::ONE)]);
        }
        K_CONS => {
            // A level before a list, one longer than it.
            let (h, t) = (s.val(own(Part::X(0))), s.val(own(Part::X(1))));
            let p = s.val(own(Part::P));
            s.taint = Some(p);
            let me = |c: usize| TR::Child(Box::new(TR::Own), c);
            s.level(h, me(0), None, None, None);
            let nt = s.val(E::Tm(me(1), Part::X(2)));
            s.list(t, me(1), None, None, Some(tc(nt)));
            let n = s.out(Col::Lin(G, nt));
            s.send([k(TAG_LIST), T::Id, tc(h), tc(t), tc(n), tc(p)]);
        }
        _ => {
            let (l, r) = (s.val(own(Part::L)), s.val(own(Part::R)));
            let fv = s.val(own(Part::F));
            let p = s.val(own(Part::P));
            s.taint = Some(p);
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
                let fc = s.last_f;
                if !ablate("LBR") {
                    s.ge(l, own(Part::L), lc, E::Tm(ch.clone(), Part::L), sh);
                }
                s.ge(fv, own(Part::F), fc, E::Tm(ch, Part::F), 0);
            }
            s.send([k(TAG_TERM), T::Id, k(kind), x[0], x[1], x[2], tc(l), tc(r), tc(fv), tc(p)]);
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
        s.send([
            k(TAG_FACT),
            T::Id,
            k(tag),
            x[0],
            x[1],
            x[2],
            x[3],
            tc(r),
            tc(o),
        ])
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
    // A is a proof's type: premises j and j + 1 are `Infer(g, A, S)` and
    // `Eq(g, S, sort zero)`.
    let proof = |s: &mut Spec, j: usize, g: usize, a: usize| {
        let (st, sz) = (s.val(E::Prem(j, Part::X(2))), s.val(E::Prem(j + 1, Part::X(2))));
        s.prem(j, F_INFER, [c(g), c(a), c(st), Some(Z)], r);
        s.prem(j + 1, F_EQ, [c(g), c(st), c(sz), Some(Z)], r);
        let z = s.val(E::Tm(TR::PFld(j + 1, 2), Part::X(1)));
        s.term(sz, TR::PFld(j + 1, 2), k(K_LEAF), [Some(k(SUB_SORT)), c(z), Some(Z)], None);
        s.level(z, TR::Child(Box::new(TR::PFld(j + 1, 2)), 1), Some(k(K_LLEAF)), Some(Z), Some(Z));
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
            // As many levels as the constant has parameters.
            if rule == "Delta0" {
                s.list(ls, ch(f(1), 2), Some(Z), Some(Z), Some(one));
                s.recv([k(TAG_DECL), tc(d), one, one, tc(b), tc(ty), tc(ord)]);
            } else {
                let (lv, v) = (
                    s.val(E::Dcl(f(1), Part::X(1))),
                    s.val(E::Dcl(f(1), Part::X(2))),
                );
                s.list(ls, ch(f(1), 2), None, None, Some(tc(lv)));
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
                s.list(ls, ch(f(1), 2), Some(Z), Some(Z), Some(one));
                s.recv([k(TAG_DECL), tc(d), tc(unf), one, tc(v), tc(t), tc(ord)]);
            } else {
                let lv = s.val(E::Dcl(f(1), Part::X(1)));
                let ty = s.val(E::Dcl(f(1), Part::X(3)));
                s.list(ls, ch(f(1), 2), None, None, Some(tc(lv)));
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
        // Level substitution through a node: the same node over the parts
        // substituted (the parts' own rows do the work on levels).
        "LInstApp" | "LInstLam" | "LInstPi" | "LInstLet" | "InstLet" => {
            let inst = rule == "InstLet";
            let (e, mid, a, res) = if inst {
                let (e, kk, a, res) = (
                    fld(&mut s, 0),
                    fld(&mut s, 1),
                    fld(&mut s, 2),
                    fld(&mut s, 3),
                );
                (e, kk, Some(a), res)
            } else {
                let (e, ls, res) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
                (e, ls, None, res)
            };
            let rf = if inst { 3 } else { 2 };
            let (kind, n) = match rule {
                "LInstApp" => (K_APP, 2),
                "LInstLam" => (K_LAM, 2),
                "LInstPi" => (K_PI, 2),
                _ => (K_LET, 3),
            };
            let p: Vec<usize> = (0..n)
                .map(|i| s.val(E::Tm(ch(f(0), i), Part::Id)))
                .collect();
            let q: Vec<usize> = (0..n)
                .map(|i| s.val(E::Tm(ch(f(rf), i), Part::Id)))
                .collect();
            let third = |v: &[usize]| if n == 3 { c(v[2]) } else { Some(Z) };
            s.term(e, f(0), k(kind), [c(p[0]), c(p[1]), third(&p)], None);
            s.term(res, f(rf), k(kind), [c(q[0]), c(q[1]), third(&q)], None);
            match a {
                Some(a) => {
                    // let A v b: the body is under one more binder.
                    closed(&mut s, a);
                    let k1 = s.out(Col::Lin(G, mid));
                    s.prem(0, F_INST, [c(p[0]), c(mid), c(a), c(q[0])], r);
                    s.prem(1, F_INST, [c(p[1]), c(mid), c(a), c(q[1])], r);
                    s.prem(2, F_INST, [c(p[2]), c(k1), c(a), c(q[2])], r);
                    send(&mut s, F_INST, [tc(e), tc(mid), tc(a), tc(res)]);
                }
                None => {
                    for i in 0..n {
                        s.prem(i, F_LINST, [c(p[i]), c(mid), c(q[i]), Some(Z)], r);
                    }
                    send(&mut s, F_LINST, [tc(e), tc(mid), tc(res), Z]);
                }
            }
        }
        "LInstProj" | "InstProj" => {
            let inst = rule == "InstProj";
            let rf = if inst { 3 } else { 2 };
            let (e, mid) = (fld(&mut s, 0), fld(&mut s, 1));
            let a = inst.then(|| fld(&mut s, 2));
            let res = fld(&mut s, rf);
            let (st, ix) = (
                s.val(E::Tm(f(0), Part::X(0))),
                s.val(E::Tm(f(0), Part::X(1))),
            );
            let (x, y) = (
                s.val(E::Tm(ch(f(0), 2), Part::Id)),
                s.val(E::Tm(ch(f(rf), 2), Part::Id)),
            );
            s.term(e, f(0), k(K_PROJ), [c(st), c(ix), c(x)], None);
            s.term(res, f(rf), k(K_PROJ), [c(st), c(ix), c(y)], None);
            match a {
                Some(a) => {
                    closed(&mut s, a);
                    s.prem(0, F_INST, [c(x), c(mid), c(a), c(y)], r);
                    send(&mut s, F_INST, [tc(e), tc(mid), tc(a), tc(res)]);
                }
                None => {
                    s.prem(0, F_LINST, [c(x), c(mid), c(y), Some(Z)], r);
                    send(&mut s, F_LINST, [tc(e), tc(mid), tc(res), Z]);
                }
            }
        }
        "CongProj" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (st, ix) = (
                s.val(E::Tm(f(1), Part::X(0))),
                s.val(E::Tm(f(1), Part::X(1))),
            );
            let (x, y) = (
                s.val(E::Tm(ch(f(1), 2), Part::Id)),
                s.val(E::Tm(ch(f(2), 2), Part::Id)),
            );
            s.term(a, f(1), k(K_PROJ), [c(st), c(ix), c(x)], None);
            s.term(b, f(2), k(K_PROJ), [c(st), c(ix), c(y)], None);
            s.prem(0, F_EQ, [c(g), c(x), c(y), Some(Z)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "FVar" => {
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let n = s.val(E::Tm(f(1), Part::X(0)));
            s.term(e, f(1), k(K_FVAR), [c(n), Some(Z), Some(Z)], None);
            s.prem(0, F_HAS, [c(g), c(n), c(t), Some(Z)], r);
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "Zeta" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (v, body) = (
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
                s.val(E::Tm(ch(f(1), 2), Part::Id)),
            );
            s.term(a, f(1), k(K_LET), [None, c(v), c(body)], None);
            s.prem(0, F_INST, [c(body), Some(one), c(v), c(b)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "HasLast" => {
            // The last entry: the context's length is n + 1.
            let (g, n, a) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let p = s.val(E::Tm(f(0), Part::X(0)));
            let len = s.out(Col::Lin(G, n));
            s.recv([k(TAG_CTX), tc(g), tc(p), tc(a), tc(len)]);
            send(&mut s, F_HAS, [tc(g), tc(n), tc(a), Z]);
        }
        "HasUp" => {
            let (g, n, a) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (p, last, len) = (
                s.val(E::Tm(f(0), Part::X(0))),
                s.val(E::Tm(f(0), Part::X(1))),
                s.val(E::Tm(f(0), Part::X(2))),
            );
            s.recv([k(TAG_CTX), tc(g), tc(p), tc(last), tc(len)]);
            s.prem(0, F_HAS, [c(p), c(n), c(a), Some(Z)], r);
            send(&mut s, F_HAS, [tc(g), tc(n), tc(a), Z]);
        }
        "Lam" => {
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (a, b) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
            );
            let cod = s.val(E::Tm(ch(f(2), 1), Part::Id));
            s.term(e, f(1), k(K_LAM), [c(a), c(b), Some(Z)], None);
            s.term(t, f(2), k(K_PI), [c(a), c(cod), Some(Z)], None);
            // The context's length n, the variable fvar n, and the context
            // with the binder's type added.
            let (gp_, ga, ng) = (
                s.val(E::Tm(f(0), Part::X(0))),
                s.val(E::Tm(f(0), Part::X(1))),
                s.val(E::Tm(f(0), Part::X(2))),
            );
            s.recv([k(TAG_CTX), tc(g), tc(gp_), tc(ga), tc(ng)]);
            let var = s.val(E::Prem(2, Part::X(2)));
            s.term(var, TR::PFld(2, 2), k(K_FVAR), [c(ng), Some(Z), Some(Z)], None);
            let ext = s.val(E::Prem(3, Part::X(0)));
            let ng1 = s.out(Col::Lin(G, ng));
            s.recv([k(TAG_CTX), tc(ext), tc(g), tc(a), tc(ng1)]);
            // The codomain does not mention the variable: fv(B) ≤ n.
            s.term(cod, ch(f(2), 1), T::Id, [None; 3], None);
            let fc = s.last_f;
            s.ge(ng, E::Tm(f(0), Part::X(2)), fc, E::Tm(ch(f(2), 1), Part::F), 0);
            let sa = s.val(E::Prem(0, Part::X(2)));
            let su = s.val(E::Prem(1, Part::X(2)));
            let u = s.val(E::Tm(TR::PFld(1, 2), Part::X(1)));
            s.term(su, TR::PFld(1, 2), k(K_LEAF), [Some(k(SUB_SORT)), c(u), Some(Z)], None);
            let bo = s.val(E::Prem(2, Part::X(3)));
            let bt = s.val(E::Prem(3, Part::X(2)));
            s.prem(0, F_INFER, [c(g), c(a), c(sa), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(sa), c(su), Some(Z)], r);
            s.prem(2, F_INST, [c(b), Some(one), c(var), c(bo)], r);
            s.prem(3, F_INFER, [c(ext), c(bo), c(bt), Some(Z)], r);
            s.prem(4, F_INST, [c(cod), Some(one), c(var), c(bt)], r);
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "CongLam" | "CongPi" => {
            let kind = if rule == "CongLam" { K_LAM } else { K_PI };
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (a1, b1) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
            );
            let (a2, b2) = (
                s.val(E::Tm(ch(f(2), 0), Part::Id)),
                s.val(E::Tm(ch(f(2), 1), Part::Id)),
            );
            s.term(a, f(1), k(kind), [c(a1), c(b1), Some(Z)], None);
            s.term(b, f(2), k(kind), [c(a2), c(b2), Some(Z)], None);
            let (gp_, ga, ng) = (
                s.val(E::Tm(f(0), Part::X(0))),
                s.val(E::Tm(f(0), Part::X(1))),
                s.val(E::Tm(f(0), Part::X(2))),
            );
            s.recv([k(TAG_CTX), tc(g), tc(gp_), tc(ga), tc(ng)]);
            let var = s.val(E::Prem(1, Part::X(2)));
            s.term(var, TR::PFld(1, 2), k(K_FVAR), [c(ng), Some(Z), Some(Z)], None);
            let ext = s.val(E::Prem(3, Part::X(0)));
            let ng1 = s.out(Col::Lin(G, ng));
            s.recv([k(TAG_CTX), tc(ext), tc(g), tc(a1), tc(ng1)]);
            // Neither body mentions the variable already.
            for (body, tr) in [(b1, ch(f(1), 1)), (b2, ch(f(2), 1))] {
                s.term(body, tr.clone(), T::Id, [None; 3], None);
                let fc = s.last_f;
                s.ge(ng, E::Tm(f(0), Part::X(2)), fc, E::Tm(tr, Part::F), 0);
            }
            let o2 = s.val(E::Prem(1, Part::X(3)));
            let o1 = s.val(E::Prem(2, Part::X(3)));
            s.prem(0, F_EQ, [c(g), c(a1), c(a2), Some(Z)], r);
            s.prem(1, F_INST, [c(b2), Some(one), c(var), c(o2)], r);
            s.prem(2, F_INST, [c(b1), Some(one), c(var), c(o1)], r);
            s.prem(3, F_EQ, [c(ext), c(o1), c(o2), Some(Z)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        "Let" => {
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (a, v, b) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
                s.val(E::Tm(ch(f(1), 2), Part::Id)),
            );
            s.term(e, f(1), k(K_LET), [c(a), c(v), c(b)], None);
            let sa = s.val(E::Prem(0, Part::X(2)));
            let su = s.val(E::Prem(1, Part::X(2)));
            let u = s.val(E::Tm(TR::PFld(1, 2), Part::X(1)));
            s.term(su, TR::PFld(1, 2), k(K_LEAF), [Some(k(SUB_SORT)), c(u), Some(Z)], None);
            let vt = s.val(E::Prem(2, Part::X(2)));
            let b2 = s.val(E::Prem(4, Part::X(3)));
            s.prem(0, F_INFER, [c(g), c(a), c(sa), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(sa), c(su), Some(Z)], r);
            s.prem(2, F_INFER, [c(g), c(v), c(vt), Some(Z)], r);
            s.prem(3, F_EQ, [c(g), c(vt), c(a), Some(Z)], r);
            s.prem(4, F_INST, [c(b), Some(one), c(v), c(b2)], r);
            s.prem(5, F_INFER, [c(g), c(b2), c(t), Some(Z)], r);
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "LLeRefl" | "LLeZero" | "LLeSucc" | "LLeSuccSucc" | "LLeMaxR0" | "LLeMaxR1" | "LLeIMaxR" | "LLeMaxL"
        | "LLeIMaxL" => {
            // l ≤ m for every assignment of the parameters.
            let (l, m) = (fld(&mut s, 0), fld(&mut s, 1));
            let le = |s: &mut Spec, j: usize, a: usize, b: usize| {
                s.prem(j, F_LLE, [c(a), c(b), Some(Z), Some(Z)], r);
            };
            let part = |s: &mut Spec, x: usize, i: usize| s.val(E::Tm(ch(f(x), i), Part::Id));
            match rule {
                "LLeRefl" => {}
                "LLeZero" => {
                    s.level(l, f(0), Some(k(K_LLEAF)), Some(Z), Some(Z));
                }
                // l ≤ m' gives l ≤ succ m'; l' ≤ m' gives succ l' ≤ succ m'.
                "LLeSucc" | "LLeSuccSucc" => {
                    let m1 = part(&mut s, 1, 0);
                    s.level(m, f(1), Some(k(K_LSUCC)), c(m1), Some(Z));
                    let l1 = if rule == "LLeSucc" {
                        l
                    } else {
                        let l1 = part(&mut s, 0, 0);
                        s.level(l, f(0), Some(k(K_LSUCC)), c(l1), Some(Z));
                        l1
                    };
                    le(&mut s, 0, l1, m1);
                }
                // At most one side of a maximum; at most the second of an
                // impredicative one (imax a b ≥ b).
                "LLeMaxR0" | "LLeMaxR1" | "LLeIMaxR" => {
                    let (a, b) = (part(&mut s, 1, 0), part(&mut s, 1, 1));
                    let kind = if rule == "LLeIMaxR" { K_LIMAX } else { K_LMAX };
                    s.level(m, f(1), Some(k(kind)), c(a), c(b));
                    le(&mut s, 0, l, if rule == "LLeMaxR0" { a } else { b });
                }
                // Both sides at most m (imax a b ≤ max a b).
                _ => {
                    let (a, b) = (part(&mut s, 0, 0), part(&mut s, 0, 1));
                    let kind = if rule == "LLeIMaxL" { K_LIMAX } else { K_LMAX };
                    s.level(l, f(0), Some(k(kind)), c(a), c(b));
                    le(&mut s, 0, a, m);
                    le(&mut s, 1, b, m);
                }
            }
            send(&mut s, F_LLE, [tc(l), tc(m), Z, Z]);
        }
        // Application spines. A count is an exponent, one more a step.
        "SpineNil" | "PreNil" => {
            let t = fld(&mut s, 0);
            send(&mut s, if rule == "SpineNil" { F_SPINE } else { F_PRE }, [tc(t), tc(t), one, Z]);
        }
        "SpineApp" | "PreApp" => {
            let tag = if rule == "SpineApp" { F_SPINE } else { F_PRE };
            let (t, h) = (fld(&mut s, 0), fld(&mut s, 1));
            let fun = s.val(E::Tm(ch(f(0), 0), Part::Id));
            s.term(t, f(0), k(K_APP), [c(fun), None, Some(Z)], None);
            let m = s.val(E::Prem(0, Part::X(2)));
            s.prem(0, tag, [c(fun), c(h), c(m), Some(Z)], r);
            let n = s.out(Col::Lin(G, m));
            send(&mut s, tag, [tc(t), tc(h), tc(n), Z]);
        }
        "ExtNil" => {
            let (u, v) = (fld(&mut s, 0), fld(&mut s, 1));
            send(&mut s, F_EXT, [tc(u), tc(v), tc(u), one]);
        }
        "ExtApp" => {
            let (u, v, w) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (v1, x, w1) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
                s.val(E::Tm(ch(f(2), 0), Part::Id)),
            );
            s.term(v, f(1), k(K_APP), [c(v1), c(x), Some(Z)], None);
            s.term(w, f(2), k(K_APP), [c(w1), c(x), Some(Z)], None);
            let m = s.val(E::Prem(0, Part::X(3)));
            s.prem(0, F_EXT, [c(u), c(v1), c(w1), c(m)], r);
            let n = s.out(Col::Lin(G, m));
            send(&mut s, F_EXT, [tc(u), tc(v), tc(w), tc(n)]);
        }
        // proj S i (c ps fs) = f_i: the term without the fields after i
        // ends in the field, and what is before it is the constructor c of
        // S at params + i arguments.
        "ProjCtorAt" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (st, ix) = (s.val(E::Tm(f(1), Part::X(0))), s.val(E::Tm(f(1), Part::X(1))));
            let x = s.val(E::Tm(ch(f(1), 2), Part::Id));
            s.term(a, f(1), k(K_PROJ), [c(st), c(ix), c(x)], None);
            let (p, gj) = (s.val(E::Prem(0, Part::X(1))), s.val(E::Prem(0, Part::X(2))));
            s.prem(0, F_PRE, [c(x), c(p), c(gj), Some(Z)], r);
            let q = s.val(E::Tm(ch(TR::PFld(0, 1), 0), Part::Id));
            s.term(p, TR::PFld(0, 1), k(K_APP), [c(q), c(b), Some(Z)], None);
            let h = s.val(E::Prem(1, Part::X(1)));
            let d = s.val(E::Tm(TR::PFld(1, 1), Part::X(1)));
            s.term(h, TR::PFld(1, 1), k(K_LEAF), [Some(k(SUB_CONST)), c(d), None], None);
            let params = s.val(E::Dcl(TR::PFld(1, 1), Part::X(7)));
            // fields = j + i + 1 and the count before the field is params + i.
            let fields = s.out(Col::MulC(G, gj, ix));
            s.recv([k(TAG_INFO), tc(d), k(2), tc(st), tc(params), tc(fields)]);
            let m = s.out(Col::Mul(params, ix));
            s.prem(1, F_SPINE, [c(q), c(h), c(m), Some(Z)], r);
            // The index is a number: the count is at least the parameters.
            s.ge(m, E::Prem(1, Part::X(2)), params, E::Dcl(TR::PFld(1, 1), Part::X(7)), 0);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        // rec ps ms mins idxs (c qs fs) es = rhs ps ms mins fs es, with
        // rhs the rule of rec for c at the levels.
        "IotaL" => {
            let (g, a, b) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let pf = |j: usize, x: usize| TR::PFld(j, x);
            // a without the arguments after the major premise ends in it.
            let (a1, ge_) = (s.val(E::Prem(1, Part::X(1))), s.val(E::Prem(1, Part::X(2))));
            s.prem(1, F_PRE, [c(a), c(a1), c(ge_), Some(Z)], r);
            let (a2, major) = (s.val(E::Tm(ch(pf(1, 1), 0), Part::Id)), s.val(E::Tm(ch(pf(1, 1), 1), Part::Id)));
            s.term(a1, pf(1, 1), k(K_APP), [c(a2), c(major), Some(Z)], None);
            // Before the indices: the recursor at its parameters, motives and minors.
            let (a3, gind) = (s.val(E::Prem(2, Part::X(1))), s.val(E::Prem(2, Part::X(2))));
            s.prem(2, F_PRE, [c(a2), c(a3), c(gind), Some(Z)], r);
            let (h, gbefore) = (s.val(E::Prem(3, Part::X(1))), s.val(E::Prem(3, Part::X(2))));
            s.prem(3, F_SPINE, [c(a3), c(h), c(gbefore), Some(Z)], r);
            let (rd, ls) = (s.val(E::Tm(pf(3, 1), Part::X(1))), s.val(E::Tm(pf(3, 1), Part::X(2))));
            s.term(h, pf(3, 1), k(K_LEAF), [Some(k(SUB_CONST)), c(rd), c(ls)], None);
            s.recv([k(TAG_INFO), tc(rd), k(3), Z, tc(gbefore), tc(gind)]);
            // As many levels as the recursor has parameters.
            let dcl: Vec<usize> = (0..4).map(|i| s.val(E::Dcl(pf(3, 1), Part::X(i)))).collect();
            let ord = s.val(E::Dcl(pf(3, 1), Part::R));
            s.list(ls, ch(pf(3, 1), 2), None, None, Some(tc(dcl[1])));
            s.recv([k(TAG_DECL), tc(rd), tc(dcl[0]), tc(dcl[1]), tc(dcl[2]), tc(dcl[3]), tc(ord)]);
            // The major premise: a constructor at its parameters and fields.
            let mh = s.val(E::Prem(4, Part::X(1)));
            let cd = s.val(E::Tm(pf(4, 1), Part::X(1)));
            s.term(mh, pf(4, 1), k(K_LEAF), [Some(k(SUB_CONST)), c(cd), None], None);
            let (ind, cp, gfields) = (
                s.val(E::Dcl(pf(4, 1), Part::X(9))),
                s.val(E::Dcl(pf(4, 1), Part::X(7))),
                s.val(E::Dcl(pf(4, 1), Part::X(8))),
            );
            s.recv([k(TAG_INFO), tc(cd), k(2), tc(ind), tc(cp), tc(gfields)]);
            let mn = s.out(Col::Mul(cp, gfields));
            s.prem(4, F_SPINE, [c(major), c(mh), c(mn), Some(Z)], r);
            // The rule, its right-hand side at the levels.
            let (rhs, rhs2) = (s.val(E::Prem(0, Part::X(0))), s.val(E::Prem(0, Part::X(2))));
            s.recv([k(TAG_RULE), tc(rd), tc(cd), tc(gfields), tc(rhs)]);
            s.prem(0, F_LINST, [c(rhs), c(ls), c(rhs2), Some(Z)], r);
            // b: the right-hand side at the arguments before the indices,
            // then the fields, then the arguments after the major premise.
            let (b0, b1) = (s.val(E::Prem(5, Part::X(2))), s.val(E::Prem(6, Part::X(2))));
            s.prem(5, F_EXT, [c(rhs2), c(a3), c(b0), c(gbefore)], r);
            s.prem(6, F_EXT, [c(b0), c(major), c(b1), c(gfields)], r);
            s.prem(7, F_EXT, [c(b1), c(a), c(b), c(ge_)], r);
            send(&mut s, F_EQ, [tc(g), tc(a), tc(b), Z]);
        }
        // The type of a projection, by a walk of the constructor's type.
        // It starts at the structure's constant: the type of its one
        // constructor at the levels.
        "PTelNil" => {
            let (g, q, rr) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (d, ls) = (s.val(E::Tm(f(1), Part::X(1))), s.val(E::Tm(f(1), Part::X(2))));
            s.term(q, f(1), k(K_LEAF), [Some(k(SUB_CONST)), c(d), c(ls)], None);
            let (cd, params, indices) = (
                s.val(E::Dcl(f(1), Part::X(9))),
                s.val(E::Dcl(f(1), Part::X(7))),
                s.val(E::Dcl(f(1), Part::X(8))),
            );
            s.recv([k(TAG_INFO), tc(d), k(1), tc(cd), tc(params), tc(indices)]);
            // The constructor's declaration, at as many levels as it has.
            let dcl: Vec<usize> = (0..4).map(|i| s.val(E::Ctor(f(1), Part::X(i)))).collect();
            let ord = s.val(E::Ctor(f(1), Part::R));
            s.list(ls, ch(f(1), 2), None, None, Some(tc(dcl[1])));
            s.recv([k(TAG_DECL), tc(cd), tc(dcl[0]), tc(dcl[1]), tc(dcl[2]), tc(dcl[3]), tc(ord)]);
            s.ge(o, E::Own(Part::O), ord, E::Ctor(f(1), Part::R), 1);
            s.prem(0, F_LINST, [c(dcl[3]), c(ls), c(rr), Some(Z)], r);
            send(&mut s, F_PTEL, [tc(g), tc(q), tc(rr), Z]);
        }
        // One more parameter: the type so far is a pi, its body at the argument.
        "PTelApp" => {
            let (g, q, rr) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (q1, a) = (s.val(E::Tm(ch(f(1), 0), Part::Id)), s.val(E::Tm(ch(f(1), 1), Part::Id)));
            s.term(q, f(1), k(K_APP), [c(q1), c(a), Some(Z)], None);
            let (r1, w) = (s.val(E::Prem(0, Part::X(2))), s.val(E::Prem(1, Part::X(2))));
            let body = s.val(E::Tm(ch(TR::PFld(1, 2), 1), Part::Id));
            s.term(w, TR::PFld(1, 2), k(K_PI), [None, c(body), Some(Z)], None);
            s.prem(0, F_PTEL, [c(g), c(q1), c(r1), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(r1), c(w), Some(Z)], r);
            s.prem(2, F_INST, [c(body), Some(one), c(a), c(rr)], r);
            send(&mut s, F_PTEL, [tc(g), tc(q), tc(rr), Z]);
        }
        // Before the first field: x's type is the structure at exactly its
        // parameters and indices, and the walk has passed the parameters.
        // Without the proposition checks the type's sort is not zero (a
        // successor, a maximum or an impredicative maximum).
        "FTelNilP" | "FTelNilS" | "FTelNilM" | "FTelNilI" => {
            let (g, pt, rr) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (st, x) = (s.val(E::Tm(f(1), Part::X(0))), s.val(E::Tm(ch(f(1), 2), Part::Id)));
            s.term(pt, f(1), k(K_PROJ), [c(st), Some(one), c(x)], None);
            let (e_, ty) = (s.val(E::Prem(0, Part::X(2))), s.val(E::Prem(1, Part::X(2))));
            s.prem(0, F_INFER, [c(g), c(x), c(e_), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(e_), c(ty), Some(Z)], r);
            let (q, gind) = (s.val(E::Prem(2, Part::X(1))), s.val(E::Prem(2, Part::X(2))));
            s.prem(2, F_PRE, [c(ty), c(q), c(gind), Some(Z)], r);
            let (h, gpar) = (s.val(E::Prem(3, Part::X(1))), s.val(E::Prem(3, Part::X(2))));
            s.prem(3, F_SPINE, [c(q), c(h), c(gpar), Some(Z)], r);
            s.term(h, TR::PFld(3, 1), k(K_LEAF), [Some(k(SUB_CONST)), c(st), None], None);
            let cd = s.val(E::Dcl(TR::PFld(3, 1), Part::X(9)));
            s.recv([k(TAG_INFO), tc(st), k(1), tc(cd), tc(gpar), tc(gind)]);
            s.prem(4, F_PTEL, [c(g), c(q), c(rr), Some(Z)], r);
            let flag = if rule == "FTelNilP" {
                k(1)
            } else {
                let (se, so) = (s.val(E::Prem(5, Part::X(2))), s.val(E::Prem(6, Part::X(2))));
                s.prem(5, F_INFER, [c(g), c(e_), c(se), Some(Z)], r);
                s.prem(6, F_EQ, [c(g), c(se), c(so), Some(Z)], r);
                let l = s.val(E::Tm(TR::PFld(6, 2), Part::X(1)));
                s.term(so, TR::PFld(6, 2), k(K_LEAF), [Some(k(SUB_SORT)), c(l), Some(Z)], None);
                let kind = match rule {
                    "FTelNilS" => K_LSUCC,
                    "FTelNilM" => K_LMAX,
                    _ => K_LIMAX,
                };
                s.level(l, ch(TR::PFld(6, 2), 1), Some(k(kind)), None, None);
                if rule != "FTelNilS" {
                    s.prem(7, F_NZ, [c(l), Some(Z), Some(Z), Some(Z)], r);
                }
                Z
            };
            send(&mut s, F_FTEL, [tc(g), tc(pt), tc(rr), flag]);
        }
        // One more field: the body at the projection before. Under the
        // proposition checks, the body does not mention the field (P0) or
        // the field's type is a proposition (P1).
        "FTelStep" | "FTelStepP0" | "FTelStepP1" => {
            let (g, pt, rr) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let flag = if rule == "FTelStep" { Z } else { k(1) };
            let (st, x) = (s.val(E::Tm(f(1), Part::X(0))), s.val(E::Tm(ch(f(1), 2), Part::Id)));
            let pp = s.val(E::Prem(0, Part::X(1)));
            let ix = s.val(E::Tm(TR::PFld(0, 1), Part::X(1)));
            s.term(pp, TR::PFld(0, 1), k(K_PROJ), [c(st), c(ix), c(x)], None);
            let ix1 = s.out(Col::Lin(G, ix));
            s.term(pt, f(1), k(K_PROJ), [c(st), c(ix1), c(x)], None);
            let (r1, w) = (s.val(E::Prem(0, Part::X(2))), s.val(E::Prem(1, Part::X(2))));
            let (a, body) =
                (s.val(E::Tm(ch(TR::PFld(1, 2), 0), Part::Id)), s.val(E::Tm(ch(TR::PFld(1, 2), 1), Part::Id)));
            s.term(w, TR::PFld(1, 2), k(K_PI), [c(a), c(body), Some(Z)], None);
            s.prem(0, F_FTEL, [c(g), c(pp), c(r1), Some(flag)], r);
            s.prem(1, F_EQ, [c(g), c(r1), c(w), Some(Z)], r);
            s.prem(2, F_INST, [c(body), Some(one), c(pp), c(rr)], r);
            match rule {
                "FTelStepP0" => {
                    s.term(body, ch(TR::PFld(1, 2), 1), T::Id, [None; 3], Some(one));
                }
                "FTelStepP1" => proof(&mut s, 3, g, a),
                _ => {}
            }
            send(&mut s, F_FTEL, [tc(g), tc(pt), tc(rr), flag]);
        }
        // The projection's type is the domain of what the walk has left.
        "ProjTy" | "ProjTyP" => {
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let flag = if rule == "ProjTy" { Z } else { k(1) };
            let (r1, w) = (s.val(E::Prem(0, Part::X(2))), s.val(E::Prem(1, Part::X(2))));
            s.term(w, TR::PFld(1, 2), k(K_PI), [c(t), None, Some(Z)], None);
            s.prem(0, F_FTEL, [c(g), c(e), c(r1), Some(flag)], r);
            s.prem(1, F_EQ, [c(g), c(r1), c(w), Some(Z)], r);
            if rule == "ProjTyP" {
                proof(&mut s, 2, g, t);
            }
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        // A level never read as zero: a successor, a parameter, a maximum
        // with such a side, an impredicative maximum with such a second side.
        "NZSucc" | "NZPar0" | "NZPar1" | "NZPar2" | "NZMaxL" | "NZMaxR" | "NZIMax" => {
            let l = fld(&mut s, 0);
            match rule {
                "NZSucc" => {
                    s.level(l, f(0), Some(k(K_LSUCC)), None, Some(Z));
                }
                "NZPar0" | "NZPar1" | "NZPar2" => {
                    let i = (rule.as_bytes()[rule.len() - 1] - b'0') as u64;
                    s.level(l, f(0), Some(k(K_LLEAF)), Some(k(i + 1)), Some(Z));
                }
                _ => {
                    let side = if rule == "NZMaxL" { 0 } else { 1 };
                    let x = s.val(E::Tm(ch(f(0), side), Part::Id));
                    let kind = if rule == "NZIMax" { K_LIMAX } else { K_LMAX };
                    let (a, b) = if side == 0 { (c(x), None) } else { (None, c(x)) };
                    s.level(l, f(0), Some(k(kind)), a, b);
                    s.prem(0, F_NZ, [c(x), Some(Z), Some(Z), Some(Z)], r);
                }
            }
            send(&mut s, F_NZ, [tc(l), Z, Z, Z]);
        }
        // A literal's type is the constant `Nat` (`String`) without levels:
        // the constant's key is the one the verifier names ([`leaf`] adds
        // that receive, as it adds a checked declaration's send).
        "LitNat" | "LitStr" => {
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let sub = if rule == "LitNat" { SUB_LIT } else { SUB_STR };
            s.term(e, f(1), k(K_LEAF), [Some(k(sub)), None, None], None);
            let (d, ls) = (s.val(E::Tm(f(2), Part::X(1))), s.val(E::Tm(f(2), Part::X(2))));
            s.term(t, f(2), k(K_LEAF), [Some(k(SUB_CONST)), c(d), c(ls)], None);
            s.list(ls, ch(f(2), 2), Some(Z), Some(Z), Some(one));
            let (k0, k1) = (s.val(E::Dcl(f(2), Part::X(4))), s.val(E::Dcl(f(2), Part::X(5))));
            s.recv([k(TAG_KEY), tc(d), tc(k0), tc(k1)]);
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "LEqLe" => {
            let (l, m) = (fld(&mut s, 0), fld(&mut s, 1));
            s.prem(0, F_LLE, [c(l), c(m), Some(Z), Some(Z)], r);
            s.prem(1, F_LLE, [c(m), c(l), Some(Z), Some(Z)], r);
            send(&mut s, F_LEQ, [tc(l), tc(m), Z, Z]);
        }
        "PiZero" | "PiIMax" | "PiV" | "PiMax" | "PiU" => {
            // pi A B : sort w, from A : sort u and, with a fresh variable
            // for the binder, B : sort v. w is imax u v, in the form the
            // rule's name says.
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (a, b) = (
                s.val(E::Tm(ch(f(1), 0), Part::Id)),
                s.val(E::Tm(ch(f(1), 1), Part::Id)),
            );
            s.term(e, f(1), k(K_PI), [c(a), c(b), Some(Z)], None);
            let (gp_, ga, ng) = (
                s.val(E::Tm(f(0), Part::X(0))),
                s.val(E::Tm(f(0), Part::X(1))),
                s.val(E::Tm(f(0), Part::X(2))),
            );
            s.recv([k(TAG_CTX), tc(g), tc(gp_), tc(ga), tc(ng)]);
            let var = s.val(E::Prem(2, Part::X(2)));
            s.term(var, TR::PFld(2, 2), k(K_FVAR), [c(ng), Some(Z), Some(Z)], None);
            let ext = s.val(E::Prem(3, Part::X(0)));
            let ng1 = s.out(Col::Lin(G, ng));
            s.recv([k(TAG_CTX), tc(ext), tc(g), tc(a), tc(ng1)]);
            // The body does not mention the variable already.
            s.term(b, ch(f(1), 1), T::Id, [None; 3], None);
            let fc = s.last_f;
            s.ge(ng, E::Tm(f(0), Part::X(2)), fc, E::Tm(ch(f(1), 1), Part::F), 0);
            let sa = s.val(E::Prem(0, Part::X(2)));
            let su = s.val(E::Prem(1, Part::X(2)));
            let u = s.val(E::Tm(TR::PFld(1, 2), Part::X(1)));
            s.term(su, TR::PFld(1, 2), k(K_LEAF), [Some(k(SUB_SORT)), c(u), Some(Z)], None);
            let bo = s.val(E::Prem(2, Part::X(3)));
            let sb = s.val(E::Prem(3, Part::X(2)));
            let sv = s.val(E::Prem(4, Part::X(2)));
            let v = s.val(E::Tm(TR::PFld(4, 2), Part::X(1)));
            s.term(sv, TR::PFld(4, 2), k(K_LEAF), [Some(k(SUB_SORT)), c(v), Some(Z)], None);
            s.prem(0, F_INFER, [c(g), c(a), c(sa), Some(Z)], r);
            s.prem(1, F_EQ, [c(g), c(sa), c(su), Some(Z)], r);
            s.prem(2, F_INST, [c(b), Some(one), c(var), c(bo)], r);
            s.prem(3, F_INFER, [c(ext), c(bo), c(sb), Some(Z)], r);
            s.prem(4, F_EQ, [c(ext), c(sb), c(sv), Some(Z)], r);
            // 1 ≤ v, as premise j: its left side is succ zero.
            let positive = |s: &mut Spec, j: usize| {
                let one_ = s.val(E::Prem(j, Part::X(0)));
                let zero = s.val(E::Tm(ch(TR::PFld(j, 0), 0), Part::Id));
                s.level(one_, TR::PFld(j, 0), Some(k(K_LSUCC)), c(zero), Some(Z));
                s.level(zero, ch(TR::PFld(j, 0), 0), Some(k(K_LLEAF)), Some(Z), Some(Z));
                s.prem(j, F_LLE, [c(one_), c(v), Some(Z), Some(Z)], r);
            };
            let w = match rule {
                "PiZero" => {
                    s.level(v, ch(TR::PFld(4, 2), 1), Some(k(K_LLEAF)), Some(Z), Some(Z));
                    v
                }
                "PiV" => {
                    s.prem(5, F_LLE, [c(u), c(v), Some(Z), Some(Z)], r);
                    v
                }
                "PiU" => {
                    s.prem(5, F_LLE, [c(v), c(u), Some(Z), Some(Z)], r);
                    positive(&mut s, 6);
                    u
                }
                _ => {
                    let w = s.val(E::Tm(f(2), Part::X(1)));
                    let kind = if rule == "PiMax" { K_LMAX } else { K_LIMAX };
                    s.level(w, ch(f(2), 1), Some(k(kind)), c(u), c(v));
                    if rule == "PiMax" {
                        positive(&mut s, 5);
                    }
                    w
                }
            };
            s.term(t, f(2), k(K_LEAF), [Some(k(SUB_SORT)), c(w), Some(Z)], None);
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "Sort" => {
            // sort l : sort (succ l).
            let (g, e, t) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (l, l2) = (s.val(E::Tm(f(1), Part::X(1))), s.val(E::Tm(f(2), Part::X(1))));
            s.term(e, f(1), k(K_LEAF), [Some(k(SUB_SORT)), c(l), Some(Z)], None);
            s.term(t, f(2), k(K_LEAF), [Some(k(SUB_SORT)), c(l2), Some(Z)], None);
            s.level(l2, ch(f(2), 1), Some(k(K_LSUCC)), c(l), Some(Z));
            send(&mut s, F_INFER, [tc(g), tc(e), tc(t), Z]);
        }
        "CongSort" | "LInstSort" => {
            // Sorts over levels that are equivalent, or one substituted.
            let cong = rule == "CongSort";
            let (x0, x1, x2) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (ia, ib) = if cong { (1, 2) } else { (0, 2) };
            let (a, b) = if cong { (x1, x2) } else { (x0, x2) };
            let (l, m) = (s.val(E::Tm(f(ia), Part::X(1))), s.val(E::Tm(f(ib), Part::X(1))));
            s.term(a, f(ia), k(K_LEAF), [Some(k(SUB_SORT)), c(l), Some(Z)], None);
            s.term(b, f(ib), k(K_LEAF), [Some(k(SUB_SORT)), c(m), Some(Z)], None);
            if cong {
                s.prem(0, F_LEQ, [c(l), c(m), Some(Z), Some(Z)], r);
                send(&mut s, F_EQ, [tc(x0), tc(a), tc(b), Z]);
            } else {
                s.prem(0, F_LINSTL, [c(l), c(x1), c(m), Some(Z)], r);
                send(&mut s, F_LINST, [tc(a), tc(x1), tc(b), Z]);
            }
        }
        "CongConst1" | "CongConst2" | "CongConst3" | "LInstConst1" | "LInstConst2" | "LInstConst3" => {
            // The same constant over level lists of this length, level by
            // level equivalent or substituted.
            let cong = rule.starts_with("Cong");
            let n = (rule.as_bytes()[rule.len() - 1] - b'0') as usize;
            let (x0, x1, x2) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (ia, ib) = if cong { (1, 2) } else { (0, 2) };
            let (a, b) = if cong { (x1, x2) } else { (x0, x2) };
            let d = s.val(E::Tm(f(ia), Part::X(1)));
            let (ms, ms2) = (s.val(E::Tm(f(ia), Part::X(2))), s.val(E::Tm(f(ib), Part::X(2))));
            s.term(a, f(ia), k(K_LEAF), [Some(k(SUB_CONST)), c(d), c(ms)], None);
            s.term(b, f(ib), k(K_LEAF), [Some(k(SUB_CONST)), c(d), c(ms2)], None);
            let (mut la, mut lb) = (ms, ms2);
            let (mut ta, mut tb) = (ch(f(ia), 2), ch(f(ib), 2));
            for j in 0..n {
                // The last cell has length one: both lists end here.
                let len = (j + 1 == n).then_some(T::K(G));
                let cell = |s: &mut Spec, id: usize, tr: &TR| {
                    let h = s.val(E::Tm(ch(tr.clone(), 0), Part::Id));
                    let t = s.val(E::Tm(ch(tr.clone(), 1), Part::Id));
                    s.list(id, tr.clone(), c(h), c(t), len);
                    (h, t)
                };
                let (h, t) = cell(&mut s, la, &ta);
                let (h2, t2) = cell(&mut s, lb, &tb);
                if cong {
                    s.prem(j, F_LEQ, [c(h), c(h2), Some(Z), Some(Z)], r);
                } else {
                    s.prem(j, F_LINSTL, [c(h), c(x1), c(h2), Some(Z)], r);
                }
                (la, lb) = (t, t2);
                (ta, tb) = (ch(ta, 1), ch(tb, 1));
            }
            if cong {
                send(&mut s, F_EQ, [tc(x0), tc(a), tc(b), Z]);
            } else {
                send(&mut s, F_LINST, [tc(a), tc(x1), tc(b), Z]);
            }
        }
        "LInstNone" => {
            // No level parameter occurs: nothing changes.
            let (e, ls) = (fld(&mut s, 0), fld(&mut s, 1));
            s.term_p(e, f(0), T::Id, [None; 3], None, Some(Z));
            send(&mut s, F_LINST, [tc(e), tc(ls), tc(e), Z]);
        }
        "LInstLZero" => {
            let (l, ls) = (fld(&mut s, 0), fld(&mut s, 1));
            s.level(l, f(0), Some(k(K_LLEAF)), Some(Z), Some(Z));
            send(&mut s, F_LINSTL, [tc(l), tc(ls), tc(l), Z]);
        }
        "LInstLSucc" | "LInstLMax" | "LInstLIMax" => {
            // The same level over the parts substituted.
            let (l, ls, res) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            let (kind, n) = match rule {
                "LInstLSucc" => (K_LSUCC, 1),
                "LInstLMax" => (K_LMAX, 2),
                _ => (K_LIMAX, 2),
            };
            let p: Vec<usize> = (0..n).map(|i| s.val(E::Tm(ch(f(0), i), Part::Id))).collect();
            let q: Vec<usize> = (0..n).map(|i| s.val(E::Tm(ch(f(2), i), Part::Id))).collect();
            let second = |v: &[usize]| if n == 2 { c(v[1]) } else { Some(Z) };
            s.level(l, f(0), Some(k(kind)), c(p[0]), second(&p));
            s.level(res, f(2), Some(k(kind)), c(q[0]), second(&q));
            for i in 0..n {
                s.prem(i, F_LINSTL, [c(p[i]), c(ls), c(q[i]), Some(Z)], r);
            }
            send(&mut s, F_LINSTL, [tc(l), tc(ls), tc(res), Z]);
        }
        "LInstLParam0" | "LInstLParam1" | "LInstLParam2" => {
            // Parameter i gives the list's entry i.
            let i = (rule.as_bytes()[rule.len() - 1] - b'0') as usize;
            let (l, ls, res) = (fld(&mut s, 0), fld(&mut s, 1), fld(&mut s, 2));
            s.level(l, f(0), Some(k(K_LLEAF)), Some(k(i as u64 + 1)), Some(Z));
            let (mut id, mut tr) = (ls, f(1));
            for _ in 0..i {
                let t = s.val(E::Tm(ch(tr.clone(), 1), Part::Id));
                s.list(id, tr.clone(), None, c(t), None);
                (id, tr) = (t, ch(tr, 1));
            }
            s.list(id, tr, c(res), None, None);
            send(&mut s, F_LINSTL, [tc(l), tc(ls), tc(res), Z]);
        }
        r => panic!("no table for {r}"),
    }
    s
}

impl Rows {
    /// Judgement rows per rule, and whether the circuit checks that rule
    /// (`false`: the row's fact goes to the assumed table).
    pub fn rule_census(&self, g: &Rows) -> HashMap<String, (usize, bool)> {
        let mut m: HashMap<String, (usize, bool)> = HashMap::new();
        for i in 0..self.facts.len() {
            let checked = table_of(g, self, i) != Of::Assumed;
            let e = m.entry(self.facts[i].rule.clone()).or_insert((0, checked));
            e.0 += 1;
        }
        m
    }

    /// Counts of cited declarations: (checked, block members, free).
    pub fn decl_census(&self) -> usize {
        self.decls.len()
    }
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
        "LInstApp" => "LInstApp",
        "LInstBind" => {
            if g.terms[f.raw[0] as usize].kind == K_LAM {
                "LInstLam"
            } else {
                "LInstPi"
            }
        }
        "LInstLet" => "LInstLet",
        "InstLet" => "InstLet",
        "LInstProj" => "LInstProj",
        "InstProj" => "InstProj",
        "CongProj" => "CongProj",
        "FVar" => "FVar",
        "Zeta" => "Zeta",
        "HasLast" => "HasLast",
        "HasUp" => "HasUp",
        "Lam" => "Lam",
        "CongLam" => "CongLam",
        "CongPi" => "CongPi",
        "Let" => "Let",
        "Sort" => "Sort",
        "LLeRefl" => "LLeRefl",
        "LLeZero" => "LLeZero",
        "LLeSucc" => "LLeSucc",
        "LLeSuccSucc" => "LLeSuccSucc",
        "LLeMaxR0" => "LLeMaxR0",
        "LLeMaxR1" => "LLeMaxR1",
        "LLeIMaxR" => "LLeIMaxR",
        "LLeMaxL" => "LLeMaxL",
        "LLeIMaxL" => "LLeIMaxL",
        "LEqLe" => "LEqLe",
        "SpineNil" => "SpineNil",
        "SpineApp" => "SpineApp",
        "PreNil" => "PreNil",
        "PreApp" => "PreApp",
        "ExtNil" => "ExtNil",
        "ExtApp" => "ExtApp",
        "ProjCtorAt" => "ProjCtorAt",
        "IotaL" => "IotaL",
        "PTelNil" => "PTelNil",
        "LitNat" => "LitNat",
        "LitStr" => "LitStr",
        "NZSucc" => "NZSucc",
        "NZPar0" => "NZPar0",
        "NZPar1" => "NZPar1",
        "NZPar2" => "NZPar2",
        "NZMaxL" => "NZMaxL",
        "NZMaxR" => "NZMaxR",
        "NZIMax" => "NZIMax",
        "PTelApp" => "PTelApp",
        "FTelNilP" => "FTelNilP",
        "FTelNilS" => "FTelNilS",
        "FTelNilM" => "FTelNilM",
        "FTelNilI" => "FTelNilI",
        "FTelStep" => "FTelStep",
        "FTelStepP0" => "FTelStepP0",
        "FTelStepP1" => "FTelStepP1",
        "ProjTy" => "ProjTy",
        "ProjTyP" => "ProjTyP",
        "PiZero" => "PiZero",
        "PiIMax" => "PiIMax",
        "PiV" => "PiV",
        "PiMax" => "PiMax",
        "PiU" => "PiU",
        "LInstNone" => "LInstNone",
        "CongSort" => "CongSort",
        "LInstSort" => "LInstSort",
        "LInstLZero" => "LInstLZero",
        "LInstLSucc" => "LInstLSucc",
        "LInstLMax" => "LInstLMax",
        "LInstLIMax" => "LInstLIMax",
        "LInstLParam" => match g.terms[f.raw[0] as usize].x[0].f.lo {
            1 => "LInstLParam0",
            2 => "LInstLParam1",
            3 => "LInstLParam2",
            _ => return Of::Assumed,
        },
        "CongConst" => match f.premises.len() {
            1 => "CongConst1",
            2 => "CongConst2",
            3 => "CongConst3",
            _ => return Of::Assumed,
        },
        "LInstConst" => match f.premises.len() {
            1 => "LInstConst1",
            2 => "LInstConst2",
            3 => "LInstConst3",
            _ => return Of::Assumed,
        },
        _ => return Of::Assumed,
    };
    Of::Fact(r)
}

const FACT_TABLES: [&str; 92] = [
    "LitNat",
    "LitStr",
    "NZSucc",
    "NZPar0",
    "NZPar1",
    "NZPar2",
    "NZMaxL",
    "NZMaxR",
    "NZIMax",
    "PTelNil",
    "PTelApp",
    "FTelNilP",
    "FTelNilS",
    "FTelNilM",
    "FTelNilI",
    "FTelStep",
    "FTelStepP0",
    "FTelStepP1",
    "ProjTy",
    "ProjTyP",
    "SpineNil",
    "SpineApp",
    "PreNil",
    "PreApp",
    "ExtNil",
    "ExtApp",
    "ProjCtorAt",
    "IotaL",
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
    "LInstApp",
    "LInstLam",
    "LInstPi",
    "LInstLet",
    "InstLet",
    "LInstProj",
    "InstProj",
    "CongProj",
    "FVar",
    "Zeta",
    "HasLast",
    "HasUp",
    "Lam",
    "CongLam",
    "CongPi",
    "Let",
    "Sort",
    "LInstNone",
    "CongSort",
    "LInstSort",
    "LInstLZero",
    "LInstLSucc",
    "LInstLMax",
    "LInstLIMax",
    "LInstLParam0",
    "LInstLParam1",
    "LInstLParam2",
    "CongConst1",
    "CongConst2",
    "CongConst3",
    "LInstConst1",
    "LInstConst2",
    "LInstConst3",
    "LLeRefl",
    "LLeZero",
    "LLeSucc",
    "LLeSuccSucc",
    "LLeMaxR0",
    "LLeMaxR1",
    "LLeIMaxR",
    "LLeMaxL",
    "LLeIMaxL",
    "LEqLe",
    "PiZero",
    "PiIMax",
    "PiV",
    "PiMax",
    "PiU",
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

/// Fixed multiplicity widths per table (see [`set_mult_bits`]).
static MULT_BITS: std::sync::Mutex<Option<Vec<usize>>> = std::sync::Mutex::new(None);

/// Fix each table's multiplicity bit width, for circuits built after this
/// call (`None`: each side takes the width its own counts need). Sides of
/// one cone built with the cone's widths share one leaf circuit.
pub fn set_mult_bits(bits: Option<Vec<usize>>) {
    *MULT_BITS.lock().unwrap() = bits;
}

impl LeanCircuit {
    /// Per table, the widest multiplicity of its send ports (0: none).
    pub fn mult_bits(&self) -> Vec<usize> {
        self.specs
            .iter()
            .map(|s| {
                s.ports
                    .iter()
                    .filter(|p| p.send)
                    .map(|p| p.mult.len())
                    .max()
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Every bus tuple of chunk `ch` of table `s`, natively, as a leaf puts
    /// it on the bus: `f(table, send, tuple, count)` per row and port (a
    /// receive counts once per real row, a send its count bits). For
    /// balance checks without proving.
    pub fn for_each_tuple(&self, s: usize, ch: usize, mut f: impl FnMut(usize, bool, &[F128], u64)) {
        let nu = self.nu;
        let sp = &self.specs[s];
        let cols: Vec<Col> = sp.cols.iter().map(|c| c.0).collect();
        let t = &self.tables[s];
        let rows = &t[(ch << nu).min(t.len())..((ch + 1) << nu).min(t.len())];
        let word = ((self.chunk_base[s] + ch) as u64) << 32;
        let mut tup = Vec::with_capacity(W);
        for (j, z0) in rows.iter().enumerate() {
            let mut z = z0.clone();
            Gate::complete(&cols, &mut z);
            for p in &sp.ports {
                tup.clear();
                for term in &p.t {
                    tup.push(match *term {
                        T::C(c) => z[c],
                        T::K(v) => v,
                        T::Id => F128::new(word | j as u64, 0),
                    });
                }
                let n = if p.mult.is_empty() {
                    1
                } else {
                    p.mult.iter().enumerate().fold(0u64, |a, (i, &c)| {
                        a | (u64::from(z[c] == F128::ONE) << i)
                    })
                };
                f(s, p.send, &tup, n);
            }
        }
    }

    /// Each table spec, as text, to compare two sides' leaf circuits.
    pub fn spec_texts(&self) -> Vec<String> {
        self.specs.iter().map(|s| format!("{s:?}")).collect()
    }

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
        for kd in TERM_TABLES {
            ofs.push(Of::Term(kd));
        }
        ofs.extend([Of::DeclChecked, Of::DeclBlock, Of::DeclFree, Of::RecRule]);
        for r in FACT_TABLES {
            if r == "Assumed" {
                ofs.extend([Of::Rare(0), Of::Rare(1), Of::Rare(2), Of::Assumed]);
            } else if rare_place(r).is_none() {
                ofs.push(Of::Fact(r));
            }
        }
        ofs.extend([Of::Goal(true), Of::Goal(false)]);
        let rare_of = |of: Of| match of {
            Of::Fact(r) => rare_place(r),
            _ => None,
        };
        let at = |of: Of| {
            let of = rare_of(of).map_or(of, |(w, _)| Of::Rare(w));
            ofs.iter().position(|&o| o == of).unwrap()
        };
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); ofs.len()];
        // In the table of rare rules, each row's rule.
        let mut rule_of: Vec<Vec<usize>> = vec![Vec::new(); ofs.len()];
        let mut term_at = vec![(0usize, 0usize); g.terms.len()];
        for (i, t) in g.terms.iter().enumerate() {
            let s = at(Of::Term(table_kind(t)));
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
            let of = table_of(g, b, i);
            let s = at(of);
            fact_at[i] = (s, members[s].len());
            members[s].push(i);
            if let Some((_, r)) = rare_of(of) {
                rule_of[s].push(r);
            }
        }
        for (i, goal) in b.goals.iter().enumerate() {
            members[at(Of::Goal(goal.value.is_some()))].push(i);
        }
        members[at(Of::RecRule)] = (0..g.rules.len()).collect();
        // Range limbs wide enough for every rank, bound and order: 2^(2L) > span.
        let limb = g.limb.unwrap_or_else(|| {
            let span = b
                .facts
                .iter()
                .map(|f| f.rank.max(f.owner))
                .chain(g.terms.iter().map(|t| (t.rank + t.lbr).max(t.fv + 1).max(t.x[2].e + 1)))
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
            } else if (f.tag == F_INST || f.tag == F_HAS) && x == 1 {
                ve(f.raw[1] as i64)
            } else if (matches!(f.tag, F_SPINE | F_PRE) && x == 2) || (f.tag == F_EXT && x == 3) {
                ve(f.raw[x] as i64)
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
                Part::F => ve(t.fv),
                // A level leaf says it with its own field.
                Part::P if t.kind == K_LLEAF => t.x[0],
                Part::P => vi(t.lp as u64),
                Part::O => unreachable!(),
            }
        };
        let decl_by_key: HashMap<(u64, u64), usize> = g.decls.iter().enumerate().map(|(i, d)| (d.key, i)).collect();
        let rule_part = |i: usize, p: Part| -> V {
            let r = &g.rules[i];
            match p {
                Part::X(0) => V { f: decl_id(r.rec), e: 0 },
                Part::X(1) => V { f: decl_id(decl_by_key[&r.ctor]), e: 0 },
                Part::X(2) => ve(r.fields as i64),
                Part::X(3) => V { f: term_id(r.rhs), e: 0 },
                Part::X(4) => vi(g.decls[r.rec].key.0),
                Part::X(5) => vi(g.decls[r.rec].key.1),
                Part::X(6) => vi(r.ctor.0),
                Part::X(7) => vi(r.ctor.1),
                _ => unreachable!(),
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
                Part::X(1) => ve(d.levels as i64),
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
                Part::X(6) => vi(d.info.what),
                Part::X(7) => ve(d.info.a as i64),
                Part::X(8) => ve(d.info.b as i64),
                // A constructor's inductive type; the one constructor
                // of an inductive type that has one.
                Part::X(9) => match (d.info.what, decl_by_key.get(&d.info.key)) {
                    (2, Some(&t)) => V { f: decl_id(t), e: 0 },
                    (1, Some(&t)) if d.info.key != (0, 0) => V { f: decl_id(t), e: 0 },
                    _ => V::default(),
                },
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
                    TR::Child(t, c) => {
                        g.terms[go(t, own, of, g, b, prem_of)].child[*c].expect("a child")
                    }
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
                Part::L | Part::F | Part::P => unreachable!(),
            }
        };
        let eval = |e: &E, of: Of, own: usize| -> V {
            match e {
                E::Own(p) => match of {
                    Of::Term(_) => term_part(own, *p),
                    Of::DeclChecked | Of::DeclBlock | Of::DeclFree => decl_part(own, *p),
                    Of::RecRule => rule_part(own, *p),
                    Of::Goal(_) => {
                        let d = b.goals[own].decl;
                        match p {
                            Part::X(6) => term_part(b.goals[own].ctx, Part::Id),
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
                E::Ctor(tr, p) => {
                    let t = &g.terms[resolve(tr, own, of)];
                    decl_part(decl_by_key[&g.decls[t.x[1].f.lo as usize].info.key], *p)
                }
            }
        };
        let h = gp(1 << limb);
        let lmask = (1i64 << limb) - 1;
        // Rows, without count bits.
        let mut specs: Vec<Spec> = ofs.iter().map(|&o| spec(o)).collect();
        let mut tables: Vec<Vec<Vec<F128>>> = Vec::new();
        // An input's value for the row `own` of kind `of`.
        let value = |src: &Src, of: &Of, own: usize, name: &str| -> Result<F128, String> {
                    Ok(match src {
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
                                return Err(format!("{name}: {} ≥ {} + {cc} fails", x.e, y.e));
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
                    })
        };
        for (s, of) in ofs.iter().enumerate() {
            let sp = &specs[s];
            let cols: Vec<Col> = sp.cols.iter().map(|c| c.0).collect();
            let mut out = Vec::new();
            for (j, &own) in members[s].iter().enumerate().filter(|_| mine(s)) {
                let mut z = vec![F128::ZERO; cols.len()];
                // In a table of several rules the row is of its rule's kind.
                let group = (!sp.groups.is_empty()).then(|| &sp.groups[rule_of[s][j]]);
                let of = group.map_or(of, |g| &g.of);
                for (y, (_, src)) in sp.cols.iter().enumerate() {
                    let Some(src) = src else { continue };
                    z[y] = value(src, of, own, &sp.name)?;
                }
                if let Some(g) = group {
                    z[g.live] = F128::ONE;
                    for (y, src) in &g.inputs {
                        z[*y] = value(src, of, own, &sp.name)?;
                    }
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
                    if p.preset && z[p.mult[0]] != F128::ONE {
                        continue;
                    }
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
        // A rule's fact goes to its own row, once.
        for (s, sp) in specs.iter().enumerate() {
            for p in sp.ports.iter().filter(|p| p.send && p.preset) {
                for (j, z) in tables[s].iter().enumerate() {
                    if z[p.mult[0]] != F128::ONE {
                        continue;
                    }
                    let t = tuple(&p.t, z, s, j);
                    match asked.get_mut(&t) {
                        Some(n) if *n > 1 => *n -= 1,
                        Some(_) => {
                            asked.remove(&t);
                        }
                        None => {
                            return Err(format!(
                                "{}: row {j} of rule {:?} does not derive its fact",
                                sp.name,
                                sp.groups[rule_of[s][j]].of
                            ));
                        }
                    }
                }
            }
        }
        // Count bits for each send port.
        for s in 0..specs.len() {
            let sends: Vec<usize> = (0..specs[s].ports.len())
                .filter(|&p| specs[s].ports[p].send && !specs[s].ports[p].preset)
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
                let need = (64 - counts.iter().max().copied().unwrap_or(0).leading_zeros()).max(1)
                    as usize;
                // A fixed width per table keeps every side's leaf one
                // circuit (the recursion tower folds one leaf circuit).
                let bits = match MULT_BITS.lock().unwrap().as_ref() {
                    Some(w) => {
                        let b = w.get(s).copied().unwrap_or(need);
                        if need > b {
                            return Err(format!(
                                "table {s} needs {need} multiplicity bits, the fixed width is {b}"
                            ));
                        }
                        b
                    }
                    None => need,
                };
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
                    if k == 0 {
                        sb.fixed_public_input(F128::ZERO)
                    } else {
                        sb.input()
                    }
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
        r.build_ms = self.build_ms + t.elapsed().as_secs_f64() * 1e3
            - r.witness_ms
            - r.prove_ms
            - r.verify_ms;
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
pub type LeafSlot<'a> = Option<SlotRows<'a>>;

/// The rows of one chunk and its chunk number, from whichever side holds
/// it (or from a copy kept elsewhere).
#[derive(Clone, Copy)]
pub struct SlotRows<'a> {
    pub rows: &'a [Vec<F128>],
    pub chunk: u64,
    pub nu: usize,
}

impl LeanCircuit {
    /// Chunk `ch` of table `s`, for a leaf slot.
    pub fn slot(&self, s: usize, ch: usize) -> SlotRows<'_> {
        let t = &self.tables[s];
        SlotRows {
            rows: &t[ch << self.nu..((ch + 1) << self.nu).min(t.len())],
            chunk: (self.chunk_base[s] + ch) as u64,
            nu: self.nu,
        }
    }
}

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
pub fn leaf(
    proto: &LeanCircuit,
    profile: &Profile,
    slots: &[LeafSlot<'_>],
    index: usize,
) -> Result<Piece, String> {
    let nu = profile.nu;
    if slots.len() != profile.slots.len() {
        return Err("one entry per profile slot".into());
    }
    let mut sb = ShapeBuilder::new(nu);
    let mut values = Vec::new();
    let mut out_slots = Vec::new();
    let mut word_ws: Vec<Wire> = Vec::new();
    let gate = crate::leanxmss::gadgets::swap_gate(nu).map_err(|e| format!("{e:?}"))?;
    let bslot = sb.slot(gate.clone());
    let w: Vec<Wire> = (0..3)
        .map(|k| {
            values.push(F128::ZERO);
            if k == 0 {
                sb.public_input()
            } else {
                sb.input()
            }
        })
        .collect();
    sb.gate(bslot, &w);
    for (k, &s) in profile.slots.iter().enumerate() {
        let sp = &proto.specs[s];
        let (rows, chunk_no): (&[Vec<F128>], u64) = match slots[k] {
            Some(c) => {
                if c.nu != nu {
                    return Err(format!("a side at nu {} in a profile at nu {nu}", c.nu));
                }
                (c.rows, c.chunk)
            }
            None => (&[], pad_chunk(index, profile.slots.len(), k)),
        };
        // The spec's columns, then the chunk word, then a live bit per port
        // without counts.
        let mut cols: Vec<Col> = sp.cols.iter().map(|c| c.0).collect();
        let word_col = cols.len();
        cols.push(Col::In);
        // One live bit for the whole row: its receives count that bit, and
        // a counted send carries its tag times the bit, so a row that
        // switches its premises off sends a tuple of tag 0, which nothing
        // receives.
        let mut live_col = vec![None; sp.ports.len()];
        let mut tag_col = vec![None; sp.ports.len()];
        if sp.ports.iter().any(|p| p.mult.is_empty()) {
            let live = cols.len();
            cols.push(Col::In);
            for (pi, p) in sp.ports.iter().enumerate() {
                if p.mult.is_empty() {
                    live_col[pi] = Some(live);
                } else if p.preset {
                    // Counted by its rule's own live bit.
                } else if let T::K(tag) = p.t[0] {
                    assert!(p.send, "a receive without a live bit");
                    tag_col[pi] = Some(cols.len());
                    cols.push(Col::Lin(tag, live));
                } else {
                    panic!("a counted send without a constant tag");
                }
            }
        }
        let io = io_index(&cols);
        let slot = sb.slot(Gate::new(cols.clone()));
        for (pi, p) in sp.ports.iter().enumerate() {
            let terms =
                p.t.iter()
                    .enumerate()
                    .map(|(ti, t)| match *t {
                        T::K(_) if ti == 0 && tag_col[pi].is_some() => BusTerm::word(io[tag_col[pi].unwrap()]),
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
        word_ws.push(word_w);
        // A checked declaration names its key in the statement: it sends
        // (STMT, key) exactly when it receives its CHECKED fact (the same
        // live bit), and the verifier receives the keys the statement claims.
        if sp.name == "decl_checked" {
            let pubs: Vec<usize> = sp
                .cols
                .iter()
                .enumerate()
                .filter(|(_, c)| matches!(c.1, Some(Src::Pub(_))))
                .map(|(y, _)| y)
                .collect();
            let recv = sp.ports.iter().position(|p| !p.send).expect("the CHECKED receive");
            let live = live_col[recv].expect("a receive has a live bit");
            let mut terms = vec![
                BusTerm {
                    word: None,
                    constant: int(TAG_STMT),
                    row: None,
                },
                BusTerm::word(io[pubs[0]]),
                BusTerm::word(io[pubs[1]]),
            ];
            terms.resize(W, BusTerm { word: None, constant: F128::ZERO, row: None });
            sb.bus_port_counted(slot, BusDirection::Send, terms, vec![io[live]]);
        }
        // A literal's rule reads the key of its type from the verifier, who
        // sends the statement's key of `Nat` (`String`) once per such row.
        if let Some(number) = prim_of(&sp.name) {
            let keys = prim_key_cols(sp);
            let live = live_col.iter().flatten().next().copied().expect("a rule has a live bit");
            let konst = |v: u64| BusTerm { word: None, constant: int(v), row: None };
            let mut terms = vec![konst(TAG_PRIM), konst(number), BusTerm::word(io[keys[0]]), BusTerm::word(io[keys[1]])];
            terms.resize(W, BusTerm { word: None, constant: F128::ZERO, row: None });
            sb.bus_port_counted(slot, BusDirection::Recv, terms, vec![io[live]]);
        }
        let n_live = live_col.iter().any(Option::is_some) as usize;
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
        out_slots.push((
            slot,
            sp.name.clone(),
            1usize << nu,
            sp.ports.len(),
            cols.len(),
        ));
    }
    // Every slot's chunk word goes on the bus once, from a table whose rows
    // are wired to those words; the verifier receives the list of words
    // the statement gives, all distinct. With as many words as slots in
    // the whole cone, balance leaves no two slots one word, so no two rows
    // one id.
    let tslot = sb.slot(Gate::new(vec![Col::In]));
    let mut terms = vec![
        BusTerm {
            word: None,
            constant: int(TAG_CHUNK),
            row: None,
        },
        BusTerm::word(0),
    ];
    terms.resize(W, BusTerm { word: None, constant: F128::ZERO, row: None });
    sb.bus_port_counted(tslot, BusDirection::Send, terms, Vec::new());
    for &w in &word_ws {
        sb.gate(tslot, &[w]);
    }
    out_slots.push((tslot, "chunk_words".to_string(), word_ws.len(), 1, 1));
    let shape = sb.finish().map_err(|e| format!("{e:?}"))?;
    Ok(Piece {
        nu,
        shape,
        values,
        slots: out_slots,
        tower: Some((bslot, gate)),
    })
}

/// A padding slot's chunk number: above every real one, and distinct for
/// every (leaf, slot) of a cone.
pub fn pad_chunk(index: usize, slots: usize, k: usize) -> u64 {
    // The same for leaves of different profiles: not by this leaf's slots.
    assert!(slots <= 4096 && k < slots);
    PAD_CHUNK + (index * 4096 + k) as u64
}

/// The index of the rare rules' table among a side's tables.
/// The table of the recursors' rules.
pub fn rule_table() -> usize {
    2 + TERM_TABLES.len() + 3
}

pub fn rare_table() -> usize {
    2 + TERM_TABLES.len() + 4 + FACT_TABLES.iter().filter(|r| **r != "Assumed" && rare_place(r).is_none()).count()
}

/// The constant a table's rule reads from the verifier, if it reads one.
fn prim_of(name: &str) -> Option<u64> {
    match name {
        "LitNat" => Some(PRIM_NAT),
        "LitStr" => Some(PRIM_STRING),
        _ => None,
    }
}

/// The columns of such a table that hold the constant's key.
fn prim_key_cols(sp: &Spec) -> Vec<usize> {
    let cols: Vec<usize> = sp
        .cols
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.1, Some(Src::Val(E::Dcl(_, Part::X(4) | Part::X(5))))))
        .map(|(y, _)| y)
        .collect();
    assert_eq!(cols.len(), 2, "the two key words");
    cols
}

/// The constants a leaf's rows read from the verifier: per row, the number
/// and the key the row holds for it ([`prims_product`]).
pub fn leaf_prims(proto: &LeanCircuit, profile: &Profile, slots: &[LeafSlot<'_>]) -> Vec<(u64, [F128; 2])> {
    let mut out = Vec::new();
    for (k, &s) in profile.slots.iter().enumerate() {
        let sp = &proto.specs[s];
        if let (Some(c), Some(number)) = (slots[k], prim_of(&sp.name)) {
            let keys = prim_key_cols(sp);
            out.extend(c.rows.iter().map(|z| (number, [z[keys[0]], z[keys[1]]])));
        }
    }
    out
}

/// The verifier's sends: per reading row, the key of the constant read.
/// A statement names one key for each constant (the keys of `Nat` and
/// `String`, which whoever reads the statement compares with the store's);
/// rows that hold different keys for one constant are refused.
pub fn prims_product(ch: &SharedChallenges, prims: &[(u64, [F128; 2])]) -> Result<F128, String> {
    let mut named: HashMap<u64, [F128; 2]> = HashMap::new();
    let mut p = F128::ONE;
    for &(number, key) in prims {
        if *named.entry(number).or_insert(key) != key {
            return Err(format!("two keys for the constant {number}"));
        }
        let mut acc = ch.beta;
        for (x, &g) in [int(TAG_PRIM), int(number), key[0], key[1]].iter().zip(&ch.gamma) {
            acc += g * *x;
        }
        p *= acc;
    }
    Ok(p)
}

/// What a leaf contributes to the statement: its slots' chunk words, and
/// the keys of the declarations its checked-declaration rows check.
pub fn leaf_statement(
    proto: &LeanCircuit,
    profile: &Profile,
    slots: &[LeafSlot<'_>],
    index: usize,
) -> (Vec<u64>, Vec<[F128; 2]>) {
    let mut words = Vec::new();
    let mut keys = Vec::new();
    for (k, &s) in profile.slots.iter().enumerate() {
        match slots[k] {
            Some(c) => {
                words.push(c.chunk << 32);
                let sp = &proto.specs[s];
                if sp.name == "decl_checked" {
                    let pubs: Vec<usize> = sp
                        .cols
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| matches!(c.1, Some(Src::Pub(_))))
                        .map(|(y, _)| y)
                        .collect();
                    for z in c.rows {
                        keys.push([z[pubs[0]], z[pubs[1]]]);
                    }
                }
            }
            None => words.push(pad_chunk(index, profile.slots.len(), k) << 32),
        }
    }
    (words, keys)
}

/// The verifier's side of the statement on the bus: one receive per chunk
/// word and per claimed key. Refuses a word list with a repeat or a word
/// that is not `c·2^32` for `c < 2^32` (a row's id is the word plus its
/// row, so ids could collide).
pub fn statement_product(
    ch: &SharedChallenges,
    words: &[u64],
    keys: &[[F128; 2]],
) -> Result<F128, String> {
    let mut sorted = words.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err("a chunk word repeats".into());
    }
    if sorted.iter().any(|w| w & 0xffff_ffff != 0) {
        return Err("a chunk word with low bits".into());
    }
    let f = |t: &[F128]| {
        let mut acc = ch.beta;
        for (x, &g) in t.iter().zip(&ch.gamma) {
            acc += g * *x;
        }
        acc
    };
    let mut p = F128::ONE;
    for &w in words {
        p *= f(&[int(TAG_CHUNK), int(w)]);
    }
    for k in keys {
        p *= f(&[int(TAG_STMT), k[0], k[1]]);
    }
    Ok(p)
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
    /// The piece's circuit shape.
    /// Number of input values held.
    pub fn n_values(&self) -> usize {
        self.values.len()
    }

    pub fn shape(&self) -> &CircuitShape {
        &self.shape
    }

    /// The tower posture's boolean gate (it owns the lincheck circuit).
    pub fn tower_gate(&self) -> Option<&BooleanGate> {
        self.tower.as_ref().map(|(_, g)| g)
    }

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
        BuiltParts {
            shape: &self.shape,
            values: &self.values,
        }
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
            Some(s) => std::borrow::Cow::Owned(
                self.shape.circuit.clone().with_shared_challenges(s.clone()),
            ),
        }
    }

    /// A tower leaf's parts: the circuit (with the shared challenges when
    /// given) and its witness, the proof and its commitment. The piece
    /// must be in the tower's posture.
    pub(crate) fn tower_parts(
        &self,
        shared: Option<&SharedChallenges>,
        label: &[u8],
    ) -> (
        flock_core::circuit::builder::BuiltCircuit,
        R1csProofCircuitMerged,
        Commitment,
    ) {
        assert!(
            self.tower.is_some(),
            "a tower leaf is a piece in the tower's posture"
        );
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
    pub fn prove(
        &self,
        shared: Option<&SharedChallenges>,
        label: &[u8],
        tamper: bool,
    ) -> PieceProof {
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

/// [`shared_balance`] with the statement's receives ([`statement_product`]):
/// the sends of every piece equal its receives times the statement's.
pub fn shared_balance_with<'a>(
    proofs: impl IntoIterator<Item = &'a R1csProofCircuitMerged>,
    statement: F128,
) -> bool {
    let (mut l, mut r) = (F128::ONE, F128::ONE);
    for b in proofs.into_iter().filter_map(|p| p.wiring.bus.as_ref()) {
        l *= b.gkr.top_lhs;
        r *= b.gkr.top_rhs;
    }
    l == r * statement
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
            .replace("F 2 Beta eq 1 5 4 0 : 1", "F 2 Beta eq 1 5 0 0 : 1")
            .replace("F 3 Trans eq 1 5 4 0 : 2 0", "F 3 Refl eq 1 5 5 0 :")
            .replace("F 4 Symm eq 1 4 5 0 : 3", "F 4 Refl eq 1 4 4 0 :");
        let rows = Rows::parse(&bad).unwrap();
        assert!(LeanCircuit::build(&rows).is_err());
        let c = LeanCircuit::build_with(&rows, false).unwrap();
        assert!(c.prove_and_verify(false).is_err());
    }

    /// A context entry claimed with another type, and one claimed at
    /// another position: neither is the context's row.
    /// Level rules claiming too much: a term with a level parameter passed
    /// as unchanged, a parameter given another level, a sort typed by
    /// itself.
    #[test]
    fn a_false_level_step_is_refused() {
        for (was, bad) in [
            ("F 12 LInstNone linst 0 1 0 0 :", "F 12 LInstNone linst 6 1 6 0 :"),
            ("F 13 LInstLParam linstl 2 1 1 0 :", "F 13 LInstLParam linstl 2 1 0 0 :"),
            ("F 15 Sort infer 1 7 0 0 :", "F 15 Sort infer 1 0 0 0 :"),
        ] {
            assert!(SMALL.contains(was));
            let rows = Rows::parse(&SMALL.replace(was, bad).replace("F 14 LInstSort linst 6 1 0 0 : 13", "F 14 Given linst 6 1 0 0 :")).unwrap();
            assert!(LeanCircuit::build(&rows).is_err(), "{bad}");
            let c = LeanCircuit::build_with(&rows, false).unwrap();
            assert!(c.prove_and_verify(false).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_false_context_entry_is_refused() {
        for (was, bad) in [
            ("F 6 HasLast has 1 0 0 0 :", "F 6 HasLast has 1 0 2 0 :"),
            ("F 6 HasLast has 1 0 0 0 :", "F 6 HasLast has 1 1 0 0 :"),
        ] {
            assert!(SMALL.contains(was));
            let rows = Rows::parse(&SMALL.replace(was, bad).replace("F 7 FVar infer 1 4 0 0 : 6", "F 7 Given infer 1 4 0 0 :")).unwrap();
            assert!(LeanCircuit::build(&rows).is_err());
            let c = LeanCircuit::build_with(&rows, false).unwrap();
            assert!(c.prove_and_verify(false).is_err());
        }
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
            assert!(
                same_commitment(&pp.commitment, &statement[i].1),
                "proving commits as the first phase did"
            );
            ok &= p
                .verify(Some(&ch), &label, &statement[i].1, &pp.proof)
                .is_ok();
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
        assert!(
            !shared(&c, total / 2 + 1, true).1,
            "a piece left out unbalances the bus"
        );
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
        p.verify(None, b"", &pp.commitment, &pp.proof)
            .expect("the tower piece verifies");
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
            .enumerate()
            .map(|(i, d)| {
                let slots: Vec<LeafSlot<'_>> = d.iter().zip(&profile.slots).map(|(x, &s)| x.map(|ch| c.slot(s, ch))).collect();
                leaf(&c, &profile, &slots, i).unwrap()
            })
            .collect();
        assert_eq!(
            leaves[0].digest(),
            leaves[1].digest(),
            "one circuit for every leaf"
        );
        let statement: Vec<([u8; 32], Commitment)> =
            leaves.iter().map(|p| (p.digest(), p.commit())).collect();
        let ch = shared_challenges(&statement);
        let mut proofs = Vec::new();
        for (i, p) in leaves.iter().enumerate() {
            let label = (i as u64).to_le_bytes();
            let pp = p.prove(Some(&ch), &label, false);
            p.verify(Some(&ch), &label, &statement[i].1, &pp.proof)
                .expect("the leaf verifies");
            proofs.push(pp.proof);
        }
        let (mut words, mut keys) = (Vec::new(), Vec::new());
        for (i, d) in dealt.iter().enumerate() {
            let slots: Vec<LeafSlot<'_>> = d.iter().zip(&profile.slots).map(|(x, &s)| x.map(|ch| c.slot(s, ch))).collect();
            let (w, k) = leaf_statement(&c, &profile, &slots, i);
            words.extend(w);
            keys.extend(k);
        }
        let st = statement_product(&ch, &words, &keys).unwrap();
        assert!(!shared_balance(&proofs), "the statement is part of the balance");
        assert!(shared_balance_with(&proofs, st), "the leaves balance with the statement");
        if let Some(k) = keys.first().copied() {
            let mut more = keys.clone();
            more.push(k);
            let st2 = statement_product(&ch, &words, &more).unwrap();
            assert!(!shared_balance_with(&proofs, st2), "a key claimed twice unbalances the bus");
        }
        let mut other = words.clone();
        other[0] = (PAD_CHUNK + (1 << 30)) << 32;
        let st3 = statement_product(&ch, &other, &keys).unwrap();
        assert!(!shared_balance_with(&proofs, st3), "another chunk word unbalances the bus");
        proofs.pop();
        assert!(
            !shared_balance_with(&proofs, st),
            "a leaf left out unbalances the bus"
        );
    }

    /// Two copies of one leaf: every row's id twice, each id naming two
    /// rows. The statement's word list then repeats (refused) or, made
    /// distinct, lists words no slot sends (unbalanced).
    #[test]
    fn repeated_chunk_words_are_refused() {
        let rows = Rows::merge((0..4000).map(|_| Rows::parse(SMALL).unwrap()).collect());
        let c = LeanCircuit::build(&rows).unwrap();
        let mut per_table: Vec<Vec<usize>> =
            vec![Vec::new(); c.chunks().iter().map(|x| x.0 + 1).max().unwrap()];
        for (s, ch, _) in c.chunks() {
            per_table[s].push(ch);
        }
        let n: Vec<usize> = per_table.iter().map(|t| t.len()).collect();
        let profile = Profile::even(c.nu, &n, 2);
        let dealt = profile.deal(&per_table, 2);
        let dealt = vec![dealt[0].clone(), dealt[0].clone()];
        let leaves: Vec<Piece> = dealt
            .iter()
            .map(|d| {
                let slots: Vec<LeafSlot<'_>> = d.iter().zip(&profile.slots).map(|(x, &s)| x.map(|ch| c.slot(s, ch))).collect();
                leaf(&c, &profile, &slots, 0).unwrap()
            })
            .collect();
        let statement: Vec<([u8; 32], Commitment)> =
            leaves.iter().map(|p| (p.digest(), p.commit())).collect();
        let ch = shared_challenges(&statement);
        let proofs: Vec<_> = leaves
            .iter()
            .enumerate()
            .map(|(i, p)| p.prove(Some(&ch), &(i as u64).to_le_bytes(), false).proof)
            .collect();
        let (mut words, mut keys) = (Vec::new(), Vec::new());
        for d in &dealt {
            let slots: Vec<LeafSlot<'_>> = d.iter().zip(&profile.slots).map(|(x, &s)| x.map(|ch| c.slot(s, ch))).collect();
            let (w, k) = leaf_statement(&c, &profile, &slots, 0);
            words.extend(w);
            keys.extend(k);
        }
        assert!(statement_product(&ch, &words, &keys).is_err());
        let mut fresh = words.clone();
        fresh.sort_unstable();
        fresh.dedup();
        let mut next = (PAD_CHUNK + (1 << 30)) << 32;
        while fresh.len() < words.len() {
            fresh.push(next);
            next += 1 << 32;
        }
        let st = statement_product(&ch, &fresh, &keys).unwrap();
        assert!(!shared_balance_with(&proofs, st));
    }

    #[test]
    fn a_false_beta_is_refused_in_pieces() {
        let bad = SMALL
            .replace("F 2 Beta eq 1 5 4 0 : 1", "F 2 Beta eq 1 5 0 0 : 1")
            .replace("F 3 Trans eq 1 5 4 0 : 2 0", "F 3 Refl eq 1 5 5 0 :")
            .replace("F 4 Symm eq 1 4 5 0 : 3", "F 4 Refl eq 1 4 4 0 :");
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
        Cone {
            shared: Rows::default(),
            terms: HashMap::new(),
            decls: HashMap::new(),
            span: 0,
        }
    }

    /// Adds one declaration's derivation: its terms and cited
    /// declarations join the shared side, its judgements and goal are
    /// appended to `batch` in the shared numbering.
    pub fn add(&mut self, r: Rows, batch: &mut Rows) -> Result<(), String> {
        // Declarations by key.
        let mut decl = Vec::with_capacity(r.decls.len());
        for d in &r.decls {
            if d.key == (0, 0) {
                return Err("a declaration without a key (an old dump)".into());
            }
            let i = match self.decls.get(&d.key) {
                Some(&i) => {
                    let e = &self.shared.decls[i];
                    if (e.unfold, e.levels, e.kind, e.order)
                        != (d.unfold, d.levels, d.kind, d.order)
                    {
                        return Err(format!(
                            "declaration {:?} differs between derivations",
                            d.key
                        ));
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
            if d.info.what != 0 {
                self.shared.decls[i].info = d.info;
            }
            self.span = self.span.max(d.order + 1);
            decl.push(i);
        }
        // Terms, with the levels, level lists and contexts among them.
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
                    self.span = self.span.max(t.rank + t.lbr).max(t.fv + 1).max(t.x[2].e + 1);
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
                        Some(_) => {
                            return Err(format!("declaration {:?} has two types or values", d.key));
                        }
                    }
                }
            }
        }
        for r in &r.rules {
            self.shared.rules.push(RuleRec { rec: decl[r.rec], ctor: r.ctor, fields: r.fields, rhs: tm[r.rhs] });
        }
        // Judgements and the goal.
        let base = batch.facts.len();
        for mut f in r.facts {
            for x in 0..4 {
                if f.is_term[x] {
                    f.raw[x] = tm[f.raw[x] as usize] as u64;
                }
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
                ctx: tm[g.ctx],
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
