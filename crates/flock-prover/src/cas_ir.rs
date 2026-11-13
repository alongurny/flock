//! Block circuits of the cas formats given as types (cas `spec/types.md`),
//! derived from the type alone. This file knows nothing of a prover: it
//! builds each circuit as a list of XOR and AND nodes, which is evaluated
//! here and lowered to a prover's gates elsewhere.
//!
//! A blob is 64-byte blocks. A record with no name has a fixed size, so its
//! place follows from the blob's length and one row checks one record. A
//! list of records that end in a name is checked by one row per block that
//! carries state to the next block.
//!
//! A name is 1 to 255 bytes, each an ASCII letter, a digit, `.`, `-`, `_` or
//! a space; it is not `.` alone and has no two dots in a row.

use std::cell::RefCell;
use std::collections::HashMap;

/// A bit of a circuit being built. `F` and `T` are the constants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct B(pub u32);
pub const F: B = B(0);
pub const T: B = B(1);

#[derive(Clone, Copy, Debug)]
pub enum Node {
    /// `F` (node 0) or `T` (node 1).
    Const,
    /// Bit `i` of the inputs: word `i / 128`, bit `i % 128`.
    In(u32),
    Xor(B, B),
    And(B, B),
}

#[derive(Default)]
struct Ir {
    nodes: Vec<Node>,
    memo: HashMap<(bool, B, B), B>,
}

thread_local!(static IR: RefCell<Ir> = RefCell::new(Ir::default()));

fn node(and: bool, a: B, b: B) -> B {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    IR.with(|ir| {
        let mut ir = ir.borrow_mut();
        if let Some(n) = ir.memo.get(&(and, a, b)) {
            return *n;
        }
        let n = B(ir.nodes.len() as u32);
        ir.nodes.push(if and { Node::And(a, b) } else { Node::Xor(a, b) });
        ir.memo.insert((and, a, b), n);
        n
    })
}

impl B {
    pub fn and(self, o: B) -> B {
        match (self, o) {
            (F, _) | (_, F) => F,
            (T, x) | (x, T) => x,
            (a, b) if a == b => a,
            (a, b) => node(true, a, b),
        }
    }
    pub fn xor(self, o: B) -> B {
        match (self, o) {
            (F, x) | (x, F) => x,
            (a, b) if a == b => F,
            (a, b) => node(false, a, b),
        }
    }
    pub fn not(self) -> B {
        self.xor(T)
    }
    pub fn or(self, o: B) -> B {
        self.not().and(o.not()).not()
    }
    /// `a` when `s`, else `b`.
    pub fn mux(s: B, a: B, b: B) -> B {
        b.xor(s.and(a.xor(b)))
    }
}

fn all(v: &[B]) -> B {
    v.iter().fold(T, |a, &b| a.and(b))
}
fn any(v: &[B]) -> B {
    v.iter().fold(F, |a, &b| a.or(b))
}
fn eq_const(x: &[B], c: u8) -> B {
    all(&(0..8).map(|i| if c >> (7 - i) & 1 == 1 { x[i] } else { x[i].not() }).collect::<Vec<_>>())
}
/// One-hot of the number `x` holds, most significant bit first.
fn decode(x: &[B]) -> Vec<B> {
    let n = x.len();
    (0..1usize << n)
        .map(|v| {
            all(&(0..n).map(|i| if v >> (n - 1 - i) & 1 == 1 { x[i] } else { x[i].not() }).collect::<Vec<_>>())
        })
        .collect()
}
/// `out[i]`: the number that `hot` is the one-hot of is above `i`.
fn below(hot: &[B], n: usize) -> Vec<B> {
    let mut out = vec![F; n];
    let mut acc = hot.iter().skip(n + 1).fold(F, |a, &b| a.xor(b));
    for i in (0..n).rev() {
        if i + 1 < hot.len() {
            acc = acc.xor(hot[i + 1]);
        }
        out[i] = acc;
    }
    out
}
/// A byte a name may hold: a letter, a digit, `+`, `.`, `-`, `_` or a space.
fn name_byte(b: &[B]) -> B {
    let (h, l) = (&b[..4], &b[4..]);
    let l32 = l[0].and(l[1]);
    let ge1 = any(l);
    let le9 = l[0].not().or(l[1].not().and(l[2].not()));
    let le10 = l[0].not().or(l[1].not().and(l[2].and(l[3]).not()));
    // 0x41..=0x4f and 0x50..=0x5a and 0x5f, the same with 0x20 added.
    let letter = B::mux(h[3], le10, ge1).or(h[2].not().and(h[3]).and(l32).and(l[2]).and(l[3]));
    // 0x30..=0x39, and 0x20, 0x2b, 0x2d, 0x2e.
    let plus = l[0].and(l[1].not()).and(l[2]).and(l[3]);
    let other = B::mux(h[3], le9, ge1.not().or(l32.and(l[2].xor(l[3]))).or(plus));
    h[0].not().and(B::mux(h[1], letter, h[2].and(other)))
}
/// Whether name bytes are wrong, and whether the last is a dot: a byte
/// inside the name is not one a name may hold or is a dot after a dot
/// (`before`: the byte before the first is a dot), or a byte outside the
/// name is not zero.
fn name_bytes(name: &[B], inside: &[B], before: B) -> (B, B) {
    let (mut bad, mut dots) = (F, before);
    for (i, &ins) in inside.iter().enumerate() {
        let b = &name[i * 8..i * 8 + 8];
        let zero = all(&b.iter().map(|x| x.not()).collect::<Vec<_>>());
        bad = bad.or(zero.xor(ins).not());
        bad = bad.or(ins.and(name_byte(b).not()));
        let dot = eq_const(b, b'.');
        bad = bad.or(ins.and(dot).and(dots));
        dots = dot;
    }
    (bad, dots)
}
/// Continue comparing `cur` with `prev`, bytes most significant first:
/// (`cur` is greater, they are equal so far).
fn compare(cur: &[B], prev: &[B], mut gt: B, mut eq: B) -> (B, B) {
    for i in 0..cur.len() {
        let e = eq.and(cur[i].xor(prev[i]));
        gt = gt.xor(e.and(cur[i]));
        eq = eq.xor(e);
    }
    (gt, eq)
}
/// A tag byte is not 1..=n.
fn tag_bad(t: &[B], n: u8) -> B {
    let hot = decode(&t[5..8]);
    any(&t[..5]).or(any(&hot[1..=n as usize]).not())
}

// The types.

#[derive(Clone, Copy)]
pub struct KeyF {
    pub optional: bool,
    /// 0: one target; n: one of n, with a tag byte.
    pub union: u8,
}
#[derive(Clone)]
pub struct Rec {
    pub keys: Vec<KeyF>,
    /// Choices that are not a key's target: the number of kinds of each.
    pub choices: Vec<u8>,
    /// Integers; true: at least 1.
    pub ints: Vec<bool>,
    pub name: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Sorted,
    Runs,
}
pub struct Type {
    pub alias: &'static str,
    pub head: Option<Rec>,
    pub list: Option<(Rec, Order)>,
    pub nonempty: bool,
    /// A text: bytes of any number, read by this automaton, with no head
    /// and no list.
    pub text: Option<Grammar>,
}

/// A finite automaton over classes of bytes. It starts in state 0.
pub struct Grammar {
    /// The byte ranges of each class; a byte in no class is refused.
    pub classes: Vec<Vec<(u8, u8)>>,
    /// The next state, by state and class.
    pub delta: Vec<Vec<Option<usize>>>,
    pub accept: Vec<bool>,
}

/// `text`: UTF-8 with no control character (none below 32, no 127, none of
/// U+0080 to U+009F).
pub fn text_grammar() -> Grammar {
    // Classes: 0 ASCII, 1 80-8F, 2 90-9F, 3 A0-BF, 4 C3-DF, 5 E0,
    // 6 E1-EC EE-EF, 7 ED, 8 F0, 9 F1-F3, 10 F4, 11 C2.
    let classes = vec![
        vec![(0x20, 0x7e)],
        vec![(0x80, 0x8f)],
        vec![(0x90, 0x9f)],
        vec![(0xa0, 0xbf)],
        vec![(0xc3, 0xdf)],
        vec![(0xe0, 0xe0)],
        vec![(0xe1, 0xec), (0xee, 0xef)],
        vec![(0xed, 0xed)],
        vec![(0xf0, 0xf0)],
        vec![(0xf1, 0xf3)],
        vec![(0xf4, 0xf4)],
        vec![(0xc2, 0xc2)],
    ];
    // States: 0 start, 1 one more, 2 two more, 3 after E0, 4 after ED,
    // 5 after F0, 6 three more, 7 after F4, 8 after C2.
    let mut d = vec![vec![None; 12]; 9];
    for (class, to) in [(0, 0), (4, 1), (5, 3), (6, 2), (7, 4), (8, 5), (9, 6), (10, 7), (11, 8)] {
        d[0][class] = Some(to);
    }
    for c in 1..=3 {
        d[1][c] = Some(0);
        d[2][c] = Some(1);
        d[6][c] = Some(2);
    }
    d[3][3] = Some(1);
    d[4][1] = Some(1);
    d[4][2] = Some(1);
    d[5][2] = Some(2);
    d[5][3] = Some(2);
    d[7][1] = Some(2);
    d[8][3] = Some(0);
    let mut accept = vec![false; 9];
    accept[0] = true;
    Grammar { classes, delta: d, accept }
}

/// `prose`: UTF-8 with no control character but the line feed.
pub fn prose_grammar() -> Grammar {
    let mut g = text_grammar();
    g.classes[0] = vec![(0x0a, 0x0a), (0x20, 0x7e)];
    g
}

/// `semver`: a version as Semantic Versioning 2.0.0 writes it. Three numbers
/// without leading zeros joined by dots; then maybe `-` and dot-separated
/// identifiers of letters, digits and hyphens, an identifier of digits alone
/// having no leading zero; then maybe `+` and such identifiers, any digits.
pub fn semver_grammar() -> Grammar {
    // Classes: 0 `0`, 1 `1`-`9`, 2 letters, 3 `-`, 4 `.`, 5 `+`.
    let classes = vec![
        vec![(b'0', b'0')],
        vec![(b'1', b'9')],
        vec![(b'A', b'Z'), (b'a', b'z')],
        vec![(b'-', b'-')],
        vec![(b'.', b'.')],
        vec![(b'+', b'+')],
    ];
    // States: 0 start; 1 major `0`; 2 major digits; 3 after the first dot;
    // 4 minor `0`; 5 minor digits; 6 after the second dot; 7 patch `0`;
    // 8 patch digits; 9 start of a pre-release identifier; 10 identifier
    // `0`; 11 digits after a leading `0`; 12 digits from `1`-`9`; 13 an
    // identifier with a letter or hyphen; 14 start of a build identifier;
    // 15 in a build identifier.
    let mut d = vec![vec![None; 6]; 16];
    let number = |d: &mut Vec<Vec<Option<usize>>>, from: usize, zero: usize, digits: usize| {
        d[from][0] = Some(zero);
        d[from][1] = Some(digits);
        d[digits][0] = Some(digits);
        d[digits][1] = Some(digits);
    };
    number(&mut d, 0, 1, 2);
    number(&mut d, 3, 4, 5);
    number(&mut d, 6, 7, 8);
    for s in [1, 2] {
        d[s][4] = Some(3);
    }
    for s in [4, 5] {
        d[s][4] = Some(6);
    }
    for s in [7, 8] {
        d[s][3] = Some(9);
        d[s][5] = Some(14);
    }
    // Pre-release identifiers.
    d[9][0] = Some(10);
    d[9][1] = Some(12);
    d[9][2] = Some(13);
    d[9][3] = Some(13);
    for s in [10, 11] {
        d[s][0] = Some(11);
        d[s][1] = Some(11);
    }
    for s in [10, 11, 12, 13] {
        d[s][2] = Some(13);
        d[s][3] = Some(13);
    }
    d[12][0] = Some(12);
    d[12][1] = Some(12);
    d[13][0] = Some(13);
    d[13][1] = Some(13);
    for s in [10, 12, 13] {
        d[s][4] = Some(9);
        d[s][5] = Some(14);
    }
    // Build identifiers.
    for c in 0..4 {
        d[14][c] = Some(15);
        d[15][c] = Some(15);
    }
    d[15][4] = Some(14);
    let mut accept = vec![false; 16];
    for s in [7, 8, 10, 12, 13, 15] {
        accept[s] = true;
    }
    Grammar { classes, delta: d, accept }
}

impl Grammar {
    /// Whether the automaton accepts `b`.
    pub fn accepts(&self, b: &[u8]) -> bool {
        let mut s = 0;
        for &x in b {
            let Some(c) = self.classes.iter().position(|rs| rs.iter().any(|&(lo, hi)| (lo..=hi).contains(&x))) else {
                return false;
            };
            match self.delta[s][c] {
                Some(t) => s = t,
                None => return false,
            }
        }
        self.accept[s]
    }
}

fn ge_const(x: &[B], c: u8) -> B {
    let (mut gt, mut eq) = (F, T);
    for i in 0..8 {
        if c >> (7 - i) & 1 == 1 {
            eq = eq.and(x[i]);
        } else {
            gt = gt.xor(eq.and(x[i]));
            eq = eq.and(x[i].not());
        }
    }
    gt.xor(eq)
}
fn in_range(x: &[B], lo: u8, hi: u8) -> B {
    if lo == hi {
        return eq_const(x, lo);
    }
    let above = ge_const(x, lo);
    if hi == 255 { above } else { above.and(ge_const(x, hi + 1).not()) }
}

/// The row for one 64-byte block of a text. Inputs: the block's four
/// words; a word with the automaton's state, one bit for each state (the
/// first state is the word 1); a word whose bit `i` says byte `i` is
/// inside the text (the bytes after a text's end are zero); and a word
/// whose lowest bit says this is the text's last block. Outputs: the next
/// state.
pub fn text_circuit(t: &Type) -> Circuit {
    let g = t.text.as_ref().expect("a text");
    let n = g.delta.len();
    build(format!("cas.{}.text", t.alias), 7, 1, |inp| {
        let blk = bytes(inp, 0, 64);
        let mut st: Vec<B> = (0..n).map(|s| inp[4 * 128 + s]).collect();
        let last = inp[6 * 128];
        let mut bad = F;
        for i in 0..64 {
            let (x, inside) = (&blk[8 * i..8 * i + 8], inp[5 * 128 + i]);
            let cl: Vec<B> =
                g.classes.iter().map(|rs| rs.iter().fold(F, |a, &(lo, hi)| a.xor(in_range(x, lo, hi)))).collect();
            let mut next = vec![F; n];
            for s in 0..n {
                for (to, next) in next.iter_mut().enumerate() {
                    let on = (0..cl.len()).filter(|&c| g.delta[s][c] == Some(to)).fold(F, |a, c| a.xor(cl[c]));
                    *next = next.xor(st[s].and(on));
                }
            }
            // Inside the text the automaton moves and must have a move;
            // outside it the byte is zero.
            bad = bad.or(B::mux(inside, any(&next).not(), any(x)));
            st = (0..n).map(|s| B::mux(inside, next[s], st[s])).collect();
        }
        let accepted = (0..n).filter(|&s| g.accept[s]).fold(F, |a, s| a.xor(st[s]));
        (vec![bad, last.and(accepted.not())], st)
    })
}

impl Rec {
    fn tags(&self) -> Vec<u8> {
        self.keys.iter().filter(|k| k.union > 0).map(|k| k.union).chain(self.choices.iter().copied()).collect()
    }
    /// The blocks of a record with no name.
    pub fn blocks(&self) -> usize {
        assert!(!self.name);
        (32 * self.keys.len() + self.tags().len() + 8 * self.ints.len()).div_ceil(64)
    }
}

fn key(optional: bool, union: u8) -> KeyF {
    KeyF { optional, union }
}
fn rec(keys: Vec<KeyF>, choices: Vec<u8>, ints: Vec<bool>, name: bool) -> Rec {
    Rec { keys, choices, ints, name }
}

/// The formats of cas `version/src/typed.rs`, in the same words.
pub fn types() -> Vec<Type> {
    let (must, may) = (key(false, 0), key(true, 0));
    vec![
        Type {
            alias: "directory",
            head: None,
            list: Some((rec(vec![key(false, 4)], vec![], vec![], true), Order::Sorted)),
            nonempty: false,
            text: None,
        },
        Type {
            alias: "commit",
            head: Some(rec(vec![must, must, may], vec![], vec![false], false)),
            list: Some((rec(vec![must, may], vec![], vec![], false), Order::Sorted)),
            nonempty: false,
            text: None,
        },
        Type {
            alias: "annotation",
            head: Some(rec(vec![may], vec![], vec![], false)),
            list: Some((rec(vec![key(false, 2)], vec![], vec![], true), Order::Sorted)),
            nonempty: false,
            text: None,
        },
        Type {
            alias: "lines",
            head: Some(rec(vec![may, may, may], vec![], vec![], false)),
            list: None,
            nonempty: false,
            text: None,
        },
        Type { alias: "text", head: None, list: None, nonempty: false, text: Some(text_grammar()) },
        Type { alias: "prose", head: None, list: None, nonempty: false, text: Some(prose_grammar()) },
        Type { alias: "tag", head: None, list: None, nonempty: false, text: Some(semver_grammar()) },
        Type {
            alias: "runs",
            head: None,
            list: Some((rec(vec![may], vec![], vec![true], false), Order::Runs)),
            nonempty: true,
            text: None,
        },
        Type {
            alias: "release",
            head: Some(rec(vec![must, must], vec![], vec![], false)),
            list: Some((rec(vec![must, must], vec![4], vec![], false), Order::Sorted)),
            nonempty: false,
            text: None,
        },
    ]
}

// Circuits.

/// A row's circuit: `inputs` and `outputs` count 128-bit words. The lowest
/// bit of the last input word turns the row off: nothing is then required
/// of the other inputs (a padding row).
pub struct Circuit {
    pub name: String,
    pub inputs: usize,
    pub outputs: usize,
    pub nodes: Vec<Node>,
    /// Bits that must be zero.
    pub zero: Vec<B>,
    /// The output bits, `outputs * 128` of them.
    pub out: Vec<B>,
    pub ands: usize,
}

fn build(name: String, inputs: usize, outputs: usize, f: impl FnOnce(&[B]) -> (Vec<B>, Vec<B>)) -> Circuit {
    IR.with(|ir| {
        let mut ir = ir.borrow_mut();
        ir.memo.clear();
        ir.nodes = vec![Node::Const, Node::Const];
        ir.nodes.extend((0..(inputs as u32 + 1) * 128).map(Node::In));
    });
    let inp: Vec<B> = (0..(inputs as u32 + 1) * 128).map(|i| B(i + 2)).collect();
    let (zero, mut out) = f(&inp[..inputs * 128]);
    let on = inp[inputs * 128].not();
    let zero: Vec<B> = zero.into_iter().map(|z| z.and(on)).collect();
    let inputs = inputs + 1;
    assert!(out.len() <= outputs * 128);
    out.resize(outputs * 128, F);
    let nodes = IR.with(|ir| std::mem::take(&mut ir.borrow_mut().nodes));
    let ands = nodes.iter().filter(|n| matches!(n, Node::And(..))).count();
    Circuit { name, inputs, outputs, nodes, zero, out, ands }
}

impl Circuit {
    /// The output words for input words (low 64 bits first), or `None`
    /// when a bit that must be zero is not.
    pub fn eval(&self, inputs: &[u128]) -> Option<Vec<u128>> {
        assert_eq!(inputs.len(), self.inputs);
        let mut v = vec![false; self.nodes.len()];
        v[1] = true;
        for (i, n) in self.nodes.iter().enumerate().skip(2) {
            v[i] = match *n {
                Node::Const => unreachable!(),
                Node::In(j) => inputs[j as usize / 128] >> (j % 128) & 1 == 1,
                Node::Xor(a, b) => v[a.0 as usize] ^ v[b.0 as usize],
                Node::And(a, b) => v[a.0 as usize] & v[b.0 as usize],
            };
        }
        if self.zero.iter().any(|z| v[z.0 as usize]) {
            return None;
        }
        let mut out = vec![0u128; self.outputs];
        for (j, o) in self.out.iter().enumerate() {
            out[j / 128] |= u128::from(v[o.0 as usize]) << (j % 128);
        }
        Some(out)
    }
}

/// The bits of `n` bytes starting at input word `word`, each byte's most
/// significant bit first. A word holds 16 bytes, the first in its low bits.
fn bytes(inp: &[B], word: usize, n: usize) -> Vec<B> {
    (0..n).flat_map(|p| (0..8).map(move |i| inp[(word + p / 16) * 128 + 8 * (p % 16) + (7 - i)])).collect()
}

/// A record with no name is wrong.
fn rec_bad(r: &Rec, x: &[B]) -> B {
    let mut bad = F;
    for (i, k) in r.keys.iter().enumerate() {
        if !k.optional {
            bad = bad.or(any(&x[256 * i..256 * i + 256]).not());
        }
    }
    let mut p = 256 * r.keys.len();
    for n in r.tags() {
        bad = bad.or(tag_bad(&x[p..p + 8], n));
        p += 8;
    }
    for &positive in &r.ints {
        if positive {
            bad = bad.or(any(&x[p..p + 64]).not());
        }
        p += 64;
    }
    bad.or(any(&x[p..]))
}

/// The row for a head record (`head`), or for one record of a list of
/// records with no name. Inputs: the record's blocks, four words each;
/// for a list record also the two words of the record before's first key
/// and a word whose lowest bit says there is none before.
pub fn rec_circuit(t: &Type, head: bool) -> Circuit {
    let (r, order) = match (head, &t.head, &t.list) {
        (true, Some(h), _) => (h, None),
        (false, _, Some((r, o))) => (r, Some(*o)),
        _ => panic!("{} has no such record", t.alias),
    };
    let s = r.blocks();
    let inputs = 4 * s + if head { 0 } else { 3 };
    build(format!("cas.{}.{}", t.alias, if head { "head" } else { "rec" }), inputs, 0, |inp| {
        let x = bytes(inp, 0, 64 * s);
        let mut zero = vec![rec_bad(r, &x)];
        if let Some(order) = order {
            let prev = bytes(inp, 4 * s, 32);
            let first = inp[(4 * s + 2) * 128];
            let (gt, eq) = compare(&x[..256], &prev, F, T);
            let wrong = if order == Order::Sorted { gt.not() } else { eq };
            zero.push(wrong.and(first.not()));
        }
        (zero, Vec::new())
    })
}

/// Name bytes in a head block after key, tag and length.
const N: usize = 30;
/// Earlier name blocks of an entry that are kept to compare the next with.
const SLOTS: usize = 4;
/// The words of the state a named list's rows carry.
pub const STATE: usize = 19;

struct State {
    /// The name bytes of the entry before's head block.
    head: Vec<B>,
    /// Its continuation blocks.
    slot: Vec<Vec<B>>,
    pv: Vec<B>,
    cv: Vec<B>,
    idx: Vec<B>,
    /// Name bytes still to come in continuation blocks.
    left: Vec<B>,
    gt: B,
    eq: B,
    /// The last name byte of the block before is a dot.
    dots: B,
}

impl State {
    /// Words 0..18 hold `head` and `slot`; the last word holds `gt` in its
    /// lowest bit (so the first state is the word 1 after zeros), then
    /// `eq`, `left`, `pv`, `cv`, `idx`, `dots`.
    fn read(inp: &[B], word: usize) -> State {
        let at = |i: usize| inp[word * 128 + i];
        let last = |i: usize| at((STATE - 1) * 128 + i);
        State {
            head: (0..N * 8).map(at).collect(),
            slot: (0..SLOTS).map(|j| (0..512).map(|i| at(N * 8 + 512 * j + i)).collect()).collect(),
            gt: last(0),
            eq: last(1),
            left: (0..8).map(|i| last(2 + i)).collect(),
            pv: (0..SLOTS).map(|i| last(10 + i)).collect(),
            cv: (0..SLOTS).map(|i| last(10 + SLOTS + i)).collect(),
            idx: (0..SLOTS).map(|i| last(10 + 2 * SLOTS + i)).collect(),
            dots: last(10 + 3 * SLOTS),
        }
    }
    fn write(&self) -> Vec<B> {
        let mut out = self.head.clone();
        for s in &self.slot {
            out.extend(s);
        }
        out.resize((STATE - 1) * 128, F);
        out.extend([self.gt, self.eq]);
        out.extend(&self.left);
        out.extend(&self.pv);
        out.extend(&self.cv);
        out.extend(&self.idx);
        out.push(self.dots);
        out
    }
}

/// The row for one block of a list of records that end in a name (key,
/// tag, length, name). Inputs: the block's four words, the state's
/// [`STATE`] words, and a word whose lowest bit says this is the blob's
/// last block. Outputs: the next state.
pub fn step_circuit(t: &Type) -> Circuit {
    let (r, _) = t.list.as_ref().expect("a list");
    assert!(r.name && r.keys.len() == 1 && r.choices.is_empty() && r.ints.is_empty());
    let k = r.keys[0];
    build(format!("cas.{}.step", t.alias), 4 + STATE + 1, STATE, |inp| {
        let blk = bytes(inp, 0, 64);
        let s = State::read(inp, 4);
        let last = inp[(4 + STATE) * 128];

        let cont = any(&s.left);
        let is_head = cont.not();
        // Read as the head block of an entry.
        let (tag, len, name) = (&blk[256..264], &blk[264..272], &blk[272..512]);
        let mut hbad = if k.union > 0 { tag_bad(tag, k.union) } else { F };
        if !k.optional {
            hbad = hbad.or(any(&blk[..256]).not());
        }
        let big = len[0].or(len[1]).or(len[2]);
        let hot = decode(&len[3..8]);
        hbad = hbad.or(big.not().and(hot[0]));
        let inside: Vec<B> = below(&hot, 31)[..N].iter().map(|&b| b.or(big)).collect();
        let (wrong, hdots) = name_bytes(name, &inside, F);
        hbad = hbad.or(wrong).or(big.not().and(hot[1]).and(eq_const(&name[0..8], b'.')));
        // The entry before was greater than the one before it.
        hbad = hbad.or(s.gt.not());
        let over = big.or(hot[31]);
        let mut borrow = F;
        let mut diff = vec![F; 8];
        for i in (0..8).rev() {
            let c = 30u8 >> (7 - i) & 1 == 1;
            diff[i] = if c { len[i].not().xor(borrow) } else { len[i].xor(borrow) };
            borrow = if c { len[i].not().or(borrow) } else { len[i].not().and(borrow) };
        }
        let hleft: Vec<B> = diff.iter().map(|&d| d.and(over)).collect();
        let (hgt, heq) = compare(name, &s.head, F, T);

        // Read as a continuation block: 64 name bytes, or the last ones.
        let ge64 = s.left[0].or(s.left[1]);
        let hot6 = decode(&s.left[2..8]);
        let cin: Vec<B> = below(&hot6, 64).iter().map(|&b| b.or(ge64)).collect();
        let (cbad, cdots) = name_bytes(&blk, &cin, s.dots);
        let mut cleft = vec![s.left[0].and(s.left[1]), s.left[0].and(s.left[1].not())];
        cleft.extend(s.left[2..].iter().map(|&b| b.and(ge64)));
        let mut old = vec![F; 512];
        for j in 0..SLOTS {
            let sel = s.idx[j].and(s.pv[j]);
            for i in 0..512 {
                old[i] = old[i].xor(sel.and(s.slot[j][i]));
            }
        }
        let (cgt, ceq) = compare(&blk, &old, s.gt, s.eq);

        let head: Vec<B> = (0..N * 8).map(|i| B::mux(is_head, name[i], s.head[i])).collect();
        let mut slot = s.slot.clone();
        let (mut pv, mut cv, mut idx) = (vec![F; SLOTS], vec![F; SLOTS], vec![F; SLOTS]);
        for j in 0..SLOTS {
            let w = cont.and(s.idx[j]);
            for i in 0..512 {
                slot[j][i] = B::mux(w, blk[i], s.slot[j][i]);
            }
            pv[j] = B::mux(is_head, s.cv[j], s.pv[j]);
            cv[j] = is_head.not().and(s.cv[j].or(s.idx[j]));
            idx[j] = if j == 0 { is_head } else { cont.and(s.idx[j - 1]) };
        }
        let next = State {
            head,
            slot,
            pv,
            cv,
            idx,
            left: (0..8).map(|i| B::mux(is_head, hleft[i], cleft[i])).collect(),
            gt: B::mux(is_head, hgt, cgt),
            eq: B::mux(is_head, heq, ceq),
            dots: B::mux(is_head, hdots, cdots),
        };
        let zero = vec![B::mux(is_head, hbad, cbad), last.and(any(&next.left).or(next.gt.not()))];
        (zero, next.write())
    })
}

// A blob as rows.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Head,
    Rec,
    Step,
    Text,
}

/// One row of a blob: its kind, its first block and how many blocks it
/// takes, whether it is the first or the last of its kind in the blob, and
/// for a text how many bytes of its block are inside the text.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub kind: Kind,
    pub block: usize,
    pub blocks: usize,
    pub first: bool,
    pub last: bool,
    pub bytes: usize,
}

/// The rows that check a blob of `len` bytes, or `None` when no blob of
/// that length is valid.
pub fn plan(t: &Type, len: usize) -> Option<Vec<Row>> {
    if let Some(g) = &t.text {
        // No block is read of an empty text: it is valid only if the
        // automaton accepts at its start.
        if len == 0 && !g.accept[0] {
            return None;
        }
        let n = len.div_ceil(64);
        return Some(
            (0..n)
                .map(|b| Row {
                    kind: Kind::Text,
                    block: b,
                    blocks: 1,
                    first: b == 0,
                    last: b + 1 == n,
                    bytes: (len - 64 * b).min(64),
                })
                .collect(),
        );
    }
    if len % 64 != 0 {
        return None;
    }
    let blocks = len / 64;
    let mut rows = Vec::new();
    let mut p = 0;
    if let Some(h) = &t.head {
        if blocks < h.blocks() {
            return None;
        }
        rows.push(Row { kind: Kind::Head, block: 0, blocks: h.blocks(), first: true, last: true, bytes: 64 * h.blocks() });
        p = h.blocks();
    }
    match &t.list {
        None if p != blocks => return None,
        None => {}
        Some(_) if t.nonempty && p == blocks => return None,
        Some((r, _)) if r.name => {
            rows.extend((p..blocks).map(|b| Row {
                kind: Kind::Step,
                block: b,
                blocks: 1,
                first: b == p,
                last: b + 1 == blocks,
                bytes: 64,
            }));
        }
        Some((r, _)) => {
            let s = r.blocks();
            if (blocks - p) % s != 0 {
                return None;
            }
            rows.extend(
                (p..blocks)
                    .step_by(s)
                    .map(|b| Row { kind: Kind::Rec, block: b, blocks: s, first: b == p, last: b + s == blocks, bytes: 64 * s }),
            );
        }
    }
    Some(rows)
}

/// The circuits of a type: head, record, step, text, where it has them.
pub struct Circuits {
    pub head: Option<Circuit>,
    pub rec: Option<Circuit>,
    pub step: Option<Circuit>,
    pub text: Option<Circuit>,
}

/// The circuit of one kind of row of a type.
pub fn circuit(t: &Type, kind: Kind) -> Circuit {
    match kind {
        Kind::Head => rec_circuit(t, true),
        Kind::Rec => rec_circuit(t, false),
        Kind::Step => step_circuit(t),
        Kind::Text => text_circuit(t),
    }
}

/// The word whose bit `i` says byte `i` of a block is inside a text that
/// has `bytes` bytes in the block.
pub fn inside(bytes: usize) -> u128 {
    if bytes >= 64 { u128::from(u64::MAX) } else { (1u128 << bytes) - 1 }
}

pub fn circuits(t: &Type) -> Circuits {
    let named = t.list.as_ref().is_some_and(|(r, _)| r.name);
    Circuits {
        head: t.head.as_ref().map(|_| rec_circuit(t, true)),
        rec: (t.list.is_some() && !named).then(|| rec_circuit(t, false)),
        step: named.then(|| step_circuit(t)),
        text: t.text.as_ref().map(|_| text_circuit(t)),
    }
}

/// The four words of a 64-byte block.
pub fn words(block: &[u8]) -> [u128; 4] {
    std::array::from_fn(|i| u128::from_le_bytes(block[16 * i..16 * i + 16].try_into().unwrap()))
}

/// Whether `blob` is valid, by its rows.
pub fn check(t: &Type, c: &Circuits, blob: &[u8]) -> bool {
    let Some(rows) = plan(t, blob.len()) else { return false };
    let mut padded = blob.to_vec();
    padded.resize(blob.len().next_multiple_of(64), 0);
    let blob = &padded[..];
    let mut state = vec![0u128; STATE];
    state[STATE - 1] = 1;
    let mut text = 1u128;
    let mut prev = [0u128; 2];
    for r in rows {
        let mut inputs: Vec<u128> =
            blob[64 * r.block..64 * (r.block + r.blocks)].chunks_exact(64).flat_map(words).collect();
        let m = [inputs[0], inputs[1]];
        let out = match r.kind {
            Kind::Head => {
                inputs.push(0);
                c.head.as_ref().unwrap().eval(&inputs)
            }
            Kind::Rec => {
                inputs.extend([prev[0], prev[1], u128::from(r.first), 0]);
                c.rec.as_ref().unwrap().eval(&inputs)
            }
            Kind::Text => {
                inputs.extend([text, inside(r.bytes), u128::from(r.last), 0]);
                c.text.as_ref().unwrap().eval(&inputs)
            }
            Kind::Step => {
                inputs.extend(&state);
                inputs.extend([u128::from(r.last), 0]);
                c.step.as_ref().unwrap().eval(&inputs)
            }
        };
        match out {
            None => return false,
            Some(o) if r.kind == Kind::Text => text = o[0],
            Some(o) if r.kind == Kind::Step => state = o,
            Some(_) => prev = m,
        }
    }
    true
}
