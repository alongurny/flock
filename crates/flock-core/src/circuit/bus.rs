//! The tuple bus: a multiset equality between tuples that gate rows SEND and
//! tuples they RECEIVE, proven by a grand product next to the wiring argument.
//!
//! The wiring argument proves copy constraints along a permutation σ fixed
//! when the circuit is built. Some dataflow is not fixed: which row of a
//! hash-chain table steps which chain depends on the witness. A bus moves
//! such values by CONTENT instead of by position. Each [`BusPort`] is one
//! gate type's rows, every live row contributing one tuple of
//! [`Bus::width`] coordinates, and the argument proves
//!
//! ```text
//! { tuple(p, j) : p a Send port, j < count(p) }
//!   = { tuple(p, j) : p a Recv port, j < count(p) }        (as multisets)
//! ```
//!
//! A coordinate ([`BusTerm`]) is a committed word of the row (one of its
//! type's gate cell-slots), plus a constant, plus a static [`RowTag`] — the
//! part that ties a tuple to a region of rows (a signature, say) without
//! committing anything. `docs/tuple-bus-design.md` has the soundness
//! argument and how leanXMSS uses it.
//!
//! ## Protocol
//!
//! After the wiring GKR, on the same transcript: sample `β` (ground like the
//! wiring's `α`) and `γ_1..γ_w`; a tuple `t` becomes the leaf
//! `β + Σ_k γ_k·t_k`. Send ports fill the left product's groups and receive
//! ports the right's, one group of `2^ν` rows per port, dead rows and unused
//! groups being 1. [`product_gkr::prove_leaves`] proves the products equal
//! and reduces both sides to one point `ρ = (ρ_row, ρ_port)`. Each side's
//! leaf MLE there is affine in the committed words' MLEs at `ρ_row`:
//!
//! ```text
//! L̂(ρ) = 1 + Σ_p eq(ρ_port, p)·[ (β + 1)·E_p + Σ_k γ_k·(ŵ_{p,k}(ρ_row)
//!                                   + c_{p,k}·E_p + T_{p,k}(ρ_row)) ]
//! ```
//!
//! with `E_p = Σ_{j<count} eq(ρ_row, j)` and `T` the row tag's MLE over the
//! live rows. The prover sends one gather value per committed word used —
//! the same packed-direct claims the wiring's gather produces, bound to the
//! commitment by the opening — and the verifier rebuilds both sides.
//!
//! ## Multiplicities
//!
//! A port may count each row `m(j) = Σ_i 2^i·m_i(j)` times, the bits `m_i`
//! being committed words of the row ([`BusPort::mult`]); a lookup table then
//! supplies each entry once with the number of requests. LogUp would sum
//! `m/(β + t)`, which in characteristic 2 only sees `m mod 2`, so the bus
//! stays a product and multiplies `f^m = ∏_i (f^{2^i})^{m_i}` instead, with
//! `f = β + Σ_k γ_k·t_k`. Such a port fills one group per bit, row `j` of
//! group `i` holding the leaf `1 + m_i·(1 + f^{2^i})`: `f^{2^i}` when the
//! bit is set, 1 when not.
//!
//! That leaf is not affine in the committed words, so the verifier does not
//! rebuild it. The prover sends each such group's `S = Σ_j eq(ρ_row, j)·
//! m_i(j)·(1 + F_i(j))` (`F_i = f^{2^i}`), which enters the side in place
//! of the affine bracket, and proves all of them, together with
//! `Σ_j eq(ρ_row, j)·m_i(j)·(1 + m_i(j)) = 0` (each bit is 0 or 1: the zero
//! check of `m² + m` at the random `ρ_row`), in one degree-3 sumcheck over
//! the rows, batched by fresh `λ_g, μ_g`. It ends at a row point `r` and
//! needs `m̂_i(r)` and `F̂_i(r)`. Frobenius is additive and fixes the
//! Boolean cube, so `eq(σ, j)^{2^i} = eq(σ^{2^i}, j)` and
//!
//! ```text
//! F̂_i(r) = Σ_j eq(r, j)·f(j)^{2^i} = f̂(σ_i)^{2^i},   σ_i = r^{2^{-i}}
//! ```
//!
//! coordinate-wise, where `f̂` is affine in the words' MLEs. So the prover
//! opens each bit at `r` and each word of the port at every `σ_i`, all of
//! them packed-direct claims like the gather's. A bit that is not 0 or 1
//! would let the prover set a leaf to anything; the zero check is what
//! stops that. The fingerprint's degree in `β` grows to `Σ m`, which
//! [`live_entries`] counts for grinding.

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    ParallelSliceMut,
};
use serde::{Deserialize, Serialize};

use crate::{
    challenger::Challenger,
    circuit::{CellSlot, Circuit, CircuitError, WiringError},
    field::F128,
    pcs::{DirectEqInd, PackedDirectClaim},
    product_gkr::{
        BatchedGrinding, LiveMask, ProductGkrLeavesProof, ProductLeaves, prove_leaves,
        verify_leaves, verify_leaves_open,
    },
    scratch::take_f128,
    zerocheck::univariate_skip::build_eq,
};

const DOMAIN_BUS: &[u8] = b"flock-tuple-bus-v1";

/// Which product a port's tuples join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusDirection {
    Send,
    Recv,
}

/// A static per-row value: row `j` of the port gets the integer
/// `((offset + j) / div) mod modulo` (no reduction when `modulo = 0`), as the
/// field element with that bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowTag {
    pub offset: u64,
    pub div: u64,
    pub modulo: u64,
}

impl RowTag {
    #[inline]
    pub fn value(&self, row: usize) -> F128 {
        let q = (self.offset + row as u64) / self.div;
        F128::new(if self.modulo == 0 { q } else { q % self.modulo }, 0)
    }
}

/// One tuple coordinate: `word + constant + row tag`, each part optional. In
/// a circuit, `word` is a gate cell-slot of the port's type; at the builder
/// (`ShapeBuilder::bus_port`), it is an index into the gate's io schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusTerm {
    pub word: Option<usize>,
    pub constant: F128,
    pub row: Option<RowTag>,
}

impl BusTerm {
    /// A committed word of the row.
    pub const fn word(word: usize) -> Self {
        Self {
            word: Some(word),
            constant: F128::ZERO,
            row: None,
        }
    }

    /// A per-row static value.
    pub const fn row(tag: RowTag) -> Self {
        Self {
            word: None,
            constant: F128::ZERO,
            row: Some(tag),
        }
    }

    /// The same term plus a constant.
    pub const fn plus(self, constant: F128) -> Self {
        Self { constant, ..self }
    }
}

/// One gate type's live rows, each contributing a tuple to one side.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusPort {
    /// Registry slot index.
    pub ty: usize,
    pub dir: BusDirection,
    pub terms: Vec<BusTerm>,
    /// The row's multiplicity, as bits, least significant first: cell-slots
    /// of the port's type, each holding 0 or 1 (the bus proves it). Row `j`
    /// then counts `Σ_i 2^i·m_i(j)` times, so 0 leaves it out. Empty: every
    /// live row counts once. See the module's "Multiplicities" section.
    #[serde(default)]
    pub mult: Vec<usize>,
}

/// A circuit's bus: every port's tuple has `width` coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bus {
    pub width: usize,
    pub ports: Vec<BusPort>,
    /// Challenges fixed outside this proof, for a bus shared by several
    /// proofs (pieces). The caller derives them from every piece's
    /// commitment, so no piece chooses its witness after seeing them. With
    /// them, this proof no longer checks `∏ send = ∏ recv` on its own: the
    /// two products are public in [`BusProof::gkr`] and the caller checks
    /// that the products of all pieces balance ([`shared_balance`]).
    #[serde(default)]
    pub shared: Option<SharedChallenges>,
}

/// The fingerprint challenges of a bus shared across proofs: `β` and one
/// `γ_k` per coordinate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedChallenges {
    pub beta: F128,
    pub gamma: Vec<F128>,
}

/// Whether the pieces' shared buses balance: the product of every piece's
/// send side equals the product of every receive side. Each piece's own
/// verification already checked its products are nonzero and correct.
pub fn shared_balance<'a>(proofs: impl IntoIterator<Item = &'a BusProof>) -> bool {
    let (mut l, mut r) = (F128::ONE, F128::ONE);
    for p in proofs {
        l *= p.gkr.top_lhs;
        r *= p.gkr.top_rhs;
    }
    l == r
}

/// The bus half of a wiring proof: the grand product's transcript, one
/// gather value per committed word the ports read (ascending cell-slot), and
/// the fingerprint challenge's PoW nonce when grinding is on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusProof {
    pub gkr: ProductGkrLeavesProof,
    pub gather: Vec<F128>,
    pub fingerprint_nonce: Vec<u64>,
    /// The multiplicity groups' sumcheck, when a port has multiplicities.
    #[serde(default)]
    pub mult: Option<MultProof>,
}

/// The multiplicity groups' half of a [`BusProof`]: each group's `S` (in
/// [`Bus::mult_groups`] order), the sumcheck's rounds (`s(0), s(x), s(x+1)`
/// for the generator `x`; `s(1)` follows from the claim), the bits at the
/// end point `r` ([`Bus::mult_bits`] order) and the words at each `σ_i`
/// ([`Bus::mult_words`] order).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultProof {
    pub sums: Vec<F128>,
    pub rounds: Vec<[F128; 3]>,
    pub bits: Vec<F128>,
    pub words: Vec<F128>,
}

/// A static column of the bus's leaf reconstruction: a port type's live-row
/// indicator, or a row tag over that type's live rows. The verifier reads
/// each at `ρ_row`; recursion defers them to the circuit-structure table
/// ([`super::SigmaAssertion::bus`]) instead of evaluating them in-circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusColumn {
    Live { ty: usize },
    Tag { ty: usize, tag: RowTag },
}

impl BusColumn {
    /// The column's entry at `row`: zero past the type's live rows.
    pub fn entry(&self, circuit: &Circuit, row: usize) -> F128 {
        let (ty, tag) = match *self {
            BusColumn::Live { ty } => (ty, None),
            BusColumn::Tag { ty, tag } => (ty, Some(tag)),
        };
        if row >= circuit.counts()[ty] {
            return F128::ZERO;
        }
        tag.map_or(F128::ONE, |t| t.value(row))
    }

    /// `Σ_{j<live} eq(ρ_row, j)·entry(j)`.
    pub fn eval(&self, circuit: &Circuit, rho_row: &[F128], eq_row: &[F128]) -> F128 {
        let rows = eq_row.len();
        match *self {
            BusColumn::Live { ty } => {
                LiveMask::eq_prefix_sum(rho_row, circuit.counts()[ty].min(rows))
            }
            BusColumn::Tag { ty, tag } => {
                let live = circuit.counts()[ty].min(rows);
                eq_row[..live]
                    .iter()
                    .enumerate()
                    .fold(F128::ZERO, |a, (j, &e)| a + e * tag.value(j))
            }
        }
    }
}

/// The bus's deferred static evaluations: `ρ_row` and each
/// [`Bus::structure_columns`] entry's value there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusStructure {
    pub rho_row: Vec<F128>,
    pub values: Vec<F128>,
}

/// The multiplicity sumcheck's deferred static evaluations: each `σ_i`
/// and the row-tag columns' values there ([`Bus::mult_tag_columns`] order).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultStructure {
    pub sigma: Vec<Vec<F128>>,
    /// [`Bus::mult_tag_columns`]: `(i, column)` per value.
    pub columns: Vec<(usize, usize)>,
    pub values: Vec<F128>,
}

impl Bus {
    /// Check the bus against the circuit it rides on: every port names a
    /// registry type, carries `width` terms, and reads only that type's own
    /// cell-slots.
    pub(crate) fn validate(&self, circuit: &Circuit) -> Result<(), CircuitError> {
        let bad = |why: &'static str| Err(CircuitError::InvalidBus(why));
        if self.width == 0 {
            return bad("a bus tuple needs a coordinate");
        }
        if self.ports.is_empty() {
            return bad("a bus needs a port");
        }
        let slots = circuit.cells().slots();
        for port in &self.ports {
            if port.ty >= circuit.counts().len() {
                return bad("a port names no registry type");
            }
            if port.terms.len() != self.width {
                return bad("a port's tuple has the wrong width");
            }
            if port.mult.len() > 63 {
                return bad("a multiplicity has at most 63 bits");
            }
            for (i, iota) in port.mult.iter().enumerate() {
                if port.mult[..i].contains(iota) {
                    return bad("a multiplicity names a bit twice");
                }
                match slots.get(*iota) {
                    Some(CellSlot::Gate { ty, .. }) if *ty == port.ty => {}
                    _ => return bad("a multiplicity bit is outside its port's type"),
                }
            }
            for term in &port.terms {
                if term.row.is_some_and(|t| t.div == 0) {
                    return bad("a row tag divides by zero");
                }
                if let Some(iota) = term.word {
                    match slots.get(iota) {
                        Some(CellSlot::Gate { ty, .. }) if *ty == port.ty => {}
                        _ => return bad("a term reads a word outside its port's type"),
                    }
                }
            }
        }
        Ok(())
    }

    /// The digest encoding: fixed-width fields, length-prefixed lists.
    pub(crate) fn absorb_into(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(DOMAIN_BUS);
        buf.extend_from_slice(&(self.width as u32).to_le_bytes());
        buf.extend_from_slice(&(self.ports.len() as u32).to_le_bytes());
        for port in &self.ports {
            buf.extend_from_slice(&(port.ty as u32).to_le_bytes());
            buf.push(matches!(port.dir, BusDirection::Recv) as u8);
            for term in &port.terms {
                match term.word {
                    Some(iota) => {
                        buf.push(1);
                        buf.extend_from_slice(&(iota as u32).to_le_bytes());
                    }
                    None => buf.push(0),
                }
                buf.extend_from_slice(&term.constant.lo.to_le_bytes());
                buf.extend_from_slice(&term.constant.hi.to_le_bytes());
                match term.row {
                    Some(t) => {
                        buf.push(1);
                        for x in [t.offset, t.div, t.modulo] {
                            buf.extend_from_slice(&x.to_le_bytes());
                        }
                    }
                    None => buf.push(0),
                }
            }
        }
        // The bus is the digest's last part, so a trailer only when some
        // port counts rows keeps every earlier digest.
        if self.ports.iter().any(|p| !p.mult.is_empty()) {
            buf.extend_from_slice(b"multiplicities");
            for port in &self.ports {
                buf.extend_from_slice(&(port.mult.len() as u32).to_le_bytes());
                for &iota in &port.mult {
                    buf.extend_from_slice(&(iota as u32).to_le_bytes());
                }
            }
        }
    }

    /// The static columns the leaf reconstruction reads, each once: per port
    /// without multiplicities, in declaration order, its type's live
    /// indicator, then each row tag.
    pub fn structure_columns(&self) -> Vec<BusColumn> {
        let mut cols = Vec::new();
        for port in self.ports.iter().filter(|p| p.mult.is_empty()) {
            let live = BusColumn::Live { ty: port.ty };
            let tags = port
                .terms
                .iter()
                .filter_map(|t| t.row)
                .map(|tag| BusColumn::Tag { ty: port.ty, tag });
            for col in std::iter::once(live).chain(tags) {
                if !cols.contains(&col) {
                    cols.push(col);
                }
            }
        }
        cols
    }

    /// The bus plane of the circuit-structure table: the
    /// [`Self::structure_columns`], then each row-tag column only the ports
    /// with multiplicities read (evaluated at the `σ_i`).
    pub fn table_columns(&self) -> Vec<BusColumn> {
        let mut cols = self.structure_columns();
        for port in self.ports.iter().filter(|p| !p.mult.is_empty()) {
            for tag in port.terms.iter().filter_map(|t| t.row) {
                let col = BusColumn::Tag { ty: port.ty, tag };
                if !cols.contains(&col) {
                    cols.push(col);
                }
            }
        }
        cols
    }

    /// The column index of `col` in [`Self::structure_columns`].
    pub fn column_index(cols: &[BusColumn], col: BusColumn) -> usize {
        cols.iter().position(|&c| c == col).expect("a bus column")
    }

    /// The cell-slots the ports without multiplicities read, ascending — the
    /// gather order.
    pub fn words(&self) -> Vec<usize> {
        let mut words: Vec<usize> = self
            .ports
            .iter()
            .filter(|p| p.mult.is_empty())
            .flat_map(|p| p.terms.iter().filter_map(|t| t.word))
            .collect();
        words.sort_unstable();
        words.dedup();
        words
    }

    /// Ports per side, in declaration order.
    pub fn side(&self, dir: BusDirection) -> Vec<&BusPort> {
        self.ports.iter().filter(|p| p.dir == dir).collect()
    }

    /// Whether some port counts its rows.
    pub fn has_mult(&self) -> bool {
        self.ports.iter().any(|p| !p.mult.is_empty())
    }

    /// One side's product groups, in order: a port without multiplicities
    /// is one group (`None`), a port with `b` bits is `b` groups, bit `i`
    /// holding `f^{2^i}` where it is set (`Some(i)`).
    pub fn groups(&self, dir: BusDirection) -> Vec<(usize, Option<usize>)> {
        let mut out = Vec::new();
        for (p, port) in self.ports.iter().enumerate().filter(|(_, p)| p.dir == dir) {
            if port.mult.is_empty() {
                out.push((p, None));
            } else {
                out.extend((0..port.mult.len()).map(|i| (p, Some(i))));
            }
        }
        out
    }

    /// The multiplicity groups `(port, bit)` in proof order: the send side's,
    /// then the receive side's.
    pub fn mult_groups(&self) -> Vec<(usize, usize)> {
        [BusDirection::Send, BusDirection::Recv]
            .into_iter()
            .flat_map(|d| self.groups(d))
            .filter_map(|(p, i)| i.map(|i| (p, i)))
            .collect()
    }

    /// The multiplicity bits' cell-slots, ascending: opened at `r`.
    pub fn mult_bits(&self) -> Vec<usize> {
        let mut bits: Vec<usize> = self.ports.iter().flat_map(|p| p.mult.clone()).collect();
        bits.sort_unstable();
        bits.dedup();
        bits
    }

    /// `(i, column)` for each row-tag column (an index into
    /// [`Self::table_columns`]) a port with at least `i + 1` bits reads,
    /// by `i` then column: evaluated at `σ_i`, deferred like the columns at
    /// `ρ_row` ([`MultStructure`]).
    pub fn mult_tag_columns(&self) -> Vec<(usize, usize)> {
        let cols = self.table_columns();
        let most = self.ports.iter().map(|p| p.mult.len()).max().unwrap_or(0);
        let mut out = Vec::new();
        for i in 0..most {
            let mut at: Vec<usize> = self
                .ports
                .iter()
                .filter(|p| p.mult.len() > i)
                .flat_map(|p| {
                    p.terms
                        .iter()
                        .filter_map(|t| t.row.map(|tag| BusColumn::Tag { ty: p.ty, tag }))
                })
                .map(|c| Self::column_index(&cols, c))
                .collect();
            at.sort_unstable();
            at.dedup();
            out.extend(at.into_iter().map(|c| (i, c)));
        }
        out
    }

    /// `(i, cell-slot)` for each word a port with at least `i + 1` bits
    /// reads, by `i` then cell-slot: opened at `σ_i`.
    pub fn mult_words(&self) -> Vec<(usize, usize)> {
        let most = self.ports.iter().map(|p| p.mult.len()).max().unwrap_or(0);
        let mut out = Vec::new();
        for i in 0..most {
            let mut words: Vec<usize> = self
                .ports
                .iter()
                .filter(|p| p.mult.len() > i)
                .flat_map(|p| p.terms.iter().filter_map(|t| t.word))
                .collect();
            words.sort_unstable();
            words.dedup();
            out.extend(words.into_iter().map(|w| (i, w)));
        }
        out
    }

    /// Port-group bits: both sides share a `2^c` grid of port groups.
    pub fn c_bits(&self) -> usize {
        let most = self
            .groups(BusDirection::Send)
            .len()
            .max(self.groups(BusDirection::Recv).len())
            .max(1);
        most.next_power_of_two().trailing_zeros() as usize
    }
}

/// The bus's challenges: `β` and one `γ_k` per coordinate.
struct Challenges {
    beta: F128,
    gamma: Vec<F128>,
}

/// Bind a shared bus's challenges into this proof's transcript.
fn observe_shared<C: Challenger>(shared: &SharedChallenges, ch: &mut C) {
    ch.observe_f128(shared.beta);
    for &g in &shared.gamma {
        ch.observe_f128(g);
    }
}

/// The fingerprint product's degree in `β`: one per live row, `2^b − 1` for
/// a row of a port with `b` multiplicity bits.
fn live_entries(bus: &Bus, circuit: &Circuit) -> usize {
    let rows = 1usize << circuit.cells().nu();
    bus.ports
        .iter()
        .map(|p| {
            let most = (1usize << p.mult.len().min(40)) - 1;
            circuit.counts()[p.ty].min(rows).saturating_mul(most.max(1))
        })
        .fold(0usize, usize::saturating_add)
}

/// Prove the bus balances on the committed buffer `packed` (the same padded
/// union buffer [`super::prove_wiring`] reads). Returns the proof and one
/// packed-direct claim per committed word read — the gather's, then the
/// multiplicity sumcheck's — for the opening.
pub(crate) fn prove_bus<C: Challenger>(
    circuit: &Circuit,
    bus: &Bus,
    packed: &[F128],
    grinding: BatchedGrinding,
    ch: &mut C,
) -> (BusProof, Vec<PackedDirectClaim>) {
    let cells = circuit.cells();
    let nu = cells.nu();
    let rows = 1usize << nu;
    ch.observe_label(DOMAIN_BUS);
    let mut fingerprint_nonce = Vec::new();
    let challenges = if let Some(shared) = &bus.shared {
        observe_shared(shared, ch);
        Challenges {
            beta: shared.beta,
            gamma: shared.gamma.clone(),
        }
    } else {
        let beta = match grinding.fingerprint_target_for(live_entries(bus, circuit)) {
            Some(target) => {
                let (nonce, beta) = ch.grind_pow_below_and_sample_f128(target);
                fingerprint_nonce.push(nonce);
                beta
            }
            None => ch.sample_f128(),
        };
        let gamma: Vec<F128> = (0..bus.width).map(|_| ch.sample_f128()).collect();
        Challenges { beta, gamma }
    };
    let word = |iota: usize, j: usize| packed[cells.gate_word_addr(iota, j)];

    let groups = 1usize << bus.c_bits();
    let leaves = |dir: BusDirection| -> ProductLeaves {
        let gs = bus.groups(dir);
        let mut buf = take_f128(groups * rows);
        let lens: Vec<usize> = (0..groups)
            .map(|g| {
                gs.get(g)
                    .map_or(0, |&(p, _)| circuit.counts()[bus.ports[p].ty].min(rows))
            })
            .collect();
        // Parallel over groups and, within a group, chunks of rows.
        const CHUNK: usize = 1 << 13;
        buf[..groups * rows]
            .par_chunks_mut(rows)
            .zip(gs.par_iter())
            .zip(lens.par_iter())
            .for_each(|((group, &(p, bit)), &live)| {
                let port = &bus.ports[p];
                group[..live]
                    .par_chunks_mut(CHUNK)
                    .enumerate()
                    .for_each(|(c, out)| {
                        for (i, leaf) in out.iter_mut().enumerate() {
                            let j = c * CHUNK + i;
                            let f = fingerprint(port, j, &challenges, word);
                            *leaf = match bit {
                                None => f,
                                Some(b) => {
                                    F128::ONE + word(port.mult[b], j) * (F128::ONE + frob(f, b))
                                }
                            };
                        }
                    });
            });
        ProductLeaves { buf, lens, rows }
    };
    let lhs = leaves(BusDirection::Send);
    let rhs = leaves(BusDirection::Recv);
    let (gkr, claim) = prove_leaves(lhs, rhs, grinding, ch);
    debug_assert!(bus.shared.is_some() || gkr.top_lhs == gkr.top_rhs);

    let rho_row = &claim.rho[..nu];
    let eq_row = build_eq(rho_row);
    let words = bus.words();
    let gather: Vec<F128> = words
        .par_iter()
        .map(|&iota| column_at(circuit, iota, &eq_row, word))
        .collect();
    let mut claims: Vec<PackedDirectClaim> = words
        .iter()
        .zip(&gather)
        .map(|(&iota, &value)| direct_claim(circuit, iota, rho_row, value))
        .collect();
    let mult = bus.has_mult().then(|| {
        let (proof, opened) = prove_mult(circuit, bus, word, &challenges, &eq_row, ch);
        claims.extend(
            opened
                .into_iter()
                .map(|(iota, point, value)| direct_claim(circuit, iota, &point, value)),
        );
        proof
    });
    (
        BusProof {
            gkr,
            gather,
            fingerprint_nonce,
            mult,
        },
        claims,
    )
}

/// Replay the bus and check both products' leaf claims against the gather
/// values (and the multiplicity groups' sums, which their sumcheck proves).
/// Returns the claims `(point, value)` in proof order and the static
/// columns' evaluations the check read.
pub(crate) fn verify_bus<C: Challenger>(
    circuit: &Circuit,
    bus: &Bus,
    proof: &BusProof,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<(Vec<(Vec<F128>, F128)>, BusStructure, Option<MultStructure>), WiringError> {
    let cells = circuit.cells();
    let nu = cells.nu();
    let rows = 1usize << nu;
    let words = bus.words();
    if proof.gather.len() != words.len() || proof.mult.is_some() != bus.has_mult() {
        return Err(WiringError::MalformedProof);
    }
    ch.observe_label(DOMAIN_BUS);
    let challenges = if let Some(shared) = &bus.shared {
        if !proof.fingerprint_nonce.is_empty() || shared.gamma.len() != bus.width {
            return Err(WiringError::MalformedProof);
        }
        observe_shared(shared, ch);
        Challenges {
            beta: shared.beta,
            gamma: shared.gamma.clone(),
        }
    } else {
        let beta = if let Some(target) = grinding.fingerprint_target_for(live_entries(bus, circuit)) {
            let [nonce] = proof.fingerprint_nonce[..] else {
                return Err(WiringError::BusGkr(
                    crate::product_gkr::ProductGkrError::InvalidGrinding,
                ));
            };
            ch.verify_pow_below_and_sample_f128(nonce, target)
                .ok_or(WiringError::BusGkr(
                    crate::product_gkr::ProductGkrError::InvalidGrinding,
                ))?
        } else {
            if !proof.fingerprint_nonce.is_empty() {
                return Err(WiringError::MalformedProof);
            }
            ch.sample_f128()
        };
        let gamma: Vec<F128> = (0..bus.width).map(|_| ch.sample_f128()).collect();
        Challenges { beta, gamma }
    };

    let c = bus.c_bits();
    let claim = if bus.shared.is_some() {
        verify_leaves_open(nu + c, &proof.gkr, grinding, ch)
    } else {
        verify_leaves(nu + c, &proof.gkr, grinding, ch)
    }
    .map_err(WiringError::BusGkr)?;
    let (rho_row, rho_port) = claim.rho.split_at(nu);
    let eq_row = build_eq(rho_row);
    debug_assert_eq!(eq_row.len(), rows);
    let eq_port = build_eq(rho_port);
    let gathered = |iota: usize| proof.gather[words.binary_search(&iota).expect("a bus word")];
    let cols = bus.structure_columns();
    let values: Vec<F128> = cols
        .iter()
        .map(|c| c.eval(circuit, rho_row, &eq_row))
        .collect();
    let column = |col: BusColumn| values[Bus::column_index(&cols, col)];
    let mut opened = Vec::new();
    let mut mult_structure = None;
    let sums: &[F128] = match &proof.mult {
        Some(mult) => {
            let (o, m) = verify_mult(circuit, bus, mult, &challenges, rho_row, ch)?;
            opened = o;
            mult_structure = Some(m);
            &mult.sums
        }
        None => &[],
    };
    let mult_groups = bus.mult_groups();

    // Each side: 1 + Σ_g eq(ρ_port, g)·[(β + 1)·E_p + Σ_k γ_k·(coordinate
    // MLE)], or the group's sum S for a multiplicity bit.
    let side = |dir: BusDirection| -> F128 {
        let mut acc = F128::ONE;
        for (g, (p, bit)) in bus.groups(dir).into_iter().enumerate() {
            let port = &bus.ports[p];
            let inner = match bit {
                Some(i) => {
                    let at = mult_groups.iter().position(|&x| x == (p, i));
                    sums[at.expect("a multiplicity group")]
                }
                None => {
                    let mass = column(BusColumn::Live { ty: port.ty });
                    let mut inner = (challenges.beta + F128::ONE) * mass;
                    for (term, &gk) in port.terms.iter().zip(&challenges.gamma) {
                        let mut value = term.constant * mass;
                        if let Some(iota) = term.word {
                            value += gathered(iota);
                        }
                        if let Some(tag) = term.row {
                            value += column(BusColumn::Tag { ty: port.ty, tag });
                        }
                        inner += gk * value;
                    }
                    inner
                }
            };
            acc += eq_port[g] * inner;
        }
        acc
    };
    if side(BusDirection::Send) != claim.lhs || side(BusDirection::Recv) != claim.rhs {
        return Err(WiringError::BusRecombination);
    }
    let gather = words
        .iter()
        .zip(&proof.gather)
        .map(|(&iota, &value)| (cells.gate_claim_point(iota, rho_row), value))
        .chain(
            opened
                .into_iter()
                .map(|(iota, point, value)| (cells.gate_claim_point(iota, &point), value)),
        )
        .collect();
    Ok((
        gather,
        BusStructure {
            rho_row: rho_row.to_vec(),
            values,
        },
        mult_structure,
    ))
}

// ---------------------------------------------------------------------------
// Multiplicities: the groups' sumcheck
// ---------------------------------------------------------------------------

pub const DOMAIN_MULT: &[u8] = b"flock-tuple-bus-multiplicities-v1";

/// An opening the multiplicity sumcheck asks for: cell-slot, row point,
/// value.
type Opened = (usize, Vec<F128>, F128);

/// `x^{2^i}`.
#[inline]
pub fn frob(mut x: F128, i: usize) -> F128 {
    for _ in 0..i {
        x = x.square();
    }
    x
}

/// `x^{2^{-i}}`: Frobenius has order 128 on GF(2^128).
pub fn frob_inv(x: F128, i: usize) -> F128 {
    frob(x, (128 - i % 128) % 128)
}

/// The sumcheck's third node: the generator `x`; the nodes are `0, 1, x,
/// x + 1`.
pub const NODE: F128 = F128::new(2, 0);

/// The cubic through `(0, v0), (1, v1), (x, vx), (x + 1, vx1)` at `r`.
pub fn interpolate(v: [F128; 4], r: F128) -> F128 {
    let nodes = [F128::ZERO, F128::ONE, NODE, NODE + F128::ONE];
    let mut acc = F128::ZERO;
    for k in 0..4 {
        let (mut num, mut den) = (F128::ONE, F128::ONE);
        for l in (0..4).filter(|&l| l != k) {
            num *= r + nodes[l];
            den *= nodes[k] + nodes[l];
        }
        acc += v[k] * num * den.inv();
    }
    acc
}

/// `eq(a, b)` for two points.
pub fn eq_at(a: &[F128], b: &[F128]) -> F128 {
    a.iter().zip(b).fold(F128::ONE, |acc, (&x, &y)| {
        acc * (x * y + (F128::ONE + x) * (F128::ONE + y))
    })
}

/// The live rows of `iota`'s type.
fn live_of(circuit: &Circuit, iota: usize, rows: usize) -> usize {
    let CellSlot::Gate { ty, .. } = circuit.cells().slots()[iota] else {
        unreachable!("validated: bus words are gate cell-slots")
    };
    circuit.counts()[ty].min(rows)
}

/// `Σ_{j<live} eq[j]·word(iota, j)`: the column's MLE at the point `eq` was
/// built from.
fn column_at(
    circuit: &Circuit,
    iota: usize,
    eq: &[F128],
    word: impl Fn(usize, usize) -> F128,
) -> F128 {
    let live = live_of(circuit, iota, eq.len());
    eq[..live]
        .iter()
        .enumerate()
        .fold(F128::ZERO, |acc, (j, &e)| acc + e * word(iota, j))
}

fn direct_claim(circuit: &Circuit, iota: usize, row: &[F128], value: F128) -> PackedDirectClaim {
    let point = circuit.cells().gate_claim_point(iota, row);
    PackedDirectClaim {
        value,
        eq_ind: DirectEqInd::EqPoint(point.clone()),
        point,
    }
}

/// Row `j`'s fingerprint over all `2^ν` rows: the live rows' tuples, and
/// `β + Σ_k γ_k·c_k` past them (words and tags are zero there), so that its
/// MLE is affine in the words' MLEs.
fn fingerprint_everywhere(
    port: &BusPort,
    j: usize,
    live: usize,
    ch: &Challenges,
    word: impl Fn(usize, usize) -> F128,
) -> F128 {
    if j < live {
        return fingerprint(port, j, ch, word);
    }
    port.terms
        .iter()
        .zip(&ch.gamma)
        .fold(ch.beta, |acc, (t, &g)| acc + g * t.constant)
}

/// Sample the batching challenges `λ_g, μ_g` after the sums.
fn batching<C: Challenger>(sums: &[F128], ch: &mut C) -> (Vec<F128>, Vec<F128>) {
    ch.observe_label(DOMAIN_MULT);
    ch.observe_f128_slice(sums);
    let lambda = sums.iter().map(|_| ch.sample_f128()).collect();
    let mu = sums.iter().map(|_| ch.sample_f128()).collect();
    (lambda, mu)
}

/// Prove every multiplicity group's sum `S_g` and that its bits are 0 or 1:
/// `Σ_j eq(ρ_row, j)·Σ_g m_g(j)·[λ_g·(1 + F_g(j)) + μ_g·(1 + m_g(j))] =
/// Σ_g λ_g·S_g`, one degree-3 sumcheck binding the row bits low first.
fn prove_mult<C: Challenger>(
    circuit: &Circuit,
    bus: &Bus,
    word: impl Fn(usize, usize) -> F128 + Copy + Sync,
    challenges: &Challenges,
    eq_row: &[F128],
    ch: &mut C,
) -> (MultProof, Vec<Opened>) {
    let rows = eq_row.len();
    let groups = bus.mult_groups();
    // Per group: the bits m_g and F_g = f^{2^i} over all rows.
    let tables: Vec<(Vec<F128>, Vec<F128>)> = groups
        .par_iter()
        .map(|&(p, i)| {
            let port = &bus.ports[p];
            let live = circuit.counts()[port.ty].min(rows);
            let m: Vec<F128> = (0..rows)
                .map(|j| {
                    if j < live {
                        word(port.mult[i], j)
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            let f: Vec<F128> = (0..rows)
                .map(|j| frob(fingerprint_everywhere(port, j, live, challenges, word), i))
                .collect();
            (m, f)
        })
        .collect();
    let sums: Vec<F128> = tables
        .iter()
        .map(|(m, f)| {
            (0..rows).fold(F128::ZERO, |acc, j| {
                acc + eq_row[j] * m[j] * (F128::ONE + f[j])
            })
        })
        .collect();
    let (lambda, mu) = batching(&sums, ch);
    let mut claim = sums
        .iter()
        .zip(&lambda)
        .fold(F128::ZERO, |acc, (&s, &l)| acc + l * s);
    let mut e = eq_row.to_vec();
    let (mut ms, mut gs): (Vec<Vec<F128>>, Vec<Vec<F128>>) = tables
        .into_iter()
        .zip(lambda.iter().zip(&mu))
        .map(|((m, f), (&l, &u))| {
            let g = (0..rows)
                .map(|j| l * (F128::ONE + f[j]) + u * (F128::ONE + m[j]))
                .collect();
            (m, g)
        })
        .unzip();

    let nodes = [F128::ZERO, NODE, NODE + F128::ONE];
    let mut rounds = Vec::new();
    let mut r = Vec::new();
    while e.len() > 1 {
        let half = e.len() / 2;
        let s: [F128; 3] = (0..half)
            .into_par_iter()
            .map(|t| {
                let mut out = [F128::ZERO; 3];
                for (o, &x) in out.iter_mut().zip(&nodes) {
                    let at = |v: &[F128]| v[2 * t] + x * (v[2 * t] + v[2 * t + 1]);
                    let inner = ms
                        .iter()
                        .zip(&gs)
                        .fold(F128::ZERO, |acc, (m, g)| acc + at(m) * at(g));
                    *o = at(&e) * inner;
                }
                out
            })
            .reduce(|| [F128::ZERO; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
        ch.observe_f128_slice(&s);
        let rk = ch.sample_f128();
        claim = interpolate([s[0], claim + s[0], s[1], s[2]], rk);
        rounds.push(s);
        r.push(rk);
        let fold = |v: &mut Vec<F128>| {
            *v = (0..half)
                .map(|t| v[2 * t] + rk * (v[2 * t] + v[2 * t + 1]))
                .collect();
        };
        fold(&mut e);
        ms.iter_mut().for_each(fold);
        gs.iter_mut().for_each(fold);
    }
    // The claim closes on e[0]·Σ_g m_g[0]·g_g[0] exactly when every bit is
    // 0 or 1; otherwise the verifier rejects, which the tests rely on.

    let mut opened = Vec::new();
    let eq_r = build_eq(&r);
    let bits: Vec<F128> = bus
        .mult_bits()
        .into_iter()
        .map(|iota| {
            let v = column_at(circuit, iota, &eq_r, word);
            opened.push((iota, r.clone(), v));
            v
        })
        .collect();
    let mut sigma: Vec<(Vec<F128>, Vec<F128>)> = Vec::new();
    let words: Vec<F128> = bus
        .mult_words()
        .into_iter()
        .map(|(i, iota)| {
            while sigma.len() <= i {
                let point: Vec<F128> = r.iter().map(|&x| frob_inv(x, sigma.len())).collect();
                let eq = build_eq(&point);
                sigma.push((point, eq));
            }
            let v = column_at(circuit, iota, &sigma[i].1, word);
            opened.push((iota, sigma[i].0.clone(), v));
            v
        })
        .collect();
    ch.observe_f128_slice(&bits);
    ch.observe_f128_slice(&words);
    (
        MultProof {
            sums,
            rounds,
            bits,
            words,
        },
        opened,
    )
}

/// Replay [`prove_mult`]: the rounds, then the end point's value from the
/// opened bits and words. Returns the openings, which the PCS checks.
fn verify_mult<C: Challenger>(
    circuit: &Circuit,
    bus: &Bus,
    proof: &MultProof,
    challenges: &Challenges,
    rho_row: &[F128],
    ch: &mut C,
) -> Result<(Vec<Opened>, MultStructure), WiringError> {
    let groups = bus.mult_groups();
    let bit_slots = bus.mult_bits();
    let word_slots = bus.mult_words();
    if proof.sums.len() != groups.len()
        || proof.rounds.len() != rho_row.len()
        || proof.bits.len() != bit_slots.len()
        || proof.words.len() != word_slots.len()
    {
        return Err(WiringError::MalformedProof);
    }
    let (lambda, mu) = batching(&proof.sums, ch);
    let mut claim = proof
        .sums
        .iter()
        .zip(&lambda)
        .fold(F128::ZERO, |acc, (&s, &l)| acc + l * s);
    let mut r = Vec::with_capacity(rho_row.len());
    for s in &proof.rounds {
        ch.observe_f128_slice(s);
        let rk = ch.sample_f128();
        claim = interpolate([s[0], claim + s[0], s[1], s[2]], rk);
        r.push(rk);
    }
    ch.observe_f128_slice(&proof.bits);
    ch.observe_f128_slice(&proof.words);

    let most = bus.ports.iter().map(|p| p.mult.len()).max().unwrap_or(0);
    let sigma: Vec<Vec<F128>> = (0..most)
        .map(|i| r.iter().map(|&x| frob_inv(x, i)).collect())
        .collect();
    let eq_sigma: Vec<Vec<F128>> = sigma.iter().map(|s| build_eq(s)).collect();
    // The row-tag columns at each σ_i: static, so deferred in recursion.
    let cols = bus.table_columns();
    let tag_cols = bus.mult_tag_columns();
    let tag_values: Vec<F128> = tag_cols
        .iter()
        .map(|&(i, c)| cols[c].eval(circuit, &sigma[i], &eq_sigma[i]))
        .collect();
    let tag_at = |i: usize, col: BusColumn| {
        tag_values[tag_cols
            .binary_search(&(i, Bus::column_index(&cols, col)))
            .expect("a multiplicity tag column")]
    };
    let bit = |iota: usize| proof.bits[bit_slots.binary_search(&iota).expect("a bit")];
    let word = |i: usize, iota: usize| {
        proof.words[word_slots
            .binary_search(&(i, iota))
            .expect("a multiplicity word")]
    };
    let mut total = F128::ZERO;
    for (g, &(p, i)) in groups.iter().enumerate() {
        let port = &bus.ports[p];
        let m = bit(port.mult[i]);
        // f̂(σ_i) = β + Σ_k γ_k·(c_k + ŵ_k(σ_i) + T̂_k(σ_i)).
        let mut f = challenges.beta;
        for (term, &gk) in port.terms.iter().zip(&challenges.gamma) {
            let mut value = term.constant;
            if let Some(iota) = term.word {
                value += word(i, iota);
            }
            if let Some(tag) = term.row {
                value += tag_at(i, BusColumn::Tag { ty: port.ty, tag });
            }
            f += gk * value;
        }
        let big_f = frob(f, i);
        total += m * (lambda[g] * (F128::ONE + big_f) + mu[g] * (F128::ONE + m));
    }
    if eq_at(rho_row, &r) * total != claim {
        return Err(WiringError::BusRecombination);
    }
    let mut opened: Vec<Opened> = bit_slots
        .iter()
        .zip(&proof.bits)
        .map(|(&iota, &v)| (iota, r.clone(), v))
        .collect();
    opened.extend(
        word_slots
            .iter()
            .zip(&proof.words)
            .map(|(&(i, iota), &v)| (iota, sigma[i].clone(), v)),
    );
    Ok((
        opened,
        MultStructure {
            sigma,
            columns: tag_cols,
            values: tag_values,
        },
    ))
}

/// Row `j`'s leaf in `port`: `β + Σ_k γ_k·t_k`, reading committed words
/// through `word(cell_slot, row)`.
#[inline]
fn fingerprint(
    port: &BusPort,
    j: usize,
    ch: &Challenges,
    word: impl Fn(usize, usize) -> F128,
) -> F128 {
    let mut acc = ch.beta;
    for (term, &g) in port.terms.iter().zip(&ch.gamma) {
        let mut t = term.constant;
        if let Some(iota) = term.word {
            t += word(iota, j);
        }
        if let Some(tag) = term.row {
            t += tag.value(j);
        }
        acc += g * t;
    }
    acc
}

/// The tuple of `port`'s row `j` — what the fingerprint encodes. For
/// witness-side balance checks.
pub fn tuple(port: &BusPort, j: usize, word: impl Fn(usize, usize) -> F128) -> Vec<F128> {
    port.terms
        .iter()
        .map(|term| {
            let mut t = term.constant;
            if let Some(iota) = term.word {
                t += word(iota, j);
            }
            if let Some(tag) = term.row {
                t += tag.value(j);
            }
            t
        })
        .collect()
}

/// How many times `port`'s row `j` counts: its multiplicity bits read as an
/// integer (a bit word other than 0 or 1 is `None`), 1 without bits. For
/// witness-side balance checks.
pub fn multiplicity(port: &BusPort, j: usize, word: impl Fn(usize, usize) -> F128) -> Option<u64> {
    if port.mult.is_empty() {
        return Some(1);
    }
    port.mult.iter().enumerate().try_fold(0u64, |acc, (i, &iota)| {
        let b = word(iota, j);
        if b == F128::ZERO {
            Some(acc)
        } else if b == F128::ONE {
            Some(acc | 1 << i)
        } else {
            None
        }
    })
}
