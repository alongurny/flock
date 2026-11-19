//! Verification keys for SPARSE fixed tables (a circuit's constraint
//! matrices), in the style of Spark (Spartan's sparse commitment).
//!
//! A root claim on a sparse `M` reads `Σ_{r,c} R(r)·C(c)·M[r,c] = v` with
//! tensor weights `R`, `C`. Over the nonzero entries `k` (row `a_k`,
//! column `b_k`, value `m_k`) that is `Σ_k m_k·R(a_k)·C(b_k)`. The key
//! commits, once, to the entries and to their memory-checking timestamps;
//! a claim is opened by
//!
//! 1. committing `e_R[k] = R(a_k)`, `e_C[k] = C(b_k)` (fresh, per claim),
//! 2. a degree-3 sumcheck `Σ_k m_k·e_R[k]·e_C[k] = v`,
//! 3. offline memory checking that every `e_R[k]` is `R` read at `a_k`
//!    (and likewise `e_C`): the multisets `Init ∪ Writes` and
//!    `Reads ∪ Final` agree, as one grand product
//!    ([`crate::product_gkr::prove_leaves`]), with tuples
//!    `(address, value, timestamp)` fingerprinted `a + γ·v + γ²·t + τ`,
//! 4. dense openings ([`crate::table_key`]) of the key and of the fresh
//!    commitment at the points steps 2 and 3 land on.
//!
//! The verifier evaluates `R` and `C` (tensor weights) and the address tag
//! itself, so it never reads `M`. Timestamps are the key's (an honest
//! preprocessing of `M`'s access pattern): the write timestamps are
//! committed beside the read timestamps instead of derived from them, and
//! every address and timestamp is encoded by its bit pattern (`s_id`).

use serde::{Deserialize, Serialize};

use crate::{
    challenger::Challenger,
    field::F128,
    matrix_fold::{MatrixClaim, Weight},
    product_gkr::{BatchedGrinding, ProductGkrLeavesProof, ProductLeaves, prove_leaves, verify_leaves},
    table_key::{DenseKey, DenseOpening, DenseTable, TensorBasis, verify_dense_many},
};

/// Key vectors, each `size` long, as the columns of the dense key table.
const ROW: usize = 0;
const COL: usize = 1;
const RTS_R: usize = 2;
const WTS_R: usize = 3;
const FTS_R: usize = 4;
const RTS_C: usize = 5;
const WTS_C: usize = 6;
const FTS_C: usize = 7;
const VAL: usize = 8;
const SLOTS_LOG: usize = 4;

/// The public key of one sparse table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseKey {
    pub rows_log: usize,
    pub cols_log: usize,
    /// The key holds `2^mats_log` slots for matrices (a union: matrix `m`'s
    /// row `r` is address `(m << rows_log) | r`), opened together.
    pub mats_log: usize,
    /// How many matrices the key holds.
    pub n_mats: usize,
    /// `log2` of the common length of the entry and address vectors.
    pub size_log: usize,
    pub dense: DenseKey,
}

/// The prover's side of a sparse key.
pub struct SparseTable {
    pub key: SparseKey,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<F128>,
    dense: DenseTable,
}

/// An opening of one claim against a sparse key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseOpening {
    pub fresh: DenseKey,
    pub rounds: Vec<[F128; 4]>,
    pub gkr: ProductGkrLeavesProof,
    /// At the memory-checking point: the eight key vectors, then `e_R`,
    /// `e_C`. At the sumcheck point: `m`, `e_R`, `e_C`.
    pub at_rho: [F128; 10],
    pub at_z: [F128; 3],
    pub key_open: DenseOpening,
    pub fresh_open: DenseOpening,
}

fn enc(i: usize) -> F128 {
    F128::new(i as u64, 0)
}

/// The `s_id` tag's MLE at `z`: `Σ_i 2^i·z_i` as a field element.
fn enc_eval(z: &[F128]) -> F128 {
    z.iter().enumerate().fold(F128::ZERO, |a, (i, &r)| {
        let b = if i < 64 { F128::new(1u64 << i, 0) } else { F128::new(0, 1u64 << (i - 64)) };
        a + b * r
    })
}

/// The four sumcheck evaluation points.
fn points() -> [F128; 4] {
    let x = F128::new(2, 0);
    [F128::ZERO, F128::ONE, x, x + F128::ONE]
}

/// Lagrange interpolation of a degree-3 polynomial from its values at
/// [`points`], evaluated at `r`.
fn interpolate(ys: &[F128; 4], r: F128) -> F128 {
    let xs = points();
    let mut out = F128::ZERO;
    for i in 0..4 {
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..4 {
            if j != i {
                num *= r + xs[j];
                den *= xs[i] + xs[j];
            }
        }
        out += ys[i] * num * den.inv();
    }
    out
}

fn fold(v: &mut Vec<F128>, r: F128) {
    *v = (0..v.len() / 2)
        .map(|i| v[2 * i] + r * (v[2 * i] + v[2 * i + 1]))
        .collect();
}

/// A weight as a basis over exactly `bits` bits (a narrower matrix pads
/// with zeros: its entries there are zero).
fn weight_basis(w: &Weight, bits: usize) -> Option<TensorBasis> {
    let (s, f) = TensorBasis::weight_factors(w, bits, true)?;
    Some(TensorBasis::new(s, f))
}

fn slot_weight(slot: usize, bits: usize) -> Weight {
    Weight::eq((0..bits).map(|j| if (slot >> j) & 1 == 1 { F128::ONE } else { F128::ZERO }).collect())
}

/// The key-vector claims at `rho` (eight) and `z` (the values).
fn key_claims(rho: &[F128], z: &[F128], at_rho: &[F128; 10], at_z: &[F128; 3]) -> Vec<MatrixClaim> {
    let mut v: Vec<MatrixClaim> = (0..8)
        .map(|s| MatrixClaim {
            row: Weight::eq(rho.to_vec()),
            col: slot_weight(s, SLOTS_LOG),
            value: at_rho[s],
        })
        .collect();
    v.push(MatrixClaim {
        row: Weight::eq(z.to_vec()),
        col: slot_weight(VAL, SLOTS_LOG),
        value: at_z[0],
    });
    v
}

fn fresh_claims(rho: &[F128], z: &[F128], at_rho: &[F128; 10], at_z: &[F128; 3]) -> Vec<MatrixClaim> {
    let mut v = Vec::new();
    for s in 0..2 {
        v.push(MatrixClaim {
            row: Weight::eq(rho.to_vec()),
            col: slot_weight(s, 1),
            value: at_rho[8 + s],
        });
        v.push(MatrixClaim {
            row: Weight::eq(z.to_vec()),
            col: slot_weight(s, 1),
            value: at_z[1 + s],
        });
    }
    v
}

/// Bind the key and every claim; draw `ρ`, which combines the matrices'
/// claims into one (`Σ ρ^m·v_m`).
fn bind<Ch: Challenger>(ch: &mut Ch, key: &SparseKey, claims: &[MatrixClaim]) -> F128 {
    ch.observe_label(b"flock-table-key-sparse-v0");
    for x in [key.rows_log, key.cols_log, key.mats_log, key.n_mats, key.size_log] {
        ch.observe_bytes(&(x as u64).to_le_bytes());
    }
    for h in &key.dense.cap {
        ch.observe_bytes(h.as_ref());
    }
    for claim in claims {
        for w in [&claim.row, &claim.col] {
            ch.observe_bytes(&(w.low.len() as u64).to_le_bytes());
            ch.observe_f128_slice(&w.low);
            ch.observe_bytes(&(w.point.len() as u64).to_le_bytes());
            ch.observe_f128_slice(&w.point);
        }
        ch.observe_f128(claim.value);
    }
    ch.sample_f128()
}

/// Per matrix, its row and column weights as bases over the matrix's own
/// `rows_log` / `cols_log` bits. `None` if a claim does not fit.
fn claim_bases(key: &SparseKey, claims: &[MatrixClaim]) -> Option<Vec<(TensorBasis, TensorBasis)>> {
    if claims.len() != key.n_mats {
        return None;
    }
    claims
        .iter()
        .map(|c| {
            Some((
                weight_basis(&c.row, key.rows_log)?,
                weight_basis(&c.col, key.cols_log)?,
            ))
        })
        .collect()
}

/// The two memories' contents: `T_R[(m << rows_log) | r] = R_m(r)`,
/// `T_C[(m << cols_log) | c] = ρ^m·C_m(c)`, zero elsewhere.
fn memories(key: &SparseKey, bases: &[(TensorBasis, TensorBasis)], rho: F128) -> (Vec<F128>, Vec<F128>) {
    let size = 1usize << key.size_log;
    let mut t_r = vec![F128::ZERO; size];
    let mut t_c = vec![F128::ZERO; size];
    let mut p = F128::ONE;
    for (m, (rb, cb)) in bases.iter().enumerate() {
        for (r, x) in rb.materialize().into_iter().enumerate() {
            t_r[(m << key.rows_log) | r] = x;
        }
        for (c, x) in cb.materialize().into_iter().enumerate() {
            t_c[(m << key.cols_log) | c] = p * x;
        }
        p *= rho;
    }
    (t_r, t_c)
}

/// The verifier's MLEs of the two memories at `p`.
fn memories_at(key: &SparseKey, bases: &[(TensorBasis, TensorBasis)], rho: F128, p: &[F128]) -> (F128, F128) {
    let part = |bits: usize, m: usize| {
        let sel = p[bits..bits + key.mats_log]
            .iter()
            .enumerate()
            .fold(F128::ONE, |a, (j, &x)| a * if (m >> j) & 1 == 1 { x } else { F128::ONE + x });
        let rest = p[bits + key.mats_log..].iter().fold(F128::ONE, |a, &x| a * (F128::ONE + x));
        sel * rest
    };
    let (mut r_hat, mut c_hat) = (F128::ZERO, F128::ZERO);
    let mut q = F128::ONE;
    for (m, (rb, cb)) in bases.iter().enumerate() {
        r_hat += part(key.rows_log, m) * rb.eval_at(&p[..key.rows_log]);
        c_hat += q * part(key.cols_log, m) * cb.eval_at(&p[..key.cols_log]);
        q *= rho;
    }
    (r_hat, c_hat)
}

fn combined(claims: &[MatrixClaim], rho: F128) -> F128 {
    let mut p = F128::ONE;
    claims.iter().fold(F128::ZERO, |a, c| {
        let x = a + p * c.value;
        p *= rho;
        x
    })
}

/// `(γ, τ)` after the sumcheck.
fn fingerprint<Ch: Challenger>(ch: &mut Ch) -> (F128, F128) {
    ch.observe_label(b"flock-table-key-sparse-memory");
    (ch.sample_f128(), ch.sample_f128())
}

/// The verifier's leaf MLEs at `rho = (row bits, two group bits)`.
#[allow(clippy::too_many_arguments)]
fn leaf_evals(
    rho: &[F128],
    at: &[F128; 10],
    r_hat: F128,
    c_hat: F128,
    gamma: F128,
    tau: F128,
) -> (F128, F128) {
    let n = rho.len() - 2;
    let p = &rho[..n];
    let (g0, g1) = (rho[n], rho[n + 1]);
    let w = [
        (F128::ONE + g0) * (F128::ONE + g1),
        g0 * (F128::ONE + g1),
        (F128::ONE + g0) * g1,
        g0 * g1,
    ];
    let a = enc_eval(p);
    let g2 = gamma * gamma;
    let h = |addr: F128, v: F128, t: F128| addr + gamma * v + g2 * t + tau;
    let (e_r, e_c) = (at[8], at[9]);
    let lhs = [
        h(a, r_hat, F128::ZERO),
        h(at[ROW], e_r, at[WTS_R]),
        h(a, c_hat, F128::ZERO),
        h(at[COL], e_c, at[WTS_C]),
    ];
    let rhs = [
        h(a, r_hat, at[FTS_R]),
        h(at[ROW], e_r, at[RTS_R]),
        h(a, c_hat, at[FTS_C]),
        h(at[COL], e_c, at[RTS_C]),
    ];
    let dot = |x: [F128; 4]| x.iter().zip(&w).fold(F128::ZERO, |s, (a, b)| s + *a * *b);
    (dot(lhs), dot(rhs))
}

impl SparseTable {
    /// Commit to `M` given by its nonzero entries `(row, col, value)`.
    pub fn new(entries: &[(usize, usize, F128)], rows_log: usize, cols_log: usize) -> Option<Self> {
        Self::new_many(&[entries], rows_log, cols_log)
    }

    /// Commit to several matrices at once (each at most `2^rows_log ×
    /// 2^cols_log`), opened together by one claim per matrix.
    pub fn new_many(mats: &[&[(usize, usize, F128)]], rows_log: usize, cols_log: usize) -> Option<Self> {
        let n_rows = 1usize << rows_log;
        let n_cols = 1usize << cols_log;
        if mats.is_empty() || mats.iter().flat_map(|e| e.iter()).any(|&(r, c, _)| r >= n_rows || c >= n_cols) {
            return None;
        }
        let mats_log = mats.len().next_power_of_two().trailing_zeros() as usize;
        let total: usize = mats.iter().map(|e| e.len()).sum();
        let size = (n_rows << mats_log)
            .max(n_cols << mats_log)
            .max(total)
            .next_power_of_two()
            .max(2);
        let size_log = size.trailing_zeros() as usize;
        let mut table = vec![F128::ZERO; size << SLOTS_LOG];
        let mut col = |s: usize, i: usize, x: F128| table[(s << size_log) | i] = x;
        let mut reads_r = vec![0usize; size];
        let mut reads_c = vec![0usize; size];
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let addressed = mats.iter().enumerate().flat_map(|(m, e)| {
            e.iter().map(move |&(r, c, v)| ((m << rows_log) | r, (m << cols_log) | c, v))
        });
        for (k, (r, c, v)) in addressed.enumerate() {
            col(ROW, k, enc(r));
            col(COL, k, enc(c));
            col(RTS_R, k, enc(reads_r[r]));
            col(WTS_R, k, enc(reads_r[r] + 1));
            col(RTS_C, k, enc(reads_c[c]));
            col(WTS_C, k, enc(reads_c[c] + 1));
            col(VAL, k, v);
            reads_r[r] += 1;
            reads_c[c] += 1;
            rows.push(r);
            cols.push(c);
            vals.push(v);
        }
        for i in 0..size {
            col(FTS_R, i, enc(reads_r[i]));
            col(FTS_C, i, enc(reads_c[i]));
        }
        let dense = DenseTable::new(&table, size_log)?;
        Some(Self {
            key: SparseKey {
                rows_log,
                cols_log,
                mats_log,
                n_mats: mats.len(),
                size_log,
                dense: dense.key.clone(),
            },
            rows,
            cols,
            vals,
            dense,
        })
    }

    /// Prove `claim` (a key of one matrix). `None` if its weights do not
    /// fit or it is false.
    pub fn open<Ch: Challenger>(&self, claim: &MatrixClaim, ch: &mut Ch) -> Option<SparseOpening> {
        self.open_many(std::slice::from_ref(claim), ch)
    }

    /// Prove one claim per matrix with one opening. `None` if a claim does
    /// not fit or is false.
    pub fn open_many<Ch: Challenger>(&self, claims: &[MatrixClaim], ch: &mut Ch) -> Option<SparseOpening> {
        let k = &self.key;
        let size = 1usize << k.size_log;
        let bases = claim_bases(k, claims)?;
        // Every claim must hold on its own: the honest prover refuses a
        // false one outright.
        let tabs: Vec<(Vec<F128>, Vec<F128>)> =
            bases.iter().map(|(rb, cb)| (rb.materialize(), cb.materialize())).collect();
        let (rmask, cmask) = ((1usize << k.rows_log) - 1, (1usize << k.cols_log) - 1);
        let mut per = vec![F128::ZERO; claims.len()];
        for ((&r, &c), &v) in self.rows.iter().zip(&self.cols).zip(&self.vals) {
            let m = r >> k.rows_log;
            per[m] += v * tabs[m].0[r & rmask] * tabs[m].1[c & cmask];
        }
        if per.iter().zip(claims).any(|(x, c)| *x != c.value) {
            return None;
        }
        let rho_m = bind(ch, k, claims);
        let (r_tab, c_tab) = memories(k, &bases, rho_m);
        let mut e_r = vec![F128::ZERO; size];
        let mut e_c = vec![F128::ZERO; size];
        let mut m = vec![F128::ZERO; size];
        for (i, ((&r, &c), &v)) in self.rows.iter().zip(&self.cols).zip(&self.vals).enumerate() {
            e_r[i] = r_tab[r];
            e_c[i] = c_tab[c];
            m[i] = v;
        }
        let mut fresh_vals = e_r.clone();
        fresh_vals.extend_from_slice(&e_c);
        let fresh = DenseTable::new(&fresh_vals, k.size_log)?;
        for h in &fresh.key.cap {
            ch.observe_bytes(h.as_ref());
        }

        // ---- the sumcheck Σ_k m·e_R·e_C ----
        let (mut fm, mut fr, mut fc) = (m.clone(), e_r.clone(), e_c.clone());
        let mut rounds = Vec::with_capacity(k.size_log);
        let mut z = Vec::with_capacity(k.size_log);
        for _ in 0..k.size_log {
            let mut ys = [F128::ZERO; 4];
            for (y, &t) in ys.iter_mut().zip(&points()) {
                for i in 0..fm.len() / 2 {
                    let at = |v: &[F128]| v[2 * i] + t * (v[2 * i] + v[2 * i + 1]);
                    *y += at(&fm) * at(&fr) * at(&fc);
                }
            }
            ch.observe_f128_slice(&ys);
            let r = ch.sample_f128();
            fold(&mut fm, r);
            fold(&mut fr, r);
            fold(&mut fc, r);
            rounds.push(ys);
            z.push(r);
        }
        let at_z = [fm[0], fr[0], fc[0]];
        ch.observe_f128_slice(&at_z);

        // ---- memory checking ----
        let (gamma, tau) = fingerprint(ch);
        let g2 = gamma * gamma;
        let h = |a: F128, v: F128, t: F128| a + gamma * v + g2 * t + tau;
        let key_col = |s: usize, i: usize| self.dense_val(s, i);
        let mut lhs = vec![F128::ZERO; 4 * size];
        let mut rhs = vec![F128::ZERO; 4 * size];
        for i in 0..size {
            let a = enc(i);
            lhs[i] = h(a, r_tab[i], F128::ZERO);
            rhs[i] = h(a, r_tab[i], key_col(FTS_R, i));
            lhs[size + i] = h(key_col(ROW, i), e_r[i], key_col(WTS_R, i));
            rhs[size + i] = h(key_col(ROW, i), e_r[i], key_col(RTS_R, i));
            lhs[2 * size + i] = h(a, c_tab[i], F128::ZERO);
            rhs[2 * size + i] = h(a, c_tab[i], key_col(FTS_C, i));
            lhs[3 * size + i] = h(key_col(COL, i), e_c[i], key_col(WTS_C, i));
            rhs[3 * size + i] = h(key_col(COL, i), e_c[i], key_col(RTS_C, i));
        }
        let side = |buf| ProductLeaves {
            buf,
            lens: vec![size; 4],
            rows: size,
        };
        let (gkr, leaves) = prove_leaves(side(lhs), side(rhs), BatchedGrinding::disabled(), ch);
        let rho = leaves.rho.clone();
        let p = &rho[..k.size_log];
        let mle = |v: &[F128]| {
            let mut v = v.to_vec();
            for &r in p {
                fold(&mut v, r);
            }
            v[0]
        };
        let mut at_rho = [F128::ZERO; 10];
        for (s, x) in at_rho.iter_mut().enumerate().take(8) {
            *x = mle(&(0..size).map(|i| key_col(s, i)).collect::<Vec<_>>());
        }
        at_rho[8] = mle(&e_r);
        at_rho[9] = mle(&e_c);
        ch.observe_f128_slice(&at_rho);

        // ---- the openings ----
        let key_open = self.dense.open_many(&key_claims(p, &z, &at_rho, &at_z), ch)?;
        let fresh_open = fresh.open_many(&fresh_claims(p, &z, &at_rho, &at_z), ch)?;
        Some(SparseOpening {
            fresh: fresh.key.clone(),
            rounds,
            gkr,
            at_rho,
            at_z,
            key_open,
            fresh_open,
        })
    }

    fn dense_val(&self, slot: usize, i: usize) -> F128 {
        self.dense.val((slot << self.key.size_log) | i)
    }
}

/// Check an opening of `claim` against a one-matrix `key`. Reads no
/// table entry.
pub fn verify_sparse<Ch: Challenger>(
    key: &SparseKey,
    claim: &MatrixClaim,
    o: &SparseOpening,
    ch: &mut Ch,
) -> bool {
    verify_sparse_many(key, std::slice::from_ref(claim), o, ch)
}

/// Check one opening of one claim per matrix ([`SparseTable::open_many`]).
pub fn verify_sparse_many<Ch: Challenger>(
    key: &SparseKey,
    claims: &[MatrixClaim],
    o: &SparseOpening,
    ch: &mut Ch,
) -> bool {
    let n = key.size_log;
    let Some(bases) = claim_bases(key, claims) else {
        return false;
    };
    if o.rounds.len() != n
        || o.fresh.rows_log != n
        || o.fresh.cols_log != 1
        || key.dense.rows_log != n
        || key.dense.cols_log != SLOTS_LOG
    {
        return false;
    }
    let rho_m = bind(ch, key, claims);
    for h in &o.fresh.cap {
        ch.observe_bytes(h.as_ref());
    }
    let mut run = combined(claims, rho_m);
    let mut z = Vec::with_capacity(n);
    for ys in &o.rounds {
        if ys[0] + ys[1] != run {
            return false;
        }
        ch.observe_f128_slice(ys);
        let r = ch.sample_f128();
        run = interpolate(ys, r);
        z.push(r);
    }
    if o.at_z[0] * o.at_z[1] * o.at_z[2] != run {
        return false;
    }
    ch.observe_f128_slice(&o.at_z);
    let (gamma, tau) = fingerprint(ch);
    let Ok(leaves) = verify_leaves(n + 2, &o.gkr, BatchedGrinding::disabled(), ch) else {
        return false;
    };
    let rho = &leaves.rho;
    let p = &rho[..n];
    let (r_hat, c_hat) = memories_at(key, &bases, rho_m, p);
    let (l, r) = leaf_evals(rho, &o.at_rho, r_hat, c_hat, gamma, tau);
    if l != leaves.lhs || r != leaves.rhs {
        return false;
    }
    ch.observe_f128_slice(&o.at_rho);
    verify_dense_many(&key.dense, &key_claims(p, &z, &o.at_rho, &o.at_z), &o.key_open, ch)
        && verify_dense_many(&o.fresh, &fresh_claims(p, &z, &o.at_rho, &o.at_z), &o.fresh_open, ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::{FsChallenger, RandomChallenger};

    fn setup(seed: u64) -> (RandomChallenger, Vec<(usize, usize, F128)>) {
        let mut rng = RandomChallenger::new(seed);
        let mut entries = Vec::new();
        for i in 0..900usize {
            let r = (i * 37 + 11) % 256;
            let c = (i * 101 + 5) % 128;
            let v = if i % 3 == 0 { F128::ONE } else { rng.sample_f128() };
            entries.push((r, c, v));
        }
        (rng, entries)
    }

    fn honest(entries: &[(usize, usize, F128)], row: &Weight, col: &Weight) -> F128 {
        let (rw, cw) = (row.materialize(), col.materialize());
        entries.iter().fold(F128::ZERO, |a, &(r, c, v)| a + v * rw[r] * cw[c])
    }

    #[test]
    fn sparse_key_opens_and_verifies() {
        let (mut rng, entries) = setup(0x5BA2_7E57);
        let t = SparseTable::new(&entries, 8, 7).unwrap();
        for (lr, lc) in [(0usize, 0usize), (2, 1)] {
            let row = Weight::low_eq(
                (0..1 << lr).map(|_| rng.sample_f128()).collect(),
                (0..8 - lr).map(|_| rng.sample_f128()).collect(),
            );
            let col = Weight::low_eq(
                (0..1 << lc).map(|_| rng.sample_f128()).collect(),
                (0..7 - lc).map(|_| rng.sample_f128()).collect(),
            );
            let value = honest(&entries, &row, &col);
            let c = MatrixClaim { row, col, value };
            let o = t.open(&c, &mut FsChallenger::new(b"s")).expect("an honest claim opens");
            assert!(verify_sparse(&t.key, &c, &o, &mut FsChallenger::new(b"s")), "honest at {lr} {lc}");

            let mut bad = c.clone();
            bad.value += F128::ONE;
            assert!(!verify_sparse(&t.key, &bad, &o, &mut FsChallenger::new(b"s")));
            assert!(t.open(&bad, &mut FsChallenger::new(b"s")).is_none());
        }
    }

    #[test]
    fn sparse_key_opens_several_matrices_at_once() {
        let (mut rng, a) = setup(0x3A75_0001);
        let (_, b) = setup(0x3A75_0002);
        let c: Vec<(usize, usize, F128)> = a.iter().take(300).map(|&(r, c, v)| (c, r % 128, v + F128::ONE)).collect();
        let mats: [&[(usize, usize, F128)]; 3] = [&a, &b[..500], &c];
        let t = SparseTable::new_many(&mats, 8, 7).unwrap();
        assert_eq!((t.key.mats_log, t.key.n_mats), (2, 3));
        let claims: Vec<MatrixClaim> = mats
            .iter()
            .map(|e| {
                let row = Weight::eq((0..8).map(|_| rng.sample_f128()).collect());
                let col = Weight::low_eq(vec![rng.sample_f128(), rng.sample_f128()], (0..6).map(|_| rng.sample_f128()).collect());
                let value = honest(e, &row, &col);
                MatrixClaim { row, col, value }
            })
            .collect();
        let o = t.open_many(&claims, &mut FsChallenger::new(b"u")).unwrap();
        assert!(verify_sparse_many(&t.key, &claims, &o, &mut FsChallenger::new(b"u")));
        for i in 0..3 {
            let mut bad = claims.clone();
            bad[i].value += F128::ONE;
            assert!(!verify_sparse_many(&t.key, &bad, &o, &mut FsChallenger::new(b"u")), "claim {i} changed");
            assert!(t.open_many(&bad, &mut FsChallenger::new(b"u")).is_none());
        }
        let mut swapped = claims.clone();
        swapped.swap(0, 2);
        assert!(!verify_sparse_many(&t.key, &swapped, &o, &mut FsChallenger::new(b"u")));
        assert!(!verify_sparse_many(&t.key, &claims[..2], &o, &mut FsChallenger::new(b"u")));
    }

    #[test]
    fn sparse_key_rejects_tampering() {
        let (mut rng, entries) = setup(0x7A3B_E2ED);
        let t = SparseTable::new(&entries, 8, 7).unwrap();
        let row = Weight::eq((0..8).map(|_| rng.sample_f128()).collect());
        let col = Weight::eq((0..7).map(|_| rng.sample_f128()).collect());
        let value = honest(&entries, &row, &col);
        let c = MatrixClaim { row, col, value };
        let o = t.open(&c, &mut FsChallenger::new(b"t")).unwrap();
        let ok = |o: &SparseOpening| verify_sparse(&t.key, &c, o, &mut FsChallenger::new(b"t"));
        assert!(ok(&o));
        for i in 0..10 {
            let mut b = o.clone();
            b.at_rho[i] += F128::ONE;
            assert!(!ok(&b), "at_rho[{i}] changed");
        }
        for i in 0..3 {
            let mut b = o.clone();
            b.at_z[i] += F128::ONE;
            assert!(!ok(&b), "at_z[{i}] changed");
        }
        let mut b = o.clone();
        b.rounds[2][3] += F128::ONE;
        assert!(!ok(&b), "a sumcheck round changed");

        // Another table's key rejects the opening.
        let mut other = entries.clone();
        other[17].2 += F128::ONE;
        let u = SparseTable::new(&other, 8, 7).unwrap();
        assert!(!verify_sparse(&u.key, &c, &o, &mut FsChallenger::new(b"t")));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::challenger::{FsChallenger, RandomChallenger};

    /// `cargo test --release -p flock-core --lib sparse_key::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sparse_key_cost() {
        let mut rng = RandomChallenger::new(0x5BE7_C4);
        for nnz_log in [16usize, 20] {
            let (rl, cl) = (nnz_log - 2, nnz_log - 3);
            let entries: Vec<(usize, usize, F128)> = (0..1usize << nnz_log)
                .map(|i| ((i * 2654435761) % (1 << rl), (i * 40503 + 7) % (1 << cl), F128::ONE))
                .collect();
            let t0 = std::time::Instant::now();
            let t = SparseTable::new(&entries, rl, cl).unwrap();
            let key_ms = t0.elapsed().as_secs_f64() * 1e3;
            let row = Weight::eq((0..rl).map(|_| rng.sample_f128()).collect());
            let col = Weight::eq((0..cl).map(|_| rng.sample_f128()).collect());
            let (rw, cw) = (row.materialize(), col.materialize());
            let value = entries.iter().fold(F128::ZERO, |a, &(r, c, v)| a + v * rw[r] * cw[c]);
            let c = MatrixClaim { row, col, value };
            let t1 = std::time::Instant::now();
            let o = t.open(&c, &mut FsChallenger::new(b"b")).unwrap();
            let open_ms = t1.elapsed().as_secs_f64() * 1e3;
            let bytes = bincode::serialize(&o).unwrap().len();
            let t2 = std::time::Instant::now();
            assert!(verify_sparse(&t.key, &c, &o, &mut FsChallenger::new(b"b")));
            let verify_ms = t2.elapsed().as_secs_f64() * 1e3;
            let t3 = std::time::Instant::now();
            let direct = entries.iter().fold(F128::ZERO, |a, &(r, cc, v)| a + v * rw[r] * cw[cc]);
            assert_eq!(direct, value);
            let direct_ms = t3.elapsed().as_secs_f64() * 1e3;
            println!(
                "sparse key 2^{nnz_log} entries: key {key_ms:.0} ms, open {open_ms:.0} ms, proof {} KiB, verify {verify_ms:.2} ms (direct {direct_ms:.2} ms)",
                bytes / 1024
            );
        }
    }
}
