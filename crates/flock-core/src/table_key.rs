//! Verification keys for a circuit's fixed tables.
//!
//! A root discharge checks claims of one shape: for a fixed public table
//! `M`, `Σ_{r,c} row(r)·col(c)·M[r,c] = value`, where `row` and `col` are
//! tensor weights (`low ⊗ eq(point)`). Evaluating `M` directly reads every
//! entry. A **key** commits to `M` once instead; a claim is then proved by
//! a short opening against the key, and checked without reading `M`.
//!
//! This module holds the **dense** key: the table is a flat column-major
//! vector (`M[r, c] = vals[(c << rows_log) | r]`, the layout of
//! [`crate::matrix_fold::DenseMatrix`] and of the sigma table), committed
//! with Ligerito under an embedded security config. The claim's weight on
//! that vector is the tensor `row ⊗ col` (zero on the padding), so the
//! opening is Ligerito's inner-product proof `⟨vals, b⟩ = value` against
//! the key's cap, and the verifier evaluates `b` succinctly at the
//! protocol's residual point (`O(log)` work plus the explicit low factors).
//!
//! Soundness rests on the cap: the verifier must take the [`DenseKey`]
//! from the statement (or rebuild it from the circuit), never from the
//! prover.

use serde::{Deserialize, Serialize};

use crate::{
    challenger::Challenger,
    field::{F128, F256},
    matrix_fold::{MatrixClaim, Weight},
    merkle::Hash,
    ntt::additive_ntt_f128::AdditiveNttF128,
    pcs::{
        LOG_PACKING,
        ligerito::{
            LigeritoProfile, LigeritoProof, ProverConfig, VerifierConfig, embedded_initial_k,
            extension::recursive_verifier_with_basis_succinct, ligero_commit, prover_config_for,
            recursive_prover_with_basis, verifier_config_for,
        },
    },
};

/// The smallest and largest committed sizes (variables of the flat
/// vector) with an embedded Ligerito security config.
const MIN_LOG_N: usize = 22 - LOG_PACKING;
const MAX_LOG_N: usize = 35 - LOG_PACKING;
const PROFILE: LigeritoProfile = LigeritoProfile::Fast;

/// The public key of one dense table: its shape and the commitment's cap.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseKey {
    /// Row bits of the table (`M` has `2^rows_log` rows).
    pub rows_log: usize,
    /// Column bits of the table.
    pub cols_log: usize,
    /// Variables of the committed vector (`rows_log + cols_log`, padded up
    /// to the smallest configured size).
    pub log_n: usize,
    /// The level-zero commitment cap.
    pub cap: Vec<Hash>,
}

/// The prover's side of a dense key: the table and its commitment.
pub struct DenseTable {
    pub key: DenseKey,
    vals: Vec<F128>,
    mat: Vec<F128>,
    tree: Vec<Hash>,
    config: ProverConfig,
}

/// An opening of one claim against a dense key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseOpening {
    pub proof: LigeritoProof,
}

fn configs(log_n: usize) -> Option<(ProverConfig, VerifierConfig)> {
    let k = embedded_initial_k(log_n + LOG_PACKING, PROFILE)?;
    let p = prover_config_for(log_n, k, PROFILE).ok()?;
    let v = verifier_config_for(log_n, k, PROFILE).ok()?;
    Some((p, v))
}

impl DenseTable {
    /// Commit to a column-major table with `2^rows_log` rows. `vals.len()`
    /// must be a multiple of `2^rows_log` and a power of two.
    pub fn new(vals: &[F128], rows_log: usize) -> Option<Self> {
        let n = vals.len();
        if !n.is_power_of_two() || n < (1usize << rows_log) {
            return None;
        }
        let total = n.trailing_zeros() as usize;
        let cols_log = total - rows_log;
        let log_n = total.max(MIN_LOG_N);
        if log_n > MAX_LOG_N {
            return None;
        }
        let (config, vconfig) = configs(log_n)?;
        let mut padded = vals.to_vec();
        padded.resize(1usize << log_n, F128::ZERO);
        let k = config.initial_k;
        let rate = config.log_inv_rates[0];
        let ntt = AdditiveNttF128::standard(log_n - k + rate);
        let w = ligero_commit(&padded, log_n - k, k, rate, &ntt, config.merkle_hash);
        let cap = w.cap(vconfig.level_zero_cap_depth()).to_vec();
        Some(Self {
            key: DenseKey {
                rows_log,
                cols_log,
                log_n,
                cap,
            },
            vals: padded,
            mat: w.mat.clone(),
            tree: w.tree.clone(),
            config,
        })
    }

    /// Entry `i` of the committed (padded) vector.
    pub fn val(&self, i: usize) -> F128 {
        self.vals[i]
    }

    /// Prove `claim` against the key. `None` if the claim's weights do not
    /// fit the table or the claim is false.
    pub fn open<Ch: Challenger>(&self, claim: &MatrixClaim, ch: &mut Ch) -> Option<DenseOpening> {
        self.open_many(std::slice::from_ref(claim), ch)
    }

    /// Prove every claim with ONE opening: a random combination
    /// `Σ ρ^i·b_i` of their bases against `Σ ρ^i·value_i`, `ρ` drawn after
    /// all claims are bound. `None` if any claim does not fit or is false.
    pub fn open_many<Ch: Challenger>(&self, claims: &[MatrixClaim], ch: &mut Ch) -> Option<DenseOpening> {
        let bases: Vec<TensorBasis> =
            claims.iter().map(|c| TensorBasis::of(&self.key, c)).collect::<Option<_>>()?;
        let mut b = vec![F128::ZERO; self.vals.len()];
        let mut mats = Vec::with_capacity(bases.len());
        for (basis, c) in bases.iter().zip(claims) {
            let m = basis.materialize();
            let honest = self.vals.iter().zip(&m).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
            if honest != c.value {
                return None;
            }
            mats.push(m);
        }
        let rho = bind(ch, &self.key, claims);
        let mut target = F128::ZERO;
        let mut p = F128::ONE;
        for (m, c) in mats.iter().zip(claims) {
            for (x, y) in b.iter_mut().zip(m) {
                *x += p * *y;
            }
            target += p * c.value;
            p *= rho;
        }
        let proof = recursive_prover_with_basis(
            &self.config,
            self.vals.clone(),
            b,
            target,
            &self.mat,
            &self.tree,
            ch,
        );
        Some(DenseOpening { proof })
    }
}

/// Check an opening of `claim` against `key`. Reads no table entry.
pub fn verify_dense<Ch: Challenger>(
    key: &DenseKey,
    claim: &MatrixClaim,
    opening: &DenseOpening,
    ch: &mut Ch,
) -> bool {
    verify_dense_many(key, std::slice::from_ref(claim), opening, ch)
}

/// Check one opening of every claim ([`DenseTable::open_many`]).
pub fn verify_dense_many<Ch: Challenger>(
    key: &DenseKey,
    claims: &[MatrixClaim],
    opening: &DenseOpening,
    ch: &mut Ch,
) -> bool {
    if claims.is_empty() {
        return false;
    }
    let Some(bases) = claims.iter().map(|c| TensorBasis::of(key, c)).collect::<Option<Vec<_>>>() else {
        return false;
    };
    let Some((_, vconfig)) = configs(key.log_n) else {
        return false;
    };
    let rho = bind(ch, key, claims);
    let mut target = F128::ZERO;
    let mut p = F128::ONE;
    let mut powers = Vec::with_capacity(claims.len());
    for c in claims {
        target += p * c.value;
        powers.push(F256::from(p));
        p *= rho;
    }
    recursive_verifier_with_basis_succinct(
        &vconfig,
        &opening.proof,
        key.log_n,
        target,
        &key.cap,
        None,
        |prefix, residual_log| {
            let mut acc = vec![F256::ZERO; 1usize << residual_log];
            for (basis, &q) in bases.iter().zip(&powers) {
                for (a, x) in acc.iter_mut().zip(basis.fold_residual(prefix, residual_log)) {
                    *a += q * x;
                }
            }
            acc
        },
        ch,
    )
}

fn bind<Ch: Challenger>(ch: &mut Ch, key: &DenseKey, claims: &[MatrixClaim]) -> F128 {
    ch.observe_label(b"flock-table-key-dense-v0");
    for h in &key.cap {
        ch.observe_bytes(h.as_ref());
    }
    ch.observe_bytes(&(claims.len() as u64).to_le_bytes());
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

/// One factor of a tensor weight, over consecutive index bits.
#[derive(Clone, Debug)]
pub(crate) enum Factor {
    /// An explicit table over `log2(len)` bits.
    Explicit(Vec<F128>),
    /// `eq(point, ·)` over `point.len()` bits.
    Eq(Vec<F128>),
}

impl Factor {
    fn bits(&self) -> usize {
        match self {
            Factor::Explicit(v) => v.len().trailing_zeros() as usize,
            Factor::Eq(p) => p.len(),
        }
    }
}

/// The basis `b` on the committed vector: `scale · ⊗ factors`, LSB first.
#[derive(Clone, Debug)]
pub(crate) struct TensorBasis {
    scale: F128,
    factors: Vec<Factor>,
}

impl TensorBasis {
    /// The weight `row ⊗ col` restricted to the table, zero on padding.
    pub(crate) fn of(key: &DenseKey, claim: &MatrixClaim) -> Option<Self> {
        let mut scale = F128::ONE;
        let mut factors = Vec::new();
        for (w, bits) in [(&claim.row, key.rows_log), (&claim.col, key.cols_log)] {
            let (s, f) = restrict(w, bits)?;
            scale *= s;
            factors.extend(f);
        }
        let used = key.rows_log + key.cols_log;
        if key.log_n < used {
            return None;
        }
        if key.log_n > used {
            factors.push(Factor::Eq(vec![F128::ZERO; key.log_n - used]));
        }
        Some(Self { scale, factors })
    }

    pub(crate) fn materialize(&self) -> Vec<F128> {
        let mut out = vec![self.scale];
        for f in &self.factors {
            let t = match f {
                Factor::Explicit(v) => v.clone(),
                Factor::Eq(p) => Weight::eq(p.clone()).materialize(),
            };
            let mut next = Vec::with_capacity(out.len() * t.len());
            for &y in &t {
                for &x in &out {
                    next.push(x * y);
                }
            }
            out = next;
        }
        out
    }

    /// `b(prefix, y)` for every `y` over the last `residual_log` bits.
    /// The basis MLE at a full point `z` (LSB first).
    pub(crate) fn eval_at(&self, z: &[F128]) -> F128 {
        let mut acc = self.scale;
        let mut at = 0usize;
        for f in &self.factors {
            let bits = f.bits();
            let zs = &z[at..at + bits];
            acc *= match f {
                Factor::Explicit(v) => {
                    let mut vals = v.clone();
                    for &r in zs {
                        vals = (0..vals.len() / 2)
                            .map(|i| vals[2 * i] + r * (vals[2 * i] + vals[2 * i + 1]))
                            .collect();
                    }
                    vals[0]
                }
                Factor::Eq(p) => p
                    .iter()
                    .zip(zs)
                    .fold(F128::ONE, |a, (&q, &r)| a * (F128::ONE + q + r)),
            };
            at += bits;
        }
        assert_eq!(at, z.len(), "the point covers the basis");
        acc
    }

    pub(crate) fn fold_residual(&self, prefix: &[F256], residual_log: usize) -> Vec<F256> {
        let total: usize = self.factors.iter().map(Factor::bits).sum();
        assert_eq!(prefix.len() + residual_log, total, "prefix and residual cover b");
        let mut scale = F256::from(self.scale);
        let mut residual: Vec<F256> = vec![F256::ONE];
        let mut at = 0usize;
        for f in &self.factors {
            let bits = f.bits();
            let take = prefix.len().saturating_sub(at).min(bits);
            let rs = &prefix[at.min(prefix.len())..][..take];
            let rest: Vec<F256> = match f {
                Factor::Explicit(v) => {
                    let mut vals: Vec<F256> = v.iter().copied().map(F256::from).collect();
                    for &r in rs {
                        vals = (0..vals.len() / 2)
                            .map(|i| vals[2 * i] + r * (vals[2 * i] + vals[2 * i + 1]))
                            .collect();
                    }
                    vals
                }
                Factor::Eq(p) => {
                    for (&z, &r) in p.iter().zip(rs) {
                        scale = scale * (F256::ONE + F256::from(z) + r);
                    }
                    Weight::eq(p[take..].to_vec())
                        .materialize()
                        .into_iter()
                        .map(F256::from)
                        .collect()
                }
            };
            if rest.len() > 1 || take < bits {
                let mut next = Vec::with_capacity(residual.len() * rest.len());
                for &y in &rest {
                    for &x in &residual {
                        next.push(x * y);
                    }
                }
                residual = next;
            } else {
                scale = scale * rest[0];
            }
            at += bits;
        }
        assert_eq!(residual.len(), 1usize << residual_log);
        residual.into_iter().map(|x| x * scale).collect()
    }
}

/// `w` restricted to its first `2^bits` entries, as `(scale, factors)`.
fn restrict(w: &Weight, bits: usize) -> Option<(F128, Vec<Factor>)> {
    let s = w.low.len().trailing_zeros() as usize;
    if !w.low.len().is_power_of_two() || s > bits || s + w.point.len() < bits {
        return None;
    }
    let keep = bits - s;
    let scale = w.point[keep..]
        .iter()
        .fold(F128::ONE, |a, &p| a * (F128::ONE + p));
    let mut f = Vec::new();
    if s > 0 || w.low[0] != F128::ONE {
        f.push(Factor::Explicit(w.low.clone()));
    }
    if keep > 0 {
        f.push(Factor::Eq(w.point[..keep].to_vec()));
    }
    Some((scale, f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        challenger::{FsChallenger, RandomChallenger},
        matrix_fold::{DenseMatrix, bilinear},
    };

    fn rand_vec(rng: &mut RandomChallenger, n: usize) -> Vec<F128> {
        (0..n).map(|_| rng.sample_f128()).collect()
    }

    fn claim(rng: &mut RandomChallenger, m: &DenseMatrix, row: (usize, usize), col: (usize, usize)) -> MatrixClaim {
        let row = Weight::low_eq(rand_vec(rng, 1 << row.0), rand_vec(rng, row.1));
        let col = Weight::low_eq(rand_vec(rng, 1 << col.0), rand_vec(rng, col.1));
        let (rw, cw) = (row.materialize(), col.materialize());
        let n_rows = 1usize << m.n_rows_log;
        let n_cols = m.vals.len() >> m.n_rows_log;
        let mut value = F128::ZERO;
        for c in 0..n_cols.min(cw.len()) {
            for r in 0..n_rows.min(rw.len()) {
                value += rw[r] * cw[c] * m.vals[(c << m.n_rows_log) | r];
            }
        }
        MatrixClaim { row, col, value }
    }

    #[test]
    fn dense_key_opens_and_verifies() {
        let mut rng = RandomChallenger::new(0x7AB1_E5EE);
        let rows_log = 9;
        let vals = rand_vec(&mut rng, 1 << 16);
        let m = DenseMatrix {
            vals: vals.clone(),
            n_rows_log: rows_log,
        };
        let t = DenseTable::new(&vals, rows_log).expect("a configured size");
        assert_eq!(t.key.log_n, MIN_LOG_N.max(16));
        // Plain eq claims, explicit low factors, and weights wider than
        // the table (their high coordinates see the zero padding).
        for (row, col) in [((0, 9), (0, 7)), ((2, 7), (1, 6)), ((0, 11), (3, 6))] {
            let c = claim(&mut rng, &m, row, col);
            if row.0 + row.1 == rows_log {
                assert_eq!(c.value, bilinear(&c.row, &c.col, &m));
            }
            let mut pc = FsChallenger::new(b"table-key-test");
            let o = t.open(&c, &mut pc).expect("an honest claim opens");
            let mut vc = FsChallenger::new(b"table-key-test");
            assert!(verify_dense(&t.key, &c, &o, &mut vc), "honest opening at {row:?} {col:?}");

            // A wrong value is rejected.
            let mut bad = c.clone();
            bad.value += F128::ONE;
            let mut vc = FsChallenger::new(b"table-key-test");
            assert!(!verify_dense(&t.key, &bad, &o, &mut vc), "a wrong value");
            assert!(t.open(&bad, &mut FsChallenger::new(b"x")).is_none());
        }
    }

    #[test]
    fn dense_key_opens_many_claims_at_once() {
        let mut rng = RandomChallenger::new(0x3A11_C1A1);
        let rows_log = 7;
        let vals = rand_vec(&mut rng, 1 << 15);
        let m = DenseMatrix { vals: vals.clone(), n_rows_log: rows_log };
        let t = DenseTable::new(&vals, rows_log).unwrap();
        let cs: Vec<MatrixClaim> = (0..5).map(|i| claim(&mut rng, &m, (i % 2, 7 - i % 2), (0, 8))).collect();
        let o = t.open_many(&cs, &mut FsChallenger::new(b"m")).unwrap();
        assert!(verify_dense_many(&t.key, &cs, &o, &mut FsChallenger::new(b"m")));
        // Any one claim changed, dropped or reordered is rejected.
        let mut bad = cs.clone();
        bad[3].value += F128::ONE;
        assert!(!verify_dense_many(&t.key, &bad, &o, &mut FsChallenger::new(b"m")));
        assert!(!verify_dense_many(&t.key, &cs[..4], &o, &mut FsChallenger::new(b"m")));
        let mut swapped = cs.clone();
        swapped.swap(0, 1);
        assert!(!verify_dense_many(&t.key, &swapped, &o, &mut FsChallenger::new(b"m")));
    }

    #[test]
    fn dense_key_binds_the_table() {
        let mut rng = RandomChallenger::new(0x0B1D_7AB1);
        let rows_log = 8;
        let vals = rand_vec(&mut rng, 1 << 15);
        let mut other = vals.clone();
        other[12345] += F128::ONE;
        let m = DenseMatrix {
            vals: vals.clone(),
            n_rows_log: rows_log,
        };
        let t = DenseTable::new(&vals, rows_log).unwrap();
        let u = DenseTable::new(&other, rows_log).unwrap();
        assert_ne!(t.key.cap, u.key.cap);
        let c = claim(&mut rng, &m, (0, 8), (0, 7));
        let o = t.open(&c, &mut FsChallenger::new(b"k")).unwrap();
        // The opening is for t's table: u's key rejects it.
        assert!(!verify_dense(&u.key, &c, &o, &mut FsChallenger::new(b"k")));
        assert!(verify_dense(&t.key, &c, &o, &mut FsChallenger::new(b"k")));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::challenger::{FsChallenger, RandomChallenger};

    /// `cargo test --release -p flock-core --lib table_key::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dense_key_cost() {
        let mut rng = RandomChallenger::new(0xBE7C_4);
        for log_n in [18usize, 22] {
            let rows_log = log_n / 2;
            let vals: Vec<F128> = (0..1usize << log_n).map(|_| rng.sample_f128()).collect();
            let t0 = std::time::Instant::now();
            let t = DenseTable::new(&vals, rows_log).unwrap();
            let key_ms = t0.elapsed().as_secs_f64() * 1e3;
            let row = Weight::eq((0..rows_log).map(|_| rng.sample_f128()).collect());
            let col = Weight::eq((0..log_n - rows_log).map(|_| rng.sample_f128()).collect());
            let b = TensorBasis::of(&t.key, &MatrixClaim { row: row.clone(), col: col.clone(), value: F128::ZERO })
                .unwrap()
                .materialize();
            let value = t.vals.iter().zip(&b).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
            let c = MatrixClaim { row, col, value };
            let t1 = std::time::Instant::now();
            let o = t.open(&c, &mut FsChallenger::new(b"b")).unwrap();
            let open_ms = t1.elapsed().as_secs_f64() * 1e3;
            let bytes = bincode::serialize(&o).unwrap().len();
            let t2 = std::time::Instant::now();
            assert!(verify_dense(&t.key, &c, &o, &mut FsChallenger::new(b"b")));
            let verify_ms = t2.elapsed().as_secs_f64() * 1e3;
            println!(
                "dense key 2^{log_n}: key {key_ms:.0} ms, open {open_ms:.0} ms, proof {} KiB, verify {verify_ms:.2} ms (direct check reads 2^{log_n} entries)",
                bytes / 1024
            );
        }
    }
}
