//! The root's structure-table checks against a VERIFICATION KEY.
//!
//! A tower root ends with claims `Σ R(r)·C(c)·M[r,c] = v` on each circuit's
//! structure table ([`SigmaAssertion::matrix`]): eight planes over the
//! cell space (masked ids, live mask, wiring sigma, element constants,
//! Boolean pins, bus and public columns). All planes but sigma follow from
//! a few counts and constants, so the verifier evaluates them itself in
//! `O(rows + slots)`. The sigma plane holds one wire index per live cell,
//! and evaluating it reads every live cell (millions): that plane is what
//! the key commits.
//!
//! Live cells form a jagged table (slot `y` is live on rows `r < h_y`), so
//! the key commits their sigma entries densely, column after column
//! ([`flock_core::pcs::jagged`]), all circuits' tables as blocks of one
//! dense key ([`flock_core::table_key`]). A claim's sigma part is reduced
//! by the jagged sumcheck to one evaluation of the dense vector, and every
//! such evaluation is proved by one opening. Several claims on one table
//! are first folded into one ([`prove_fold`]).

use flock_core::{
    challenger::{Challenger, FsChallenger},
    circuit::{Circuit, SigmaAssertion, StructureTable},
    matrix_fold::{FoldMatrix, FoldProof, MatrixClaim, prove_fold, verify_fold},
    pcs::jagged::{self, JaggedAssistProof, JaggedParams, JaggedSumcheckProof},
    table_key::{DenseKey, DenseOpening, DenseTable, Factor, TensorBasis, verify_dense_bases},
};
use serde::{Deserialize, Serialize};

use crate::tower::F128;

const DOMAIN: &[u8] = b"flock-tower-root-key-v1";
/// The structure table's planes (`SigmaAssertion`'s column order).
const PLANES_LOG: usize = 3;
const SIGMA_PLANE: usize = 2;

fn log2_ceil(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

fn enc(i: usize) -> F128 {
    F128::new(i as u64, 0)
}

/// One structure table's public shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructureShape {
    /// The circuit's digest.
    pub digest: [u8; 32],
    /// Row bits (`ν`).
    pub nu: usize,
    /// Slot bits of one plane (`log2 base_cols`).
    pub slots_log: usize,
    /// Live rows per slot, `2^slots_log` of them.
    pub heights: Vec<u64>,
    /// Dense bits of the committed sigma entries.
    pub m: usize,
}

impl StructureShape {
    fn of(t: &StructureTable) -> (Self, Vec<F128>) {
        let heights: Vec<u64> = (0..t.base)
            .map(|y| t.counts.get(y).map_or(0, |&c| c as u64))
            .collect();
        let area: u64 = heights.iter().sum();
        let m = log2_ceil(area as usize).max(1);
        let sigma = t.sigma.as_ref().expect("the prover's table");
        let mut q: Vec<F128> = sigma.iter().map(|&x| enc(x as usize)).collect();
        q.resize(1 << m, F128::ZERO);
        (
            Self {
                digest: t.digest,
                nu: t.nu,
                slots_log: log2_ceil(t.base),
                heights,
                m,
            },
            q,
        )
    }

    fn params(&self) -> JaggedParams {
        JaggedParams::from_heights(&self.heights, self.nu, self.m)
    }
}

/// The public key: every structure table's shape and one dense commitment
/// to all their sigma entries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootKey {
    pub tables: Vec<StructureShape>,
    /// Bits of one block (the largest `m`).
    pub block_log: usize,
    pub dense: DenseKey,
}

/// The prover's side.
pub struct RootKeyTables {
    pub key: RootKey,
    structures: Vec<StructureTable>,
    q: Vec<Vec<F128>>,
    dense: DenseTable,
}

/// Per table, the fold of its claims (when several) and the jagged
/// reduction of its sigma part (when its sigma weight is nonzero); then
/// one dense opening.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootKeyProof {
    pub folds: Vec<Option<FoldProof>>,
    pub jagged: Vec<Option<(JaggedSumcheckProof, JaggedAssistProof)>>,
    pub dense: Option<DenseOpening>,
}

/// A claim's split: the non-sigma planes' value (computed from the
/// circuit), the sigma plane's scale and points.
struct Split {
    rest: F128,
    scale: F128,
    z_row: Vec<F128>,
    z_col: Vec<F128>,
}

/// Split `c` on `circuit`'s structure table. `None` unless both weights are
/// plain (`low` of length one) and match the table's shape.
fn split(m: &StructureTable, shape: &StructureShape, c: &MatrixClaim) -> Option<Split> {
    if c.row.low.len() != 1
        || c.col.low.len() != 1
        || c.row.point.len() != shape.nu
        || c.col.point.len() != shape.slots_log + PLANES_LOG
        || m.n_rows() != 1 << shape.nu
        || m.n_cols() != 1 << (shape.slots_log + PLANES_LOG)
    {
        return None;
    }
    let base = 1usize << shape.slots_log;
    let rw = c.row.materialize();
    let mut cw = c.col.materialize();
    cw[SIGMA_PLANE * base..(SIGMA_PLANE + 1) * base].fill(F128::ZERO);
    let g = m.row_marginal(&cw, rw.len());
    let rest = rw.iter().zip(&g).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
    let plane_eq =
        c.col.point[shape.slots_log..]
            .iter()
            .enumerate()
            .fold(F128::ONE, |a, (j, &p)| {
                a * if (SIGMA_PLANE >> j) & 1 == 1 {
                    p
                } else {
                    F128::ONE + p
                }
            });
    Some(Split {
        rest,
        scale: c.row.low[0] * c.col.low[0] * plane_eq,
        z_row: c.row.point.clone(),
        z_col: c.col.point[..shape.slots_log].to_vec(),
    })
}

fn bind<Ch: Challenger>(ch: &mut Ch, key: &RootKey, claims: &[(usize, MatrixClaim)]) {
    ch.observe_label(DOMAIN);
    ch.observe_bytes(&bincode::serialize(key).expect("a key serializes"));
    ch.observe_bytes(&bincode::serialize(claims).expect("claims serialize"));
}

impl RootKey {
    fn blocks_log(&self) -> usize {
        log2_ceil(self.tables.len())
    }

    /// The dense basis that reads block `b` at `point` (`m` bits).
    fn basis(&self, b: usize, m: usize, point: &[F128]) -> TensorBasis {
        let mut f = vec![Factor::Eq(point.to_vec())];
        if self.block_log > m {
            f.push(Factor::Eq(vec![F128::ZERO; self.block_log - m]));
        }
        let bl = self.blocks_log();
        if bl > 0 {
            f.push(Factor::Eq(
                (0..bl)
                    .map(|j| {
                        if (b >> j) & 1 == 1 {
                            F128::ONE
                        } else {
                            F128::ZERO
                        }
                    })
                    .collect(),
            ));
        }
        let used = self.block_log + bl;
        if self.dense.log_n > used {
            f.push(Factor::Eq(vec![F128::ZERO; self.dense.log_n - used]));
        }
        TensorBasis::new(F128::ONE, f)
    }

    /// Check every `(table index, claim)`. `vks[i]` is table `i`'s structure
    /// without sigma ([`StructureTable::without_sigma`]): its counts and
    /// constants give the non-sigma planes.
    pub fn verify(
        &self,
        vks: &[StructureTable],
        claims: &[(usize, MatrixClaim)],
        proof: &RootKeyProof,
    ) -> bool {
        let t_all = std::time::Instant::now();
        let n = self.tables.len();
        if vks.len() != n
            || proof.folds.len() != n
            || proof.jagged.len() != n
            || claims.iter().any(|(t, _)| *t >= n)
        {
            return false;
        }
        for (shape, c) in self.tables.iter().zip(vks) {
            if c.digest != shape.digest
                || c.nu != shape.nu
                || 1usize << shape.slots_log != c.base
                || shape.heights.len() != c.base
                || shape
                    .heights
                    .iter()
                    .enumerate()
                    .any(|(y, &h)| h != c.counts.get(y).map_or(0, |&x| x as u64))
            {
                return false;
            }
        }
        if std::env::var_os("ROOT_KEY_TRACE").is_some() {
            eprintln!(
                "root key: shape checks {:.1} ms",
                t_all.elapsed().as_secs_f64() * 1e3
            );
        }
        let mut ch = FsChallenger::new(DOMAIN);
        bind(&mut ch, self, claims);
        let mut items = Vec::new();
        for i in 0..n {
            let group: Vec<MatrixClaim> = claims
                .iter()
                .filter(|(t, _)| *t == i)
                .map(|(_, c)| c.clone())
                .collect();
            let one = match (group.len(), &proof.folds[i]) {
                (0, None) => None,
                (1, None) => Some(group[0].clone()),
                (k, Some(p)) if k > 1 => match verify_fold(&group, p, &mut ch) {
                    Ok(c) => Some(c),
                    Err(_) => return false,
                },
                _ => return false,
            };
            let shape = &self.tables[i];
            match (one, &proof.jagged[i]) {
                (None, None) => {}
                (Some(c), jp) => {
                    let t_split = std::time::Instant::now();
                    let Some(s) = split(&vks[i], shape, &c) else {
                        return false;
                    };
                    let sig = c.value + s.rest; // characteristic 2
                    let split_ms = t_split.elapsed().as_secs_f64() * 1e3;
                    let t_j = std::time::Instant::now();
                    match jp {
                        None if s.scale.is_zero() && sig.is_zero() => {}
                        Some(jp) if !s.scale.is_zero() => {
                            let v = sig * s.scale.inv();
                            match jagged::verify_with_assist(
                                &shape.params(),
                                &s.z_row,
                                &s.z_col,
                                v,
                                &jp.0,
                                &jp.1,
                                &mut ch,
                            ) {
                                Some(d) => {
                                    if std::env::var_os("ROOT_KEY_TRACE").is_some() {
                                        eprintln!(
                                            "root key table {i}: split {split_ms:.1} ms, jagged {:.1} ms",
                                            t_j.elapsed().as_secs_f64() * 1e3
                                        );
                                    }
                                    items.push((self.basis(i, shape.m, &d.point), d.alpha))
                                }
                                None => return false,
                            }
                        }
                        _ => return false,
                    }
                }
                _ => return false,
            }
        }
        let t_d = std::time::Instant::now();
        let out = match (&proof.dense, items.is_empty()) {
            (None, true) => true,
            (Some(o), false) => verify_dense_bases(&self.dense, &items, o, &mut ch),
            _ => false,
        };
        if std::env::var_os("ROOT_KEY_TRACE").is_some() {
            eprintln!(
                "root key: dense opening {:.1} ms",
                t_d.elapsed().as_secs_f64() * 1e3
            );
        }
        out
    }
}

impl RootKeyTables {
    /// Build the key over the circuits' structure tables (a circuit named
    /// twice is kept once).
    pub fn new(circuits: &[&Circuit]) -> Option<Self> {
        Self::from_tables(
            circuits
                .iter()
                .map(|c| StructureTable::from_circuit(c, true))
                .collect(),
        )
    }

    /// Build the key over prover's structure tables (with sigma); a digest
    /// named twice is kept once.
    pub fn from_tables(tables: Vec<StructureTable>) -> Option<Self> {
        let mut uniq: Vec<StructureTable> = Vec::new();
        for t in tables {
            if !uniq.iter().any(|u| u.digest == t.digest) {
                uniq.push(t);
            }
        }
        let (tables, q): (Vec<StructureShape>, Vec<Vec<F128>>) =
            uniq.iter().map(|c| StructureShape::of(c)).unzip();
        let block_log = tables.iter().map(|t| t.m).max()?;
        let blocks = tables.len().next_power_of_two();
        let mut vals = vec![F128::ZERO; blocks << block_log];
        for (b, qb) in q.iter().enumerate() {
            vals[b << block_log..][..qb.len()].copy_from_slice(qb);
        }
        let dense = DenseTable::new(&vals, block_log)?;
        Some(Self {
            key: RootKey {
                tables,
                block_log,
                dense: dense.key.clone(),
            },
            structures: uniq,
            q,
            dense,
        })
    }

    /// The verifier's tables in key order (what [`RootKey::verify`] takes).
    pub fn vks(&self) -> Vec<StructureTable> {
        self.structures
            .iter()
            .map(StructureTable::without_sigma)
            .collect()
    }

    /// The index of the table with this circuit digest.
    pub fn index(&self, digest: &[u8; 32]) -> Option<usize> {
        self.key.tables.iter().position(|t| t.digest == *digest)
    }

    /// Prove every `(table index, claim)`. `None` if a claim is false or
    /// not of the plain shape a root carries.
    pub fn prove(&self, claims: &[(usize, MatrixClaim)]) -> Option<RootKeyProof> {
        let n = self.key.tables.len();
        let mut ch = FsChallenger::new(DOMAIN);
        bind(&mut ch, &self.key, claims);
        let mut folds = vec![None; n];
        let mut jag = vec![None; n];
        let mut items = Vec::new();
        for i in 0..n {
            let group: Vec<MatrixClaim> = claims
                .iter()
                .filter(|(t, _)| *t == i)
                .map(|(_, c)| c.clone())
                .collect();
            let m = &self.structures[i];
            let one = match group.len() {
                0 => continue,
                1 => group[0].clone(),
                _ => {
                    let combs: Vec<Vec<F128>> = group
                        .iter()
                        .map(|c| m.col_marginal(&c.row.materialize(), m.n_cols()))
                        .collect();
                    let (p, c) = prove_fold(m, &combs, &group, &mut ch);
                    folds[i] = Some(p);
                    c
                }
            };
            let shape = &self.key.tables[i];
            let s = split(m, shape, &one)?;
            let sig = one.value + s.rest; // characteristic 2
            if s.scale.is_zero() {
                if !sig.is_zero() {
                    return None;
                }
                continue;
            }
            let v = sig * s.scale.inv();
            let params = shape.params();
            let mut replay = ch.clone();
            let (jp, ja, honest) =
                jagged::prove_with_assist(&params, &self.q[i], &s.z_row, &s.z_col, &mut ch);
            if honest != v {
                return None;
            }
            let d =
                jagged::verify_with_assist(&params, &s.z_row, &s.z_col, v, &jp, &ja, &mut replay)?;
            items.push((self.key.basis(i, shape.m, &d.point), d.alpha));
            jag[i] = Some((jp, ja));
        }
        let dense = if items.is_empty() {
            None
        } else {
            Some(self.dense.open_bases(&items, &mut ch)?)
        };
        Some(RootKeyProof {
            folds,
            jagged: jag,
            dense,
        })
    }
}

/// One line per circuit: its structure table's shape and live cells.
pub fn describe(circuits: &[&Circuit]) -> String {
    let mut out = String::new();
    for c in circuits {
        let m = SigmaAssertion::matrix(c);
        let live = c.live_mask();
        let n_live: usize = live.counts.iter().sum();
        out.push_str(&format!(
            "structure {}x{} cells 2^{} (nu {}), live {}\n",
            m.n_rows(),
            m.n_cols(),
            c.cells().mu(),
            c.cells().nu(),
            n_live
        ));
    }
    out
}
