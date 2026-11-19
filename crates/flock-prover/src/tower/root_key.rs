//! The root's table checks against a VERIFICATION KEY.
//!
//! A tower root ends with claims `Σ R(r)·C(c)·M[r,c] = v` on the fixed
//! tables of its circuits: the constraint matrices of each boolean and
//! element type (sparse) and each circuit's structure table (dense: wiring
//! sigma, live mask, element constants, bus and public columns). The plain
//! discharge evaluates every table. Here the tables are committed once, as
//! a [`RootKey`]: every sparse matrix in one Spark-style key
//! ([`flock_core::sparse_key`]) and every structure table in one dense key
//! ([`flock_core::table_key`]). The root's claims are grouped by table,
//! several claims on one table are folded into one ([`prove_fold`]), and
//! one opening per key proves them all. The verifier replays the folds and
//! checks the two openings; it reads no table.
//!
//! The jagged layouts stay direct (their check is `O(runs)`, about 2 ms).

use flock_core::{
    challenger::{Challenger, FsChallenger},
    circuit::{Circuit, SigmaAssertion},
    element_r1cs::SparseF128Matrix,
    matrix_fold::{FoldMatrix, FoldProof, MatrixClaim, Weight, prove_fold, verify_fold},
    r1cs::SparseBinaryMatrix,
    sparse_key::{SparseKey, SparseOpening, SparseTable, verify_sparse_many},
    table_key::{DenseKey, DenseOpening, DenseTable, Factor, TensorBasis, verify_dense_bases},
};
use serde::{Deserialize, Serialize};

use crate::tower::F128;

const DOMAIN: &[u8] = b"flock-tower-root-key-v0";

/// One fixed table a root claim may name.
#[derive(Clone, Copy)]
pub enum Table<'a> {
    Boolean(&'a SparseBinaryMatrix),
    Element(&'a SparseF128Matrix),
    Structure(&'a Circuit),
}

impl Table<'_> {
    /// Identity for grouping: the same table object, wherever it is named.
    fn id(&self) -> (u8, usize) {
        match self {
            Table::Boolean(m) => (0, *m as *const _ as usize),
            Table::Element(m) => (1, *m as *const _ as usize),
            Table::Structure(c) => (2, *c as *const _ as usize),
        }
    }
}

fn log2_ceil(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

/// The public key: the two commitments and every table's shape, in the
/// order the tables were given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootKey {
    /// Per table: `None` for a sparse table (its slot is its rank among the
    /// sparse ones), `Some((rows_log, cols_log))` for a dense one.
    pub shapes: Vec<Option<(usize, usize)>>,
    pub sparse: Option<SparseKey>,
    pub dense: Option<DenseKey>,
    /// Bits of one dense block (`max rows_log + cols_log`).
    pub block_log: usize,
}

/// The prover's side.
pub struct RootKeyTables<'a> {
    pub key: RootKey,
    tables: Vec<Table<'a>>,
    sparse: Option<SparseTable>,
    dense: Option<DenseTable>,
}

/// What a root proves: per table, the fold of its claims (when there are
/// several), then one opening per key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootKeyProof {
    pub folds: Vec<Option<FoldProof>>,
    pub sparse: Option<SparseOpening>,
    pub dense: Option<DenseOpening>,
}

fn entries_bool(m: &SparseBinaryMatrix) -> Vec<(usize, usize, F128)> {
    m.rows
        .iter()
        .enumerate()
        .flat_map(|(r, cols)| cols.iter().map(move |&c| (r, c, F128::ONE)))
        .collect()
}

fn entries_el(m: &SparseF128Matrix) -> Vec<(usize, usize, F128)> {
    m.rows
        .iter()
        .enumerate()
        .flat_map(|(r, cols)| cols.iter().map(move |&(c, v)| (r, c, v)))
        .collect()
}

impl<'a> RootKeyTables<'a> {
    /// Build the key over `tables` (distinct tables; a table named twice
    /// is kept once).
    pub fn new(tables: &[Table<'a>]) -> Option<Self> {
        let mut uniq: Vec<Table<'a>> = Vec::new();
        for t in tables {
            if !uniq.iter().any(|u| u.id() == t.id()) {
                uniq.push(*t);
            }
        }
        let mut sparse_entries: Vec<Vec<(usize, usize, F128)>> = Vec::new();
        let (mut rl, mut cl) = (0usize, 0usize);
        let mut dense_mats = Vec::new();
        let mut shapes = Vec::new();
        for t in &uniq {
            match t {
                Table::Boolean(m) => {
                    rl = rl.max(log2_ceil(m.num_rows));
                    cl = cl.max(log2_ceil(m.num_cols));
                    sparse_entries.push(entries_bool(m));
                    shapes.push(None);
                }
                Table::Element(m) => {
                    rl = rl.max(log2_ceil(m.num_rows));
                    cl = cl.max(log2_ceil(m.num_cols));
                    sparse_entries.push(entries_el(m));
                    shapes.push(None);
                }
                Table::Structure(c) => {
                    let d = SigmaAssertion::matrix(c).materialize();
                    let cols_log = log2_ceil(d.vals.len() >> d.n_rows_log);
                    shapes.push(Some((d.n_rows_log, cols_log)));
                    dense_mats.push(d);
                }
            }
        }
        let sparse = if sparse_entries.is_empty() {
            None
        } else {
            let refs: Vec<&[(usize, usize, F128)]> = sparse_entries.iter().map(|e| e.as_slice()).collect();
            Some(SparseTable::new_many(&refs, rl, cl)?)
        };
        let block_log = shapes.iter().flatten().map(|(r, c)| r + c).max().unwrap_or(0);
        let dense = if dense_mats.is_empty() {
            None
        } else {
            let blocks = dense_mats.len().next_power_of_two();
            let mut vals = vec![F128::ZERO; blocks << block_log];
            for (b, d) in dense_mats.iter().enumerate() {
                vals[b << block_log..][..d.vals.len()].copy_from_slice(&d.vals);
            }
            Some(DenseTable::new(&vals, block_log)?)
        };
        Some(Self {
            key: RootKey {
                shapes,
                sparse: sparse.as_ref().map(|s| s.key.clone()),
                dense: dense.as_ref().map(|d| d.key.clone()),
                block_log,
            },
            tables: uniq,
            sparse,
            dense,
        })
    }

    /// The index of `t` in the key, if it holds it.
    pub fn index(&self, t: &Table<'_>) -> Option<usize> {
        self.tables.iter().position(|u| u.id() == t.id())
    }

    fn fold_matrix(&self, i: usize) -> Box<dyn FoldMatrix + '_> {
        match self.tables[i] {
            Table::Boolean(m) => Box::new(m.clone()),
            Table::Element(m) => Box::new(m.clone()),
            Table::Structure(c) => Box::new(SigmaAssertion::matrix(c)),
        }
    }

    /// Prove every `(table index, claim)`. `None` if a claim is false or
    /// does not fit its table.
    pub fn prove(&self, claims: &[(usize, MatrixClaim)]) -> Option<RootKeyProof> {
        let n = self.tables.len();
        let mut ch = FsChallenger::new(DOMAIN);
        bind(&mut ch, &self.key, claims);
        let mut folds = vec![None; n];
        let mut one = vec![None; n];
        for i in 0..n {
            let group: Vec<MatrixClaim> = claims.iter().filter(|(t, _)| *t == i).map(|(_, c)| c.clone()).collect();
            match group.len() {
                0 => {}
                1 => one[i] = Some(group[0].clone()),
                _ => {
                    let m = self.fold_matrix(i);
                    let combs: Vec<Vec<F128>> =
                        group.iter().map(|c| m.col_marginal(&c.row.materialize(), m.n_cols())).collect();
                    let (p, c) = prove_fold(m.as_ref(), &combs, &group, &mut ch);
                    folds[i] = Some(p);
                    one[i] = Some(c);
                }
            }
        }
        let (sparse_claims, dense_items) = self.key.final_claims(&one)?;
        let sparse = match (&self.sparse, sparse_claims) {
            (Some(t), Some(cs)) => Some(t.open_many(&cs, &mut ch)?),
            (None, None) => None,
            _ => return None,
        };
        let dense = match (&self.dense, dense_items) {
            (Some(t), Some(items)) => Some(t.open_bases(&items, &mut ch)?),
            (None, None) => None,
            _ => return None,
        };
        Some(RootKeyProof {
            folds,
            sparse,
            dense,
        })
    }
}

fn bind<Ch: Challenger>(ch: &mut Ch, key: &RootKey, claims: &[(usize, MatrixClaim)]) {
    ch.observe_label(DOMAIN);
    ch.observe_bytes(&bincode::serialize(key).expect("a key serializes"));
    ch.observe_bytes(&bincode::serialize(claims).expect("claims serialize"));
}

/// A claim nobody makes on a table in the key: zero weight, value zero.
fn zero_claim(rows_log: usize, cols_log: usize) -> MatrixClaim {
    MatrixClaim {
        row: Weight::low_eq(vec![F128::ZERO], vec![F128::ZERO; rows_log]),
        col: Weight::eq(vec![F128::ZERO; cols_log]),
        value: F128::ZERO,
    }
}

impl RootKey {
    /// The one claim per table, split into the sparse key's claims (one per
    /// sparse table, in order) and the dense key's bases.
    #[allow(clippy::type_complexity)]
    fn final_claims(
        &self,
        one: &[Option<MatrixClaim>],
    ) -> Option<(Option<Vec<MatrixClaim>>, Option<Vec<(TensorBasis, F128)>>)> {
        let mut sparse = Vec::new();
        let mut dense = Vec::new();
        let mut block = 0usize;
        let blocks_log = self
            .shapes
            .iter()
            .filter(|s| s.is_some())
            .count()
            .next_power_of_two()
            .trailing_zeros() as usize;
        for (i, shape) in self.shapes.iter().enumerate() {
            match shape {
                None => {
                    let k = self.sparse.as_ref()?;
                    sparse.push(one[i].clone().unwrap_or_else(|| zero_claim(k.rows_log, k.cols_log)));
                }
                Some((rl, cl)) => {
                    if let Some(c) = &one[i] {
                        let (s1, mut f) = TensorBasis::weight_factors(&c.row, *rl, false)?;
                        let (s2, f2) = TensorBasis::weight_factors(&c.col, *cl, false)?;
                        f.extend(f2);
                        if self.block_log > rl + cl {
                            f.push(Factor::Eq(vec![F128::ZERO; self.block_log - rl - cl]));
                        }
                        f.push(Factor::Eq(
                            (0..blocks_log)
                                .map(|j| if (block >> j) & 1 == 1 { F128::ONE } else { F128::ZERO })
                                .collect(),
                        ));
                        let k = self.dense.as_ref()?;
                        let used = self.block_log + blocks_log;
                        if k.log_n > used {
                            f.push(Factor::Eq(vec![F128::ZERO; k.log_n - used]));
                        }
                        dense.push((TensorBasis::new(s1 * s2, f), c.value));
                    }
                    block += 1;
                }
            }
        }
        let sparse = (!sparse.is_empty()).then_some(sparse);
        let dense = (!dense.is_empty()).then_some(dense);
        Some((sparse, dense))
    }

    /// Check every `(table index, claim)` against the key. Reads no table.
    pub fn verify(&self, claims: &[(usize, MatrixClaim)], proof: &RootKeyProof) -> bool {
        let n = self.shapes.len();
        if proof.folds.len() != n || claims.iter().any(|(t, _)| *t >= n) {
            return false;
        }
        let mut ch = FsChallenger::new(DOMAIN);
        bind(&mut ch, self, claims);
        let mut one = vec![None; n];
        for i in 0..n {
            let group: Vec<MatrixClaim> = claims.iter().filter(|(t, _)| *t == i).map(|(_, c)| c.clone()).collect();
            match (group.len(), &proof.folds[i]) {
                (0, None) => {}
                (1, None) => one[i] = Some(group[0].clone()),
                (k, Some(p)) if k > 1 => match verify_fold(&group, p, &mut ch) {
                    Ok(c) => one[i] = Some(c),
                    Err(_) => return false,
                },
                _ => return false,
            }
        }
        let Some((sparse_claims, dense_items)) = self.final_claims(&one) else {
            return false;
        };
        let sparse_ok = match (&self.sparse, sparse_claims, &proof.sparse) {
            (Some(k), Some(cs), Some(o)) => verify_sparse_many(k, &cs, o, &mut ch),
            (None, None, None) => true,
            _ => false,
        };
        sparse_ok
            && match (&self.dense, dense_items, &proof.dense) {
                (Some(k), Some(items), Some(o)) => verify_dense_bases(k, &items, o, &mut ch),
                (None, None, None) => true,
                _ => false,
            }
    }
}

/// One line per distinct table: kind, shape and nonzeros (dense: entries).
pub fn describe(tables: &[Table<'_>]) -> String {
    let mut seen: Vec<(u8, usize)> = Vec::new();
    let mut out = String::new();
    for t in tables {
        if seen.contains(&t.id()) {
            continue;
        }
        seen.push(t.id());
        let line = match t {
            Table::Boolean(m) => format!(
                "boolean {}x{} nnz {}",
                m.num_rows,
                m.num_cols,
                m.rows.iter().map(|r| r.len()).sum::<usize>()
            ),
            Table::Element(m) => format!(
                "element {}x{} nnz {}",
                m.num_rows,
                m.num_cols,
                m.rows.iter().map(|r| r.len()).sum::<usize>()
            ),
            Table::Structure(c) => {
                let m = SigmaAssertion::matrix(c);
                format!("structure {}x{}", m.n_rows(), m.n_cols())
            }
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}
