//! **A LEAN CONE'S PROOF, ON ITS OWN.** What the segments of a Lean tower
//! run hand a verifier who was not there: a [`LeanBundle`] of verification
//! keys and segment root proofs, checked against the cone's
//! [`LeanStatement`] by [`verify_lean_bundle`] with nothing else.
//!
//! A [`LeanVk`] is the circuits a segment's root is checked against (the
//! leaf, the first-level node, the base and the steady spine node) as
//! [`CircuitRecipe`]s, plus the shape constants of the root's published
//! accumulator blocks. The verifier rebuilds each circuit through its
//! validating constructors and computes its digest; the VK's fingerprint
//! ([`LeanVk::fingerprint`]) is the BLAKE3 hash of its bytes, so a
//! consumer who regenerates the VK from the prover's code at the same
//! parameters compares one hash.
//!
//! Per segment the verifier then does what the prover's root did, with
//! nothing taken from the prover's memory: (1) the root proof verifies
//! against the steady circuit, its lincheck tables (from the registry's own
//! matrices) and the VK's PCS parameters; (2) the publics give the
//! segment's commitment root, the challenges' seed and the products of the
//! bus's open sides; (3) the published accumulators are reassembled from
//! the publics alone ([`super::verify::reassemble_named`]) and discharged
//! against the leaf, first-level, base and root tables, sigma included.
//! Over all segments, [`segments_balanced`] binds the challenge root and
//! balances the bus against the statement.

use flock_core::{
    aggregate::{Accumulator, ElementMatrices, TypeMatrices},
    circuit::{Circuit, recipe::CircuitRecipe},
    lincheck::{CscCircuit, LincheckCircuit},
    matrix_fold::{FoldMatrix, MatrixClaim},
    pcs::{Commitment, LOG_PACKING, PcsParams, jagged::JaggedParams},
    schedule::{Registry, TableType},
    union::UnionInstance,
};
use serde::{Deserialize, Serialize};

use super::{
    DOMAIN, F128, FsChallenger, MixedProof,
    config::outer_union,
    lean_leaf::f128_digest,
    lean_tower::{LeanStatement, SegmentClaim, lean_segments_root, segments_balanced},
    node::{digest_f128, entry_live},
    verify::{AccLayout, reassemble_named},
};

/// One outer (node) circuit of a VK, with the PCS parameters its proofs
/// are opened under and the dense bits its jagged layout reads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OuterVk {
    pub circuit: CircuitRecipe,
    pub pcs: PcsParams,
    pub m: usize,
}

/// The verification key of one segment shape.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeanVk {
    /// The envelope `(nu, lanes, m)` the outers were built at.
    pub envelope: (usize, usize, usize),
    pub leaf: CircuitRecipe,
    pub leaf_full_height: bool,
    pub leaf_m: usize,
    pub fl: OuterVk,
    pub base: OuterVk,
    pub steady: OuterVk,
    pub(super) layout: AccLayout,
    /// Where the root's statement words start in its publics.
    pub app_base: usize,
}

const VK_MAGIC: &[u8; 16] = b"cas-lean-vkey001";

impl LeanVk {
    /// A key file: the 16 bytes `cas-lean-vkey001`, then the key's bincode
    /// encoding compressed with zstd.
    pub fn to_bytes(&self) -> Vec<u8> {
        let raw = bincode::serialize(self).expect("a VK serializes");
        let mut out = VK_MAGIC.to_vec();
        out.extend(zstd::bulk::compress(&raw, 9).expect("zstd compresses"));
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let body = b
            .strip_prefix(VK_MAGIC.as_slice())
            .ok_or("not a Lean verification key (bad header)")?;
        let mut raw = Vec::new();
        zstd::stream::copy_decode(body, &mut raw)
            .map_err(|e| format!("the key does not decompress: {e}"))?;
        bincode::deserialize(&raw).map_err(|e| format!("the key does not decode: {e}"))
    }

    /// The BLAKE3 hash of the VK's bytes.
    pub fn fingerprint(&self) -> [u8; 32] {
        *blake3::hash(&bincode::serialize(self).expect("a VK serializes")).as_bytes()
    }
}

/// One segment's root proof: which VK, which leaves, and the proof.
#[derive(Clone, Serialize, Deserialize)]
pub struct LeanSegmentProof {
    /// The index of its VK in the bundle.
    pub vk: u32,
    pub first_leaf: u64,
    pub leaves: u64,
    pub public: Vec<F128>,
    pub(crate) proof: MixedProof,
    pub commitment: Commitment,
}

impl std::fmt::Debug for LeanSegmentProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeanSegmentProof")
            .field("vk", &self.vk)
            .field("first_leaf", &self.first_leaf)
            .field("leaves", &self.leaves)
            .finish_non_exhaustive()
    }
}

/// What a cone proves, as the verifier receives it on the shared bus: the
/// chunk words, the keys of the declarations checked, the constants the
/// rules read and the steps of arithmetic on literals.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeanStatementWire {
    pub chunk_words: Vec<u64>,
    pub keys: Vec<[F128; 2]>,
    pub prims: Vec<(u64, [F128; 2])>,
    /// `(op, op key, x, y, out)` per step.
    pub arith: Vec<(u64, [F128; 2], u64, u64, [F128; 2])>,
}

impl From<&LeanStatement> for LeanStatementWire {
    fn from(s: &LeanStatement) -> Self {
        Self {
            chunk_words: s.chunk_words.clone(),
            keys: s.keys.clone(),
            prims: s.prims.clone(),
            arith: s.arith.iter().map(|a| (a.op, a.key, a.x, a.y, a.out)).collect(),
        }
    }
}

impl LeanStatementWire {
    pub fn statement(&self) -> LeanStatement {
        LeanStatement {
            chunk_words: self.chunk_words.clone(),
            keys: self.keys.clone(),
            prims: self.prims.clone(),
            arith: self
                .arith
                .iter()
                .map(|&(op, key, x, y, out)| crate::lean_rows::Arith { op, key, x, y, out })
                .collect(),
        }
    }

    /// The statement's identity: the BLAKE3 hash of its bytes.
    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&bincode::serialize(self).expect("a statement serializes")).as_bytes()
    }
}

/// A whole cone's proof: what it proves, the verification keys, and one
/// root proof per segment.
#[derive(Clone, Serialize, Deserialize)]
pub struct LeanBundle {
    pub statement: LeanStatementWire,
    /// The fingerprints of the verification keys the segments name (a key
    /// is large and shared by every proof at its parameters, so it travels
    /// in a file of its own: [`LeanVk::to_bytes`]).
    pub vks: Vec<[u8; 32]>,
    pub segments: Vec<LeanSegmentProof>,
}

const BUNDLE_MAGIC: &[u8; 16] = b"cas-lean-bundle1";

impl LeanBundle {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = BUNDLE_MAGIC.to_vec();
        bincode::serialize_into(&mut out, self).expect("a bundle serializes");
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let body = b
            .strip_prefix(BUNDLE_MAGIC.as_slice())
            .ok_or("not a Lean cone bundle (bad header)")?;
        bincode::deserialize(body).map_err(|e| format!("the bundle does not decode: {e}"))
    }
}

/// What a green verification established.
#[derive(Clone, Debug)]
pub struct LeanVerified {
    pub statement: [u8; 32],
    /// Per VK: its fingerprint and its four circuits' digests (leaf, FL,
    /// base, steady).
    pub vks: Vec<([u8; 32], [[u8; 32]; 4])>,
    pub segments: usize,
    pub leaves: u64,
    pub challenge_root: [u8; 32],
    pub verify_ms: f64,
}

/// A VK's circuits, rebuilt.
struct Material {
    leaf: Circuit,
    fl: Circuit,
    base: Circuit,
    steady: Circuit,
    leaf_jp: JaggedParams,
    fl_jp: JaggedParams,
    base_jp: JaggedParams,
    steady_jp: JaggedParams,
}

fn jagged_of(u: &UnionInstance<'_>, m: usize) -> Result<JaggedParams, String> {
    let dense = m
        .checked_sub(LOG_PACKING)
        .ok_or("a VK's dense bits are below the packing")?;
    Ok(JaggedParams::from_heights(&u.jagged_heights(), u.n_log(), dense))
}

/// The tables a key names instead of shipping: the BLAKE3 block every
/// outer verifies its children's hashes with (about 170 MiB as matrices).
const CHAIN_BLAKE3: &str = "flock-chain-blake3";

/// The prover's namer for an outer registry at `nu` rows.
pub(super) fn outer_namer(nu: usize) -> impl Fn(&TableType) -> Option<(String, usize)> {
    let b3 = super::fl_node::chain_blake_r1cs(nu);
    move |t: &TableType| {
        std::sync::Arc::ptr_eq(&t.a_0.rows, &b3.a_0.rows).then(|| (CHAIN_BLAKE3.to_string(), nu))
    }
}

/// The verifier's resolver: the named tables, built from this code.
pub(super) fn resolve(name: &str, nu: usize) -> Option<TableType> {
    (name == CHAIN_BLAKE3 && (8..=24).contains(&nu))
        .then(|| TableType::from_block_r1cs(&super::fl_node::chain_blake_r1cs(nu)))
}

fn rebuild(vk: &LeanVk) -> Result<Material, String> {
    let r = |c: &CircuitRecipe, what: &str| {
        c.rebuild_with(&resolve)
            .map_err(|e| format!("the VK's {what} circuit does not rebuild: {e:?}"))
    };
    let leaf = r(&vk.leaf, "leaf")?;
    let fl = r(&vk.fl.circuit, "first-level")?;
    let base = r(&vk.base.circuit, "base")?;
    let steady = r(&vk.steady.circuit, "steady")?;
    let leaf_jp = {
        let mut u = UnionInstance::new(leaf.registry(), leaf.counts().to_vec());
        if vk.leaf_full_height {
            u.set_full_height_commit();
        }
        jagged_of(&u, vk.leaf_m)?
    };
    let outer = |c: &Circuit, m: usize| {
        jagged_of(&outer_union(c.registry(), c.counts().to_vec()), m)
    };
    let fl_jp = outer(&fl, vk.fl.m)?;
    let base_jp = outer(&base, vk.base.m)?;
    let steady_jp = outer(&steady, vk.steady.m)?;
    Ok(Material {
        leaf,
        fl,
        base,
        steady,
        leaf_jp,
        fl_jp,
        base_jp,
        steady_jp,
    })
}

fn boolean_mats(r: &Registry) -> Vec<TypeMatrices<'_>> {
    r.boolean_types().iter().map(|t| (&t.a_0, &t.b_0)).collect()
}

fn element_mats(r: &Registry) -> Vec<ElementMatrices<'_>> {
    r.element_types()
        .iter()
        .filter_map(|t| t.element_type())
        .map(|e| (e.a_0(), e.b_0()))
        .collect()
}

/// The root's accumulated claims against the VK's tables: the lane's
/// against the leaf, the main fold's against the first-level, root and
/// base circuits, and the passenger's against the base.
fn discharge(
    m: &Material,
    lane: &Accumulator,
    main: &Accumulator,
    passenger: &[([F128; 2], MatrixClaim)],
    expect_passenger: bool,
) -> Result<(), String> {
    // The lane, about the leaves.
    let leaf_d = m.leaf.digest();
    if !lane.discharge(&boolean_mats(m.leaf.registry())) {
        return Err("the lane's boolean claims".into());
    }
    if !lane.discharge_element(&element_mats(m.leaf.registry())) {
        return Err("the lane's element claims".into());
    }
    if !lane.discharge_sigma(&[&m.leaf]) {
        return Err("the lane's wiring claim".into());
    }
    if !lane.discharge_jagged(&[(leaf_d, &m.leaf_jp)]) {
        return Err("the lane's jagged claim".into());
    }
    // The main fold, about the first-level node, the root and the base.
    let fold: Vec<(&dyn FoldMatrix, &dyn FoldMatrix)> = m
        .fl
        .registry()
        .boolean_types()
        .iter()
        .map(|t| (&t.a_0 as &dyn FoldMatrix, &t.b_0 as &dyn FoldMatrix))
        .collect();
    if !main.discharge_dyn(&fold) {
        return Err("the main fold's boolean claims".into());
    }
    if !main.discharge_element(&element_mats(m.fl.registry())) {
        return Err("the main fold's element claims".into());
    }
    if !main.discharge_sigma(&[&m.fl, &m.steady, &m.base]) {
        return Err("the main fold's wiring claims".into());
    }
    let layouts = [
        (m.fl.digest(), &m.fl_jp),
        (m.steady.digest(), &m.steady_jp),
        (m.base.digest(), &m.base_jp),
    ];
    if !main.discharge_jagged(&layouts) {
        return Err("the main fold's jagged claims".into());
    }
    // The passenger: the base's two orphaned claims.
    let live = passenger.iter().any(|(_, c)| entry_live(c));
    if live != expect_passenger {
        return Err("the passenger's liveness".into());
    }
    if live {
        let base_d = m.base.digest();
        if passenger.len() != 2
            || passenger.iter().any(|(k, c)| *k != digest_f128(&base_d) || !entry_live(c))
        {
            return Err("the passenger's keys".into());
        }
        let pacc = Accumulator {
            registry_digest: main.registry_digest,
            per_type: Vec::new(),
            per_element: Vec::new(),
            per_linear: Vec::new(),
            sigma: vec![(base_d, passenger[0].1.clone())],
            jagged: vec![(base_d, passenger[1].1.clone())],
        };
        if !pacc.discharge_sigma(&[&m.base]) || !pacc.discharge_jagged(&[(base_d, &m.base_jp)]) {
            return Err("the passenger's claims".into());
        }
    }
    Ok(())
}

/// Verify one segment's root against its VK: the claim it makes on the
/// shared bus.
fn verify_segment(
    vk: &LeanVk,
    m: &Material,
    s: &LeanSegmentProof,
) -> Result<SegmentClaim, String> {
    if s.leaves < 6 || s.leaves % 2 != 0 {
        return Err("a segment is a spine over at least three leaf pairs".into());
    }
    // (1) The root proof, against the steady circuit and its own tables.
    let st = &m.steady;
    let u = outer_union(st.registry(), st.counts().to_vec());
    let csc: Vec<CscCircuit> = st
        .registry()
        .boolean_types()
        .iter()
        .map(|t| CscCircuit::from_matrices(&t.a_0, &t.b_0).with_const_pin(t.const_pin))
        .collect();
    let lcs: Vec<&dyn LincheckCircuit> = csc.iter().map(|c| c as &dyn LincheckCircuit).collect();
    if s.commitment.params != vk.steady.pcs {
        return Err("the root's commitment is not under the VK's parameters".into());
    }
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    s.proof
        .verify_circuit(&u, st, &s.public, &lcs, &s.commitment, &vk.steady.pcs, &mut ch)
        .map_err(|e| format!("the root proof does not verify: {e:?}"))?;
    // (2) The statement words.
    let p = &s.public;
    let a = vk.app_base;
    if p.len() < a + 6 {
        return Err("the publics are shorter than the statement".into());
    }
    let root = f128_digest([p[a], p[a + 1]]);
    let seed = f128_digest([p[a + 2], p[a + 3]]);
    let (send, recv) = (p[a + 4], p[a + 5]);
    if send == F128::ZERO {
        return Err("a zero product".into());
    }
    // (3) The accumulators, from the publics alone, then their discharge.
    let (main, passenger, lane) = reassemble_named(
        p,
        &vk.layout,
        [m.fl.digest(), m.steady.digest(), m.base.digest()],
        st.registry(),
        m.leaf.digest(),
        m.leaf.registry().digest(),
    )
    .map_err(|e| format!("the published accumulators: {e:?}"))?;
    let k = s.leaves / 2;
    discharge(m, &lane, &main, &passenger, k >= 4)?;
    Ok(SegmentClaim {
        root,
        seed,
        send,
        recv,
    })
}

/// One segment alone against one VK (the prover's check of what it
/// saves): its claim on the shared bus.
pub fn check_segment(vk: &LeanVk, s: &LeanSegmentProof) -> Result<SegmentClaim, String> {
    under_envelope(vk, || verify_segment(vk, &rebuild(vk)?, s))
}

/// Run `f` with the process-wide envelope set to `vk`'s (the outers' unions
/// read it), and a panic inside a check turned into a refusal.
fn under_envelope<T>(vk: &LeanVk, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let (nu, lanes, m) = vk.envelope;
    super::envelope::set_envelope_override(Some((nu, lanes, m)));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    super::envelope::set_envelope_override(None);
    r.unwrap_or_else(|p| {
        let msg = p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        Err(format!("a check failed: {msg}"))
    })
}

/// **VERIFY A LEAN CONE BUNDLE.** Rebuild every VK's circuits, verify
/// every segment's root against its VK, and check that the segments
/// together prove `statement`: one challenge root over theirs, and the
/// shared bus balanced against the statement's chunk words, checked keys,
/// constants and arithmetic.
pub fn verify_lean_bundle(bundle: &LeanBundle, keys: &[LeanVk]) -> Result<LeanVerified, String> {
    let t = std::time::Instant::now();
    let statement = &bundle.statement.statement();
    if bundle.segments.is_empty() {
        return Err("no segments".into());
    }
    let prints: Vec<[u8; 32]> = keys.iter().map(LeanVk::fingerprint).collect();
    let vks: Vec<&LeanVk> = bundle
        .vks
        .iter()
        .map(|f| {
            prints
                .iter()
                .position(|p| p == f)
                .map(|i| &keys[i])
                .ok_or_else(|| format!("no verification key blake3:{}", hex(f)))
        })
        .collect::<Result<_, _>>()?;
    let mats: Vec<Material> = vks
        .iter()
        .map(|vk| under_envelope(vk, || rebuild(vk)))
        .collect::<Result<_, _>>()?;
    let mut claims = Vec::with_capacity(bundle.segments.len());
    let mut next = 0u64;
    for (i, s) in bundle.segments.iter().enumerate() {
        if s.first_leaf != next {
            return Err(format!("segment {i} does not start where the last ended"));
        }
        next += s.leaves;
        let v = s.vk as usize;
        let (vk, m) = vks
            .get(v)
            .zip(mats.get(v))
            .ok_or(format!("segment {i} names no VK"))?;
        claims.push(
            under_envelope(vk, || verify_segment(vk, m, s)).map_err(|e| format!("segment {i}: {e}"))?,
        );
    }
    let roots: Vec<[u8; 32]> = claims.iter().map(|c| c.root).collect();
    let challenge_root = if roots.len() == 1 {
        roots[0]
    } else {
        lean_segments_root(&roots)
    };
    if !segments_balanced(&challenge_root, &claims, statement)? {
        return Err("the shared bus does not balance against the statement".into());
    }
    Ok(LeanVerified {
        statement: bundle.statement.digest(),
        vks: bundle
            .vks
            .iter()
            .zip(&mats)
            .map(|(f, m)| {
                (
                    *f,
                    [m.leaf.digest(), m.fl.digest(), m.base.digest(), m.steady.digest()],
                )
            })
            .collect(),
        segments: bundle.segments.len(),
        leaves: next,
        challenge_root,
        verify_ms: t.elapsed().as_secs_f64() * 1e3,
    })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
