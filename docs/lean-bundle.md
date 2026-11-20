# The Lean cone bundle, format 3

A **bundle** is one file that proves a Lean cone: the statement it proves, the
fingerprints of the verification keys its proofs are checked against, and one
root proof per segment of the tower. A **verification key** is a file of its
own (`vk-<fingerprint>.vk`): it is shared by every proof made at the same
parameters, so it is fetched once. `lean_tower verify BUNDLE [KEY ...]` reads
the bundle and its keys (by default the `vk-*.vk` files beside the bundle) and
nothing else. Code: `crates/flock-prover/src/tower/lean_verify.rs`
(types, `verify_lean_bundle`) and `crates/flock-core/src/circuit/recipe.rs`
(circuits as recipes).

## Encoding

The file is the 16 bytes `cas-lean-bundle3` followed by the
[bincode 1.3](https://docs.rs/bincode/1.3) encoding (its default
configuration) of a `LeanBundle`:

- integers are little-endian at fixed width; `usize` is 8 bytes;
- a sequence (`Vec`, map) is a `u64` length and its elements;
- an `Option` is one byte (0 none, 1 some) and the value;
- an enum is a `u32` variant index and the variant's fields;
- a struct or tuple is its fields in order, with no tags;
- a field element `F128` is two `u64`, `lo` then `hi`;
- a hash or digest `[u8; 32]` is 32 bytes.

The last character of the header is the format number. Any change to any
type below, or to the proof types it contains, is a new format number;
readers refuse a header they do not know.

## Contents

```
LeanBundle {
    statement: LeanStatementWire,
    vks:       Vec<[u8; 32]>,          // key fingerprints, named by index
    segments:  Vec<LeanSegmentProof>,
}

LeanStatementWire {                 // what the verifier sends on the shared bus
    chunk_words: Vec<u64>,          // one distinct word per row chunk
    keys:        Vec<[F128; 2]>,    // the keys of the declarations checked
    prims:       Vec<(u64, [F128; 2])>,              // constant number, its key
    arith:       Vec<(u64, [F128; 2], u64, u64, [F128; 2])>, // op, op key, x, y, out
    facts:       Vec<[(u64, u64); 11]>, // what the declarations are (below)
}

LeanVk {
    envelope:         (usize, usize, usize),   // nu, lanes, m of the outers
    leaf:             CircuitRecipe,
    leaf_full_height: bool,
    leaf_m:           usize,
    fl, base, steady: OuterVk,                 // first-level node, base, spine node
    layout:           AccLayout,               // where the root publishes its accumulators
    app_base:         usize,                   // where the root's statement words start
}

OuterVk { circuit: CircuitRecipe, pcs: PcsParams, m: usize }

LeanSegmentProof {
    vk:         u32,        // index into vks
    first_leaf: u64,
    leaves:     u64,
    public:     Vec<F128>,  // the root's public words
    proof:      MixedProof, // the root's circuit proof
    commitment: Commitment, // the root's commitment
}
```

A key file is the 16 bytes `cas-lean-vkey001` followed by the
[zstd](https://facebook.github.io/zstd/) compression of the bincode encoding
of a `LeanVk`.

`CircuitRecipe` holds the inputs of a circuit's validating constructors
(registry types and their matrices, gate counts, fixed public words, the
wiring, the builder's optional steps and the bus), never its derived tables.
The wiring is a byte string of LEB128 varints: the number of classes, then
per class its length and each cell index's distance from the one before it
(the first cell's from the previous class's first; classes are ascending and
ordered by their least cell). A registry type is a Boolean table (its three
matrices), an element table, or a **named** table the verifier builds from
its own code: today only `flock-chain-blake3` at a row count, the BLAKE3
block every node uses to check its children (about 170 MiB as matrices). The
verifier rebuilds every circuit through its constructors, which check it
again, and computes its digest. `MixedProof`,
`Commitment`, `PcsParams` and `AccLayout` are the flock types of the same
names, encoded field by field as above.

The **statement digest** is the BLAKE3 hash of the bincode encoding of
`LeanStatementWire`; a **verification key's fingerprint** is the BLAKE3 hash of
the encoding of its `LeanVk`. The verifier prints both.

Sizes for the shuffle paper's cone: about 490 KiB per segment proof, about
800 KiB per verification key; reading and checking one segment takes about
9 s, most of it rebuilding the key's circuits.

## What verification checks

Per verification key: every circuit rebuilds. Per segment, against its key:
the root proof verifies against the spine circuit (its lincheck tables are
the registry's own matrices; the commitment is under the key's PCS
parameters); the publics give the segment's commitment root, the challenges'
seed and the bus's two products; the accumulators the root publishes are
read from the publics alone and discharged against the leaf, first-level,
base and spine circuits (wiring included). Segments must cover the leaves in
order. Over all segments: the challenge root is the hash of the segments'
roots, every segment drew its challenges from it, and the shared bus
balances against the statement.

## What the declarations are

A checked declaration is named by its key, which is a hash of its name. So
that the proof says what the declarations *are*, the verifier also receives
`facts`: the tuples the shared side's rows send for every declaration (its
row: unfolding, level count, value and type ids; its key; what the
reduction rules read of it), every recursor rule, and every term, level and
level list the declarations' types, the values of those other than theorems
and the rules' right-hand sides reach. Those rows send each once more, so
the bus balances only if the cone's rows are these. `check_facts`
(`lean_rows.rs`) compares them, by structure and whatever the ids, with the
declarations' text: cas-lean turns the declarations of its bundle into rows
and refuses a cone that has a declaration its text does not, or one that
differs in anything above. For the shuffle paper's cone, about 155 000
facts (148 000 terms of 4.2 million). Format 1 (`cas-lean-bundle1`) had no
facts; it is still read, and says so.

## What it does not check yet

The circuits in a key are rebuilt and checked for consistency, but nothing
in the file says they are *the* Lean checker's circuits: a key is trusted by
its fingerprint. Compare the printed fingerprint with the one published for
the prover version that made the proof. Regenerating a key from the prover's
code and parameters is the next step.
