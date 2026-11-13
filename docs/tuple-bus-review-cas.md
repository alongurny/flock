# The tuple bus as a lookup for cas: a soundness review

Review of `docs/tuple-bus-design.md` and `flock-core/src/circuit/bus.rs`
(at `612fcdf`), for the uses cas plans: chunk-with-path rows (a changed
chunk proven against the old blob's BLAKE3 tree, alon/cas issue #8), a
manifest checked against its commit (#1), and the table checker for Lean
declarations (#10). Asked in alon/cas issue #9.

## Verdict

The argument is sound for what it states: the multiset of sent tuples equals
the multiset of received tuples, except with probability at most
`N / 2^128` (N live entries), brought to about `2^-128` by the grind. It is
not yet a lookup argument: it has no multiplicities. cas can use it as it is
for one-to-one routing (each value used once), and needs one addition,
below, before a table can be read many times.

## What was checked

1. **Challenges after the commitment.** `prove_bus` / `verify_bus` observe
   the label `flock-tuple-bus-v1` and sample `β` and then `γ_1..γ_w` on the
   wiring's transcript, after the wiring GKR, which runs after the trace
   commitment and the circuit digest (the bus is part of the digest). So the
   tuples are fixed before `(β, γ)` are drawn. Correct.
2. **Field size.** Challenges are GF(2^128) elements. With one challenge per
   coordinate, each fingerprint `β + Σ γ_k t_k` is of degree 1, the product
   of each side has degree N, and Schwartz–Zippel gives `N / 2^128`. For cas
   sizes (a commit's blobs: up to about 10^6 entries) that is `2^-108`
   before grinding and `2^-128` after the target grind (§1.9). Correct.
3. **Characteristic 2.** Unique factorization holds in `F[B, G_1..G_w]` in
   any characteristic, so equal products mean equal multisets, repeated
   tuples included (`Φ^2` on both sides is fine). The zero-product check
   (§1.3) closes the one degenerate case. Correct.
4. **Dead rows.** Leaves of dead rows are 1; the binding of gathers to the
   commitment relies on dead rows being zero in the committed buffer (the
   "dead-row invariant", wiring item 5). cas's circuits use
   `set_full_height_commit`, where the invariant is enforced by the pin
   rows; any new cas table must keep it.

## Findings for cas

1. **A row tag added to a word does not separate anything.** A coordinate
   is `word + constant + row_tag`, and addition in GF(2^128) is XOR. So
   `(word a, tag x)` and `(word a ⊕ x ⊕ y, tag y)` give the same coordinate.
   leanXMSS is safe because its tags sit in their own coordinate (`S`) or
   on rows where the word is fixed. Rule for cas: a table id, a blob index
   or a row position goes in a coordinate of its own, never added to a
   witness word.
2. **No multiplicities.** Each tuple is sent once and received once (§1.5).
   A lookup reads a table entry any number of times, so as it stands the
   table side would need one row per read, with the count known when the
   circuit is built. For chunk paths that is acceptable (each parent node of
   a BLAKE3 tree is read by at most two children, known from the lengths in
   the manifest). For the Lean tables it is not: a term is cited an unknown
   number of times.
   - LogUp (`Σ 1/(β + φ_read) = Σ m/(β + φ_table)`) does **not** carry over:
     in characteristic 2 the multiplicity `m` is reduced mod 2.
   - The fix that does carry over is the one Binius's M3 channels use: a
     committed multiplicity column `m` in bits `m_0..m_{k-1}` and the
     table's leaf `∏_i (β + φ)^{2^i · m_i}`, the powers made by repeated
     squaring (a linear map in characteristic 2, so cheap in the GKR's
     leaves). The degree bound becomes `Σ m ≤ N`, unchanged in order.
3. **Keys are two coordinates.** A BLAKE3 key is 256 bits: two GF(2^128)
   coordinates, each with its own `γ`. Fine with per-coordinate challenges;
   do not pack a key into one coordinate by a power of one challenge.
4. **Recursion.** The tower's in-circuit verifier does not replay the bus
   (§1.5). cas does not aggregate proofs by recursion yet; when it does, the
   bus has to be taught to the child walker first.
5. **The counting argument is per circuit.** leanXMSS proves its routing
   correct by a count (§2.3) that depends on gadget constraints. Every cas
   use needs its own such argument, written next to the circuit, saying why
   a balanced bus means the right rows were matched.

## Decision

Use the bus for cas's chunk-path rows (one-to-one, finding 2 does not bite),
with tags in their own coordinates. Add multiplicities (finding 2) before
the Lean table checker. A prover with an established lookup is not needed
for either.
