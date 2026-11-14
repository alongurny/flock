//! Proves a cone of Lean declarations with linked citations: one shared
//! numbering of terms and declarations (proven once, the shared side) and
//! batches of judgements and goals over it, all pieces of one proof with a
//! shared bus. A citation of a declaration receives the row that its own
//! goal (in whichever batch) vouches for, and names an earlier declaration.
//!
//! ```text
//! lean_cone cone BATCHES ROWSDIR OUT        # OUT/cone.bin, OUT/b<k>.bin
//! lean_cone commit K CAP OUT                # batch K: OUT/K.asks, OUT/K.commit
//! lean_cone commit-shared N CAP OUT         # OUT/s.commit
//! lean_cone challenges N OUT
//! lean_cone prove K CAP OUT                 # OUT/K.<i>.proof, verified
//! lean_cone prove-shared N CAP OUT          # OUT/s.<i>.proof, verified
//! lean_cone check N OUT
//! ```

use std::{collections::HashMap, fs, path::Path, time::Instant};

use flock_core::{circuit::bus::SharedChallenges, pcs::Commitment, proof::R1csProofCircuitMerged};
use flock_prover::lean_rows::{
    Cone, LeanCircuit, Rows, SHARED_BASE, Side, Tuple, same_commitment, shared_balance,
    shared_challenges,
};

type Statement = Vec<([u8; 32], Commitment)>;

fn read<T: serde::de::DeserializeOwned>(p: impl AsRef<Path>) -> T {
    let bytes = fs::read(p.as_ref()).unwrap_or_else(|e| panic!("{}: {e}", p.as_ref().display()));
    bincode::deserialize(&bytes).unwrap()
}
fn write<T: serde::Serialize>(p: impl AsRef<Path>, x: &T) {
    fs::write(p, bincode::serialize(x).unwrap()).unwrap();
}

fn batch_base(k: usize) -> usize {
    (k + 1) << 20
}

/// The tag of a tuple (its first word).
fn tag(t: &Tuple) -> u64 {
    t[0].0
}

fn build_batch(k: usize, out: &str) -> (LeanCircuit, HashMap<Tuple, u64>) {
    let g: Rows = read(format!("{out}/cone.bin"));
    let b: Rows = read(format!("{out}/b{k}.bin"));
    let (c, left) = LeanCircuit::build_side(&g, &b, Side::Batch, batch_base(k), &HashMap::new())
        .unwrap_or_else(|e| panic!("batch {k}: {e}"));
    // What is left is asked of the shared side: terms and declarations.
    let bad: Vec<_> = left.keys().filter(|t| ![1, 3].contains(&tag(t))).collect();
    assert!(
        bad.is_empty(),
        "batch {k} asks for {} tuples no one gives, e.g. {:?}",
        bad.len(),
        bad[0]
    );
    (c, left)
}

fn build_shared(n: usize, out: &str) -> LeanCircuit {
    let g: Rows = read(format!("{out}/cone.bin"));
    let mut outside: HashMap<Tuple, u64> = HashMap::new();
    for k in 0..n {
        for (t, c) in read::<Vec<(Tuple, u64)>>(format!("{out}/{k}.asks")) {
            *outside.entry(t).or_default() += c;
        }
    }
    let (c, left) =
        LeanCircuit::build_side(&g, &Rows::default(), Side::Shared, SHARED_BASE, &outside)
            .unwrap_or_else(|e| panic!("shared side: {e}"));
    let bad: Vec<_> = left.keys().filter(|t| tag(t) != 6).collect();
    assert!(
        bad.is_empty(),
        "the batches ask for {} tuples the shared side lacks, e.g. {:?}",
        bad.len(),
        bad[0]
    );
    eprintln!(
        "shared side: {} checked declarations await their goals",
        left.len()
    );
    c
}

fn commit(c: &LeanCircuit, cap: usize, name: &str, out: &str) {
    let mut st: Statement = Vec::new();
    for (i, chunks) in c.pieces(cap).iter().enumerate() {
        let p = c.piece(chunks).unwrap();
        st.push((p.digest(), p.commit()));
        eprintln!("{name}.{i}: {} rows committed", p.rows());
    }
    write(format!("{out}/{name}.commit"), &st);
}

fn prove(c: &LeanCircuit, cap: usize, name: &str, out: &str) {
    let ch: SharedChallenges = read(format!("{out}/challenges"));
    let st: Statement = read(format!("{out}/{name}.commit"));
    let pieces = c.pieces(cap);
    assert_eq!(pieces.len(), st.len(), "the pieces of the first phase");
    for (i, chunks) in pieces.iter().enumerate() {
        let p = c.piece(chunks).unwrap();
        assert_eq!(p.digest(), st[i].0, "the circuit of the first phase");
        let label = format!("cas-lean-cone {name} {i}").into_bytes();
        let pp = p.prove(Some(&ch), &label, false);
        assert!(
            same_commitment(&pp.commitment, &st[i].1),
            "the commitment of the first phase"
        );
        let (prove_ms, witness_ms) = (pp.prove_ms, pp.witness_ms);
        let path = format!("{out}/{name}.{i}.proof");
        write(&path, &pp.proof);
        drop(pp);
        let proof: R1csProofCircuitMerged = read(&path);
        let verify_ms = p
            .verify(Some(&ch), &label, &st[i].1, &proof)
            .expect("the piece verifies");
        println!(
            "{name}.{i}\trows {}\twitness {witness_ms:.0} ms\tprove {prove_ms:.0} ms\tverify {verify_ms:.1} ms\tproof {} KiB",
            p.rows(),
            fs::metadata(&path).unwrap().len() / 1024
        );
    }
}

fn names(n: usize) -> Vec<String> {
    std::iter::once("s".to_string())
        .chain((0..n).map(|k| k.to_string()))
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n = |i: usize| a[i].parse::<usize>().unwrap();
    match a[0].as_str() {
        "cone" => {
            let (batches, dir, out) = (&a[1], &a[2], &a[3]);
            let mut by: Vec<Vec<String>> = Vec::new();
            for l in fs::read_to_string(batches).unwrap().lines() {
                let (b, f) = l.split_once('\t').unwrap();
                let b: usize = b.parse().unwrap();
                if by.len() <= b {
                    by.resize(b + 1, Vec::new());
                }
                by[b].push(f.to_string());
            }
            let t = Instant::now();
            let mut cone = Cone::new();
            let (mut facts, mut goals) = (0, 0);
            for (k, files) in by.iter().enumerate() {
                let mut batch = Rows::default();
                for f in files {
                    let text = fs::read_to_string(Path::new(dir).join(f)).unwrap();
                    let r = Rows::parse(&text).unwrap_or_else(|e| panic!("{f}: {e}"));
                    cone.add(r, &mut batch)
                        .unwrap_or_else(|e| panic!("{f}: {e}"));
                }
                let (_, _, nf) = batch.counts();
                facts += nf;
                goals += files.len();
                write(format!("{out}/b{k}.bin"), &batch);
                eprintln!("batch {k}: {} files, {nf} facts", files.len());
            }
            let (terms, checked, block, free) = cone.stats();
            write(format!("{out}/cone.bin"), &cone.finish());
            println!(
                "{} batches, {goals} goals, {facts} facts; shared: {terms} terms, declarations {checked} checked, {block} in inductive blocks, {free} free; {:.0} s",
                by.len(),
                t.elapsed().as_secs_f64()
            );
        }
        "commit" => {
            let (k, cap, out) = (n(1), n(2), &a[3]);
            let (c, left) = build_batch(k, out);
            let asks: Vec<(Tuple, u64)> = left.into_iter().collect();
            write(format!("{out}/{k}.asks"), &asks);
            commit(&c, cap, &k.to_string(), out);
        }
        "commit-shared" => {
            let (cnt, cap, out) = (n(1), n(2), &a[3]);
            let c = build_shared(cnt, out);
            commit(&c, cap, "s", out);
        }
        "challenges" => {
            let (cnt, out) = (n(1), &a[2]);
            let st: Statement = names(cnt)
                .iter()
                .flat_map(|m| read::<Statement>(format!("{out}/{m}.commit")))
                .collect();
            write(format!("{out}/challenges"), &shared_challenges(&st));
            println!("{} pieces, challenges drawn", st.len());
        }
        "prove" => {
            let (k, cap, out) = (n(1), n(2), &a[3]);
            let (c, _) = build_batch(k, out);
            prove(&c, cap, &k.to_string(), out);
        }
        "prove-shared" => {
            let (cnt, cap, out) = (n(1), n(2), &a[3]);
            let c = build_shared(cnt, out);
            prove(&c, cap, "s", out);
        }
        "check" => {
            let (cnt, out) = (n(1), &a[2]);
            let mut st: Statement = Vec::new();
            let mut proofs = Vec::new();
            let mut bytes = 0;
            for m in names(cnt) {
                let s: Statement = read(format!("{out}/{m}.commit"));
                for i in 0..s.len() {
                    let path = format!("{out}/{m}.{i}.proof");
                    bytes += fs::metadata(&path).unwrap().len();
                    proofs.push(read::<R1csProofCircuitMerged>(&path));
                }
                st.extend(s);
            }
            let ch: SharedChallenges = read(format!("{out}/challenges"));
            assert_eq!(
                ch,
                shared_challenges(&st),
                "challenges drawn from every commitment"
            );
            let ok = shared_balance(&proofs);
            println!(
                "{} pieces, {} KiB of proofs, balance {}",
                proofs.len(),
                bytes / 1024,
                if ok { "holds" } else { "FAILS" }
            );
            assert!(ok);
        }
        m => panic!("unknown step {m}"),
    }
}
