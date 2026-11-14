//! Proves many derivations as pieces of one proof with a shared bus: every
//! piece commits first, the bus challenges are drawn from all commitments,
//! then each piece proves with those challenges and leaves its two bus
//! products public; the pieces together verify when every piece verifies
//! and the products balance over all of them.
//!
//! Batches come from a TSV of `<batch> <file>` lines; each batch is merged
//! into one derivation, built at chunk base `batch << 20` (so row tags never
//! collide across batches), and split into pieces of about CAP rows. One
//! process per step keeps memory flat:
//!
//! ```text
//! lean_pieces commit K CAP BATCHES ROWSDIR OUT   # OUT/K.commit
//! lean_pieces challenges N OUT                   # OUT/challenges, from 0..N
//! lean_pieces prove K CAP BATCHES ROWSDIR OUT    # OUT/K.<i>.proof, verified
//! lean_pieces check N OUT                        # challenges and balance
//! ```

use std::{fs, path::Path};

use flock_core::{circuit::bus::SharedChallenges, pcs::Commitment, proof::R1csProofCircuitMerged};
use flock_prover::lean_rows::{
    LeanCircuit, Rows, same_commitment, shared_balance, shared_challenges,
};

type Statement = Vec<([u8; 32], Commitment)>;

fn batch(k: usize, batches: &str, dir: &str) -> Rows {
    let files: Vec<String> = fs::read_to_string(batches)
        .unwrap()
        .lines()
        .filter_map(|l| {
            let (b, f) = l.split_once('\t')?;
            (b.parse::<usize>().ok()? == k).then(|| f.to_string())
        })
        .collect();
    assert!(!files.is_empty(), "batch {k} is empty");
    Rows::merge(
        files
            .iter()
            .map(|f| {
                Rows::parse(&fs::read_to_string(Path::new(dir).join(f)).unwrap())
                    .unwrap_or_else(|e| panic!("{f}: {e}"))
            })
            .collect(),
    )
}

fn circuit(k: usize, batches: &str, dir: &str) -> LeanCircuit {
    let rows = batch(k, batches, dir);
    LeanCircuit::build_at(&rows, true, k << 20).expect("a consistent derivation")
}

fn read<T: serde::de::DeserializeOwned>(p: impl AsRef<Path>) -> T {
    bincode::deserialize(
        &fs::read(p.as_ref()).unwrap_or_else(|e| panic!("{}: {e}", p.as_ref().display())),
    )
    .unwrap()
}

fn statement(n: usize, out: &str) -> Statement {
    (0..n)
        .flat_map(|k| read::<Statement>(format!("{out}/{k}.commit")))
        .collect()
}

fn label(k: usize, i: usize) -> Vec<u8> {
    format!("cas-lean-piece {k} {i}").into_bytes()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n = |i: usize| a[i].parse::<usize>().unwrap();
    match a[0].as_str() {
        "commit" => {
            let (k, cap, out) = (n(1), n(2), &a[5]);
            let c = circuit(k, &a[3], &a[4]);
            let mut st: Statement = Vec::new();
            for (i, chunks) in c.pieces(cap).iter().enumerate() {
                let t = std::time::Instant::now();
                let p = c.piece(chunks).unwrap();
                st.push((p.digest(), p.commit()));
                eprintln!(
                    "batch {k} piece {i}: {} rows, committed in {:.0} ms",
                    p.rows(),
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            fs::write(
                format!("{out}/{k}.commit"),
                bincode::serialize(&st).unwrap(),
            )
            .unwrap();
        }
        "challenges" => {
            let (count, out) = (n(1), &a[2]);
            let st = statement(count, out);
            let ch = shared_challenges(&st);
            fs::write(
                format!("{out}/challenges"),
                bincode::serialize(&ch).unwrap(),
            )
            .unwrap();
            println!("{} pieces, challenges drawn", st.len());
        }
        "prove" => {
            let (k, cap, out) = (n(1), n(2), &a[5]);
            let ch: SharedChallenges = read(format!("{out}/challenges"));
            let st: Statement = read(format!("{out}/{k}.commit"));
            let c = circuit(k, &a[3], &a[4]);
            let pieces = c.pieces(cap);
            assert_eq!(pieces.len(), st.len(), "the pieces of the first phase");
            for (i, chunks) in pieces.iter().enumerate() {
                let p = c.piece(chunks).unwrap();
                assert_eq!(p.digest(), st[i].0, "the circuit of the first phase");
                let pp = p.prove(Some(&ch), &label(k, i), false);
                assert!(
                    same_commitment(&pp.commitment, &st[i].1),
                    "the commitment of the first phase"
                );
                let (prove_ms, witness_ms) = (pp.prove_ms, pp.witness_ms);
                let bytes = bincode::serialize(&pp.proof).unwrap();
                let path = format!("{out}/{k}.{i}.proof");
                fs::write(&path, &bytes).unwrap();
                drop(pp);
                let proof: R1csProofCircuitMerged = read(&path);
                let verify_ms = p
                    .verify(Some(&ch), &label(k, i), &st[i].1, &proof)
                    .expect("the piece verifies");
                println!(
                    "{k}.{i}\trows {}\twitness {witness_ms:.0} ms\tprove {prove_ms:.0} ms\tverify {verify_ms:.1} ms\tproof {} KiB",
                    p.rows(),
                    bytes.len() / 1024
                );
            }
        }
        "check" => {
            let (count, out) = (n(1), &a[2]);
            let st = statement(count, out);
            let ch: SharedChallenges = read(format!("{out}/challenges"));
            assert_eq!(
                ch,
                shared_challenges(&st),
                "challenges drawn from every commitment"
            );
            let mut proofs = Vec::new();
            let mut bytes = 0;
            for k in 0..count {
                let pieces = read::<Statement>(format!("{out}/{k}.commit")).len();
                for i in 0..pieces {
                    let path = format!("{out}/{k}.{i}.proof");
                    bytes += fs::metadata(&path).unwrap().len();
                    proofs.push(read::<R1csProofCircuitMerged>(&path));
                }
            }
            let t = std::time::Instant::now();
            let ok = shared_balance(&proofs);
            println!(
                "{} pieces, {} KiB of proofs, balance {} ({:.2} ms)",
                proofs.len(),
                bytes / 1024,
                if ok { "holds" } else { "FAILS" },
                t.elapsed().as_secs_f64() * 1e3
            );
            assert!(ok);
        }
        m => panic!("unknown step {m}"),
    }
}
