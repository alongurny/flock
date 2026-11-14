//! One proof for each commit of a replayed history: every blob the commit
//! adds, of every type, proven valid together.
//!
//! cas_commits <percommit.bin> [<first> [<count> [<step>]]]: the file is
//! what `percommit` in proof-cost writes. One line for each commit proven:
//! its number, blobs, bytes, compressions, type rows, table size, and the
//! seconds of setup, witness and proof, and verification.
use std::time::Instant;

use flock_prover::{cas_ir, cas_typed::TypedCircuit};

fn main() {
    let a: Vec<String> = std::env::args().filter(|a| a != "--bench").collect();
    let arg = |i: usize, d: usize| a.get(i).map_or(d, |n| n.parse().unwrap());
    let (first, count, step) = (arg(2, 0), arg(3, usize::MAX), arg(4, 1));
    let data = std::fs::read(&a[1]).unwrap();
    let types = cas_ir::types();
    let ty = |alias: &str| {
        let alias = if alias == "note" || alias == "author" {
            "text"
        } else {
            alias
        };
        types
            .iter()
            .find(|t| t.alias == alias)
            .unwrap_or_else(|| panic!("no type {alias}"))
    };
    let (mut p, mut commit, mut done) = (0, 0, 0);
    println!(
        "commit,blobs,bytes,compressions,rows,nu,setup_s,prove_s,prove_again_s,verify_s,proof_bytes"
    );
    while p < data.len() && done < count {
        let n = u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let (mut typed, mut keys, mut blobs) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let l = data[p] as usize;
            let alias = std::str::from_utf8(&data[p + 1..p + 1 + l]).unwrap();
            p += 1 + l;
            let key: [u8; 32] = data[p..p + 32].try_into().unwrap();
            let len = u32::from_le_bytes(data[p + 32..p + 36].try_into().unwrap()) as usize;
            typed.push((ty(alias), len));
            keys.push(key);
            blobs.push(data[p + 36..p + 36 + len].to_vec());
            p += 36 + len;
        }
        commit += 1;
        if commit <= first || (commit - first - 1) % step != 0 {
            continue;
        }
        done += 1;
        let bytes: usize = blobs.iter().map(Vec::len).sum();
        let start = Instant::now();
        let circuit = TypedCircuit::new(&typed, true);
        let ts = start.elapsed();
        // A circuit's first proof is slower than its later ones: both are
        // given, the first as `prove_s`.
        let mut first = 0.0;
        for timed in [false, true] {
            let start = Instant::now();
            let w = circuit.witness(&keys, &blobs);
            let proof = circuit.prove(&w);
            let tp = start.elapsed();
            let size = bincode::serialize(&proof.proof).unwrap().len()
                + bincode::serialize(&proof.commitment).unwrap().len();
            let start = Instant::now();
            circuit.verify(&keys, &proof).expect("verifies");
            let tv = start.elapsed();
            if !timed {
                first = tp.as_secs_f64();
            }
            if timed {
                println!(
                    "{},{n},{bytes},{},{},{},{:.4},{first:.4},{:.4},{:.4},{size}",
                    commit - 1,
                    circuit.rows.0,
                    circuit.rows.1,
                    circuit.nu,
                    ts.as_secs_f64(),
                    tp.as_secs_f64(),
                    tv.as_secs_f64()
                );
                break;
            }
        }
    }
}
