//! `CAS_N` blobs (default 1024) of `CAS_K` runs records (default 16) in one proof.
use std::{env::var, time::Instant};

use flock_prover::cas::RunsCircuit;

fn main() {
    let get = |k: &str, d: usize| var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, k, runs) = (get("CAS_N", 1024), get("CAS_K", 16), get("CAS_RUNS", 3));
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let blobs: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            let mut b = vec![0u8; 64 * k];
            for r in b.chunks_exact_mut(64) {
                for c in r[..32].chunks_exact_mut(8) {
                    c.copy_from_slice(&next().to_le_bytes());
                }
                r[32..40].copy_from_slice(&(next() % 1000 + 1).to_le_bytes());
            }
            b
        })
        .collect();
    let keys: Vec<[u8; 32]> = blobs.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
    let t = Instant::now();
    let circuit = RunsCircuit::new(n, k);
    println!("n={n} k={k} bytes={} setup {:?}", n * k * 64, t.elapsed());
    for _ in 0..runs {
        let t = Instant::now();
        let w = circuit.witness(&keys, &blobs);
        let tw = t.elapsed();
        let proof = circuit.prove(&w);
        let tp = t.elapsed();
        let t = Instant::now();
        circuit.verify(&keys, &proof).expect("verifies");
        let tv = t.elapsed();
        let mut bad = keys.clone();
        bad[0][0] ^= 1;
        assert!(circuit.verify(&bad, &proof).is_err(), "a wrong key must be rejected");
        println!(
            "witness {tw:?} witness+prove {tp:?} verify {tv:?}: {:.0} blocks/s, {:.2} MB/s",
            (n * k) as f64 / tp.as_secs_f64(),
            (n * k * 64) as f64 / tp.as_secs_f64() / 1e6
        );
    }
}
