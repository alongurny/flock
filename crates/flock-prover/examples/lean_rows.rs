//! Proves the derivation rows that cas-lean's `lean_dump` writes, and
//! reports the tables and timings. Usage: `lean_rows FILE.rows...`.

use flock_prover::lean_rows::{LeanCircuit, Rows};

fn main() {
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).expect("a rows file");
        let rows = Rows::parse(&text).expect("valid rows");
        let (t, d, f) = rows.counts();
        let c = LeanCircuit::build(&rows).expect("a consistent derivation");
        let r = c.prove_and_verify(false).expect("the proof verifies");
        let name = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        println!(
            "{name}\tterms {t}\tdecls {d}\tfacts {f}\tnu {}\trows {}\tcells {}\tbuild {:.0} ms\twitness {:.0} ms\tprove {:.0} ms\tverify {:.1} ms\tproof {} KiB",
            r.nu,
            r.rows,
            r.cells,
            r.build_ms,
            r.witness_ms,
            r.prove_ms,
            r.verify_ms,
            r.proof_bytes / 1024
        );
        for (n, rows, ports, cols) in &r.tables {
            eprintln!("  {n:>11} {rows:>8} rows {ports:>3} ports {cols:>3} columns");
        }
    }
}
