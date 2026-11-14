//! Proves the derivation rows that cas-lean's `lean_dump` writes, and
//! reports the tables and timings. Usage: `lean_rows [--merge] FILE.rows...`;
//! with `--merge`, all files in one proof.

use flock_prover::lean_rows::{LeanCircuit, Rows};

fn run(name: &str, rows: &Rows) -> (f64, f64, usize) {
    let (t, d, f) = rows.counts();
    let c = LeanCircuit::build(rows).expect("a consistent derivation");
    let r = c.prove_and_verify(false).expect("the proof verifies");
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
    (
        r.prove_ms + r.build_ms + r.witness_ms,
        r.verify_ms,
        r.proof_bytes,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--batch FACTS DIR`: every .rows file of DIR, in proofs of about FACTS
    // facts each; prints each proof and the totals.
    if args.first().is_some_and(|a| a == "--batch") {
        let cap: usize = args[1].parse().unwrap();
        let mut files: Vec<_> = std::fs::read_dir(&args[2])
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        let (mut total_prove, mut total_verify, mut total_bytes, mut proofs, mut facts_all) =
            (0.0, 0.0, 0, 0, 0);
        let mut batch: Vec<Rows> = Vec::new();
        let mut facts = 0;
        let t = std::time::Instant::now();
        let mut flush = |batch: &mut Vec<Rows>, facts: &mut usize| {
            if batch.is_empty() {
                return;
            }
            let n = batch.len();
            let rows = Rows::merge(std::mem::take(batch));
            let (p, v, b) = run(&format!("batch of {n}"), &rows);
            total_prove += p;
            total_verify += v;
            total_bytes += b;
            proofs += 1;
            facts_all += *facts;
            *facts = 0;
        };
        for f in &files {
            let rows = Rows::parse(&std::fs::read_to_string(f).unwrap()).expect("valid rows");
            let n = rows.counts().2;
            if facts + n > cap && !batch.is_empty() {
                flush(&mut batch, &mut facts);
            }
            facts += n;
            batch.push(rows);
        }
        flush(&mut batch, &mut facts);
        println!(
            "total\t{} declarations\t{facts_all} facts\t{proofs} proofs\tprove {:.1} s (build and witness included)\tverify {:.0} ms\tproofs {} KiB\twall {:.1} s",
            files.len(),
            total_prove / 1e3,
            total_verify,
            total_bytes / 1024,
            t.elapsed().as_secs_f64()
        );
        return;
    }
    let merge = args.first().is_some_and(|a| a == "--merge");
    let files: Vec<&String> = args.iter().filter(|a| *a != "--merge").collect();
    let parse = |path: &String| {
        Rows::parse(&std::fs::read_to_string(path).expect("a rows file")).expect("valid rows")
    };
    if merge {
        let rows = Rows::merge(files.iter().map(|p| parse(p)).collect());
        run(&format!("{} merged", files.len()), &rows);
    } else {
        for path in files {
            let name = std::path::Path::new(path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            run(&name, &parse(path));
        }
    }
}
