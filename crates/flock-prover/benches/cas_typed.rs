//! Real blobs of a cas type proven valid, in batches of one proof each.
//!
//! cas_typed <alias> <blobs.bin> [<max blobs>]: the file holds, for each
//! blob, its 32-byte key, a 32-bit length and its bytes (`irtest` in
//! proof-cost writes it from a store). `CAS_ROWS` bounds the compressions
//! of a batch (default 120000).
use std::{
    panic,
    time::{Duration, Instant},
};

use flock_prover::{
    cas_ir,
    cas_typed::{TypedCircuit, compressions},
};

fn main() {
    let a: Vec<String> = std::env::args().filter(|a| a != "--bench").collect();
    let (alias, file) = (&a[1], &a[2]);
    let max: usize = a.get(3).map_or(usize::MAX, |n| n.parse().unwrap());
    let rows: usize = std::env::var("CAS_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(120_000);
    let data = std::fs::read(file).unwrap();
    let (mut keys, mut blobs, mut p) = (Vec::new(), Vec::new(), 0);
    while p < data.len() && blobs.len() < max {
        let key: [u8; 32] = data[p..p + 32].try_into().unwrap();
        let len = u32::from_le_bytes(data[p + 32..p + 36].try_into().unwrap()) as usize;
        let blob = data[p + 36..p + 36 + len].to_vec();
        assert_eq!(blake3::hash(&blob).as_bytes(), &key);
        keys.push(key);
        blobs.push(blob);
        p += 36 + len;
    }
    let types = cas_ir::types();
    let t = types.iter().find(|t| t.alias == alias).expect("a type");
    let longest = blobs.iter().map(Vec::len).max().unwrap_or(0);
    println!("{alias}: {} blobs, the longest {longest} bytes", blobs.len());

    let (mut at, mut batch) = (0, 0);
    let (mut sum_blobs, mut sum_bytes, mut sum_comp, mut sum_rows) = (0, 0, 0, 0);
    let (mut setup, mut prove, mut verify) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    while at < blobs.len() {
        // As many blobs as the batch has room for, then cut to fit.
        let (mut n, mut used) = (0, 0);
        while at + n < blobs.len() && used + compressions(blobs[at + n].len() / 64) <= rows {
            used += compressions(blobs[at + n].len() / 64);
            n += 1;
        }
        if n == 0 {
            break;
        }
        let (k, b) = (&keys[at..at + n], &blobs[at..at + n]);
        let blocks: Vec<usize> = b.iter().map(|b| b.len() / 64).collect();
        let bytes: usize = b.iter().map(Vec::len).sum();
        let start = Instant::now();
        let typed: Vec<(&cas_ir::Type, usize)> = blocks.iter().map(|&k| (t, 64 * k)).collect();
        let circuit = TypedCircuit::new(&typed, true);
        let ts = start.elapsed();
        // The first proof of a process is slower: it is run and left out.
        for timed in [batch > 0, true] {
            let start = Instant::now();
            let w = circuit.witness(k, b);
            let proof = circuit.prove(&w);
            let tp = start.elapsed();
            let size = bincode::serialize(&proof.proof).unwrap().len() + bincode::serialize(&proof.commitment).unwrap().len();
            let start = Instant::now();
            circuit.verify(k, &proof).expect("verifies");
            let tv = start.elapsed();
            if !timed {
                continue;
            }
            let mut bad = k.to_vec();
            bad[0][0] ^= 1;
            assert!(circuit.verify(&bad, &proof).is_err(), "a wrong key must be rejected");
            println!(
                "batch {batch}: {n} blobs, {bytes} bytes, {} compressions, {} type rows; setup {ts:?} witness+prove {tp:?} verify {tv:?} proof {size} bytes",
                circuit.rows.0, circuit.rows.1
            );
            (sum_blobs, sum_bytes, sum_comp, sum_rows) =
                (sum_blobs + n, sum_bytes + bytes, sum_comp + circuit.rows.0, sum_rows + circuit.rows.1);
            (setup, prove, verify) = (setup + ts, prove + tp, verify + tv);
            break;
        }
        at += n;
        batch += 1;
    }
    println!(
        "{alias} total: {sum_blobs} of {} blobs in {batch} proofs, {sum_bytes} bytes, {sum_comp} compressions, {sum_rows} type rows; setup {setup:?} witness+prove {prove:?} verify {verify:?}: {:.0} compressions/s, {:.2} MB/s, {:.0} blobs/s",
        blobs.len(),
        sum_comp as f64 / prove.as_secs_f64(),
        sum_bytes as f64 / prove.as_secs_f64() / 1e6,
        sum_blobs as f64 / prove.as_secs_f64()
    );

    // A blob that is not valid, under its true hash: no proof of it may
    // verify. A few blobs suffice.
    let n = blobs.len().min(64);
    let (mut keys, mut blobs) = (keys[..n].to_vec(), blobs[..n].to_vec());
    let blocks: Vec<usize> = blobs.iter().map(|b| b.len() / 64).collect();
    let i = blobs.iter().position(|b| !b.is_empty()).expect("a blob with a block");
    let c = cas_ir::circuits(t);
    let end = (0..blobs[i].len())
        .rev()
        .find(|&p| {
            let mut b = blobs[i].clone();
            b[p] ^= 0x41;
            !cas_ir::check(t, &c, &b)
        })
        .expect("a byte that matters");
    blobs[i][end] ^= 0x41;
    keys[i] = *blake3::hash(&blobs[i]).as_bytes();
    let typed: Vec<(&cas_ir::Type, usize)> = blocks.iter().map(|&k| (t, 64 * k)).collect();
    let small = TypedCircuit::new(&typed, false);
    let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let w = small.witness(&keys, &blobs);
        let proof = small.prove(&w);
        small.verify(&keys, &proof)
    }));
    match outcome {
        Ok(Ok(())) => panic!("a proof of a blob that is not valid verified"),
        Ok(Err(e)) => println!("a damaged blob under its true hash: the proof is refused ({e:?})"),
        Err(_) => println!("a damaged blob under its true hash: the prover stops"),
    }
}
