//! The Lean cone as ONE proof: the sides of `lean_cone` (the shared side
//! and the batches) dealt into uniform leaves of one profile, proved under
//! one shared bus and folded by the recursion tower into a single root,
//! verified with the cone's statement.
//!
//! ```text
//! LEAN_ROWS_NU=NU lean_tower census N OUT          # OUT/census.tsv
//! LEAN_ROWS_NU=NU lean_tower run N OUT SLOTS [ENV_NU LANES M]
//! ```
//!
//! `N` batches (the shared side is side 0, batch k side k + 1), `OUT` the
//! `lean_cone` directory (cone.bin, b<k>.bin, <k>.asks). `run` picks a
//! profile of about `SLOTS` slots from the census, commits every leaf
//! (first pass: the leaves' words), then proves the tower holding one side
//! and one leaf pair at a time (second pass, from the last leaf back).

use std::{collections::HashMap, fs, path::Path, time::Instant};

use flock_prover::{
    lean_rows::{LeafSlot, LeanCircuit, Profile, Rows, SHARED_BASE, Side, Tuple, leaf},
    tower::{LeanEnvelope, TowerConfig, leaf_word, prove_lean_tower_streaming},
};

fn read<T: serde::de::DeserializeOwned>(p: impl AsRef<Path>) -> T {
    let bytes = fs::read(p.as_ref()).unwrap_or_else(|e| panic!("{}: {e}", p.as_ref().display()));
    bincode::deserialize(&bytes).unwrap()
}

fn batch_base(k: usize) -> usize {
    (k + 1) << 20
}

/// Side 0 is the shared side, side k + 1 batch k (as `lean_cone` builds them).
fn build(side: usize, n: usize, out: &str) -> LeanCircuit {
    let g: Rows = read(format!("{out}/cone.bin"));
    if side == 0 {
        let mut outside: HashMap<Tuple, u64> = HashMap::new();
        for k in 0..n {
            for (t, c) in read::<Vec<(Tuple, u64)>>(format!("{out}/{k}.asks")) {
                *outside.entry(t).or_default() += c;
            }
        }
        let (c, _) = LeanCircuit::build_side(&g, &Rows::default(), Side::Shared, SHARED_BASE, &outside)
            .unwrap_or_else(|e| panic!("shared side: {e}"));
        c
    } else {
        let k = side - 1;
        let b: Rows = read(format!("{out}/b{k}.bin"));
        let (c, _) = LeanCircuit::build_side(&g, &b, Side::Batch, batch_base(k), &HashMap::new())
            .unwrap_or_else(|e| panic!("batch {k}: {e}"));
        c
    }
}

/// Per table, the side's chunk numbers.
fn per_table(c: &LeanCircuit, n_tables: usize) -> Vec<Vec<usize>> {
    let mut v = vec![Vec::new(); n_tables];
    for (s, ch, _) in c.chunks() {
        v[s].push(ch);
    }
    v
}

fn census_path(out: &str) -> String {
    format!("{out}/census.tsv")
}

fn read_census(out: &str) -> (usize, Vec<Vec<usize>>) {
    let text = fs::read_to_string(census_path(out)).expect("run census first");
    let mut lines = text.lines();
    let nu: usize = lines.next().unwrap().parse().unwrap();
    let rows = lines
        .map(|l| l.split('\t').skip(1).map(|x| x.parse().unwrap()).collect())
        .collect();
    (nu, rows)
}

/// About `slots` slots, table s taking its share of every side's chunks.
fn profile(nu: usize, census: &[Vec<usize>], slots: usize) -> Profile {
    let n_tables = census[0].len();
    let totals: Vec<usize> = (0..n_tables).map(|s| census.iter().map(|r| r[s]).sum()).collect();
    let all: usize = totals.iter().sum();
    let mut v = Vec::new();
    for (s, &t) in totals.iter().enumerate() {
        if t > 0 {
            let k = ((slots * t) as f64 / all as f64).round().max(1.0) as usize;
            v.extend(std::iter::repeat_n(s, k));
        }
    }
    Profile { nu, slots: v }
}

/// A side's leaf count under `profile`.
fn side_leaves(profile: &Profile, counts: &[usize]) -> usize {
    counts
        .iter()
        .enumerate()
        .map(|(s, &n)| {
            let k = profile.slots.iter().filter(|&&x| x == s).count();
            if n == 0 { 0 } else { n.div_ceil(k) }
        })
        .max()
        .unwrap_or(0)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| a[i].parse::<usize>().unwrap();
    let (n, out) = (num(1), a[2].clone());
    let t0 = Instant::now();
    match a[0].as_str() {
        "census" => {
            let mut text = String::new();
            let mut nu = None;
            for side in 0..=n {
                let c = build(side, n, &out);
                assert!(nu.is_none_or(|x| x == c.nu), "one nu for every side (set LEAN_ROWS_NU)");
                nu = Some(c.nu);
                let n_tables = c.chunks().iter().map(|x| x.0 + 1).max().unwrap_or(0);
                let pt = per_table(&c, n_tables);
                let counts: Vec<String> = pt.iter().map(|t| t.len().to_string()).collect();
                eprintln!("side {side}: {} chunks ({:.0} s)", pt.iter().map(Vec::len).sum::<usize>(), t0.elapsed().as_secs_f64());
                text.push_str(&format!("{side}\t{}\n", counts.join("\t")));
            }
            fs::write(census_path(&out), format!("{}\n{text}", nu.unwrap())).unwrap();
        }
        "run" => {
            let slots = num(3);
            let env = LeanEnvelope {
                nu: a.get(4).map_or(15, |x| x.parse().unwrap()),
                lanes: a.get(5).map_or(40, |x| x.parse().unwrap()),
                m: a.get(6).map_or(30, |x| x.parse().unwrap()),
            };
            let (nu, mut census) = read_census(&out);
            let n_tables = census.iter().map(Vec::len).max().unwrap();
            for r in &mut census {
                r.resize(n_tables, 0);
            }
            let profile = profile(nu, &census, slots);
            // Leaves in order, side by side: (side, leaf in side); an empty
            // leaf pads an odd count.
            let mut index: Vec<(usize, usize)> = Vec::new();
            for (side, counts) in census.iter().enumerate() {
                for l in 0..side_leaves(&profile, counts) {
                    index.push((side, l));
                }
            }
            if index.len() % 2 == 1 {
                index.push((n, usize::MAX));
            }
            println!(
                "profile: nu {nu}, {} slots; {} leaves over {} sides",
                profile.slots.len(),
                index.len(),
                n + 1
            );
            let piece_of = |c: &LeanCircuit, l: usize, counts: &[usize]| {
                let leaves = side_leaves(&profile, counts);
                let mut pt = per_table(c, n_tables);
                pt.resize(n_tables, Vec::new());
                let slots: Vec<Option<usize>> = if l == usize::MAX {
                    vec![None; profile.slots.len()]
                } else {
                    profile.deal(&pt, leaves).swap_remove(l)
                };
                let s: Vec<LeafSlot<'_>> = slots.iter().map(|x| x.map(|ch| (c, ch))).collect();
                leaf(c, &profile, &s).unwrap()
            };

            // ---- FIRST PASS: every leaf's word ----
            let mut words = Vec::with_capacity(index.len());
            let mut side_c: Option<(usize, LeanCircuit)> = None;
            for &(side, l) in &index {
                if side_c.as_ref().is_none_or(|(s, _)| *s != side) {
                    side_c = None;
                    side_c = Some((side, build(side, n, &out)));
                }
                let c = &side_c.as_ref().unwrap().1;
                assert_eq!(c.nu, nu, "the census's nu");
                words.push(leaf_word(&piece_of(c, l, &census[side])));
            }
            println!("first pass: {} leaves committed ({:.0} s)", words.len(), t0.elapsed().as_secs_f64());

            // ---- SECOND PASS: the tower, from the last leaf back ----
            side_c = None;
            let c0 = build(index[0].0, n, &out);
            let p0 = piece_of(&c0, index[0].1, &census[index[0].0]);
            drop(c0);
            let t1 = Instant::now();
            let r = prove_lean_tower_streaming(
                TowerConfig::Chain128,
                &words,
                p0,
                |i| {
                    let (side, l) = index[i];
                    if side_c.as_ref().is_none_or(|(s, _)| *s != side) {
                        side_c = None;
                        side_c = Some((side, build(side, n, &out)));
                    }
                    piece_of(&side_c.as_ref().unwrap().1, l, &census[side])
                },
                env,
                |line| println!("{line} ({:.0} s)", t1.elapsed().as_secs_f64()),
            );
            println!("{r:?}");
            println!(
                "ONE PROOF for the cone: {} leaves; root proof {} KiB; verify {:.1} ms (root proof) + {:.1} ms (root-side checks); wall {:.0} s",
                r.leaves,
                r.root_proof_bytes / 1024,
                r.root_verify_ms,
                r.discharge_ms,
                t0.elapsed().as_secs_f64()
            );
        }
        m => panic!("unknown step {m}"),
    }
}
