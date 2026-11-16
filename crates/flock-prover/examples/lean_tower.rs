//! The Lean cone as ONE proof: the sides of `lean_cone` (the shared side
//! and the batches) dealt into uniform leaves of one profile, proved under
//! one shared bus and folded by the recursion tower into a single root,
//! verified with the cone's statement.
//!
//! ```text
//! LEAN_ROWS_NU=NU lean_tower census N OUT          # OUT/census.tsv
//! LEAN_ROWS_NU=NU lean_tower run N OUT SLOTS [ENV_NU LANES M]
//! LEAN_ROWS_NU=NU lean_tower probe N OUT SLOTS [ENV_NU LANES M]   # four leaves
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
        let (c, _) =
            LeanCircuit::build_side(&g, &Rows::default(), Side::Shared, SHARED_BASE, &outside)
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

/// Per table, the widest multiplicity over every side: one leaf circuit.
fn bits_path(out: &str) -> String {
    format!("{out}/mult-bits.tsv")
}

fn read_bits(out: &str) -> Vec<usize> {
    let text = fs::read_to_string(bits_path(out)).expect("run census first");
    text.trim()
        .split('\t')
        .map(|x| x.parse().unwrap())
        .collect()
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
    let totals: Vec<usize> = (0..n_tables)
        .map(|s| census.iter().map(|r| r[s]).sum())
        .collect();
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

/// The process's resident and peak memory, for the progress lines.
/// Counts live heap bytes, to tell held memory from allocator slack.
struct Counting;
static LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
// SAFETY: forwards to the system allocator and only counts sizes.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(l) }
    }
    // Forwarded, not the default alloc-then-memset: the prover relies on
    // calloc's lazy zero pages for its big zeroed buffers.
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        LIVE.fetch_sub(l.size(), std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        LIVE.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        LIVE.fetch_sub(l.size(), std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.realloc(p, l, n) }
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

/// Return freed heap pages to the OS (glibc keeps a large transient's
/// fragments otherwise).
fn trim() {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    // SAFETY: glibc's malloc_trim only returns free heap pages to the OS.
    unsafe {
        malloc_trim(0);
    }
}

fn rss() -> String {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    // SAFETY: glibc's malloc_trim only returns free heap pages to the OS.
    unsafe {
        malloc_trim(0);
    }
    let st = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |k: &str| {
        st.lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(0.0, |kb| kb as f64 / 1048576.0)
    };
    format!(
        "rss {:.1} GB, peak {:.1} GB, live {:.1} GB",
        get("VmRSS:"),
        get("VmHWM:"),
        LIVE.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1073741824.0
    )
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| a[i].parse::<usize>().unwrap();
    let (n, out) = (num(1), a[2].clone());
    let t0 = Instant::now();
    match a[0].as_str() {
        "asks" => {
            // Each batch's requests to the shared side, at this nu: the
            // shared side's counts read them, and a row's id depends on nu.
            let g: Rows = read(format!("{out}/cone.bin"));
            for k in 0..n {
                let b: Rows = read(format!("{out}/b{k}.bin"));
                let (_, asks) =
                    LeanCircuit::build_side(&g, &b, Side::Batch, batch_base(k), &HashMap::new())
                        .unwrap_or_else(|e| panic!("batch {k}: {e}"));
                let v: Vec<(Tuple, u64)> = asks.into_iter().collect();
                fs::write(format!("{out}/{k}.asks.new"), bincode::serialize(&v).unwrap()).unwrap();
                eprintln!("batch {k}: {} asks ({:.0} s)", v.len(), t0.elapsed().as_secs_f64());
            }
        }
        "cmpasks" => {
            for k in 0..n {
                let old: HashMap<Tuple, u64> =
                    read::<Vec<(Tuple, u64)>>(format!("{out}/{k}.asks")).into_iter().collect();
                let new: HashMap<Tuple, u64> =
                    read::<Vec<(Tuple, u64)>>(format!("{out}/{k}.asks.new")).into_iter().collect();
                let diff = new.iter().filter(|(t, c)| old.get(*t) != Some(*c)).count()
                    + old.keys().filter(|t| !new.contains_key(*t)).count();
                if diff > 0 {
                    eprintln!("batch {k}: {diff} differ of {}", new.len());
                }
            }
            eprintln!("compared {n}");
        }
        "rules" => {
            let g: Rows = read(format!("{out}/cone.bin"));
            let mut tot: HashMap<String, (usize, bool)> = HashMap::new();
            for k in 0..n {
                let b: Rows = read(format!("{out}/b{k}.bin"));
                for (r, (c, ok)) in b.rule_census(&g) {
                    let e = tot.entry(r).or_insert((0, ok));
                    e.0 += c;
                }
            }
            let mut v: Vec<_> = tot.into_iter().collect();
            v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            for (r, (c, ok)) in v {
                println!("{r}\t{c}\t{}", if ok { "checked" } else { "assumed" });
            }
        }
        "census" => {
            let mut text = String::new();
            let mut nu = None;
            let mut bits: Vec<usize> = Vec::new();
            for side in 0..=n {
                let c = build(side, n, &out);
                assert!(
                    nu.is_none_or(|x| x == c.nu),
                    "one nu for every side (set LEAN_ROWS_NU)"
                );
                nu = Some(c.nu);
                let b = c.mult_bits();
                bits.resize(bits.len().max(b.len()), 0);
                for (x, y) in bits.iter_mut().zip(b) {
                    *x = (*x).max(y);
                }
                let n_tables = c.chunks().iter().map(|x| x.0 + 1).max().unwrap_or(0);
                let pt = per_table(&c, n_tables);
                let counts: Vec<String> = pt.iter().map(|t| t.len().to_string()).collect();
                eprintln!(
                    "side {side}: {} chunks ({:.0} s)",
                    pt.iter().map(Vec::len).sum::<usize>(),
                    t0.elapsed().as_secs_f64()
                );
                text.push_str(&format!("{side}\t{}\n", counts.join("\t")));
            }
            fs::write(census_path(&out), format!("{}\n{text}", nu.unwrap())).unwrap();
            let bits: Vec<String> = bits.iter().map(usize::to_string).collect();
            fs::write(bits_path(&out), bits.join("\t") + "\n").unwrap();
        }
        "run" | "probe" | "sizes" => {
            let slots = num(3);
            let env = LeanEnvelope {
                nu: a.get(4).map_or(15, |x| x.parse().unwrap()),
                lanes: a.get(5).map_or(40, |x| x.parse().unwrap()),
                m: a.get(6).map_or(30, |x| x.parse().unwrap()),
            };
            let (nu, mut census) = read_census(&out);
            flock_prover::lean_rows::set_mult_bits(Some(read_bits(&out)));
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
            if a[0] == "probe" {
                // Four leaves from the middle of the cone: the envelope's fit
                // and the per-proof costs, before a full run.
                let at = std::env::var("LEAN_PROBE_AT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(index.len() / 2)
                    & !1;
                index = index[at..at + 4].to_vec();
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
            if std::env::var_os("LEAN_SPECS").is_some() {
                let (x, y) = (index[0].0, index[index.len() - 1].0);
                let (cx, cy) = (
                    build(x, n, &out).spec_texts(),
                    build(y, n, &out).spec_texts(),
                );
                for (i, (p, q)) in cx.iter().zip(&cy).enumerate() {
                    if p != q {
                        println!(
                            "table {i} differs between sides {x} and {y}:\n  {}\n  {}",
                            &p[..p.len().min(600)],
                            &q[..q.len().min(600)]
                        );
                    }
                }
                println!("{} vs {} tables", cx.len(), cy.len());
                return;
            }
            if let Ok(i) = std::env::var("LEAN_SIDE_OF") {
                let (side, l) = index[i.parse::<usize>().unwrap()];
                println!("leaf {i} is ({side}, {l}); side counts {:?}", census[side]);
                let c = build(side, n, &out);
                println!("side built ({})", rss());
                let p = piece_of(&c, l, &census[side]);
                drop(c);
                println!("piece built ({})", rss());
                drop(p);
                return;
            }
            if std::env::var_os("LEAN_BALANCE").is_some() {
                // The cone's bus natively, leaf by leaf as the tower deals
                // it: the net count of every tuple, which must be zero.
                let mut net: HashMap<[u8; 16], (i64, usize, bool)> = HashMap::new();
                let mut side_c: Option<(usize, LeanCircuit)> = None;
                for &(side, l) in &index {
                    if l == usize::MAX {
                        continue;
                    }
                    if side_c.as_ref().is_none_or(|(s, _)| *s != side) {
                        side_c = None;
                        side_c = Some((side, build(side, n, &out)));
                        eprintln!("side {side} ({:.0} s, {} tuples)", t0.elapsed().as_secs_f64(), net.len());
                    }
                    let c = &side_c.as_ref().unwrap().1;
                    let leaves = side_leaves(&profile, &census[side]);
                    let mut pt = per_table(c, n_tables);
                    pt.resize(n_tables, Vec::new());
                    let slots = profile.deal(&pt, leaves).swap_remove(l);
                    for (k, x) in slots.iter().enumerate() {
                        if let Some(ch) = *x {
                            c.for_each_tuple(profile.slots[k], ch, |s, send, t, m| {
                                if m == 0 {
                                    return;
                                }
                                let mut h = blake3::Hasher::new();
                                for v in t {
                                    h.update(&v.lo.to_le_bytes());
                                    h.update(&v.hi.to_le_bytes());
                                }
                                let key: [u8; 16] = h.finalize().as_bytes()[..16].try_into().unwrap();
                                let e = net.entry(key).or_insert((0, s, send));
                                e.0 += if send { m as i64 } else { -(m as i64) };
                            });
                        }
                    }
                }
                let bad: Vec<_> = net.values().filter(|e| e.0 != 0).collect();
                println!("{} tuples, {} unbalanced", net.len(), bad.len());
                let mut by: HashMap<(usize, bool), (usize, i64)> = HashMap::new();
                for e in &bad {
                    let b = by.entry((e.1, e.2)).or_default();
                    b.0 += 1;
                    b.1 += e.0;
                }
                let mut by: Vec<_> = by.into_iter().collect();
                by.sort();
                for ((s, send), (k, sum)) in by {
                    println!("  first seen in table {s} ({}): {k} tuples, net {sum}", if send { "send" } else { "recv" });
                }
                return;
            }
            if a[0] == "sizes" {
                let (side, l) = index[index.len() / 2];
                let c = build(side, n, &out);
                println!("side built ({})", rss());
                let p = piece_of(&c, l, &census[side]);
                drop(c);
                println!("piece built ({})", rss());
                let sh = p.shape();
                let cl = sh.clone();
                println!("shape cloned ({})", rss());
                drop(cl);
                println!("  values: {:.1} MB", p.n_values() as f64 * 16.0 / 1e6);
                for (k, b) in sh.heap_report() {
                    println!("  {k}: {:.1} MB ({b})", b as f64 / 1e6);
                }
                let wires: usize = sh.circuit.wires().iter().map(Vec::len).sum();
                println!(
                    "cells mu {}, wire classes {}, wired cells {}, types {}",
                    sh.circuit.cells().mu(),
                    sh.circuit.wires().len(),
                    wires,
                    sh.registry.types().len()
                );
                let mut ent = 0usize;
                let mut rows = 0usize;
                for ty in sh.registry.types() {
                    if let Some(e) = ty.element_type() {
                        for m in [e.a_0(), e.b_0()] {
                            rows += m.rows.len();
                            ent += m.rows.iter().map(Vec::len).sum::<usize>();
                        }
                    }
                }
                println!("element matrices: {rows} rows, {ent} entries");
                drop(p);
                println!("piece dropped ({})", rss());
                return;
            }

            // ---- FIRST PASS: every leaf's word, kept on disk: a run that
            // fails later restarts at the tower (the tower checks every word
            // against its leaf, so a stale file fails loudly) ----
            let words_file = format!("{out}/words-{}-{slots}-{}.bin", a[0], index.len());
            let mut words: Vec<[u8; 32]> = fs::read(&words_file)
                .map(|b| b.chunks_exact(32).map(|c| c.try_into().unwrap()).collect())
                .unwrap_or_default();
            let mut side_c: Option<(usize, LeanCircuit)> = None;
            if words.len() != index.len() {
                words.clear();
                for &(side, l) in &index {
                    if side_c.as_ref().is_none_or(|(s, _)| *s != side) {
                        side_c = None;
                        side_c = Some((side, build(side, n, &out)));
                    }
                    let c = &side_c.as_ref().unwrap().1;
                    assert_eq!(c.nu, nu, "the census's nu");
                    let p = piece_of(c, l, &census[side]);
                    if std::env::var_os("LEAN_DIGESTS").is_some() {
                        println!(
                            "leaf ({side}, {l}): circuit blake3:{:02x?}",
                            &p.digest()[..6]
                        );
                    }
                    words.push(leaf_word(&p));
                }
                fs::write(&words_file, words.concat()).unwrap();
            }
            println!(
                "first pass: {} leaves committed ({:.0} s, {})",
                words.len(),
                t0.elapsed().as_secs_f64(),
                rss()
            );

            // ---- SECOND PASS: the tower, from the last leaf back ----
            drop(side_c);
            let mut pending: Option<(usize, flock_prover::lean_rows::Piece)> = None;
            let t1 = Instant::now();
            let r = prove_lean_tower_streaming(
                TowerConfig::Chain128,
                &words,
                |i| {
                    // A side's circuit is large (7.6 GB at the peak to build
                    // the shared side): build it with nothing proven held
                    // (the pool cleared), take leaf i and, when of the same
                    // side, leaf i - 1 (asked next: its piece is 0.8 GB),
                    // drop it and hand its pages back before a leaf proves.
                    if let Some((j, p)) = pending.take() {
                        if j == i {
                            return p;
                        }
                    }
                    let (side, l) = index[i];
                    flock_core::scratch::clear();
                    let c = build(side, n, &out);
                    let p = piece_of(&c, l, &census[side]);
                    if i % 2 == 1 && index[i - 1].0 == side {
                        pending = Some((i - 1, piece_of(&c, index[i - 1].1, &census[side])));
                    }
                    drop(c);
                    trim();
                    p
                },
                env,
                |line| println!("{line} ({:.0} s, {})", t1.elapsed().as_secs_f64(), rss()),
            );
            println!("{r:?}");
            if a[0] == "run" {
                assert!(r.balanced, "the shared bus balances over the whole cone");
            }
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
