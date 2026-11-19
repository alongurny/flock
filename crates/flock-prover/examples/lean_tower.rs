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

use flock_field::F128;

use flock_prover::{
    lean_rows::{
        Arith, LeafSlot, LeanCircuit, Profile, Rows, SHARED_BASE, Side, SlotRows, Tuple,
        arith_entry, leaf, leaf_arith, leaf_prims, leaf_statement, lit_hash,
    },
    tower::{
        LeanEnvelope, LeanStatement, SegmentClaim, TowerConfig, leaf_word, lean_segment_root,
        lean_segments_root, prove_lean_segment, prove_lean_tower_streaming, segments_balanced,
    },
};

fn read<T: serde::de::DeserializeOwned>(p: impl AsRef<Path>) -> T {
    let bytes = fs::read(p.as_ref()).unwrap_or_else(|e| panic!("{}: {e}", p.as_ref().display()));
    bincode::deserialize(&bytes).unwrap()
}

type P2 = (u64, u64);
/// A statement on disk: chunk words, checked keys, the constants read and
/// the steps of arithmetic.
type StmtS = (
    Vec<u64>,
    Vec<[P2; 2]>,
    Vec<(u64, [P2; 2])>,
    Vec<(u64, [P2; 2], u64, u64, [P2; 2])>,
);

fn stmt_ser(s: &LeanStatement) -> StmtS {
    let p = |k: &[F128; 2]| [(k[0].lo, k[0].hi), (k[1].lo, k[1].hi)];
    (
        s.chunk_words.clone(),
        s.keys.iter().map(p).collect(),
        s.prims.iter().map(|(n, k)| (*n, p(k))).collect(),
        s.arith
            .iter()
            .map(|a| (a.op, p(&a.key), a.x, a.y, p(&a.out)))
            .collect(),
    )
}

fn stmt_de((chunk_words, keys, prims, arith): StmtS) -> LeanStatement {
    let f = |k: [P2; 2]| [F128::new(k[0].0, k[0].1), F128::new(k[1].0, k[1].1)];
    LeanStatement {
        chunk_words,
        keys: keys.into_iter().map(f).collect(),
        prims: prims.into_iter().map(|(n, k)| (n, f(k))).collect(),
        arith: arith
            .into_iter()
            .map(|(op, key, x, y, out)| Arith {
                op,
                key: f(key),
                x,
                y,
                out: f(out),
            })
            .collect(),
    }
}

/// `lean_dump`'s LEAN_DUMP_NAT file: the kernel's constants by key
/// (`P number key0 key1`) and the literals' values by hash (`N value`).
/// Empty when there is no file: a cone without arithmetic needs none.
fn nat_file(path: &str) -> (HashMap<[u64; 2], u64>, HashMap<u64, u64>) {
    let (mut numbers, mut lits) = (HashMap::new(), HashMap::new());
    for line in fs::read_to_string(path).unwrap_or_default().lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let n = |i: usize| {
            w[i].parse::<u64>()
                .unwrap_or_else(|_| panic!("{path}: {line}"))
        };
        match w[0] {
            "P" => drop(numbers.insert([n(2), n(3)], n(1))),
            "N" => drop(lits.insert(lit_hash(n(1)), n(1))),
            _ => panic!("{path}: {line}"),
        }
    }
    (numbers, lits)
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

/// Where a shared chunk's rows are kept (`lean_tower shared` writes them).
fn chunk_file(out: &str, s: usize, ch: usize) -> String {
    format!("{out}/shared/{s}-{ch}.bin")
}

/// A chunk's number and rows: three u64 (chunk, rows, columns), then the
/// cells as two u64 each, row by row.
fn write_chunk(path: &str, chunk: u64, rows: &[Vec<F128>]) {
    let cols = rows.first().map_or(0, Vec::len);
    let mut b: Vec<u8> = Vec::with_capacity(24 + rows.len() * cols * 16);
    for v in [chunk, rows.len() as u64, cols as u64] {
        b.extend(v.to_le_bytes());
    }
    for r in rows {
        for x in r {
            b.extend(x.lo.to_le_bytes());
            b.extend(x.hi.to_le_bytes());
        }
    }
    fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn read_chunk(path: &str) -> (u64, Vec<Vec<F128>>) {
    let b =
        fs::read(path).unwrap_or_else(|e| panic!("{path}: {e} (run `lean_tower shared` first)"));
    let u = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    let (chunk, n, cols) = (u(0), u(1) as usize, u(2) as usize);
    let rows = (0..n)
        .map(|r| {
            (0..cols)
                .map(|c| F128::new(u(3 + 2 * (r * cols + c)), u(4 + 2 * (r * cols + c))))
                .collect()
        })
        .collect();
    (chunk, rows)
}

/// What the leaves are made from: the shared side's chunks from disk, and
/// the batches' chunks, copied out of a batch built when a leaf needs it
/// (with those of the leaves around it, so a batch is built once per
/// stretch of leaves, and never held while a leaf proves).
struct Sources {
    out: String,
    g: Rows,
    proto: LeanCircuit,
    chunks: HashMap<(usize, usize, usize), (u64, Vec<Vec<F128>>)>,
    around: usize,
}

type Plan = [Vec<Option<(usize, usize)>>];

impl Sources {
    fn new(out: &str) -> Sources {
        let g: Rows = read(format!("{out}/cone.bin"));
        let (proto, _) = LeanCircuit::build_side(
            &g,
            &Rows::default(),
            Side::Batch,
            batch_base(0),
            &HashMap::new(),
        )
        .unwrap_or_else(|e| panic!("an empty batch: {e}"));
        let around = std::env::var("LEAN_AROUND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        Sources {
            out: out.into(),
            g,
            proto,
            chunks: HashMap::new(),
            around,
        }
    }

    fn keys(
        profile: &Profile,
        slots: &[Option<(usize, usize)>],
    ) -> impl Iterator<Item = (usize, usize, usize)> {
        slots
            .iter()
            .zip(&profile.slots)
            .filter_map(|(x, &s)| x.map(|(side, ch)| (side, s, ch)))
    }

    /// Makes leaf i's chunks present; keeps only those of the leaves within
    /// `around` of it.
    fn load(&mut self, profiles: &[Profile], plan: &Plan, i: usize) {
        let profile = &profiles[i];
        let (lo, hi) = (
            i.saturating_sub(self.around),
            (i + self.around + 1).min(plan.len()),
        );
        let near: std::collections::HashSet<(usize, usize, usize)> = (lo..hi)
            .flat_map(|j| Self::keys(&profiles[j], &plan[j]))
            .filter(|k| k.0 > 0)
            .chain(Self::keys(profile, &plan[i]).filter(|k| k.0 == 0))
            .collect();
        self.chunks.retain(|k, _| near.contains(k));
        let mut sides: Vec<usize> = Self::keys(profile, &plan[i])
            .filter(|k| !self.chunks.contains_key(k))
            .map(|k| k.0)
            .collect();
        sides.sort_unstable();
        sides.dedup();
        for side in sides {
            if side == 0 {
                for k in Self::keys(profile, &plan[i]).filter(|k| k.0 == 0) {
                    let c = read_chunk(&chunk_file(&self.out, k.1, k.2));
                    self.chunks.insert(k, c);
                }
                continue;
            }
            let k = side - 1;
            let b: Rows = read(format!("{}/b{k}.bin", self.out));
            let (c, _) =
                LeanCircuit::build_side(&self.g, &b, Side::Batch, batch_base(k), &HashMap::new())
                    .unwrap_or_else(|e| panic!("batch {k}: {e}"));
            assert_eq!(c.nu, profile.nu, "the census's nu");
            for key in near.iter().filter(|x| x.0 == side) {
                if !self.chunks.contains_key(key) {
                    let x = c.slot(key.1, key.2);
                    self.chunks.insert(*key, (x.chunk, x.rows.to_vec()));
                }
            }
        }
    }

    /// Drops the shared chunks (read again from disk when asked).
    fn rest(&mut self) {
        self.chunks.retain(|k, _| k.0 > 0);
    }

    fn slots<'a>(
        &'a self,
        profile: &Profile,
        slots: &[Option<(usize, usize)>],
    ) -> Vec<LeafSlot<'a>> {
        let nu = profile.nu;
        slots
            .iter()
            .zip(&profile.slots)
            .map(|(x, &s)| {
                x.map(|(side, ch)| {
                    let (chunk, rows) = &self.chunks[&(side, s, ch)];
                    SlotRows {
                        rows,
                        chunk: *chunk,
                        nu,
                    }
                })
            })
            .collect()
    }

    fn piece(
        &self,
        profile: &Profile,
        slots: &[Option<(usize, usize)>],
        i: usize,
    ) -> flock_prover::lean_rows::Piece {
        leaf(&self.proto, profile, &self.slots(profile, slots), i).unwrap()
    }

    fn statement(
        &self,
        profile: &Profile,
        slots: &[Option<(usize, usize)>],
        i: usize,
    ) -> (Vec<u64>, Vec<[F128; 2]>) {
        leaf_statement(&self.proto, profile, &self.slots(profile, slots), i)
    }

    fn prims(&self, profile: &Profile, slots: &[Option<(usize, usize)>]) -> Vec<(u64, [F128; 2])> {
        leaf_prims(&self.proto, profile, &self.slots(profile, slots))
    }

    fn arith(&self, profile: &Profile, slots: &[Option<(usize, usize)>]) -> Vec<[F128; 7]> {
        leaf_arith(&self.proto, profile, &self.slots(profile, slots))
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
                fs::write(
                    format!("{out}/{k}.asks.new"),
                    bincode::serialize(&v).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "batch {k}: {} asks ({:.0} s)",
                    v.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        "cmpasks" => {
            for k in 0..n {
                let old: HashMap<Tuple, u64> = read::<Vec<(Tuple, u64)>>(format!("{out}/{k}.asks"))
                    .into_iter()
                    .collect();
                let new: HashMap<Tuple, u64> =
                    read::<Vec<(Tuple, u64)>>(format!("{out}/{k}.asks.new"))
                        .into_iter()
                        .collect();
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
            v.sort_by(|a, b| b.1.0.cmp(&a.1.0));
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
        "shared" => {
            // The shared side's chunks, kept on disk: every leaf holds a
            // few, and building the side takes minutes and gigabytes.
            flock_prover::lean_rows::set_mult_bits(Some(read_bits(&out)));
            let c = build(0, n, &out);
            fs::create_dir_all(format!("{out}/shared")).unwrap();
            let mut k = 0;
            for (s, ch, _) in c.chunks() {
                let x = c.slot(s, ch);
                write_chunk(&chunk_file(&out, s, ch), x.chunk, x.rows);
                k += 1;
            }
            eprintln!(
                "{k} chunks written ({:.0} s, {})",
                t0.elapsed().as_secs_f64(),
                rss()
            );
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
            // Leaves of the rare rules' table alone (LEAN_RARE_SLOTS=k slots
            // each), in a segment of their own: the leaf's verifier grows
            // with the leaf's ports, and that table has as many as all the
            // others together.
            let rare_k: usize = std::env::var("LEAN_RARE_SLOTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let rare = flock_prover::lean_rows::rare_table();
            // The table of the spine rules follows it and shares its leaves
            // (LEAN_SPINE_SLOTS slots each, 2 unless said).
            let spine_k: usize = std::env::var("LEAN_SPINE_SLOTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2);
            // And the table of the rules that type a projection (LEAN_TEL_SLOTS, 1 unless said).
            let tel_k: usize = std::env::var("LEAN_TEL_SLOTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            let late: Vec<(usize, usize)> = vec![
                (rare, rare_k),
                (rare + 1, spine_k),
                (rare + 2, tel_k),
                (flock_prover::lean_rows::rule_table(), 1),
            ];
            let late_census: Vec<Vec<usize>> = late
                .iter()
                .map(|&(t, _)| census.iter().map(|r| r[t]).collect())
                .collect();
            if rare_k > 0 {
                for r in &mut census {
                    for &(t, _) in &late {
                        r[t] = 0;
                    }
                }
            }
            let profile = profile(nu, &census, slots);
            // Every table's chunks over the whole cone, side by side: leaf
            // l takes the next k_s of table s, so a leaf holds terms of
            // the shared side beside the judgements of a few batches.
            // The batches come in order and a leaf takes chunks from at
            // most `window` of them at once (a batch is built to read its
            // chunks), so a table whose chunks run out leaves its slots
            // empty until the oldest batch is used up. The shared side's
            // chunks fill their own slots at an even rate.
            let window = std::env::var("LEAN_WINDOW")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4usize);
            let k_of: Vec<usize> = (0..n_tables)
                .map(|s| profile.slots.iter().filter(|&&x| x == s).count())
                .collect();
            let mut queues: Vec<std::collections::VecDeque<(usize, usize)>> =
                vec![std::collections::VecDeque::new(); n_tables];
            for (s, &c) in census[0].iter().enumerate() {
                queues[s].extend((0..c).map(|ch| (0, ch)));
            }
            let mut next = 1;
            let mut plan: Vec<Vec<Option<(usize, usize)>>> = Vec::new();
            loop {
                // Bring batches in while some table is short and the window allows.
                loop {
                    let oldest = queues
                        .iter()
                        .flatten()
                        .map(|x| x.0)
                        .filter(|&x| x > 0)
                        .min()
                        .unwrap_or(next);
                    let short = (0..n_tables)
                        .any(|s| census[1..].iter().any(|c| c[s] > 0) && queues[s].len() < k_of[s]);
                    if next > n || !short || next - oldest >= window {
                        break;
                    }
                    for (s, &c) in census[next].iter().enumerate() {
                        queues[s].extend((0..c).map(|ch| (next, ch)));
                    }
                    next += 1;
                }
                if queues.iter().all(|q| q.is_empty()) {
                    break;
                }
                let mut seen = vec![0usize; n_tables];
                let leaf: Vec<Option<(usize, usize)>> = profile
                    .slots
                    .iter()
                    .map(|&s| {
                        seen[s] += 1;
                        queues[s].pop_front()
                    })
                    .collect();
                plan.push(leaf);
            }
            if plan.len() % 2 == 1 {
                plan.push(vec![None; profile.slots.len()]);
            }
            let n_common = plan.len();
            let mut profiles: Vec<Profile> = vec![profile.clone(); n_common];
            if rare_k > 0 {
                let rare_profile = Profile {
                    nu,
                    slots: late.iter().flat_map(|&(t, k)| vec![t; k]).collect(),
                };
                let all: Vec<Vec<(usize, usize)>> = late_census
                    .iter()
                    .map(|c| {
                        c.iter()
                            .enumerate()
                            .flat_map(|(side, &c)| (0..c).map(move |ch| (side, ch)))
                            .collect()
                    })
                    .collect();
                let n_late = late
                    .iter()
                    .zip(&all)
                    .map(|(&(_, k), a)| a.len().div_ceil(k.max(1)))
                    .max()
                    .unwrap_or(0);
                let width: usize = late.iter().map(|x| x.1).sum();
                for l in 0..n_late {
                    let mut leaf: Vec<Option<(usize, usize)>> = Vec::with_capacity(width);
                    for (&(_, k), a) in late.iter().zip(&all) {
                        leaf.extend((l * k..(l + 1) * k).map(|i| a.get(i).copied()));
                    }
                    plan.push(leaf);
                }
                while (plan.len() - n_common) % 2 == 1 || plan.len() - n_common < 4 {
                    plan.push(vec![None; width]);
                }
                profiles.resize(plan.len(), rare_profile);
            }
            let n_leaves = plan.len();
            // The cone's index of each leaf (padding words depend on it).
            let mut index: Vec<usize> = (0..n_leaves).collect();
            if a[0] == "probe" {
                // Four leaves from the middle of the cone: the envelope's fit
                // and the per-proof costs, before a full run.
                let at = std::env::var("LEAN_PROBE_AT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(index.len() / 2)
                    & !1;
                index = index[at..at + 4].to_vec();
                plan = plan[at..at + 4].to_vec();
                profiles = profiles[at..at + 4].to_vec();
            }
            let full: usize = plan.iter().flatten().flatten().count();
            println!(
                "profile: nu {nu}, {} slots; {} leaves over {} sides; {} of {} slots hold a chunk",
                profile.slots.len(),
                index.len(),
                n + 1,
                full,
                profiles.iter().map(|p| p.slots.len()).sum::<usize>()
            );
            if std::env::var_os("LEAN_PLAN").is_some() {
                for (i, p) in plan.iter().enumerate() {
                    let mut sides: Vec<usize> = p.iter().flatten().map(|x| x.0).collect();
                    sides.sort_unstable();
                    sides.dedup();
                    println!(
                        "leaf {i}: {} chunks from sides {sides:?}",
                        p.iter().flatten().count()
                    );
                }
                return;
            }
            if std::env::var_os("LEAN_BALANCE").is_some() {
                // The cone's bus natively, side by side: the net count of
                // every tuple, which must be zero.
                let mut net: HashMap<[u8; 16], (i64, usize, bool)> = HashMap::new();
                for side in 0..=n {
                    let c = build(side, n, &out);
                    eprintln!(
                        "side {side} ({:.0} s, {} tuples)",
                        t0.elapsed().as_secs_f64(),
                        net.len()
                    );
                    for (s0, ch, _) in c.chunks() {
                        c.for_each_tuple(s0, ch, |s, send, t, m| {
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
                    println!(
                        "  first seen in table {s} ({}): {k} tuples, net {sum}",
                        if send { "send" } else { "recv" }
                    );
                }
                return;
            }
            let mut src = Sources::new(&out);
            if a[0] == "sizes" {
                let at = index.len() / 2;
                src.load(&profiles, &plan, at);
                println!("sources loaded ({})", rss());
                let p = src.piece(&profiles[at], &plan[at], index[at]);
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
            let stmt_file = format!("{out}/statement-{}-{slots}-{}.bin", a[0], index.len());
            let mut statement: LeanStatement = fs::read(&stmt_file)
                .ok()
                .map(|b| {
                    stmt_de(bincode::deserialize(&b).expect("a statement file of this version"))
                })
                .unwrap_or_default();
            if words.len() != index.len() || statement.chunk_words.is_empty() {
                words.clear();
                statement = LeanStatement::default();
                // What names the arithmetic's steps: the kernel's constants
                // and the literals' values (`lean_dump` with LEAN_DUMP_NAT).
                let (numbers, lits) = nat_file(&format!("{out}/nat.rows"));
                // A first pass that was cut short goes on from its last
                // checkpoint: the words and the statement of the leaves done.
                type Part = (Vec<[u8; 32]>, StmtS);
                let part_file = format!("{words_file}.part");
                if let Some((w, s)) = fs::read(&part_file)
                    .ok()
                    .and_then(|b| bincode::deserialize::<Part>(&b).ok())
                {
                    eprintln!("first pass: {} leaves from the checkpoint", w.len());
                    words = w;
                    statement = stmt_de(s);
                }
                let done = words.len();
                for (i, p) in plan.iter().enumerate().skip(done) {
                    if i > done && i % 8 == 0 {
                        let part: Part = (words.clone(), stmt_ser(&statement));
                        fs::write(
                            format!("{part_file}.new"),
                            bincode::serialize(&part).unwrap(),
                        )
                        .unwrap();
                        fs::rename(format!("{part_file}.new"), &part_file).unwrap();
                    }
                    src.load(&profiles, &plan, i);
                    let piece = src.piece(&profiles[i], p, index[i]);
                    let (w, k) = src.statement(&profiles[i], p, index[i]);
                    statement.chunk_words.extend(w);
                    statement.keys.extend(k);
                    statement.prims.extend(src.prims(&profiles[i], p));
                    for raw in src.arith(&profiles[i], p) {
                        statement.arith.push(
                            arith_entry(&raw, &numbers, &lits)
                                .unwrap_or_else(|e| panic!("leaf {i}: {e}")),
                        );
                    }
                    if std::env::var_os("LEAN_DIGESTS").is_some() {
                        println!("leaf {i}: circuit blake3:{:02x?}", &piece.digest()[..6]);
                    }
                    words.push(leaf_word(&piece));
                    if i % 8 == 0 {
                        eprintln!(
                            "leaf {i} committed ({:.0} s, {})",
                            t0.elapsed().as_secs_f64(),
                            rss()
                        );
                    }
                }
                fs::write(&words_file, words.concat()).unwrap();
                fs::write(
                    &stmt_file,
                    bincode::serialize(&stmt_ser(&statement)).unwrap(),
                )
                .unwrap();
                let _ = fs::remove_file(&part_file);
            }
            println!(
                "statement: {} chunk words, {} checked keys, {} steps of arithmetic",
                statement.chunk_words.len(),
                statement.keys.len(),
                statement.arith.len()
            );
            let mut named: Vec<(u64, [F128; 2], usize)> = Vec::new();
            let ops = statement.arith.iter().map(|a| (a.op, a.key));
            for (n, key) in statement.prims.iter().copied().chain(ops) {
                match named.iter_mut().find(|x| x.0 == n && x.1 == key) {
                    Some(x) => x.2 += 1,
                    None => named.push((n, key, 1)),
                }
            }
            named.sort_unstable_by_key(|x| x.0);
            for (n, key, rows) in named {
                println!(
                    "statement: constant {n} has the key {:016x}{:016x} {:016x}{:016x} ({rows} rows read it)",
                    key[0].hi, key[0].lo, key[1].hi, key[1].lo
                );
            }
            println!(
                "first pass: {} leaves committed ({:.0} s, {})",
                words.len(),
                t0.elapsed().as_secs_f64(),
                rss()
            );

            // ---- SECOND PASS: the tower, from the last leaf back ----
            let t1 = Instant::now();
            let mut make_piece = |i: usize| {
                // A leaf's sources are read with nothing proven held (the
                // pool cleared) and their spare pages handed back before
                // the leaf proves.
                flock_core::scratch::clear();
                src.load(&profiles, &plan, i);
                let p = src.piece(&profiles[i], &plan[i], index[i]);
                src.rest();
                trim();
                p
            };
            // ---- IN SEGMENTS (LEAN_SEGMENTS=S): each segment a proof of
            // its own, kept on disk, so a run that is cut off resumes at
            // the first segment not yet proved; LEAN_SEGMENT=i proves only
            // segment i. The segments' claims are then joined. ----
            if let Some(s_n) = std::env::var("LEAN_SEGMENTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                type Kept = (
                    [u8; 32],
                    [u8; 32],
                    [u8; 32],
                    (u64, u64),
                    (u64, u64),
                    usize,
                    f64,
                    f64,
                    f64,
                );
                // The common leaves in s_n segments, then the rare rules'
                // leaves (another leaf circuit) in one of their own.
                let n_common = if a[0] == "run" { n_common } else { words.len() };
                let k = n_common / 2;
                let mut bounds: Vec<(usize, usize)> = (0..s_n)
                    .map(|s| (2 * (s * k / s_n), 2 * ((s + 1) * k / s_n)))
                    .collect();
                if n_common < words.len() {
                    bounds.push((n_common, words.len()));
                }
                let s_n = bounds.len();
                assert!(
                    bounds.iter().all(|(a, b)| b - a >= 4),
                    "at least four leaves a segment"
                );
                let roots: Vec<[u8; 32]> = bounds
                    .iter()
                    .map(|(a, b)| lean_segment_root(&words[*a..*b]))
                    .collect();
                let all = lean_segments_root(&roots);
                let only = std::env::var("LEAN_SEGMENT")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok());
                let file =
                    |s: usize| format!("{out}/segment-{slots}-{}-{s_n}-{s}.bin", index.len());
                let kept = |s: usize| -> Option<Kept> {
                    let k: Kept = bincode::deserialize(&fs::read(file(s)).ok()?).ok()?;
                    (k.0 == all && k.1 == roots[s]).then_some(k)
                };
                for s in 0..s_n {
                    if kept(s).is_some() || only.is_some_and(|o| o != s) {
                        continue;
                    }
                    let (a, b) = bounds[s];
                    let t = Instant::now();
                    let o = prove_lean_segment(
                        TowerConfig::Chain128,
                        &words[a..b],
                        a,
                        Some(all),
                        &mut make_piece,
                        env,
                        |line| {
                            println!(
                                "segment {s}: {line} ({:.0} s, {})",
                                t1.elapsed().as_secs_f64(),
                                rss()
                            )
                        },
                    );
                    let r = &o.report;
                    let k: Kept = (
                        all,
                        o.root,
                        o.seed,
                        (o.send.lo, o.send.hi),
                        (o.recv.lo, o.recv.hi),
                        r.root_proof_bytes,
                        r.root_verify_ms,
                        r.discharge_ms,
                        t.elapsed().as_secs_f64(),
                    );
                    fs::write(file(s), bincode::serialize(&k).unwrap()).unwrap();
                    println!(
                        "segment {s} of {s_n} proved: leaves {a}..{b}; proof {} KiB; verify {:.1} ms + {:.1} ms; {:.0} s",
                        r.root_proof_bytes / 1024,
                        r.root_verify_ms,
                        r.discharge_ms,
                        t.elapsed().as_secs_f64()
                    );
                }
                let done: Vec<Option<Kept>> = (0..s_n).map(kept).collect();
                let have = done.iter().flatten().count();
                if have < s_n {
                    println!("{have} of {s_n} segments proved");
                    return;
                }
                let done: Vec<Kept> = done.into_iter().flatten().collect();
                let claims: Vec<SegmentClaim> = done
                    .iter()
                    .map(|k| SegmentClaim {
                        root: k.1,
                        seed: k.2,
                        send: F128::new(k.3.0, k.3.1),
                        recv: F128::new(k.4.0, k.4.1),
                    })
                    .collect();
                let balanced = segments_balanced(&all, &claims, &statement);
                println!("the segments joined: {balanced:?}");
                assert_eq!(
                    balanced,
                    Ok(true),
                    "the shared bus balances over the whole cone"
                );
                println!(
                    "THE CONE in {s_n} proofs: {} leaves; proofs {} KiB together; verify {:.1} ms (proofs) + {:.1} ms (root-side checks); proving {:.0} s",
                    words.len(),
                    done.iter().map(|k| k.5).sum::<usize>() / 1024,
                    done.iter().map(|k| k.6).sum::<f64>(),
                    done.iter().map(|k| k.7).sum::<f64>(),
                    done.iter().map(|k| k.8).sum::<f64>()
                );
                return;
            }
            let r = prove_lean_tower_streaming(
                TowerConfig::Chain128,
                &words,
                &mut make_piece,
                &statement,
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
