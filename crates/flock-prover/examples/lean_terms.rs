//! Counts terms and declarations across rows files, per file and after
//! sharing equal structure across all of them: what one global term table
//! would hold. Usage: `lean_terms DIR`.

use std::collections::HashMap;

fn main() {
    let dir = std::env::args().nth(1).expect("a directory");
    let mut global: HashMap<String, u32> = HashMap::new();
    let (mut terms, mut decls, mut facts, mut files) = (0usize, 0usize, 0usize, 0usize);
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    for p in paths {
        let text = std::fs::read_to_string(&p).unwrap();
        files += 1;
        let mut local: Vec<u32> = Vec::new();
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            match w.first().copied() {
                Some("T") => {
                    terms += 1;
                    let m = |i: usize| local[w[i].parse::<usize>().unwrap()].to_string();
                    // Declarations are local numbers: keep them file-scoped,
                    // which undercounts sharing of constants (an upper bound).
                    let key = match w[2] {
                        "app" | "lam" | "pi" => format!("{} {} {}", w[2], m(3), m(4)),
                        "let" => format!("let {} {} {}", m(3), m(4), m(5)),
                        "proj" => format!("proj {}:{} {} {}", files, w[3], w[4], m(5)),
                        "leaf" if w[3] == "const" => format!("const {files}:{} {}", w[4], w[5]),
                        _ => w[2..].join(" "),
                    };
                    let n = global.len() as u32;
                    local.push(*global.entry(key).or_insert(n));
                }
                Some("D") => decls += 1,
                Some("F") => facts += 1,
                _ => {}
            }
        }
    }
    println!(
        "{files} files: terms {terms}, shared across files at most {}, decls {decls}, facts {facts}",
        global.len()
    );
}
