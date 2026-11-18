//! The rows a cone's certification leaves to be taken as given, per rule:
//! parses each rows file with the declaration info, as `lean_cone` does,
//! and counts the rules no table checks. With `LEAN_SHOW_LEFT` set, the
//! certification also prints each leftover universe row's levels.
//!
//! ```text
//! lean_left INFO ROWSDIR LIST     # LIST: one rows file name per line
//! ```

use flock_prover::lean_rows::Rows;
use std::collections::BTreeMap;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let info = Rows::parse(&std::fs::read_to_string(&a[1]).expect("the info rows")).expect("info rows");
    let mut left: BTreeMap<String, usize> = BTreeMap::new();
    for name in std::fs::read_to_string(&a[3]).expect("a list").lines() {
        let Ok(text) = std::fs::read_to_string(format!("{}/{name}", a[2])) else { continue };
        let rows = Rows::parse_with(&text, Some(&info)).unwrap_or_else(|e| panic!("{name}: {e}"));
        if let Ok(want) = std::env::var("LEAN_FIND") {
            let census = rows.rule_census(&rows);
            for w in want.split(',') {
                if census.get(w).is_some_and(|x| x.1) {
                    eprintln!("FOUND {w} {} {name}", text.lines().count());
                }
            }
        }
        for (rule, (n, checked)) in rows.rule_census(&rows) {
            if !checked {
                *left.entry(rule).or_default() += n;
            }
        }
    }
    for (rule, n) in left {
        println!("{rule}\t{n}");
    }
}
