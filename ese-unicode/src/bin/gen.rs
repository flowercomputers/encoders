//! Generates `src/tables.rs` by asking the reference crates about every
//! code point. Run with `cargo run -p ese-unicode --features gen --bin gen`.
//!
//! Layout (see `lib.rs`): u8 class pages for every code point; for the
//! pages that contain any normalization mapping, a 256-bit bitmap whose
//! rank gives each mapped code point's position in a per-page run of
//! `RESULTS`, one u32 each holding the strip-plus-lowercase form inline (or
//! an escape to the multi-character pool) plus flags from which the
//! strip-only and lowercase-only forms are derived; the few characters whose
//! forms are not derivable go to `EXCEPTIONS`.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::char::decompose_canonical;

const CONTROL: u8 = 1;
const WHITESPACE: u8 = 1 << 1;
const PUNCTUATION: u8 = 1 << 2;
const CJK: u8 = 1 << 3;
const NONSPACING_MARK: u8 = 1 << 4;
const HANGUL_SYLLABLE: u8 = 1 << 5;

// Result word encoding (must match lib.rs).
const CP_MASK: u32 = 0x1F_FFFF;
const EMPTY: u32 = 0x11_0000;
const MULTI: u32 = 1 << 31;
const STRIP_IS_IDENTITY: u32 = 1 << 30;
const LOWER_IS_IDENTITY: u32 = 1 << 29;
const EXCEPTION: u32 = 1 << 28;

fn is_chinese_char(cp: u32) -> bool {
    matches!(cp,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F |
        0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

fn class(c: char) -> u8 {
    let cat = get_general_category(c);
    let mut k = 0;
    if matches!(
        cat,
        GeneralCategory::Control | GeneralCategory::Format | GeneralCategory::PrivateUse
    ) {
        k |= CONTROL;
    }
    if c.is_whitespace() {
        k |= WHITESPACE;
    }
    if matches!(
        cat,
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    ) {
        k |= PUNCTUATION;
    }
    if is_chinese_char(c as u32) {
        k |= CJK;
    }
    if cat == GeneralCategory::NonspacingMark {
        k |= NONSPACING_MARK;
    }
    if (0xAC00..=0xD7A3).contains(&(c as u32)) {
        k |= HANGUL_SYLLABLE;
    }
    k
}

fn strip(c: char) -> String {
    let mut s = String::new();
    decompose_canonical(c, |d| {
        if get_general_category(d) != GeneralCategory::NonspacingMark {
            s.push(d);
        }
    });
    s
}

fn lower(s: &str) -> String {
    s.chars().flat_map(char::to_lowercase).collect()
}

/// Encodes a normalized form as a result word (without flags).
struct Multi {
    strings: Vec<String>,
    ids: HashMap<String, u32>,
}

impl Multi {
    fn encode(&mut self, s: &str) -> u32 {
        let mut it = s.chars();
        match (it.next(), it.next()) {
            (None, _) => EMPTY,
            (Some(c), None) => c as u32,
            _ => {
                let id = *self.ids.entry(s.to_string()).or_insert_with(|| {
                    self.strings.push(s.to_string());
                    (self.strings.len() - 1) as u32
                });
                assert!(id < 0xFFFF);
                MULTI | id
            }
        }
    }
}

/// Reference crate versions, read from the pins in this crate's manifest so
/// the header cannot drift from what the generator was built against.
fn pinned_version(manifest: &str, krate: &str) -> String {
    for line in manifest.lines() {
        if let Some(rest) = line.trim().strip_prefix(krate)
            && let Some(v) = rest.split("version = \"=").nth(1)
        {
            return v.split('"').next().unwrap().to_string();
        }
    }
    panic!("no pinned version for {krate} in Cargo.toml");
}

fn main() {
    let manifest = include_str!("../../Cargo.toml");
    let ver_un = pinned_version(manifest, "unicode-normalization");
    let ver_ugc = pinned_version(manifest, "unicode-general-category");

    let mut multi = Multi {
        strings: Vec::new(),
        ids: HashMap::new(),
    };

    // Per code point: class, and the three forms if any differs from identity.
    let mut class_pages: Vec<[u8; 256]> = Vec::new();
    let mut class_page_ids: HashMap<[u8; 256], u8> = HashMap::new();
    let mut class_index: Vec<u8> = Vec::new();

    let mut map_pages: Vec<[u64; 4]> = Vec::new();
    let mut map_index: Vec<u8> = Vec::new();
    let mut map_base: Vec<u16> = Vec::new();
    let mut results: Vec<u32> = Vec::new();
    let mut exceptions: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut stats = (0usize, 0usize, 0usize); // mapped, derivable-A, derivable-B

    for page in 0..(0x110000 >> 8) {
        let mut cls = [0u8; 256];
        let mut map = [0u64; 4];
        let mut page_results: Vec<u32> = Vec::new();
        for low in 0..256 {
            let cp = (page << 8 | low) as u32;
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let k = class(c);
            cls[low] = k;
            if k & HANGUL_SYLLABLE != 0 {
                continue;
            }
            let ident = c.to_string();
            let st = strip(c);
            let lo = lower(&ident);
            let sl = lower(&st);
            if st == ident && lo == ident && sl == ident {
                continue;
            }
            stats.0 += 1;
            let mut word = multi.encode(&sl);
            let a = st == ident; // no decomposition
            let b = lo == ident; // no lowercase
            // Derivations the runtime applies:
            //   strip-only  = c if A, else SL if B, else exception
            //   lower-only  = c if B, else SL if A, else exception
            let strip_ok = if a {
                true
            } else if b {
                st == sl
            } else {
                false
            };
            let lower_ok = if b {
                true
            } else if a {
                lo == sl
            } else {
                false
            };
            if a {
                word |= STRIP_IS_IDENTITY;
                stats.1 += 1;
            }
            if b {
                word |= LOWER_IS_IDENTITY;
                stats.2 += 1;
            }
            if !(strip_ok && lower_ok) {
                word |= EXCEPTION;
                let s_word = multi.encode(&st);
                let l_word = multi.encode(&lo);
                exceptions.insert(cp, (s_word, l_word));
            }
            page_results.push(word);
            map[low >> 6] |= 1u64 << (low & 63);
        }
        let id = *class_page_ids.entry(cls).or_insert_with(|| {
            class_pages.push(cls);
            (class_pages.len() - 1) as u8
        });
        assert!(
            class_pages.len() <= 255,
            "more than 255 distinct class pages"
        );
        class_index.push(id);
        if page_results.is_empty() {
            map_index.push(0);
        } else {
            map_pages.push(map);
            assert!(map_pages.len() <= 255, "more than 255 mapping pages");
            map_index.push(map_pages.len() as u8);
            assert!(results.len() <= u16::MAX as usize);
            map_base.push(results.len() as u16);
            results.extend(page_results);
        }
    }

    let mut out = String::new();
    writeln!(
        out,
        "// Generated by `cargo run -p ese-unicode --features gen --bin gen`. Do not edit."
    )
    .unwrap();
    writeln!(
        out,
        "// Reference crates: unicode-normalization {ver_un}, unicode-general-category {ver_ugc}."
    )
    .unwrap();
    writeln!(
        out,
        "// {} class pages, {} mapping pages, {} mapped code points ({} strip-identity, {} lower-identity, {} exceptions), {} multi-char forms.",
        class_pages.len(), map_pages.len(), results.len(), stats.1, stats.2, exceptions.len(), multi.strings.len()
    ).unwrap();
    writeln!(out, "pub const REFERENCE_CRATES: &str = \"unicode-normalization {ver_un}, unicode-general-category {ver_ugc}\";").unwrap();
    writeln!(
        out,
        "pub const UNICODE_VERSION: (u8, u8, u8) = {:?};",
        unicode_normalization::UNICODE_VERSION
    )
    .unwrap();
    writeln!(
        out,
        "pub static CLASS_INDEX: [u8; {}] = {:?};",
        class_index.len(),
        class_index
    )
    .unwrap();
    writeln!(
        out,
        "pub static CLASS_PAGES: [[u8; 256]; {}] = [",
        class_pages.len()
    )
    .unwrap();
    for p in &class_pages {
        writeln!(out, "{p:?},").unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(
        out,
        "pub static MAP_INDEX: [u8; {}] = {:?};",
        map_index.len(),
        map_index
    )
    .unwrap();
    writeln!(
        out,
        "pub static MAP_PAGES: [[u64; 4]; {}] = [",
        map_pages.len()
    )
    .unwrap();
    for p in &map_pages {
        writeln!(out, "{p:?},").unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(
        out,
        "pub static MAP_BASE: [u16; {}] = {:?};",
        map_base.len(),
        map_base
    )
    .unwrap();
    writeln!(
        out,
        "pub static RESULTS: [u32; {}] = {:?};",
        results.len(),
        results
    )
    .unwrap();
    writeln!(
        out,
        "pub static MULTI: [&str; {}] = {:?};",
        multi.strings.len(),
        multi.strings
    )
    .unwrap();
    writeln!(
        out,
        "/// `(code point, strip-only form, lowercase-only form)`, sorted by code point."
    )
    .unwrap();
    writeln!(
        out,
        "pub static EXCEPTIONS: [(u32, u32, u32); {}] = [",
        exceptions.len()
    )
    .unwrap();
    for (cp, (s, l)) in &exceptions {
        writeln!(out, "({cp}, {s}, {l}),").unwrap();
    }
    writeln!(out, "];").unwrap();

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tables.rs");
    std::fs::write(path, &out).unwrap();
    let bytes = class_index.len()
        + class_pages.len() * 256
        + map_index.len()
        + map_pages.len() * 32
        + map_base.len() * 2
        + results.len() * 4
        + multi.strings.iter().map(|s| s.len() + 16).sum::<usize>()
        + exceptions.len() * 12;
    eprintln!(
        "wrote {path}: {} class pages, {} mapping pages, {} results, {} exceptions, {} multi; about {} KB of tables",
        class_pages.len(),
        map_pages.len(),
        results.len(),
        exceptions.len(),
        multi.strings.len(),
        bytes / 1024
    );
}
