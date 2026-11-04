//! Two-level perfect hash construction over the runtime's own hash
//! functions (`ese::phf`), so the two sides cannot drift.
//!
//! `bucket(hash_key(key))` picks a bucket, and `seeded(hash_key(key),
//! seeds[bucket])` masked to the table picks the slot. The full seeded value
//! is the slot's verifier (the emitter packs the row exponent into its low
//! byte).

use std::collections::HashSet;

use ese::phf::{bucket, hash_key, seeded};

use crate::Error;

pub(crate) struct Phf {
    pub table_size: usize,
    pub num_buckets: usize,
    /// Per-bucket displacement seed. Kept to 16 bits so the runtime's seed
    /// table stays small enough for L1; the table is enlarged instead if a
    /// bucket cannot be placed within that range.
    pub seeds: Vec<u16>,
    /// `slot -> index into the input key list`, `None` for empty slots.
    pub slots: Vec<Option<usize>>,
    /// Per-slot verifier: the full seeded hash of the key (0 for empty slots).
    pub verify: Vec<u64>,
}

pub(crate) fn build(keys: &[&[u8]]) -> Result<Phf, Error> {
    let n = keys.len();
    if n == 0 {
        return Err(Error::Invalid(
            "cannot build a hash table for an empty vocabulary".into(),
        ));
    }
    let mut table_size = n.next_power_of_two();
    loop {
        match build_sized(keys, table_size)? {
            Some(phf) => return Ok(phf),
            None if table_size < (1 << 40) => table_size *= 2,
            None => {
                return Err(Error::Invalid(
                    "perfect hash construction failed at every table size".into(),
                ));
            }
        }
    }
}

/// One construction attempt; `None` if some bucket needs a seed above
/// `u16::MAX` at this table size.
fn build_sized(keys: &[&[u8]], table_size: usize) -> Result<Option<Phf>, Error> {
    let n = keys.len();
    let table_mask = table_size - 1;
    let num_buckets = n;

    let hashes: Vec<u64> = keys.iter().map(|k| hash_key(k)).collect();
    let mut buckets: Vec<Vec<usize>> = vec![vec![]; num_buckets];
    for (i, &h) in hashes.iter().enumerate() {
        buckets[bucket(h, num_buckets)].push(i);
    }
    let mut order: Vec<usize> = (0..num_buckets).collect();
    order.sort_by_key(|&b| std::cmp::Reverse(buckets[b].len()));

    let mut seeds = vec![0u16; num_buckets];
    let mut slots = vec![None; table_size];
    let mut verify = vec![0u64; table_size];
    let mut trial: Vec<usize> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();

    for &b in &order {
        if buckets[b].is_empty() {
            continue;
        }
        let mut i: u32 = 0;
        loop {
            let seed = i | 1;
            i += 1;
            if seed > u16::MAX as u32 {
                return Ok(None);
            }
            trial.clear();
            seen.clear();
            let ok = buckets[b].iter().all(|&k| {
                let slot = (seeded(hashes[k], seed as u16) as usize) & table_mask;
                if slots[slot].is_some() || !seen.insert(slot) {
                    return false;
                }
                trial.push(slot);
                true
            });
            if ok {
                seeds[b] = seed as u16;
                for (&k, &slot) in buckets[b].iter().zip(&trial) {
                    slots[slot] = Some(k);
                    verify[slot] = seeded(hashes[k], seed as u16);
                }
                break;
            }
        }
    }

    Ok(Some(Phf {
        table_size,
        num_buckets,
        seeds,
        slots,
        verify,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the runtime lookup: the verifier alone decides, the key is
    /// never compared, so this also detects false positives.
    fn lookup(p: &Phf, key: &[u8]) -> Option<usize> {
        let h = hash_key(key);
        let x = seeded(h, p.seeds[bucket(h, p.num_buckets)]);
        let slot = (x as usize) & (p.table_size - 1);
        (p.verify[slot] == x).then_some(p.slots[slot]).flatten()
    }

    #[test]
    fn every_key_resolves_to_itself() {
        let owned: Vec<String> = (0..5000).map(|i| format!("tok{i}")).collect();
        let keys: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
        let p = build(&keys).unwrap();
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(lookup(&p, k), Some(i));
        }
        assert_eq!(lookup(&p, b"absent"), None);
    }

    /// Regression: with the seed mixed in before the key, "laz" under seed 7
    /// and "haz" under seed 3 hashed identically and a 105k-key table
    /// answered thousands of random short strings.
    #[test]
    fn no_false_positives_on_short_keys() {
        let mut owned: Vec<String> = Vec::new();
        for a in b'a'..=b'z' {
            for b in b'a'..=b'z' {
                for &c in b"aeiouzy" {
                    owned.push(format!("{}{}{}", a as char, b as char, c as char));
                    owned.push(format!("##{}{}{}", a as char, b as char, c as char));
                }
            }
        }
        for i in 0..60_000 {
            owned.push(format!("w{i}"));
        }
        let keys: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
        let set: std::collections::HashSet<&[u8]> = keys.iter().copied().collect();
        let p = build(&keys).unwrap();
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut false_positives = 0;
        let mut probes = 0;
        for _ in 0..300_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = 2 + (state % 4) as usize;
            let s: Vec<u8> = (0..len)
                .map(|i| b'a' + ((state >> (8 * i)) % 26) as u8)
                .collect();
            if set.contains(s.as_slice()) {
                continue;
            }
            probes += 1;
            if lookup(&p, &s).is_some() {
                false_positives += 1;
            }
        }
        assert!(probes > 100_000);
        assert_eq!(
            false_positives, 0,
            "{false_positives} false positives in {probes} probes"
        );
    }
}
