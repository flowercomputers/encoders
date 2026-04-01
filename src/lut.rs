include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

#[repr(C, align(4))]
struct AlignedWeights([u8; TABLE_SIZE * DIMENSIONS * DTYPE_SIZE]);
static ALIGNED_W: AlignedWeights =
    AlignedWeights(*include_bytes!(concat!(env!("OUT_DIR"), "/weights.bin")));
static WEIGHTS: &'static [Param; TABLE_SIZE] = unsafe { std::mem::transmute(&ALIGNED_W.0) };

#[repr(C, align(4))]
struct AlignedSeeds([u8; NUM_BUCKETS * 4]);
static ALIGNED_S: AlignedSeeds =
    AlignedSeeds(*include_bytes!(concat!(env!("OUT_DIR"), "/seeds.bin")));
static SEEDS: &'static [u32; NUM_BUCKETS] = unsafe { std::mem::transmute(&ALIGNED_S.0) };

#[repr(C, align(8))]
struct AlignedVerify([u8; TABLE_SIZE * 8]);
static ALIGNED_V: AlignedVerify =
    AlignedVerify(*include_bytes!(concat!(env!("OUT_DIR"), "/verify.bin")));
static VERIFY: &'static [u64; TABLE_SIZE] = unsafe { std::mem::transmute(&ALIGNED_V.0) };

#[inline(always)]
fn phf_hash(key: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325 ^ seed;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[inline(always)]
pub fn lookup(key: &str) -> Option<&'static Param> {
    let bytes = key.as_bytes();
    let bucket = (phf_hash(bytes, 0) as usize) % NUM_BUCKETS;
    unsafe {
        let slot = (phf_hash(bytes, *(SEEDS.get_unchecked(bucket)) as u64) as usize) & TABLE_MASK;
        if *(VERIFY.get_unchecked(slot)) == phf_hash(bytes, u64::MAX) {
            Some(WEIGHTS.get_unchecked(slot))
        } else {
            None
        }
    }
}
