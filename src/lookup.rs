include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

// All supported DIMENSIONS are multiples of 8, so this layout has no
// padding and matches the packed bytes in weights.bin exactly.
#[repr(C, align(8))]
pub struct Slot {
    verify: u64,
    /// per-row quantization parameters (0.0 / 1.0 for unquantized builds)
    pub min: f32,
    pub scale: f32,
    pub weight: Param,
}

const _: () = assert!(std::mem::size_of::<Slot>() == SLOT_SIZE);

#[repr(C, align(8))]
struct AlignedSlots([u8; TABLE_SIZE * SLOT_SIZE]);
static ALIGNED_SLOTS: AlignedSlots =
    AlignedSlots(*include_bytes!(concat!(env!("OUT_DIR"), "/weights.bin")));
static SLOTS: &'static [Slot; TABLE_SIZE] = unsafe { std::mem::transmute(&ALIGNED_SLOTS.0) };

#[repr(C, align(4))]
struct AlignedSeeds([u8; NUM_BUCKETS * 4]);
static ALIGNED_S: AlignedSeeds =
    AlignedSeeds(*include_bytes!(concat!(env!("OUT_DIR"), "/seeds.bin")));
static SEEDS: &'static [u32; NUM_BUCKETS] = unsafe { std::mem::transmute(&ALIGNED_S.0) };

#[inline(always)]
pub fn lookup(key: &str) -> Option<&'static Slot> {
    let bytes = key.as_bytes();
    let bucket = (hash(bytes, 0) as usize) % NUM_BUCKETS;
    unsafe {
        let slot = (hash(bytes, *SEEDS.get_unchecked(bucket) as u64) as usize) & TABLE_MASK;
        let entry = SLOTS.get_unchecked(slot);
        if entry.verify == hash(bytes, u64::MAX) {
            Some(entry)
        } else {
            None
        }
    }
}

#[inline(always)]
fn hash(key: &[u8], seed: u64) -> u64 {
    let mut h = seed ^ 0x517cc1b727220a95;
    for &b in key {
        h = (h ^ b as u64).wrapping_mul(0x2127599bf4325c37);
    }
    h ^ (h >> 32)
}
