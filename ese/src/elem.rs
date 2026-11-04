/// Storage type of an embedding element in the compiled-in table, and how
/// raw stored values are summed.
///
/// Dequantization is affine per column (`value = min + q * scale`), and the
/// output is a mean, so the inner loop only ever sums raw stored values in
/// `Acc` lanes. The per-column constants are applied once at the end.
pub trait Elem: Copy + Send + Sync + 'static {
    /// Lane type for the running sum of raw stored values.
    type Acc: Copy + Send + Sync + 'static;
    /// The stored value widened to f32 (for boosted rows).
    fn to_f32(self) -> f32;
    const ZERO: Self::Acc;
    /// `false` for f32 rows, whose per-column constants are the identity.
    const QUANTIZED: bool;
    /// How many rows can be summed into one `Acc` lane without overflow.
    /// `u32::MAX` means never.
    const MAX_BATCH: u32;
    fn add(acc: Self::Acc, e: Self) -> Self::Acc;
    fn acc_to_f32(acc: Self::Acc) -> f32;
}

impl Elem for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    type Acc = f32;
    const ZERO: f32 = 0.0;
    const QUANTIZED: bool = false;
    const MAX_BATCH: u32 = u32::MAX;
    #[inline(always)]
    fn add(acc: f32, e: f32) -> f32 {
        acc + e
    }
    #[inline(always)]
    fn acc_to_f32(acc: f32) -> f32 {
        acc
    }
}

impl Elem for u8 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    type Acc = u16;
    const ZERO: u16 = 0;
    const QUANTIZED: bool = true;
    // 257 * 255 = 65535 fits a u16 lane exactly.
    const MAX_BATCH: u32 = 257;
    #[inline(always)]
    fn add(acc: u16, e: u8) -> u16 {
        acc + e as u16
    }
    #[inline(always)]
    fn acc_to_f32(acc: u16) -> f32 {
        acc as f32
    }
}

impl Elem for u16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    type Acc = u32;
    const ZERO: u32 = 0;
    const QUANTIZED: bool = true;
    // 65537 * 65535 fits a u32 lane exactly.
    const MAX_BATCH: u32 = 65537;
    #[inline(always)]
    fn add(acc: u32, e: u16) -> u32 {
        acc + e as u32
    }
    #[inline(always)]
    fn acc_to_f32(acc: u32) -> f32 {
        acc as f32
    }
}
