//! Mean pooling over stored rows without dequantizing them.
//!
//! A quantized row carries a power-of-two exponent chosen at build time so
//! that small-norm rows (an untrained `[UNK]`, SIF-down-weighted frequent
//! tokens) still use the full 8 or 16 bits: the stored bytes represent
//! `2^-exp * (min + q * scale)`.
//!
//! Rows are not summed as they arrive. Their addresses are collected and
//! the sum runs afterwards one column block at a time with the integer
//! lanes held in registers: each row byte is read once and the accumulator
//! never touches memory. (Explicitly touching each row's cache lines at
//! lookup time was measured on real data and made no difference; the
//! out-of-order core already overlaps those misses.) Rows with exponent 0,
//! which is nearly all of them, go through that path; the few boosted rows
//! are widened and added to an f32 side array scaled by `2^-exp`. The
//! per-column constants are applied once at the end, directly into the
//! caller's output when the text fit a single batch.

use core::mem::MaybeUninit;

use crate::Elem;

/// Largest row exponent ese-build emits.
pub const MAX_ROW_EXP: usize = 8;

/// Rows collected before a sum pass. Bounded so that integer lanes cannot
/// overflow for any [`Elem`] and so the address list stays on the stack.
pub const BATCH: usize = 256;

/// Columns summed per register-resident pass.
const BLOCK: usize = 32;

/// Pending row addresses plus the running sum.
///
/// This struct is several KB; build it in place with [`Accumulator::init`]
/// and borrow it, never move it. The f32 sum is only initialized when a
/// batch drains or a boosted row arrives; a text that fits one batch and
/// has no boosted rows is dequantized straight into the output.
pub struct Accumulator<E: Elem, const D: usize> {
    pending: [MaybeUninit<&'static [E; D]>; BATCH],
    npending: usize,
    sum: MaybeUninit<[f32; D]>,
    started: bool,
    /// Sum over boosted rows of `2^-exp`; exponent-0 rows count 1 each via
    /// `count - boosted`.
    boosted_weight: f32,
    boosted: u32,
    count: u32,
    /// Tokens seen so far, pooled or dropped; stops pooling at `limit`.
    produced: u32,
    limit: u32,
}

#[inline(always)]
fn exp_factor(exp: u8) -> f32 {
    1.0 / (1u32 << exp) as f32
}

/// Sum `rows[.., off..off + B]` in integer lanes and widen to f32.
#[inline(always)]
fn sum_block<E: Elem, const D: usize, const B: usize>(rows: &[&[E; D]], off: usize) -> [f32; B] {
    debug_assert!(off + B <= D);
    let mut acc = [E::ZERO; B];
    for row in rows {
        // SAFETY: the caller keeps `off + B <= D`.
        let r: &[E; B] = unsafe { &*(row.as_ptr().add(off) as *const [E; B]) };
        for j in 0..B {
            acc[j] = E::add(acc[j], r[j]);
        }
    }
    let mut out = [0.0f32; B];
    for j in 0..B {
        out[j] = E::acc_to_f32(acc[j]);
    }
    out
}

/// Sum `rows` and hand each column block's f32 totals to `emit(offset,
/// totals)`, in column order.
#[inline(always)]
fn sum_rows<E: Elem, const D: usize>(rows: &[&[E; D]], mut emit: impl FnMut(usize, &[f32])) {
    let full = D - D % BLOCK;
    let mut off = 0;
    while off < full {
        let s = sum_block::<E, D, BLOCK>(rows, off);
        emit(off, &s);
        off += BLOCK;
    }
    if off < D {
        let mut s = [0.0f32; BLOCK];
        for j in off..D {
            let mut a = E::ZERO;
            for row in rows {
                a = E::add(a, row[j]);
            }
            s[j - off] = E::acc_to_f32(a);
        }
        emit(off, &s[..D - off]);
    }
}

impl<E: Elem, const D: usize> Accumulator<E, D> {
    const _BATCH_FITS: () = assert!(BATCH as u32 <= E::MAX_BATCH);

    /// Initialize an accumulator in caller-provided storage. Returning the
    /// struct by value would copy it; this writes only the small fields.
    #[inline(always)]
    pub fn init(slot: &mut MaybeUninit<Self>) -> &mut Self {
        let () = Self::_BATCH_FITS;
        let p = slot.as_mut_ptr();
        // SAFETY: each field is written through a raw pointer into `slot`
        // without reading it; `pending` and `sum` are MaybeUninit.
        unsafe {
            core::ptr::addr_of_mut!((*p).npending).write(0);
            core::ptr::addr_of_mut!((*p).started).write(false);
            core::ptr::addr_of_mut!((*p).boosted_weight).write(0.0);
            core::ptr::addr_of_mut!((*p).boosted).write(0);
            core::ptr::addr_of_mut!((*p).count).write(0);
            core::ptr::addr_of_mut!((*p).produced).write(0);
            core::ptr::addr_of_mut!((*p).limit).write(u32::MAX);
            slot.assume_init_mut()
        }
    }

    /// Number of rows added so far.
    #[inline(always)]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Stop after `max_tokens` tokens (pooled or dropped unknowns), the way
    /// `tokenizers` truncation does before model2vec drops `[UNK]` ids.
    /// `0` means no limit.
    #[inline(always)]
    pub fn set_limit(&mut self, max_tokens: usize) {
        self.limit = if max_tokens == 0 {
            u32::MAX
        } else {
            max_tokens.min(u32::MAX as usize) as u32
        };
    }

    /// Whether the token limit has been reached; later tokens are ignored.
    #[inline(always)]
    pub fn full(&self) -> bool {
        self.produced >= self.limit
    }

    /// Count a token that is not pooled (a dropped unknown) toward the limit.
    #[inline(always)]
    pub fn skip(&mut self) {
        if !self.full() {
            self.produced += 1;
        }
    }

    /// Add a row stored with exponent `exp` (always 0 for f32 rows).
    #[inline(always)]
    pub fn add(&mut self, row: &'static [E; D], exp: u8) {
        if self.produced >= self.limit {
            return;
        }
        self.produced += 1;
        if exp == 0 {
            if self.npending == BATCH {
                self.drain();
            }
            self.pending[self.npending].write(row);
            self.npending += 1;
        } else {
            self.add_boosted(row, exp);
        }
        self.count += 1;
    }

    #[inline(always)]
    fn pending(&self) -> &[&'static [E; D]] {
        // SAFETY: `pending[..npending]` has been written by `add`.
        unsafe { core::slice::from_raw_parts(self.pending.as_ptr() as *const _, self.npending) }
    }

    #[inline(always)]
    fn sum_mut(&mut self) -> &mut [f32; D] {
        if !self.started {
            self.sum.write([0.0; D]);
            self.started = true;
        }
        // SAFETY: initialized just above.
        unsafe { self.sum.assume_init_mut() }
    }

    #[inline(never)]
    fn add_boosted(&mut self, row: &[E; D], exp: u8) {
        let f = exp_factor(exp);
        let sum = self.sum_mut();
        for (s, &e) in sum.iter_mut().zip(row.iter()) {
            *s += e.to_f32() * f;
        }
        self.boosted_weight += f;
        self.boosted += 1;
    }

    /// Sum the pending rows into the f32 side array.
    #[inline(never)]
    fn drain(&mut self) {
        if self.npending == 0 {
            return;
        }
        let Self {
            pending,
            npending,
            sum,
            started,
            ..
        } = self;
        // SAFETY: `pending[..npending]` has been written by `add`.
        let rows: &[&'static [E; D]] =
            unsafe { core::slice::from_raw_parts(pending.as_ptr() as *const _, *npending) };
        let out = sum.as_mut_ptr() as *mut f32;
        if *started {
            sum_rows(rows, |off, s| {
                for (j, v) in s.iter().enumerate() {
                    // SAFETY: `sum` is initialized and `off + j < D`.
                    unsafe { *out.add(off + j) += v };
                }
            });
        } else {
            sum_rows(rows, |off, s| {
                for (j, v) in s.iter().enumerate() {
                    // SAFETY: `off + j < D`; every lane is written exactly once.
                    unsafe { out.add(off + j).write(*v) };
                }
            });
            *started = true;
        }
        *npending = 0;
    }

    /// Write the dequantized mean, L2-normalized if requested, into `out`.
    /// Zero if nothing was added. `min` and `scale` are the per-column
    /// dequantization constants and are ignored for f32 rows.
    #[inline]
    pub fn finish_into(
        &mut self,
        out: &mut MaybeUninit<[f32; D]>,
        min: &[f32; D],
        scale: &[f32; D],
        normalize: bool,
    ) {
        if self.count == 0 {
            out.write([0.0; D]);
            return;
        }
        let inv_n = 1.0 / self.count as f32;
        // Weight carried by the per-column `min`: 1 per plain row, 2^-exp
        // per boosted row.
        let weight_sum = (self.count - self.boosted) as f32 + self.boosted_weight;
        let o = out.as_mut_ptr() as *mut f32;
        // Sum of squares for normalization, accumulated in the same pass in
        // eight independent lanes (a serial chain costs ~180 ns for 256
        // lanes).
        let mut sq = [0.0f32; 8];
        let mut dequant = |off: usize, s: &[f32]| {
            for (j, &v) in s.iter().enumerate() {
                let k = off + j;
                let y = if E::QUANTIZED {
                    (v * scale[k] + min[k] * weight_sum) * inv_n
                } else {
                    v * inv_n
                };
                // SAFETY: `k < D`; every lane is written exactly once.
                unsafe { o.add(k).write(y) };
            }
            if normalize {
                // SAFETY: `out[off..off + s.len()]` was written just above.
                let ys = unsafe { core::slice::from_raw_parts(o.add(off), s.len()) };
                let mut chunks = ys.chunks_exact(8);
                for c in &mut chunks {
                    for (l, y) in sq.iter_mut().zip(c) {
                        *l += y * y;
                    }
                }
                for (l, y) in sq.iter_mut().zip(chunks.remainder()) {
                    *l += y * y;
                }
            }
        };
        if self.started {
            self.drain();
            // SAFETY: `started` is only set after `sum` is written.
            let sum = unsafe { self.sum.assume_init_ref() };
            dequant(0, sum);
        } else {
            // Common case: one batch, no boosted rows, no side array.
            sum_rows(self.pending(), dequant);
        }
        // SAFETY: every lane was written above.
        let out = unsafe { out.assume_init_mut() };
        if normalize {
            let norm = sq.iter().sum::<f32>().sqrt();
            if norm > 0.0 {
                let inv = 1.0 / norm;
                for v in out.iter_mut() {
                    *v *= inv;
                }
            }
        }
    }

    /// [`finish_into`](Self::finish_into) returning the vector by value.
    #[inline]
    pub fn finish(&mut self, min: &[f32; D], scale: &[f32; D], normalize: bool) -> [f32; D] {
        let mut out = MaybeUninit::uninit();
        self.finish_into(&mut out, min, scale, normalize);
        // SAFETY: `finish_into` writes every lane.
        unsafe { out.assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_sums_drain_before_overflow() {
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<u8, 4>::init(&mut slot);
        static ROW: [u8; 4] = [255, 0, 128, 1];
        for _ in 0..600 {
            acc.add(&ROW, 0);
        }
        assert_eq!(acc.count(), 600);
        let out = acc.finish(&[0.0; 4], &[1.0; 4], false);
        assert_eq!(out, [255.0, 0.0, 128.0, 1.0]);
    }

    #[test]
    fn odd_widths_use_the_tail_path() {
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<u8, 37>::init(&mut slot);
        static A: [u8; 37] = [1; 37];
        static B: [u8; 37] = [3; 37];
        acc.add(&A, 0);
        acc.add(&B, 0);
        let out = acc.finish(&[0.0; 37], &[1.0; 37], false);
        assert_eq!(out, [2.0; 37]);
        // and through the side array
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<u8, 37>::init(&mut slot);
        for _ in 0..300 {
            acc.add(&A, 0);
            acc.add(&B, 0);
        }
        let out = acc.finish(&[0.0; 37], &[1.0; 37], false);
        assert_eq!(out, [2.0; 37]);
    }

    #[test]
    fn per_column_dequantization() {
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<u8, 2>::init(&mut slot);
        acc.add(&[0, 255], 0);
        acc.add(&[255, 255], 0);
        // column 0: values -1 and 1 -> mean 0; column 1: both 3 -> 3
        let out = acc.finish(&[-1.0, 1.0], &[2.0 / 255.0, 2.0 / 255.0], false);
        assert!((out[0]).abs() < 1e-6 && (out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn row_exponents_scale_rows_and_min() {
        // min = 1, scale = 1: a stored q with exponent e is 2^-e * (1 + q).
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<u8, 2>::init(&mut slot);
        acc.add(&[7, 0], 0); // 8, 1
        acc.add(&[3, 1], 2); // (1+3)/4 = 1, (1+1)/4 = 0.5
        let out = acc.finish(&[1.0, 1.0], &[1.0, 1.0], false);
        assert!((out[0] - 4.5).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.75).abs() < 1e-6, "{out:?}");
        // the same through a drain
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<u8, 2>::init(&mut slot);
        for _ in 0..300 {
            acc.add(&[7, 0], 0);
            acc.add(&[3, 1], 2);
        }
        let out = acc.finish(&[1.0, 1.0], &[1.0, 1.0], false);
        assert!(
            (out[0] - 4.5).abs() < 1e-4 && (out[1] - 0.75).abs() < 1e-4,
            "{out:?}"
        );
        // only boosted rows
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<u8, 2>::init(&mut slot);
        acc.add(&[3, 1], 8);
        let out = acc.finish(&[1.0, 1.0], &[1.0, 1.0], false);
        assert!((out[0] - 4.0 / 256.0).abs() < 1e-7 && (out[1] - 2.0 / 256.0).abs() < 1e-7);
    }

    #[test]
    fn f32_mean_and_normalize() {
        let mut slot = MaybeUninit::uninit();
        let acc = Accumulator::<f32, 3>::init(&mut slot);
        acc.add(&[3.0, 0.0, 0.0], 0);
        acc.add(&[5.0, 0.0, 0.0], 0);
        assert_eq!(acc.finish(&[0.0; 3], &[1.0; 3], true), [1.0, 0.0, 0.0]);
        let mut empty = MaybeUninit::uninit();
        assert_eq!(
            Accumulator::<f32, 3>::init(&mut empty).finish(&[0.0; 3], &[1.0; 3], true),
            [0.0; 3]
        );
    }
}
