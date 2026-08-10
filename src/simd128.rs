//! 4-lane backend on wasm simd128

use {crate::lanes::Lanes, std::arch::wasm32::*};

#[derive(Clone, Copy)]
pub struct Simd128(pub(crate) v128);

/// Rotates right by `R`: wasm has no vector rotate, so shift pair + or
#[inline(always)]
fn rot<const R: u32>(x: v128) -> v128 {
    v128_or(u32x4_shr(x, R), u32x4_shl(x, 32 - R))
}

/// Transposes four lanes' worth of four consecutive words
#[inline(always)]
fn transpose4(a: [v128; 4]) -> [v128; 4] {
    let lo01 = i32x4_shuffle::<0, 4, 1, 5>(a[0], a[1]);
    let lo23 = i32x4_shuffle::<0, 4, 1, 5>(a[2], a[3]);
    let hi01 = i32x4_shuffle::<2, 6, 3, 7>(a[0], a[1]);
    let hi23 = i32x4_shuffle::<2, 6, 3, 7>(a[2], a[3]);
    [
        i32x4_shuffle::<0, 1, 4, 5>(lo01, lo23),
        i32x4_shuffle::<2, 3, 6, 7>(lo01, lo23),
        i32x4_shuffle::<0, 1, 4, 5>(hi01, hi23),
        i32x4_shuffle::<2, 3, 6, 7>(hi01, hi23),
    ]
}

/// Loads 16 bytes at `off`, byte-swapping each word to native order
///
/// # Safety
///
/// `p.add(off)` must be valid for reads of 16 bytes.
#[inline(always)]
unsafe fn load_be(p: *const u8, off: usize) -> v128 {
    let raw = v128_load(p.add(off).cast());
    // Reverse the four bytes within each 32-bit word.
    i8x16_shuffle::<3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12>(raw, raw)
}

/// The one concrete instantiation of the flat kernel for this backend
///
/// Entry points reach it through `GroupFn` so the kernel is compiled exactly
/// once.
pub(crate) fn group(msgs: &[crate::batch::Message<'_>], out: &mut [[u8; 32]]) {
    crate::batch::hash_lanes::<Simd128, { <Simd128 as Lanes>::N }>(msgs, out)
}

/// # Safety
///
/// An engine without simd128 rejects the module at instantiation, so a call
/// that runs at all is safe.
pub(crate) unsafe fn steps(h: &mut [[u32; 8]], n: u64) {
    crate::chain::steps_lanes::<Simd128, { <Simd128 as Lanes>::N }>(h, n)
}

impl Lanes for Simd128 {
    const N: usize = 4;

    #[inline(always)]
    unsafe fn transpose(ptrs: &[*const u8], n: usize) -> [Self; 16] {
        debug_assert!((1..=4).contains(&n));
        let mut w = [Self::splat(0); 16];
        unsafe {
            // Inactive lanes reuse lane 0 so every lane holds valid data.
            let mut b = [ptrs[0]; 4];
            b[..n].copy_from_slice(&ptrs[..n]);
            // 16 words per block, four lanes at a time.
            for g in 0..4 {
                let off = g * 16;
                let loaded = [
                    load_be(b[0], off),
                    load_be(b[1], off),
                    load_be(b[2], off),
                    load_be(b[3], off),
                ];
                let t = transpose4(loaded);
                for (j, tj) in t.into_iter().enumerate() {
                    w[g * 4 + j] = Simd128(tj);
                }
            }
        }
        w
    }

    #[inline(always)]
    fn splat(v: u32) -> Self {
        Simd128(u32x4_splat(v))
    }

    #[inline(always)]
    fn load(v: &[u32]) -> Self {
        debug_assert_eq!(v.len(), 4);
        // v128_load is alignment-tolerant in wasm.
        Simd128(unsafe { v128_load(v.as_ptr().cast()) })
    }

    #[inline(always)]
    fn store(self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 4);
        unsafe { v128_store(out.as_mut_ptr().cast(), self.0) }
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Simd128(u32x4_add(self.0, o.0))
    }

    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        Simd128(v128_xor(self.0, o.0))
    }

    #[inline(always)]
    fn and(self, o: Self) -> Self {
        Simd128(v128_and(self.0, o.0))
    }

    #[inline(always)]
    fn not_and(self, o: Self) -> Self {
        // v128_andnot is `a & !b`, so the operands swap to give `!self & o`.
        Simd128(v128_andnot(o.0, self.0))
    }

    #[inline(always)]
    fn shr<const B: u32>(self) -> Self {
        Simd128(u32x4_shr(self.0, B))
    }

    #[inline(always)]
    fn rotr<const B: u32>(self) -> Self {
        Simd128(rot::<B>(self.0))
    }

    /// `Ch(x, y, z)` is exactly `bitselect(y, z, x)`: the two masked terms are
    /// disjoint, so their xor is an or, which is the select. One instruction.
    #[inline(always)]
    fn ch(self, y: Self, z: Self) -> Self {
        Simd128(v128_bitselect(y.0, z.0, self.0))
    }

    /// Where `y == z` the majority is that value; where they differ, `x`
    /// decides. `bitselect(x, z, y ^ z)` says precisely that, in two
    /// operations against the generic four.
    #[inline(always)]
    fn maj(self, y: Self, z: Self) -> Self {
        Simd128(v128_bitselect(self.0, z.0, v128_xor(y.0, z.0)))
    }
}

/// `W` interleaved 4-lane waves, hashing `4*W` messages per group.
#[derive(Clone, Copy)]
pub struct Waves<const W: usize>([Simd128; W]);

// One concrete instantiation per wave count, reached through GroupFn like
// every other kernel so each is compiled exactly once.
pub(crate) fn group8(msgs: &[crate::batch::Message<'_>], out: &mut [[u8; 32]]) {
    crate::batch::hash_lanes::<Waves<2>, { <Waves<2> as Lanes>::N }>(msgs, out)
}

pub(crate) fn group16(msgs: &[crate::batch::Message<'_>], out: &mut [[u8; 32]]) {
    crate::batch::hash_lanes::<Waves<4>, { <Waves<4> as Lanes>::N }>(msgs, out)
}

impl<const W: usize> Lanes for Waves<W> {
    const N: usize = 4 * W;

    #[inline(always)]
    unsafe fn transpose(ptrs: &[*const u8], n: usize) -> [Self; 16] {
        debug_assert!((1..=Self::N).contains(&n));
        unsafe {
            // One 4-lane transpose per wave. A wave with no active lanes
            // still needs valid data everywhere; lane 0 serves, as in the
            // scalar gather.
            let per: [[Simd128; 16]; W] = std::array::from_fn(|i| {
                let active = n.saturating_sub(4 * i).min(4);
                if active == 0 {
                    <Simd128 as Lanes>::transpose(ptrs, 1)
                } else {
                    <Simd128 as Lanes>::transpose(&ptrs[4 * i..], active)
                }
            });
            std::array::from_fn(|j| Waves(std::array::from_fn(|i| per[i][j])))
        }
    }

    #[inline(always)]
    fn splat(v: u32) -> Self {
        Waves([Simd128::splat(v); W])
    }

    #[inline(always)]
    fn load(v: &[u32]) -> Self {
        debug_assert_eq!(v.len(), Self::N);
        Waves(std::array::from_fn(|i| Simd128::load(&v[4 * i..4 * i + 4])))
    }

    #[inline(always)]
    fn store(self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), Self::N);
        for (i, w) in self.0.into_iter().enumerate() {
            w.store(&mut out[4 * i..4 * i + 4]);
        }
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].add(o.0[i])))
    }

    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].xor(o.0[i])))
    }

    #[inline(always)]
    fn and(self, o: Self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].and(o.0[i])))
    }

    #[inline(always)]
    fn not_and(self, o: Self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].not_and(o.0[i])))
    }

    #[inline(always)]
    fn shr<const B: u32>(self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].shr::<B>()))
    }

    #[inline(always)]
    fn rotr<const B: u32>(self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].rotr::<B>()))
    }

    #[inline(always)]
    fn ch(self, y: Self, z: Self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].ch(y.0[i], z.0[i])))
    }

    #[inline(always)]
    fn maj(self, y: Self, z: Self) -> Self {
        Waves(std::array::from_fn(|i| self.0[i].maj(y.0[i], z.0[i])))
    }
}
