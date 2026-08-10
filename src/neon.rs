//! 4-lane backend on AArch64 NEON
//!
//! NEON is baseline on AArch64, so no runtime detection.

use {crate::lanes::Lanes, std::arch::aarch64::*};

#[derive(Clone, Copy)]
pub struct Neon(pub(crate) uint32x4_t);

/// Rotates right by `R`, given its precomputed complement `L = 32 - R`
///
/// Two ops, not the shift/shift/or triple: `vsri` shifts right and inserts
/// into the destination's remaining bits, so the left-shifted half doubles as
/// the insertion target. Measured 1.08x on the whole kernel together with the
/// bit-select Ch/Maj below (M4 Max, paired A/B).
#[inline(always)]
unsafe fn rot<const R: i32, const L: i32>(x: uint32x4_t) -> uint32x4_t {
    // A mispaired complement is a wrong hash, not a crash; fail the build.
    const { assert!(R + L == 32) };
    vsriq_n_u32::<R>(vshlq_n_u32::<L>(x), x)
}

/// Transposes four lanes' worth of four consecutive words
///
/// Input `a[i]` holds words `4g..4g+4` of lane `i`, already byte-swapped;
/// output `j` holds word `4g+j` of every lane.
#[inline(always)]
unsafe fn transpose4(a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
    // vtrnq only interleaves adjacent pairs, so recombining the low/high
    // halves of the two results is what finishes the 4x4 transpose.
    let t01 = vtrnq_u32(a[0], a[1]);
    let t23 = vtrnq_u32(a[2], a[3]);
    [
        vcombine_u32(vget_low_u32(t01.0), vget_low_u32(t23.0)),
        vcombine_u32(vget_low_u32(t01.1), vget_low_u32(t23.1)),
        vcombine_u32(vget_high_u32(t01.0), vget_high_u32(t23.0)),
        vcombine_u32(vget_high_u32(t01.1), vget_high_u32(t23.1)),
    ]
}

/// Loads 16 bytes at `off`, byte-swapping each word to native order
///
/// # Safety
///
/// `p.add(off)` must be valid for reads of 16 bytes.
#[inline(always)]
unsafe fn load_be(p: *const u8, off: usize) -> uint32x4_t {
    let raw = vld1q_u8(p.add(off));
    vreinterpretq_u32_u8(vrev32q_u8(raw))
}

/// The one concrete instantiation of the flat kernel for this backend
///
/// No feature gate: NEON is baseline on AArch64. Entry points reach it through
/// `GroupFn` so the kernel is compiled exactly once.
pub(crate) fn group(msgs: &[crate::batch::Message<'_>], out: &mut [[u8; 32]]) {
    crate::batch::hash_lanes::<Neon, { <Neon as Lanes>::N }>(msgs, out)
}

/// # Safety
///
/// NEON is baseline on AArch64, so this is always safe to call there.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn steps(h: &mut [[u32; 8]], n: u64) {
    crate::chain::steps_lanes::<Neon, { <Neon as Lanes>::N }>(h, n)
}

impl Lanes for Neon {
    const N: usize = 4;

    /// `Ch(x, y, z)` is exactly `vbsl(x, y, z)`: the two masked terms are
    /// disjoint, so their xor and a bit-select agree. One op instead of the
    /// default's and/bic/eor; simd128 does the same via `v128_bitselect`, and
    /// LLVM only fuses the default into `bsl` in some inlined copies.
    #[inline(always)]
    fn ch(self, y: Self, z: Self) -> Self {
        unsafe { Neon(vbslq_u32(self.0, y.0, z.0)) }
    }

    /// Majority: where `y` and `z` disagree, `x` casts the deciding vote.
    /// `vbsl(y ^ z, x, y)` says precisely that, in two ops against the
    /// default's four.
    #[inline(always)]
    fn maj(self, y: Self, z: Self) -> Self {
        unsafe { Neon(vbslq_u32(veorq_u32(y.0, z.0), self.0, y.0)) }
    }

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
                    w[g * 4 + j] = Neon(tj);
                }
            }
        }
        w
    }

    #[inline(always)]
    fn splat(v: u32) -> Self {
        unsafe { Neon(vdupq_n_u32(v)) }
    }

    #[inline(always)]
    fn load(v: &[u32]) -> Self {
        debug_assert_eq!(v.len(), 4);
        unsafe { Neon(vld1q_u32(v.as_ptr())) }
    }

    #[inline(always)]
    fn store(self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 4);
        unsafe { vst1q_u32(out.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { Neon(vaddq_u32(self.0, o.0)) }
    }

    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        unsafe { Neon(veorq_u32(self.0, o.0)) }
    }

    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { Neon(vandq_u32(self.0, o.0)) }
    }

    #[inline(always)]
    fn not_and(self, o: Self) -> Self {
        // vbic is `a & !b`, so the operands swap to give `!self & o`.
        unsafe { Neon(vbicq_u32(o.0, self.0)) }
    }

    #[inline(always)]
    fn shr<const B: u32>(self) -> Self {
        unsafe {
            let x = self.0;
            // SHA-256 only ever shifts by 3 and 10.
            let r = match B {
                3 => vshrq_n_u32::<3>(x),
                10 => vshrq_n_u32::<10>(x),
                _ => unreachable!(),
            };
            Neon(r)
        }
    }

    #[inline(always)]
    fn rotr<const B: u32>(self) -> Self {
        unsafe {
            let x = self.0;
            // The ten rotations SHA-256 uses.
            let r = match B {
                2 => rot::<2, 30>(x),
                6 => rot::<6, 26>(x),
                7 => rot::<7, 25>(x),
                11 => rot::<11, 21>(x),
                13 => rot::<13, 19>(x),
                17 => rot::<17, 15>(x),
                18 => rot::<18, 14>(x),
                19 => rot::<19, 13>(x),
                22 => rot::<22, 10>(x),
                25 => rot::<25, 7>(x),
                _ => unreachable!(),
            };
            Neon(r)
        }
    }
}
