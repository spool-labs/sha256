//! 16-lane backend on x86-64 AVX-512

use {
    crate::lanes::Lanes,
    std::arch::x86_64::{
        __m512i, _mm512_add_epi32, _mm512_and_si512, _mm512_andnot_si512, _mm512_loadu_si512,
        _mm512_ror_epi32, _mm512_set1_epi32, _mm512_shuffle_epi8, _mm512_shuffle_i32x4,
        _mm512_srli_epi32, _mm512_storeu_si512, _mm512_ternarylogic_epi32, _mm512_unpackhi_epi32,
        _mm512_unpackhi_epi64, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64, _mm512_xor_si512,
    },
};

#[derive(Clone, Copy)]
pub struct Avx512(__m512i);

/// Byte-reversal pattern for `vpshufb`.
#[rustfmt::skip]
static BSWAP32: [u8; 64] = [
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
];

/// Stages 2..4 of the 16x16 transpose ladder.
#[inline(always)]
unsafe fn transpose_rest(t: [__m512i; 16]) -> [__m512i; 16] {
    let mut s = [_mm512_set1_epi32(0); 16];
    for g in 0..4 {
        let b = g * 4;
        s[b] = _mm512_unpacklo_epi64(t[b], t[b + 2]);
        s[b + 1] = _mm512_unpackhi_epi64(t[b], t[b + 2]);
        s[b + 2] = _mm512_unpacklo_epi64(t[b + 1], t[b + 3]);
        s[b + 3] = _mm512_unpackhi_epi64(t[b + 1], t[b + 3]);
    }
    let mut u = [_mm512_set1_epi32(0); 16];
    for g in 0..2 {
        let b = g * 8;
        for i in 0..4 {
            u[b + i] = _mm512_shuffle_i32x4::<0x88>(s[b + i], s[b + i + 4]);
            u[b + i + 4] = _mm512_shuffle_i32x4::<0xdd>(s[b + i], s[b + i + 4]);
        }
    }
    let mut o = [_mm512_set1_epi32(0); 16];
    for i in 0..8 {
        o[i] = _mm512_shuffle_i32x4::<0x88>(u[i], u[i + 8]);
        o[i + 8] = _mm512_shuffle_i32x4::<0xdd>(u[i], u[i + 8]);
    }
    o
}

/// The one concrete instantiation of the flat kernel for this backend
///
/// Entry points reach it through `GroupFn` so the kernel is compiled exactly
/// once.
///
/// # Safety
///
/// The running CPU must support AVX-512F and AVX-512BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn group(msgs: &[crate::batch::Message<'_>], out: &mut [[u8; 32]]) {
    crate::batch::hash_lanes::<Avx512, { <Avx512 as Lanes>::N }>(msgs, out)
}

/// # Safety
///
/// The running CPU must support AVX-512F and AVX-512BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn steps(h: &mut [[u32; 8]], n: u64) {
    crate::chain::steps_lanes::<Avx512, { <Avx512 as Lanes>::N }>(h, n)
}

impl Lanes for Avx512 {
    const N: usize = 16;

    // `vpternlogd` does any three-input boolean from a truth-table byte in one
    // instruction, so the six sigma XORs plus Ch and Maj each collapse to a
    // single op. Spelled out rather than left to the optimiser: LLVM does not
    // reliably re-fuse the generic sequences back into ternlog.
    //
    // Truth table is indexed with src1 as the high bit, index = a*4 + b*2 + c,
    // result bit `imm8 >> index & 1`. All eight combinations are checked in
    // tests/ternlog_model.rs:
    //   0x96 = a ^ b ^ c
    //   0xCA = (a & b) ^ (!a & c)  -- Ch, with a=x, b=y, c=z
    //   0xE8 = majority(a, b, c)   -- Maj

    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        unsafe { Self(_mm512_ternarylogic_epi32::<0x96>(self.0, b.0, c.0)) }
    }

    #[inline(always)]
    fn ch(self, y: Self, z: Self) -> Self {
        unsafe { Self(_mm512_ternarylogic_epi32::<0xCA>(self.0, y.0, z.0)) }
    }

    #[inline(always)]
    fn maj(self, y: Self, z: Self) -> Self {
        unsafe { Self(_mm512_ternarylogic_epi32::<0xE8>(self.0, y.0, z.0)) }
    }

    #[inline(always)]
    unsafe fn transpose(ptrs: &[*const u8], n: usize) -> [Self; 16] {
        debug_assert!((1..=16).contains(&n));
        unsafe {
            // Inactive lanes reuse lane 0 so every lane holds valid data.
            let mut src = [ptrs[0]; 16];
            src[..n].copy_from_slice(&ptrs[..n]);

            // Stage 1: byte swap per input, 32-bit unpack per output.
            let mask = _mm512_loadu_si512(BSWAP32.as_ptr() as *const _);
            let mut r = [_mm512_set1_epi32(0); 16];
            for (i, ri) in r.iter_mut().enumerate() {
                let raw = _mm512_loadu_si512(src[i] as *const _);
                *ri = _mm512_shuffle_epi8(raw, mask);
            }
            let mut t = [_mm512_set1_epi32(0); 16];
            for i in 0..8 {
                t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
                t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
            }
            let o = transpose_rest(t);
            o.map(Self)
        }
    }

    #[inline(always)]
    fn splat(v: u32) -> Self {
        unsafe { Self(_mm512_set1_epi32(v as i32)) }
    }

    #[inline(always)]
    fn load(v: &[u32]) -> Self {
        debug_assert_eq!(v.len(), 16);
        unsafe { Self(_mm512_loadu_si512(v.as_ptr() as *const _)) }
    }

    #[inline(always)]
    fn store(self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 16);
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut _, self.0) }
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { Self(_mm512_add_epi32(self.0, o.0)) }
    }

    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        unsafe { Self(_mm512_xor_si512(self.0, o.0)) }
    }

    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, o.0)) }
    }

    #[inline(always)]
    fn not_and(self, o: Self) -> Self {
        // vpandnd is exactly `(!a) & b`.
        unsafe { Self(_mm512_andnot_si512(self.0, o.0)) }
    }

    #[inline(always)]
    fn shr<const B: u32>(self) -> Self {
        unsafe {
            let x = self.0;
            // SHA-256 only ever shifts by 3 and 10.
            let r = match B {
                3 => _mm512_srli_epi32::<3>(x),
                10 => _mm512_srli_epi32::<10>(x),
                _ => unreachable!(),
            };
            Self(r)
        }
    }

    #[inline(always)]
    fn rotr<const B: u32>(self) -> Self {
        unsafe {
            let x = self.0;
            // vprord is a native rotate: one instruction, no shift/shift/or.
            let r = match B {
                2 => _mm512_ror_epi32::<2>(x),
                6 => _mm512_ror_epi32::<6>(x),
                7 => _mm512_ror_epi32::<7>(x),
                11 => _mm512_ror_epi32::<11>(x),
                13 => _mm512_ror_epi32::<13>(x),
                17 => _mm512_ror_epi32::<17>(x),
                18 => _mm512_ror_epi32::<18>(x),
                19 => _mm512_ror_epi32::<19>(x),
                22 => _mm512_ror_epi32::<22>(x),
                25 => _mm512_ror_epi32::<25>(x),
                _ => unreachable!(),
            };
            Self(r)
        }
    }
}
