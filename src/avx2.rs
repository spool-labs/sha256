//! 8-lane backend on x86-64 AVX2

use {
    crate::lanes::Lanes,
    std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_loadu_si256,
        _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi32, _mm256_setr_epi8,
        _mm256_shuffle_epi8, _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_si256,
        _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
        _mm256_xor_si256,
    },
};

#[derive(Clone, Copy)]
pub struct Avx2(pub(crate) __m256i);

/// Rotates right by `R`, given its precomputed complement `L = 32 - R`
#[inline(always)]
unsafe fn rot<const R: i32, const L: i32>(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32::<R>(x), _mm256_slli_epi32::<L>(x))
}

/// Byte-swaps each 32-bit word, big-endian message bytes to native
#[inline(always)]
pub(crate) unsafe fn bswap32(x: __m256i) -> __m256i {
    #[rustfmt::skip]
    let mask = _mm256_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    );
    _mm256_shuffle_epi8(x, mask)
}

/// Transposes an 8x8 matrix of 32-bit words, one row per register
#[inline(always)]
pub(crate) unsafe fn transpose8(r: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);

    let s0 = _mm256_unpacklo_epi64(t0, t2);
    let s1 = _mm256_unpackhi_epi64(t0, t2);
    let s2 = _mm256_unpacklo_epi64(t1, t3);
    let s3 = _mm256_unpackhi_epi64(t1, t3);
    let s4 = _mm256_unpacklo_epi64(t4, t6);
    let s5 = _mm256_unpackhi_epi64(t4, t6);
    let s6 = _mm256_unpacklo_epi64(t5, t7);
    let s7 = _mm256_unpackhi_epi64(t5, t7);

    [
        _mm256_permute2x128_si256::<0x20>(s0, s4),
        _mm256_permute2x128_si256::<0x20>(s1, s5),
        _mm256_permute2x128_si256::<0x20>(s2, s6),
        _mm256_permute2x128_si256::<0x20>(s3, s7),
        _mm256_permute2x128_si256::<0x31>(s0, s4),
        _mm256_permute2x128_si256::<0x31>(s1, s5),
        _mm256_permute2x128_si256::<0x31>(s2, s6),
        _mm256_permute2x128_si256::<0x31>(s3, s7),
    ]
}

/// The one concrete instantiation of the flat kernel for this backend.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn group(msgs: &[crate::batch::Message<'_>], out: &mut [[u8; 32]]) {
    crate::batch::hash_lanes::<Avx2, { <Avx2 as Lanes>::N }>(msgs, out)
}

/// # Safety
///
/// The running CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn steps(h: &mut [[u32; 8]], n: u64) {
    crate::chain::steps_lanes::<Avx2, { <Avx2 as Lanes>::N }>(h, n)
}

impl Lanes for Avx2 {
    const N: usize = 8;

    #[inline(always)]
    unsafe fn transpose(ptrs: &[*const u8], n: usize) -> [Self; 16] {
        debug_assert!((1..=8).contains(&n));
        let mut w = [Self::splat(0); 16];
        unsafe {
            // Inactive lanes reuse lane 0 so every lane holds valid data.
            let mut b = [ptrs[0]; 8];
            b[..n].copy_from_slice(&ptrs[..n]);
            // 16 words per block, eight lanes at a time.
            for g in 0..2 {
                let off = g * 32;
                let mut rows = [_mm256_set1_epi32(0); 8];
                for (lane, row) in rows.iter_mut().enumerate() {
                    let raw = _mm256_loadu_si256(b[lane].add(off) as *const __m256i);
                    *row = bswap32(raw);
                }
                for (j, tj) in transpose8(rows).into_iter().enumerate() {
                    w[g * 8 + j] = Avx2(tj);
                }
            }
        }
        w
    }

    #[inline(always)]
    fn splat(v: u32) -> Self {
        unsafe { Avx2(_mm256_set1_epi32(v as i32)) }
    }

    #[inline(always)]
    fn load(v: &[u32]) -> Self {
        debug_assert_eq!(v.len(), 8);
        unsafe { Avx2(_mm256_loadu_si256(v.as_ptr() as *const __m256i)) }
    }

    #[inline(always)]
    fn store(self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 8);
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) }
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { Avx2(_mm256_add_epi32(self.0, o.0)) }
    }

    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        unsafe { Avx2(_mm256_xor_si256(self.0, o.0)) }
    }

    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { Avx2(_mm256_and_si256(self.0, o.0)) }
    }

    #[inline(always)]
    fn not_and(self, o: Self) -> Self {
        // vpandn is exactly `(!a) & b`.
        unsafe { Avx2(_mm256_andnot_si256(self.0, o.0)) }
    }

    #[inline(always)]
    fn shr<const B: u32>(self) -> Self {
        unsafe {
            let x = self.0;
            // SHA-256 only ever shifts by 3 and 10.
            let r = match B {
                3 => _mm256_srli_epi32::<3>(x),
                10 => _mm256_srli_epi32::<10>(x),
                _ => unreachable!(),
            };
            Avx2(r)
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
            Avx2(r)
        }
    }
}
