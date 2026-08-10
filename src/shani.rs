//! Multi-stream backend on the x86 SHA-NI extension

use {
    crate::{
        batch::{stage_prefix_block, Message, Shape, BLOCK},
        core::{H0, K},
    },
    std::arch::x86_64::*,
};

/// Number of independent messages kept in flight
pub const STREAMS: usize = 4;

/// Shuffle pattern turning big-endian message bytes into native words
const BSWAP: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];

/// SHA-NI keeps the working state as ABEF and CDGH rather than ABCD and EFGH
#[derive(Clone, Copy)]
pub(crate) struct State {
    abef: __m128i,
    cdgh: __m128i,
}

impl State {
    /// The inverse of `words`: eight in-order state words into ABEF/CDGH
    #[inline(always)]
    pub(crate) unsafe fn pack(abcd: __m128i, efgh: __m128i) -> Self {
        let t = _mm_shuffle_epi32(abcd, 0xB1);
        let e = _mm_shuffle_epi32(efgh, 0x1B);
        State {
            abef: _mm_alignr_epi8(t, e, 8),
            cdgh: _mm_blend_epi16(e, t, 0xF0),
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn init() -> Self {
        Self::pack(
            _mm_loadu_si128(H0.as_ptr() as *const __m128i),
            _mm_loadu_si128(H0.as_ptr().add(4) as *const __m128i),
        )
    }

    /// Undoes the ABEF/CDGH interleave, giving the eight state words in order
    ///
    /// Split from `digest` because the chain needs the words themselves: they
    /// are already the next block's, and byte-swapping out and back would
    /// cancel.
    #[inline(always)]
    pub(crate) unsafe fn words(self) -> (__m128i, __m128i) {
        let t = _mm_shuffle_epi32(self.abef, 0x1B);
        let s1 = _mm_shuffle_epi32(self.cdgh, 0xB1);
        (_mm_blend_epi16(t, s1, 0xF0), _mm_alignr_epi8(s1, t, 8))
    }

    /// Emits the digest big-endian
    #[inline(always)]
    pub(crate) unsafe fn digest(self) -> [u8; 32] {
        let (abcd, efgh) = self.words();
        let mask = _mm_loadu_si128(BSWAP.as_ptr() as *const __m128i);
        let mut out = [0u8; 32];
        _mm_storeu_si128(
            out.as_mut_ptr() as *mut __m128i,
            _mm_shuffle_epi8(abcd, mask),
        );
        _mm_storeu_si128(
            out.as_mut_ptr().add(16) as *mut __m128i,
            _mm_shuffle_epi8(efgh, mask),
        );
        out
    }
}

/// Loads a 64-byte block as four big-endian-corrected message vectors
#[inline(always)]
pub(crate) unsafe fn load_msg(block: &[u8]) -> [__m128i; 4] {
    debug_assert!(block.len() >= BLOCK);
    let mask = _mm_loadu_si128(BSWAP.as_ptr() as *const __m128i);
    let mut m = [_mm_setzero_si128(); 4];
    for (i, mi) in m.iter_mut().enumerate() {
        let raw = _mm_loadu_si128(block.as_ptr().add(i * 16) as *const __m128i);
        *mi = _mm_shuffle_epi8(raw, mask);
    }
    m
}

/// One group of four rounds for a single stream
///
/// Split from the interlaced loop so both the serial and interlaced forms
/// use the same schedule recurrence and cannot drift.
#[inline(always)]
unsafe fn round4(st: &mut State, m: &mut [__m128i; 4], i: usize) {
    let k = _mm_loadu_si128(K.as_ptr().add(i * 4) as *const __m128i);
    let wk = _mm_add_epi32(m[i & 3], k);
    st.cdgh = _mm_sha256rnds2_epu32(st.cdgh, st.abef, wk);
    if (3..=14).contains(&i) {
        let t = _mm_alignr_epi8(m[i & 3], m[(i + 3) & 3], 4);
        m[(i + 1) & 3] = _mm_sha256msg2_epu32(_mm_add_epi32(m[(i + 1) & 3], t), m[i & 3]);
    }
    let wk_hi = _mm_shuffle_epi32(wk, 0x0E);
    st.abef = _mm_sha256rnds2_epu32(st.abef, st.cdgh, wk_hi);
    // msg1 supplies the `W[i-16] + sigma0(W[i-15])` half of each schedule
    // word. It must run through group 12: that is where the slot consumed by
    // the last msg2 (group 14) gets its half, and dropping it silently
    // corrupts only W[60..64].
    if (1..=12).contains(&i) {
        m[(i + 3) & 3] = _mm_sha256msg1_epu32(m[(i + 3) & 3], m[i & 3]);
    }
}

/// Compresses one block for a single stream
#[inline(always)]
unsafe fn compress_block(st: &mut State, m: &mut [__m128i; 4]) {
    let save = *st;
    // Literal group indices for the reason spelled out on `compress_interleaved`:
    // rolled, the msg1/msg2 range tests survive as per-iteration branches.
    macro_rules! groups {
        ($($i:literal)*) => { $( round4(st, m, $i); )* };
    }
    groups!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
    st.abef = _mm_add_epi32(st.abef, save.abef);
    st.cdgh = _mm_add_epi32(st.cdgh, save.cdgh);
}

/// Compresses one block of every stream, interlaced at round granularity.
#[inline(always)]
pub(crate) unsafe fn compress_interleaved(
    st: &mut [State; STREAMS],
    msg: &mut [[__m128i; 4]; STREAMS],
) {
    let save = *st;

    // Literal group indices, not `for i in 0..16`: LLVM keeps the rolled
    // loop rolled, which turns the msg1/msg2 range tests into per-iteration
    // branches and pushes the schedule state into stack slots behind
    // computed addresses.
    macro_rules! groups {
        ($($i:literal)*) => { $(
            for (s, m) in st.iter_mut().zip(msg.iter_mut()) {
                round4(s, m, $i);
            }
        )* };
    }
    groups!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);

    for (s, o) in st.iter_mut().zip(save.iter()) {
        s.abef = _mm_add_epi32(s.abef, o.abef);
        s.cdgh = _mm_add_epi32(s.cdgh, o.cdgh);
    }
}

/// Hashes one message on its own. Used for ragged tails, where the
/// interlace has nothing left to weave in.
#[inline(always)]
unsafe fn hash_one(m: &Message<'_>, out: &mut [u8; 32]) {
    let mut st = State::init();
    let mut staging = [0u8; BLOCK];
    for k in 0..m.blocks() {
        let mut msg = if m.block_is_interior(k) {
            load_msg(m.interior_block(k))
        } else {
            m.fill_block(k, &mut staging);
            load_msg(&staging)
        };
        compress_block(&mut st, &mut msg);
    }
    *out = st.digest();
}

/// Hashes exactly STREAMS equal-length messages with their chains interlaced
#[inline(always)]
unsafe fn hash_uniform_group(msgs: &[Message<'_>], out: &mut [[u8; 32]], blocks: usize) {
    debug_assert_eq!(msgs.len(), STREAMS);

    let mut st = [State::init(); STREAMS];
    let mut staging = [[0u8; BLOCK]; STREAMS];
    let mut msg = [[_mm_setzero_si128(); 4]; STREAMS];

    let shape = Shape::of(msgs);
    let mut bases = [std::ptr::null::<u8>(); STREAMS];
    for (b, m) in bases.iter_mut().zip(msgs) {
        *b = m.body.as_ptr();
    }

    let staged0 = stage_prefix_block(msgs, &shape, &mut staging);
    for k in 0..blocks {
        if shape.same && k >= shape.k_lo && k < shape.k_hi {
            let off = k * BLOCK - shape.plen;
            for (slot, base) in msg.iter_mut().zip(bases.iter()) {
                // SAFETY: the interior bound documented on `Shape`; bodies
                // stay borrowed for the whole call.
                *slot = load_msg(std::slice::from_raw_parts(base.add(off), BLOCK));
            }
        } else if k == 0 && staged0 {
            for (slot, s) in msg.iter_mut().zip(staging.iter()) {
                *slot = load_msg(s);
            }
        } else {
            let mut interior = [false; STREAMS];
            for (lane, m) in msgs.iter().enumerate() {
                interior[lane] = m.block_is_interior(k);
                if !interior[lane] {
                    m.fill_block(k, &mut staging[lane]);
                }
            }
            for (lane, m) in msgs.iter().enumerate() {
                msg[lane] = if interior[lane] {
                    load_msg(m.interior_block(k))
                } else {
                    load_msg(&staging[lane])
                };
            }
        }
        compress_interleaved(&mut st, &mut msg);
    }

    for (o, s) in out.iter_mut().zip(st.iter()) {
        *o = s.digest();
    }
}

/// Hashes up to STREAMS messages: the unit of work the shared drivers dispatch
/// through.
///
/// # Safety
///
/// The running CPU must support SHA-NI, SSSE3, and SSE4.1.
#[target_feature(enable = "sha,ssse3,sse4.1")]
pub(crate) unsafe fn group(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
    debug_assert!(msgs.len() <= STREAMS);
    debug_assert_eq!(msgs.len(), out.len());
    let blocks = match msgs.first() {
        Some(m) => m.blocks(),
        None => return,
    };
    if msgs.len() == STREAMS && msgs.iter().all(|m| m.blocks() == blocks) {
        hash_uniform_group(msgs, out, blocks);
    } else {
        // Ragged group: interlacing a mixed-length group would mean masking
        // off finished streams every block, and the Merkle workload is uniform.
        for (m, oi) in msgs.iter().zip(out.iter_mut()) {
            hash_one(m, oi);
        }
    }
}

/// one compression per link, and nothing else
///
/// The state words are the next block's first eight words already, so the pair
/// of byte swaps that `digest` and `load_msg` would do is dropped entirely and
/// only the ABEF/CDGH unpack survives. Words 8..16 are loop-invariant.
///
/// # Safety
///
/// The running CPU must support SHA-NI, SSSE3, and SSE4.1.
#[target_feature(enable = "sha,ssse3,sse4.1")]
pub(crate) unsafe fn chain(seed: &[u8; 32], n: u64) -> [u8; 32] {
    let mask = _mm_loadu_si128(BSWAP.as_ptr() as *const __m128i);
    let mut abcd = _mm_shuffle_epi8(_mm_loadu_si128(seed.as_ptr() as *const __m128i), mask);
    let mut efgh = _mm_shuffle_epi8(
        _mm_loadu_si128(seed.as_ptr().add(16) as *const __m128i),
        mask,
    );
    let pad0 = _mm_loadu_si128(crate::chain::PAD.as_ptr() as *const __m128i);
    let pad1 = _mm_loadu_si128(crate::chain::PAD.as_ptr().add(4) as *const __m128i);

    let init = State::init();
    for _ in 0..n {
        let mut st = init;
        let mut m = [abcd, efgh, pad0, pad1];
        compress_block(&mut st, &mut m);
        (abcd, efgh) = st.words();
    }

    let mut out = [0u8; 32];
    _mm_storeu_si128(
        out.as_mut_ptr() as *mut __m128i,
        _mm_shuffle_epi8(abcd, mask),
    );
    _mm_storeu_si128(
        out.as_mut_ptr().add(16) as *mut __m128i,
        _mm_shuffle_epi8(efgh, mask),
    );
    out
}

/// Advances `STREAMS` interlaced chains `n` links: this unit's step kernel
///
/// The serial chain leaves the SHA unit's issue slots mostly idle behind one
/// dependency chain; independent chains fill them, which is what replay has
/// and generation does not. The scheduler owns raggedness, so this loop is
/// compressions and nothing else, with the byte swaps paid there once per
/// chain; only the ABEF/CDGH pack and unpack remain per link.
///
/// # Safety
///
/// The running CPU must support SHA-NI, SSSE3, and SSE4.1.
#[target_feature(enable = "sha,ssse3,sse4.1")]
pub(crate) unsafe fn steps4(h: &mut [[u32; 8]], n: u64) {
    debug_assert_eq!(h.len(), STREAMS);

    let pad0 = _mm_loadu_si128(crate::chain::PAD.as_ptr() as *const __m128i);
    let pad1 = _mm_loadu_si128(crate::chain::PAD.as_ptr().add(4) as *const __m128i);

    let mut abcd = [_mm_setzero_si128(); STREAMS];
    let mut efgh = [_mm_setzero_si128(); STREAMS];
    for lane in 0..STREAMS {
        abcd[lane] = _mm_loadu_si128(h[lane].as_ptr() as *const __m128i);
        efgh[lane] = _mm_loadu_si128(h[lane].as_ptr().add(4) as *const __m128i);
    }

    let init = [State::init(); STREAMS];
    for _ in 0..n {
        let mut st = init;
        let mut msg = [[_mm_setzero_si128(); 4]; STREAMS];
        for (m, (a, e)) in msg.iter_mut().zip(abcd.iter().zip(efgh.iter())) {
            *m = [*a, *e, pad0, pad1];
        }
        compress_interleaved(&mut st, &mut msg);
        for (lane, s) in st.iter().enumerate() {
            let (a, e) = s.words();
            abcd[lane] = a;
            efgh[lane] = e;
        }
    }

    for lane in 0..STREAMS {
        _mm_storeu_si128(h[lane].as_mut_ptr() as *mut __m128i, abcd[lane]);
        _mm_storeu_si128(h[lane].as_mut_ptr().add(4) as *mut __m128i, efgh[lane]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::compress,
        lanes::{Lanes, Scalar},
    };

    /// Isolates one block compression against the portable core, so a failure
    /// points at the SHA-NI round sequence rather than the driver.
    #[test]
    fn shani_block_matches_portable() {
        if !std::arch::is_x86_feature_detected!("sha") {
            return;
        }
        unsafe { run() }
    }

    /// init -> digest with no rounds must return H0 big-endian. Isolates the
    /// ABEF/CDGH packing from the round sequence.
    #[test]
    fn shani_state_packing_roundtrips() {
        if !std::arch::is_x86_feature_detected!("sha") {
            return;
        }
        unsafe {
            let d = State::init().digest();
            let mut got = [0u32; 8];
            for (i, g) in got.iter_mut().enumerate() {
                *g = u32::from_be_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]]);
            }
            assert_eq!(
                got.map(|x| format!("{x:08x}")),
                H0.map(|x| format!("{x:08x}")),
                "init/digest packing does not round-trip"
            );
        }
    }

    /// Steps a textbook block compression alongside the SHA-NI one, comparing
    /// after every group of four rounds, so a mismatch names the exact group.
    #[test]
    fn shani_groups_match_reference() {
        if !std::arch::is_x86_feature_detected!("sha") {
            return;
        }
        unsafe { step() }
    }

    #[target_feature(enable = "sha,ssse3,sse4.1")]
    unsafe fn step() {
        let mut block = [0u8; 64];
        for (i, b) in block.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(1);
        }

        // textbook reference
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = H0;

        let mut st = State::init();
        let mut m = load_msg(&block);

        for g in 0..16 {
            // four reference rounds
            for r in 4 * g..4 * g + 4 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[r])
                    .wrapping_add(w[r]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let mj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(mj);
                v = [
                    t1.wrapping_add(t2),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }

            round4(&mut st, &mut m, g);

            let mut abef = [0u32; 4];
            let mut cdgh = [0u32; 4];
            _mm_storeu_si128(abef.as_mut_ptr() as *mut __m128i, st.abef);
            _mm_storeu_si128(cdgh.as_mut_ptr() as *mut __m128i, st.cdgh);
            // lanes are low-to-high: abef = [F, E, B, A], cdgh = [H, G, D, C]
            let got = [
                abef[3], abef[2], cdgh[3], cdgh[2], abef[1], abef[0], cdgh[1], cdgh[0],
            ];
            assert_eq!(
                got.map(|x| format!("{x:08x}")),
                v.map(|x| format!("{x:08x}")),
                "diverged at group {g} (rounds {}..{})",
                4 * g,
                4 * g + 4
            );
        }
    }

    #[target_feature(enable = "sha,ssse3,sse4.1")]
    unsafe fn run() {
        let mut block = [0u8; 64];
        for (i, b) in block.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(1);
        }

        let mut st = H0.map(Scalar::<1>::splat);
        let srcs: [*const u8; 1] = [block.as_ptr()];
        let w = <Scalar<1> as Lanes>::transpose(&srcs, 1);
        compress::<Scalar<1>>(&mut st, w);
        let mut want = [0u32; 8];
        for (i, s) in st.iter().enumerate() {
            let mut o = [0u32; 1];
            s.store(&mut o);
            want[i] = o[0];
        }

        let mut s2 = State::init();
        let mut m = load_msg(&block);
        compress_block(&mut s2, &mut m);
        let d = s2.digest();
        let mut got = [0u32; 8];
        for (i, g) in got.iter_mut().enumerate() {
            *g = u32::from_be_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]]);
        }

        assert_eq!(
            got.map(|x| format!("{x:08x}")),
            want.map(|x| format!("{x:08x}")),
            "shani compress_block != portable"
        );
    }
}
