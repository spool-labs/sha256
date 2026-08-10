//! Interlace: two 16-lane AVX-512 waves woven into one thread
//!
//! A single 16-lane compression is one dependency chain: at ~2 IPC the
//! vector ports sit a third idle waiting on it, which is exactly the slack
//! an SMT sibling harvests for 1.5x. This kernel claims that slack from
//! within the thread instead: two independent waves with their rounds
//! interlaced, so wave B issues while wave A's chain stalls.
//!
//! The 48 live vectors against 32 ZMM guarantee spills, and whether the
//! latency hidden outweighs the spill traffic is a per-microarchitecture
//! verdict, not a principle.

use crate::{
    avx512::Avx512,
    batch::{stage_prefix_block, write_digest, Message, Shape, BLOCK, MAX_LANES},
    core::{big_sigma0, big_sigma1, small_sigma0, small_sigma1, H0, K},
    lanes::Lanes,
};

const WAVE: usize = 16;
pub(crate) const WIDTH: usize = 32;

// The drivers stage WIDTH messages at a time; a wider kernel would silently
// overrun their staging arrays, so fail the build instead.
const _: () = assert!(WIDTH <= crate::batch::MAX_WIDTH);

/// Compresses one block of each wave, rounds alternating A/B
///
/// Same structure as `core::compress`, twice over: the round and schedule
/// macros are restated because they capture the local working variables,
/// but the sigma arithmetic is shared with the single-wave core. The A/B
/// alternation is textual so the scheduler sees both chains at once. Each
/// `rounds16x2` expansion is flat for the same reason `core`'s `rounds16` is:
/// literal indices keep both schedule windows in registers. Its four
/// invocations are not — see the loop at the end of this function.
#[inline(always)]
pub(crate) fn compress2<L: Lanes>(
    sa: &mut [L; 8],
    sb: &mut [L; 8],
    mut wa: [L; 16],
    mut wb: [L; 16],
) {
    let [mut a0, mut b0, mut c0, mut d0, mut e0, mut f0, mut g0, mut h0] = *sa;
    let [mut a1, mut b1, mut c1, mut d1, mut e1, mut f1, mut g1, mut h1] = *sb;

    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident, $w:expr, $k:expr) => {{
            let t1 = $w
                .add(L::splat($k))
                .add($h.add(big_sigma1($e)).add($e.ch($f, $g)));
            let t2 = big_sigma0($a).add($a.maj($b, $c));
            $d = $d.add(t1);
            $h = t1.add(t2);
        }};
    }
    // `direct` reads a block word (rounds 0..16); `extend` advances the
    // named wave's window in place (rounds 16..64).
    macro_rules! direct {
        ($w:ident, $i:expr) => {
            $w[$i]
        };
    }
    macro_rules! extend {
        ($w:ident, $i:expr) => {{
            const I: usize = $i;
            let s1 = small_sigma1($w[(I + 14) & 15]);
            let s0 = small_sigma0($w[(I + 1) & 15]);
            $w[I] = s1.add($w[(I + 9) & 15]).add(s0).add($w[I]);
            $w[I]
        }};
    }
    // One round per line on purpose; rustfmt would fold the listing into a
    // vertical argument stack, which buries the A/B alternation the layout
    // exists to show.
    #[rustfmt::skip]
    macro_rules! rounds16x2 {
        ($k:expr, $sched:ident) => {
            round!(a0,b0,c0,d0,e0,f0,g0,h0, $sched!(wa, 0), $k[0]);
            round!(a1,b1,c1,d1,e1,f1,g1,h1, $sched!(wb, 0), $k[0]);
            round!(h0,a0,b0,c0,d0,e0,f0,g0, $sched!(wa, 1), $k[1]);
            round!(h1,a1,b1,c1,d1,e1,f1,g1, $sched!(wb, 1), $k[1]);
            round!(g0,h0,a0,b0,c0,d0,e0,f0, $sched!(wa, 2), $k[2]);
            round!(g1,h1,a1,b1,c1,d1,e1,f1, $sched!(wb, 2), $k[2]);
            round!(f0,g0,h0,a0,b0,c0,d0,e0, $sched!(wa, 3), $k[3]);
            round!(f1,g1,h1,a1,b1,c1,d1,e1, $sched!(wb, 3), $k[3]);
            round!(e0,f0,g0,h0,a0,b0,c0,d0, $sched!(wa, 4), $k[4]);
            round!(e1,f1,g1,h1,a1,b1,c1,d1, $sched!(wb, 4), $k[4]);
            round!(d0,e0,f0,g0,h0,a0,b0,c0, $sched!(wa, 5), $k[5]);
            round!(d1,e1,f1,g1,h1,a1,b1,c1, $sched!(wb, 5), $k[5]);
            round!(c0,d0,e0,f0,g0,h0,a0,b0, $sched!(wa, 6), $k[6]);
            round!(c1,d1,e1,f1,g1,h1,a1,b1, $sched!(wb, 6), $k[6]);
            round!(b0,c0,d0,e0,f0,g0,h0,a0, $sched!(wa, 7), $k[7]);
            round!(b1,c1,d1,e1,f1,g1,h1,a1, $sched!(wb, 7), $k[7]);
            round!(a0,b0,c0,d0,e0,f0,g0,h0, $sched!(wa, 8), $k[8]);
            round!(a1,b1,c1,d1,e1,f1,g1,h1, $sched!(wb, 8), $k[8]);
            round!(h0,a0,b0,c0,d0,e0,f0,g0, $sched!(wa, 9), $k[9]);
            round!(h1,a1,b1,c1,d1,e1,f1,g1, $sched!(wb, 9), $k[9]);
            round!(g0,h0,a0,b0,c0,d0,e0,f0, $sched!(wa, 10), $k[10]);
            round!(g1,h1,a1,b1,c1,d1,e1,f1, $sched!(wb, 10), $k[10]);
            round!(f0,g0,h0,a0,b0,c0,d0,e0, $sched!(wa, 11), $k[11]);
            round!(f1,g1,h1,a1,b1,c1,d1,e1, $sched!(wb, 11), $k[11]);
            round!(e0,f0,g0,h0,a0,b0,c0,d0, $sched!(wa, 12), $k[12]);
            round!(e1,f1,g1,h1,a1,b1,c1,d1, $sched!(wb, 12), $k[12]);
            round!(d0,e0,f0,g0,h0,a0,b0,c0, $sched!(wa, 13), $k[13]);
            round!(d1,e1,f1,g1,h1,a1,b1,c1, $sched!(wb, 13), $k[13]);
            round!(c0,d0,e0,f0,g0,h0,a0,b0, $sched!(wa, 14), $k[14]);
            round!(c1,d1,e1,f1,g1,h1,a1,b1, $sched!(wb, 14), $k[14]);
            round!(b0,c0,d0,e0,f0,g0,h0,a0, $sched!(wa, 15), $k[15]);
            round!(b1,c1,d1,e1,f1,g1,h1,a1, $sched!(wb, 15), $k[15]);
        };
    }

    // Rounds 16..64 roll instead of unrolling: 16 rounds is two full
    // rotations of a0..h0, so every pass re-enters with the same naming.
    // Fully unrolled the body is ~30 KB against a 32 KB L1I, close enough
    // to the edge that unrelated code shifting its alignment costs 60%.
    let (kc, _) = K.as_chunks::<16>();
    rounds16x2!(kc[0], direct);
    for k in &kc[1..] {
        rounds16x2!(k, extend);
    }

    sa[0] = sa[0].add(a0);
    sa[1] = sa[1].add(b0);
    sa[2] = sa[2].add(c0);
    sa[3] = sa[3].add(d0);
    sa[4] = sa[4].add(e0);
    sa[5] = sa[5].add(f0);
    sa[6] = sa[6].add(g0);
    sa[7] = sa[7].add(h0);
    sb[0] = sb[0].add(a1);
    sb[1] = sb[1].add(b1);
    sb[2] = sb[2].add(c1);
    sb[3] = sb[3].add(d1);
    sb[4] = sb[4].add(e1);
    sb[5] = sb[5].add(f1);
    sb[6] = sb[6].add(g1);
    sb[7] = sb[7].add(h1);
}

#[inline(always)]
fn write_digests<L: Lanes>(state: &[L; 8], out: &mut [[u8; 32]]) {
    let mut unpacked = [[0u32; MAX_LANES]; 8];
    for (i, s) in state.iter().enumerate() {
        s.store(&mut unpacked[i][..L::N]);
    }
    for (lane, o) in out.iter_mut().enumerate() {
        write_digest(&unpacked, lane, o);
    }
}

/// Hashes exactly 32 equal-block-count messages, 16 per wave
#[inline(always)]
unsafe fn fused(msgs: &[Message<'_>], out: &mut [[u8; 32]], blocks: usize) {
    debug_assert_eq!(msgs.len(), WIDTH);

    let mut sa = H0.map(Avx512::splat);
    let mut sb = H0.map(Avx512::splat);

    let shape = Shape::of(msgs);
    let mut bases = [std::ptr::null::<u8>(); WIDTH];
    for (b, m) in bases.iter_mut().zip(msgs) {
        *b = m.body.as_ptr();
    }

    let mut staging = [[0u8; BLOCK]; WIDTH];
    let mut srcs = [std::ptr::null::<u8>(); WIDTH];

    let staged0 = stage_prefix_block(msgs, &shape, &mut staging);
    for k in 0..blocks {
        if shape.same && k >= shape.k_lo && k < shape.k_hi {
            let off = k * BLOCK - shape.plen;
            for (s, b) in srcs.iter_mut().zip(bases.iter()) {
                // SAFETY: the interior bound documented on `Shape`.
                *s = b.add(off);
            }
        } else if k == 0 && staged0 {
            for (s, b) in srcs.iter_mut().zip(staging.iter()) {
                *s = b.as_ptr();
            }
        } else {
            for (lane, m) in msgs.iter().enumerate() {
                if m.block_is_interior(k) {
                    srcs[lane] = m.interior_block(k).as_ptr();
                } else {
                    m.fill_block(k, &mut staging[lane]);
                    srcs[lane] = staging[lane].as_ptr();
                }
            }
        }
        // SAFETY: every source points at a full 64-byte block, either
        // interior to a borrowed body or freshly staged above.
        let wa = Avx512::transpose(&srcs[..WAVE], WAVE);
        let wb = Avx512::transpose(&srcs[WAVE..], WAVE);
        compress2(&mut sa, &mut sb, wa, wb);
    }

    write_digests(&sa, &mut out[..WAVE]);
    write_digests(&sb, &mut out[WAVE..]);
}

/// Hashes up to 32 messages: the full interlace when full and uniform,
/// else split across the single-wave kernel
///
/// # Safety
///
/// The running CPU must support AVX-512F and AVX-512BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn group2(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
    debug_assert!(msgs.len() <= WIDTH);
    debug_assert_eq!(msgs.len(), out.len());
    let blocks = match msgs.first() {
        Some(m) => m.blocks(),
        None => return,
    };
    if msgs.len() == WIDTH && msgs.iter().all(|m| m.blocks() == blocks) {
        fused(msgs, out, blocks);
    } else {
        // Ragged or partial: 16-lane groups through the shared driver; only
        // the cross-wave overlap is lost.
        crate::batch::drive(WAVE, crate::avx512::group, msgs, out);
    }
}

/// Advances 32 chains as two interlaced 16-lane waves
///
/// # Safety
///
/// The running CPU must support AVX-512F and AVX-512BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn steps2(h: &mut [[u32; 8]], n: u64) {
    crate::chain::steps_lanes2::<Avx512, WAVE>(h, n)
}
