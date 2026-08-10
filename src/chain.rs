//! Proof of history: SHA-256 fed back into itself

use crate::{
    core::{big_sigma0, big_sigma1, compress, small_sigma0, small_sigma1, H0, K},
    lanes::{Lanes, Scalar},
};

/// Words 8..16 of a padded 32-byte message: the terminator, then the bit length
///
/// Every link shares them, so backends hoist these out of the loop.
pub(crate) const PAD: [u32; 8] = [0x8000_0000, 0, 0, 0, 0, 0, 0, 256];

/// The widest chain kernel: two 16-lane AVX-512 waves
const MAX_WIDTH: usize = 32;

/// Advances every lane of `h` by `n` links; `h.len()` is the kernel's width
///
/// The same shape as `batch::GroupFn`: an unsafe pointer to a
/// `#[target_feature]` kernel, paired with its probe by the dispatcher.
pub(crate) type StepsFn = unsafe fn(&mut [[u32; 8]], u64);

/// A digest as the eight native words the next block wants
#[inline(always)]
fn seed_words(seed: &[u8; 32]) -> [u32; 8] {
    let mut h = [0u32; 8];
    for (hj, c) in h.iter_mut().zip(seed.chunks_exact(4)) {
        *hj = u32::from_be_bytes(c.try_into().unwrap());
    }
    h
}

/// The inverse: native state words back out as a big-endian digest
#[inline(always)]
fn word_bytes(h: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (c, hj) in out.chunks_exact_mut(4).zip(h) {
        c.copy_from_slice(&hj.to_be_bytes());
    }
    out
}

/// Portable chain, for targets with no SHA-256 unit
pub(crate) fn portable(seed: &[u8; 32], n: u64) -> [u8; 32] {
    let mut h = seed_words(seed);

    let mut w = [0u32; 16];
    w[8..].copy_from_slice(&PAD);
    for _ in 0..n {
        w[..8].copy_from_slice(&h);
        let mut st = H0.map(Scalar::<1>::splat);
        compress::<Scalar<1>>(&mut st, w.map(Scalar::<1>::splat));
        for (hi, s) in h.iter_mut().zip(st) {
            *hi = s.0[0];
        }
    }

    word_bytes(&h)
}

/// `small_sigma0` of a compile-time word, for folding the pad's schedule terms
const fn sig0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

/// `small_sigma1`, likewise
const fn sig1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// Round constants 8..16 with the pad words pre-added
///
/// Those rounds read only pad, so `w + k` is a compile-time constant.
const PADK: [u32; 8] = {
    let mut a = [0u32; 8];
    let mut i = 0;
    while i < 8 {
        a[i] = K[8 + i].wrapping_add(PAD[i]);
        i += 1;
    }
    a
};

/// One chain link per lane: the generic compression with the pad's schedule folded
///
/// W16..W31 is the band where the constant pad is still in reach of the
/// expansion recurrence: the six zero words drop their terms, sigmas of the
/// two live pad words fold to constants, and W25..W29 collapse to two terms
/// each. From W32 every input is live again. Flat-only because the band is
/// sixteen distinct formulas; `steps_lanes` routes rolled backends to the
/// generic compression instead.
#[inline(always)]
pub(crate) fn compress_chain<L: Lanes>(state: &mut [L; 8], w8: &[L; 8]) {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    let mut w = [L::splat(0); 16];
    w[..8].copy_from_slice(w8);

    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident, $wk:expr) => {{
            let t1 = $wk.add($h.add(big_sigma1($e)).add($e.ch($f, $g)));
            let t2 = big_sigma0($a).add($a.maj($b, $c));
            $d = $d.add(t1);
            $h = t1.add(t2);
        }};
    }
    // Eight rounds is one full rotation of the working variables, so each
    // block below re-enters with the same naming.
    macro_rules! rounds8 {
        ($sched:ident, $base:literal) => {
            round!(a, b, c, d, e, f, g, h, $sched!($base, 0));
            round!(h, a, b, c, d, e, f, g, $sched!($base, 1));
            round!(g, h, a, b, c, d, e, f, $sched!($base, 2));
            round!(f, g, h, a, b, c, d, e, $sched!($base, 3));
            round!(e, f, g, h, a, b, c, d, $sched!($base, 4));
            round!(d, e, f, g, h, a, b, c, $sched!($base, 5));
            round!(c, d, e, f, g, h, a, b, $sched!($base, 6));
            round!(b, c, d, e, f, g, h, a, $sched!($base, 7));
        };
    }
    // The schedules: `live` reads a digest word, `padk` is the folded
    // constant, `band` reads W16..W31 precomputed below, `ext` is the generic
    // recurrence from W32 on.
    macro_rules! live {
        ($b:literal, $j:literal) => {
            w[$b + $j].add(L::splat(K[$b + $j]))
        };
    }
    macro_rules! padk {
        ($b:literal, $j:literal) => {
            L::splat(PADK[$j])
        };
    }
    macro_rules! band {
        ($b:literal, $j:literal) => {
            w[$b - 16 + $j].add(L::splat(K[$b + $j]))
        };
    }
    macro_rules! ext {
        ($b:literal, $j:literal) => {{
            const I: usize = ($b + $j) & 15;
            let s1 = small_sigma1(w[(I + 14) & 15]);
            let s0 = small_sigma0(w[(I + 1) & 15]);
            w[I] = s1.add(w[(I + 9) & 15]).add(s0).add(w[I]);
            w[I].add(L::splat(K[$b + $j]))
        }};
    }

    rounds8!(live, 0);
    rounds8!(padk, 8);

    // W16..W31, in window-slot order. Each line is the recurrence with the
    // pad's zeros dropped and its sigmas folded; the differential test holds
    // every line against `core::compress`.
    w[0] = w[0].add(small_sigma0(w[1]));
    w[1] = w[1].add(small_sigma0(w[2])).add(L::splat(sig1(PAD[7])));
    w[2] = w[2].add(small_sigma0(w[3])).add(small_sigma1(w[0]));
    w[3] = w[3].add(small_sigma0(w[4])).add(small_sigma1(w[1]));
    w[4] = w[4].add(small_sigma0(w[5])).add(small_sigma1(w[2]));
    w[5] = w[5].add(small_sigma0(w[6])).add(small_sigma1(w[3]));
    w[6] = w[6]
        .add(small_sigma0(w[7]))
        .add(L::splat(PAD[7]))
        .add(small_sigma1(w[4]));
    w[7] = w[7]
        .add(L::splat(sig0(PAD[0])))
        .add(w[0])
        .add(small_sigma1(w[5]));
    w[8] = L::splat(PAD[0]).add(w[1]).add(small_sigma1(w[6]));
    w[9] = w[2].add(small_sigma1(w[7]));
    w[10] = w[3].add(small_sigma1(w[8]));
    w[11] = w[4].add(small_sigma1(w[9]));
    w[12] = w[5].add(small_sigma1(w[10]));
    w[13] = w[6].add(small_sigma1(w[11]));
    w[14] = L::splat(sig0(PAD[7])).add(w[7]).add(small_sigma1(w[12]));
    w[15] = L::splat(PAD[7])
        .add(small_sigma0(w[0]))
        .add(w[8])
        .add(small_sigma1(w[13]));

    rounds8!(band, 16);
    rounds8!(band, 24);
    rounds8!(ext, 32);
    rounds8!(ext, 40);
    rounds8!(ext, 48);
    rounds8!(ext, 56);

    state[0] = state[0].add(a);
    state[1] = state[1].add(b);
    state[2] = state[2].add(c);
    state[3] = state[3].add(d);
    state[4] = state[4].add(e);
    state[5] = state[5].add(f);
    state[6] = state[6].add(g);
    state[7] = state[7].add(h);
}

/// Advances `W` lanes of chains `n` links: the step kernel for lane backends.
#[inline(always)]
pub(crate) fn steps_lanes<L: Lanes, const W: usize>(h: &mut [[u32; 8]], n: u64) {
    const { assert!(L::N == W) };
    debug_assert_eq!(h.len(), W);

    let mut scratch = [0u32; W];
    let mut hv = [L::splat(0); 8];
    for (j, hj) in hv.iter_mut().enumerate() {
        for (lane, s) in scratch.iter_mut().enumerate() {
            *s = h[lane][j];
        }
        *hj = L::load(&scratch);
    }

    let init = H0.map(L::splat);
    if L::FLAT_ROUNDS {
        for _ in 0..n {
            let mut st = init;
            compress_chain::<L>(&mut st, &hv);
            hv = st;
        }
    } else {
        // The folded band is flat by nature; a backend that must roll its
        // rounds takes the generic compression and keeps its loop.
        let pad = PAD.map(L::splat);
        for _ in 0..n {
            let mut w = [L::splat(0); 16];
            w[..8].copy_from_slice(&hv);
            w[8..].copy_from_slice(&pad);
            let mut st = init;
            compress::<L>(&mut st, w);
            hv = st;
        }
    }

    for (j, hj) in hv.iter().enumerate() {
        hj.store(&mut scratch);
        for (lane, s) in scratch.iter().enumerate() {
            h[lane][j] = *s;
        }
    }
}

/// `2 * W` lanes as two waves with their rounds interlaced.
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
#[inline(always)]
pub(crate) fn steps_lanes2<L: Lanes, const W: usize>(h: &mut [[u32; 8]], n: u64) {
    const { assert!(L::N == W) };
    debug_assert_eq!(h.len(), 2 * W);

    let mut scratch = [0u32; W];
    let mut wa = [L::splat(0); 8];
    let mut wb = [L::splat(0); 8];
    for j in 0..8 {
        for (lane, s) in scratch.iter_mut().enumerate() {
            *s = h[lane][j];
        }
        wa[j] = L::load(&scratch);
        for (lane, s) in scratch.iter_mut().enumerate() {
            *s = h[W + lane][j];
        }
        wb[j] = L::load(&scratch);
    }

    let pad = PAD.map(L::splat);
    let init = H0.map(L::splat);
    for _ in 0..n {
        let mut ba = [L::splat(0); 16];
        ba[..8].copy_from_slice(&wa);
        ba[8..].copy_from_slice(&pad);
        let mut bb = [L::splat(0); 16];
        bb[..8].copy_from_slice(&wb);
        bb[8..].copy_from_slice(&pad);

        let (mut sta, mut stb) = (init, init);
        crate::avx512x2::compress2::<L>(&mut sta, &mut stb, ba, bb);
        wa = sta;
        wb = stb;
    }

    for j in 0..8 {
        wa[j].store(&mut scratch);
        for (lane, s) in scratch.iter().enumerate() {
            h[lane][j] = *s;
        }
        wb[j].store(&mut scratch);
        for (lane, s) in scratch.iter().enumerate() {
            h[W + lane][j] = *s;
        }
    }
}

/// The scalar steps, so the portable rows ride the same scheduler
///
/// `unsafe` only to fit `StepsFn`; the body needs no CPU feature.
pub(crate) unsafe fn steps_scalar1(h: &mut [[u32; 8]], n: u64) {
    steps_lanes::<Scalar<1>, 1>(h, n)
}

/// As `steps_scalar1`, at the aarch64 register file's width
pub(crate) unsafe fn steps_scalar8(h: &mut [[u32; 8]], n: u64) {
    steps_lanes::<Scalar<8>, 8>(h, n)
}

/// Pulls the next unfinished chain into `lane`, retiring empty ones in passing
///
/// Returns false once the queue is dry; the lane then keeps hashing whatever
/// it holds, which is free.
#[allow(clippy::too_many_arguments)]
fn feed(
    lane: usize,
    order: &[u32],
    next: &mut usize,
    seeds: &[[u8; 32]],
    lens: &[u64],
    out: &mut [[u8; 32]],
    h: &mut [[u32; 8]],
    rem: &mut [u64],
    who: &mut [usize],
) -> bool {
    while *next < order.len() {
        let i = order[*next] as usize;
        *next += 1;
        if lens[i] == 0 {
            out[i] = seeds[i];
            continue;
        }
        h[lane] = seed_words(&seeds[i]);
        rem[lane] = lens[i];
        who[lane] = i;
        return true;
    }
    false
}

/// Runs every chain through `width` lanes: longest first, refill on retire
///
/// The step kernel never sees raggedness. This runs it to the nearest finish
/// line, retires the lanes that arrived, refills them from the queue, and
/// goes again, so state crosses the kernel boundary once per chain rather
/// than per link. Longest-first ordering makes the queue drain into its
/// shortest chains, which is the least time lanes can idle at the tail.
///
/// # Safety
///
/// `steps` must be safe to call on the running CPU: the caller pairs it with
/// the feature probe that admitted it, as `steps_for` does.
pub(crate) unsafe fn run_scheduled(
    width: usize,
    steps: StepsFn,
    seeds: &[[u8; 32]],
    lens: &[u64],
    out: &mut [[u8; 32]],
) {
    let n = seeds.len();
    debug_assert!((1..=MAX_WIDTH).contains(&width));
    debug_assert!(n == lens.len() && n == out.len());
    debug_assert!(u32::try_from(n).is_ok());

    // Longest chains first. Batches within the width sort on the stack; the
    // index tie-break keeps equal lengths in caller order on both paths.
    let mut small = [0u32; MAX_WIDTH];
    let mut big: Vec<u32>;
    let order: &[u32] = if n <= MAX_WIDTH {
        for (s, i) in small.iter_mut().zip(0..n as u32) {
            *s = i;
        }
        for i in 1..n {
            let mut j = i;
            while j > 0 && lens[small[j] as usize] > lens[small[j - 1] as usize] {
                small.swap(j - 1, j);
                j -= 1;
            }
        }
        &small[..n]
    } else {
        big = (0..n as u32).collect();
        big.sort_unstable_by(|&a, &b| lens[b as usize].cmp(&lens[a as usize]).then(a.cmp(&b)));
        &big
    };

    let mut h = [[0u32; 8]; MAX_WIDTH];
    let mut rem = [u64::MAX; MAX_WIDTH];
    let mut who = [0usize; MAX_WIDTH];
    let mut next = 0usize;
    let mut live = 0usize;
    for lane in 0..width {
        if feed(
            lane, order, &mut next, seeds, lens, out, &mut h, &mut rem, &mut who,
        ) {
            live += 1;
        }
    }

    while live > 0 {
        // A live lane exists, so the min is a real length; empty lanes sit at
        // u64::MAX and never win it.
        let run = rem[..width].iter().copied().min().unwrap();
        steps(&mut h[..width], run);
        for lane in 0..width {
            if rem[lane] == u64::MAX {
                continue;
            }
            rem[lane] -= run;
            if rem[lane] == 0 {
                out[who[lane]] = word_bytes(&h[lane]);
                rem[lane] = u64::MAX;
                live -= 1;
                if feed(
                    lane, order, &mut next, seeds, lens, out, &mut h, &mut rem, &mut who,
                ) {
                    live += 1;
                }
            }
        }
    }
}

/// Which chain kernel this build and CPU resolve to
///
/// One question decides it, where the multi-buffer ladder weighs several:
/// does the CPU have a SHA-256 unit. Lane count cannot matter when there is
/// never more than one live message.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    ShaNi,
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    NeonSha2,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Portable => "portable",
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            Kernel::ShaNi => "shani",
            #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
            Kernel::NeonSha2 => "neon-sha2",
        }
    }
}

#[inline]
#[allow(clippy::needless_return)]
fn select() -> Kernel {
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    {
        if crate::dispatch::have_shani() {
            return Kernel::ShaNi;
        }
        return Kernel::Portable;
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    {
        if std::arch::is_aarch64_feature_detected!("sha2") {
            return Kernel::NeonSha2;
        }
        return Kernel::Portable;
    }
    #[cfg(not(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(feature = "scalar")
    )))]
    return Kernel::Portable;
}

pub(crate) fn hash_chain(seed: &[u8; 32], n: u64) -> [u8; 32] {
    match select() {
        Kernel::Portable => portable(seed, n),
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        // SAFETY: `select` returns this arm only when the probe passed.
        Kernel::ShaNi => unsafe { crate::shani::chain(seed, n) },
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        // SAFETY: as above.
        Kernel::NeonSha2 => unsafe { crate::neon_sha2::chain(seed, n) },
    }
}

pub(crate) fn backend() -> &'static str {
    select().name()
}

/// Drives many independent chains through one scheduled kernel.
pub(crate) fn hash_chains(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
    match pick(crate::dispatch::select(), seeds.len()) {
        Some(k) => {
            let (width, steps) = steps_for(k, seeds.len());
            // SAFETY: `steps_for` pairs each steps fn with the probe that
            // admitted its kernel.
            unsafe { run_scheduled(width, steps, seeds, lens, out) }
        }
        None => {
            // Too few live lanes to pay for any group, so take the
            // latency-optimal kernel one chain at a time.
            for i in 0..seeds.len() {
                out[i] = hash_chain(&seeds[i], lens[i]);
            }
        }
    }
}

/// The kernel worth running for a batch of `remaining` chains, if any
///
/// a batch too small to pay for the wide kernel is often still
/// wide enough for the four-lane SHA unit, whose step is far smaller.
fn pick(wide: crate::dispatch::Kernel, remaining: usize) -> Option<crate::dispatch::Kernel> {
    if remaining >= chain_min(wide) {
        return Some(wide);
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    if remaining >= 3 && crate::dispatch::have_shani() {
        return Some(crate::dispatch::Kernel::ShaNiX4);
    }
    None
}

/// Live chains below which a kernel loses to what `pick` would fall back to.
fn chain_min(kernel: crate::dispatch::Kernel) -> usize {
    use crate::dispatch::Kernel;
    #[allow(unused_variables)]
    let hw = !matches!(select(), self::Kernel::Portable);
    match kernel {
        // Multi-buffer in general-purpose registers loses to one lane on every
        // target measured, chains included.
        Kernel::Portable1 | Kernel::Portable8 => usize::MAX,
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        Kernel::Neon4 => {
            if hw {
                usize::MAX
            } else {
                2
            }
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        Kernel::NeonSha2x4 => 3,
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::ShaNiX4 => 3,
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::Avx2_8 => {
            if hw {
                usize::MAX
            } else {
                2
            }
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::Avx512_16 => {
            if hw {
                10
            } else {
                2
            }
        }
        // Interlaced above 16 lanes and a single wave at or below, so the
        // threshold that matters is the single wave's.
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::Avx512_16x2 => {
            if hw {
                10
            } else {
                2
            }
        }
        // wasm has no SHA unit, so its serial chain is the portable one.
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            not(feature = "scalar")
        ))]
        Kernel::Simd128_4 => 2,
    }
}

/// The step kernel and width for a picked dispatch kernel
///
/// The one width that depends on the batch.
#[allow(unused_variables)]
fn steps_for(kernel: crate::dispatch::Kernel, n: usize) -> (usize, StepsFn) {
    use crate::dispatch::Kernel;
    match kernel {
        Kernel::Portable1 => (1, steps_scalar1 as StepsFn),
        Kernel::Portable8 => (8, steps_scalar8 as StepsFn),
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        Kernel::Neon4 => (4, crate::neon::steps as StepsFn),
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        Kernel::NeonSha2x4 => (4, crate::neon_sha2::steps4 as StepsFn),
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::ShaNiX4 => (4, crate::shani::steps4 as StepsFn),
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::Avx2_8 => (8, crate::avx2::steps as StepsFn),
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::Avx512_16 => (16, crate::avx512::steps as StepsFn),
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Kernel::Avx512_16x2 => {
            if n <= 16 {
                (16, crate::avx512::steps as StepsFn)
            } else {
                (32, crate::avx512x2::steps2 as StepsFn)
            }
        }
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            not(feature = "scalar")
        ))]
        Kernel::Simd128_4 => (4, crate::simd128::steps as StepsFn),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a wrong constant or a mis-slotted band word diverges on the first input
    #[test]
    fn folded_schedule() {
        let mut x = 0x243f_6a88_85a3_08d3u64;
        let mut rand = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u32
        };

        for _ in 0..100 {
            let h = [(); 8].map(|_| Scalar::<1>([rand()]));

            let mut w = [Scalar::<1>([0]); 16];
            w[..8].copy_from_slice(&h);
            for (wj, p) in w[8..].iter_mut().zip(PAD) {
                *wj = Scalar::splat(p);
            }
            let mut want = H0.map(Scalar::<1>::splat);
            compress(&mut want, w);

            let mut got = H0.map(Scalar::<1>::splat);
            compress_chain(&mut got, &h);

            assert_eq!(want.map(|s| s.0), got.map(|s| s.0));
        }
    }

    // distinct lengths force a retire-and-refill at every finish line
    #[test]
    fn scheduler_refill() {
        for n in [1usize, 2, 5, 8, 9, 17, 33] {
            let seeds: Vec<[u8; 32]> = (0..n).map(|i| [i as u8; 32]).collect();
            let lens: Vec<u64> = (0..n).map(|i| (i as u64 * 3) % 7).collect();
            let mut out = vec![[0u8; 32]; n];
            // SAFETY: the scalar steps need no CPU feature.
            unsafe { run_scheduled(8, steps_scalar8, &seeds, &lens, &mut out) };
            for i in 0..n {
                assert_eq!(out[i], hash_chain(&seeds[i], lens[i]), "chain {i}");
            }
        }
    }
}
