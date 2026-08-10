//! Solana's proof-of-history chain, backends against `sha2`
//!
//! `sha2` is the baseline because it is what the Solana validator's hasher
//! wraps, and its proof-of-history loop is exactly this:
//!
//! ```text
//! for _ in 0..num_hashes { self.hash = hash(self.hash.as_ref()); }
//! ```
//!
//! There is no parallelism here to find. Link `i + 1` hashes link `i`, so the
//! multi-buffer kernels this crate exists for have nothing to put in their
//! lanes and are not in the table. What differs between the rows is only how
//! much wrapper is left around one 64-byte compression: padding construction,
//! state packing, and a digest byte swap the next link immediately undoes.
//!
//! The `hash_many, 1/link` row is the control. It runs the same SHA-256 unit
//! as `hash_chain` through the ordinary batch entry point, one message per
//! call, so the gap between those two rows is the specialisation alone and
//! not the hardware.
//!
//! Lengths are the Solana validator's own. `DEFAULT_HASHES_PER_BATCH` = 500
//! is the granularity its service hashes at; `DEFAULT_HASHES_PER_TICK`
//! = 62,500 is one tick. `DEFAULT_HASHES_PER_SECOND` = 10,000,000 is the
//! clock a leader has to hold on one pinned core, which is 100 ns per link.
//!
//! # Read the baseline before the ratio
//!
//! `sha2` picks its backend by platform and the validator enables no features
//! on it. On x86_64 it reaches SHA-NI by runtime detection, so production PoH
//! already runs on hardware and the ratio is a wrapper measurement. On aarch64
//! the hardware path is behind `sha2/asm`, which is off, so the default
//! baseline is software and the ratio is a hardware-against-software one that
//! says nothing about the production gap. `--features sha2-asm-bench` turns
//! that baseline into hardware; the header line says which one ran.
//!
//! Run with `cargo bench --bench poh_chain`.

use {
    sha2::{Digest, Sha256},
    std::time::Instant,
    tape_sha256::{backends, hash_chain, hash_chains, hash_many},
};

/// The Solana validator's DEFAULT_HASHES_PER_BATCH
const BATCH: u64 = 500;
/// The Solana validator's DEFAULT_HASHES_PER_TICK
const TICK: u64 = 62_500;
/// The Solana validator's DEFAULT_HASHES_PER_SECOND, the rate a leader must hold
const POH_CLOCK: f64 = 10_000_000.0;

/// Which compression `sha2` was built with, since the ratio is meaningless
/// without it
fn sha2_backend() -> &'static str {
    if cfg!(feature = "sha2-asm-bench") {
        if cfg!(target_arch = "aarch64") {
            "asm (crypto ext)"
        } else {
            "asm"
        }
    } else if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        "auto (SHA-NI when present) -- what validators run"
    } else {
        "software -- what validators run here, but not what the silicon can do"
    }
}

fn sha2_chain(seed: &[u8; 32], n: u64) -> [u8; 32] {
    let mut h = *seed;
    for _ in 0..n {
        let mut d = Sha256::new();
        d.update(h);
        h = d.finalize().into();
    }
    h
}

/// The same unit `hash_chain` uses, reached through the batch entry point
fn batched_chain(seed: &[u8; 32], n: u64) -> [u8; 32] {
    let mut h = *seed;
    let mut out = [[0u8; 32]; 1];
    for _ in 0..n {
        hash_many(&[&h[..]], &mut out);
        h = out[0];
    }
    h
}

/// Seconds per link, by the min over many short windows
fn bench<F: FnMut()>(name: &str, links: u64, mut f: F) -> f64 {
    // Many rounds because the min has to find a fast core, not just a quiet
    // one: on a hybrid CPU an unlucky row runs entirely on an efficiency core
    // and reports a clean, wrong, uniformly slow number.
    const ROUNDS: usize = 40;
    /// Short on purpose: a preemption lands inside one window, which the min
    /// then discards, rather than being averaged into a long one.
    const WINDOW: f64 = 0.003;

    for _ in 0..3 {
        f();
    }
    // Calibrated to wall clock, not to a fixed iteration count. The rows here
    // differ by 4x, so a fixed count would hand the fastest row the shortest
    // window and the noisiest spread.
    let t0 = Instant::now();
    f();
    let one = t0.elapsed().as_secs_f64().max(1e-9);
    let iters = ((WINDOW / one) as u64).max(1);

    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let per_link = t0.elapsed().as_secs_f64() / (iters * links) as f64;
        best = best.min(per_link);
        worst = worst.max(per_link);
    }
    let spread = 100.0 * (worst - best) / best;
    println!(
        "{name:<26}{:>9.2} ns/link{:>10.2} MH/s   (spread {spread:>4.1}%)",
        best * 1e9,
        1e-6 / best
    );
    best
}

fn run(links: u64, label: &str) -> (f64, f64) {
    let seed = [0x11u8; 32];

    // A chain kernel that skips a byte swap produces a fast wrong answer, not
    // a slow one, so every row is gated against sha2 before it is timed.
    let expect = sha2_chain(&seed, links);
    for (name, got) in [
        ("hash_chain", hash_chain(&seed, links)),
        ("hash_many", batched_chain(&seed, links)),
        ("chain_portable", backends::chain_portable(&seed, links)),
    ] {
        assert_eq!(got, expect, "{name} disagrees with sha2 -- wrong, not slow");
    }

    println!("{label}: chains of {links} links");
    let base = bench("sha2 (baseline)", links, || {
        std::hint::black_box(sha2_chain(&seed, links));
    });
    let batched = bench("hash_many, 1/link", links, || {
        std::hint::black_box(batched_chain(&seed, links));
    });
    let chain = bench("hash_chain", links, || {
        std::hint::black_box(hash_chain(&seed, links));
    });
    let portable = bench("hash_chain portable", links, || {
        std::hint::black_box(backends::chain_portable(&seed, links));
    });

    println!();
    println!("  vs sha2:  hash_chain {:.2}x", base / chain);
    println!("            hash_many  {:.2}x", base / batched);
    println!("            portable   {:.2}x", base / portable);
    println!(
        "  specialisation alone (hash_chain vs hash_many): {:.2}x",
        batched / chain
    );
    println!(
        "  PoH clock (10 MH/s = 100 ns/link): sha2 {:.0}% of budget, \
         hash_chain {:.0}%",
        100.0 * base * POH_CLOCK,
        100.0 * chain * POH_CLOCK
    );
    println!();

    (base, chain)
}

/// Independent chains in one slot of replay: 64 ticks, before transaction entries
const ENTRIES: usize = 64;
/// Links per entry in the replay rows
///
/// Far below a tick's 62,500 only to keep the timing windows short. The two
/// lengths above already establish that per-link cost does not move with
/// chain length.
const REPLAY_LINKS: u64 = 1024;

/// What replay has that generation does not: independent chains
///
/// Every entry publishes its ending hash, so entry `i`'s segment starts from
/// entry `i - 1`'s known hash and all segments can run at once. Validators
/// already spend that across threads, one serial chain per thread; these rows
/// are what a single thread gets by running the segments in lockstep instead.
///
/// The sweep is over group size rather than a single number because that is
/// the question -- how many independent chains it takes before the SHA-256
/// unit stops being latency-bound, and whether the dispatched width is right.
fn replay() {
    let seeds: Vec<[u8; 32]> = (0..ENTRIES)
        .map(|i| [(i as u8).wrapping_mul(37); 32])
        .collect();
    let lens = vec![REPLAY_LINKS; ENTRIES];
    let mut out = vec![[0u8; 32]; ENTRIES];
    let total = ENTRIES as u64 * REPLAY_LINKS;

    let expect: Vec<[u8; 32]> = seeds.iter().map(|s| sha2_chain(s, REPLAY_LINKS)).collect();
    hash_chains(&seeds, &lens, &mut out);
    assert_eq!(out, expect, "hash_chains disagrees with sha2");

    println!("replay: {ENTRIES} independent chains of {REPLAY_LINKS} links");
    let base = bench("sha2, one at a time", total, || {
        for s in &seeds {
            std::hint::black_box(sha2_chain(s, REPLAY_LINKS));
        }
    });
    let serial = bench("hash_chain, one at a time", total, || {
        for s in &seeds {
            std::hint::black_box(hash_chain(s, REPLAY_LINKS));
        }
    });

    let mut best = (0usize, f64::INFINITY);
    for group in [2usize, 3, 4, 8, 16, 32, 64] {
        let t = bench(&format!("hash_chains, {group} at a time"), total, || {
            for c in (0..ENTRIES).step_by(group) {
                let end = (c + group).min(ENTRIES);
                hash_chains(&seeds[c..end], &lens[..end - c], &mut out[c..end]);
            }
            std::hint::black_box(&out);
        });
        if t < best.1 {
            best = (group, t);
        }
    }

    println!();
    println!("  vs sha2 one at a time:");
    println!("    hash_chain (serial)        {:.2}x", base / serial);
    println!(
        "    hash_chains x{:<3}           {:.2}x",
        best.0,
        base / best.1
    );
    println!(
        "  latency floor -> throughput floor: {:.2}x  ({:.2} -> {:.2} ns/link)",
        serial / best.1,
        serial * 1e9,
        best.1 * 1e9
    );
    println!(
        "  dispatched width is {}, best measured width is {}",
        tape_sha256::lane_width(),
        best.0
    );
    println!();
}

/// Ragged lengths: what the refill scheduler buys over fixed groups
///
/// Long chains stand in for tick entries among short transaction entries.
/// A caller chunking by width pins each long chain to a group and pays its
/// full length while the other lanes idle; one call schedules longest-first
/// and refills lanes as chains retire. Same work, only the scheduling differs.
fn ragged() {
    const LONG: u64 = 2048;
    const SHORT: u64 = 128;
    let seeds: Vec<[u8; 32]> = (0..ENTRIES)
        .map(|i| [(i as u8).wrapping_mul(59); 32])
        .collect();
    let lens: Vec<u64> = (0..ENTRIES)
        .map(|i| if i % 8 == 0 { LONG } else { SHORT })
        .collect();
    let mut out = vec![[0u8; 32]; ENTRIES];
    let total: u64 = lens.iter().sum();
    let width = tape_sha256::lane_width();

    let expect: Vec<[u8; 32]> = seeds
        .iter()
        .zip(&lens)
        .map(|(s, &l)| sha2_chain(s, l))
        .collect();
    hash_chains(&seeds, &lens, &mut out);
    assert_eq!(out, expect, "ragged hash_chains disagrees with sha2");

    println!(
        "ragged replay: {} chains of {LONG} links among {} of {SHORT}, arrival order",
        ENTRIES / 8,
        ENTRIES - ENTRIES / 8
    );
    let serial = bench("hash_chain, one at a time", total, || {
        for (s, &l) in seeds.iter().zip(&lens) {
            std::hint::black_box(hash_chain(s, l));
        }
    });
    let grouped = bench(&format!("hash_chains, chunks of {width}"), total, || {
        for c in (0..ENTRIES).step_by(width) {
            let end = (c + width).min(ENTRIES);
            hash_chains(&seeds[c..end], &lens[c..end], &mut out[c..end]);
        }
        std::hint::black_box(&out);
    });
    let scheduled = bench("hash_chains, one call", total, || {
        hash_chains(&seeds, &lens, &mut out);
        std::hint::black_box(&out);
    });

    println!();
    println!(
        "  scheduling alone (one call vs chunks): {:.2}x   (vs serial: {:.2}x)",
        grouped / scheduled,
        serial / scheduled
    );
    println!();
}

type ChainsFn = unsafe fn(&[[u8; 32]], &[u64], &mut [[u8; 32]]);

/// Per-kernel step cost, which is what sets the break-even group size
///
/// A partial group costs a full group -- the dead lanes still run every round
/// -- so a kernel whose full-width step costs `T` only beats the serial
/// chain's `S` per link above `T / S` live chains. Both terms are measured
/// here so the dispatch threshold is a number off this table rather than a
/// guess.
fn kernels() {
    let seeds: Vec<[u8; 32]> = (0..ENTRIES)
        .map(|i| [(i as u8).wrapping_mul(37); 32])
        .collect();
    let lens = vec![REPLAY_LINKS; ENTRIES];

    let serial = bench("hash_chain (serial, S)", REPLAY_LINKS, || {
        std::hint::black_box(hash_chain(&seeds[0], REPLAY_LINKS));
    });
    println!();

    let row = |name: &str, width: usize, f: ChainsFn| {
        if width > ENTRIES {
            return;
        }
        let groups = ENTRIES / width;
        let total = (groups * width) as u64 * REPLAY_LINKS;
        let mut out = vec![[0u8; 32]; ENTRIES];

        // A lockstep kernel that leaks one lane into another produces a fast
        // wrong answer, so gate before timing.
        unsafe { f(&seeds[..width], &lens[..width], &mut out[..width]) };
        for i in 0..width {
            assert_eq!(
                out[i],
                sha2_chain(&seeds[i], REPLAY_LINKS),
                "{name} lane {i} disagrees with sha2"
            );
        }

        let t = bench(name, total, || {
            for c in 0..groups {
                let (a, b) = (c * width, (c + 1) * width);
                unsafe { f(&seeds[a..b], &lens[a..b], &mut out[a..b]) };
            }
            std::hint::black_box(&out);
        });
        println!(
            "{:<26}step {:>8.1} ns, beats the serial chain above {:.0} live chains",
            "",
            t * width as f64 * 1e9,
            (t * width as f64 / serial).ceil()
        );
    };

    println!("per-kernel, at each kernel's own width:");
    row("portable-1", 1, |s, l, o| {
        backends::chains::portable1(s, l, o)
    });
    row("portable-8", 8, |s, l, o| {
        backends::chains::portable8(s, l, o)
    });
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    {
        if std::arch::is_aarch64_feature_detected!("sha2") {
            row("neon-sha2-x4", 4, backends::chains::neon_sha2x4);
            row("neon-sha2-x8", 8, backends::chains::neon_sha2x8);
        }
        row("neon-4", 4, backends::chains::neon4);
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    {
        if is_x86_feature_detected!("sha") {
            row("shani-x4", 4, backends::chains::shani_x4);
        }
        if is_x86_feature_detected!("avx2") {
            row("avx2-8", 8, backends::chains::avx2_8);
        }
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            row("avx512-16", 16, backends::chains::avx512_16);
            row("avx512-2x16", 32, backends::chains::avx512_16x2);
        }
    }
    println!();
}

fn main() {
    println!("multi-buffer backend: {}", tape_sha256::backend());
    println!("chain backend:        {}", tape_sha256::chain_backend());
    println!("sha2 baseline:        {}", sha2_backend());
    println!();

    run(BATCH, "PohService batch");
    let (base, chain) = run(TICK, "one tick");
    replay();
    ragged();
    kernels();

    println!("one 400 ms slot is 64 ticks = {} links", 64 * TICK);
    println!(
        "  sha2       {:>7.1} ms of the 400 ms slot",
        base * (64 * TICK) as f64 * 1e3
    );
    println!("  hash_chain {:>7.1} ms", chain * (64 * TICK) as f64 * 1e3);
}
