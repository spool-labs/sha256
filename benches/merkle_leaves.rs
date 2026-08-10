//! Hashes one Solana erasure batch of Merkle leaves, backends against `sha2`
//!
//! `sha2` is the baseline because it is what the validator's shredder and recovery
//! path use for leaf hashing today.
//!
//! The workload is the real one: 64 leaves (32 data + 32 coding shreds), each
//! a 26-byte domain-separation prefix and ~1KB of shred payload. All 64 land
//! on the same block count, so no lane sits idle.
//!
//! Run with `cargo bench --bench merkle_leaves`.

use {
    sha2::{Digest, Sha256},
    std::time::Instant,
    tape_sha256::{backends, Message},
};

const PREFIX: &[u8] = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
/// Leaf size for a data shred in a 32:32 batch with 6 proof entries
const DATA_LEAF: usize = 1019;
/// Leaf size for a coding shred in the same batch
const CODE_LEAF: usize = 1044;
const DATA_SHREDS: usize = 32;
const CODE_SHREDS: usize = 32;
const LEAVES: usize = DATA_SHREDS + CODE_SHREDS;

/// Firedancer's batch SHA-256, purely as a benchmark reference
///
/// Enabled with `--features firedancer-bench` and `FD_LIB_DIR` pointing at a
/// Firedancer build. Nothing from Firedancer is vendored or linked into the
/// library itself.
///
/// The `fd_sha256_batch_*` entry points are `static inline` in Firedancer's
/// header, so they are not linkable symbols; the kernels behind them are. We
/// call those directly, which also lets us time each width separately rather
/// than whichever one the header happened to be compiled for.
///
/// Each takes three parallel arrays indexed `[0, batch_cnt)`, which Firedancer
/// documents as 32-byte aligned, with `batch_cnt` in `[1, FD_SHA256_BATCH_MAX]`.
#[cfg(feature = "firedancer-bench")]
mod firedancer {
    use std::os::raw::c_void;

    pub const AVX_LANES: usize = 8;
    pub const AVX512_LANES: usize = 16;

    unsafe extern "C" {
        pub fn fd_sha256_private_batch_avx(
            batch_cnt: u64,
            batch_data: *const *const c_void,
            batch_sz: *const u64,
            batch_hash: *const *mut c_void,
        );
        // Only present when the linked Firedancer build enabled AVX-512, so
        // the declaration is gated too: referencing it against a build without
        // it is a link error, not a runtime fallback.
        #[cfg(feature = "firedancer-bench-avx512")]
        pub fn fd_sha256_private_batch_avx512(
            batch_cnt: u64,
            batch_data: *const *const c_void,
            batch_sz: *const u64,
            batch_hash: *const *mut c_void,
        );
    }

    /// 64-byte-aligned argument arrays, over the documented 32-byte minimum
    #[repr(C, align(64))]
    pub struct Args {
        pub data: [*const c_void; AVX512_LANES],
        pub sz: [u64; AVX512_LANES],
        pub hash: [*mut c_void; AVX512_LANES],
    }

    impl Default for Args {
        fn default() -> Self {
            Args {
                data: [std::ptr::null(); AVX512_LANES],
                sz: [0; AVX512_LANES],
                hash: [std::ptr::null_mut(); AVX512_LANES],
            }
        }
    }
}

/// isa-l_crypto's multi-buffer SHA-256, purely as a benchmark reference
///
/// Enabled with `--features isal-bench` and `ISAL_DIR` pointing at a built
/// isa-l_crypto tree. Nothing from isa-l is vendored or linked into the
/// library itself.
///
/// Everything crosses the boundary through `benches/isal_shim.c` rather than a
/// repr(C) mirror of `ISAL_SHA256_HASH_CTX`. A wrong mirror would not fail to
/// link, it would corrupt the manager's state, so the layout stays on the C
/// side and Rust sees an opaque handle.
///
/// Unlike Firedancer's kernels these are a job-manager API: submit N jobs,
/// flush until drained. That scheduling is real work the kernel-level calls do
/// not do, so this row is what an isa-l caller pays, not a bare kernel time.
#[cfg(feature = "isal-bench")]
mod isal {
    use std::os::raw::c_void;

    unsafe extern "C" {
        pub fn isal_bench_new() -> *mut c_void;
        pub fn isal_bench_free(h: *mut c_void);
        pub fn isal_bench_run_avx512(
            h: *mut c_void,
            n: u32,
            bufs: *const *const u8,
            lens: *const u32,
            out: *mut [u8; 32],
        ) -> i32;
        pub fn isal_bench_run_avx512_ni(
            h: *mut c_void,
            n: u32,
            bufs: *const *const u8,
            lens: *const u32,
            out: *mut [u8; 32],
        ) -> i32;
        pub fn isal_bench_run_dispatch(
            h: *mut c_void,
            n: u32,
            bufs: *const *const u8,
            lens: *const u32,
            out: *mut [u8; 32],
        ) -> i32;
    }
}

fn make_leaves() -> Vec<Vec<u8>> {
    (0..LEAVES)
        .map(|i| {
            let len = if i < DATA_SHREDS {
                DATA_LEAF
            } else {
                CODE_LEAF
            };
            (0..len)
                .map(|j| (j as u8).wrapping_mul(31).wrapping_add(i as u8))
                .collect()
        })
        .collect()
}

/// Times `f` and reports the best of several rounds
///
/// A single mean gave the same backend a 37% spread between invocations here,
/// enough to invert a comparison. Interference (frequency scaling, other
/// processes, migration between core types) only ever makes a run slower, so
/// the minimum is the most stable estimate of the code's real cost. The spread
/// is printed alongside so a noisy run is visible rather than averaged in.
fn bench<F: FnMut()>(name: &str, bytes_per_iter: usize, mut f: F) -> f64 {
    // Many short windows beat few long ones for a min-estimator: a preemption
    // or frequency dip lands in one short window, which the min then discards,
    // instead of being averaged into a long one.
    const ROUNDS: usize = 15;
    const ITERS: usize = 50;

    for _ in 0..20 {
        f();
    }
    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        for _ in 0..ITERS {
            f();
        }
        let per_iter = t0.elapsed().as_secs_f64() / ITERS as f64;
        best = best.min(per_iter);
        worst = worst.max(per_iter);
    }
    let mbps = bytes_per_iter as f64 / best / 1e6;
    let spread = 100.0 * (worst - best) / best;
    println!(
        "{name:<28}{:>9.2} us/batch{:>11.0} MB/s   (spread {spread:>4.1}%)",
        best * 1e6,
        mbps
    );
    best
}

/// Two AVX-512 workers on an SMT sibling pair, half the batch each
///
/// The reference harness for the SMT pattern in the crate docs. Threads are
/// caller-owned; the library never spawns or pins. Homogeneous and 50/50, so
/// there is no balancing and no straggler, which is why this measures with a
/// sub-1% spread where an uneven split did not.
///
/// Measures throughput, not per-batch latency: each worker runs its share
/// ITERS times with no per-iteration synchronisation, so the only sync is at
/// start and end. Run it under `taskset -c <a>,<b>` where a and b are thread
/// siblings of one core; on any other pairing this degrades to ordinary
/// multicore and measures something else entirely.
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
fn bench_smt2(name: &str, bytes_per_iter: usize, ma: &[Message<'_>], mb: &[Message<'_>]) -> f64 {
    const ROUNDS: usize = 15;
    const ITERS: usize = 400;
    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for r in 0..=ROUNDS {
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for part in [ma, mb] {
                sc.spawn(move || {
                    let mut o = vec![[0u8; 32]; part.len()];
                    for _ in 0..ITERS {
                        unsafe { backends::avx512_16(part, &mut o) };
                        std::hint::black_box(&o);
                    }
                });
            }
        });
        let per_iter = t0.elapsed().as_secs_f64() / ITERS as f64;
        if r == 0 {
            continue; // warmup
        }
        best = best.min(per_iter);
        worst = worst.max(per_iter);
    }
    let mbps = bytes_per_iter as f64 / best / 1e6;
    let spread = 100.0 * (worst - best) / best;
    println!(
        "{name:<28}{:>9.2} us/batch{:>11.0} MB/s   (spread {spread:>4.1}%)",
        best * 1e6,
        mbps
    );
    best
}

fn main() {
    let leaves = make_leaves();
    let refs: Vec<&[u8]> = leaves.iter().map(|l| l.as_slice()).collect();
    let msgs: Vec<Message<'_>> = refs.iter().map(|l| Message::prefixed(PREFIX, l)).collect();
    let total_bytes: usize = leaves.iter().map(|l| l.len() + PREFIX.len()).sum();

    println!(
        "{LEAVES} leaves, {total_bytes} bytes/batch, active backend: {}",
        tape_sha256::backend()
    );
    println!();

    let mut out = vec![[0u8; 32]; LEAVES];

    // The baseline: what validators do today, one hash call per leaf.
    let sha2_time = bench("sha2 (serial, current)", total_bytes, || {
        for (i, leaf) in refs.iter().enumerate() {
            let mut h = Sha256::new();
            h.update(PREFIX);
            h.update(leaf);
            out[i] = h.finalize().into();
        }
        std::hint::black_box(&out);
    });

    // Our own core at one lane, separating the multi-buffer win from the core
    // simply being faster or slower than sha2.
    bench("tape serial (1 lane)", total_bytes, || {
        backends::serial(&msgs, &mut out);
        std::hint::black_box(&out);
    });

    let portable = bench("tape portable (8 lane)", total_bytes, || {
        backends::portable8(&msgs, &mut out);
        std::hint::black_box(&out);
    });

    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    let neon = bench("tape neon (4 lane)", total_bytes, || {
        backends::neon4(&msgs, &mut out);
        std::hint::black_box(&out);
    });

    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    let neon_sha2 = if std::arch::is_aarch64_feature_detected!("sha2") {
        Some(bench("tape neon-sha2 (x4 stream)", total_bytes, || {
            unsafe { backends::neon_sha2x4(&msgs, &mut out) };
            std::hint::black_box(&out);
        }))
    } else {
        None
    };

    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    let shani = if is_x86_feature_detected!("sha") {
        Some(bench("tape shani (x4 stream)", total_bytes, || {
            unsafe { backends::shani_x4(&msgs, &mut out) };
            std::hint::black_box(&out);
        }))
    } else {
        None
    };

    // Reference harness for the SMT pattern documented in the crate docs: two
    // caller-owned threads on a sibling pair, half the batch each. Homogeneous
    // and 50/50, so there is nothing to balance and no straggler.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");

    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    let smt = if has_avx512 {
        let (ha, hb) = msgs.split_at(LEAVES / 2);
        Some(bench_smt2("tape smt pair (2x avx512)", total_bytes, ha, hb))
    } else {
        None
    };

    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    let avx2 = if is_x86_feature_detected!("avx2") {
        Some(bench("tape avx2 (8 lane)", total_bytes, || {
            unsafe { backends::avx2_8(&msgs, &mut out) };
            std::hint::black_box(&out);
        }))
    } else {
        None
    };

    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    let avx512 = if has_avx512 {
        Some(bench("tape avx512 (16 lane)", total_bytes, || {
            unsafe { backends::avx512_16(&msgs, &mut out) };
            std::hint::black_box(&out);
        }))
    } else {
        None
    };

    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    let avx512x2 = if has_avx512 {
        Some(bench("tape avx512 (2x16 interlace)", total_bytes, || {
            unsafe { backends::avx512_16x2(&msgs, &mut out) };
            std::hint::black_box(&out);
        }))
    } else {
        None
    };

    // Firedancer takes one contiguous (data, sz) per message, so the prefix
    // must be concatenated up front. That copy is hoisted out of the timed
    // region, so this measures its kernel and not memcpy. Our crate avoids the
    // copy entirely via `Message::prefixed`, so in the real Merkle use it
    // starts ahead by that much.
    #[cfg(feature = "firedancer-bench")]
    let (fd_avx_time, fd_avx512_time) = {
        use firedancer::*;
        let joined: Vec<Vec<u8>> = refs.iter().map(|leaf| [PREFIX, leaf].concat()).collect();

        // Submits every leaf through one Firedancer kernel, `lanes` at a time.
        macro_rules! run_fd {
            ($kernel:path, $lanes:expr) => {{
                let mut args = Args::default();
                for (chunk, hashes) in joined.chunks($lanes).zip(out.chunks_mut($lanes)) {
                    for (i, (msg, hash)) in chunk.iter().zip(hashes.iter_mut()).enumerate() {
                        args.data[i] = msg.as_ptr() as *const _;
                        args.sz[i] = msg.len() as u64;
                        args.hash[i] = hash.as_mut_ptr() as *mut _;
                    }
                    unsafe {
                        $kernel(
                            chunk.len() as u64,
                            args.data.as_ptr(),
                            args.sz.as_ptr(),
                            args.hash.as_ptr(),
                        );
                    }
                }
                std::hint::black_box(&out);
            }};
        }

        let avx = if is_x86_feature_detected!("avx2") {
            Some(bench("firedancer avx (8 lane)", total_bytes, || {
                run_fd!(fd_sha256_private_batch_avx, AVX_LANES)
            }))
        } else {
            None
        };
        #[cfg(feature = "firedancer-bench-avx512")]
        let avx512 = if is_x86_feature_detected!("avx512f") {
            Some(bench("firedancer avx512 (16 lane)", total_bytes, || {
                run_fd!(fd_sha256_private_batch_avx512, AVX512_LANES)
            }))
        } else {
            None
        };
        #[cfg(not(feature = "firedancer-bench-avx512"))]
        let avx512: Option<f64> = None;
        (avx, avx512)
    };
    #[cfg(not(feature = "firedancer-bench"))]
    let (fd_avx_time, fd_avx512_time): (Option<f64>, Option<f64>) = (None, None);

    // isa-l_crypto, same treatment as Firedancer: the prefix concat is hoisted
    // out of the timed region, so this measures its scheduler and kernels
    // rather than memcpy.
    #[cfg(feature = "isal-bench")]
    let (isal_avx512_time, isal_avx512_ni_time, isal_dispatch_time) = {
        let joined: Vec<Vec<u8>> = refs.iter().map(|leaf| [PREFIX, leaf].concat()).collect();
        let bufs: Vec<*const u8> = joined.iter().map(|m| m.as_ptr()).collect();
        let lens: Vec<u32> = joined.iter().map(|m| m.len() as u32).collect();

        let handle = unsafe { isal::isal_bench_new() };
        assert!(!handle.is_null(), "isal_bench_new failed");

        // A mis-wired FFI produces a fast wrong answer rather than a link
        // error, so every entry point is gated against sha2 before it is
        // timed. The digest byte order in particular is easy to get wrong:
        // isa-l leaves the words native-endian.
        let expect: Vec<[u8; 32]> = joined
            .iter()
            .map(|m| {
                let mut h = Sha256::new();
                h.update(m);
                h.finalize().into()
            })
            .collect();

        macro_rules! isal_row {
            ($name:expr, $f:path) => {{
                let mut got = vec![[0u8; 32]; LEAVES];
                let rc = unsafe {
                    $f(
                        handle,
                        LEAVES as u32,
                        bufs.as_ptr(),
                        lens.as_ptr(),
                        got.as_mut_ptr(),
                    )
                };
                if rc != 0 {
                    println!("{:<28}unavailable (rc {rc})", $name);
                    None
                } else if got != expect {
                    panic!("{} disagrees with sha2 -- FFI is wrong, not slow", $name);
                } else {
                    Some(bench($name, total_bytes, || {
                        let rc = unsafe {
                            $f(
                                handle,
                                LEAVES as u32,
                                bufs.as_ptr(),
                                lens.as_ptr(),
                                out.as_mut_ptr(),
                            )
                        };
                        debug_assert_eq!(rc, 0);
                        std::hint::black_box(&out);
                    }))
                }
            }};
        }

        let a = isal_row!("isa-l mb (x16 avx512)", isal::isal_bench_run_avx512);
        let b = isal_row!("isa-l mb (x16 avx512-ni)", isal::isal_bench_run_avx512_ni);
        let c = isal_row!("isa-l mb (dispatched)", isal::isal_bench_run_dispatch);

        unsafe { isal::isal_bench_free(handle) };
        (a, b, c)
    };
    #[cfg(not(feature = "isal-bench"))]
    let (isal_avx512_time, isal_avx512_ni_time, isal_dispatch_time): (
        Option<f64>,
        Option<f64>,
        Option<f64>,
    ) = (None, None, None);

    println!();
    println!("speedup vs sha2 serial baseline:");
    println!("  portable-8  {:.2}x", sha2_time / portable);
    match (fd_avx_time, fd_avx512_time) {
        (None, None) => println!(
            "  firedancer  not linked (build with --features firedancer-bench \
             and FD_LIB_DIR set)"
        ),
        (avx, avx512) => {
            if let Some(t) = avx {
                println!("  fd-avx-8    {:.2}x", sha2_time / t);
            }
            if let Some(t) = avx512 {
                println!("  fd-avx512   {:.2}x", sha2_time / t);
            }
        }
    }
    match (isal_avx512_time, isal_avx512_ni_time, isal_dispatch_time) {
        (None, None, None) => println!(
            "  isa-l       not linked (build with --features isal-bench \
             and ISAL_DIR set)"
        ),
        (a, b, c) => {
            if let Some(t) = a {
                println!("  isal-avx512 {:.2}x", sha2_time / t);
            }
            if let Some(t) = b {
                println!("  isal-av512ni{:.2}x", sha2_time / t);
            }
            if let Some(t) = c {
                println!("  isal-disp   {:.2}x", sha2_time / t);
            }
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    {
        println!("  neon-4      {:.2}x", sha2_time / neon);
        match neon_sha2 {
            Some(t) => println!("  neon-sha2   {:.2}x", sha2_time / t),
            None => println!("  neon-sha2   unavailable on this CPU"),
        }
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    {
        match shani {
            Some(t) => println!("  shani-x4    {:.2}x", sha2_time / t),
            None => println!("  shani-x4    unavailable on this CPU"),
        }
        if let Some(t) = smt {
            println!(
                "  smt-pair    {:.2}x  (2 threads, one physical core)",
                sha2_time / t
            );
        }
        match avx2 {
            Some(t) => println!("  avx2-8      {:.2}x", sha2_time / t),
            None => println!("  avx2-8      unavailable on this CPU"),
        }
        match avx512 {
            Some(t) => println!("  avx512-16   {:.2}x", sha2_time / t),
            None => println!("  avx512-16   unavailable on this CPU"),
        }
        if let Some(t) = avx512x2 {
            println!("  avx512-2x16 {:.2}x", sha2_time / t);
        }
        println!();
        println!(
            "note: sha2's x86 baseline uses SHA-NI when present. Check with \
             `lscpu | grep -o sha_ni` -- without it sha2 falls back to its \
             software path and these ratios will look far more favourable \
             than on a SHA-NI machine."
        );
    }
}
