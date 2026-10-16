//! Hashes one tape blob commitment: the protocol's own Merkle shapes
//!
//! tape commits every erasure-coded blob to a Merkle root over its slice
//! group: up to 20 slices of up to 4 KiB, each leaf hashed as
//! sha256("LEAF" || slice), and pairs joined as
//! sha256("LEFT" || left || "RIGHT" || right). The slicer builds these
//! trees on every write, and the gateway and archive nodes rebuild them to
//! verify reads; tape-sdk does the same hashing with the serial sha2 crate
//! today, which is the baseline row here.
//!
//! The two layers are benched separately because their shapes differ: 20
//! leaves of 64 blocks each dominate the work, and the 73-byte pair joins
//! are the small-message regime. Run with `cargo bench --bench blob_group`.

use {
    sha2::{Digest, Sha256},
    std::time::Instant,
    tape_sha256::{hash_many, hash_many_prefixed},
};

const LEAF_LABEL: &[u8] = b"LEAF";
const LEFT_LABEL: &[u8] = b"LEFT";
const RIGHT_LABEL: &[u8] = b"RIGHT";

/// Firedancer's batch kernels, for the comparison rows; see merkle_leaves
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
        #[cfg(feature = "firedancer-bench-avx512")]
        pub fn fd_sha256_private_batch_avx512(
            batch_cnt: u64,
            batch_data: *const *const c_void,
            batch_sz: *const u64,
            batch_hash: *const *mut c_void,
        );
    }

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

/// tape_core::erasure::GROUP_SIZE, slices per erasure group
const GROUP_SIZE: usize = 20;
/// tape's MAX_n, the largest slice the slicer produces
const SLICE_BYTES: usize = 4096;
/// Pair joins in one 20-leaf tree of height 5, interior levels 10+5+3+2+1
const JOINS: usize = 21;

fn bench<F: FnMut()>(name: &str, bytes_per_iter: usize, mut f: F) -> f64 {
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
        "{name:<30}{:>9.2} us/group{:>10.0} MB/s   (spread {spread:>4.1}%)",
        best * 1e6,
        mbps
    );
    best
}

fn main() {
    let slices: Vec<Vec<u8>> = (0..GROUP_SIZE)
        .map(|i| {
            (0..SLICE_BYTES)
                .map(|j| (j as u8).wrapping_mul(31).wrapping_add(i as u8))
                .collect()
        })
        .collect();
    let refs: Vec<&[u8]> = slices.iter().map(|s| s.as_slice()).collect();
    let leaf_bytes = GROUP_SIZE * (LEAF_LABEL.len() + SLICE_BYTES);

    // Pair joins over placeholder digests; the join message is
    // LEFT || left || RIGHT || right, 73 bytes.
    let children: Vec<[u8; 32]> = (0..2 * JOINS).map(|i| [i as u8; 32]).collect();
    let joined: Vec<Vec<u8>> = children
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| [LEFT_LABEL, &p[0], RIGHT_LABEL, &p[1]].concat())
        .collect();
    let joined_refs: Vec<&[u8]> = joined.iter().map(|j| j.as_slice()).collect();
    let join_bytes: usize = joined.iter().map(|j| j.len()).sum();

    println!(
        "{GROUP_SIZE} slices x {SLICE_BYTES} B + {JOINS} pair joins, active backend: {}",
        tape_sha256::backend()
    );
    println!();

    let mut out = vec![[0u8; 32]; GROUP_SIZE.max(JOINS)];

    // What tape-sdk does today: one serial hash per leaf.
    let serial_leaves = bench("leaves sha2 (serial, current)", leaf_bytes, || {
        for (i, s) in refs.iter().enumerate() {
            let mut h = Sha256::new();
            h.update(LEAF_LABEL);
            h.update(s);
            out[i] = h.finalize().into();
        }
        std::hint::black_box(&out);
    });

    let batch_leaves = bench("leaves tape hash_many_prefixed", leaf_bytes, || {
        hash_many_prefixed(LEAF_LABEL, &refs, &mut out[..GROUP_SIZE]);
        std::hint::black_box(&out);
    });

    let serial_joins = bench("joins sha2 (serial, current)", join_bytes, || {
        for (i, j) in joined_refs.iter().enumerate() {
            let mut h = Sha256::new();
            h.update(j);
            out[i] = h.finalize().into();
        }
        std::hint::black_box(&out);
    });

    let batch_joins = bench("joins tape hash_many", join_bytes, || {
        hash_many(&joined_refs, &mut out[..JOINS]);
        std::hint::black_box(&out);
    });

    // Firedancer rows. Its batch API takes one contiguous buffer per
    // message, so prefix concatenation happens outside the timed region,
    // and its only tail option is a partial batch through the same wide
    // kernel, which is exactly the difference under test.
    #[cfg(feature = "firedancer-bench")]
    {
        use firedancer::*;

        let leaf_joined: Vec<Vec<u8>> = refs.iter().map(|s| [LEAF_LABEL, s].concat()).collect();

        macro_rules! run_fd {
            ($kernel:path, $lanes:expr, $msgs:expr) => {{
                let mut args = Args::default();
                for (chunk, hashes) in $msgs.chunks($lanes).zip(out.chunks_mut($lanes)) {
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

        if is_x86_feature_detected!("avx2") {
            bench("leaves fd avx (8 lane)", leaf_bytes, || {
                run_fd!(fd_sha256_private_batch_avx, AVX_LANES, leaf_joined)
            });
            bench("joins fd avx (8 lane)", join_bytes, || {
                run_fd!(fd_sha256_private_batch_avx, AVX_LANES, joined)
            });
        }
        #[cfg(feature = "firedancer-bench-avx512")]
        if is_x86_feature_detected!("avx512f") {
            bench("leaves fd avx512 (16 lane)", leaf_bytes, || {
                run_fd!(fd_sha256_private_batch_avx512, AVX512_LANES, leaf_joined)
            });
            bench("joins fd avx512 (16 lane)", join_bytes, || {
                run_fd!(fd_sha256_private_batch_avx512, AVX512_LANES, joined)
            });
        }
    }

    // The batch rows must agree with the serial ones before the ratios mean
    // anything.
    let mut want = vec![[0u8; 32]; GROUP_SIZE];
    for (i, s) in refs.iter().enumerate() {
        let mut h = Sha256::new();
        h.update(LEAF_LABEL);
        h.update(s);
        want[i] = h.finalize().into();
    }
    let mut got = vec![[0u8; 32]; GROUP_SIZE];
    hash_many_prefixed(LEAF_LABEL, &refs, &mut got);
    assert_eq!(got, want, "leaf batch diverges from serial");

    let mut want = vec![[0u8; 32]; JOINS];
    for (i, j) in joined_refs.iter().enumerate() {
        let mut h = Sha256::new();
        h.update(j);
        want[i] = h.finalize().into();
    }
    let mut got = vec![[0u8; 32]; JOINS];
    hash_many(&joined_refs, &mut got);
    assert_eq!(got, want, "join batch diverges from serial");

    println!();
    println!("leaf level speedup: {:.2}x", serial_leaves / batch_leaves);
    println!("join level speedup: {:.2}x", serial_joins / batch_joins);
    let serial_total = serial_leaves + serial_joins;
    let batch_total = batch_leaves + batch_joins;
    println!(
        "whole commitment:   {:.2}x  ({:.2} -> {:.2} us/group)",
        serial_total / batch_total,
        serial_total * 1e6,
        batch_total * 1e6
    );
}
