//! SHA-256 in bulk: many independent messages at once, or iterated chains
//!
//! One hash cannot be vectorised, since each of the 64 rounds depends on the
//! last. Independent messages can, though: run them in lockstep, one per SIMD
//! lane, and every round becomes an elementwise vector operation.
//!
//! Worth reaching for when you have many messages and need none of them early.
//! The canonical case is a Merkle tree, where the whole bottom level can be
//! hashed in one pass.
//!
//! ```
//! use tape_sha256::hash_many;
//!
//! let msgs: Vec<&[u8]> = vec![b"one", b"two", b"three"];
//! let mut out = vec![[0u8; 32]; msgs.len()];
//! hash_many(&msgs, &mut out);
//! ```
//!
//! When every leaf shares a domain-separation prefix, hash_many_prefixed
//! avoids materialising `prefix || body` per message:
//!
//! ```
//! use tape_sha256::hash_many_prefixed;
//!
//! const LEAF: &[u8] = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
//! let leaves: Vec<&[u8]> = vec![&[1u8; 64], &[2u8; 64]];
//! let mut out = vec![[0u8; 32]; leaves.len()];
//! hash_many_prefixed(LEAF, &leaves, &mut out);
//! ```
//!
//! # Correctness
//!
//! Output is bit-identical to any conforming SHA-256. Every backend is gated
//! against the independent `sha2` crate over lengths covering all block and
//! padding edge cases, and each SIMD backend is cross-checked against the
//! portable one.
//!
//! # Backend selection
//!
//! By default the best kernel for the running CPU is picked and falls back
//! gracefully. The `scalar`, `avx2`, `avx512`, and `neon` features pin one
//! kernel at build time instead, skipping detection; a pinned kernel the CPU
//! lacks will fault, so pin only what the whole fleet supports.
//!
//! # Using both threads of a core
//!
//! These entry points are stateless, so two threads may call them concurrently
//! on disjoint halves of a batch. Doing that on the two SMT siblings of one
//! physical core measured 1.5x the single-thread rate on Zen 5 (9.83 vs 14.84
//! us for 64 Merkle leaves), because a single thread at ~2 instructions per
//! cycle is limited by its own dependency chains rather than by issue width,
//! and the sibling fills the gaps.
//!
//! This crate never spawns or pins a thread. Thread topology is the caller's
//! policy, not a hashing library's. Split the batch, hand each half to a thread
//! you have pinned, and join:
//!
//! ```no_run
//! use tape_sha256::hash_many;
//!
//! # let msgs: Vec<&[u8]> = vec![];
//! # let mut out = vec![[0u8; 32]; 0];
//! let (m0, m1) = msgs.split_at(msgs.len() / 2);
//! let (o0, o1) = out.split_at_mut(msgs.len() / 2);
//! std::thread::scope(|s| {
//!     s.spawn(|| hash_many(m0, o0));
//!     s.spawn(|| hash_many(m1, o1));
//! });
//! ```
//!
//! How much this is worth depends heavily on the microarchitecture.
//!
//! Three further caveats decide whether this is worth anything:
//!
//! The two threads must be siblings of the *same* physical core. Pinned to
//! different cores this is ordinary multicore parallelism, which spends a core
//! to get it; the 1.5x claim is specifically about using a sibling that would
//! otherwise idle. Sibling pairs are listed in
//! `/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list`. Getting the
//! pairing wrong degrades silently, so it is worth asserting.
//!
//! The sibling has to actually be idle. Under a fully loaded process there is
//! no spare thread to claim and the gain goes away. Use an equal split of the
//! same kernel: an uneven or mixed-kernel split leaves a straggler and measured
//! worse than a single thread.
//!
//! # Hash chains
//!
//! Everything above is one pass over messages that already exist. A chain
//! instead feeds each digest back in as the next message, which is what
//! Solana's proof of history is.
//!
//! [`hash_chain`] is the exception to the whole premise of this crate: one
//! chain has no independent work to lane up, so it ignores the multi-buffer
//! kernels and drives the CPU's SHA-256 unit at its latency floor.
//!
//! [`hash_chains`] puts the premise back. Independent chains, one per
//! verified entry, say run in lockstep exactly like independent messages
//! do, which trades that latency floor for a throughput one. See both.

#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
mod avx2;
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
mod avx512;
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
mod avx512x2;
mod batch;
mod chain;
mod core;
mod lanes;
#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
mod neon;
#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
mod neon_sha2;
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
mod shani;
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(feature = "scalar")
))]
mod simd128;

pub use batch::Message;

use {
    batch::{drive, drive_pairs, drive_slices},
    lanes::Scalar,
};

/// Number of messages the active backend hashes per pass
///
/// Batches at least this large amortise the transpose fully; smaller ones
/// still work but leave lanes idle.
pub fn lane_width() -> usize {
    dispatch::lane_width()
}

/// Name of the active backend, for logging and benchmarks
pub fn backend() -> &'static str {
    dispatch::backend()
}

/// Hashes every message in `msgs`, writing one digest per message to `out`
///
/// # Panics
///
/// Panics if `msgs.len() != out.len()`.
pub fn hash_many(msgs: &[&[u8]], out: &mut [[u8; 32]]) {
    dispatch::hash_slices(&[], msgs, out);
}

/// Hashes `prefix || body` for each body in `bodies`
///
/// Equivalent to concatenating and calling hash_many, but without building the
/// concatenation.
///
/// # Panics
///
/// Panics if `bodies.len() != out.len()`.
pub fn hash_many_prefixed(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
    dispatch::hash_slices(prefix, bodies, out);
}

/// Hashes `prefix || left || right` for each pair, without joining them first
///
/// For Merkle interior nodes, where a parent hashes a domain prefix and two
/// child hashes held in separate buffers. Saves the caller a staging buffer
/// and the join.
///
/// # Panics
///
/// Panics unless `left.len() == right.len() == out.len()`.
pub fn hash_pairs(prefix: &[u8], left: &[&[u8]], right: &[&[u8]], out: &mut [[u8; 32]]) {
    dispatch::hash_pairs(prefix, left, right, out);
}

/// Hashes pre-built messages, for callers that want per-message prefixes
///
/// # Panics
///
/// Panics if `msgs.len() != out.len()`.
pub fn hash_messages(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
    dispatch::hash(msgs, out);
}

/// Hashes `seed`, then that digest, and so on `n` times
///
/// The message is always 32 bytes, which makes the padded block constant above
/// word 8 and lets the state feed back without the byte swap a general
/// `sha256(digest)` would pay twice per link.
///
/// Returns `seed` unchanged when `n` is 0.
///
/// ```
/// use tape_sha256::hash_chain;
///
/// let seed = [0u8; 32];
/// assert_eq!(hash_chain(&seed, 3), hash_chain(&hash_chain(&seed, 1), 2));
/// ```
pub fn hash_chain(seed: &[u8; 32], n: u64) -> [u8; 32] {
    chain::hash_chain(seed, n)
}

/// Hashes many independent chains at once, one per lane
///
/// `out[i]` is `seeds[i]` hashed back into itself `lens[i]` times, the same
/// answer `hash_chain` gives, but the chains run in lockstep so the SHA-256
/// unit is throughput-bound instead of latency-bound.
///
/// A single chain cannot go faster than one block compression's latency, and
/// `hash_chain` already sits there. Independent chains are a different
/// problem, and Solana replay has them: every `Entry` publishes its ending
/// hash, so entry `i`'s segment starts from entry `i - 1`'s known hash and all
/// segments can be verified at once. Ragged lengths are scheduled, not paid
/// for: chains run longest-first and a lane that finishes picks up the next
/// chain, so a batch costs about `lens.iter().sum() / width` links, however
/// the lengths fall.
///
/// Uses `backend`'s kernel, not `chain_backend`'s.
///
/// # Panics
///
/// Panics unless `seeds.len() == lens.len() == out.len()`.
///
/// ```
/// use tape_sha256::{hash_chain, hash_chains};
///
/// let seeds = [[1u8; 32], [2u8; 32]];
/// let lens = [7u64, 9];
/// let mut out = [[0u8; 32]; 2];
/// hash_chains(&seeds, &lens, &mut out);
/// assert_eq!(out[0], hash_chain(&seeds[0], 7));
/// assert_eq!(out[1], hash_chain(&seeds[1], 9));
/// ```
pub fn hash_chains(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
    assert_eq!(seeds.len(), lens.len());
    assert_eq!(seeds.len(), out.len());
    chain::hash_chains(seeds, lens, out);
}

/// Name of the kernel `hash_chain` uses
///
/// Not the same choice as `backend`: the chain cares only whether a SHA-256
/// unit exists, never how wide the vector unit is.
pub fn chain_backend() -> &'static str {
    chain::backend()
}

/// Backends exposed for differential testing and benchmarking
///
/// Not stable API; callers should use hash_many and let dispatch pick. Each
/// dispatchable backend has two entry points: one taking pre-built messages,
/// and a `_slices` form that builds them on the stack so the wrappers need no
/// allocation. Both `serial` and `portable8` are dispatchable — which one the
/// no-SIMD fallback picks is per-architecture, see `dispatch::PORTABLE`.
#[doc(hidden)]
pub mod backends {
    use super::*;

    // Concrete scalar group functions; see GroupFn for why kernels are
    // instantiated once here and called through pointers.
    pub(crate) fn scalar1(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        batch::hash_lanes::<Scalar<1>, 1>(msgs, out)
    }
    pub(crate) fn scalar8(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        batch::hash_lanes::<Scalar<8>, 8>(msgs, out)
    }
    /// Portable 8-lane kernel
    ///
    /// The reference every SIMD backend is checked against, and the no-SIMD
    /// fallback on register-rich targets.
    pub fn portable8(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        unsafe { drive(8, scalar8, msgs, out) }
    }

    pub fn portable8_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        unsafe { drive_slices(8, scalar8, prefix, bodies, out) }
    }

    /// Single-lane kernel: ordinary serial SHA-256 through the same core
    ///
    /// Shows the lane machinery is not what makes output correct, and is the
    /// baseline speedups are quoted against.
    pub fn serial(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        unsafe { drive(1, scalar1, msgs, out) }
    }

    pub fn serial_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        unsafe { drive_slices(1, scalar1, prefix, bodies, out) }
    }

    /// 4-lane AArch64 NEON kernel
    ///
    /// NEON is baseline on AArch64, so this is always safe to call there.
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    pub fn neon4(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        unsafe { drive(4, crate::neon::group, msgs, out) }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    pub fn neon4_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        unsafe { drive_slices(4, crate::neon::group, prefix, bodies, out) }
    }

    /// 4-lane wasm simd128 kernel
    ///
    /// Exists only in builds with `-C target-feature=+simd128`; an engine
    /// that lacks simd128 rejects the module at instantiation, so a call
    /// that runs at all is safe.
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "scalar")
    ))]
    pub fn simd128_4(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        unsafe { drive(4, crate::simd128::group, msgs, out) }
    }

    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "scalar")
    ))]
    pub fn simd128_4_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        unsafe { drive_slices(4, crate::simd128::group, prefix, bodies, out) }
    }

    /// Two-wave 8-lane wasm simd128 kernel, for benchmarking against the
    /// single wave; never dispatched to
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "scalar")
    ))]
    pub fn simd128_8_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        unsafe { drive_slices(8, crate::simd128::group8, prefix, bodies, out) }
    }

    /// Four-wave 16-lane wasm simd128 kernel; never dispatched to
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "scalar")
    ))]
    pub fn simd128_16_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        unsafe { drive_slices(16, crate::simd128::group16, prefix, bodies, out) }
    }

    /// Multi-stream kernel on the ARMv8 SHA-256 crypto extension
    ///
    /// # Safety
    ///
    /// The running CPU must support the `sha2` extension.
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    pub unsafe fn neon_sha2x4(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        drive(
            crate::neon_sha2::STREAMS,
            crate::neon_sha2::group,
            msgs,
            out,
        )
    }

    /// # Safety
    ///
    /// The running CPU must support the `sha2` extension.
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    pub unsafe fn neon_sha2x4_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        drive_slices(
            crate::neon_sha2::STREAMS,
            crate::neon_sha2::group,
            prefix,
            bodies,
            out,
        )
    }

    /// Multi-stream kernel on the x86 SHA-NI extension
    ///
    /// # Safety
    ///
    /// The running CPU must support SHA-NI, SSSE3, and SSE4.1.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn shani_x4(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        drive(crate::shani::STREAMS, crate::shani::group, msgs, out)
    }

    /// # Safety
    ///
    /// The running CPU must support SHA-NI, SSSE3, and SSE4.1.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn shani_x4_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        drive_slices(
            crate::shani::STREAMS,
            crate::shani::group,
            prefix,
            bodies,
            out,
        )
    }

    /// 8-lane AVX2 kernel
    ///
    /// # Safety
    ///
    /// The running CPU must support AVX2.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn avx2_8(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        drive(8, crate::avx2::group, msgs, out)
    }

    /// # Safety
    ///
    /// The running CPU must support AVX2.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn avx2_8_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        drive_slices(8, crate::avx2::group, prefix, bodies, out)
    }

    /// 16-lane AVX-512 kernel
    ///
    /// # Safety
    ///
    /// The running CPU must support AVX-512F and AVX-512BW.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn avx512_16(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        drive(16, crate::avx512::group, msgs, out)
    }

    /// # Safety
    ///
    /// The running CPU must support AVX-512F and AVX-512BW.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn avx512_16_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        drive_slices(16, crate::avx512::group, prefix, bodies, out)
    }

    /// 32-message AVX-512 interlace: two 16-lane compressions with their
    /// rounds interlaced in one thread
    ///
    /// # Safety
    ///
    /// The running CPU must support AVX-512F and AVX-512BW.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn avx512_16x2(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        drive(crate::avx512x2::WIDTH, crate::avx512x2::group2, msgs, out)
    }

    /// # Safety
    ///
    /// The running CPU must support AVX-512F and AVX-512BW.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn avx512_16x2_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        drive_slices(
            crate::avx512x2::WIDTH,
            crate::avx512x2::group2,
            prefix,
            bodies,
            out,
        )
    }

    /// Fallback where no SHA-256 unit exists
    pub fn chain_portable(seed: &[u8; 32], n: u64) -> [u8; 32] {
        crate::chain::portable(seed, n)
    }

    /// Chain on the x86 SHA-NI extension
    ///
    /// # Safety
    ///
    /// The running CPU must support SHA-NI, SSSE3, and SSE4.1.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    pub unsafe fn chain_shani(seed: &[u8; 32], n: u64) -> [u8; 32] {
        crate::shani::chain(seed, n)
    }

    /// Chain on the ARMv8 SHA-256 crypto extension
    ///
    /// # Safety
    ///
    /// The running CPU must support the `sha2` extension.
    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    pub unsafe fn chain_neon_sha2(seed: &[u8; 32], n: u64) -> [u8; 32] {
        crate::neon_sha2::chain(seed, n)
    }

    /// Lockstep chain kernels, one entry per dispatchable backend
    ///
    /// Each pins one step kernel under the shared scheduler, so any number of
    /// chains streams through that kernel's width. `hash_chains` picks between
    /// them and the serial chain; these exist so the choice can be measured
    /// rather than asserted.
    pub mod chains {
        use crate::chain::run_scheduled;

        pub fn portable1(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            // SAFETY: the scalar steps need no CPU feature.
            unsafe { run_scheduled(1, crate::chain::steps_scalar1, seeds, lens, out) }
        }
        pub fn portable8(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            // SAFETY: as above.
            unsafe { run_scheduled(8, crate::chain::steps_scalar8, seeds, lens, out) }
        }

        /// # Safety
        ///
        /// The running CPU must support the `sha2` extension.
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        pub unsafe fn neon_sha2x4(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            run_scheduled(4, crate::neon_sha2::steps4, seeds, lens, out)
        }

        /// # Safety
        ///
        /// The running CPU must support the `sha2` extension.
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        pub unsafe fn neon_sha2x8(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            run_scheduled(8, crate::neon_sha2::steps8, seeds, lens, out)
        }

        /// # Safety
        ///
        /// NEON is baseline on AArch64.
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        pub unsafe fn neon4(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            run_scheduled(4, crate::neon::steps, seeds, lens, out)
        }

        /// # Safety
        ///
        /// The running CPU must support SHA-NI, SSSE3, and SSE4.1.
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        pub unsafe fn shani_x4(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            run_scheduled(4, crate::shani::steps4, seeds, lens, out)
        }

        /// # Safety
        ///
        /// The running CPU must support AVX2.
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        pub unsafe fn avx2_8(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            run_scheduled(8, crate::avx2::steps, seeds, lens, out)
        }

        /// # Safety
        ///
        /// The running CPU must support AVX-512F and AVX-512BW.
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        pub unsafe fn avx512_16(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            run_scheduled(16, crate::avx512::steps, seeds, lens, out)
        }

        /// # Safety
        ///
        /// The running CPU must support AVX-512F and AVX-512BW.
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        pub unsafe fn avx512_16x2(seeds: &[[u8; 32]], lens: &[u64], out: &mut [[u8; 32]]) {
            run_scheduled(32, crate::avx512x2::steps2, seeds, lens, out)
        }
    }
}

mod dispatch {
    use super::*;

    /// The kernel to fall back on where no SIMD exists
    ///
    /// Multi-buffer costs a real general-purpose register per lane in scalar
    /// code rather than riding along free in a vector lane, so the width that
    /// wins is set by the register file. x86-64 has 16 GPRs and cannot absorb
    /// even two lanes: eight measured 1.9x slower than one on Zen 4 (418.78 vs
    /// 218.30 us). aarch64 has 31, and eight lanes measure 13% faster than one
    /// (93.7 vs 106.5).
    ///
    /// One lane is the default because it is the safe side of that trade: it
    /// gives up ~13% where the registers exist, while the wide kernel loses
    /// 1.9x where they do not, and most targets are nearer x86-64 than aarch64
    /// (32-bit x86 has 8 GPRs; wasm is JIT-ed onto whatever the host has).
    /// aarch64 is listed because it was measured, not because it is 64-bit.
    ///
    /// Unreachable on an aarch64 build without `scalar`, since NEON is
    /// baseline there and nothing falls through to it.
    #[allow(dead_code)]
    const PORTABLE: Kernel = if cfg!(target_arch = "aarch64") {
        Kernel::Portable8
    } else {
        Kernel::Portable1
    };

    /// Which kernel this build and CPU resolve to
    ///
    /// Entry points, reported name, and lane width all read from a single
    /// select, so they cannot drift. An earlier version kept two detection
    /// ladders and derived the width by string-matching the backend name,
    /// where a rename would silently have changed the reported width.
    // Which variants exist depends on target and features; the rest are only
    // match arms, which dead_code would otherwise flag.
    #[allow(dead_code)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Kernel {
        Portable1,
        Portable8,
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        Neon4,
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        NeonSha2x4,
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        ShaNiX4,
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Avx2_8,
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Avx512_16,
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        Avx512_16x2,
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            not(feature = "scalar")
        ))]
        Simd128_4,
    }

    impl Kernel {
        pub(crate) fn name(self) -> &'static str {
            match self {
                Kernel::Portable1 => "portable-1",
                Kernel::Portable8 => "portable-8",
                #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
                Kernel::Neon4 => "neon-4",
                #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
                Kernel::NeonSha2x4 => "neon-sha2-x4",
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::ShaNiX4 => "shani-x4",
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::Avx2_8 => "avx2-8",
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::Avx512_16 => "avx512-16",
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::Avx512_16x2 => "avx512-2x16",
                #[cfg(all(
                    target_arch = "wasm32",
                    target_feature = "simd128",
                    not(feature = "scalar")
                ))]
                Kernel::Simd128_4 => "simd128-4",
            }
        }

        pub(crate) fn width(self) -> usize {
            match self {
                Kernel::Portable1 => 1,
                Kernel::Portable8 => 8,
                #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
                Kernel::Neon4 => 4,
                #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
                Kernel::NeonSha2x4 => crate::neon_sha2::STREAMS,
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::ShaNiX4 => crate::shani::STREAMS,
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::Avx2_8 => 8,
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::Avx512_16 => 16,
                #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
                Kernel::Avx512_16x2 => crate::avx512x2::WIDTH,
                #[cfg(all(
                    target_arch = "wasm32",
                    target_feature = "simd128",
                    not(feature = "scalar")
                ))]
                Kernel::Simd128_4 => 4,
            }
        }
    }

    /// Effective AMD CPU family, or 0 on anything else
    ///
    /// 0x19 is Zen 3 / Zen 4, 0x1a is Zen 5; the AVX-512 dispatch below is
    /// per-family because the kernels rank differently on each. Cached:
    /// `select` runs on every public call, and two `cpuid`s per call would
    /// be measurable against a single small message.
    #[cfg(all(
        target_arch = "x86_64",
        not(any(feature = "scalar", feature = "avx2", feature = "avx512"))
    ))]
    fn amd_family() -> u32 {
        use std::sync::OnceLock;
        static FAMILY: OnceLock<u32> = OnceLock::new();
        *FAMILY.get_or_init(|| {
            use std::arch::x86_64::__cpuid;
            // SAFETY: leaves 0 and 1 exist on every x86_64 CPU. The blocks
            // are redundant from 1.95, where `__cpuid` became safe, but the
            // MSRV toolchain still requires them.
            #[allow(unused_unsafe)]
            let v = unsafe { __cpuid(0) };
            // "AuthenticAMD" in the ebx/edx/ecx registers.
            if (v.ebx, v.edx, v.ecx) != (0x6874_7541, 0x6974_6e65, 0x444d_4163) {
                return 0;
            }
            #[allow(unused_unsafe)]
            let eax = unsafe { __cpuid(1) }.eax;
            // AMD reports base family 0xf and puts the rest in the extended
            // field, so the effective family is their sum.
            ((eax >> 8) & 0xf) + ((eax >> 20) & 0xff)
        })
    }

    /// Selects the best kernel for this target and CPU
    #[inline]
    #[allow(clippy::needless_return)]
    pub(crate) fn select() -> Kernel {
        #[cfg(feature = "scalar")]
        return PORTABLE;

        #[cfg(all(target_arch = "x86_64", feature = "avx512", not(feature = "scalar")))]
        return Kernel::Avx512_16;
        #[cfg(all(target_arch = "x86_64", feature = "avx2", not(feature = "scalar")))]
        return Kernel::Avx2_8;
        #[cfg(all(target_arch = "aarch64", feature = "neon", not(feature = "scalar")))]
        return Kernel::Neon4;

        #[cfg(not(any(
            feature = "scalar",
            all(target_arch = "x86_64", any(feature = "avx2", feature = "avx512")),
            all(target_arch = "aarch64", feature = "neon"),
        )))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                use std::sync::OnceLock;
                // Each feature probe is itself a cached atomic load, but the
                // ladder runs on every public call and stacks half a dozen
                // of them; resolve the kernel once instead.
                static KERNEL: OnceLock<Kernel> = OnceLock::new();
                return *KERNEL.get_or_init(|| {
                    let sha = have_shani();
                    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                        // AVX-512 capability alone does not decide; the family
                        // does, and each arm here is a measurement. Zen 5 runs
                        // the 2x16 interlace (12.7 vs 14.6 us single-wave on
                        // EPYC 9B45): its native 512-bit datapath leaves
                        // dependency stalls the second wave fills.
                        //
                        // Zen 4 double-pumps 512-bit ops, so the interlace measured
                        // exactly neutral there (23.69 vs 23.72 on EPYC 9B14)
                        // and the SHA-NI streams beat every 16-lane kernel
                        // instead (21.05).
                        //
                        // Intel keeps the single wave for a different reason:
                        // the interlace wins the cycles (998 vs 1053 per block
                        // step on Emerald Rapids, 1004 vs 1048 on Granite Rapids)
                        // and loses the clock, since doubling 512-bit density
                        // costs 7 to 11% of frequency.
                        let fam = amd_family();
                        if fam == 0x1a {
                            return Kernel::Avx512_16x2;
                        }
                        if !(sha && fam == 0x19) {
                            return Kernel::Avx512_16;
                        }
                    }
                    // Without AVX-512 (or on Zen 4), the dedicated SHA unit beats
                    // 8-lane integer multi-buffer by a wide margin.
                    // Measured on Zen 3, AVX2 loses even to serial SHA-NI.
                    if sha {
                        return Kernel::ShaNiX4;
                    }
                    if is_x86_feature_detected!("avx2") {
                        return Kernel::Avx2_8;
                    }
                    PORTABLE
                });
            }
            #[cfg(target_arch = "aarch64")]
            {
                // The dedicated SHA-256 unit beats integer multi-buffer, so
                // prefer it and fall back to NEON lanes.
                if std::arch::is_aarch64_feature_detected!("sha2") {
                    return Kernel::NeonSha2x4;
                }
                Kernel::Neon4
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                // Wasm has no runtime probing; simd128 is a build-time fact.
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    Kernel::Simd128_4
                }
                #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
                {
                    PORTABLE
                }
            }
        }
    }

    pub(crate) fn hash(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
        match select() {
            Kernel::Portable1 => backends::serial(msgs, out),
            Kernel::Portable8 => backends::portable8(msgs, out),
            #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
            Kernel::Neon4 => backends::neon4(msgs, out),
            #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
            Kernel::NeonSha2x4 => unsafe { backends::neon_sha2x4(msgs, out) },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            Kernel::ShaNiX4 => unsafe { backends::shani_x4(msgs, out) },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            Kernel::Avx2_8 => unsafe { backends::avx2_8(msgs, out) },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            k @ (Kernel::Avx512_16 | Kernel::Avx512_16x2) => unsafe {
                let (w, group) = avx512_parts(k);
                let cut = avx512_tail_cut(msgs.len());
                let (head, tail) = msgs.split_at(cut);
                let (out_head, out_tail) = out.split_at_mut(cut);
                drive(w, group, head, out_head);
                if !tail.is_empty() {
                    backends::shani_x4(tail, out_tail);
                }
            },
            #[cfg(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                not(feature = "scalar")
            ))]
            Kernel::Simd128_4 => backends::simd128_4(msgs, out),
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    #[inline]
    pub(crate) fn have_shani() -> bool {
        is_x86_feature_detected!("sha")
            && is_x86_feature_detected!("ssse3")
            && is_x86_feature_detected!("sse4.1")
    }

    /// Where to split an avx512 batch so its remainder runs on SHA-NI
    ///
    /// A remainder under 13 messages costs a flat 16-lane group mostly
    /// running empty lanes, while SHA-NI streams pay only for the messages
    /// present; at 13 or more the flat group is cheaper again. Routed when
    /// the CPU has SHA-NI. Returns the batch length when no split should
    /// happen.
    ///
    /// The cut is at wave (16-lane) granularity for both AVX-512 kernels:
    /// the interlace processes 32 per chunk but degrades internally to
    /// 16-lane groups for a partial chunk, so its routable straggler is
    /// also `len % 16` -- cutting at `len % 32` would strand a 17..=28
    /// remainder as 16 plus the exact mostly-empty group this exists to
    /// avoid.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    #[inline]
    fn avx512_tail_cut(len: usize) -> usize {
        let rem = len % 16;
        if (1..=12).contains(&rem) && have_shani() {
            len - rem
        } else {
            len
        }
    }

    /// Chunk width and group function for an AVX-512 kernel choice
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    fn avx512_parts(k: Kernel) -> (usize, batch::GroupFn) {
        match k {
            Kernel::Avx512_16x2 => (
                crate::avx512x2::WIDTH,
                crate::avx512x2::group2 as batch::GroupFn,
            ),
            _ => (16, crate::avx512::group as batch::GroupFn),
        }
    }

    pub(crate) fn hash_slices(prefix: &[u8], bodies: &[&[u8]], out: &mut [[u8; 32]]) {
        match select() {
            Kernel::Portable1 => backends::serial_slices(prefix, bodies, out),
            Kernel::Portable8 => backends::portable8_slices(prefix, bodies, out),
            #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
            Kernel::Neon4 => backends::neon4_slices(prefix, bodies, out),
            #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
            Kernel::NeonSha2x4 => unsafe { backends::neon_sha2x4_slices(prefix, bodies, out) },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            Kernel::ShaNiX4 => unsafe { backends::shani_x4_slices(prefix, bodies, out) },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            Kernel::Avx2_8 => unsafe { backends::avx2_8_slices(prefix, bodies, out) },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            k @ (Kernel::Avx512_16 | Kernel::Avx512_16x2) => unsafe {
                let (w, group) = avx512_parts(k);
                let cut = avx512_tail_cut(bodies.len());
                let (head, tail) = bodies.split_at(cut);
                let (out_head, out_tail) = out.split_at_mut(cut);
                drive_slices(w, group, prefix, head, out_head);
                if !tail.is_empty() {
                    backends::shani_x4_slices(prefix, tail, out_tail);
                }
            },
            #[cfg(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                not(feature = "scalar")
            ))]
            Kernel::Simd128_4 => backends::simd128_4_slices(prefix, bodies, out),
        }
    }

    pub(crate) fn hash_pairs(prefix: &[u8], left: &[&[u8]], right: &[&[u8]], out: &mut [[u8; 32]]) {
        let k = select();
        let width = k.width();
        match k {
            Kernel::Portable1 => unsafe {
                drive_pairs(width, backends::scalar1, prefix, left, right, out)
            },
            Kernel::Portable8 => unsafe {
                drive_pairs(width, backends::scalar8, prefix, left, right, out)
            },
            #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
            Kernel::Neon4 => unsafe {
                drive_pairs(width, crate::neon::group, prefix, left, right, out)
            },
            #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
            Kernel::NeonSha2x4 => unsafe {
                drive_pairs(width, crate::neon_sha2::group, prefix, left, right, out)
            },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            Kernel::ShaNiX4 => unsafe {
                drive_pairs(width, crate::shani::group, prefix, left, right, out)
            },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            Kernel::Avx2_8 => unsafe {
                drive_pairs(width, crate::avx2::group, prefix, left, right, out)
            },
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            k @ (Kernel::Avx512_16 | Kernel::Avx512_16x2) => unsafe {
                let (w, group) = avx512_parts(k);
                let cut = avx512_tail_cut(left.len());
                drive_pairs(
                    w,
                    group,
                    prefix,
                    &left[..cut],
                    &right[..cut],
                    &mut out[..cut],
                );
                if cut < left.len() {
                    drive_pairs(
                        crate::shani::STREAMS,
                        crate::shani::group,
                        prefix,
                        &left[cut..],
                        &right[cut..],
                        &mut out[cut..],
                    );
                }
            },
            #[cfg(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                not(feature = "scalar")
            ))]
            Kernel::Simd128_4 => unsafe {
                drive_pairs(width, crate::simd128::group, prefix, left, right, out)
            },
        }
    }

    pub(crate) fn lane_width() -> usize {
        select().width()
    }

    pub(crate) fn backend() -> &'static str {
        select().name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_message() {
        let mut out = [[0u8; 32]; 1];
        hash_many(&[b""], &mut out);
        // FIPS 180-4 known answer for the empty string.
        let expect = hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(out[0], expect);
    }

    #[test]
    fn abc() {
        let mut out = [[0u8; 32]; 1];
        hash_many(&[b"abc"], &mut out);
        let expect = hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(out[0], expect);
    }

    #[test]
    fn prefix_matches_concatenation() {
        let prefix = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
        let bodies: Vec<Vec<u8>> = (0..8usize).map(|i| vec![i as u8; 100 + i * 37]).collect();
        let body_refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();

        let mut via_prefix = vec![[0u8; 32]; bodies.len()];
        hash_many_prefixed(prefix, &body_refs, &mut via_prefix);

        let joined: Vec<Vec<u8>> = bodies
            .iter()
            .map(|b| [prefix.as_slice(), b].concat())
            .collect();
        let joined_refs: Vec<&[u8]> = joined.iter().map(|b| b.as_slice()).collect();
        let mut via_concat = vec![[0u8; 32]; bodies.len()];
        hash_many(&joined_refs, &mut via_concat);

        assert_eq!(via_prefix, via_concat);
    }

    fn hex(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }
}
