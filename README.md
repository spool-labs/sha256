# tape-sha256

[![Crates.io](https://img.shields.io/crates/v/tape-sha256.svg)](https://crates.io/crates/tape-sha256)
[![Documentation](https://docs.rs/tape-sha256/badge.svg)](https://docs.rs/tape-sha256)
[![License](https://img.shields.io/crates/l/tape-sha256.svg)](LICENSE)


Pure-Rust, hardware-accelerated **SHA-256** in two shapes: hash many
independent messages at once, one per SIMD lane, or run iterated hash chains
on the CPU's SHA-256 unit.

SHA-256 is inherently serial *within* one message, since each of the 64 rounds
depends on the last, so a single hash cannot be vectorised. But `N`
independent messages can run in lockstep, one per lane, turning every round
into an elementwise vector operation. The canonical workload is a Merkle
tree: every leaf is independent, so the whole bottom level hashes in one
pass.

```rust
use tape_sha256::hash_many_prefixed;

// tape's Merkle leaves: sha256("LEAF" || slice), one call per slice group
let slices: Vec<&[u8]> = vec![&[1u8; 4096], &[2u8; 4096]];
let mut out = vec![[0u8; 32]; slices.len()];
hash_many_prefixed(b"LEAF", &slices, &mut out);
```

`hash_many_prefixed` hashes `prefix || body` without ever materialising the
concatenation: the kernel reads straight out of both slices. For
domain-separated Merkle leaves, tape's and agave's alike, that skips a full
copy of every message per batch, an advantage baked into the API shape
rather than an optimisation flag.

## Hash chains

The other shape is the opposite one: a single message hashed back into itself,
over and over, as in Solana's proof of history. No vector width helps, because
link `i + 1` is link `i`'s digest. What is left to exploit is the message. It
is always the previous 32-byte digest, so the padded block is constant above
word 8, and a SHA-256 unit already holds its state in the word order the next
block wants. One compression per link is all that remains, which is the
silicon's latency floor.

```rust
use tape_sha256::{hash_chain, hash_chains};

let end = hash_chain(&[0u8; 32], 62_500); // one Solana tick

// Independent chains run in lockstep, one per lane.
let seeds = [[1u8; 32], [2u8; 32]];
let mut out = [[0u8; 32]; 2];
hash_chains(&seeds, &[62_500, 62_500], &mut out);
```

Chains stop being serial once there is more than one, and verification has
many: each entry publishes its ending hash, so every segment starts from a
known one and all of them can run at once. Nanoseconds per link, over 64
independent chains:

| chains per call | Zen 5 | Apple M-series |
|---|---:|---:|
| `sha2`, serial baseline | 45.4 | 27.7 \* |
| 1 (`hash_chain`) | 33.8 | 22.5 |
| 4 | 18.9 | 14.5 |
| 32 | **7.3** | **14.5** |

\* with `sha2`'s `asm` feature; without it that baseline is ~105 ns/link.

Zen 5 ends up **6.0x** under `sha2`. AArch64 saturates at 1.55x, because its
crypto extension has four streams and NEON integer lanes lose to them
outright. Below the break-even `hash_chains` steps down to the SHA unit's
streams and then to the serial chain, on thresholds `cargo bench --bench
poh_chain` measures rather than assumes.

## In tape

tape commits every erasure-coded blob to a Merkle root over its slice group:
up to 20 slices of up to 4 KiB, each leaf hashed as `sha256("LEAF" || slice)`
and pairs joined as `sha256("LEFT" || left || "RIGHT" || right)`. The slicer
builds that tree on every write, the gateway and archive nodes rebuild it to
verify reads, and tape-sdk exposes the same hashing natively, over FFI to
wasmtime hosts, and to browsers through wasm-bindgen.

The whole commitment measures **6.85x** on Apple silicon (133.1 -> 19.4
us/group, about 51,000 groups per second per core) and 1.74x on Zen 4.
`cargo bench --bench blob_group`.

## Backends

The best kernel for the running CPU is chosen at load time and falls back
gracefully:

| backend | selected when | lanes |
|---|---|---|
| `avx512-2x16` | AMD Zen 5 (two 16-lane waves interlaced) | 32 |
| `avx512-16` | other x86-64 with AVX-512F+BW | 16 |
| `shani-x4` | x86-64 with SHA-NI but no AVX-512, and AMD Zen 4/3 | 4 streams |
| `avx2-8` | x86-64 with AVX2 only | 8 |
| `neon-sha2-x4` | AArch64 with the SHA-256 crypto extension | 4 streams |
| `neon-4` | other AArch64 | 4 |
| `simd128-4` | wasm32 built with `-C target-feature=+simd128` | 4 |
| `portable-8` | other AArch64 without SIMD (`scalar` builds) | 8 |
| `portable-1` | everything else | 1 |

All SIMD backends share one compression function written against a lane trait;
a backend supplies only the word arithmetic. The `scalar`, `avx2`, `avx512`,
and `neon` features pin one kernel at build time instead of detecting; a pinned
kernel the CPU lacks will fault at runtime, so pin only what your whole fleet
supports. (Pinning `neon` selects the integer NEON kernel, not the faster
crypto extension.) The same table drives `hash_chains`, since a chain link is
one block compression; `hash_chain` is the exception, asking only whether a
SHA-256 unit exists.

## Benchmarks

Full numbers, per-hardware analysis, and methodology are in
[BENCHMARKS.md](BENCHMARKS.md) and the wasm/JS story in
`benches/wasm/README.md`; the chain tables come from
`cargo bench --bench poh_chain`. The batch
workload throughout is one Solana erasure batch of Merkle leaves (64 leaves,
~1 KB each, 67,680 bytes), measured against the serial `sha2` path and against
Firedancer's batch kernels built from source on the same machine.

![Batch SHA-256 throughput per core, MB/s: tape leads Firedancer on Zen 5 (8654 vs 5393), Zen 4 (3242 vs 3006) and Zen 3 (3264 vs 1476), sits level on Intel Granite Rapids (3923 vs 4006), and is the only batch kernel on Apple silicon at 4166; the serial sha2 baseline runs 610 to 2443](charts/speedup.svg)

| hardware | best backend | vs `sha2` | vs Firedancer |
|---|---|---|---|
| AMD Zen 5 | `avx512-2x16` | 3.5x | **60% faster** |
| AMD Zen 4 | `shani-x4` | 2.1x | **~8% faster** |
| AMD Zen 3 | `shani-x4` | 2.1x | **~2.2x faster** |
| Intel EMR / GNR | `avx512-16` | 2.0-2.1x | ~1% slower (cycles) |
| AArch64 | `neon-sha2-x4` | 6.8x | no comparison |

The per-family AMD dispatch is measurement, not theory: the 2x16 interlace
wins 38% over the single wave on Zen 5's native 512-bit datapath, measures
exactly neutral on double-pumped Zen 4, and is deliberately not enabled
anywhere it has not been benched. The Zen 5 column is one pinned invocation so
every bar shares a run; three reps put the interlace at 7.82-8.34 us/batch.

Integrated into agave, per core against the serial `sha2` path it uses today:
shred recovery is **14-21% faster end to end**, and 1.55x over its 7-thread
rayon pool on one thread; shred verify is **1.40x** across the whole receive
path on Zen 5, where the Merkle rebuild is 54% of the cost and batches 2.13x.
Whole-process gains depend on each path's SHA-256 share, and nothing is
claimed beyond the paths measured.

## Correctness

Output is bit-identical to any conforming SHA-256. Every backend is gated
differentially against the independent `sha2` crate across message lengths
spanning all block-boundary and padding edge cases, checked against every
other backend, and the error-prone shuffle machinery (the 8x8 and 16x16
transpose ladders, the `vpternlogd` control bytes) is validated against scalar
models of the Intel intrinsic semantics. The chain entry points are gated link
by link on top of that, ragged and zero-length groups included. See `tests/`.

A kernel only *executes* where its instructions exist, so a green run on one
architecture compiles the others but cannot exercise them. The suite has been
run to completion on AMD Zen 3, AMD Zen 4, AMD Zen 5, Intel Emerald Rapids,
Intel Granite Rapids, and Apple M-series, so every backend has been validated
on real silicon of the vendor it targets. Run `cargo test --release` on a
representative machine before trusting a new build there;
`tests/differential.rs` prints which kernels a run actually exercised.

## License

Apache-2.0
