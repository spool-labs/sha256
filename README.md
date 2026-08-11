# tape-sha256

[![Crates.io](https://img.shields.io/crates/v/tape-sha256.svg)](https://crates.io/crates/tape-sha256)
[![Documentation](https://docs.rs/tape-sha256/badge.svg)](https://docs.rs/tape-sha256)
[![License](https://img.shields.io/crates/l/tape-sha256.svg)](LICENSE)

Pure-Rust, hardware-accelerated SHA-256 for batch hashing and iterated hash
chains.

`tape-sha256` provides two kinds of hashing:

- **Batch hashing** processes many independent messages in parallel, one per
  SIMD lane. This is useful for Merkle trees, data verification, and other
  workloads that hash many inputs at once.
- **Hash chains** repeatedly hash a 32-byte digest, either as one serial chain
  or as many independent chains in parallel. This is useful for workloads such
  as Solana proof-of-history verification.

The crate automatically selects the best implementation available on the
running CPU, with portable fallbacks for unsupported targets.

## Installation

```toml
[dependencies]
tape-sha256 = "0.2"
```

## Batch hashing

Use `hash_many` to hash a group of independent messages:

```rust
use tape_sha256::hash_many;

let messages: Vec<&[u8]> = vec![b"one", b"two", b"three"];
let mut digests = vec![[0u8; 32]; messages.len()];

hash_many(&messages, &mut digests);
```

When every message shares a prefix, `hash_many_prefixed` hashes
`prefix || body` without allocating or copying the concatenated inputs:

```rust
use tape_sha256::hash_many_prefixed;

let leaves: Vec<&[u8]> = vec![&[1u8; 4096], &[2u8; 4096]];
let mut digests = vec![[0u8; 32]; leaves.len()];

hash_many_prefixed(b"LEAF", &leaves, &mut digests);
```

For Merkle-tree parent nodes, `hash_pairs` similarly hashes
`prefix || left || right` without constructing temporary buffers.

## Hash chains

Use `hash_chain` when each digest becomes the input to the next hash:

```rust
use tape_sha256::hash_chain;

let seed = [0u8; 32];
let end = hash_chain(&seed, 62_500);
```

When several chains are independent, `hash_chains` runs them in parallel:

```rust
use tape_sha256::hash_chains;

let seeds = [[1u8; 32], [2u8; 32]];
let lengths = [62_500, 62_500];
let mut ends = [[0u8; 32]; 2];

hash_chains(&seeds, &lengths, &mut ends);
```

## Hardware acceleration

At runtime, `tape-sha256` selects an implementation for the current platform,
including AVX-512, AVX2, SHA-NI, ARM SHA-2, NEON, WebAssembly SIMD, and portable
Rust backends.

The `scalar`, `avx2`, `avx512`, and `neon` Cargo features pin a backend at
compile time. A pinned backend must be supported by every CPU that runs the
binary; the default runtime selection is safer for general-purpose builds.

Use `backend()` and `lane_width()` to inspect the selected batch backend, or
`chain_backend()` to inspect the single-chain backend.

## Performance and correctness

See [BENCHMARKS.md](BENCHMARKS.md) for measurements, hardware-specific analysis,
and methodology. WebAssembly results are documented in
[benches/wasm/README.md](benches/wasm/README.md).

Every available backend is tested against the independent `sha2` crate across
message lengths covering block and padding boundaries. Run the test suite with:

```sh
cargo test --release
```

## License

Apache-2.0
