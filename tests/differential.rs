//! Correctness gate: every backend must match an independent SHA-256.

use {
    rand::{rngs::StdRng, Rng, SeedableRng},
    sha2::{Digest, Sha256},
    tape_sha256::{hash_many, hash_many_prefixed, hash_messages, Message},
};

fn reference(msg: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(msg);
    h.finalize().into()
}

/// Lengths bracketing every padding and block-boundary transition
///
/// Empty, tiny, either side of each 64-byte block, and either side of 55/56
/// where the 9 padding bytes force an extra block.
fn edge_lengths() -> Vec<usize> {
    let mut v = vec![0usize, 1, 2, 3];
    for block in 0..6usize {
        let b = block * 64;
        for d in [53, 54, 55, 56, 57, 62, 63, 64, 65, 66] {
            v.push(b + d);
        }
    }
    // Around the real Merkle-leaf sizes this crate was built for.
    v.extend([955, 980, 1019, 1044, 1045, 1070]);
    v.sort_unstable();
    v.dedup();
    v
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

#[test]
fn single_message_every_edge_length() {
    for len in edge_lengths() {
        let msg = pattern(len, 7);
        let mut got = [[0u8; 32]; 1];
        hash_many(&[&msg], &mut got);
        assert_eq!(got[0], reference(&msg), "length {len}");
    }
}

/// A full batch of equal lengths, driving the uniform fast path
#[test]
fn uniform_batch_every_edge_length() {
    for len in edge_lengths() {
        let bodies: Vec<Vec<u8>> = (0..8).map(|i| pattern(len, i as u8)).collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; bodies.len()];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "length {len} lane {i}");
        }
    }
}

/// Lanes of differing lengths in one batch
///
/// Early finishers must have their digest snapped off while other lanes keep
/// going, so this is the likeliest place to capture a lane at the wrong round.
#[test]
fn ragged_batch_lengths() {
    let lens = edge_lengths();
    for window in lens.windows(8) {
        let bodies: Vec<Vec<u8>> = window
            .iter()
            .enumerate()
            .map(|(i, &l)| pattern(l, i as u8))
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; bodies.len()];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "lens {window:?} lane {i}");
        }
    }
}

/// Every batch size up to several lane widths, covering partial final batches
/// and inactive lanes
#[test]
fn every_batch_size() {
    for count in 0..40usize {
        let bodies: Vec<Vec<u8>> = (0..count).map(|i| pattern(100 + i * 13, i as u8)).collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; count];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "count {count} lane {i}");
        }
    }
}

/// The prefixed form must equal hashing the concatenation, including when the
/// prefix alone spans a block and when it is empty
#[test]
fn prefixed_matches_concatenation() {
    let prefixes: Vec<Vec<u8>> = vec![
        vec![],
        b"\x00SOLANA_MERKLE_SHREDS_LEAF".to_vec(),
        vec![0xab; 63],
        vec![0xcd; 64],
        vec![0xef; 65],
    ];
    for prefix in prefixes {
        for len in edge_lengths() {
            let bodies: Vec<Vec<u8>> = (0..8).map(|i| pattern(len, i as u8)).collect();
            let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
            let mut got = vec![[0u8; 32]; bodies.len()];
            hash_many_prefixed(&prefix, &refs, &mut got);
            for (i, b) in bodies.iter().enumerate() {
                let joined = [prefix.as_slice(), b.as_slice()].concat();
                assert_eq!(
                    got[i],
                    reference(&joined),
                    "prefix {} body {len} lane {i}",
                    prefix.len()
                );
            }
        }
    }
}

/// Per-message prefixes, mixing prefixed and bare messages in one batch
#[test]
fn mixed_per_message_prefixes() {
    let pa = b"alpha".to_vec();
    let pb = vec![0x5au8; 70];
    let bodies: Vec<Vec<u8>> = (0..8).map(|i| pattern(200 + i * 29, i as u8)).collect();
    let msgs: Vec<Message<'_>> = bodies
        .iter()
        .enumerate()
        .map(|(i, b)| match i % 3 {
            0 => Message::new(b),
            1 => Message::prefixed(&pa, b),
            _ => Message::prefixed(&pb, b),
        })
        .collect();
    let mut got = vec![[0u8; 32]; bodies.len()];
    hash_messages(&msgs, &mut got);
    for (i, b) in bodies.iter().enumerate() {
        let joined = match i % 3 {
            0 => b.clone(),
            1 => [pa.as_slice(), b].concat(),
            _ => [pb.as_slice(), b].concat(),
        };
        assert_eq!(got[i], reference(&joined), "lane {i}");
    }
}

/// Broad randomised sweep on top of the targeted cases above
#[test]
fn randomised_sweep() {
    let mut rng = StdRng::seed_from_u64(0xD1FF_5EED);
    for _ in 0..300 {
        let count = rng.gen_range(0..24usize);
        let bodies: Vec<Vec<u8>> = (0..count)
            .map(|_| {
                let len = rng.gen_range(0..1200usize);
                (0..len).map(|_| rng.r#gen::<u8>()).collect()
            })
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; count];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "random lane {i} len {}", b.len());
        }
    }
}

/// Every backend must agree with every other, not merely with the reference
///
/// A backend that dispatch never selects on this host still gets exercised
/// here, provided the host can actually run it.
#[test]
fn backends_agree() {
    use tape_sha256::backends;

    let lens = edge_lengths();
    for window in lens.windows(9) {
        let bodies: Vec<Vec<u8>> = window
            .iter()
            .enumerate()
            .map(|(i, &l)| pattern(l, i as u8))
            .collect();
        let prefix = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
        let msgs: Vec<Message<'_>> = bodies
            .iter()
            .map(|b| Message::prefixed(prefix, b))
            .collect();

        let mut want = vec![[0u8; 32]; bodies.len()];
        backends::serial(&msgs, &mut want);
        for (i, b) in bodies.iter().enumerate() {
            let joined = [prefix.as_slice(), b.as_slice()].concat();
            assert_eq!(want[i], reference(&joined), "serial lane {i}");
        }

        let mut got = vec![[0u8; 32]; bodies.len()];
        backends::portable8(&msgs, &mut got);
        assert_eq!(got, want, "portable8 vs serial, lens {window:?}");

        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        {
            let mut got = vec![[0u8; 32]; bodies.len()];
            backends::neon4(&msgs, &mut got);
            assert_eq!(got, want, "neon4 vs serial, lens {window:?}");

            if std::arch::is_aarch64_feature_detected!("sha2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::neon_sha2x4(&msgs, &mut got) };
                assert_eq!(got, want, "neon_sha2x4 vs serial, lens {window:?}");
            }
        }

        // Only ever run on real x86. The AVX kernels are developed on an
        // AArch64 host where they compile but never execute, so this is the
        // gate that must pass before they are trusted anywhere.
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        {
            if is_x86_feature_detected!("sha") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::shani_x4(&msgs, &mut got) };
                assert_eq!(got, want, "shani_x4 vs serial, lens {window:?}");
            }
            if is_x86_feature_detected!("avx2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx2_8(&msgs, &mut got) };
                assert_eq!(got, want, "avx2_8 vs serial, lens {window:?}");
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16(&msgs, &mut got) };
                assert_eq!(got, want, "avx512_16 vs serial, lens {window:?}");

                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16x2(&msgs, &mut got) };
                assert_eq!(got, want, "avx512_16x2 vs serial, lens {window:?}");
            }
        }
    }
}

/// The `_slices` forms must agree with the `Message`-taking ones
///
/// These are the paths the public wrappers actually dispatch to, and without
/// this only the host's active kernel would ever exercise them. Uniform
/// batches drive the same-shape fast path, ragged windows the general one.
#[test]
fn slices_forms_agree() {
    use tape_sha256::backends;

    let prefix = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
    let lens = edge_lengths();
    // 33 = one full chunk of the widest kernel (the 32-message interlace)
    // plus a straggler, so the fused path and its remainder are exercised
    // through the `_slices` drivers, not only the dedicated test below.
    let mut cases: Vec<Vec<usize>> = lens.iter().map(|&l| vec![l; 33]).collect();
    cases.extend(lens.windows(9).map(|w| w.to_vec()));

    for case in &cases {
        let bodies: Vec<Vec<u8>> = case
            .iter()
            .enumerate()
            .map(|(i, &l)| pattern(l, i as u8))
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let msgs: Vec<Message<'_>> = bodies
            .iter()
            .map(|b| Message::prefixed(prefix, b))
            .collect();

        let mut want = vec![[0u8; 32]; bodies.len()];
        backends::serial(&msgs, &mut want);

        let mut got = vec![[0u8; 32]; bodies.len()];
        backends::portable8_slices(prefix, &refs, &mut got);
        assert_eq!(got, want, "portable8_slices, lens {case:?}");

        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        {
            let mut got = vec![[0u8; 32]; bodies.len()];
            backends::neon4_slices(prefix, &refs, &mut got);
            assert_eq!(got, want, "neon4_slices, lens {case:?}");

            if std::arch::is_aarch64_feature_detected!("sha2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::neon_sha2x4_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "neon_sha2x4_slices, lens {case:?}");
            }
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        {
            if is_x86_feature_detected!("sha") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::shani_x4_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "shani_x4_slices, lens {case:?}");
            }
            if is_x86_feature_detected!("avx2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx2_8_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "avx2_8_slices, lens {case:?}");
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "avx512_16_slices, lens {case:?}");

                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16x2_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "avx512_16x2_slices, lens {case:?}");
            }
        }
    }
}

/// The two-wave kernel's fused path needs a full uniform 32; the windows in
/// backends_agree only reach its split path. Covers fused (32), fused plus
/// the driver's next chunk (33), and ragged (24, split), at every edge
/// length.
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
#[test]
fn avx512_two_wave_full_group() {
    use tape_sha256::backends;

    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
        return;
    }
    let prefix = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
    for len in edge_lengths() {
        for count in [24usize, 32, 33] {
            let bodies: Vec<Vec<u8>> = (0..count).map(|i| pattern(len, i as u8)).collect();
            let msgs: Vec<Message<'_>> = bodies
                .iter()
                .map(|b| Message::prefixed(prefix, b))
                .collect();
            let mut want = vec![[0u8; 32]; count];
            backends::serial(&msgs, &mut want);
            let mut got = vec![[0u8; 32]; count];
            unsafe { backends::avx512_16x2(&msgs, &mut got) };
            assert_eq!(got, want, "two-wave len {len} count {count}");
        }
    }
}

/// Reports which kernels this run actually exercised
///
/// A green run on a host that silently skipped the AVX kernels is not a
/// validated one, and only this output tells the two apart.
#[test]
fn report_exercised_backends() {
    println!("active dispatch backend: {}", tape_sha256::backend());
    #[cfg(target_arch = "x86_64")]
    {
        println!("avx2 available:   {}", is_x86_feature_detected!("avx2"));
        println!(
            "avx512 available: {}",
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
        );
    }
    #[cfg(target_arch = "aarch64")]
    println!(
        "aarch64 sha2 ext: {}",
        std::arch::is_aarch64_feature_detected!("sha2")
    );
    #[cfg(not(target_arch = "x86_64"))]
    println!("not x86_64: AVX kernels compiled but NOT executed by this run");
}

/// `hash_pairs` must equal hashing the materialised `prefix || left || right`
///
/// This is the Merkle interior-node shape: a node hashes a 26-byte domain
/// prefix and two 20-byte proof entries held in separate buffers. The
/// three-segment path must not drift from simple concatenation, at any
/// length or batch size.
#[test]
fn hash_pairs_matches_concatenation() {
    use tape_sha256::hash_pairs;

    let prefixes: Vec<Vec<u8>> = vec![
        vec![],
        b"\x01SOLANA_MERKLE_SHREDS_NODE".to_vec(),
        vec![0xab; 63],
        vec![0xcd; 64],
        vec![0xef; 65],
    ];
    // 20 is the real proof-entry size; the rest bracket block boundaries.
    for half in [0usize, 1, 20, 31, 32, 63, 64, 65, 500] {
        for prefix in &prefixes {
            for count in [1usize, 3, 8, 9, 16, 17, 20] {
                let lefts: Vec<Vec<u8>> = (0..count).map(|i| pattern(half, i as u8)).collect();
                let rights: Vec<Vec<u8>> =
                    (0..count).map(|i| pattern(half, (i + 99) as u8)).collect();
                let lr: Vec<&[u8]> = lefts.iter().map(|b| b.as_slice()).collect();
                let rr: Vec<&[u8]> = rights.iter().map(|b| b.as_slice()).collect();

                let mut got = vec![[0u8; 32]; count];
                hash_pairs(prefix, &lr, &rr, &mut got);

                for i in 0..count {
                    let joined = [prefix.as_slice(), &lefts[i], &rights[i]].concat();
                    assert_eq!(
                        got[i],
                        reference(&joined),
                        "prefix {} half {half} count {count} pair {i}",
                        prefix.len()
                    );
                }
            }
        }
    }
}

/// `hash_chain` must equal `sha2` fed back into itself.
#[test]
fn hash_chain_matches_sha2() {
    use tape_sha256::hash_chain;

    for seed_byte in [0u8, 1, 0xff, 0x5a] {
        let seed = [seed_byte; 32];
        let mut want = seed;
        for n in 0..=64u64 {
            assert_eq!(hash_chain(&seed, n), want, "seed {seed_byte:#x} n {n}");
            assert_eq!(
                tape_sha256::backends::chain_portable(&seed, n),
                want,
                "portable, seed {seed_byte:#x} n {n}"
            );
            want = reference(&want);
        }
    }
}

/// A chain of `a + b` links must be a chain of `a` continued by one of `b`
///
/// Entry verification restarts the chain at every entry boundary, so a kernel
/// whose first link differs from its steady-state one would still pass a
/// fixed-length check.
#[test]
fn hash_chain_composes() {
    use tape_sha256::hash_chain;

    let seed = [7u8; 32];
    for a in [0u64, 1, 2, 3, 63, 64, 65, 500] {
        for b in [0u64, 1, 2, 3, 500] {
            assert_eq!(
                hash_chain(&seed, a + b),
                hash_chain(&hash_chain(&seed, a), b),
                "a {a} b {b}"
            );
        }
    }
}

/// A long chain, so a kernel that drifts only after many links is caught
#[test]
fn hash_chain_long_run() {
    use tape_sha256::hash_chain;

    const LINKS: u64 = 62_500;
    let seed = [0x33u8; 32];

    let mut want = seed;
    for _ in 0..LINKS {
        want = reference(&want);
    }
    assert_eq!(hash_chain(&seed, LINKS), want);
    assert_eq!(tape_sha256::backends::chain_portable(&seed, LINKS), want);
}

/// `hash_chains` must agree with `hash_chain` lane for lane
///
/// The scheduler retires a lane at its finish line and refills it from the
/// queue, so ragged and short batches are where a chain could pick up another
/// chain's state. Swept over batch sizes around every kernel width, and over
/// lengths that finish in every order, including zero.
#[test]
fn hash_chains_matches_hash_chain() {
    use tape_sha256::{hash_chain, hash_chains};

    let seeds: Vec<[u8; 32]> = (0..70u8).map(|i| [i.wrapping_mul(37); 32]).collect();

    for count in [
        1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 70,
    ] {
        for shape in 0..4 {
            let lens: Vec<u64> = (0..count)
                .map(|i| match shape {
                    // uniform, the tick-entry case
                    0 => 37,
                    // ascending, descending, and a spread including zero
                    1 => i as u64,
                    2 => (count - i) as u64,
                    _ => ((i * 7) % 11) as u64,
                })
                .collect();

            let mut got = vec![[0u8; 32]; count];
            hash_chains(&seeds[..count], &lens, &mut got);

            for i in 0..count {
                assert_eq!(
                    got[i],
                    hash_chain(&seeds[i], lens[i]),
                    "count {count} shape {shape} lane {i} len {}",
                    lens[i]
                );
            }
        }
    }
}

/// The replay shape: many equal-length chains, checked against sha2 directly
#[test]
fn hash_chains_replay_shape() {
    use tape_sha256::hash_chains;

    const ENTRIES: usize = 64;
    const LINKS: u64 = 500;

    let seeds: Vec<[u8; 32]> = (0..ENTRIES).map(|i| [i as u8; 32]).collect();
    let lens = vec![LINKS; ENTRIES];
    let mut got = vec![[0u8; 32]; ENTRIES];
    hash_chains(&seeds, &lens, &mut got);

    for (i, seed) in seeds.iter().enumerate() {
        let mut want = *seed;
        for _ in 0..LINKS {
            want = reference(&want);
        }
        assert_eq!(got[i], want, "entry {i}");
    }
}

/// Every chains backend must agree with the serial chain
///
/// `hash_chains` only ever exercises the kernel dispatch picks, so each step
/// kernel the CPU can run is driven through the scheduler here. Batch sizes
/// sit below, at, and past every kernel width, and the lengths are ragged
/// with zeros mixed in, so filling, retiring, and refilling all happen at
/// every width.
#[test]
fn chains_backends_agree() {
    use tape_sha256::{backends::chains, hash_chain};

    let seeds: Vec<[u8; 32]> = (0..71u8).map(|i| [i.wrapping_mul(29); 32]).collect();
    let lens: Vec<u64> = (0..71u64).map(|i| (i * 13) % 97).collect();
    let want: Vec<[u8; 32]> = seeds
        .iter()
        .zip(&lens)
        .map(|(s, &l)| hash_chain(s, l))
        .collect();

    macro_rules! check {
        ($name:literal, $f:expr) => {
            for n in [0usize, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 71] {
                let mut got = vec![[0u8; 32]; n];
                $f(&seeds[..n], &lens[..n], &mut got);
                assert_eq!(got, want[..n], concat!($name, " n={}"), n);
            }
        };
    }

    check!("portable1", chains::portable1);
    check!("portable8", chains::portable8);

    #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
    {
        check!("neon4", |s, l, o: &mut [_]| unsafe {
            chains::neon4(s, l, o)
        });
        if std::arch::is_aarch64_feature_detected!("sha2") {
            check!("neon_sha2x4", |s, l, o: &mut [_]| unsafe {
                chains::neon_sha2x4(s, l, o)
            });
            check!("neon_sha2x8", |s, l, o: &mut [_]| unsafe {
                chains::neon_sha2x8(s, l, o)
            });
        }
    }

    // Only ever run on real x86, same as `backends_agree`: the gate the AVX
    // chain kernels must pass before they are trusted anywhere.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    {
        if is_x86_feature_detected!("sha") {
            check!("shani_x4", |s, l, o: &mut [_]| unsafe {
                chains::shani_x4(s, l, o)
            });
        }
        if is_x86_feature_detected!("avx2") {
            check!("avx2_8", |s, l, o: &mut [_]| unsafe {
                chains::avx2_8(s, l, o)
            });
        }
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            check!("avx512_16", |s, l, o: &mut [_]| unsafe {
                chains::avx512_16(s, l, o)
            });
            check!("avx512_16x2", |s, l, o: &mut [_]| unsafe {
                chains::avx512_16x2(s, l, o)
            });
        }
    }
}
