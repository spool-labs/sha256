# Benchmarks

Workload: one Solana erasure batch of Merkle leaves, 64 leaves
(32 data + 32 coding shreds), each a 26-byte domain-separation prefix plus
~1 KB of payload, 67,680 bytes per batch. Reported time is the best of
15 rounds of 50 iterations (`cargo bench --bench merkle_leaves`).

Every figure below is single-threaded: one thread, pinned with `taskset`
to one physical core of an otherwise idle machine, with that core's SMT
sibling also idle. The only exception is the SMT pairing section, which is
explicitly two threads on the two siblings of one core. Firedancer was
built from source on the same machine with its matching profile, so each
comparison is same-session; absolute numbers drift a few percent between
sessions on cloud hardware, so only same-session pairs are meaningful.

**AMD Zen 5 (family 0x1a), 2026-08-08:**

```
sha2 (serial, current)          27.71 us/batch       2443 MB/s   (spread 0.8%)
tape serial (1 lane)           109.77 us/batch        617 MB/s   (spread 0.3%)
tape portable (8 lane)         255.13 us/batch        265 MB/s   (spread 0.2%)
tape avx2 (8 lane)              29.34 us/batch       2307 MB/s   (spread 0.7%)
tape shani (x4 stream)          14.76 us/batch       4584 MB/s   (spread 0.4%)
tape avx512 (16 lane)           10.82 us/batch       6254 MB/s   (spread 0.5%)
tape smt pair (2x avx512)       10.76 us/batch       6289 MB/s   (spread 1.3%)
tape avx512 (2x16 interlace)     7.82 us/batch       8654 MB/s   (spread 3.9%)
firedancer avx (8 lane)         24.97 us/batch       2711 MB/s   (spread 1.7%)
firedancer avx512 (16 lane)     12.55 us/batch       5393 MB/s   (spread 0.3%)
isa-l mb (x16 avx512)           10.92 us/batch       6195 MB/s   (spread 1.1%)
isa-l mb (x16 avx512-ni)        10.85 us/batch       6237 MB/s   (spread 1.6%)
isa-l mb (dispatched)           10.89 us/batch       6213 MB/s   (spread 1.8%)
```

One pinned invocation, so every row above shares a run and no pairing is
carried across sessions. Three pinned repetitions put the interlace at
7.82, 8.30 and 8.34 us; this is the best of them and the rest of the table
moves under 1% between reps.

This is a faster part than the c4d used through 2026-07-29, where the same
`sha2` baseline measured 1863 MB/s against 2443 here. The exact model was
not captured in the run logs; what is certain is AMD family 0x1a, since
that is the only family the `avx512-2x16` dispatch is enabled for and it
is the backend the binary selected.

The dispatched kernel is the `avx512-2x16` interlace: two 16-lane
compressions with their rounds interlaced in one thread, filling the
dependency stalls a single wave leaves idle. **3.54x over the serial
baseline, 38% over the single wave, 60% faster than Firedancer's AVX-512
kernel, and 39% faster than isa-l_crypto's best.** Enabled on AMD family
0x1a only: measured neutral on Zen 4, and on Intel it wins the cycles but
loses the clock to the AVX-512 licence. Both keep the single wave; see
those sections.

isa-l_crypto is the other production multi-buffer implementation worth
measuring against, and it lands within 1% of tape's *single* 16-lane wave
(6237 against 6254) while both of its AVX-512 entry points and its own
dispatcher agree to within 0.7%. Its `avx512-ni` variant, which uses the
SHA extension inside the multi-buffer manager, buys it nothing here. The
comparison is not quite like for like in isa-l's favour or against: its
API is a job manager rather than a kernel call, so the row includes submit
and flush scheduling that tape's does not do, and the prefix concatenation
is hoisted out of the timed region for both.

Building with `-C target-cpu=native` makes the interlace **worse**: the
same source measured 8.19-8.32 us on the default build and 8.88-9.47 us
with native, an 8-14% loss, while `shani-x4` and the single AVX-512 wave
moved under 2%. That is the instruction-cache footprint below biting
again, and it is a reason not to recommend `target-cpu=native` for this
crate. Native does help the two kernels dispatch never selects on this
part -- `avx2-8` went 29.49 to 20.56 us and `portable-8` 242.43 to 147.28
-- which is why the loss shows up only where it costs anything.

Rounds 16..64 are rolled rather than fully unrolled, which is a
measurement-integrity matter rather than a style one. Fully unrolled the
kernel body is 30,748 bytes against Zen 5's 32 KB 8-way L1 instruction
cache — roughly 7.5 of 8 ways per set, so whether it fits is decided by
where the linker happens to put it. In that state this row moved between
18.15 and 11.31 us on the 07-29 c4d purely from 35 KB of unrelated code
elsewhere in the crate, with the hot loop byte-identical and the spill
counts equal; the same source built with
`-Cllvm-args=-align-all-functions=9` went from 18.14 to 11.2. Rolling
brings the body to 22,333 bytes, and the cliff goes away. Anything that
grows this kernel should re-check that number.

That is 8.18 M hashes/sec for ~1 KB messages, or **139 M SHA-256 block
compressions/sec** on one core (every message here pads to 17 blocks, so
a batch is 1,088 block compressions). The block figure is the one worth
quoting, since hashes/sec depends entirely on message length.

The interlace still does **not** beat a correctly pinned SMT pair. The
`smt pair` row in the table above reads 10.76 us, but that is the
`taskset -c 2` artifact: confining both threads to one logical CPU is not
an SMT pairing, and the row is meaningless under the pinning the rest of
the table needs. An unpinned invocation in the same session restores it at
**5.88 us against the interlace's 8.49**, so two real threads on the two
siblings of one core remain 1.44x ahead. The interlace's value is that it
captures most of that gain without needing an idle sibling, which a loaded
validator does not have; it is not a win over two real threads.

**AMD Zen 4 (EPYC 9B14, GCP c3d):**

```
sha2 (serial, current)          43.71 us/batch       1548 MB/s
tape shani (x4 stream)          20.88 us/batch       3242 MB/s
tape avx512 (16 lane)           23.79 us/batch       2845 MB/s
firedancer avx (8 lane)         27.93 us/batch       2423 MB/s
firedancer avx512 (16 lane)     22.51 us/batch       3006 MB/s
```

After the interlaced rounds were unrolled, the SHA-NI streams beat every
16-lane kernel
on Zen 4 — ours, Firedancer's, and the two-thread avx512 SMT pair (23.84).
Dispatch therefore prefers `shani-x4` on Zen 4, detected as AMD family
0x19 with AVX-512 (the family spans Zen 3 and Zen 4, and Zen 3 has no
AVX-512; Zen 5 is family 0x1a and keeps `avx512-16`). That makes tape
**2.09x over the serial baseline and ~8% faster than Firedancer's best
kernel here** — Firedancer ships no SHA-NI batch kernel to route to.
Kernel for kernel at 16 lanes its AVX-512 still leads ours by ~5% on this
part; the shani flip wins the machine anyway. Firedancer was built
natively on this box (`linux_gcc_zen4`, GCC 15.2, real Genoa).

The 2x16 interlace was probed here too and measured exactly neutral
(23.69 vs 23.72 single-wave, same session, spreads under 3%): Zen 4
double-pumps 512-bit ops through a 256-bit datapath, so a single wave
already saturates the pipes and leaves no stalls for the second wave to
fill — the same reason its SMT gain is only 1.17x. Dispatch keeps
`shani-x4` on Zen 4.

Rolling the interlace's rounds, which is worth 40% on Zen 5 by getting the
kernel back inside the L1 instruction cache, was tried here and is worth
1.3% (22.10 -> 21.82 best of four, against `shani-x4` at 20.66 in the same
session). Zen 4 has the same 32 KB L1I, so it sat on the same cliff, but
fetch was never its limit — the pipes were. This is measured, not assumed;
do not re-run it.

**AMD Zen 3 (EPYC 7B13, GCP c2d):**

```
sha2 (serial, current)          42.66 us/batch       1587 MB/s
tape shani (x4 stream)          20.73 us/batch       3264 MB/s
tape avx2 (8 lane)              46.81 us/batch       1446 MB/s
firedancer avx (8 lane)         45.87 us/batch       1476 MB/s
```

**2.06x over the serial baseline, and ~2.2x faster than Firedancer.** The
reason is structural rather than clever: Firedancer ships no SHA-NI batch
kernel, so without AVX-512 it falls back to 8-lane AVX, which loses even to
serial SHA-NI. Note our own AVX2 row also loses to the baseline: on a
machine with SHA-NI and no AVX-512, integer multi-buffer is the wrong tool
and dispatch picks `shani-x4`.

**Intel Emerald Rapids (Xeon Platinum 8581C, GCP c4):**

First, the original run, kept because the correction below is the point. It
came from an earlier session on a c4 instance clocking about 15% lower than the
one used for everything after it, so compare its ratios and not its absolutes:

```
sha2 (serial, current)          49.18 us/batch       1376 MB/s
tape shani (x4 stream)          41.04 us/batch       1649 MB/s
tape avx512 (16 lane)           24.49 us/batch       2763 MB/s
firedancer avx (8 lane)         29.20 us/batch       2318 MB/s
firedancer avx512 (16 lane)     21.82 us/batch       3102 MB/s
```


SHA-NI interlacing is also much weaker on Intel, 1.19x over the serial
baseline versus 1.84x on Zen 5, which matches Intel not executing
`sha256rnds2` concurrently while AMD does. The interlace unroll that
bought 1.30x on Zen 3 is worth ~3% here (same-session paired, 32.61 to
31.62 us) for the same reason: a serialized unit is bound by round
latency, not by how fast the loop feeds it.

The `sha2` row is what agave's leaf hashing does today (one hash per leaf;
`sha2` uses SHA-NI hardware on all these CPUs). The Firedancer rows call
`fd_sha256_private_batch_{avx,avx512}` directly, the strongest batch
implementation we know of.

One methodological note: Firedancer's batch API takes each message as a
single contiguous buffer, so the harness concatenates `prefix || leaf` for
it outside the timed region. That reflects production accurately rather than
charitably. Firedancer's shredder writes the 26-byte Merkle prefix into
dead space at the tail of each shred's signature field
(`fd_shredder.c`), so its real per-shred prefix cost is a 26-byte write.
`Message::prefixed` reaches the same zero-copy result without requiring
writable dead space in front of the payload; that is a generality
difference in the API, not a performance edge, and no number is claimed
for it.

**Intel Granite Rapids (Xeon 6985P-C, GCP c4-standard-8-lssd):**

```
sha2 (serial, current)          36.87 us/batch       1835 MB/s
tape serial (1 lane)           167.08 us/batch        405 MB/s
tape portable (8 lane)         308.31 us/batch        220 MB/s
tape shani (x4 stream)          29.20 us/batch       2318 MB/s
tape smt pair (2x avx512)       17.35 us/batch       3900 MB/s
tape avx2 (8 lane)              44.11 us/batch       1534 MB/s
tape avx512 (16 lane)           17.25 us/batch       3923 MB/s
tape avx512 (2x16 interlace)    17.52 us/batch       3864 MB/s
firedancer avx (8 lane)         22.56 us/batch       2999 MB/s
firedancer avx512 (16 lane)     16.89 us/batch       4006 MB/s
```

2.14x over the serial baseline, best of six invocations. Unlike the c4
instance behind the Emerald Rapids numbers, this box is quiet: most rows
repeat to under 1% and the wall clock is usable here. Cycles anyway, six
paired rounds:

```
                     cyc/block-step   instr/block-step    IPC     GHz
tape avx512-16                 1048               2234   2.13    4.12
tape avx512-2x16               1004               2483   2.47    3.68
firedancer avx512              1034               2247   2.17    4.12
```

**Firedancer is 1.3% ahead on cycles, winning all six rounds at 0.3-0.4%
spread, not the 11% previously recorded on this part.** Wall clock puts it at
2.2%. Granite Rapids therefore behaves like Emerald Rapids, and the gap is a
percent or two Intel-wide rather than double digits.

One honest loose end. The earlier 11% came from a run with Firedancer at
16.03 us, about 6% faster than any build produced here. Its kernel was given
gcc 12.2 and gcc 14.2, and `-march=icelake-server` and `-march=graniterapids`,
and lands at 1034 to 1042 cycles per block step in every combination, on this
part and on Emerald Rapids alike. So the build is at least self-consistent,
but the older figure is not reproduced and its provenance is unknown. Compiler
tuning is dead as an explanation for the gap on both parts.

The 2x16 interlace again wins the cycles, 1004 against Firedancer's 1034, and
again cannot spend them: its clock averages 3.68 GHz against the single wave's
4.12, and it ends up 1.6% behind in wall time. Treat that margin loosely. This
kernel is the volatile one everywhere it has been measured, spreading 7% on
cycles and 3.7% within a single bench run, and its clock figure is a mean over
those noisy rounds rather than a stable operating point.

A later session put a number on that volatility and it is worse than the
within-run spread suggests: across ten invocations on Granite Rapids the
interlace ranged 17.61 to 20.15 us (14%), against the single wave's 3.7%,
and the spread is bimodal rather than noisy — each process sits in one mode
or the other, and disabling ASLR collapses it to a single tight cluster. The
cause is per-process memory layout and is not understood; it is not the 4 KB
schedule stride, which was ruled out by padding. Nothing rides on it while
Intel dispatches the single wave, but it does mean the best-of-six figures
above flatter the interlace on this part, and a median would serve better. What is solid is the
direction, and it is the same on both Intel parts: ahead on cycles, behind on
the clock, so dispatch keeps the single wave.

**AArch64 (Apple M-series):**

```
sha2 (serial, current)         110.95 us/batch        610 MB/s
tape neon (4 lane)              53.44 us/batch       1267 MB/s
tape neon-sha2 (x4 stream)      16.25 us/batch       4166 MB/s
```

(`sha2` is on its software path here.) The `neon-sha2` backend keeps four
hardware SHA-256 streams in flight to cover the crypto instructions'
latency, reaching 6.8x the serial baseline. Firedancer has no AArch64 batch
kernel, so there is nothing to compare against.

**SMT sibling pairing**: the entry points are stateless, so two
caller-pinned threads on the SMT siblings of one physical core can each
hash half a batch. What that is worth is entirely microarchitecture, one
pair per row, each measured same-session against its own single-thread
number:

```
AMD Zen 5         9.53 vs 14.15 us    1.48x
AMD Zen 4       20.16 vs 23.66 us    1.17x
Intel EMR       20.77 vs 23.88 us    1.15x
```

Zen 5's single thread is dependency-bound on its native 512-bit path, so
the sibling fills real gaps; Zen 4's double-pumped datapath leaves it less
room, and an early run that pinned both threads to one logical CPU
measured no gain at all, so verify the sibling pinning before trusting any
SMT number. Both caveats from the crate docs apply: the
sibling must actually be idle (this measures spare capacity, which a
loaded validator does not have), and the threads are the caller's, since
the library never spawns or pins. The `smt pair` row in `merkle_leaves`
is the reference harness.

**WebAssembly**: a `simd128-4` backend exists and is differentially gated
like the rest. Its honest scope, measured in Node and in a real Chrome tab:
for batched ~1 KB messages in a browser, WebCrypto's hardware-backed
`subtle.digest` wins (75 vs 80-90 us for this workload in Chrome), so use
WebCrypto there. Where the wasm backend earns its keep is small messages in
tight synchronous loops, address grinding being the canonical case: 4.55 M
candidates/s in a Chrome tab, 3.7x the web3.js stack and 1.6x `hash-wasm`.
PDA derivation is bound by the ed25519 curve check (94-99% of the cost), so
no SHA-256 implementation moves it. Numbers, methodology, and the full
where-it-wins/where-it-loses map are in `benches/wasm/README.md`.

**The no-SIMD fallback width is per-architecture**, because multi-buffer in
scalar code costs a real general-purpose register per lane rather than
riding along free in a vector lane. Measured across widths on one erasure
batch:

```
                        x86-64 (Zen 4)   aarch64 (Apple M)
tape serial (1 lane)        218.30 us          106.5 us
tape portable (2 lane)      368.66 us              -
tape portable (4 lane)      365.01 us              -
tape portable (8 lane)      418.78 us           93.7 us
```

x86-64 has 16 GPRs and cannot absorb even two lanes: the state alone is 16
live words at N=2, so it spills before the extra independent chains buy
anything, and every width loses to plain serial. aarch64 has 31 and eight
lanes come out 13% ahead. Dispatch falls back to `portable-8` on aarch64 and
`portable-1` everywhere else: one lane is the safe default, giving up ~13%
where the registers exist against the 1.9x the wide kernel loses where they
do not, and aarch64 is named because it was measured rather than because it
is 64-bit. `portable8` stays exported as the differential-test reference
regardless of which one dispatches.

## tape blob commitments

tape's own workload (`cargo bench --bench blob_group`): one blob commitment
is a Merkle tree over an erasure group of up to 20 slices of up to 4 KiB,
leaves hashed as `sha256("LEAF" || slice)`, pairs as
`sha256("LEFT" || left || "RIGHT" || right)`. The slicer builds it per
write, the gateway and archive nodes rebuild it per verified read, and
tape-sdk runs the same hashing on the serial `sha2` crate today.

Apple M-series, min of 15 rounds, batch rows asserted equal to serial:

```
leaves sha2 (serial, current)    128.92 us/group       636 MB/s
leaves tape hash_many_prefixed    18.72 us/group      4381 MB/s
joins sha2 (serial, current)       4.19 us/group       366 MB/s
joins tape hash_many               0.72 us/group      2122 MB/s

leaf level 6.9x, join level 5.8x, whole commitment 6.85x
```

That is ~51,000 slice groups per second on one core, about 4.2 GB/s of
blob payload. The 20-slice group also happens to divide evenly into the
4-stream ARM backend, so no lane sits idle.

On x86 the group does not divide the 16-lane kernel, and that is where
dispatch's tail routing earns its keep: a remainder of 1 to 12 messages
goes to the SHA-NI streams instead of a mostly-empty wide group. Measured
on Zen 4 (EPYC 9B14, Firedancer built natively on the same box,
same-session):

```
leaves sha2 (serial, current)     50.36 us/group      1628 MB/s
leaves tape hash_many_prefixed    28.78 us/group      2850 MB/s
leaves fd avx512 (16 lane)        32.43 us/group      2528 MB/s
joins sha2 (serial, current)       1.82 us/group       844 MB/s
joins tape hash_many               1.28 us/group      1199 MB/s
joins fd avx (8 lane)              1.32 us/group      1158 MB/s

whole commitment: tape 30.05 us vs firedancer 33.75 = 1.12x faster
```

Without the routing the leaf level ran at 42.7 us (a 4-message tail
occupying a 16-lane group), so the routing alone is worth 1.48x on this
shape. Firedancer pays the full tail tax: its only option for a partial
batch is the same wide kernel with idle lanes, and it ships no SHA-NI
kernel to route to.

## Summary

| hardware | best backend | vs `sha2` | vs Firedancer |
|---|---|---|---|
| AMD Zen 5 | `avx512-2x16` | 3.5x | **60% faster** |
| AMD Zen 4 | `shani-x4` | 2.1x | **~8% faster** |
| AMD Zen 3 | `shani-x4` | 2.1x | **~2.2x faster** |
| Intel EMR / GNR | `avx512-16` | 2.0-2.1x | ~1% slower (cycles) |
| AArch64 | `neon-sha2-x4` | 6.8x | no comparison |

In agave integration measurements, switching shred recovery's leaf hashing
to this crate made recovery 14-21% faster end-to-end, and the
single-threaded SIMD path outperformed the existing 7-thread rayon pool by
1.55x, returning six cores to the rest of the validator.

## Running the reference comparisons

Both reference implementations are behind bench-only features and neither
is vendored into or linked against the library itself.

To include the Firedancer rows in `merkle_leaves` or `blob_group`, point
`FD_LIB_DIR` at a Firedancer build's lib directory and enable the feature:

```sh
FD_LIB_DIR=/path/to/firedancer/build/linux/gcc/zen5/lib \
  cargo bench --bench merkle_leaves --features firedancer-bench
```

`firedancer-bench` links the 8-lane kernel only. Add the AVX-512 row with
`--features firedancer-bench-avx512`, but only against a Firedancer build
whose machine profile enabled AVX-512. The symbol does not exist otherwise
and the bench will fail to link rather than fall back.

For the isa-l_crypto rows, point `ISAL_DIR` at an isa-l_crypto tree that
has been built in place and enable `isal-bench`:

```sh
ISAL_DIR=/path/to/isa-l_crypto \
  cargo bench --bench merkle_leaves --features isal-bench
```

The build accepts either a built-in-place tree (headers under `include/`,
archive under `.libs/`) or an installed prefix. The arch-specific entry
points it times are exported from `libisal_crypto.a` but declared only in
`include/internal/`, so `benches/isal_shim.c` redeclares them against the
public types; every entry point is checked against `sha2` before it is
timed, since a mis-wired one produces a fast wrong answer rather than a
link error.
