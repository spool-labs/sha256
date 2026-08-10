//! Multi-stream backend on the ARMv8 SHA-256 crypto extension.

use {
    crate::{
        batch::{stage_prefix_block, Message, Shape, BLOCK},
        core::{H0, K},
    },
    std::arch::aarch64::*,
};

/// Number of independent messages kept in flight
///
/// Enough to cover `sha256h`'s latency without exhausting the register file:
/// each stream needs two state vectors plus four message vectors.
pub const STREAMS: usize = 4;

#[derive(Clone, Copy)]
struct State {
    abcd: uint32x4_t,
    efgh: uint32x4_t,
}

impl State {
    /// The instructions want the IV split `abcd`/`efgh`, which is just the two
    /// halves of the shared `H0`
    #[inline(always)]
    unsafe fn init() -> Self {
        State {
            abcd: vld1q_u32(H0.as_ptr()),
            efgh: vld1q_u32(H0.as_ptr().add(4)),
        }
    }

    #[inline(always)]
    unsafe fn digest(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        // The instructions keep state native-endian; the output is big-endian.
        vst1q_u8(
            out.as_mut_ptr(),
            vrev32q_u8(vreinterpretq_u8_u32(self.abcd)),
        );
        vst1q_u8(
            out.as_mut_ptr().add(16),
            vrev32q_u8(vreinterpretq_u8_u32(self.efgh)),
        );
        out
    }
}

/// Loads a 64-byte block as four big-endian-corrected message vectors
#[inline(always)]
unsafe fn load_msg(block: &[u8]) -> [uint32x4_t; 4] {
    debug_assert!(block.len() >= BLOCK);
    let mut m = [vdupq_n_u32(0); 4];
    for (i, mi) in m.iter_mut().enumerate() {
        let raw = vld1q_u8(block.as_ptr().add(i * 16));
        *mi = vreinterpretq_u32_u8(vrev32q_u8(raw));
    }
    m
}

/// Compresses one block into `st` using the crypto extension.
#[inline(always)]
unsafe fn compress_block(st: &mut State, msg: &mut [uint32x4_t; 4]) {
    let (mut s0, mut s1) = (st.abcd, st.efgh);
    // Literal round indices, so the `i < 12` schedule test folds away instead
    // of staying a branch.
    macro_rules! round {
        ($i:literal) => {{
            let slot = $i & 3;
            let wk = vaddq_u32(msg[slot], vld1q_u32(K.as_ptr().add($i * 4)));
            if $i < 12 {
                msg[slot] = vsha256su0q_u32(msg[slot], msg[($i + 1) & 3]);
            }
            let saved = s0;
            s0 = vsha256hq_u32(s0, s1, wk);
            s1 = vsha256h2q_u32(s1, saved, wk);
            if $i < 12 {
                msg[slot] = vsha256su1q_u32(msg[slot], msg[($i + 2) & 3], msg[($i + 3) & 3]);
            }
        }};
    }
    macro_rules! rounds {
        ($($i:literal)*) => { $( round!($i); )* };
    }
    rounds!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
    st.abcd = vaddq_u32(st.abcd, s0);
    st.efgh = vaddq_u32(st.efgh, s1);
}

/// Compresses one block of every stream, interlaced at round granularity
///
/// The `for lane` loops are over a compile-time constant and unroll, so each
/// phase becomes a run of back-to-back independent instructions. Generic over
/// the stream count because the right number is the SHA pipe count times the
/// latency cover, which differs per core.
#[inline(always)]
unsafe fn compress_interleaved<const S: usize>(
    st: &mut [State; S],
    msg: &mut [[uint32x4_t; 4]; S],
) {
    let mut a = [vdupq_n_u32(0); S];
    let mut e = [vdupq_n_u32(0); S];
    for lane in 0..S {
        a[lane] = st[lane].abcd;
        e[lane] = st[lane].efgh;
    }

    for i in 0..16 {
        let slot = i & 3;
        let kv = vld1q_u32(K.as_ptr().add(i * 4));

        let mut wk = [vdupq_n_u32(0); S];
        for lane in 0..S {
            wk[lane] = vaddq_u32(msg[lane][slot], kv);
        }
        if i < 12 {
            for m in msg.iter_mut() {
                m[slot] = vsha256su0q_u32(m[slot], m[(i + 1) & 3]);
            }
        }
        // Two independent instructions per stream, so 2 * S issue before
        // anything has to wait.
        for lane in 0..S {
            let saved = a[lane];
            a[lane] = vsha256hq_u32(a[lane], e[lane], wk[lane]);
            e[lane] = vsha256h2q_u32(e[lane], saved, wk[lane]);
        }
        if i < 12 {
            for m in msg.iter_mut() {
                m[slot] = vsha256su1q_u32(m[slot], m[(i + 2) & 3], m[(i + 3) & 3]);
            }
        }
    }

    for lane in 0..S {
        st[lane].abcd = vaddq_u32(st[lane].abcd, a[lane]);
        st[lane].efgh = vaddq_u32(st[lane].efgh, e[lane]);
    }
}

/// Hashes one message on its own, for ragged tails with nothing to weave in
#[inline(always)]
unsafe fn hash_one(m: &Message<'_>, out: &mut [u8; 32]) {
    let mut st = State::init();
    let mut staging = [0u8; BLOCK];
    for k in 0..m.blocks() {
        let mut msg = if m.block_is_interior(k) {
            load_msg(m.interior_block(k))
        } else {
            m.fill_block(k, &mut staging);
            load_msg(&staging)
        };
        compress_block(&mut st, &mut msg);
    }
    *out = st.digest();
}

/// Hashes exactly `STREAMS` equal-length messages, chains interlaced
///
/// Both conditions matter: equal block counts keep every stream live for every
/// iteration, and a full group wastes no lane. `group` routes anything else to
/// the serial path.
#[inline(always)]
unsafe fn hash_uniform_group(msgs: &[Message<'_>], out: &mut [[u8; 32]], blocks: usize) {
    debug_assert_eq!(msgs.len(), STREAMS);

    let mut st = [State::init(); STREAMS];
    let mut staging = [[0u8; BLOCK]; STREAMS];
    let mut msg = [[vdupq_n_u32(0); 4]; STREAMS];

    let shape = Shape::of(msgs);
    let mut bases = [std::ptr::null::<u8>(); STREAMS];
    for (b, m) in bases.iter_mut().zip(msgs) {
        *b = m.body.as_ptr();
    }

    let staged0 = stage_prefix_block(msgs, &shape, &mut staging);
    for k in 0..blocks {
        if shape.same && k >= shape.k_lo && k < shape.k_hi {
            let off = k * BLOCK - shape.plen;
            for (slot, base) in msg.iter_mut().zip(bases.iter()) {
                // SAFETY: the interior bound documented on `Shape`; bodies
                // stay borrowed for the whole call.
                *slot = load_msg(std::slice::from_raw_parts(base.add(off), BLOCK));
            }
        } else if k == 0 && staged0 {
            for (slot, s) in msg.iter_mut().zip(staging.iter()) {
                *slot = load_msg(s);
            }
        } else {
            // Stage first so the loads below are all independent.
            let mut interior = [false; STREAMS];
            for (lane, m) in msgs.iter().enumerate() {
                interior[lane] = m.block_is_interior(k);
                if !interior[lane] {
                    m.fill_block(k, &mut staging[lane]);
                }
            }
            for (lane, m) in msgs.iter().enumerate() {
                msg[lane] = if interior[lane] {
                    load_msg(m.interior_block(k))
                } else {
                    load_msg(&staging[lane])
                };
            }
        }
        compress_interleaved(&mut st, &mut msg);
    }

    for (lane, o) in out.iter_mut().enumerate() {
        *o = st[lane].digest();
    }
}

/// one compression per link
///
/// The crypto extension keeps its state in the same native word order the next
/// block wants, so `digest`'s byte swap and `load_msg`'s undo of it both go
/// away and the state feeds straight back in. Words 8..16 are loop-invariant.
///
/// # Safety
///
/// The running CPU must support the ARMv8 SHA-256 extension (`sha2`).
#[target_feature(enable = "sha2")]
pub(crate) unsafe fn chain(seed: &[u8; 32], n: u64) -> [u8; 32] {
    let mut abcd = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(seed.as_ptr())));
    let mut efgh = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(seed.as_ptr().add(16))));
    let pad0 = vld1q_u32(crate::chain::PAD.as_ptr());
    let pad1 = vld1q_u32(crate::chain::PAD.as_ptr().add(4));

    for _ in 0..n {
        let mut st = State::init();
        let mut m = [abcd, efgh, pad0, pad1];
        compress_block(&mut st, &mut m);
        abcd = st.abcd;
        efgh = st.efgh;
    }

    let mut out = [0u8; 32];
    vst1q_u8(out.as_mut_ptr(), vrev32q_u8(vreinterpretq_u8_u32(abcd)));
    vst1q_u8(
        out.as_mut_ptr().add(16),
        vrev32q_u8(vreinterpretq_u8_u32(efgh)),
    );
    out
}

/// Advances `S` interlaced chains `n` links: this unit's step kernel
///
/// The serial chain leaves the SHA unit's issue slots mostly idle behind one
/// dependency chain; independent chains fill them, which is what replay has
/// and generation does not.
#[inline(always)]
unsafe fn steps_interlaced<const S: usize>(h: &mut [[u32; 8]], n: u64) {
    debug_assert_eq!(h.len(), S);

    let pad0 = vld1q_u32(crate::chain::PAD.as_ptr());
    let pad1 = vld1q_u32(crate::chain::PAD.as_ptr().add(4));

    let mut abcd = [vdupq_n_u32(0); S];
    let mut efgh = [vdupq_n_u32(0); S];
    for lane in 0..S {
        abcd[lane] = vld1q_u32(h[lane].as_ptr());
        efgh[lane] = vld1q_u32(h[lane].as_ptr().add(4));
    }

    for _ in 0..n {
        let mut st = [State::init(); S];
        let mut msg = [[vdupq_n_u32(0); 4]; S];
        for (m, (a, e)) in msg.iter_mut().zip(abcd.iter().zip(efgh.iter())) {
            *m = [*a, *e, pad0, pad1];
        }
        compress_interleaved(&mut st, &mut msg);
        for (lane, s) in st.iter().enumerate() {
            abcd[lane] = s.abcd;
            efgh[lane] = s.efgh;
        }
    }

    for lane in 0..S {
        vst1q_u32(h[lane].as_mut_ptr(), abcd[lane]);
        vst1q_u32(h[lane].as_mut_ptr().add(4), efgh[lane]);
    }
}

/// # Safety
///
/// The running CPU must support the ARMv8 SHA-256 extension (`sha2`).
#[target_feature(enable = "sha2")]
pub(crate) unsafe fn steps4(h: &mut [[u32; 8]], n: u64) {
    steps_interlaced::<STREAMS>(h, n)
}

/// Eight chains in flight: for cores with a second SHA-256 pipe
///
/// Four streams cover one pipe's `sha256h` latency exactly; a second pipe
/// needs eight, at the cost of spills the crypto latency hides.
///
/// # Safety
///
/// The running CPU must support the ARMv8 SHA-256 extension (`sha2`).
#[target_feature(enable = "sha2")]
pub(crate) unsafe fn steps8(h: &mut [[u32; 8]], n: u64) {
    steps_interlaced::<8>(h, n)
}

/// Hashes up to `STREAMS` messages: the unit of work the shared drivers
/// dispatch through
///
/// # Safety
///
/// The running CPU must support the ARMv8 SHA-256 extension (`sha2`).
#[target_feature(enable = "sha2")]
pub(crate) unsafe fn group(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
    debug_assert!(msgs.len() <= STREAMS);
    debug_assert_eq!(msgs.len(), out.len());
    let blocks = match msgs.first() {
        Some(m) => m.blocks(),
        None => return,
    };
    if msgs.len() == STREAMS && msgs.iter().all(|m| m.blocks() == blocks) {
        hash_uniform_group(msgs, out, blocks);
    } else {
        for (m, oi) in msgs.iter().zip(out.iter_mut()) {
            hash_one(m, oi);
        }
    }
}
