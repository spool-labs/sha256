//! One message hashed in pieces, for input that is never in memory whole

use crate::{
    batch::BLOCK,
    chain::{select, Kernel},
    core::{compress, H0},
    lanes::{Lanes, Scalar},
};

/// Bytes the padding needs after the 0x80 marker for the bit length
const LENGTH_BYTES: usize = 8;

/// SHA-256 of a message that arrives in pieces
///
/// Runs on the CPU's SHA-256 unit when it has one, like `hash_chain`.
///
/// ```
/// use tape_sha256::Sha256;
///
/// let mut hasher = Sha256::new();
/// hasher.update(b"hello ");
/// hasher.update(b"world");
/// let digest = hasher.finish();
/// ```
#[derive(Clone)]
pub struct Sha256 {
    kernel: Kernel,
    state: [u32; 8],
    pending: [u8; BLOCK],
    pending_len: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// A hasher with nothing fed yet
    pub fn new() -> Self {
        Self::with_kernel(select())
    }

    /// Feeds the next piece of the message
    pub fn update(&mut self, data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        let mut rest = data;

        if self.pending_len > 0 {
            let taken = (BLOCK - self.pending_len).min(rest.len());
            self.pending[self.pending_len..self.pending_len + taken]
                .copy_from_slice(&rest[..taken]);
            self.pending_len += taken;
            rest = &rest[taken..];
            if self.pending_len < BLOCK {
                return;
            }
            compress_blocks(self.kernel, &mut self.state, &self.pending);
            self.pending_len = 0;
        }

        let whole = rest.len() - rest.len() % BLOCK;
        if whole > 0 {
            compress_blocks(self.kernel, &mut self.state, &rest[..whole]);
        }
        let tail = &rest[whole..];
        self.pending[..tail.len()].copy_from_slice(tail);
        self.pending_len = tail.len();
    }

    /// The digest of every piece fed so far
    pub fn finish(mut self) -> [u8; 32] {
        let mut padded = [0u8; 2 * BLOCK];
        padded[..self.pending_len].copy_from_slice(&self.pending[..self.pending_len]);
        padded[self.pending_len] = 0x80;
        let padded_len = match self.pending_len < BLOCK - LENGTH_BYTES {
            true => BLOCK,
            false => 2 * BLOCK,
        };
        let bit_len = self.total_len.wrapping_mul(8);
        padded[padded_len - LENGTH_BYTES..padded_len].copy_from_slice(&bit_len.to_be_bytes());
        compress_blocks(self.kernel, &mut self.state, &padded[..padded_len]);

        let mut digest = [0u8; 32];
        for (bytes, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *bytes = word.to_be_bytes();
        }
        digest
    }

    fn with_kernel(kernel: Kernel) -> Self {
        Self {
            kernel,
            state: H0,
            pending: [0u8; BLOCK],
            pending_len: 0,
            total_len: 0,
        }
    }
}

fn compress_blocks(kernel: Kernel, state: &mut [u32; 8], blocks: &[u8]) {
    match kernel {
        Kernel::Portable => compress_portable(state, blocks),
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        // SAFETY: `select` returns this arm only when the probe passed.
        Kernel::ShaNi => unsafe { crate::shani::compress_blocks(state, blocks) },
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        // SAFETY: as above.
        Kernel::NeonSha2 => unsafe { crate::neon_sha2::compress_blocks(state, blocks) },
    }
}

fn compress_portable(state: &mut [u32; 8], blocks: &[u8]) {
    let mut lanes = state.map(Scalar::<1>::splat);
    for block in blocks.as_chunks::<BLOCK>().0 {
        let mut words = [Scalar::<1>::splat(0); 16];
        for (word, bytes) in words.iter_mut().zip(block.as_chunks::<4>().0) {
            *word = Scalar::<1>::splat(u32::from_be_bytes(*bytes));
        }
        compress(&mut lanes, words);
    }
    for (word, lane) in state.iter_mut().zip(lanes) {
        *word = lane.0[0];
    }
}

#[cfg(test)]
mod tests {
    use {super::*, sha2::Digest};

    fn kernels() -> Vec<Kernel> {
        let mut kernels = vec![Kernel::Portable];
        if select() != Kernel::Portable {
            kernels.push(select());
        }
        kernels
    }

    fn reference(message: &[u8]) -> [u8; 32] {
        sha2::Sha256::digest(message).into()
    }

    fn message(len: usize) -> Vec<u8> {
        (0..len)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
            .collect()
    }

    // every length across the padding and block edges matches, fed whole
    #[test]
    fn whole_messages() {
        for kernel in kernels() {
            for len in (0..300).chain([1_000, 4_096, 65_537]) {
                let data = message(len);
                let mut hasher = Sha256::with_kernel(kernel);

                hasher.update(&data);

                assert_eq!(
                    hasher.finish(),
                    reference(&data),
                    "{} len {len}",
                    kernel.name()
                );
            }
        }
    }

    // any split of the message gives the same digest as feeding it whole
    #[test]
    fn split_pieces() {
        let data = message(1_000);
        let expected = reference(&data);
        for kernel in kernels() {
            for piece in [1, 3, 55, 56, 63, 64, 65, 127, 128, 999] {
                let mut hasher = Sha256::with_kernel(kernel);

                for part in data.chunks(piece) {
                    hasher.update(part);
                }

                assert_eq!(hasher.finish(), expected, "{} piece {piece}", kernel.name());
            }
        }
    }

    // empty pieces between real ones change nothing
    #[test]
    fn empty_pieces() {
        let data = message(130);
        let mut hasher = Sha256::new();

        hasher.update(&[]);
        hasher.update(&data[..70]);
        hasher.update(&[]);
        hasher.update(&data[70..]);

        assert_eq!(hasher.finish(), reference(&data));
    }

    // the FIPS 180-2 one-million-a vector, fed in odd pieces
    #[test]
    fn million_a() {
        let data = vec![b'a'; 1_000_000];
        let mut hasher = Sha256::new();

        for part in data.chunks(997) {
            hasher.update(part);
        }

        let expected = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
        let digest: String = hasher
            .finish()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest, expected);
    }
}
