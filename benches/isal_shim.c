/* Benchmark-only shim over isa-l_crypto's multi-buffer SHA-256.
 *
 * Built only under the `isal-bench` feature. Nothing here is linked into the
 * library. isa-l_crypto is BSD-3-Clause and is neither vendored nor
 * distributed by this crate; shipping a binary built with this feature on
 * picks up its notice conditions.
 *
 * The shim exists so ISAL_SHA256_HASH_CTX's layout comes from the real header
 * rather than a hand-written repr(C) mirror. Getting that layout wrong does
 * not fail to link, it corrupts memory inside the manager, so the C side owns
 * it and Rust only sees opaque pointers.
 *
 * The manager and context pool are allocated once and reused, so allocation
 * stays out of the timed region. Manager init does not: isa-l requires it per
 * batch, so it is part of what a caller pays.
 */

#include <isa-l_crypto/sha256_mb.h>
#include <stdint.h>
#include <stdlib.h>

#define ISAL_BENCH_MAX_JOBS 64

/* Arch-specific entry points. These are exported from libisal_crypto.a but
 * declared only in include/internal/, which `make install` does not ship, so
 * they are redeclared here against the public types.
 *
 * The leading underscore is part of the C identifier, not a mangling artefact. */
extern void
_sha256_ctx_mgr_init_avx512(ISAL_SHA256_HASH_CTX_MGR *mgr);
extern ISAL_SHA256_HASH_CTX *
_sha256_ctx_mgr_submit_avx512(ISAL_SHA256_HASH_CTX_MGR *mgr, ISAL_SHA256_HASH_CTX *ctx,
                              const void *buf, uint32_t len, ISAL_HASH_CTX_FLAG flags);
extern ISAL_SHA256_HASH_CTX *
_sha256_ctx_mgr_flush_avx512(ISAL_SHA256_HASH_CTX_MGR *mgr);

extern void
_sha256_ctx_mgr_init_avx512_ni(ISAL_SHA256_HASH_CTX_MGR *mgr);
extern ISAL_SHA256_HASH_CTX *
_sha256_ctx_mgr_submit_avx512_ni(ISAL_SHA256_HASH_CTX_MGR *mgr, ISAL_SHA256_HASH_CTX *ctx,
                                 const void *buf, uint32_t len, ISAL_HASH_CTX_FLAG flags);
extern ISAL_SHA256_HASH_CTX *
_sha256_ctx_mgr_flush_avx512_ni(ISAL_SHA256_HASH_CTX_MGR *mgr);

typedef struct {
        ISAL_SHA256_HASH_CTX_MGR *mgr;
        ISAL_SHA256_HASH_CTX *pool;
} isal_bench;

void *
isal_bench_new(void)
{
        isal_bench *b = calloc(1, sizeof(*b));
        if (!b)
                return NULL;
        /* The header asks for 16-byte alignment; the digest arrays inside want
         * 64, and the kernels load them with aligned moves. */
        if (posix_memalign((void **) &b->mgr, 64, sizeof(*b->mgr)) != 0 ||
            posix_memalign((void **) &b->pool, 64, sizeof(*b->pool) * ISAL_BENCH_MAX_JOBS) != 0) {
                free(b);
                return NULL;
        }
        return b;
}

void
isal_bench_free(void *h)
{
        isal_bench *b = h;
        if (!b)
                return;
        free(b->mgr);
        free(b->pool);
        free(b);
}

/* isa-l leaves the digest as native-endian words; the canonical hash is those
 * words big-endian. isa-l's own tests compare word arrays and so never hit
 * this, which is why it is easy to miss. */
static void
store_be(uint8_t out[32], const uint32_t d[8])
{
        for (int i = 0; i < 8; i++) {
                out[i * 4 + 0] = (uint8_t) (d[i] >> 24);
                out[i * 4 + 1] = (uint8_t) (d[i] >> 16);
                out[i * 4 + 2] = (uint8_t) (d[i] >> 8);
                out[i * 4 + 3] = (uint8_t) (d[i]);
        }
}

#define ISAL_BENCH_RUN(suffix)                                                                     \
        int isal_bench_run##suffix(void *h, uint32_t n, const uint8_t *const *bufs,                 \
                                   const uint32_t *lens, uint8_t (*out)[32])                        \
        {                                                                                          \
                isal_bench *b = h;                                                                 \
                if (!b || n > ISAL_BENCH_MAX_JOBS)                                                 \
                        return -1;                                                                 \
                _sha256_ctx_mgr_init##suffix(b->mgr);                                              \
                for (uint32_t i = 0; i < n; i++) {                                                 \
                        isal_hash_ctx_init(&b->pool[i]);                                           \
                        b->pool[i].user_data = (void *) (uintptr_t) i;                             \
                        _sha256_ctx_mgr_submit##suffix(b->mgr, &b->pool[i], bufs[i], lens[i],       \
                                                       ISAL_HASH_ENTIRE);                           \
                }                                                                                  \
                while (_sha256_ctx_mgr_flush##suffix(b->mgr) != NULL) {                            \
                }                                                                                  \
                for (uint32_t i = 0; i < n; i++) {                                                 \
                        if (b->pool[i].status != ISAL_HASH_CTX_STS_COMPLETE)                       \
                                return -2;                                                         \
                        store_be(out[i], b->pool[i].job.result_digest);                            \
                }                                                                                  \
                return 0;                                                                          \
        }

ISAL_BENCH_RUN(_avx512)
ISAL_BENCH_RUN(_avx512_ni)

/* The dispatched path, for comparison: whichever kernel isa-l's own CPU
 * detection picks on this machine. On a part with both AVX-512 and SHA-NI that
 * is not necessarily the plain _avx512 one. */
int
isal_bench_run_dispatch(void *h, uint32_t n, const uint8_t *const *bufs, const uint32_t *lens,
                        uint8_t (*out)[32])
{
        isal_bench *b = h;
        ISAL_SHA256_HASH_CTX *done;

        if (!b || n > ISAL_BENCH_MAX_JOBS)
                return -1;
        if (isal_sha256_ctx_mgr_init(b->mgr) != 0)
                return -3;
        for (uint32_t i = 0; i < n; i++) {
                isal_hash_ctx_init(&b->pool[i]);
                b->pool[i].user_data = (void *) (uintptr_t) i;
                if (isal_sha256_ctx_mgr_submit(b->mgr, &b->pool[i], &done, bufs[i], lens[i],
                                               ISAL_HASH_ENTIRE) != 0)
                        return -3;
        }
        do {
                if (isal_sha256_ctx_mgr_flush(b->mgr, &done) != 0)
                        return -3;
        } while (done != NULL);
        for (uint32_t i = 0; i < n; i++) {
                if (b->pool[i].status != ISAL_HASH_CTX_STS_COMPLETE)
                        return -2;
                store_be(out[i], b->pool[i].job.result_digest);
        }
        return 0;
}
