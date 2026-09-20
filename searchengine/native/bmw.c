// MaxScore top-k over the block-compressed index (see compress.c format).
// Integer scoring domain: impacts are u8-quantized; scores are u32 sums.
// Block skip data (last docid / width / max impact per block) lets galloping
// jump whole blocks WITHOUT decoding them — decompression cost is only paid
// for blocks that can matter.
//
// Build: clang -O2 -shared -o libbmw.dylib bmw.c
#include <stdint.h>
#include <string.h>

#define BLK 128

typedef struct {
    // static per-term info
    const uint8_t *blob;
    const uint32_t *blast;   // block last-docid, global array
    const uint8_t *bwidth;
    const uint8_t *bmaxq_arr;
    const uint64_t *boff;    // block data offset in blob
    int64_t b0, nb;          // first block (global idx), #blocks
    int64_t df;
    uint32_t maxq;
    // cursor
    int64_t cb;              // current block idx relative to b0; nb = exhausted
    int c, pos;              // count in decoded block, position
    uint32_t ids[BLK];
    uint8_t imps[BLK];
} Term;

static _Thread_local int64_t g_decoded;

static void decode_block(Term *t) {
    g_decoded++;
    int64_t gb = t->b0 + t->cb;
    int64_t remaining = t->df - t->cb * BLK;
    int c = remaining < BLK ? (int)remaining : BLK;
    int w = t->bwidth[gb];
    const uint8_t *p = t->blob + t->boff[gb];
    int64_t prev = (t->cb == 0) ? -1 : (int64_t)t->blast[gb - 1];
    uint64_t acc = 0; int bits = 0;
    uint32_t mask = w == 32 ? 0xFFFFFFFFu : ((1u << w) - 1);
    for (int i = 0; i < c; i++) {
        while (bits < w) { acc |= (uint64_t)(*p++) << bits; bits += 8; }
        uint32_t v = (uint32_t)(acc & mask);
        acc >>= w; bits -= w;
        prev = prev + 1 + v;
        t->ids[i] = (uint32_t)prev;
    }
    size_t packed = ((size_t)c * w + 7) / 8;
    memcpy(t->imps, t->blob + t->boff[gb] + packed, c);
    t->c = c; t->pos = 0;
}

static inline int exhausted(const Term *t) { return t->cb >= t->nb; }
static inline uint32_t cur_id(const Term *t) { return t->ids[t->pos]; }

static void advance(Term *t) {
    if (++t->pos >= t->c) {
        if (++t->cb < t->nb) decode_block(t);
    }
}

// locate (WITHOUT decoding) the block that could contain docid >= d.
// Returns the block's max quantized impact, or -1 if list exhausted / d is
// inside the already-decoded current block (cheap path — no bound check
// needed, decode already paid). Sets *blk to the located block index.
static int block_for(Term *t, uint32_t d, int64_t *blk) {
    if (exhausted(t)) return -1;
    if (t->pos < t->c && t->ids[t->c - 1] >= d) { *blk = t->cb; return -1; }
    int64_t lo = t->cb + 1, hi = t->nb - 1;
    if (lo > hi || t->blast[t->b0 + hi] < d) { t->cb = t->nb; return -1; }
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (t->blast[t->b0 + mid] < d) lo = mid + 1; else hi = mid;
    }
    *blk = lo;
    return t->bmaxq_arr[t->b0 + lo];
}

// position cursor at first posting >= d inside block blk (decodes if needed)
static void land(Term *t, int64_t blk, uint32_t d) {
    int l;
    if (blk != t->cb) {
        t->cb = blk;
        decode_block(t);
        l = 0;
    } else {
        l = t->pos;
    }
    int h = t->c - 1;
    while (l < h) {
        int mid = (l + h) >> 1;
        if (t->ids[mid] < d) l = mid + 1; else h = mid;
    }
    t->pos = l;
}

typedef struct { uint32_t score; uint32_t doc; } Hit;
static void heap_down(Hit *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l].score < h[m].score) m = l;
        if (r < n && h[r].score < h[m].score) m = r;
        if (m == i) break;
        Hit tmp = h[m]; h[m] = h[i]; h[i] = tmp; i = m;
    }
}
static void heap_up(Hit *h, int i) {
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].score <= h[i].score) break;
        Hit t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}

// terms pre-sorted by maxq ASCENDING. Returns hit count; scores are
// quantized-integer sums (caller rescales).
int64_t bmw_query(const uint8_t *blob, const uint32_t *blast,
                  const uint8_t *bwidth, const uint8_t *bmaxq,
                  const uint64_t *boff,
                  const int64_t *tb0, const int64_t *tnb,
                  const int64_t *tdf, const uint32_t *tmaxq,
                  int32_t nterms, int32_t k,
                  uint32_t *out_ids, uint32_t *out_scores, int64_t *stats) {
    if (nterms <= 0 || k <= 0) return 0;
    if (nterms > 64) nterms = 64;
    g_decoded = 0;
    Term terms[64];
    uint64_t prefix[64];
    for (int i = 0; i < nterms; i++) {
        Term *t = &terms[i];
        t->blob = blob; t->blast = blast; t->bwidth = bwidth;
        t->bmaxq_arr = bmaxq; t->boff = boff;
        t->b0 = tb0[i]; t->nb = tnb[i]; t->df = tdf[i]; t->maxq = tmaxq[i];
        t->cb = 0; t->pos = 0; t->c = 0;
        decode_block(t);
        prefix[i] = tmaxq[i] + (i ? prefix[i - 1] : 0);
    }
    Hit heap[1024];
    if (k > 1024) k = 1024;
    int nh = 0;
    uint64_t theta = 0;   // strictly-greater threshold once heap full
    int have_theta = 0;
    int pivot = 0;
    int64_t cands = 0;

    for (;;) {
        uint32_t d = 0xFFFFFFFFu;
        for (int i = pivot; i < nterms; i++)
            if (!exhausted(&terms[i]) && cur_id(&terms[i]) < d)
                d = cur_id(&terms[i]);
        if (d == 0xFFFFFFFFu) break;
        cands++;

        uint64_t score = 0;
        for (int i = pivot; i < nterms; i++) {
            Term *t = &terms[i];
            if (!exhausted(t) && cur_id(t) == d) {
                score += t->imps[t->pos];
                advance(t);
            }
        }
        for (int i = pivot - 1; i >= 0; i--) {
            if (have_theta && score + prefix[i] <= theta) break;
            Term *t = &terms[i];
            int64_t blk;
            int bmax = block_for(t, d, &blk);
            if (exhausted(t)) continue;
            // block-max shallow check: even if d is present in the target
            // block, its impact <= bmax; if that can't lift the total past
            // theta (with all lower-bounded terms at their term max), skip
            // the decode entirely.
            if (bmax >= 0 && have_theta) {
                uint64_t bound = score + (uint64_t)bmax
                                 + (i ? prefix[i - 1] : 0);
                if (bound <= theta) continue;
            }
            land(t, blk, d);
            if (cur_id(t) == d) {
                score += t->imps[t->pos];
                advance(t);
            }
        }
        if (nh < k) {
            heap[nh].score = (uint32_t)score; heap[nh].doc = d;
            heap_up(heap, nh); nh++;
            if (nh == k) {
                have_theta = 1;
                theta = heap[0].score;
                while (pivot < nterms && prefix[pivot] <= theta) pivot++;
            }
        } else if (score > theta) {
            heap[0].score = (uint32_t)score; heap[0].doc = d;
            heap_down(heap, nh, 0);
            theta = heap[0].score;
            while (pivot < nterms && prefix[pivot] <= theta) pivot++;
        }
    }

    int64_t count = nh;
    for (int i = nh - 1; i >= 0; i--) {
        out_ids[i] = heap[0].doc;
        out_scores[i] = heap[0].score;
        heap[0] = heap[nh - 1];
        nh--;
        if (nh > 0) heap_down(heap, nh, 0);
    }
    if (stats) { stats[0] = g_decoded; stats[1] = cands; }
    return count;
}

// ---------------------------------------------------------------------
// Conjunctive intersection: every docid present in ALL given terms.
//
// Used for exact phrase search. A document containing a phrase must
// contain all of the phrase's terms, so this yields a provably COMPLETE
// candidate set (no recall loss), which the caller then verifies against
// the stored document text. Cheapest term drives the scan; the rest are
// probed with block-skipping galloping seeks.
//
// terms may be given in any order; lens/df arrays parallel to tb0/tnb.
// Returns number of docids written (stops at max_out).
int64_t bmw_intersect(const uint8_t *blob, const uint32_t *blast,
                      const uint8_t *bwidth, const uint8_t *bmaxq,
                      const uint64_t *boff,
                      const int64_t *tb0, const int64_t *tnb,
                      const int64_t *tdf,
                      int32_t nterms, uint32_t *out, int64_t max_out) {
    if (nterms <= 0 || nterms > 64) return 0;
    g_decoded = 0;
    Term terms[64];
    int order[64];
    for (int i = 0; i < nterms; i++) {
        Term *t = &terms[i];
        t->blob = blob; t->blast = blast; t->bwidth = bwidth;
        t->bmaxq_arr = bmaxq; t->boff = boff;
        t->b0 = tb0[i]; t->nb = tnb[i]; t->df = tdf[i]; t->maxq = 0;
        t->cb = 0; t->pos = 0; t->c = 0;
        if (t->nb <= 0) return 0;   // empty list => empty intersection
        decode_block(t);
        order[i] = i;
    }
    // drive the scan from the shortest list
    for (int i = 1; i < nterms; i++)
        for (int j = i; j > 0 && terms[order[j]].df < terms[order[j-1]].df; j--) {
            int tmp = order[j]; order[j] = order[j-1]; order[j-1] = tmp;
        }
    Term *lead = &terms[order[0]];
    int64_t n_out = 0;
    while (!exhausted(lead) && n_out < max_out) {
        uint32_t d = cur_id(lead);
        int ok = 1;
        for (int i = 1; i < nterms && ok; i++) {
            Term *t = &terms[order[i]];
            int64_t blk;
            block_for(t, d, &blk);
            if (exhausted(t)) { ok = -1; break; }   // -1: whole scan is done
            land(t, blk, d);
            if (exhausted(t)) { ok = -1; break; }
            if (cur_id(t) != d) ok = 0;
        }
        if (ok == -1) break;
        if (ok) out[n_out++] = d;
        advance(lead);
    }
    return n_out;
}
