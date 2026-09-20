// MaxScore top-k disjunctive query evaluation (Turtle & Flood 1995, as
// formulated in Mallia et al.'s PISA work), doc-at-a-time with galloping
// skips into non-essential lists.
//
// Operates directly on v1 index slices: docids ascending per list (uint32),
// per-posting precomputed BM25 impacts (float32), per-list max impact.
//
// Exactness: identical result set to exhaustive evaluation (modulo
// floating-point addition order; accumulation in double to minimize it).
//
// Build: clang -O2 -shared -o libmaxscore.dylib maxscore.c
#include <stdint.h>
#include <string.h>

typedef struct {
    const uint32_t *ids;
    const float *imp;
    int64_t len;
    int64_t pos;
    double max_imp;
} List;

typedef struct { double score; uint32_t doc; } Hit;

// ---- min-heap of size k on score ----
static void heap_up(Hit *h, int i) {
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].score <= h[i].score) break;
        Hit t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}
static void heap_down(Hit *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l].score < h[m].score) m = l;
        if (r < n && h[r].score < h[m].score) m = r;
        if (m == i) break;
        Hit t = h[m]; h[m] = h[i]; h[i] = t; i = m;
    }
}

// gallop cursor of list l forward to first pos with ids[pos] >= d
static inline void gallop(List *l, uint32_t d) {
    int64_t pos = l->pos, len = l->len;
    if (pos >= len || l->ids[pos] >= d) return;
    int64_t step = 1, last = pos;
    while (pos + step < len && l->ids[pos + step] < d) {
        last = pos + step;
        step <<= 1;
    }
    int64_t hi = (pos + step < len) ? pos + step : len - 1;
    if (l->ids[hi] < d) { l->pos = len; return; }
    // binary search in (last, hi]
    int64_t lo = last;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (l->ids[mid] < d) lo = mid + 1; else hi = mid;
    }
    l->pos = lo;
}

// lists must be pre-sorted by max_imp ASCENDING by the caller.
// Returns number of hits written to out_ids/out_scores (best first).
// stats[0] = postings scored (essential + probed hits),
// stats[1] = candidates evaluated.
int64_t maxscore_query(const uint32_t **ids, const float **imp,
                       const int64_t *lens, const float *max_imps,
                       int32_t nterms, int32_t k,
                       uint32_t *out_ids, float *out_scores,
                       int64_t *stats) {
    if (nterms <= 0 || k <= 0) return 0;
    List lists[64];
    if (nterms > 64) nterms = 64;
    double prefix[64]; // prefix[i] = sum of max_imp[0..i]
    for (int i = 0; i < nterms; i++) {
        lists[i] = (List){ids[i], imp[i], lens[i], 0, (double)max_imps[i]};
        prefix[i] = max_imps[i] + (i ? prefix[i - 1] : 0.0);
    }
    Hit heap[1024];
    if (k > 1024) k = 1024;
    int nh = 0;
    double theta = -1.0;
    int pivot = 0; // lists [0, pivot) are non-essential
    int64_t scored = 0, cands = 0;

    for (;;) {
        // candidate: min current docid among essential lists
        uint32_t d = UINT32_MAX;
        for (int i = pivot; i < nterms; i++)
            if (lists[i].pos < lists[i].len && lists[i].ids[lists[i].pos] < d)
                d = lists[i].ids[lists[i].pos];
        if (d == UINT32_MAX) break;
        cands++;

        double score = 0.0;
        for (int i = pivot; i < nterms; i++) {
            List *l = &lists[i];
            if (l->pos < l->len && l->ids[l->pos] == d) {
                score += l->imp[l->pos];
                l->pos++;
                scored++;
            }
        }
        // try non-essential lists, highest bound first
        for (int i = pivot - 1; i >= 0; i--) {
            if (score + prefix[i] <= theta) break;
            List *l = &lists[i];
            gallop(l, d);
            if (l->pos < l->len && l->ids[l->pos] == d) {
                score += l->imp[l->pos];
                l->pos++;
                scored++;
            }
        }
        if (score > theta || nh < k) {
            if (nh < k) {
                heap[nh].score = score; heap[nh].doc = d;
                heap_up(heap, nh); nh++;
            } else if (score > heap[0].score) {
                heap[0].score = score; heap[0].doc = d;
                heap_down(heap, nh, 0);
            }
            if (nh == k) {
                theta = heap[0].score;
                while (pivot < nterms && prefix[pivot] <= theta) pivot++;
            }
        }
    }

    // pop min repeatedly -> fill output back-to-front => best first
    int64_t count = nh;
    for (int i = nh - 1; i >= 0; i--) {
        out_ids[i] = heap[0].doc;
        out_scores[i] = (float)heap[0].score;
        heap[0] = heap[nh - 1];
        nh--;
        if (nh > 0) heap_down(heap, nh, 0);
    }
    if (stats) { stats[0] = scored; stats[1] = cands; }
    return count;
}
