// Converter: raw postings (docids u32, quantized impacts u8, term offsets
// i64) -> block-compressed format.
//
// Per 128-posting block (last block of a term may be short, count c):
//   stored delta v_i = docid_i - prev_docid - 1, where prev for the first
//   posting of a BLOCK is block_last[prev_block] (or -1 at term start), so
//   any block decodes independently given the skip arrays.
//   data: ceil(c*w/8) bytes of LSB-first bitpacked v (w = bits of max v),
//   then c bytes of u8 impacts.
// Skip arrays (one entry per block): last docid u32, width u8, max-impact u8.
//
// Inputs are read IN PLACE from .npy files via a byte offset (the array's
// data start): copying them to temp .raw files needed ~1.8GB of scratch and
// filled the disk on this machine (notes/13). Never duplicate what you can
// address.
//
// Usage: compress <docids.npy> <off> <impq.npy> <off> <offsets.npy> <off>
//                 <n_terms> <out_blob> <out_last.u32> <out_width.u8>
//                 <out_maxq.u8>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define BLK 128

static void *load(const char *p, long data_off, size_t *sz) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    *sz = (size_t)(end - data_off);
    fseek(f, data_off, SEEK_SET);
    void *b = malloc(*sz);
    if (fread(b, 1, *sz, f) != *sz) { perror("fread"); exit(1); }
    fclose(f);
    return b;
}

static inline int width_of(uint32_t v) {
    int w = 0;
    while (v) { w++; v >>= 1; }
    return w;
}

int main(int argc, char **argv) {
    if (argc != 12) { fprintf(stderr, "args\n"); return 2; }
    size_t sz;
    uint32_t *docids = load(argv[1], strtol(argv[2], 0, 10), &sz);
    uint8_t *impq = load(argv[3], strtol(argv[4], 0, 10), &sz);
    int64_t *offs = load(argv[5], strtol(argv[6], 0, 10), &sz);
    int64_t n_terms = strtoll(argv[7], 0, 10);

    FILE *fblob = fopen(argv[8], "wb");
    FILE *flast = fopen(argv[9], "wb");
    FILE *fwidth = fopen(argv[10], "wb");
    FILE *fmaxq = fopen(argv[11], "wb");
    if (!fblob || !flast || !fwidth || !fmaxq) { perror("open out"); return 1; }
    // buffered output for blob
    setvbuf(fblob, NULL, _IOFBF, 8 << 20);

    uint32_t vbuf[BLK];
    uint8_t pack[BLK * 4 + 8];
    uint64_t nblocks = 0;

    for (int64_t t = 0; t < n_terms; t++) {
        int64_t s = offs[t], e = offs[t + 1];
        int64_t prev = -1;
        for (int64_t bs = s; bs < e; bs += BLK) {
            int64_t c = e - bs < BLK ? e - bs : BLK;
            uint32_t maxv = 0; uint8_t mq = 0;
            for (int64_t i = 0; i < c; i++) {
                uint32_t d = docids[bs + i];
                vbuf[i] = (uint32_t)(d - prev - 1);
                if (vbuf[i] > maxv) maxv = vbuf[i];
                if (impq[bs + i] > mq) mq = impq[bs + i];
                prev = d;
            }
            uint8_t w = (uint8_t)width_of(maxv);
            uint32_t lastd = (uint32_t)prev;
            // pack LSB-first
            uint64_t acc = 0; int bits = 0; size_t nb = 0;
            for (int64_t i = 0; i < c; i++) {
                acc |= (uint64_t)vbuf[i] << bits;
                bits += w;
                while (bits >= 8) { pack[nb++] = (uint8_t)acc; acc >>= 8; bits -= 8; }
            }
            if (bits) pack[nb++] = (uint8_t)acc;
            fwrite(pack, 1, nb, fblob);
            fwrite(impq + bs, 1, c, fblob);
            fwrite(&lastd, 4, 1, flast);
            fwrite(&w, 1, 1, fwidth);
            fwrite(&mq, 1, 1, fmaxq);
            nblocks++;
        }
    }
    fclose(fblob); fclose(flast); fclose(fwidth); fclose(fmaxq);
    printf("%llu\n", (unsigned long long)nblocks);
    return 0;
}
