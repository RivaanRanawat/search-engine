// Native corpus scanner: tokenize + (optional Porter stem) + tf-count +
// build doc-ordered posting buffers, for one byte-shard of collection.tsv.
//
// Key designs:
// - Tokens: runs of [a-z0-9] after ASCII tolower; bytes >= 0x80 are
//   separators. (Divergence from Python str.lower() only for rare Unicode
//   uppercase whose lowercase contains ASCII — quantified in experiment.)
// - Two-level hash: raw term -> stem entry, so Porter runs once per unique
//   raw term per shard, never per token.
// - tf counting without a per-doc map: each stem entry carries
//   (cur_doc, cur_tf); postings flushed lazily on doc change -> lists are
//   naturally doc-ordered because pids in the file are ascending.
// - Shard docids come from the pid column itself: byte-splitting the file
//   needs no docid coordination between workers.
//
// Usage:
//   scanner scan  <tsv> <start_off> <end_off> <out.shard> <stem:0|1>
//   scanner stemtest        (words on stdin -> stems on stdout)
//
// Shard file layout (all little-endian):
//   u64 magic 0x53484152445F3032 ("SHARD_02")
//   u32 stem_flag, u32 min_pid, u32 n_docs, u32 n_terms, u64 total_postings
//   u64 names_bytes
//   u16 doclens[n_docs]
//   u16 name_len[n_terms]; bytes names[] (concatenated);
//   u32 df[n_terms]
//   u32 packed[total]  (docid<<6 | min(tf,63)), term-major
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ---------------- Porter stemmer (port of searchengine/porter.py) ---------
static int is_cons(const char *w, int i) {
    char c = w[i];
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return 0;
    if (c == 'y') return i == 0 ? 1 : !is_cons(w, i - 1);
    return 1;
}
static int measure(const char *w, int len) {
    int n = 0, i = 0;
    while (i < len && is_cons(w, i)) i++;
    while (i < len) {
        while (i < len && !is_cons(w, i)) i++;
        if (i == len) break;
        n++;
        while (i < len && is_cons(w, i)) i++;
    }
    return n;
}
static int vowel_in(const char *w, int len) {
    for (int i = 0; i < len; i++) if (!is_cons(w, i)) return 1;
    return 0;
}
static int doublec(const char *w, int len) {
    return len >= 2 && w[len-1] == w[len-2] && is_cons(w, len-1);
}
static int cvc(const char *w, int len) {
    if (len < 3) return 0;
    if (!(is_cons(w, len-3) && !is_cons(w, len-2) && is_cons(w, len-1))) return 0;
    char c = w[len-1];
    return c != 'w' && c != 'x' && c != 'y';
}
static int ends(const char *w, int len, const char *suf, int slen) {
    return len >= slen && memcmp(w + len - slen, suf, slen) == 0;
}
typedef struct { const char *suf; int slen; const char *rep; int rlen; } Rule;
static const Rule STEP2[] = {
    {"ational",7,"ate",3},{"tional",6,"tion",4},{"enci",4,"ence",4},
    {"anci",4,"ance",4},{"izer",4,"ize",3},{"bli",3,"ble",3},{"alli",4,"al",2},
    {"entli",5,"ent",3},{"eli",3,"e",1},{"ousli",5,"ous",3},{"ization",7,"ize",3},
    {"ation",5,"ate",3},{"ator",4,"ate",3},{"alism",5,"al",2},
    {"iveness",7,"ive",3},{"fulness",7,"ful",3},{"ousness",7,"ous",3},
    {"aliti",5,"al",2},{"iviti",5,"ive",3},{"biliti",6,"ble",3},{"logi",4,"log",3}};
static const Rule STEP3[] = {
    {"icate",5,"ic",2},{"ative",5,"",0},{"alize",5,"al",2},{"iciti",5,"ic",2},
    {"ical",4,"ic",2},{"ful",3,"",0},{"ness",4,"",0}};
static const char *STEP4[] = {"al","ance","ence","er","ic","able","ible","ant",
    "ement","ment","ent","ion","ou","ism","ate","iti","ous","ive","ize"};

// stems w (len) in place, returns new length. w must be writable, cap >= len+2.
static int porter(char *w, int len) {
    if (len <= 2) return len;
    // 1a
    if (ends(w,len,"sses",4)) len -= 2;
    else if (ends(w,len,"ies",3)) len -= 2;
    else if (w[len-1]=='s' && !(len>=2 && w[len-2]=='s')) len -= 1;
    // 1b
    if (ends(w,len,"eed",3)) {
        if (measure(w,len-3) > 0) len -= 1;
    } else {
        int cut = 0;
        if (ends(w,len,"ed",2) && vowel_in(w,len-2)) cut = 2;
        else if (ends(w,len,"ing",3) && vowel_in(w,len-3)) cut = 3;
        if (cut) {
            len -= cut;
            if (ends(w,len,"at",2)||ends(w,len,"bl",2)||ends(w,len,"iz",2))
                { w[len++]='e'; }
            else if (doublec(w,len) && w[len-1]!='l' && w[len-1]!='s' && w[len-1]!='z')
                len -= 1;
            else if (measure(w,len)==1 && cvc(w,len)) { w[len++]='e'; }
        }
    }
    // 1c
    if (w[len-1]=='y' && vowel_in(w,len-1)) w[len-1]='i';
    // 2
    for (unsigned i=0;i<sizeof(STEP2)/sizeof(Rule);i++) {
        const Rule *r=&STEP2[i];
        if (ends(w,len,r->suf,r->slen)) {
            if (measure(w,len-r->slen) > 0) {
                memcpy(w+len-r->slen, r->rep, r->rlen);
                len = len - r->slen + r->rlen;
            }
            break;
        }
    }
    // 3
    for (unsigned i=0;i<sizeof(STEP3)/sizeof(Rule);i++) {
        const Rule *r=&STEP3[i];
        if (ends(w,len,r->suf,r->slen)) {
            if (measure(w,len-r->slen) > 0) {
                memcpy(w+len-r->slen, r->rep, r->rlen);
                len = len - r->slen + r->rlen;
            }
            break;
        }
    }
    // 4
    for (unsigned i=0;i<sizeof(STEP4)/sizeof(char*);i++) {
        int slen = (int)strlen(STEP4[i]);
        if (ends(w,len,STEP4[i],slen)) {
            int stlen = len - slen;
            if (measure(w,stlen) > 1) {
                if (slen==3 && memcmp(STEP4[i],"ion",3)==0) {
                    if (stlen>0 && (w[stlen-1]=='s'||w[stlen-1]=='t')) len = stlen;
                } else len = stlen;
            }
            break;
        }
    }
    // 5a
    if (w[len-1]=='e') {
        int m = measure(w,len-1);
        if (m > 1 || (m==1 && !cvc(w,len-1))) len -= 1;
    }
    // 5b
    if (measure(w,len)>1 && doublec(w,len) && w[len-1]=='l') len -= 1;
    return len;
}

// ---------------- hash tables ----------------
static inline uint64_t fnv1a(const char *s, int len) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
    return h;
}

typedef struct {       // stem-level entry: owns postings
    uint32_t name_off, cur_doc, cur_tf;
    uint32_t *buf; uint32_t blen, bcap;
    uint16_t name_len;
} StemEntry;
typedef struct {       // raw-term entry: points to stem entry
    uint32_t name_off; uint32_t stem_id; uint16_t name_len; uint8_t used;
} RawEntry;

#define RAW_CAP  (1u<<23)   // 8.4M slots
#define STEM_CAP (1u<<22)   // 4.2M slots
static RawEntry *raw_tab;
static uint32_t *stem_slot;      // slot -> stem_id+1 (0 = empty)
static StemEntry *stems;
static uint32_t n_stems = 0;
static char *arena; static size_t arena_len = 0, arena_cap = 0;
#define NO_DOC 0xFFFFFFFFu

static uint32_t arena_put(const char *s, int len) {
    if (arena_len + len > arena_cap) {
        arena_cap = arena_cap ? arena_cap*2 : (64u<<20);
        arena = realloc(arena, arena_cap);
    }
    memcpy(arena + arena_len, s, len);
    arena_len += len;
    return (uint32_t)(arena_len - len);
}

static uint32_t stem_lookup(const char *s, int len) {
    uint64_t h = fnv1a(s, len);
    uint32_t mask = STEM_CAP - 1, i = h & mask;
    for (;;) {
        uint32_t v = stem_slot[i];
        if (!v) {
            StemEntry *e = &stems[n_stems];
            e->name_off = arena_put(s, len);
            e->name_len = (uint16_t)len;
            e->cur_doc = NO_DOC; e->cur_tf = 0;
            e->buf = NULL; e->blen = 0; e->bcap = 0;
            stem_slot[i] = ++n_stems;
            return n_stems - 1;
        }
        StemEntry *e = &stems[v-1];
        if (e->name_len == len && memcmp(arena + e->name_off, s, len) == 0)
            return v - 1;
        i = (i + 1) & mask;
    }
}

static uint32_t raw_lookup(const char *s, int len, int do_stem) {
    uint64_t h = fnv1a(s, len);
    uint32_t mask = RAW_CAP - 1, i = h & mask;
    for (;;) {
        RawEntry *e = &raw_tab[i];
        if (!e->used) {
            e->used = 1;
            e->name_off = arena_put(s, len);
            e->name_len = (uint16_t)len;
            if (do_stem && len < 250) {
                char tmp[256];
                memcpy(tmp, s, len);
                int sl = porter(tmp, len);
                e->stem_id = stem_lookup(tmp, sl);
            } else {
                e->stem_id = stem_lookup(s, len);
            }
            return e->stem_id;
        }
        if (e->name_len == len && memcmp(arena + e->name_off, s, len) == 0)
            return e->stem_id;
        i = (i + 1) & mask;
    }
}

static inline void flush_entry(StemEntry *e) {
    if (e->cur_doc == NO_DOC) return;
    if (e->blen == e->bcap) {
        e->bcap = e->bcap ? e->bcap * 2 : 4;
        e->buf = realloc(e->buf, e->bcap * 4);
    }
    uint32_t tf = e->cur_tf > 63 ? 63 : e->cur_tf;
    e->buf[e->blen++] = (e->cur_doc << 6) | tf;
    e->cur_doc = NO_DOC; e->cur_tf = 0;
}

// ---------------- main scan ----------------
static int do_scan(const char *path, size_t start, size_t end,
                   const char *out_path, int do_stem) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    if (end > (size_t)st.st_size) end = st.st_size;
    char *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(base + start, end - start, MADV_SEQUENTIAL);

    raw_tab = calloc(RAW_CAP, sizeof(RawEntry));
    stem_slot = calloc(STEM_CAP, 4);
    stems = malloc((size_t)STEM_CAP * sizeof(StemEntry));

    uint16_t *doclens = NULL; size_t dl_len = 0, dl_cap = 0;
    uint32_t min_pid = 0; int have_min = 0;

    const char *p = base + start, *lim = base + end;
    // if not at file start, caller guarantees p is at a line start
    while (p < lim) {
        // parse pid
        uint32_t pid = 0;
        while (p < lim && *p != '\t') {
            pid = pid * 10 + (uint32_t)(*p - '0');
            p++;
        }
        if (p < lim) p++; // skip tab
        if (!have_min) { min_pid = pid; have_min = 1; }
        // tokenize text until newline
        uint32_t ntok = 0;
        char tok[512]; int tl = 0;
        while (p < lim && *p != '\n') {
            unsigned char c = (unsigned char)*p++;
            if (c >= 'A' && c <= 'Z') c += 32;
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
                if (tl < 511) tok[tl++] = (char)c;
            } else if (tl) {
                uint32_t sid = raw_lookup(tok, tl, do_stem);
                StemEntry *e = &stems[sid];
                if (e->cur_doc != pid) { flush_entry(e); e->cur_doc = pid; e->cur_tf = 1; }
                else e->cur_tf++;
                ntok++; tl = 0;
            }
        }
        if (tl) {
            uint32_t sid = raw_lookup(tok, tl, do_stem);
            StemEntry *e = &stems[sid];
            if (e->cur_doc != pid) { flush_entry(e); e->cur_doc = pid; e->cur_tf = 1; }
            else e->cur_tf++;
            ntok++;
        }
        if (p < lim) p++; // newline
        if (dl_len == dl_cap) {
            dl_cap = dl_cap ? dl_cap * 2 : 1u<<20;
            doclens = realloc(doclens, dl_cap * 2);
        }
        doclens[dl_len++] = ntok > 65535 ? 65535 : (uint16_t)ntok;
    }

    // flush all pending postings; count totals
    uint64_t total = 0;
    for (uint32_t i = 0; i < n_stems; i++) {
        flush_entry(&stems[i]);
        total += stems[i].blen;
    }

    uint64_t names_bytes = 0;
    for (uint32_t i = 0; i < n_stems; i++) names_bytes += stems[i].name_len;

    FILE *out = fopen(out_path, "wb");
    if (!out) { perror("fopen"); return 1; }
    uint64_t magic = 0x53484152445F3032ULL;
    uint32_t hdr[4] = {(uint32_t)do_stem, min_pid, (uint32_t)dl_len, n_stems};
    fwrite(&magic, 8, 1, out);
    fwrite(hdr, 4, 4, out);
    fwrite(&total, 8, 1, out);
    fwrite(&names_bytes, 8, 1, out);
    fwrite(doclens, 2, dl_len, out);
    for (uint32_t i = 0; i < n_stems; i++)
        fwrite(&stems[i].name_len, 2, 1, out);
    for (uint32_t i = 0; i < n_stems; i++)
        fwrite(arena + stems[i].name_off, 1, stems[i].name_len, out);
    for (uint32_t i = 0; i < n_stems; i++)
        fwrite(&stems[i].blen, 4, 1, out);
    for (uint32_t i = 0; i < n_stems; i++)
        fwrite(stems[i].buf, 4, stems[i].blen, out);
    fclose(out);
    fprintf(stderr, "shard docs=%zu stems=%u postings=%llu\n",
            dl_len, n_stems, (unsigned long long)total);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "stemtest") == 0) {
        char line[600];
        while (fgets(line, sizeof line, stdin)) {
            int len = (int)strlen(line);
            while (len && (line[len-1]=='\n' || line[len-1]=='\r')) len--;
            if (len < 250) { len = porter(line, len); }
            fwrite(line, 1, len, stdout);
            fputc('\n', stdout);
        }
        return 0;
    }
    if (argc != 7 || strcmp(argv[1], "scan") != 0) {
        fprintf(stderr, "usage: scanner scan <tsv> <start> <end> <out> <stem>\n");
        return 2;
    }
    return do_scan(argv[2], strtoull(argv[3],0,10), strtoull(argv[4],0,10),
                   argv[5], atoi(argv[6]));
}
