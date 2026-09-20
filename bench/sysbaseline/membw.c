// Ground-truth memory benchmarks for this machine.
// Measures: memcpy bandwidth, sequential read bandwidth (sum), random-access
// latency (dependent pointer chase), and integer hash throughput (FNV-1a).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const size_t N = 1UL << 30; // 1 GiB
    char *a = malloc(N), *b = malloc(N);
    memset(a, 1, N); memset(b, 2, N); // fault pages in

    // memcpy bandwidth (counts read+write traffic once each)
    double t0 = now_s();
    for (int i = 0; i < 4; i++) memcpy(b, a, N);
    double t1 = now_s();
    printf("memcpy_GBps %.2f\n", 4.0 * N / (t1 - t0) / 1e9);

    // sequential read bandwidth via 64-bit sum
    volatile uint64_t sink = 0;
    uint64_t *w = (uint64_t *)a;
    size_t nw = N / 8;
    t0 = now_s();
    for (int r = 0; r < 4; r++) {
        uint64_t s = 0;
        for (size_t i = 0; i < nw; i++) s += w[i];
        sink += s;
    }
    t1 = now_s();
    printf("seqread_GBps %.2f\n", 4.0 * N / (t1 - t0) / 1e9);

    // random access latency: dependent pointer chase over 512 MiB
    size_t M = (1UL << 29) / 8;
    uint64_t *chase = (uint64_t *)b;
    // build a random cycle (Sattolo)
    for (size_t i = 0; i < M; i++) chase[i] = i;
    srandom(42);
    for (size_t i = M - 1; i > 0; i--) {
        size_t j = (size_t)(random() % i);
        uint64_t tmp = chase[i]; chase[i] = chase[j]; chase[j] = tmp;
    }
    uint64_t idx = 0;
    const size_t hops = 1UL << 25; // 33.5M dependent loads
    t0 = now_s();
    for (size_t i = 0; i < hops; i++) idx = chase[idx];
    t1 = now_s();
    sink += idx;
    printf("random_load_ns %.1f\n", (t1 - t0) / hops * 1e9);

    // FNV-1a hash throughput on 16-byte keys (term-hashing proxy)
    const size_t H = 1UL << 26; // 67M hashes
    uint64_t h = 1469598103934665603ULL;
    t0 = now_s();
    for (size_t i = 0; i < H; i++) {
        uint64_t x = i;
        for (int k = 0; k < 16; k++) { h ^= (x >> (k % 8)) & 0xff; h *= 1099511628211ULL; }
    }
    t1 = now_s();
    sink += h;
    printf("fnv16B_Mops %.1f\n", H / (t1 - t0) / 1e6);

    fprintf(stderr, "sink %llu\n", (unsigned long long)sink);
    free(a); free(b);
    return 0;
}
