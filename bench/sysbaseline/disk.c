// Ground-truth SSD benchmarks for this machine.
// Uses fcntl(F_NOCACHE) so macOS page cache doesn't fake the numbers.
// Measures: sequential write, sequential read (uncached), 4K random read
// (uncached), all on a temp file in the current directory.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const size_t FILE_SZ = 4UL << 30;   // 4 GiB
    const size_t BUF_SZ  = 4UL << 20;   // 4 MiB chunks
    const char *path = "diskbench.tmp";

    char *buf;
    posix_memalign((void **)&buf, 16384, BUF_SZ);
    memset(buf, 7, BUF_SZ);

    // sequential write (uncached)
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    fcntl(fd, F_NOCACHE, 1);
    double t0 = now_s();
    for (size_t off = 0; off < FILE_SZ; off += BUF_SZ)
        if (write(fd, buf, BUF_SZ) != (ssize_t)BUF_SZ) { perror("write"); return 1; }
    fsync(fd);
    double t1 = now_s();
    close(fd);
    printf("seqwrite_GBps %.2f\n", FILE_SZ / (t1 - t0) / 1e9);

    // sequential read (uncached)
    fd = open(path, O_RDONLY);
    fcntl(fd, F_NOCACHE, 1);
    t0 = now_s();
    for (size_t off = 0; off < FILE_SZ; off += BUF_SZ)
        if (read(fd, buf, BUF_SZ) != (ssize_t)BUF_SZ) { perror("read"); return 1; }
    t1 = now_s();
    printf("seqread_GBps %.2f\n", FILE_SZ / (t1 - t0) / 1e9);

    // 4K random reads (uncached) — the "posting list seek" primitive
    const int NREADS = 20000;
    srandom(1234);
    t0 = now_s();
    for (int i = 0; i < NREADS; i++) {
        off_t off = ((off_t)(random() % (FILE_SZ / 4096))) * 4096;
        if (pread(fd, buf, 4096, off) != 4096) { perror("pread"); return 1; }
    }
    t1 = now_s();
    close(fd);
    printf("rand4k_us %.1f\n", (t1 - t0) / NREADS * 1e6);
    printf("rand4k_iops %.0f\n", NREADS / (t1 - t0));

    unlink(path);
    free(buf);
    return 0;
}
