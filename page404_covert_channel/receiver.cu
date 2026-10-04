// =============================================================================
// receiver.cu — PAGE404 eviction-based covert channel (RECEIVER side)
//
// Research artifact for the PAGE404 study of NVIDIA GH200 (Hopper, sm_90).
// This is reproduction code for a published, responsibly-disclosed academic
// result; it is not a deployable attack. The receiver runs on a different MIG
// slice from the sender (see launch_overt_channel.sh) and recovers the bit
// stream by probing shared L2 cache sets.
//
// Decoding (per bit slot):
//   The receiver primes its probe lines, waits, then times one more access to
//   the same line. If the line is still resident (fast), the sender was idle
//   -> bit '0'. If it was evicted (slow), the sender was hammering -> bit '1'.
//   A run of NUM_SYNCHRONIZATION_BITS consecutive '1's marks the start of the
//   payload and aligns the receiver's slot clock with the sender's.
//
// Physical-address / cache-set mapping:
//   As in the sender, va_to_pa() + MASK0..2 parity bucket pages by their 3-bit
//   L2 set pattern; collect_receiver_counters_multi() builds the probe sets.
//
// Build:   see build.sh   (nvcc -O3 -std=c++17 -arch=sm_90)
// Launch:  see launch_overt_channel.sh
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <time.h>    // nanosleep / time
#include <unistd.h>  // sleep / usleep
#include <cooperative_groups.h>
#include <sys/mman.h>

// namespace cg = cooperative_groups;  // UNUSED: kept for the experimental
//                                     // grid-wide sync path (see grid_sync).

// ---------------------------------------------------------------------------
// Tunable defaults (overridable on the command line; see main()).
// ---------------------------------------------------------------------------
#define DEFAULT_N                 256   // "threshold": accesses per prime loop
#define DEFAULT_BLOCK_KB          128   // probe page granularity (KB)
#define MIGRATION_DETECTION      1000   // latency threshold (cycles): < => resident ('0')
#define NUMBER_OF_BITS          10000   // default number of bit slots to sample
#define NUM_OF_COUNTERS_HP        512   // probe pages per huge page / channel
#define NUM_SYNCHRONIZATION_BITS    5   // leading sync '1's that mark payload start
#define STARTING_WARMUP_ACCESS   3000   // warm-up slots before decoding
#define NUM_EXPECTED_BITS       10000   // payload length used for bandwidth calc

// --- UNUSED defaults, retained from the development version -----------------
#define STRIDE_BYTES              32    // UNUSED (one cache line)
#define MIGRATION_THRESHOLD   120000    // UNUSED

// ---------------------------------------------------------------------------
// Per-configuration channel timing (cycles). Select with NUM_CHANNELS; must
// stay paired with the sender's STARTING_DELAY / BIT1_DELAY / IDLE_TIME for the
// same NUM_CHANNELS.
//   IDLE_TIME                : slot length
//   DELAY_IN_BETWEEN         : spacing between prime and probe
//   MIGRATION_DETECTION_WAIT : dwell before the timed probe access
// ---------------------------------------------------------------------------
#ifndef NUM_CHANNELS
#define NUM_CHANNELS 1
#endif

#if   NUM_CHANNELS == 1
  #define IDLE_TIME                 (230000)
  #define DELAY_IN_BETWEEN            37000
  #define MIGRATION_DETECTION_WAIT    60000
#elif NUM_CHANNELS == 2
  #define IDLE_TIME                 (250000)
  #define DELAY_IN_BETWEEN            37000
  #define MIGRATION_DETECTION_WAIT   100000
#elif NUM_CHANNELS == 3
  #define IDLE_TIME                 (300000)
  #define DELAY_IN_BETWEEN            57000
  #define MIGRATION_DETECTION_WAIT   100000
#elif NUM_CHANNELS == 4
  #define IDLE_TIME                 (380000)
  #define DELAY_IN_BETWEEN            57000
  #define MIGRATION_DETECTION_WAIT   180000
#else
  #error "Unsupported NUM_CHANNELS (expected 1..4)"
#endif

// --- UNUSED latency-spike thresholds from an earlier detector --------------
#define SPIKE_LOW       20000ULL   // UNUSED
#define SPIKE_HIGH    1000000ULL   // UNUSED
#define PREV_LAT_LOW      850ULL   // UNUSED
#define PREV_LAT_HIGH   10000ULL   // UNUSED

// ---------------------------------------------------------------------------
// L2 cache-set parity masks for the target set (pattern 000).
// ---------------------------------------------------------------------------
#define MASK0  0x1ba4e00000ULL
#define MASK1  0x24e9c00000ULL
#define MASK2  0x49d3a00000ULL

#define SET_BIT_0  0
#define SET_BIT_1  0
#define SET_BIT_2  0

#define HP      (512UL << 20)   // 512 MB huge page (arm64 / 64 KB base pages)
#define NUM_HP  2               // UNUSED: static huge-page count (now derived)

uintptr_t va_to_pa(void* vaddr);

static inline int parity64(uint64_t x) {
    return __builtin_parityll(x);
}

// True if `pa` maps to the target L2 set (pattern 000).
// UNUSED: readable reference for the parity-mask set test; the hot path uses
// the 3-bit pattern computed inline in collect_receiver_counters_multi().
bool maps_to_set(uint64_t pa) {
    return parity64(pa & MASK0) == SET_BIT_0 &&
           parity64(pa & MASK1) == SET_BIT_1 &&
           parity64(pa & MASK2) == SET_BIT_2;
}

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Returns `num_groups` groups of probe pointers, one per 3-bit set pattern
// (group 0 -> pattern 000, group 1 -> 001, ...). Each group holds up to
// `counters_per_group` matching addresses. Scans the 16 sub-pages of every
// 2 MB page so probe lines spread evenly across the set.
// ---------------------------------------------------------------------------
std::vector<std::vector<uint32_t*>> collect_receiver_counters_multi(
        uint32_t*  arr,
        int        counters_per_group,
        size_t     num_of_128KB_pages,
        int        num_groups)          // 1..8
{
    if (num_groups < 1) num_groups = 1;
    if (num_groups > 8) num_groups = 8;

    std::vector<std::vector<uint32_t*>> result(num_groups);

    auto is_full = [&](int g) {
        return static_cast<int>(result[g].size()) >= counters_per_group;
    };

    size_t total_2MB_pages = num_of_128KB_pages / 16;

    for (size_t page_128kb = 0; page_128kb < 16; page_128kb++) {
        size_t page_offset_128KB = page_128kb * (128 * 1024 / sizeof(uint32_t));

        for (size_t page_2mb = 0; page_2mb < total_2MB_pages; page_2mb++) {
            size_t   offset = page_2mb * (2 * 1024 * 1024 / sizeof(uint32_t))
                            + page_offset_128KB;
            uint64_t pa     = va_to_pa((void *)&arr[offset]);

            int bit0    = parity64(pa & MASK0);          // LSB of pattern
            int bit1    = parity64(pa & MASK1);
            int bit2    = parity64(pa & MASK2);          // MSB of pattern
            int pattern = (bit2 << 2) | (bit1 << 1) | bit0;  // 0..7

            if (pattern < num_groups && !is_full(pattern))
                result[pattern].push_back(&arr[offset]);
        }

        // Stop scanning once every requested group is full.
        bool all_full = true;
        for (int g = 0; g < num_groups; g++)
            if (!is_full(g)) { all_full = false; break; }
        if (all_full) break;
    }

    return result;
}

// UNUSED: experimental grid-wide barrier (never called; all call sites below
// are commented out). Retained for reference.
__device__ void grid_sync(volatile int* barrier_counter, int num_blocks)
{
    atomicAdd((int*)barrier_counter, 1);
    while (*barrier_counter < num_blocks) {}

    if (threadIdx.x == 0 && blockIdx.x == 0)
        atomicExch((int*)barrier_counter, 0);

    while (*barrier_counter != 0) {}
}

__device__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

// Busy-wait for `cycles` SM clock cycles (measured from "now").
__device__ __forceinline__ void busy_wait(uint64_t cycles) {
    uint64_t start;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start));
    while (clock64() - start < cycles) {}
}

// Issue `count` volatile global loads from `p` (priming / dwell accesses).
__device__ __forceinline__ void prime(const uint32_t* p, uint64_t count) {
    uint32_t temp;
    for (uint64_t i = 0; i < count; i++)
        asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
    (void)temp;   // written by the volatile load; value intentionally discarded
}

// One probe/decode step on pointer `p`: prime the line, dwell, then time a
// single access. Returns the measured access latency (cycles) and adds the
// loaded value into `sum` (keeps the loads from being optimized away).
__device__ __forceinline__ uint64_t probe_latency(const uint32_t* p, int N, int delta_n,
                                                   uint32_t& sum) {
    uint32_t temp;

    // Prime: fill the line, leaving `delta_n` accesses for the dwell phase.
    prime(p, (uint64_t)(N - delta_n));

    busy_wait(DELAY_IN_BETWEEN);
    prime(p, (uint64_t)delta_n);
    busy_wait(MIGRATION_DETECTION_WAIT);

    // Untimed touch to settle state.
    asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
    asm volatile ("membar.gl;");
    sum += temp;

    // Timed access: short latency => line still resident.
    uint64_t t0, t1;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(t0));
    asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
    asm volatile ("membar.gl;");
    sum += temp;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(t1));

    return t1 - t0;
}

// ---------------------------------------------------------------------------
// Covert-channel kernel: one block per channel ("set"). Each block walks its
// probe pages one per slot, decoding one bit per slot and writing the recovered
// stream (post-sync) into `message`, plus the raw stream into `all_messages`.
// ---------------------------------------------------------------------------
__global__ void measure_migration(uint32_t** all_sets_receiver_pages,
                                  uint32_t*  dummy,
                                  char*      message,
                                  uint64_t   num_bits,
                                  int        N,
                                  uint64_t*  measurement_time,
                                  uint64_t*  measurements,
                                  int        delta_n,
                                  int*       num_of_captured_bits,
                                  char*      all_messages,
                                  int        num_pages,
                                  volatile int* barrier_counter,   // UNUSED (kept for ABI)
                                  int        num_blocks)           // UNUSED (kept for ABI)
{
    int set = blockIdx.x;                 // one block per channel / set

    uint32_t sum = 0;
    uint64_t measurement_start = 0, measurement_end = 0;
    uint64_t measurements_start, measurements_end;

    uint64_t bit = 0;
    uint64_t global_meaasurement_index = 0;
    uint32_t smid = get_smid();

    uint32_t** receiver_pages = all_sets_receiver_pages + set * num_pages;

    int num_sync_bits   = 0;   // consecutive '1's seen while hunting for sync
    int num_total_bits  = 0;   // payload bits recorded after sync
    int num_times_sync  = 0;   // 0 = pre-sync, >=1 = recording payload
    int page_id         = 0;

    // ---- Warm-up: settle the probe pages into steady state ----------------
    // grid_sync(barrier_counter, num_blocks);   // UNUSED
    while (bit < STARTING_WARMUP_ACCESS) {
        probe_latency(receiver_pages[page_id], N, delta_n, sum);
        page_id++;
        bit++;
    }

    bit = 0;
    if (set == 0)
        printf("set %d: Receiver started on SM %d\n", set, smid);
    // grid_sync(barrier_counter, num_blocks);   // UNUSED
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(measurement_start));

    // ---- Decode loop ------------------------------------------------------
    while (bit < num_bits) {
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(measurements_start));

        uint64_t latency = probe_latency(receiver_pages[page_id], N, delta_n, sum);

        // Resident (fast) => sender idle => '0'; evicted (slow) => '1'.
        char detected_bit = (latency < MIGRATION_DETECTION) ? '0' : '1';

        // Once synchronized, record payload bits and timestamp the end once we
        // have the expected count.
        if (num_times_sync >= 1) {
            message[num_total_bits + set * num_bits] = detected_bit;
            num_total_bits++;
            if (num_total_bits == NUM_EXPECTED_BITS)
                asm volatile ("mov.u64 %0, %%clock64;" : "=l"(measurement_end));
        }

        // Pre-sync: look for NUM_SYNCHRONIZATION_BITS consecutive '1's.
        if (num_times_sync == 0 && detected_bit == '1') {
            num_sync_bits++;
            if (num_sync_bits == NUM_SYNCHRONIZATION_BITS) {
                num_sync_bits = 0;
                num_times_sync++;
                asm volatile ("mov.u64 %0, %%clock64;" : "=l"(measurement_start));
            }
        } else {
            num_sync_bits = 0;
        }

        all_messages[bit + set * num_bits] = detected_bit;

        bit++;
        page_id++;

        // Hold the slot to its full length and record its duration.
        while (clock64() - measurements_start < IDLE_TIME) {}
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(measurements_end));
        measurements[global_meaasurement_index + set * num_bits] =
            measurements_end - measurements_start;
        global_meaasurement_index++;
    }

    measurement_time[set]      = measurement_end - measurement_start;
    num_of_captured_bits[set]  = num_total_bits;
    dummy[set]                 = sum;
    // grid_sync(barrier_counter, num_blocks);   // UNUSED
    if (set == 0)
        printf("set %d: Receiver finished\n", set);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int N                = DEFAULT_N;
    int num_pages        = 1024;
    int num_message_bits = NUMBER_OF_BITS;
    int gpu_id           = 0;      // UNUSED (device is pinned via MIG)
    int delta_n          = 6;
    int num_of_used_sets = 2;

    static struct option opts[] = {
        {"threshold", required_argument, 0, 'N'},
        {"num-pages", required_argument, 0, 'M'},
        {"num-bits",  required_argument, 0, 'B'},
        {"gpu",       required_argument, 0, 'G'},
        {"delta-n",   required_argument, 0, 'D'},
        {"num-sets",  required_argument, 0, 'S'},
        {0, 0, 0, 0}
    };
    int c, idx;
    while ((c = getopt_long(argc, argv, "N:M:B:G:D:S:", opts, &idx)) != -1) {
        switch (c) {
        case 'N': N                = atoi(optarg); break;
        case 'M': num_pages        = atoi(optarg); break;
        case 'B': num_message_bits = atoi(optarg); break;
        case 'G': gpu_id           = atoi(optarg); break;
        case 'D': delta_n          = atoi(optarg); break;
        case 'S': num_of_used_sets = atoi(optarg); break;
        }
    }
    (void)gpu_id;

    // ---- Allocate and huge-page-align the probe buffer -------------------
    int    num_hps = num_pages / NUM_OF_COUNTERS_HP;
    size_t len     = num_hps * HP;

    void *raw = mmap(NULL, len + HP, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { perror("mmap"); return 1; }

    uintptr_t raw_addr = (uintptr_t)raw;
    size_t    raw_len  = len + HP;
    uintptr_t buf      = ((uintptr_t)raw + HP - 1) & ~(HP - 1);   // align up to HP

    madvise((void*)buf, len, MADV_HUGEPAGE);

    uint32_t* arr = (uint32_t*) buf;
    memset(arr, 0, len);
    printf("Receiver initialized\n");

    size_t page_size_128KB  = (DEFAULT_BLOCK_KB * 1024);
    size_t total_128KB_pages = len / page_size_128KB;

    for (int i = 0; i < num_hps; i++) {
        uint64_t pa = va_to_pa((void*)(arr + i * HP / sizeof(uint32_t)));
        printf("Rec: Huge page %d: VA = 0x%016lx, PA = 0x%016lx\n",
               i, (unsigned long)(arr + i * HP), (unsigned long)pa);
    }

    printf("num_pages: %d, num_hps: %d, total_128KB_pages: %lu\n",
           num_pages, num_hps, total_128KB_pages);

    std::vector<std::vector<uint32_t*>> receiver_pages =
        collect_receiver_counters_multi(arr, num_pages, total_128KB_pages, num_of_used_sets);

    // ---- Device allocations ----------------------------------------------
    uint32_t** d_pages;
    CUDA_CHECK(cudaMalloc(&d_pages, num_pages * num_of_used_sets * sizeof(uint32_t*)));
    for (int i = 0; i < num_of_used_sets; i++) {
        if (receiver_pages[i].size() != (size_t)num_pages) {
            printf("Not correctly collected\n");
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(d_pages + i * num_pages, receiver_pages[i].data(),
                              num_pages * sizeof(uint32_t*), cudaMemcpyHostToDevice));
    }

    size_t dummy_size              = sizeof(uint32_t) * num_of_used_sets;
    size_t message_size            = num_of_used_sets * num_message_bits * sizeof(char);
    size_t measurement_size        = num_of_used_sets * sizeof(uint64_t);
    size_t measurements_size       = num_of_used_sets * num_message_bits * sizeof(uint64_t);
    size_t num_of_captured_bits_sz = num_of_used_sets * sizeof(int);

    // UNUSED barrier plumbing (paired with grid_sync); kept for ABI parity.
    int *barrier_device;
    CUDA_CHECK(cudaMalloc(&barrier_device, sizeof(int)));
    int *barrier_counter = (int*) malloc(sizeof(int));
    *barrier_counter = 0;
    CUDA_CHECK(cudaMemcpy(barrier_device, barrier_counter, sizeof(int), cudaMemcpyHostToDevice));

    uint32_t *d_dummy;
    CUDA_CHECK(cudaMalloc(&d_dummy, dummy_size));
    uint32_t *h_dummy = (uint32_t*) malloc(dummy_size);
    CUDA_CHECK(cudaMemset(d_dummy, 0, dummy_size));

    char *d_messages;
    CUDA_CHECK(cudaMalloc(&d_messages, message_size));
    char *h_messages = (char *) malloc(message_size);

    char *d_all_messages;
    CUDA_CHECK(cudaMalloc(&d_all_messages, message_size));
    char *h_all_messages = (char *) malloc(message_size);

    uint64_t *d_measurement;
    CUDA_CHECK(cudaMalloc(&d_measurement, measurement_size));
    uint64_t *h_measurement = (uint64_t*) malloc(measurement_size);

    uint64_t *d_measurements_all;
    CUDA_CHECK(cudaMalloc(&d_measurements_all, measurements_size));
    uint64_t *h_measurements_all = (uint64_t*) malloc(measurements_size);

    int *d_num_of_captured_bits;
    CUDA_CHECK(cudaMalloc(&d_num_of_captured_bits, num_of_captured_bits_sz));
    int *h_num_of_captured_bits = (int*) malloc(num_of_captured_bits_sz);

    // ---- Run the channel -------------------------------------------------
    // Coarse alignment with the sender's launch (see launch_overt_channel.sh).
    usleep(1697800);
    measure_migration<<<num_of_used_sets, 1>>>(
        d_pages, d_dummy, d_messages, num_message_bits, N, d_measurement,
        d_measurements_all, delta_n, d_num_of_captured_bits, d_all_messages,
        num_pages, barrier_device, num_of_used_sets);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_messages, d_messages, message_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, dummy_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_measurement, d_measurement, measurement_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_measurements_all, d_measurements_all, measurements_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_num_of_captured_bits, d_num_of_captured_bits, num_of_captured_bits_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_all_messages, d_all_messages, message_size, cudaMemcpyDeviceToHost));

    // ---- Write per-channel output and report bandwidth -------------------
    FILE* fptr;
    char  filename[256];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int   total_num_captured_bits = 0;
    float bandwidth = 0.0;
    printf("GPU clock rate: %d kHz (%.2f GHz)\n", prop.clockRate, prop.clockRate / 1.0e6);

    for (int i = 0; i < num_of_used_sets; i++) {
        printf("Number of captured bits: %d\n", h_num_of_captured_bits[i]);
        total_num_captured_bits += h_num_of_captured_bits[i];

        snprintf(filename, sizeof(filename), "./texts/parallel_channel/message_bits_%d", i);
        fptr = fopen(filename, "w");
        int usable_bits = h_num_of_captured_bits[i] - NUM_SYNCHRONIZATION_BITS;
        if (usable_bits > 0)
            for (int k = 0; k < usable_bits; k++)
                fprintf(fptr, "%c \n", *(h_messages + i * num_message_bits + k));
        fclose(fptr);

        snprintf(filename, sizeof(filename), "./texts/parallel_channel/receiver_measurements_%d", i);
        fptr = fopen(filename, "w");
        for (uint64_t k = 0; k < (uint64_t)num_message_bits; k++)
            fprintf(fptr, "%lu \n", *(h_measurements_all + i * num_message_bits + k));
        fclose(fptr);

        snprintf(filename, sizeof(filename), "./texts/parallel_channel/receiver_all_messages_%d", i);
        fptr = fopen(filename, "w");
        for (uint64_t k = 0; k < (uint64_t)num_message_bits; k++)
            fprintf(fptr, "%c \n", *(h_all_messages + i * num_message_bits + k));
        fclose(fptr);

        printf("Dummy: %d\n", h_dummy[i]);
        printf("Measurement Time: %lu\n", h_measurement[i]);
        if (h_num_of_captured_bits[i] == 0)
            h_num_of_captured_bits[i] = num_message_bits;
        bandwidth += (NUM_EXPECTED_BITS * 1.0 * prop.clockRate * 1000.0) / h_measurement[i];
    }

    printf("Bandwidth: %.2f bits/s\n", bandwidth);

    // ---- Cleanup ---------------------------------------------------------
    printf("Unmapping memory.\n");
    fflush(stdout);
    CUDA_CHECK(cudaFree(d_pages));
    CUDA_CHECK(cudaFree(d_messages));
    CUDA_CHECK(cudaFree(d_all_messages));
    CUDA_CHECK(cudaFree(d_measurements_all));
    CUDA_CHECK(cudaFree(d_num_of_captured_bits));
    CUDA_CHECK(cudaFree(d_dummy));
    CUDA_CHECK(cudaFree(d_measurement));
    CUDA_CHECK(cudaFree(barrier_device));
    CUDA_CHECK(cudaDeviceReset());   // force full driver-side teardown now

    free(h_messages);
    free(h_all_messages);
    free(h_measurement);
    free(h_measurements_all);
    free(h_num_of_captured_bits);
    free(h_dummy);
    free(barrier_counter);

    sleep(1);

    for (int i = num_hps - 1; i >= 0; i--)
        munmap((void*)(buf + (size_t)i * HP), HP);

    size_t head = buf - raw_addr;
    if (head) munmap((void*)raw_addr, head);
    size_t tail = (raw_addr + raw_len) - (buf + len);
    if (tail) munmap((void*)(buf + len), tail);

    _exit(0);
}

// Translate a virtual address to its physical address via /proc/self/pagemap.
// Requires the pages to be present (we touch them with memset first).
uintptr_t va_to_pa(void* vaddr) {
    uintptr_t virt      = (uintptr_t)vaddr;
    long      page_size = sysconf(_SC_PAGESIZE);
    uint64_t  page_idx  = virt / page_size;
    uint64_t  offset    = virt % page_size;

    FILE* pagemap = fopen("/proc/self/pagemap", "rb");
    if (!pagemap) {
        perror("fopen /proc/self/pagemap failed");
        return 0;
    }

    fseeko(pagemap, page_idx * sizeof(uint64_t), SEEK_SET);

    uint64_t entry = 0;
    if (fread(&entry, sizeof(uint64_t), 1, pagemap) != 1) {
        perror("fread pagemap failed");
        fclose(pagemap);
        return 0;
    }
    fclose(pagemap);

    if (!(entry & (1ULL << 63))) {
        fprintf(stderr, "Page not present in RAM!\n");
        return 0;
    }

    uint64_t  pfn = entry & ((1ULL << 55) - 1);
    uintptr_t pa  = (pfn * page_size) + offset;
    return pa;
}
