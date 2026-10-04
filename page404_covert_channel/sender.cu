// =============================================================================
// sender.cu — PAGE404 eviction-based covert channel (SENDER side)
//
// Research artifact for the PAGE404 study of NVIDIA GH200 (Hopper, sm_90).
// This is reproduction code for a published, responsibly-disclosed academic
// result; it is not a deployable attack. The sender and receiver run on two
// different MIG slices of the same physical GPU and communicate by modulating
// contention on shared L2 cache sets.
//
// Protocol (per bit slot, duration ~IDLE_TIME cycles):
//   bit '1' : the sender hammers a rotating series of eviction sets, evicting
//             the receiver's probe lines so the receiver observes high latency.
//   bit '0' : the sender stays idle for the slot, leaving the receiver's lines
//             resident so the receiver observes low latency.
// A fixed run of synchronization '1' bits at the start lets the receiver lock
// onto the slot boundaries before the payload begins.
//
// Physical-address / cache-set mapping:
//   The GH200 L2 set index is a set of parity functions (MASK0..2) over the
//   physical address. va_to_pa() reads /proc/self/pagemap to recover the PA of
//   each 2 MB region, and collect_sender_counters_multi() buckets regions by
//   their 3-bit set pattern so each channel ("set") drives one target bucket.
//
// Build:   see build.sh   (nvcc -O3 -std=c++17 -arch=sm_90)
// Launch:  see launch_overt_channel.sh
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <time.h>    // nanosleep / time
#include <unistd.h>  // sleep / usleep
#include <cooperative_groups.h>
#include <vector>
#include <sys/mman.h>

// namespace cg = cooperative_groups;  // UNUSED: kept for the experimental
//                                     // grid-wide sync path (see commented
//                                     // grid.sync() calls in the kernel).

// ---------------------------------------------------------------------------
// Tunable defaults (overridable on the command line; see main()).
// ---------------------------------------------------------------------------
#define DEFAULT_N                128   // per-access "threshold" loop count
#define STRIDE_BYTES              32   // one cache line
#define NUM_BITS                1000   // default payload length (bits)
#define FINISHING_DELAY   7000000000   // post-payload hold (cycles)
#define NUM_OF_COUNTERS_HP        32   // probe pointers per huge page / set
#define STARTING_WARMUP_ACCESS  3000   // warm-up slots before transmission
#define NUM_SYNCHRONIZATION_BITS  10   // leading/trailing sync '1' bits

// --- UNUSED defaults, retained from the development version -----------------
#define DEFAULT_M                400   // UNUSED
#define DEFAULT_PAGE_CHANGE        4   // UNUSED
#define DEFAULT_BLOCK_KB          64   // UNUSED

// ---------------------------------------------------------------------------
// Per-configuration channel timing (cycles).
//
// Select the number of parallel channels at compile time with NUM_CHANNELS.
// These constants set the slot length (IDLE_TIME), the inter-eviction spacing
// (BIT1_DELAY), and the one-shot alignment delay before transmission starts
// (STARTING_DELAY). They must stay paired with the receiver's IDLE_TIME /
// DELAY_IN_BETWEEN / MIGRATION_DETECTION_WAIT for the same NUM_CHANNELS.
// ---------------------------------------------------------------------------
#ifndef NUM_CHANNELS
#define NUM_CHANNELS 1
#endif

#if   NUM_CHANNELS == 1
  #define STARTING_DELAY    50888800
  #define BIT1_DELAY           10600
  #define IDLE_TIME          (230000)
#elif NUM_CHANNELS == 2
  #define STARTING_DELAY   800800000
  #define BIT1_DELAY           10500
  #define IDLE_TIME          (250000)
#elif NUM_CHANNELS == 3
  #define STARTING_DELAY   800800000
  #define BIT1_DELAY           10500
  #define IDLE_TIME          (300000)
#elif NUM_CHANNELS == 4
  #define STARTING_DELAY  1600800000
  #define BIT1_DELAY           13600
  #define IDLE_TIME          (380000)
#else
  #error "Unsupported NUM_CHANNELS (expected 1..4)"
#endif

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

uintptr_t va_to_pa(void* vaddr);

static inline int parity64(uint64_t x) {
    return __builtin_parityll(x);
}

// True if `pa` maps to the target L2 set (pattern 000).
// UNUSED: kept as a readable reference for the parity-mask set test; the hot
// path uses the 3-bit pattern computed inline in collect_sender_counters_multi.
bool maps_to_set(uint64_t pa) {
    return parity64(pa & MASK0) == SET_BIT_0 &&
           parity64(pa & MASK1) == SET_BIT_1 &&
           parity64(pa & MASK2) == SET_BIT_2;
}

// UNUSED: precomputed 2 MB-page groupings per set pattern from an earlier
// experiment. Retained for reference; the live path discovers groups at run
// time via collect_sender_counters_multi().
static uint8_t page_groups[8][32] = {
    {  0,  11,  22,  29,  39,  44,  49,  58,  69,  78,  83,  88,  98, 105, 116, 127, 134, 141, 144, 155, 161, 170, 183, 188, 195, 200, 213, 222, 228, 239, 242, 249 }, // group 0
    {  7,  12,  17,  26,  32,  43,  54,  61,  66,  73,  84,  95, 101, 110, 115, 120, 129, 138, 151, 156, 166, 173, 176, 187, 196, 207, 210, 217, 227, 232, 245, 254 }, // group 1
    {  5,  14,  19,  24,  34,  41,  52,  63,  64,  75,  86,  93, 103, 108, 113, 122, 131, 136, 149, 158, 164, 175, 178, 185, 198, 205, 208, 219, 225, 234, 247, 252 }, // group 2
    {  2,   9,  20,  31,  37,  46,  51,  56,  71,  76,  81,  90,  96, 107, 118, 125, 132, 143, 146, 153, 163, 168, 181, 190, 193, 202, 215, 220, 230, 237, 240, 251 }, // group 3
    {  6,  13,  16,  27,  33,  42,  55,  60,  67,  72,  85,  94, 100, 111, 114, 121, 128, 139, 150, 157, 167, 172, 177, 186, 197, 206, 211, 216, 226, 233, 244, 255 }, // group 4
    {  1,  10,  23,  28,  38,  45,  48,  59,  68,  79,  82,  89,  99, 104, 117, 126, 135, 140, 145, 154, 160, 171, 182, 189, 194, 201, 212, 223, 229, 238, 243, 248 }, // group 5
    {  3,   8,  21,  30,  36,  47,  50,  57,  70,  77,  80,  91,  97, 106, 119, 124, 133, 142, 147, 152, 162, 169, 180, 191, 192, 203, 214, 221, 231, 236, 241, 250 }, // group 6
    {  4,  15,  18,  25,  35,  40,  53,  62,  65,  74,  87,  92, 102, 109, 112, 123, 130, 137, 148, 159, 165, 174, 179, 184, 199, 204, 209, 218, 224, 235, 246, 253 }, // group 7
};

// ---------------------------------------------------------------------------
// Scan the huge-page region and bucket 2 MB pages by their 3-bit L2 set
// pattern. Returns `num_groups` groups (one per requested pattern, 0..7), each
// holding up to `counters_per_group` probe pointers.
// ---------------------------------------------------------------------------
std::vector<std::vector<uint32_t*>> collect_sender_counters_multi(
        uint32_t* arr,
        int       counters_per_group,
        size_t    num_of_2MB_pages,
        int       num_groups)           // 1..8
{
    if (num_groups < 1) num_groups = 1;
    if (num_groups > 8) num_groups = 8;

    std::vector<std::vector<uint32_t*>> result(num_groups);

    auto is_full = [&](int g) {
        return static_cast<int>(result[g].size()) >= counters_per_group;
    };
    auto all_full = [&]() {
        for (int g = 0; g < num_groups; g++)
            if (!is_full(g)) return false;
        return true;
    };

    for (size_t page = 0; page < num_of_2MB_pages && !all_full(); page++) {
        size_t   offset = page * (2 * 1024 * 1024) / sizeof(uint32_t);
        uint64_t pa     = va_to_pa((void *)&arr[offset]);

        int bit0    = parity64(pa & MASK0);
        int bit1    = parity64(pa & MASK1);
        int bit2    = parity64(pa & MASK2);
        int pattern = (bit2 << 2) | (bit1 << 1) | bit0;   // 0..7

        if (pattern < num_groups && !is_full(pattern))
            result[pattern].push_back(&arr[offset]);
    }

    for (int g = 0; g < num_groups; g++) {
        if ((int)result[g].size() < counters_per_group)
            printf("[WARN] Group %d (pattern %d): only %zu/%d addresses found\n",
                   g, g, result[g].size(), counters_per_group);
    }
    return result;
}

// Fill `arr` (length `size`) with the bit pattern to transmit: a leading and
// trailing block of '1' synchronization bits, random payload in between.
void fill_random_bits(char *arr, int size) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    int i = 0;
    for (; i < NUM_SYNCHRONIZATION_BITS / 2; i++)
        arr[i] = '1';                               // leading sync bits
    for (; i < size - NUM_SYNCHRONIZATION_BITS / 2; i++)
        arr[i] = (rand() % 2) ? '1' : '0';          // random payload
    for (; i < size; i++)
        arr[i] = '1';                               // trailing sync bits
}

// ---------------------------------------------------------------------------
// UNUSED kernel: standalone warm-up pass over all probe pages. It is not
// launched by main() (the covert-channel kernel does its own warm-up loop);
// retained for reference / manual experiments.
// ---------------------------------------------------------------------------
__global__ void warmup(uint32_t** pointer,
                       uint32_t*  dummy,
                       uint64_t   total_num_pages,
                       int        N)
{
    printf("Sender Warm-up started\n");
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t temp, sum = 0;
    temp = (uint32_t)(tid * sizeof(uint32_t));

    uint64_t number_elems_in_block = (64 * 1024) / sizeof(uint32_t);

    uint64_t kk = 0;
    uint32_t* p;
    while (kk < total_num_pages) {
        int kk_counter = 0;
        p = pointer[kk];
        while (kk_counter < N) {
            asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
            sum += temp;
            asm volatile ("membar.gl;");
            p = p + number_elems_in_block;
            kk_counter++;
        }
        kk++;
    }
    printf("Sender Warm-up finished\n");
    *dummy = sum;
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

// Issue volatile global loads from `p` until `kk` reaches `max_steps`.
// `kk` is passed by reference because some call sites intentionally continue a
// previous (unreset) count — see the commented phases in saturate_pool().
__device__ __forceinline__ void hammer(const uint32_t* p, uint64_t& kk,
                                        uint64_t max_steps) {
    uint32_t temp;
    while (kk < max_steps) {
        asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
        kk++;
    }
    (void)temp;   // written by the volatile load; value intentionally discarded
}

// Rotate to the next eviction set for this thread and return its probe pointer.
__device__ __forceinline__ uint32_t* next_eviction_set(uint32_t** d_pages, uint64_t tid,
                                                        int& which, int num_eviction_sets) {
    which = (which + 1) % num_eviction_sets;
    return d_pages[tid + NUM_OF_COUNTERS_HP * which];
}

// ---------------------------------------------------------------------------
// Covert-channel kernel: one block per channel ("set"), one thread per probe
// pointer. Each block transmits its own num_bits-long message by hammering (for
// a '1') or idling (for a '0') during each fixed-length slot.
// ---------------------------------------------------------------------------
__global__ void saturate_pool(uint32_t** d_pages_sender,
                              uint32_t*  dummy,
                              uint64_t   max_steps,
                              char*      messages,
                              uint64_t   num_bits,
                              int        page_change,      // UNUSED (kept for ABI)
                              uint64_t*  bit_measurement,
                              int        num_eviction_sets,
                              int        offset)
{
    int      set = blockIdx.x;
    uint64_t tid = (uint64_t)threadIdx.x;

    uint32_t temp = 0;
    uint32_t sum  = 0;
    uint64_t kk   = 0;
    uint64_t bit  = 0;
    uint64_t bit1_measurement_start, bit1_measurement_end;
    uint64_t measurement_index = 0;
    bool     timed_out = false;
    int      which_eviciton_set = 0;
    uint32_t smid = get_smid();

    uint32_t** d_pages = d_pages_sender + set * offset;
    uint32_t*  p       = d_pages[tid + NUM_OF_COUNTERS_HP * which_eviciton_set];

    __syncthreads();

    // ---- Warm-up: drive the eviction sets into steady state --------------
    // Each warm-up slot runs 7 hammer phases. Phase 1 carries an idle-time
    // guard; phase 2 intentionally continues phase 1's `kk` count (no reset),
    // matching the original implementation.
    while (bit < STARTING_WARMUP_ACCESS) {
        timed_out = false;
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(bit1_measurement_start));

        // phase 1 (guarded)
        while (kk < max_steps) {
            if (clock64() - bit1_measurement_start >= IDLE_TIME) { timed_out = true; break; }
            asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
            kk++;
        }
        __syncthreads();
        busy_wait(BIT1_DELAY);

        // phase 2 (kk NOT reset, by design)
        p = next_eviction_set(d_pages, tid, which_eviciton_set, num_eviction_sets);
        hammer(p, kk, max_steps);
        __syncthreads();
        busy_wait(BIT1_DELAY);

        // phases 3..7 (kk reset each time; no spacing after the last)
        for (int phase = 0; phase < 5; phase++) {
            kk = 0;
            p = next_eviction_set(d_pages, tid, which_eviciton_set, num_eviction_sets);
            hammer(p, kk, max_steps);
            __syncthreads();
            if (phase < 4) busy_wait(BIT1_DELAY);
        }
        bit++;
    }

    // ---- One-shot alignment delay, then begin transmission ---------------
    if (tid == 0 && set == 0)
        printf("set %d: Sender Kernel starts on SM %d\n", set, smid);
    busy_wait(STARTING_DELAY);
    // grid.sync();   // UNUSED: experimental grid-wide barrier
    __syncthreads();
    if (tid == 0 && set == 0)
        printf("set %d: Sender started\n", set);

    bit = 0;   // NOTE: kk is deliberately left as-is from warm-up.
    while (bit < num_bits) {
        timed_out = false;
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(bit1_measurement_start));

        if (messages[bit + set * num_bits] == '1') {
            // phase 1 (guarded)
            while (kk < max_steps) {
                if (clock64() - bit1_measurement_start >= IDLE_TIME) { timed_out = true; break; }
                asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
                kk++;
            }
            __syncthreads();
            if (timed_out) goto end_of_bit;
            busy_wait(BIT1_DELAY);

            // phase 2 (kk NOT reset, by design)
            p = next_eviction_set(d_pages, tid, which_eviciton_set, num_eviction_sets);
            hammer(p, kk, max_steps);
            __syncthreads();
            if (timed_out) goto end_of_bit;
            busy_wait(BIT1_DELAY);

            // phases 3..8 (kk reset each time)
            for (int phase = 0; phase < 6; phase++) {
                kk = 0;
                p = next_eviction_set(d_pages, tid, which_eviciton_set, num_eviction_sets);
                hammer(p, kk, max_steps);
                __syncthreads();
                if (timed_out) goto end_of_bit;
                busy_wait(BIT1_DELAY);
            }

            // phase 9 (final hammer: no barrier, fence instead of spacing)
            kk = 0;
            p = next_eviction_set(d_pages, tid, which_eviciton_set, num_eviction_sets);
            hammer(p, kk, max_steps);
            if (timed_out) goto end_of_bit;
            asm volatile ("membar.gl;");

            // hold the slot to its full length, then pre-rotate for next bit
            while (clock64() - bit1_measurement_start < IDLE_TIME) {}
            p = next_eviction_set(d_pages, tid, which_eviciton_set, num_eviction_sets);
        }
        else {
            // transmit '0': idle for the whole slot
            while (clock64() - bit1_measurement_start < IDLE_TIME) {}
            __syncthreads();
        }

    end_of_bit:
        kk = 0;
        bit++;
        __syncthreads();
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(bit1_measurement_end));
        if (tid == 0) {
            bit_measurement[measurement_index + set * num_bits] =
                bit1_measurement_end - bit1_measurement_start;
            measurement_index++;
        }
    }

    if (tid == 0 && set == 0)
        printf("set %d: Sender finished\n", set);
    busy_wait(FINISHING_DELAY);
    __syncthreads();
    if (tid == 0 && set == 0)
        printf("set %d: Sender kernel finished\n", set);

    (void)temp;   // written by the guarded loads; value intentionally discarded
    atomicAdd(&dummy[set], sum);
}

// Print the VA->PA mapping for each huge page (diagnostic for set targeting).
static void dump_hugepage_pas(uint32_t* arr, int num_hps) {
    for (int i = 0; i < num_hps; i++) {
        uint64_t pa = va_to_pa((void*)(arr + i * HP / sizeof(uint32_t)));
        printf("Sen: Huge page %d: VA = 0x%016lx, PA = 0x%016lx\n",
               i, (unsigned long)(arr + i * HP), (unsigned long)pa);
    }
}

int main(int argc, char **argv)
{
    int num_eviction_sets = 2;
    int N                 = DEFAULT_N;
    int gpu_id            = 0;            // UNUSED (device is pinned via MIG)
    int delta_n           = 10;
    int num_bits          = NUM_BITS;
    int num_of_used_sets  = 1;

    static struct option opts[] = {
        {"threshold",         required_argument, 0, 'N'},
        {"num-eviction-sets", required_argument, 0, 'p'},
        {"gpu",               required_argument, 0, 'g'},
        {"delta-n",           required_argument, 0, 'd'},
        {"num-bits",          required_argument, 0, 'b'},
        {"num-sets",          required_argument, 0, 's'},
        {0, 0, 0, 0}
    };
    int c, idx;
    while ((c = getopt_long(argc, argv, "N:p:g:d:b:s", opts, &idx)) != -1) {
        switch (c) {
        case 'N': N                 = atoi(optarg); break;
        case 'p': num_eviction_sets = atoi(optarg); break;
        case 'g': gpu_id            = atoi(optarg); break;
        case 'd': delta_n           = atoi(optarg); break;
        case 'b': num_bits          = atoi(optarg); break;
        case 's': num_of_used_sets  = atoi(optarg); break;
        }
    }
    (void)gpu_id;
    (void)N;   // accepted as --threshold for CLI parity; sender uses delta_n

    // ---- Allocate and huge-page-align the probe buffer -------------------
    int    num_hps = num_eviction_sets;
    size_t len     = num_hps * HP;
    num_bits       = num_bits + NUM_SYNCHRONIZATION_BITS;

    void *raw = mmap(NULL, len + HP, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { perror("mmap"); return 1; }

    uintptr_t raw_addr = (uintptr_t)raw;
    size_t    raw_len  = len + HP;
    uintptr_t buf      = ((uintptr_t)raw + HP - 1) & ~(HP - 1);   // align up to HP

    madvise((void*)buf, len, MADV_HUGEPAGE);

    uint32_t* arr = (uint32_t*) buf;
    memset(arr, 0, len);
    printf("Sender initialized\n");

    size_t total_2MB_pages = len / (2 * 1024 * 1024);

    dump_hugepage_pas(arr, num_hps);

    // ---- Discover probe pointers for each channel's target set -----------
    std::vector<std::vector<uint32_t*>> sender_counters =
        collect_sender_counters_multi(arr, num_eviction_sets * NUM_OF_COUNTERS_HP,
                                       total_2MB_pages, num_of_used_sets);

    uint32_t** d_pages;
    CUDA_CHECK(cudaMalloc(&d_pages,
        num_eviction_sets * NUM_OF_COUNTERS_HP * num_of_used_sets * sizeof(uint32_t*)));
    for (int i = 0; i < num_of_used_sets; i++) {
        if (sender_counters[i].size() != (size_t)(num_eviction_sets * NUM_OF_COUNTERS_HP)) {
            printf("Not correctly collected\n");
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(d_pages + i * num_eviction_sets * NUM_OF_COUNTERS_HP,
                              sender_counters[i].data(),
                              num_eviction_sets * NUM_OF_COUNTERS_HP * sizeof(uint32_t*),
                              cudaMemcpyHostToDevice));
    }

    // ---- Device / host allocations ---------------------------------------
    size_t dummy_size        = sizeof(uint32_t) * num_of_used_sets;
    size_t message_size      = num_of_used_sets * num_bits * sizeof(char);
    size_t measurements_size = num_of_used_sets * num_bits * sizeof(uint64_t);

    uint32_t *d_dummy;
    CUDA_CHECK(cudaMalloc(&d_dummy, dummy_size));
    uint32_t *h_dummy = (uint32_t *) malloc(dummy_size);

    char *d_message;
    CUDA_CHECK(cudaMalloc(&d_message, message_size));
    char *h_message = (char *) malloc(message_size);

    uint64_t *d_bit_measurement;
    CUDA_CHECK(cudaMalloc(&d_bit_measurement, measurements_size));
    uint64_t *h_bit_measurement = (uint64_t *) malloc(measurements_size);

    for (int message_set = 0; message_set < num_of_used_sets; message_set++)
        fill_random_bits(h_message + message_set * num_bits, num_bits);

    CUDA_CHECK(cudaMemcpy(d_message, h_message, message_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_threads = NUM_OF_COUNTERS_HP;
    int num_blocks  = num_of_used_sets;
    printf("number of blocks: %d\n", num_blocks);
    printf("number of threads per block: %d\n", num_threads);

    // Coarse alignment with the receiver's launch (see launch_overt_channel.sh).
    usleep(1136000);
    saturate_pool<<<num_blocks, num_threads>>>(
        d_pages, d_dummy, delta_n, d_message, num_bits, num_eviction_sets,
        d_bit_measurement, num_eviction_sets, num_eviction_sets * NUM_OF_COUNTERS_HP);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, dummy_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bit_measurement, d_bit_measurement, measurements_size,
                          cudaMemcpyDeviceToHost));
    printf("Dummy: %d\n", *h_dummy);

    // ---- Dump per-channel timing and the transmitted payload -------------
    FILE* fptr;
    char  filename[256];
    for (int i = 0; i < num_of_used_sets; i++) {
        snprintf(filename, sizeof(filename), "./texts/parallel_channel/sender_measurement_%d", i);
        fptr = fopen(filename, "w");
        for (uint64_t k = 0; k < (uint64_t)num_bits; k++)
            fprintf(fptr, "%lu \n", *(h_bit_measurement + i * num_bits + k));
        fclose(fptr);

        snprintf(filename, sizeof(filename), "./texts/parallel_channel/sender_message_%d", i);
        fptr = fopen(filename, "w");
        for (uint64_t k = NUM_SYNCHRONIZATION_BITS / 2; k < (uint64_t)(num_bits - NUM_SYNCHRONIZATION_BITS / 2); k++)
            fprintf(fptr, "%c \n", *(h_message + i * num_bits + k));
        fclose(fptr);
    }

    sleep(1);
    dump_hugepage_pas(arr, num_hps);

    // ---- Cleanup ---------------------------------------------------------
    free(h_message);
    free(h_dummy);
    CUDA_CHECK(cudaFree(d_dummy));
    CUDA_CHECK(cudaFree(d_message));

    dump_hugepage_pas(arr, num_hps);

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
