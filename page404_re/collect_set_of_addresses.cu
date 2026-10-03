// --- C / POSIX ---
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <time.h>       // for nanosleep
#include <unistd.h>     // for sleep / usleep
#include <sys/mman.h>
#include <sys/stat.h>   // for mkdir

// --- C++ ---
#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>

// --- CUDA ---
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------
#define DEFAULT_N            256
#define DEFAULT_DELTA_N      6
#define DEFAULT_M            8
#define DEFAULT_PAGE_MB      2
#define DEFAULT_MARGIN       8
#define STRIDE_BYTES         32   // one cache line

#define PREV_LAT_LOW      800
#define TEST_TIME           5
#define TOTAL_SETS          8


// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


__global__ void pointerchase(
    uint32_t  *pointer,
    uint32_t  *dummy,
    uint64_t  *time_log,
    uint64_t   max_steps
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t temp = 0;
    uint32_t sum  = 0;
    uint64_t start_clk, end_clk;
    uint64_t kk = 0;
    while (kk < max_steps && temp != (uint32_t) -1){
        uint32_t *p = &pointer[temp];
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_clk));
        asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
        sum += temp;
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_clk));

        uint64_t lat    = end_clk - start_clk;
        time_log[kk]    = lat;
        kk++;
    }

    atomicAdd(dummy, sum);
}

[[maybe_unused]] static void build_global_chain(uint32_t *arr,
                                uint64_t  total_pages,
                                uint64_t  page_elems,
                                uint64_t  stride_elems)
{
    uint64_t total_slots = total_pages * (page_elems / stride_elems);
    for (uint64_t i = 0; i < total_slots; i++) {
        uint64_t cur  =  i                      * stride_elems;
        uint64_t next = ((i + 1) % total_slots) * stride_elems;
        arr[cur] = (uint32_t)next;
    }
}

// Replace build_global_chain with per-page local chains:
static void build_per_page_chain(uint32_t *arr,
                                  uint64_t  total_pages,
                                  uint64_t  page_elems,
                                  uint64_t  stride_elems)
{
    uint64_t slots = page_elems / stride_elems;
    for (uint64_t pg = 0; pg < total_pages; pg++) {
        uint64_t base = pg * page_elems;
        for (uint64_t i = 0; i < slots; i++) {
            uint64_t cur  = base + i * stride_elems;
            // next is a LOCAL offset from base, not a global index
            uint64_t next_local = ((i + 1) % slots) * stride_elems;
            arr[cur] = (uint32_t)next_local;
        }
    }
}

__global__ void detect_migration(
    uint32_t  *pointer,
    uint32_t  *dummy,
    uint64_t  *time_log,
    uint64_t   max_steps,
    uint32_t  *out_next,
    bool      *migration_detected
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t temp = 0;
    uint32_t sum  = 0;
    uint64_t start_clk, end_clk;
    uint64_t kk = 0;
    uint64_t latency;
    *migration_detected = false;
    uint64_t migration_point = 300;
    while (/* kk < max_steps &&  */temp != (uint32_t) -1){
        uint32_t* p = &pointer[temp];
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_clk));
        asm volatile ("ld.volatile.global.u32 %0, [%1];" : "=r"(temp) : "l"(p));
        sum += temp;
        asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_clk));
            latency = end_clk - start_clk;
            //printf("latency: %lu\n",latency);
            time_log[kk] = end_clk -start_clk;
            if ( latency < 800){
                migration_point = kk;
                break;
            }
        kk++;
    }
    printf("migration point: %lu\n",migration_point);

    if(migration_point < 257)
        *migration_detected = true;

    atomicAdd(dummy, sum);
    *out_next = kk;
}

uintptr_t va_to_pa(void* vaddr);

bool check_page_eviction_set(uint32_t* arr, int phase_1, int phase_2, int phase_3,
                            size_t PAGE_BYTES, size_t PAGE_ELEMS, int M,
                            uint32_t* d_dummy, uint32_t* h_dummy,
                            uint64_t* d_times, uint64_t* h_times,
                            uint32_t* d_migration_point, uint32_t* h_migration_point,
                            bool* d_mig_detected, bool* h_mig_detected,
                            uint32_t general_dummy, size_t total_timings,
                            std::vector<int>& migration_results, std::vector<int>& migration_points,
                            std::vector<int>eviction_elements, int test_time, int N, uint32_t* target_page)
{
    int num_ev_elem = 0;
    bool evicted = false;
    
    for(int temp_size=eviction_elements.size(); temp_size<=eviction_elements.size();temp_size++)
    {
        pointerchase<<<1,1>>>(target_page , d_dummy, d_times+phase_1*(temp_size+1), N+1);
        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(vector_id+1), N+1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }
        sleep(1);

        cudaMemPrefetchAsync(target_page, PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page_id), PAGE_BYTES, cudaCpuDeviceId, 0);
            cudaDeviceSynchronize();
        }
        sleep(1);
        uintptr_t pa1 = va_to_pa(target_page);
        pointerchase<<<1,1>>>(target_page, d_dummy, d_times, phase_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        general_dummy += *h_dummy;

        for(int vector_id=0; vector_id<temp_size;vector_id++){
            int page_id = eviction_elements[vector_id];
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(vector_id+1), phase_2);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }

        va_to_pa(target_page);
        detect_migration<<<1,1>>>(target_page, d_dummy, d_times + phase_1 + (temp_size+1)*phase_2, phase_3, d_migration_point,  d_mig_detected);
        //pointerchase<<<1,1>>>(arr, d_dummy, d_times, delta_n);
        CUDA_CHECK(cudaDeviceSynchronize());
        //sleep(1);
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t),                     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_times, d_times, total_timings * sizeof(uint64_t),     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_migration_point, d_migration_point, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mig_detected, d_mig_detected, sizeof(bool),           cudaMemcpyDeviceToHost));
        migration_points.push_back(*h_migration_point);
        general_dummy += *h_dummy;
        uintptr_t pa2 = va_to_pa(target_page);
        if(*h_mig_detected)
        {
            migration_results.push_back(0);
            evicted = false;
            //printf("EVICTION: Migration Detected - %d pages did not distract counter\n",temp_size);
        }
        else
        {
            //eviction_elements.push_back(page_trial);
            num_ev_elem++;
            evicted = true;
            migration_results.push_back(1);
            //printf("EVICTION: No Migration - %d pages distracted counter\n",temp_size);

        }
        pointerchase<<<1,1>>>(target_page , d_dummy, d_times+phase_1*(temp_size+1), N+1);
        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(vector_id+1), N+1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }
        sleep(1);

        cudaMemPrefetchAsync(target_page, PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page_id), PAGE_BYTES, cudaCpuDeviceId, 0);
            cudaDeviceSynchronize();
        }
        va_to_pa(arr);
    }
    printf("General dummy: %d\n",general_dummy);
    return evicted;

    
}

int find_eviction_set(uint32_t* arr, int phase_1, int phase_2, int phase_3,
                      size_t PAGE_BYTES, size_t PAGE_ELEMS, int M,
                      uint32_t* d_dummy, uint32_t* h_dummy,
                      uint64_t* d_times, uint64_t* h_times,
                      uint32_t* d_migration_point, uint32_t* h_migration_point,
                      bool* d_mig_detected, bool* h_mig_detected,
                      uint32_t general_dummy, size_t total_timings,
                      std::vector<int>& migration_results, std::vector<int>& migration_points,
                      std::vector<int>eviction_elements, int test_time, int N)
{
    int num_ev_elem = 0;
    
    for(int temp_size=1; temp_size<=eviction_elements.size();temp_size++)
    {
        pointerchase<<<1,1>>>(arr , d_dummy, d_times+phase_1*(temp_size+1), N+1);
        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(vector_id+1), N+1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }
        sleep(1);

        cudaMemPrefetchAsync(arr, PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page_id), PAGE_BYTES, cudaCpuDeviceId, 0);
            cudaDeviceSynchronize();
        }
        sleep(1);
        //uintptr_t pa1 = va_to_pa(arr);
        pointerchase<<<1,1>>>(arr, d_dummy, d_times, phase_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        general_dummy += *h_dummy;

        for(int vector_id=0; vector_id<temp_size;vector_id++){
            int page_id = eviction_elements[vector_id];
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(vector_id+1), phase_2);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }

        //va_to_pa(arr);
        detect_migration<<<1,1>>>(arr, d_dummy, d_times + phase_1 + (temp_size+1)*phase_2, phase_3, d_migration_point,  d_mig_detected);
        //pointerchase<<<1,1>>>(arr, d_dummy, d_times, delta_n);
        CUDA_CHECK(cudaDeviceSynchronize());
        //sleep(1);
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t),                     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_times, d_times, total_timings * sizeof(uint64_t),     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_migration_point, d_migration_point, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mig_detected, d_mig_detected, sizeof(bool),           cudaMemcpyDeviceToHost));
        migration_points.push_back(*h_migration_point);
        general_dummy += *h_dummy;
        //uintptr_t pa2 = va_to_pa(arr);
        if(*h_mig_detected)
        {
            migration_results.push_back(0);
            //printf("EVICTION: Migration Detected - %d pages did not distract counter\n",temp_size);
            
        }
        else
        {
            //eviction_elements.push_back(page_trial);
            num_ev_elem++;
            migration_results.push_back(1);
            
            //printf("EVICTION: No Migration - %d pages distracted counter\n",temp_size);

        }
        pointerchase<<<1,1>>>(arr , d_dummy, d_times+phase_1*(temp_size+1), N+1);
        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(vector_id+1), N+1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }
        sleep(1);

        cudaMemPrefetchAsync(arr, PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();

        for(int vector_id=0; vector_id<temp_size; vector_id++)
        {
            int page_id = eviction_elements[vector_id];
            cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page_id), PAGE_BYTES, cudaCpuDeviceId, 0);
            cudaDeviceSynchronize();
        }
        //va_to_pa(arr);
    }
    return (eviction_elements.size() - num_ev_elem+1);

    printf("General dummy: %d\n",general_dummy);
}


bool check_page(uint32_t* arr, int phase_1, int phase_2, int phase_3,
                size_t PAGE_BYTES, size_t PAGE_ELEMS, int M,
                uint32_t* d_dummy, uint32_t* h_dummy,
                uint64_t* d_times, uint64_t* h_times,
                uint32_t* d_migration_point, uint32_t* h_migration_point,
                bool* d_mig_detected, bool* h_mig_detected,
                uint32_t general_dummy, size_t total_timings,
                std::vector<int>& migration_results, std::vector<int>& migration_points,
                std::vector<int>eviction_elements, int test_time, int page_num, int N,
                std::vector <int> all_eviction_elements, uint32_t* page0, int offset)
{
    printf("Start Checking an element\n\n");
    for (int page = 0; page<(page_num+1);page++){
        cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page), PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();
    }

    int test = test_time;
    int check_time = test_time;
    while(test)
    {
        pointerchase<<<1,1>>>(page0, d_dummy, d_times, phase_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        general_dummy += *h_dummy;
        for(int page_id=offset; page_id<page_num; page_id++)
        {
            if (std::count(all_eviction_elements.begin(), all_eviction_elements.end(),page_id+1)>0){
                continue;
            }
            //printf("page id: %d:  ",page_id);
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id +1), d_dummy, d_times+phase_1*(page_id+1), phase_2);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
            //va_to_pa(arr + (size_t)PAGE_ELEMS*((size_t)page_id +1));
        }

        detect_migration<<<1,1>>>(page0, d_dummy, d_times + phase_1 + page_num * phase_2, phase_3, d_migration_point,  d_mig_detected);
        //pointerchase<<<1,1>>>(arr, d_dummy, d_times, delta_n);
        CUDA_CHECK(cudaDeviceSynchronize());
        //sleep(1);
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t),                     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_times, d_times, total_timings * sizeof(uint64_t),     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_migration_point, d_migration_point, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mig_detected, d_mig_detected, sizeof(bool),           cudaMemcpyDeviceToHost));
        //migration_points.push_back(*h_migration_point);
        general_dummy += *h_dummy;

        for(int page_id=offset; page_id<page_num+1; page_id++)
        {
            if ((std::count(all_eviction_elements.begin(), all_eviction_elements.end(),page_id)>0)){
                continue;
            }
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(page_id+1), N+1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }
        sleep(1);

        for (int page = offset; page<page_num+1;page++){
            if ((std::count(all_eviction_elements.begin(), all_eviction_elements.end(),page)>0)){
                continue;
            }
            cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page), PAGE_BYTES, cudaCpuDeviceId, 0);
            cudaDeviceSynchronize();
        }

        if(*h_mig_detected)
        {
            migration_results.push_back(0);
            printf("CHECKING: Migration Detected - %d pages did not distract counter\n",page_num);
        }
        else
        {
            
            migration_results.push_back(1);
            check_time--;
            printf("CHECKING: No Migration - %d pages distracted counter\n",page_num);
        }
        test--;
    }

    bool status = true;
    if (check_time>0)
        status = false;

    return status;
}

void build_eviction_set(uint32_t* arr, int phase_1, int phase_2, int phase_3,
                                     size_t PAGE_BYTES, size_t PAGE_ELEMS, int M,
                                     uint32_t* d_dummy, uint32_t* h_dummy,
                                     uint64_t* d_times, uint64_t* h_times,
                                     uint32_t* d_migration_point, uint32_t* h_migration_point,
                                     bool* d_mig_detected, bool* h_mig_detected,
                                     uint32_t general_dummy, size_t total_timings,
                                     std::vector<int>& migration_results, std::vector<int>& migration_points,
                                     int test, int N, int eviction_set_size, std::vector <int>& eviction_elements,
                                     std::vector <int>& all_eviction_elements, uint32_t* page0, int offset
                               )
{
    

    for (int page = offset; page<(M+1);page++){
        cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page), PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();
    }

    for(int page_trial=offset+1; page_trial<=M; page_trial++)
    {
        //phase 1: Access the Page 0
        printf("\n\n");
        uintptr_t pa1 = va_to_pa(page0);
        pointerchase<<<1,1>>>(page0, d_dummy, d_times, phase_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        general_dummy += *h_dummy;
        //va_to_pa(arr);
        for(int page_id=offset; page_id<page_trial; page_id++)
        {
            if ( std::count(all_eviction_elements.begin(), all_eviction_elements.end(),page_id+1)>0){
                continue;
            }
            //printf("page id: %d:  ",page_id);
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id +1), d_dummy, d_times+phase_1*(page_id+1), phase_2);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
            //va_to_pa(arr + (size_t)PAGE_ELEMS*((size_t)page_id +1));
        }   

        
        va_to_pa(page0);
        detect_migration<<<1,1>>>(page0, d_dummy, d_times + phase_1 + page_trial*phase_2, phase_3, d_migration_point,  d_mig_detected);
        //pointerchase<<<1,1>>>(arr, d_dummy, d_times, delta_n);
        CUDA_CHECK(cudaDeviceSynchronize());
        //sleep(1);
        CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t),                     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_times, d_times, total_timings * sizeof(uint64_t),     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_migration_point, d_migration_point, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mig_detected, d_mig_detected, sizeof(bool),           cudaMemcpyDeviceToHost));
        //migration_points.push_back(*h_migration_point);
        general_dummy += *h_dummy;
        uintptr_t pa2 = va_to_pa(page0);
        if(*h_mig_detected)
        {
            migration_results.push_back(0);
            printf("Migration Detected - %d pages did not distract counter\n",page_trial);
        }
        else
        {
            
            migration_results.push_back(1);
            printf("No Migration - %d pages distracted counter\n",page_trial);
        }

        //pointerchase<<<1,1>>>(page0, d_dummy, d_times+phase_1*1, N+1);
        for(int page_id=offset; page_id<page_trial+1; page_id++)
        {
            if (std::count(all_eviction_elements.begin(), all_eviction_elements.end(),page_id)>0){
                continue;
            }
            pointerchase<<<1,1>>>(arr + (size_t)PAGE_ELEMS*((size_t)page_id), d_dummy, d_times+phase_1*(page_id+1), N+1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_dummy, d_dummy, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            general_dummy += *h_dummy;
        }
        sleep(1);

        //cudaMemPrefetchAsync(page0, PAGE_BYTES, cudaCpuDeviceId, 0);
        //cudaDeviceSynchronize();

        for (int page = offset; page<page_trial+1;page++){
            if ((std::count(all_eviction_elements.begin(), all_eviction_elements.end(),page)>0)){
                continue;
            }
            cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page), PAGE_BYTES, cudaCpuDeviceId, 0);
            cudaDeviceSynchronize();
        }
        va_to_pa(page0);
        if(!(*h_mig_detected)){
            bool add_to_set = check_page(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M, 
                                         d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                                         d_mig_detected, h_mig_detected, general_dummy, total_timings, migration_results, 
                                         migration_points, eviction_elements, test, page_trial,N,all_eviction_elements, page0, offset);
            if(add_to_set){
                eviction_elements.push_back(page_trial);
                all_eviction_elements.push_back(page_trial);
            }
        }
        if (eviction_elements.size()>=eviction_set_size){
            break;
        }
    }
    printf("GENERAL dummy: %u\n", general_dummy);

    //return eviction_elements;
}

int main(int argc, char **argv)
{
    // -----------------------------------------------------------------------
    // 1. Command-line parameters (with defaults)
    // -----------------------------------------------------------------------
    int  M       = DEFAULT_M;
    int  N       = DEFAULT_N;
    int  delta_n = DEFAULT_DELTA_N;
    int  page_mb = DEFAULT_PAGE_MB;
    [[maybe_unused]] int  eviction_set_size = 16;
    int  test_time = TEST_TIME;

    static struct option opts[] = {
        {"num-pages",        required_argument, 0, 'M'},
        {"accesses-thr",     required_argument, 0, 'N'},
        {"delta-n",          required_argument, 0, 'd'},
        {"page-size-mb",     required_argument, 0, 'p'},
        {"eviction-set-size",required_argument, 0, 'E'},
        {"test",             required_argument, 0, 'T'},
        {0, 0, 0, 0}
    };
    int c, idx;
    while ((c = getopt_long(argc, argv, "M:N:d:p:E:T", opts, &idx)) != -1) {
        switch (c) {
        case 'M': M       = atoi(optarg); break;
        case 'N': N       = atoi(optarg); break;
        case 'd': delta_n = atoi(optarg); break;
        case 'p': page_mb = atoi(optarg); break;
        case 'E': eviction_set_size = atoi(optarg); break;
        case 'T': test_time = atoi(optarg); break;
        }
    }

    // -----------------------------------------------------------------------
    // 2. Derived access-phase counts and page geometry
    // -----------------------------------------------------------------------
    int phase_1 = N - delta_n; // number of acceses to the first page before probing M pages
    int phase_2 = phase_1; // number of accesses to each M distractor pages
    int phase_3 = phase_1; // accesses until a migration occurs

    const size_t PAGE_BYTES     = (size_t)page_mb * 1024 * 1024;
    const size_t ELEM_BYTES     = sizeof(uint32_t);
    const size_t STRIDE_ELEMS   = STRIDE_BYTES / ELEM_BYTES;
    const size_t PAGE_ELEMS     = PAGE_BYTES / ELEM_BYTES;
    [[maybe_unused]] const size_t STEPS_PER_PAGE = PAGE_ELEMS / STRIDE_ELEMS;

    // -----------------------------------------------------------------------
    // 3. Allocate the probe buffer and build the pointer-chase chains
    // -----------------------------------------------------------------------
    int total_num_pages = M+1;
    size_t total_bytes = (size_t) total_num_pages * PAGE_BYTES;

    uint32_t* arr = NULL;
    if (posix_memalign((void**)&arr, PAGE_BYTES, total_bytes) != 0) {
        perror("posix_memalign failed");
        return 1;
    } 
    
    //memset(arr, 0, total_bytes);  // 

    for (int page = 0; page<(M+1);page++){
        cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page), PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();
    }

    
    build_per_page_chain(arr, total_num_pages, PAGE_ELEMS, STRIDE_ELEMS);

    
    // -----------------------------------------------------------------------
    // 4. Allocate host/device scratch buffers
    // -----------------------------------------------------------------------
    size_t total_timings = phase_1+ M * phase_2 + phase_3;

    uint32_t* d_dummy; uint32_t* h_dummy;
    uint64_t* d_times; uint64_t* h_times;

    CUDA_CHECK(cudaMalloc((void **) &d_dummy, sizeof(uint32_t)));
    h_dummy = (uint32_t *) malloc(sizeof(uint32_t));

    CUDA_CHECK(cudaMalloc((void **) &d_times, total_timings * sizeof(uint64_t)));
    h_times = (uint64_t *) malloc(total_timings * sizeof(uint64_t));

    uint32_t general_dummy = 0;

    uint32_t* d_migration_point; uint32_t* h_migration_point;

    CUDA_CHECK(cudaMalloc((void **) &d_migration_point, sizeof(uint32_t)));
    h_migration_point = (uint32_t *) malloc(sizeof(uint32_t));

    bool* d_mig_detected; bool* h_mig_detected;
    CUDA_CHECK(cudaMalloc((void **) &d_mig_detected, sizeof(bool)));
    h_mig_detected = (bool *) malloc(sizeof(bool));

    std::vector <int> migration_results;
    std::vector <int> migration_points;
    
    std::vector <int> all_eviction_elements;
    std::vector <std::vector<int>> all_eviction_sets;

    // -----------------------------------------------------------------------
    // 5. Discover the unique eviction sets (stop after TOTAL_SETS)
    // -----------------------------------------------------------------------
    for(int global_page_id = 0; global_page_id<total_num_pages; global_page_id++)
    {
        uint32_t* page_pointer = arr + (size_t) global_page_id * PAGE_ELEMS;
        std::vector <int> eviction_elements;
        if(std::count(all_eviction_elements.begin(), all_eviction_elements.end(),global_page_id)>0){
            printf("PAGE %d already captured\n",global_page_id);
            continue;
        }

        //check if there is an already eviction set that this page belongs to:
        bool taken = false;
        for(int set_id = 0; set_id<all_eviction_sets.size(); set_id++)
        {
            bool status1 = check_page_eviction_set(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M - global_page_id, 
                                                   d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                                                   d_mig_detected, h_mig_detected, general_dummy, total_timings, migration_results, 
                                                   migration_points, all_eviction_sets[set_id], test_time, N,page_pointer);
            bool status2 = check_page_eviction_set(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M - global_page_id, 
                                                   d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                                                   d_mig_detected, h_mig_detected, general_dummy, total_timings, migration_results, 
                                                   migration_points, all_eviction_sets[set_id], test_time, N,page_pointer);
            bool status3 = check_page_eviction_set(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M - global_page_id, 
                                                   d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                                                   d_mig_detected, h_mig_detected, general_dummy, total_timings, migration_results, 
                                                   migration_points, all_eviction_sets[set_id], test_time, N,page_pointer);
            bool status4 = check_page_eviction_set(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M - global_page_id, 
                                                   d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                                                   d_mig_detected, h_mig_detected, general_dummy, total_timings, migration_results, 
                                                   migration_points, all_eviction_sets[set_id], test_time, N,page_pointer);
            bool status5 = check_page_eviction_set(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M - global_page_id, 
                                                   d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                                                   d_mig_detected, h_mig_detected, general_dummy, total_timings, migration_results, 
                                                   migration_points, all_eviction_sets[set_id], test_time, N,page_pointer);

            if(status1 && status2 && status3 && status4 &&status5){
                all_eviction_sets[set_id].push_back(global_page_id);
                taken = true;
                printf("Page taken: %d\n", global_page_id);
                break;
            }
        }
        if (taken)
            continue;
        printf("Page: %d\n", global_page_id);
        eviction_elements.push_back(global_page_id);
        build_eviction_set(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M-1, 
                       d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                       d_mig_detected, h_mig_detected, general_dummy, total_timings,migration_results,
                       migration_points, test_time, N, M,eviction_elements, all_eviction_elements,page_pointer, global_page_id);
        
        all_eviction_sets.push_back(eviction_elements);
        //all_eviction_elements.insert(all_eviction_elements.end(), eviction_elements.begin(), eviction_elements.end());
        eviction_elements.clear();
        printf("Number of unique sets: %zu\n",all_eviction_sets.size());
        /* printf("Starting building eviction set\n");

        
        
        int found_set_size = find_eviction_set(arr, phase_1, phase_2, phase_3, PAGE_BYTES, PAGE_ELEMS, M, 
                                                d_dummy, h_dummy, d_times, h_times, d_migration_point, h_migration_point, 
                                                d_mig_detected, h_mig_detected, general_dummy, total_timings, migration_results, 
                                                migration_points, eviction_elements, test_time, N); */
        if (all_eviction_sets.size() == TOTAL_SETS)
            break;
    }
    
    printf("Total number of unique sets: %zu\n\n\n Eviction Sest:\n",all_eviction_sets.size());
    //printf("Found Eviction Set size: %d\n", found_set_size);
    /* for(int i=0;i<all_eviction_sets.size();i++){
        for (int k=0; k<all_eviction_sets[i].size(); k++) {
            printf("%d, ", all_eviction_sets[i][k]);
        }
        printf("\n");
    } */
    // -----------------------------------------------------------------------
    // 6. Write results to ./texts/ (create the directory if it does not exist)
    // -----------------------------------------------------------------------
    mkdir("./texts", 0755);   // no-op if it already exists

    FILE* fptr;
    fptr = fopen("./texts/all_sets_pa", "w");
    for(int i=0;i<all_eviction_sets.size();i++){
        printf("\n");
        fprintf(fptr, "\n");
        for (int k=0; k<all_eviction_sets[i].size(); k++) {
            uintptr_t pa = va_to_pa(&arr[all_eviction_sets[i][k] * PAGE_ELEMS]);
            fprintf(fptr, "0x%016lx \n", (unsigned long)pa);
            //printf("%d, ", all_eviction_sets[i][k]);
            //printf("VA: 0x%016lx  -->  PA: 0x%016lx\n",(unsigned long)virt, (unsigned long)pa);
        }
        fprintf(fptr, "============================================\n");
        printf("============================================\n");
    }   
    fclose(fptr);

    /* fptr = fopen("./texts/hw_counters_number_guess", "w");
    for (uint64_t k = 0; k < total_timings; k++) {
        //if (h_times[k] == 0) break;
        fprintf(fptr, "%lu \n", h_times[k]);
    }
    fclose(fptr);

    fptr = fopen("./texts/migration_results", "w");
    for (uint64_t k = 0; k < migration_results.size(); k++) {
        //if (h_times[k] == 0) break;
        fprintf(fptr, "%d \n", migration_results[k]);
    }
    fclose(fptr);

    fptr = fopen("./texts/migration_points", "w");
    for (uint64_t k = 0; k < migration_points.size(); k++) {
        //if (h_times[k] == 0) break;
        fprintf(fptr, "%d \n", migration_points[k]);
    }
    fclose(fptr); */


    printf("About to exit\n");
    fflush(stdout);

    // -----------------------------------------------------------------------
    // 7. Cleanup
    // -----------------------------------------------------------------------
    // Ensure all pages are back on CPU before freeing
    // This goes through the cache HIT path and restores ref state cleanly
    for (int page = 0; page < (M+1); page++) {
        cudaMemPrefetchAsync(arr + (size_t)PAGE_ELEMS*((size_t)page),
                            PAGE_BYTES, cudaCpuDeviceId, 0);
        cudaDeviceSynchronize();
    }
    
    printf("prefetch to CPU done\n");
    fflush(stdout);

    cudaFree(d_dummy);
    cudaFree(d_times);
    cudaFree(d_migration_point);
    cudaFree(d_mig_detected);
    printf("cudaFree done\n");
    fflush(stdout);

    //cudaDeviceReset();
    //printf("cudaDeviceReset done\n");
    //fflush(stdout);

    // Don't call free(arr) — let OS reclaim on process exit
    // after all kernel/UVM cleanup has fully completed
    // free(arr);  ← REMOVE THIS

    printf("exiting cleanly\n");
    fflush(stdout);
    return 0;
}


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

    uint64_t pfn  = entry & ((1ULL << 55) - 1);
    uintptr_t pa  = (pfn * page_size) + offset;

    printf("VA: 0x%016lx  -->  PA: 0x%016lx\n",
           (unsigned long)virt, (unsigned long)pa);

    return pa;
}