#include <iostream>
#include <vector>
#include <queue>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <x86intrin.h>
#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"

struct CompareByFirst {
    constexpr bool operator()(const std::pair<float, uint32_t>& a,
                              const std::pair<float, uint32_t>& b) const noexcept {
        return a.first < b.first;
    }
};

// Accurate cycle counter with serialization
static inline uint64_t rdtsc_start() {
    unsigned int aux;
    _mm_lfence();
    return __rdtscp(&aux);
}

static inline uint64_t rdtsc_end() {
    unsigned int aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

// Prevent compiler from optimizing away memory
static inline void escape(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "   HARDWARE CYCLE QUANTIFICATION MICROBENCHMARK        \n";
    std::cout << "   CPU: Intel Core i5-13400 (Raptor Lake)              \n";
    std::cout << "========================================================\n\n";

    const int WARMUP_ITERS = 10000;
    const int TEST_ITERS = 50000;

    // -------------------------------------------------------------
    // 1. MEASURE DRAM CACHE MISS LATENCY (POINTER CHASING)
    // -------------------------------------------------------------
    {
        // Allocate a 64 MB buffer (much larger than 20 MB L3 cache)
        size_t dram_size = 64 * 1024 * 1024 / sizeof(uintptr_t);
        std::vector<uintptr_t> chase_buffer(dram_size);
        
        // Build a randomized pointer chasing cycle with 64-byte strides
        size_t num_lines = dram_size / 8; // 64 bytes = 8 uintptr_t
        std::vector<size_t> indices(num_lines);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());
        
        for (size_t i = 0; i < num_lines; ++i) {
            size_t next_idx = indices[(i + 1) % num_lines];
            chase_buffer[indices[i] * 8] = reinterpret_cast<uintptr_t>(&chase_buffer[next_idx * 8]);
        }

        uintptr_t* p = &chase_buffer[indices[0] * 8];
        // Warmup
        for (int i = 0; i < 1000; ++i) {
            p = reinterpret_cast<uintptr_t*>(*p);
        }

        std::vector<uint64_t> dram_cycles;
        dram_cycles.reserve(10000);
        for (int i = 0; i < 10000; ++i) {
            uint64_t t0 = rdtsc_start();
            p = reinterpret_cast<uintptr_t*>(*p);
            uint64_t t1 = rdtsc_end();
            dram_cycles.push_back(t1 - t0);
        }
        std::sort(dram_cycles.begin(), dram_cycles.end());
        escape(p);

        std::cout << "[1] DRAM UNCACHED MEMORY ACCESS LATENCY:\n";
        std::cout << "    - Min Cycles:    " << dram_cycles[0] << " cycles\n";
        std::cout << "    - Median Cycles: " << dram_cycles[dram_cycles.size() / 2] << " cycles\n";
        std::cout << "    - P90 Cycles:    " << dram_cycles[dram_cycles.size() * 9 / 10] << " cycles\n";
        std::cout << "    - P99 Cycles:    " << dram_cycles[dram_cycles.size() * 99 / 100] << " cycles\n\n";
    }

    // -------------------------------------------------------------
    // 2. MEASURE AVX2 DISTANCE COMPUTE CYCLES (ENRON: d = 1369)
    // -------------------------------------------------------------
    {
        size_t dim_enron = 1369;
        hnswlib::L2Space space_enron(dim_enron);
        auto dist_func_enron = space_enron.get_dist_func();
        auto dist_param_enron = space_enron.get_dist_func_param();

        std::vector<float> v1(dim_enron, 0.5f);
        std::vector<float> v2(dim_enron, 0.7f);

        // Warm cache distance compute
        std::vector<uint64_t> enron_warm_cycles;
        enron_warm_cycles.reserve(TEST_ITERS);

        for (int i = 0; i < WARMUP_ITERS; ++i) {
            escape(v1.data());
            escape(v2.data());
            volatile float d = dist_func_enron(v1.data(), v2.data(), dist_param_enron);
            (void)d;
        }

        for (int i = 0; i < TEST_ITERS; ++i) {
            escape(v1.data());
            escape(v2.data());
            uint64_t t0 = rdtsc_start();
            volatile float d = dist_func_enron(v1.data(), v2.data(), dist_param_enron);
            uint64_t t1 = rdtsc_end();
            (void)d;
            enron_warm_cycles.push_back(t1 - t0);
        }
        std::sort(enron_warm_cycles.begin(), enron_warm_cycles.end());

        std::cout << "[2] AVX2 DISTANCE COMPUTE - ENRON (d = 1,369 floats = 5,476 bytes):\n";
        std::cout << "    - Min Cycles:    " << enron_warm_cycles[0] << " cycles\n";
        std::cout << "    - Median Cycles: " << enron_warm_cycles[enron_warm_cycles.size() / 2] << " cycles\n";
        std::cout << "    - P90 Cycles:    " << enron_warm_cycles[enron_warm_cycles.size() * 9 / 10] << " cycles\n";
        std::cout << "    - SIMD operations: (85 iterations x 2 AVX2 ops) + 9 residual ops = ~179 ops\n\n";
    }

    // -------------------------------------------------------------
    // 3. MEASURE AVX2 DISTANCE COMPUTE CYCLES (SIFT: d = 128)
    // -------------------------------------------------------------
    {
        size_t dim_sift = 128;
        hnswlib::L2Space space_sift(dim_sift);
        auto dist_func_sift = space_sift.get_dist_func();
        auto dist_param_sift = space_sift.get_dist_func_param();

        std::vector<float> v1(dim_sift, 0.5f);
        std::vector<float> v2(dim_sift, 0.7f);

        std::vector<uint64_t> sift_warm_cycles;
        sift_warm_cycles.reserve(TEST_ITERS);

        for (int i = 0; i < WARMUP_ITERS; ++i) {
            escape(v1.data());
            escape(v2.data());
            volatile float d = dist_func_sift(v1.data(), v2.data(), dist_param_sift);
            (void)d;
        }

        for (int i = 0; i < TEST_ITERS; ++i) {
            escape(v1.data());
            escape(v2.data());
            uint64_t t0 = rdtsc_start();
            volatile float d = dist_func_sift(v1.data(), v2.data(), dist_param_sift);
            uint64_t t1 = rdtsc_end();
            (void)d;
            sift_warm_cycles.push_back(t1 - t0);
        }
        std::sort(sift_warm_cycles.begin(), sift_warm_cycles.end());

        std::cout << "[3] AVX2 DISTANCE COMPUTE - SIFT (d = 128 floats = 512 bytes):\n";
        std::cout << "    - Min Cycles:    " << sift_warm_cycles[0] << " cycles\n";
        std::cout << "    - Median Cycles: " << sift_warm_cycles[sift_warm_cycles.size() / 2] << " cycles\n";
        std::cout << "    - P90 Cycles:    " << sift_warm_cycles[sift_warm_cycles.size() * 9 / 10] << " cycles\n";
        std::cout << "    - SIMD operations: 8 iterations x 2 AVX2 ops = 16 ops\n\n";
    }

    // -------------------------------------------------------------
    // 4. MEASURE STEP 1 SEARCH STRUCTURE SETUP CYCLES
    // -------------------------------------------------------------
    {
        typedef float dist_t;
        typedef uint32_t tableint;
        typedef unsigned short int vl_type;

        hnswlib::VisitedListPool visited_list_pool(4, 100000);
        tableint ep_id = 48192;
        dist_t dist = 1.4142f;

        std::vector<uint64_t> setup_cycles;
        setup_cycles.reserve(TEST_ITERS);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            hnswlib::VisitedList *vl = visited_list_pool.getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
            visited_array[ep_id] = visited_array_tag;

            visited_list_pool.releaseVisitedList(vl);
        }

        // Measure Step 1 Housekeeping: Structure initialization up to while(!candidate_set.empty())
        for (int i = 0; i < TEST_ITERS; ++i) {
            uint64_t t0 = rdtsc_start();

            // 1. Visited List pool access & tag acquisition
            hnswlib::VisitedList *vl = visited_list_pool.getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // 2. Priority Queue instantiations (2 heaps)
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            // 3. Flags, scalar bounds, initial emplace
            dist_t lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);

            // 4. Mark visited
            visited_array[ep_id] = visited_array_tag;

            uint64_t t1 = rdtsc_end();
            setup_cycles.push_back(t1 - t0);

            visited_list_pool.releaseVisitedList(vl);
        }
        std::sort(setup_cycles.begin(), setup_cycles.end());

        std::cout << "[4] STEP 1 SEARCH STRUCTURE SETUP LATENCY:\n";
        std::cout << "    - Min Cycles:    " << setup_cycles[0] << " cycles\n";
        std::cout << "    - Median Cycles: " << setup_cycles[setup_cycles.size() / 2] << " cycles\n";
        std::cout << "    - P90 Cycles:    " << setup_cycles[setup_cycles.size() * 9 / 10] << " cycles\n";
        std::cout << "    - P99 Cycles:    " << setup_cycles[setup_cycles.size() * 99 / 100] << " cycles\n\n";

        // Subcomponent breakdown
        // Component A: Visited list pool only
        std::vector<uint64_t> pool_cycles;
        for (int i = 0; i < 10000; ++i) {
            uint64_t t0 = rdtsc_start();
            hnswlib::VisitedList *vl = visited_list_pool.getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            escape(&visited_array);
            escape(&visited_array_tag);
            uint64_t t1 = rdtsc_end();
            pool_cycles.push_back(t1 - t0);
            visited_list_pool.releaseVisitedList(vl);
        }
        std::sort(pool_cycles.begin(), pool_cycles.end());

        // Component B: Two priority queues only
        std::vector<uint64_t> heap_cycles;
        for (int i = 0; i < 10000; ++i) {
            uint64_t t0 = rdtsc_start();
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
            escape(&top_candidates);
            escape(&candidate_set);
            uint64_t t1 = rdtsc_end();
            heap_cycles.push_back(t1 - t0);
        }
        std::sort(heap_cycles.begin(), heap_cycles.end());

        std::cout << "    Component Breakdown (Median):\n";
        std::cout << "      * VisitedList Pool + Tag:      " << pool_cycles[pool_cycles.size() / 2] << " cycles\n";
        std::cout << "      * 2 Heaps + Initial Emplace:   " << heap_cycles[heap_cycles.size() / 2] << " cycles\n\n";
    }

    std::cout << "========================================================\n";
    std::cout << "                      COMPLETE                          \n";
    std::cout << "========================================================\n";
    return 0;
}
