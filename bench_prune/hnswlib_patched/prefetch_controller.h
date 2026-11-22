#pragma once

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#define HNSW_PREFETCH_T0(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0)
#define HNSW_PREFETCH_T1(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1)
#else
#define HNSW_PREFETCH_T0(ptr) __builtin_prefetch((const void*)(ptr), 0, 3)
#define HNSW_PREFETCH_T1(ptr) __builtin_prefetch((const void*)(ptr), 0, 2)
#endif

namespace hnswlib {

enum class PrefetchMode {
    BASELINE = 0,
    STEP1_SKIP = 1,
    STEP2_SKIP_LOOKAHEAD = 2,
    STEP3_CLUSTER = 3,
    STOCK_HNSW = 4,  // hnswlib's original prefetch lines only (reference behaviour)
    IDEA2_V2 = 5,    // Idea 2.0: warm-up + lookahead j+D + optional visited filter (see v2_* below)
    AUTOTUNE = 6     // Analytical model auto-tuning
};

// Global helper: prefetch contiguous memory across 64-byte cache lines
static inline void prefetch_range(const void* address, size_t bytes, size_t max_lines = 4) {
    if (!address || bytes == 0) return;
    const char* ptr = reinterpret_cast<const char*>(address);
    size_t lines = (bytes + 63) / 64;
    if (lines > max_lines) lines = max_lines;
    for (size_t i = 0; i < lines; ++i) {
        HNSW_PREFETCH_T0(ptr + (i * 64));
    }
}

} // namespace hnswlib

#include "cluster_prefetcher.h"

namespace hnswlib {

/**
 * Analytical Microarchitectural Cost Model Configuration
 */
struct CostModelConfig {
    double t_dram = 380.0;       // DRAM round-trip latency in core cycles
    double t_visited = 15.0;     // Visited check latency
    double t_pq = 25.0;          // Priority queue operation
    double t_overhead = 10.0;    // Loop overhead
    double p_unvisited = 0.45;   // Expected fraction of unvisited neighbors
    double t_simd = 0.0;         // Measured/calibrated SIMD distance compute cycles
    double tau_iter = 0.0;       // Estimated single neighbor iteration latency
    size_t d_star = 2;           // Analytically optimal lookahead distance
    size_t l_star = 3;           // Analytically optimal prefetch lines
    size_t threads = 1;          // Concurrency level
};

/**
 * PrefetchController:
 * Modular controller for skip-aware, lookahead, and cluster graph memory prefetching.
 * Keeps hardware intrinsics and prefetch policies cleanly isolated from core graph search.
 */
class PrefetchController {
public:
    static inline thread_local bool enable_skip_prefetch = true;      // Step 1: Skip-triggered prefetch
    static inline thread_local bool enable_lookahead_prefetch = true; // Step 2: Base-layer lookahead prefetch
    static inline thread_local bool enable_cluster_prefetch = true;   // Step 3: Cluster-aware prefetch
    static inline thread_local bool enable_stock_prefetch = false;
    static inline thread_local size_t lookahead_lines = 3;            // Idea 2: cache lines of the vector to prefetch (3 = original; huge = whole vector)
    static inline thread_local size_t lookahead_dist = 2;             // Idea 2: prefetch the neighbour this many slots ahead
    // ---- Idea 2.0 knobs (used only in IDEA2_V2 mode) ----
    static inline thread_local bool   enable_v2   = false;
    static inline thread_local size_t v2_dist     = 2;     // steady state: while at neighbour j, prefetch neighbour j+D
    static inline thread_local size_t v2_lines    = 3;     // cache lines of each neighbour vector to prefetch
    static inline thread_local bool   v2_warmup   = true;  // before the loop, prefetch neighbours 1..D (they are never j+D)
    static inline thread_local bool   v2_filter   = false; // prefetch visited flag at j+2D, then vector at j+D only if not visited
    static inline thread_local bool   v2_push_ll  = true;  // stock: prefetch link list of the best candidate after each push
    static inline thread_local std::string custom_mode_name = "";

    static inline thread_local CostModelConfig cost_model;

    template <typename DistFunc>
    static inline CostModelConfig compute_optimal_parameters(
        size_t dim,
        DistFunc dist_func,
        const void* dist_param,
        const void* sample_v1 = nullptr,
        const void* sample_v2 = nullptr,
        size_t concurrency_threads = 1
    ) {
        CostModelConfig cfg;
        cfg.threads = concurrency_threads;
        if (sample_v1 && sample_v2) {
            volatile float dummy = 0;
            for (int i = 0; i < 500; i++) dummy += dist_func(sample_v1, sample_v2, dist_param);
            uint64_t c0 = __rdtsc();
            const int ITERS = 20000;
            for (int i = 0; i < ITERS; i++) dummy += dist_func(sample_v1, sample_v2, dist_param);
            uint64_t c1 = __rdtsc();
            cfg.t_simd = (double)(c1 - c0) / ITERS;
        } else {
            cfg.t_simd = std::max(20.0, (double)dim * 0.28);
        }

        cfg.tau_iter = cfg.t_visited + cfg.p_unvisited * (cfg.t_simd + cfg.t_pq) + cfg.t_overhead;
        double raw_d = cfg.t_dram / cfg.tau_iter;
        size_t d = (size_t)std::ceil(raw_d);
        if (d < 1) d = 1;
        if (d > 8) d = 8;
        cfg.d_star = d;

        size_t total_lines = (dim * sizeof(float) + 63) / 64;
        cfg.l_star = std::min((size_t)3, total_lines);

        // Contention-aware adaptation
        if (concurrency_threads >= 32) {
            cfg.l_star = 1;
            if (cfg.d_star > 1) cfg.d_star -= 1;
        } else if (concurrency_threads >= 16) {
            cfg.l_star = std::min(cfg.l_star, (size_t)2);
        }

        return cfg;
    }

    static inline void apply_cost_model(const CostModelConfig& cfg) {
        cost_model = cfg;
        enable_v2 = true;
        v2_dist = cfg.d_star;
        v2_lines = cfg.l_star;
        v2_warmup = true;
        v2_filter = true;
        v2_push_ll = true;
        custom_mode_name = "pf_filter_autotune";
    }

    static inline void set_mode(PrefetchMode mode) {
        enable_stock_prefetch = false;
        enable_v2 = false;
        switch (mode) {
            case PrefetchMode::STOCK_HNSW:
                enable_skip_prefetch = false;
                enable_lookahead_prefetch = false;
                enable_cluster_prefetch = false;
                ClusterPrefetcher::enable_cluster_prefetch = false;
                enable_stock_prefetch = true;
                custom_mode_name = "stock_hnsw_prefetch";
                break;
            case PrefetchMode::BASELINE:
                enable_skip_prefetch = false;
                enable_lookahead_prefetch = false;
                enable_cluster_prefetch = false;
                ClusterPrefetcher::enable_cluster_prefetch = false;
                custom_mode_name = "baseline";
                break;
            case PrefetchMode::STEP1_SKIP:
                enable_skip_prefetch = true;
                enable_lookahead_prefetch = false;
                enable_cluster_prefetch = false;
                ClusterPrefetcher::enable_cluster_prefetch = false;
                custom_mode_name = "step1_skip_prefetch";
                break;
            case PrefetchMode::STEP2_SKIP_LOOKAHEAD:
                enable_skip_prefetch = true;
                enable_lookahead_prefetch = true;
                enable_cluster_prefetch = false;
                ClusterPrefetcher::enable_cluster_prefetch = false;
                custom_mode_name = "step2_skip_lookahead_prefetch";
                break;
            case PrefetchMode::IDEA2_V2:
                enable_skip_prefetch = true;       // Idea 1 (shortcut landing) stays on; SHG only
                enable_lookahead_prefetch = false; // old Idea 2 hook off
                enable_cluster_prefetch = false;
                ClusterPrefetcher::enable_cluster_prefetch = false;
                enable_v2 = true;
                custom_mode_name = "idea2_v2";
                break;
            case PrefetchMode::AUTOTUNE:
                enable_skip_prefetch = true;
                enable_lookahead_prefetch = false;
                enable_cluster_prefetch = false;
                ClusterPrefetcher::enable_cluster_prefetch = false;
                enable_v2 = true;
                v2_dist = cost_model.d_star;
                v2_lines = cost_model.l_star;
                v2_warmup = true;
                v2_filter = true;
                v2_push_ll = true;
                custom_mode_name = "pf_filter_autotune";
                break;
            case PrefetchMode::STEP3_CLUSTER:
                enable_skip_prefetch = true;
                enable_lookahead_prefetch = true;
                enable_cluster_prefetch = true;
                ClusterPrefetcher::enable_cluster_prefetch = true;
                custom_mode_name = "step3_cluster_prefetch";
                break;
        }
    }

    static inline void set_mode_from_string(const std::string& mode_str) {
        if (mode_str == "baseline" || mode_str == "none" || mode_str == "off") {
            set_mode(PrefetchMode::BASELINE);
        } else if (mode_str == "step1" || mode_str == "skip") {
            set_mode(PrefetchMode::STEP1_SKIP);
        } else if (mode_str == "step2" || mode_str == "lookahead") {
            set_mode(PrefetchMode::STEP2_SKIP_LOOKAHEAD);
        } else if (mode_str == "step3" || mode_str == "cluster" || mode_str == "all") {
            set_mode(PrefetchMode::STEP3_CLUSTER);
        } else {
            // Default: step 3 (all optimizations active)
            set_mode(PrefetchMode::STEP3_CLUSTER);
        }
    }

    static inline std::string get_mode_name() {
        if (!custom_mode_name.empty()) return custom_mode_name;
        if (!enable_skip_prefetch && !enable_lookahead_prefetch && !enable_cluster_prefetch) return "baseline";
        if (enable_skip_prefetch && !enable_lookahead_prefetch && !enable_cluster_prefetch) return "step1_skip_prefetch";
        if (enable_skip_prefetch && enable_lookahead_prefetch && !enable_cluster_prefetch) return "step2_skip_lookahead_prefetch";
        if (enable_cluster_prefetch) return "step3_cluster_prefetch";
        return "custom_prefetch";
    }

    /**
     * STEP 1 & STEP 3: Skip-Triggered + Cluster-Aware Destination Prefetching
     */
    template <typename GetLinkListFn, typename GetVectorFn>
    static inline void on_shortcut_jump(
        uint32_t currObj,
        int from_level,
        int dest_level,
        GetLinkListFn&& get_linklist_fn,
        GetVectorFn&& get_vector_fn,
        size_t data_dim_bytes,
        bool is_skip
    ) {
        if (enable_skip_prefetch) {
            // 1. Prefetch destination entry node's linklist at dest_level
            void* linklist = get_linklist_fn(currObj, dest_level);
            if (linklist) {
                prefetch_range(linklist, 128, 2);

                // 2. Prefetch the destination entry node's vector data
                void* node_vector = get_vector_fn(currObj);
                if (node_vector) {
                    prefetch_range(node_vector, data_dim_bytes, 4);
                }

                // 3. Prefetch the first candidate neighbor in the entry node's linklist
                uint32_t* list_data = reinterpret_cast<uint32_t*>(linklist);
                uint32_t count = *list_data;
                if (count > 0) {
                    uint32_t first_neighbor = *(list_data + 1);
                    void* neighbor_vec = get_vector_fn(first_neighbor);
                    if (neighbor_vec) {
                        prefetch_range(neighbor_vec, data_dim_bytes, 2);
                    }
                }
            }
        }

        // STEP 3: Cluster-Aware Target Prefetching
        if (enable_cluster_prefetch && (dest_level == 0 || is_skip)) {
            ClusterPrefetcher::instance().prefetch_cluster_targets(
                currObj,
                get_vector_fn,
                data_dim_bytes
            );
        }
    }

    /**
     * STEP 2: Base-Layer Lookahead Neighbor Prefetching
     */
    template <typename VlType, typename GetVectorFn>
    static inline void on_base_layer_neighbor_lookahead(
        const int* data,
        size_t current_idx,
        size_t total_neighbors,
        const VlType* visited_array,
        GetVectorFn&& get_vector_fn,
        size_t data_dim_bytes,
        size_t lookahead = 2
    ) {
        if (!enable_lookahead_prefetch) return;

        if (current_idx + lookahead <= total_neighbors) {
            int ahead_id = *(data + current_idx + lookahead);
            if (ahead_id >= 0) {
                HNSW_PREFETCH_T0(&visited_array[ahead_id]);
                void* vec = get_vector_fn(ahead_id);
                if (vec) {
                    prefetch_range(vec, data_dim_bytes, lookahead_lines);
                }
            }
        }
    }

    /**
     * STEP 2b: Next Candidate Node Prefetching
     */
    template <typename GetLinkList0Fn>
    static inline void on_candidate_queue_next(
        uint32_t next_node_id,
        GetLinkList0Fn&& get_linklist0_fn
    ) {
        if (!enable_lookahead_prefetch && !enable_v2) return;

        void* linklist = get_linklist0_fn(next_node_id);
        if (linklist) {
            prefetch_range(linklist, 128, 2);
        }
    }

    /**
     * Idea 2.0, warm-up: neighbours 1..D are never "j+D" for any j, so prefetch them before the loop.
     * No visited filter here: their flags are not in cache yet, and reading them would stall.
     */
    template <typename VlType, typename GetVectorFn>
    static inline void v2_prologue(const int* data, size_t size, const VlType* visited,
                                   GetVectorFn&& get_vector_fn, size_t bytes) {
        if (!enable_v2 || !v2_warmup) return;
        size_t last = std::min(v2_dist, size);
        size_t flags_to = v2_filter ? std::min(2 * v2_dist, size) : last;
        for (size_t k = 1; k <= flags_to; k++) HNSW_PREFETCH_T0(&visited[*(data + k)]);
        for (size_t k = 1; k <= last; k++) {
            int id = *(data + k);
            prefetch_range(get_vector_fn(id), bytes, v2_lines);
        }
    }

    /**
     * Idea 2.0, steady state at neighbour j (1-based):
     *   no filter: prefetch flag + vector of j+D
     *   filter   : prefetch flag of j+2D; read flag of j+D (prefetched D steps ago) and
     *              prefetch its vector only if it is not visited yet
     */
    template <typename VlType, typename GetVectorFn>
    static inline void v2_step(const int* data, size_t j, size_t size, const VlType* visited, VlType tag,
                               GetVectorFn&& get_vector_fn, size_t bytes) {
        if (!enable_v2) return;
        size_t k = j + v2_dist;
        if (v2_filter) {
            size_t f = j + 2 * v2_dist;
            if (f <= size) HNSW_PREFETCH_T0(&visited[*(data + f)]);
            if (k <= size) {
                int id = *(data + k);
                if (visited[id] != tag) {
                    prefetch_range(get_vector_fn(id), bytes, v2_lines);
                }
            }
        } else if (k <= size) {
            int id = *(data + k);
            HNSW_PREFETCH_T0(&visited[id]);
            prefetch_range(get_vector_fn(id), bytes, v2_lines);
        }
    }
};

} // namespace hnswlib
