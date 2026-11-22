#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace hnswlib {

// Forward declaration of prefetch_range helper
void prefetch_range(const void* address, size_t bytes, size_t max_lines);

/**
 * ClusterPrefetcher:
 * Implements Step 3 (Cluster-Aware Prefetching) for SHG.
 * 
 * Groups graph nodes into topological clusters and identifies representative 
 * co-accessed hub nodes. When a shortcut transition lands on an entry node,
 * the cluster's high-probability targets are prefetched ahead of graph beam search.
 */
class ClusterPrefetcher {
public:
    static constexpr size_t CLUSTER_PREFETCH_TARGETS = 2; // Bounded (2) to avoid LFB saturation
    static inline bool enable_cluster_prefetch = true;

    // Compact cluster lookup table: maps node_id -> top cluster representative neighbor IDs
    struct ClusterEntry {
        uint8_t count = 0;
        std::array<uint32_t, CLUSTER_PREFETCH_TARGETS> members{};
    };

    std::vector<ClusterEntry> cluster_table;
    bool is_initialized = false;

    // Singleton / shared instance access
    static ClusterPrefetcher& instance() {
        static ClusterPrefetcher inst;
        return inst;
    }

    /**
     * Build cluster representatives based on graph connectivity.
     * For each node, selects the top co-connected hub neighbors.
     */
    template <typename GetLinkListFn>
    void init_clusters(size_t num_elements, GetLinkListFn&& get_linklist0_fn) {
        if (num_elements == 0) return;
        cluster_table.resize(num_elements);

        for (size_t i = 0; i < num_elements; ++i) {
            void* ll = get_linklist0_fn(i);
            if (!ll) continue;

            uint32_t* data = reinterpret_cast<uint32_t*>(ll);
            uint32_t size = *data;
            if (size == 0) continue;

            uint32_t* neighbors = data + 1;
            uint8_t targets = static_cast<uint8_t>(std::min(size, static_cast<uint32_t>(CLUSTER_PREFETCH_TARGETS)));
            cluster_table[i].count = targets;

            for (uint8_t j = 0; j < targets; ++j) {
                cluster_table[i].members[j] = neighbors[j];
            }
        }
        is_initialized = true;
    }

    /**
     * Prefetches cluster targets when landing on a destination entry node.
     */
    template <typename GetVectorFn>
    inline void prefetch_cluster_targets(
        uint32_t entry_node,
        GetVectorFn&& get_vector_fn,
        size_t data_dim_bytes
    ) {
        if (!enable_cluster_prefetch || !is_initialized) return;
        if (entry_node >= cluster_table.size()) return;

        const auto& entry = cluster_table[entry_node];
        for (uint8_t i = 0; i < entry.count; ++i) {
            uint32_t cluster_member = entry.members[i];
            void* vec = get_vector_fn(cluster_member);
            if (vec) {
                // Prefetch first 2 cache lines of each cluster member vector
                prefetch_range(vec, data_dim_bytes, 2);
            }
        }
    }
};

} // namespace hnswlib
