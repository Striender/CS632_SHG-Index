# Idea 2.0 filter (v2_filter): add-only patch

For a codebase that already has `PrefetchController` with BASELINE / STEP1_SKIP /
STEP2_SKIP_LOOKAHEAD / STEP3_CLUSTER (like `SHG-Index/hnswlib/prefetch_controller.h`).
Nothing existing is changed: every addition is guarded by `enable_v2`, which is false unless
`set_mode(PrefetchMode::IDEA2_V2)` is called. The other modes behave exactly as before.

Reference implementation (tested): `/home/neeraj/SHG-Index/bench_prune/hnswlib_patched/`
(`prefetch_controller.h`, `heds.h`; search for "Idea 2.0").

## What it does (base-layer neighbour loop, D = 2, L = 3 lines)
- Before the loop: prefetch the visited flags of neighbours 1..2D and the vectors (L lines) of
  neighbours 1..D.
- At neighbour j: prefetch the visited flag of j+2D. Then read the flag of j+D, which is in cache
  by now. If j+D is not visited, prefetch its vector (L lines); if it is visited, prefetch nothing.
- Also prefetch the next candidate's link list (existing `on_candidate_queue_next`), and the link
  list of the best candidate after each push.

---

## 1. prefetch_controller.h

### 1a. enum: add one value
```cpp
enum class PrefetchMode {
    BASELINE = 0,
    STEP1_SKIP = 1,
    STEP2_SKIP_LOOKAHEAD = 2,
    STEP3_CLUSTER = 3,
    IDEA2_V2 = 5          // NEW: Idea 2.0 filter
};
```
Also add `#include <algorithm>` at the top (for `std::min`).

### 1b. inside `class PrefetchController`, next to the other static flags
```cpp
    // ---- Idea 2.0 filter (used only in IDEA2_V2 mode) ----
    static inline bool   enable_v2 = false;
    static inline size_t v2_dist   = 2;   // D: vector prefetch distance (flag distance is 2D)
    static inline size_t v2_lines  = 3;   // L: cache lines of each neighbour vector
```

### 1c. in `set_mode()`: add `enable_v2 = false;` as the FIRST line (before the switch), and this case
```cpp
            case PrefetchMode::IDEA2_V2:
                enable_skip_prefetch = true;       // Idea 1 stays on (SHG shortcut jumps)
                enable_lookahead_prefetch = false; // old Idea 2 off
                enable_cluster_prefetch = false;
                ClusterPrefetcher::enable_cluster_prefetch = false;
                enable_v2 = true;
                custom_mode_name = "idea2_v2_filter";
                break;
```

### 1d. in `on_candidate_queue_next`: change the guard line only
```cpp
        if (!enable_lookahead_prefetch && !enable_v2) return;   // was: if (!enable_lookahead_prefetch) return;
```

### 1e. add these two functions to the class
```cpp
    // warm-up before the neighbour loop: neighbours 1..D are never "j+D"
    template <typename VlType, typename GetVectorFn>
    static inline void v2_prologue(const int* data, size_t size, const VlType* visited,
                                   GetVectorFn&& get_vector_fn, size_t bytes) {
        if (!enable_v2) return;
        size_t last = std::min(v2_dist, size), flags_to = std::min(2 * v2_dist, size);
        for (size_t k = 1; k <= flags_to; k++) HNSW_PREFETCH_T0(&visited[*(data + k)]);
        for (size_t k = 1; k <= last; k++) prefetch_range(get_vector_fn(*(data + k)), bytes, v2_lines);
    }

    // at neighbour j (1-based): flag of j+2D now, vector of j+D only if j+D is not visited
    template <typename VlType, typename GetVectorFn>
    static inline void v2_step(const int* data, size_t j, size_t size, const VlType* visited, VlType tag,
                               GetVectorFn&& get_vector_fn, size_t bytes) {
        if (!enable_v2) return;
        size_t f = j + 2 * v2_dist, k = j + v2_dist;
        if (f <= size) HNSW_PREFETCH_T0(&visited[*(data + f)]);
        if (k <= size) {
            int id = *(data + k);
            if (visited[id] != tag) prefetch_range(get_vector_fn(id), bytes, v2_lines);
        }
    }
```

---

## 2. heds.h: in BOTH base-layer loops (`searchBaseLayerST` and `searchBaseLayerSTPrune`)

### 2a. just before `for (size_t j = 1; j <= size; j++) {`
```cpp
            PrefetchController::v2_prologue(data, size, visited_array,
                [this](tableint id) -> void* { return (void*)getDataByInternalId(id); }, data_size_);
```

### 2b. at the top of the loop body, right after the existing `on_base_layer_neighbor_lookahead(...)` call
```cpp
                PrefetchController::v2_step(data, j, size, visited_array, visited_array_tag,
                    [this](tableint id) -> void* { return (void*)getDataByInternalId(id); }, data_size_);
```

### 2c. right after `candidate_set.emplace(-dist, candidate_id);`
```cpp
                        if (PrefetchController::enable_v2)
                            _mm_prefetch((char *) get_linklist0(candidate_set.top().second), _MM_HINT_T0);
```

---

## 3. Use
```cpp
hnswlib::PrefetchController::set_mode(hnswlib::PrefetchMode::IDEA2_V2);
```

## 4. Check
Search results (IDs) must be identical to `BASELINE` for every query: this only adds prefetches.

## Measured (M=48, efC=80, efSearch=64, K=20, 1 thread pinned, base layer, cold cache)
Speedup vs no prefetch:

|          | HNSW | SHG  |
|----------|------|------|
| Enron    | 1.14 | 1.09 |
| GIST1M   | 1.31 | 1.18 |
| SIFT1M   | 1.15 | 1.12 |
| SIFT10M  | 1.24 | 1.16 |

0 result mismatches on all datasets
(`/home/neeraj/SHG-Index/bench_prune/stall4_<dataset>/correctness.csv`).
