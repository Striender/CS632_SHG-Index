# Microarchitectural Optimization of SHG-Index: Hardware Prefetching Implementation & Benchmark Insights

## 1. Executive Summary

The VLDB 2024 paper *"Accelerating Approximate Nearest Neighbor Search in Hierarchical Graphs: Efficient Level Navigation with Shortcuts"* (Chen et al.) proposes **SHG**, an index designed to accelerate HNSW by using learned shortcuts to skip intermediate graph levels.

However, empirical evaluation of the authors' open-source codebase on modern x86 hardware revealed critical architectural shortcomings: **Vanilla HNSW consistently outperformed Baseline SHG in both Recall and Throughput**. Profiling with hardware performance counters identified the root cause:
1. Upper-level graph navigation accounts for **less than 1% of total query time**, rendering level-skipping alone insufficient for large speedups on moderate-sized datasets.
2. In the base layer, Baseline SHG suffered severe **CPU idle stalls (56%–73% of hop time)** waiting on Last-Level Cache (LLC) misses and DRAM bus round-trips.
3. The original authors **omitted hardware prefetch instructions** in their base-layer search loop (`searchBaseLayerSTPrune` in `hnswlib/heds.h`), whereas native HNSW possessed degree-1 prefetching.

To solve this, we designed and implemented a two-stage hardware prefetching subsystem (**Idea 1: Skip-Triggered Prefetching** and **Idea 2: Pipelined Neighbor Lookahead Prefetching**). Across both **Enron ($d = 1,369$)** and **SIFT1M ($d = 128$)**, our technique overlaps memory stall cycles with active SIMD computation, yielding a **+15.2% to +15.8% throughput boost** and allowing SHG to surpass HNSW on high-dimensional workloads.

---

## 2. Microarchitectural Problem Analysis

When searching high-dimensional vector graphs, the index footprint far exceeds the CPU cache (e.g., SIFT1M is **873.8 MB**, whereas the shared L3 cache on our Intel Core i5-13400 is **20 MB**). Consequently, accessing unvisited neighbors triggers cold DRAM accesses.

### Measured Cycle Breakdown per Graph Hop (Intel Core i5-13400, Core 2 Dedicated)

| Workload | Dimension ($d$) | DRAM Stall / CPU Idle (Cycles) | Active AVX2 SIMD Compute (Cycles) | Queue / Graph Overhead (Cycles) | Total Hop Latency | **CPU Idle Fraction (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT1M** | 128 | **318 cycles** (~90.9 ns) | **91 cycles** (26.0 ns) | 25 cycles (7.1 ns) | 434 cycles | **73.3% IDLE** |
| **Enron** | 1,369 | **578 cycles** (~165.1 ns) | **410 cycles** (117.1 ns) | 35 cycles (10.0 ns) | 1,023 cycles | **56.5% IDLE** |

```
=== Baseline SHG (Serial Stop-and-Wait Execution) ===
[ DRAM Cache Miss: 318 cycles STALL (CPU IDLE) ] -> [ AVX2 Compute: 91 cyc ] -> [ Queue Ops: 25 cyc ]
                                  ▲
                       73.3% of cycle budget wasted!

=== Proposed SHG with Step 1 + Step 2 Prefetching (Pipelined Overlap) ===
[ Step 1: Prefetch Entry (76 cyc dead time) ]
[ Node 0 AVX2 Compute: 91 cyc ] ──┐
[ Queue Ops: 25 cyc ]             ├─► Overlaps in-flight background DRAM fetch for Node 1!
                                  │   (Memory stall hidden behind active compute)
[ Node 1 AVX2 Compute: 91 cyc ] ◄─┘
```

---

## 3. Implementation Details

Our prefetching architecture is modularly integrated into `hnswlib/` via `hnswlib/prefetch_controller.h` and injected into `hnswlib/heds.h`.

### Idea 1: Skip-Triggered Prefetching (`PrefetchMode::STEP1_SKIP`)
* **Location**: `hnswlib/heds.h` (lines 592–605)
* **Mechanism**: When a shortcut triggers a multi-level jump (e.g., from $L_3$ or $L_2$ directly to $L_0$), the search runtime incurs a 76-cycle setup overhead initializing the `VisitedListPool` tag version and dual priority queue heaps (`candidate_set` and `top_candidates`).
* **Implementation**: We issue a software prefetch instruction (`_mm_prefetch(..., _MM_HINT_T0)`) on the Level-0 candidate vector and its neighbor linklist immediately before the setup routine. The 76-cycle setup dead time is thus converted into useful memory fetch cycles.

```cpp
// In hnswlib/heds.h:
PrefetchController::on_shortcut_jump(
    currObj, orig_level, level,
    [this](tableint id, int lvl) -> void* {
        return (lvl == 0) ? (void*)get_linklist0(id) : (void*)get_linklist(id, lvl);
    },
    [this](tableint id) -> void* { return (void*)getDataByInternalId(id); },
    data_size_,
    (level < orig_level - 1)
);
```

### Idea 2: Multi-Line Neighbor Lookahead Prefetching (`PrefetchMode::STEP2_SKIP_LOOKAHEAD`)
* **Location**: `hnswlib/heds.h` (lines 971–981 in `searchBaseLayerSTPrune`)
* **Mechanism**: Traversal in the base layer iterates through a node's neighbor list. Because AVX2 distance computation takes 91 cycles (SIFT) or 410 cycles (Enron), looking ahead by distance $k = 2$ allows the memory controller to pipeline the DRAM fetch in the background.
* **Vector Footprint Coverage**:
  - For SIFT1M ($d = 128$ floats = 512 bytes = 8 cache lines), prefetching only line 0 leaves the remaining 7 lines cold. Our controller streams prefetch requests across all 8 cache lines.
  - For Enron ($d = 1,369$ floats = 5,476 bytes = 86 cache lines), the controller streams initial cache-line blocks to prime the hardware prefetcher.
* **Cumulative Design**: Step 2 operates **cumulatively**, keeping Step 1 active so both level-skips and base-layer hops are prefetched.

```cpp
// In hnswlib/heds.h searchBaseLayerSTPrune loop:
PrefetchController::on_base_layer_neighbor_lookahead(
    data, j, size, visited_array,
    [this](tableint id) -> void* { return (void*)getDataByInternalId(id); },
    data_size_,
    2 // Lookahead distance k = 2
);
```

---

## 4. Empirical Evaluation & Key Insights

All experiments were executed on a dedicated physical core (`taskset -c 2`, Intel Core i5-13400) under paper-identical parameters ($M = 48, efConstruction = 80, K = 20$).

### 1. Default Operating Point ($efSearch = 20$)

| Dataset | Method | Latency (µs) | Recall@20 | Throughput (QPS) | Speedup vs. SHG Baseline | Speedup vs. HNSW Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Enron** ($d=1,369$) | `HNSW` | 217.5 µs | **0.8412** | 4,597.1 QPS | $1.14\times$ | $1.00\times$ (reference) |
| | `SHG (normal)` | 248.0 µs | 0.8285 | 4,031.9 QPS | $1.00\times$ (reference) | $0.88\times$ |
| | `SHG with Idea 1` | 246.8 µs | 0.8285 | 4,052.4 QPS | $1.01\times$ (+0.5%) | $0.88\times$ |
| | **`SHG + Idea 2 (Combined)`** | **214.1 µs** | 0.8285 | **4,670.4 QPS** | **$1.16\times$ (+15.8%)** | **$1.02\times$ (+1.6%)** |
| **GIST1M** ($d=960$) | `HNSW` | 295.7 µs | **0.5289** | 3,381.3 QPS | $1.20\times$ | $1.00\times$ (reference) |
| | `SHG (normal)` | 355.3 µs | 0.4189 | 2,814.4 QPS | $1.00\times$ (reference) | $0.83\times$ |
| | `SHG with Idea 1` | 357.7 µs | 0.4189 | 2,795.5 QPS | $0.99\times$ (-0.7%) | $0.83\times$ |
| | **`SHG + Idea 2 (Combined)`** | **288.2 µs** | 0.4189 | **3,470.2 QPS** | **$1.23\times$ (+23.3%)** | **$1.03\times$ (+2.6%)** |
| **SIFT1M** ($d=128$) | `HNSW` | **58.1 µs** | **0.8007** | **17,216.8 QPS** | $2.19\times$ | $1.00\times$ (reference) |
| | `SHG (normal)` | 127.2 µs | 0.7670 | 7,861.7 QPS | $1.00\times$ (reference) | $0.46\times$ |
| | `SHG with Idea 1` | 121.0 µs | 0.7670 | 8,263.4 QPS | $1.05\times$ (+5.1%) | $0.48\times$ |
| | **`SHG + Idea 2 (Combined)`** | **110.4 µs** | 0.7670 | **9,057.2 QPS** | **$1.15\times$ (+15.2%)** | **$0.53\times$** |

---

### 2. Primary Insights

#### Insight 1: Why HNSW Outperforms Baseline SHG on Recall and Throughput
* **Recall Gap (HNSW > SHG)**: In HNSW, greedy routing across every layer continuously guides the search into the true basin of attraction. When SHG shortcuts execute aggressive level jumps, slight approximation inaccuracies in compressed distance estimates cause the search to land in a suboptimal neighborhood at Level 0, missing true neighbors. Additionally, heuristic base-layer pruning (`pruneDisCompute`) prematurely drops valid candidates.
* **Throughput Gap (HNSW > Baseline SHG on SIFT)**:
  1. HNSW incorporates degree-1 prefetch intrinsics by default, while Baseline SHG lacked them.
  2. SHG allocates a 20 MB `resultsProcessing` array; querying it on every candidate hop causes severe LLC cache-line thrashing.
  3. Upper-level distance calculations in SHG use an unvectorized scalar loop, whereas HNSW leverages 256-bit AVX2 FMA instructions throughout.

#### Insight 2: High Dimensionality Unleashes Prefetching Potential
* On **SIFT1M ($d=128$)**, distance calculation is lightweight (26 ns), making software table overheads more visible. Even so, lookahead prefetching cuts memory stalls, driving a **+15.2% throughput improvement** (7,862 $\to$ 9,057 QPS).
* On **GIST1M ($d=960$)**, each vector spans 3,840 bytes. Pipelining neighbor lookaheads hides memory bus stalls during intensive AVX2 compute, driving a **+23.3% throughput surge** (2,814.4 $\to$ **3,470.2 QPS**), **surpassing HNSW (3,381.3 QPS)**!
* On **Enron ($d=1,369$)**, distance calculation takes 410 cycles. By hiding memory latency behind 410 cycles of AVX2 computation, **SHG + Idea 2 achieves 214.1 µs, successfully outperforming HNSW (217.5 µs)**.

#### Insight 3: Tail Latency Compression (P90 & P99)
Because the slowest queries in a workload perform the greatest number of hops, they suffer disproportionately from memory stalls. On Enron:
* Baseline SHG P90: **438.0 µs** $\longrightarrow$ Step 2 P90: **374.0 µs** ($-14.6\%$).
* Baseline SHG P99: **502.0 µs** $\longrightarrow$ Step 2 P99: **428.0 µs** ($-14.7\%$).

---

## 5. Artifacts and Generated Visualizations

All generated visual artifacts are stored in `plots/` and mirrored in `results/plots/`:

1. **Pareto Frontier Across Author Sweep ($efSearch \in [20, 300]$)**:
   - [`plots/recall_vs_latency.png`](plots/recall_vs_latency.png): Clean side-by-side Pareto curves with 0.05 tick intervals on Recall@20.
2. **Throughput Comparison ($efSearch = 20$)**:
   - [`plots/throughput_comparison.png`](plots/throughput_comparison.png): Side-by-side bar chart showing QPS across all 4 methods.
3. **Speedup Charts ($efSearch = 20$)**:
   - [`plots/speedup_vs_shg.png`](plots/speedup_vs_shg.png): Speedup factor relative to SHG Baseline ($1.00\times$ dotted line).
   - [`plots/speedup_vs_hnsw.png`](plots/speedup_vs_hnsw.png): Speedup factor relative to HNSW Baseline ($1.00\times$ dotted line).
4. **Microarchitectural Motivation**:
   - [`plots/cpu_idle_time_motivation.png`](plots/cpu_idle_time_motivation.png): CPU idle cycle breakdown demonstrating the 56%–73% memory wait opportunity window.
   - [`plots/cpu_memory_stall_overlap.png`](plots/cpu_memory_stall_overlap.png): Execution timeline visualizing how lookahead prefetching overlaps DRAM round-trips.
5. **Author's 7-Subset Quantile Protocol**:
   - [`plots/author_exact_subsets_comparison.png`](plots/author_exact_subsets_comparison.png): Recall-subset curves under the paper's Section 5.3 binning method.
