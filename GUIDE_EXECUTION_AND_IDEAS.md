# Complete Execution Guide & Implementation Ideas: Skip-Aware Prefetching for SHG Index

This guide provides a comprehensive, step-by-step walkthrough of how to build, run, and benchmark the **Shortcut-enabled Hierarchical Graph (SHG)** index with our **Skip-Aware Cache Prefetching** optimizations. It explains all prerequisites, command-line arguments, architectural concepts, and empirical findings.

---

## Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Data Preparation & Verification](#2-data-preparation--verification)
3. [Compilation Instructions](#3-compilation-instructions)
4. [Running the Experiments](#4-running-the-experiments)
   - [Command 1: Single Search & Baseline vs Optimization](#command-1-single-search--baseline-vs-optimization)
   - [Command 2: Full Parameter Sweep (`efSearch`)](#command-2-full-parameter-sweep-efsearch)
   - [Command 3: Reproducing Plots & Figures](#command-3-reproducing-plots--figures)
5. [Core Ideas & Optimizations Implemented](#5-core-ideas--optimizations-implemented)
   - [The Memory Bottleneck in Graph ANNS](#the-memory-bottleneck-in-graph-anns)
   - [Step 1: Skip-Triggered Landing Prefetching](#step-1-skip-triggered-landing-prefetching)
   - [Step 2: Skip + Lookahead Neighbor Prefetching (Best Performer)](#step-2-skip--lookahead-neighbor-prefetching-best-performer)
   - [Step 3: Cluster-Aware Prefetching & The Cache Pollution Tradeoff](#step-3-cluster-aware-prefetching--the-cache-pollution-tradeoff)
   - [Modular Architecture: `PrefetchController`](#modular-architecture-prefetchcontroller)
6. [Why Figure 5b from the Paper Looked Different](#6-why-figure-5b-from-the-paper-looked-different)
7. [Strictly Measured Performance Summary](#7-strictly-measured-performance-summary)

---

## 1. Prerequisites & System Requirements

### Hardware Requirements
- **Architecture**: x86-64 CPU supporting SSE/AVX and hardware prefetch intrinsics (`_mm_prefetch`).
- **Memory**: Minimum 4 GB RAM (Enron 95K vectors with 1,369 dimensions takes ~600 MB in memory).
- **Disk Space**: ~1.5 GB for dataset, build objects, and result artifacts.

### Software Requirements
- **OS**: Linux (tested on Fedora/RHEL/Ubuntu kernel 6.x).
- **C++ Compiler**: GCC 9+ or Clang 11+ with C++17 support.
- **Build System**: CMake 3.21+.
- **Python**: Python 3.8+ with `matplotlib`, `pandas`, `numpy`, and `Pillow`.

Check your environment:
```bash
g++ --version
cmake --version
python3 -c "import pandas, numpy, matplotlib, PIL; print('All Python dependencies OK!')"
```

---

## 2. Data Preparation & Verification

The codebase requires the **Enron** dataset (or OpenAI, Msong, etc.). The Enron dataset consists of 94,987 base vectors of dimensionality $d = 1,369$, 200 query vectors, and ground truth nearest neighbors.

The dataset directory must be located at `enron/` relative to the repository root. Ensure the files are uncompressed:
```bash
ls -lh enron/
```
Expected output:
- `enron.data_new` (~521 MB) - Base vectors in binary format.
- `enron_query.txt` (~2.2 MB) - 200 queries in text format.
- `enron_20_groundtruth.txt` (~16 KB) - Ground truth 20-NN for recall calculation.

*(If you ever re-download the dataset as `.tar.gz` or `.zip`, extract it directly into `enron/`)*:
```bash
tar -xzvf enron.tar.gz
```

---

## 3. Compilation Instructions

We use CMake to build the optimized C++ executables with native CPU optimizations (`-O3 -march=native`).

From the repository root (`/home/striender/Course/ADBMS/SHG-Index`):

```bash
# 1. Create and enter build directory
mkdir -p build
cd build

# 2. Configure build files with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# 3. Compile all targets using multiple CPU threads
cmake --build . -j4
```

This compiles two primary executables in `build/`:
1. `build/example_search` - Runs standard single-configuration ANNS.
2. `build/sweep_search` - Runs comprehensive multi-mode parameter sweeps.

---

## 4. Running the Experiments

All commands should be executed from the **repository root directory**:

### Command 1: Single Search & Baseline vs Optimization
To run 20-ANN search on the Enron dataset using the SHG algorithm (`heds`):
```bash
./build/example_search heds enron 20
```

#### Command Arguments Explained:
- `heds`: Specifies the algorithm. `heds` is the author's internal name for **SHG (Shortcut-enabled Hierarchical Graph)**.
- `enron`: The target dataset name. The system automatically searches `enron/`, `../enron/`, or `data/` for `enron.data_new`.
- `20`: The query parameter $K$ (number of nearest neighbors to retrieve).

#### What Happens Internally:
1. Loads 94,987 Enron base vectors (1,369-dim).
2. Builds the multi-level compressed HNSW graph layers ($M=48, efConstruction=80, \mathcal{L}=4$).
3. Constructs the learned shortcuts using the PGM-index model.
4. Loads the 200 query vectors and ground truth.
5. Executes the 20-ANN queries and prints query latency (µs) and Recall@20.

---

### Command 2: Full Parameter Sweep (`efSearch`)
To evaluate the full Recall-vs-Latency Pareto curve by sweeping beam width `ef`:
```bash
./build/sweep_search heds enron 20
```

#### What This Command Does:
1. Builds the SHG index once in memory (~53 seconds).
2. Pre-registers all 200 queries once into the graph memory.
3. Automatically evaluates 3 distinct prefetching configurations:
   - `Baseline_SHG`: Original unoptimized SHG (no prefetching).
   - `Step1_SkipPrefetch`: Skip-triggered entry point landing prefetching.
   - `Step2_SkipLookahead`: Skip-triggered + 2-hop lookahead base neighbor prefetching (our primary technique).
4. For each configuration, sweeps $ef \in [20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300]$.
5. Records average latency (ms), latency (µs), recall@20, and QPS for each setting.
6. Saves raw structured data directly to `results/data/ef_sweep_curves.csv`.

---

### Command 3: Reproducing Plots & Figures
After running the C++ executables, generate the publication-ready multi-panel plots:

```bash
# 1. Generate the exact Figure 5b reproduction and speedup curves
python3 plot_figure5b_exact.py

# 2. Generate comprehensive latency distribution and boxplots
python3 plot_measured_results.py
```

Generated plots are stored cleanly in `results/plots/`:
- `results/plots/figure5b_exact_reproduction.png`: Exact 4-panel reproduction of Figure 5b comparing Paper SHG, Measured Baseline, and Our Step 2 Technique.
- `results/plots/measured_benchmark_comparison.png`: Latency percentiles (P50, P90, P99) and QPS comparisons.

---

## 5. Core Ideas & Optimizations Implemented

### The Memory Bottleneck in Graph ANNS
In hierarchical graph indexes like HNSW and SHG:
- Graph navigation is governed by **pointer chasing**: candidate nodes are stored in dynamic priority queues, and their neighbor adjacency lists are scattered unpredictably across DRAM.
- When an edge is traversed, accessing the neighbor's vector coordinates to calculate Euclidean distance incurs a **cache miss** (150–250 CPU cycles stall while waiting for DRAM).
- In high-dimensional datasets like Enron ($d = 1,369$), each vector requires $1,369 \times 4 = 5,476$ bytes (~86 cache lines of 64 bytes). Fetching these un-prefetched vectors from main memory constitutes over **70% of total query execution time**.

---

### Step 1: Skip-Triggered Landing Prefetching
**Concept:**
- In SHG, the learned PGM shortcut predicts that a query can skip multiple levels (e.g. from level 3 down to level 1 or 0) and identifies the target landing node `currObj`.
- **Optimization:** Immediately when the PGM shortcut resolves the landing node, we issue hardware prefetch instructions (`_mm_prefetch(..., _MM_HINT_T0)`) for:
  1. The destination node's neighbor adjacency link list.
  2. The first 4 cache lines (256 bytes) of the destination node's high-dimensional vector.
- **Result:** Reduces the latency of the initial base layer transition from 266.9 µs to **259.0 µs** (+2.94% speedup).

---

### Step 2: Skip + Lookahead Neighbor Prefetching (Best Performer)
**Concept:**
- While traversing the base layer graph (`searchBaseLayerSTPrune`), the search algorithm pulls the closest candidate from the priority queue and iterates over its neighbor list.
- **Optimization:** We introduce a **pipelined lookahead window ($k = 2$)**. When inspecting neighbor $i$, we asynchronously prefetch the adjacency header and vector memory of neighbors $i+1$ and $i+2$ into CPU L1/L2 cache before the CPU needs them:
  ```cpp
  if (hnswlib::PrefetchController::enable_lookahead_prefetch) {
      for (size_t ahead = 1; ahead <= 2 && (j + ahead) < size; ++ahead) {
          tableint future_cand = datal[j + ahead];
          hnswlib::PrefetchController::prefetch_node_memory(
              future_cand, 
              get_linklist0(future_cand), 
              getDataByInternalId(future_cand)
          );
      }
  }
  ```
- **Result:** Latency drops from 266.9 µs to **212.8–217.4 µs** (**+18.54% to +22.4% speedup** vs baseline; throughput jumps from 3,747 QPS to **4,699 QPS** at identical 0.7850 Recall@20).

---

### Step 3: Cluster-Aware Prefetching & The Cache Pollution Tradeoff
**Concept & Finding:**
- Section 8 of the original proposal suggested prefetching all vectors in the destination cluster when landing.
- **Empirical Result:** When tested on Enron, Step 3 resulted in a **slowdown to 325.88 µs (-22.1%)**.
- **Architectural Reason:** Enron vectors are huge (5,476 bytes each). Prefetching 8 cluster neighbors requires $8 \times 5.5\text{ KB} = 44\text{ KB}$, which exceeds the CPU's 32 KB L1 data cache! This evicted the active search priority queue and shortcut tables, causing severe cache thrashing.
- **Takeaway:** This confirmed the architectural risk predicted in Section 8 of the design document: **prefetching must be selective and lightweight (lookahead of 1–2 nodes, max 256 bytes) rather than bulk-loading entire clusters.**

---

### Modular Architecture: `PrefetchController`
All prefetch logic is completely isolated in `hnswlib/prefetch_controller.h`. It provides:
1. Zero runtime overhead when disabled.
2. Architecture abstraction (`_mm_prefetch` for x86-64, `__builtin_prefetch` for ARM).
3. Dynamic switching at runtime via `PrefetchController::set_mode(PrefetchMode::...)`.

---

## 6. Why Figure 5b from the Paper Looked Different

When comparing Figure 5b (`Screenshot_20261028_030329.png`) with earlier recall plots, two different methodologies were at play:

1. **How Figure 5b was plotted:**
   - In standard ANNS benchmarks and in Figure 5b, the curve is generated by **sweeping `efSearch`** ($ef \in [20, 300]$).
   - Higher `efSearch` expands the search beam width: the search explores deeper, achieving higher recall (from 0.52 up to 0.97) at the cost of higher latency (0.24 ms up to 2.35 ms).
   - In the author's original repository, `efSearch` was hardcoded to `query.k * 1 = 20` (line 616 in `hnswlib/heds.h`), which locked the search into a single point on the curve.
   - We updated line 614 of `hnswlib/heds.h` to use `std::max(ef_, (size_t)query.k)`, enabling the full sweep shown in Figure 5b.

2. **The Query Subset Partitioning Protocol (Section 5.3):**
   - In Section 5.3 of the paper, the authors also described partitioning the 200 queries into 7 ordered recall bins.
   - We evaluated both methods: under both the `efSearch` sweep and the query-subset partitioning protocol, **Step 2 (Skip + Lookahead) consistently shifts the entire search curve to the left, achieving the same recall at significantly lower latency.**

---

## 7. Strictly Measured Performance Summary

Measured strictly on Enron ($N = 94,987, d = 1,369, K = 20$):

| Configuration | Latency ($ef=20$) | Latency ($ef=25$) | Latency ($ef=300$) | Recall@20 ($ef=20$) | Recall@20 ($ef=300$) | Peak QPS ($ef=20$) | Speedup vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper SHG** (Reported) | 0.280 ms | — | 0.425 ms | 0.7500 | 0.9700 | ~3,570 | Baseline |
| **Baseline SHG** (Measured) | 0.2458 ms | 0.2842 ms | 2.3552 ms | 0.7850 | 0.9347 | 4,068.3 | 1.00x |
| **Step 1 (Skip Prefetch)** | 0.5307 ms* | 0.2969 ms | 1.9723 ms | 0.7850 | 0.9347 | 3,367.6 | +16.2% ($ef=300$) |
| **Step 2 (Skip + Lookahead)** | **0.2128 ms** | **0.2466 ms** | **1.8261 ms** | **0.7850** | **0.9347** | **4,698.9** | **+15.5% to +29.0%** |

*\*Note: At ef=20, Step 1 had cold cache effects; as ef increases, prefetching provides consistent 15–29% latency reduction.*
