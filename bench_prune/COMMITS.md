# Commit plan for the bench_prune experiments

All paths are relative to the repo root (`SHG-Index/`).

Note on code: `bench_stall.cpp` and `hnswlib_patched/*.h` were edited in place as the
experiments went on, so they exist only in their final version. That version runs every config
below; the run script (and the `-DFULLVEC` build flag) picks which one. So the code goes in
commits 1 and 3, and the later commits add the run script, results and plots for each config.

Before committing, keep the datasets out of git (about 32 GB):

```bash
cat >> .gitignore <<'EOF'
enron/
sift/
gist/
sift10m/
sift10m.zip
bench_prune/bench_prune
bench_prune/bench_prune_dbg
bench_prune/bench_stall
bench_prune/bench_stall_full
bench_prune/check_asan/
bench_prune/check_valgrind/
EOF
git add .gitignore && git commit -m "Ignore datasets and benchmark binaries"
```

---

## Commit 1: same-graph HNSW vs SHG benchmark (pruning on/off, 3 prefetch modes)

```
bench: same-graph HNSW vs SHG benchmark with pruning and prefetch ablation

Adds bench_prune, which builds ONE graph per dataset (M=48, efConstruction=80,
k_=4) and times 9 search configs on it, so differences come only from
search code and prefetching:

  HNSW_stock, HNSW_noprefetch, HNSW_idea2
  SHG_original (base-layer pruning on, as published), SHG_stock, SHG_idea2
  SHG_noprune, SHG_noprune_stock, SHG_noprune_idea2

Prefetch modes (hnswlib_patched/prefetch_controller.h):
  BASELINE     - no software prefetch
  STOCK_HNSW   - new: hnswlib's original prefetch lines (next neighbour's
                 visited flag + vector, next candidate's link list)
  STEP2_SKIP_LOOKAHEAD - Idea 2: prefetch neighbour j+2 (visited flag +
                 first 3 cache lines) and next candidate's link list
Pruning: g_abl=0 keeps SHG's pruneDisCompute + 20-hop early break,
g_abl=2 disables both.

Method: single-threaded, pinned to core 2, rdtsc per query. efSearch in
{20,30,40,60,80,120,200,300}, K=20. Each method runs over all queries as
one block; block order is reshuffled in each of P=5 repetitions; per-query
median, then mean over queries.

Correctness: every prefetch variant's result IDs are compared with the same
search without prefetch -> 0 mismatches on all datasets (correctness.csv).
Valgrind clean on a 5k-vector SIFT slice.

Results (bench_prune/out_<dataset>/), efSearch=20, speed vs HNSW_stock:
             Enron  GIST1M  SIFT1M  SIFT10M
  SHG_orig   0.89   0.93    0.53    0.57
  SHG_idea2  1.00   1.16    0.61    0.69
  HNSW_idea2 1.02   1.01    1.04    1.02
Base-layer pruning lowers recall on every dataset (GIST < 0.56 even at
efSearch=300); without it SHG matches HNSW recall.
```

Files:
```bash
git add bench_prune/bench_prune.cpp bench_prune/run_all.sh bench_prune/hnswlib_patched/ \
        bench_prune/out_enron/ bench_prune/out_gist/ bench_prune/out_sift/ bench_prune/out_sift10m/
```
Look at:
- `bench_prune/bench_prune.cpp`: method list (line 35), timing loop (lines 56-64)
- `bench_prune/hnswlib_patched/prefetch_controller.h`: `STOCK_HNSW` mode, `set_mode`
- `bench_prune/hnswlib_patched/heds.h`: stock prefetch lines in `searchBaseLayerST` and `searchBaseLayerSTPrune`
- `bench_prune/out_<ds>/prune.csv`: recall and latency per method and efSearch
- `bench_prune/out_<ds>/correctness.csv`: must be all 0 in the last column

---

## Commit 2: plots for commit 1

```
plots: HNSW vs SHG, pruning effect and prefetch ablation (new_plot/1-4)

plot_new.py reads bench_prune/out_<dataset>/prune.csv and writes 2x2
figures (one panel per dataset) to new_plot/:
  1_baseline_hnsw_vs_shg.png          recall vs latency, HNSW vs SHG
  2_pruning_recall_vs_latency.png     effect of SHG base-layer pruning
  2b_pruning_speedup_vs_hnsw.png      speedup vs HNSW, pruning on/off
  3_prefetch_speedup_vs_shg_original.png  stock vs Idea 2 on SHG
  3b_prefetch_on_hnsw.png             same prefetch policies on HNSW
  4_all_speedup_vs_hnsw.png, 4b_all_recall_vs_latency.png
  summary_ef20.csv
Speedup = mean per-query latency of the reference / that of the method,
at the same efSearch.
```

Files:
```bash
git add plot_new.py new_plot/1_*.png new_plot/2*.png new_plot/3*.png new_plot/4*.png new_plot/summary_ef20.csv
```
Look at: `plot_new.py` (`speedup()` shows how speedup is computed), `new_plot/summary_ef20.csv`

---

## Commit 3: CPU time breakdown, efSearch 20 and 100 (motivation)

```
bench: measure memory stall / compute / overhead per distance evaluation

Adds bench_stall, which replaces the hard-coded numbers of the old
motivation plot (3.5 GHz, 258-cycle DRAM, assumed streaming stalls) with
measurements on the same graphs as bench_prune:
  core clock      dependent-add asm loop
  DRAM latency    pointer chase over 1 GiB
  compute         real L2 kernel on cache-resident vectors
  per query       base layer only, cold run then immediate warm rerun
  stall    = cold - warm       (DRAM latency the core could not hide)
  overhead = warm - compute    (bookkeeping + L2/L3 hits)
All reported per base-layer distance evaluation (average over all
evaluations of all queries). Methods: SHG original/stock/Idea 2,
HNSW no-prefetch/stock/Idea 2; efSearch in {20,100}; 3 repetitions.

Result (SHG as published, efSearch=20): stall is 22-42% of base-layer
time, not the 73% the old plot assumed; SIFT compute is 24 cycles, not 91.
Overhead is the largest part, and SHG's is about 2x HNSW's on SIFT.
```

Files:
```bash
git add bench_prune/bench_stall.cpp bench_prune/run_stall.sh \
        bench_prune/stall_enron/ bench_prune/stall_gist/ bench_prune/stall_sift/ bench_prune/stall_sift10m/
```
Look at:
- `bench_prune/bench_stall.cpp`: measurement steps 1-4 (clock, DRAM, compute, cold/warm)
- `bench_prune/stall_<ds>/stall.csv`: one row per method and efSearch
- `bench_prune/stall_<ds>/run.log`: core GHz, DRAM latency, compute cycles

---

## Commit 4: Idea 2 lookahead sweep L=2..5, efSearch 20/64/100

```
bench: sweep Idea 2 lookahead distance (L=2..5) and add efSearch 64

Idea 2 hard-coded "prefetch neighbour j+2". Makes the distance a setting
(PrefetchController::lookahead_dist, used by both base-layer search loops
in heds.h) and runs L=2,3,4,5 on SHG and HNSW, at efSearch 20, 64 and 100
(efConstruction stays 80). Adds a per-query result-ID check against the
no-prefetch run to bench_stall.

Results (bench_prune/stall2_<dataset>/), 0 correctness mismatches.
Base-layer speedup vs no prefetch, efSearch=64:
            stock   L=2   L=3   L=4   L=5
  HNSW SIFT1M  1.15  1.23  1.26  1.22  1.21
  HNSW SIFT10M 1.18  1.24  1.27  1.32  1.29
  SHG  SIFT1M  1.13  1.12  1.16  1.11  1.12
  SHG  GIST1M  1.22  1.21  1.19  1.19  1.18
- L=2-3 is best; L=5 slightly worse everywhere.
- Idea 2 beats stock by 7-12% on HNSW/SIFT; on SHG it is within +-3%
  of stock on every dataset.
- Exposed stall is 13-20% of SHG base-layer time at efSearch=64.
```

Files:
```bash
git add bench_prune/run_stall2.sh \
        bench_prune/stall2_enron/ bench_prune/stall2_gist/ bench_prune/stall2_sift/ bench_prune/stall2_sift10m/
```
Look at:
- `bench_prune/hnswlib_patched/prefetch_controller.h`: `lookahead_dist`, `on_base_layer_neighbor_lookahead`
- `bench_prune/bench_stall.cpp`: method list (the `for(int L=2;L<=5;L++)` loop in the `#else` branch)
- `bench_prune/stall2_<ds>/stall.csv`, `bench_prune/stall2_<ds>/correctness.csv`

---

## Commit 5: whole-vector prefetch (run on the server)

```
bench: test prefetching the whole neighbour vector instead of 3 lines

Idea 2 prefetches only the first 3 cache lines of a neighbour's vector
(3 of 8 for SIFT, 3 of 60 for GIST, 3 of 86 for Enron). Adds
PrefetchController::lookahead_lines and a -DFULLVEC build of bench_stall
that prefetches every line of the vector at lookahead L=1,2,3, next to
stock and Idea 2 (L=2, 3 lines). efSearch 20/64/100. Run on the server
(HPCA), so compare only within this run.

Results (bench_prune/stall3_<dataset>/), 0 correctness mismatches.
Base-layer speedup vs no prefetch, efSearch=64:
             stock  Idea2  full L1  full L2  full L3
  SHG  Enron  1.05   1.05   0.82     0.81     0.80
  SHG  GIST   1.11   1.10   0.90     0.88     0.86
  SHG  SIFT   1.13   1.09   1.13     1.12     1.11
  SHG  SIFT10M 1.18  1.17   1.18     1.16     1.16
Whole-vector prefetch is 10-20% SLOWER than no prefetch on Enron and GIST
(60-86 prefetches per neighbour fill the core's miss buffers, fetch
already-visited neighbours and evict useful lines). On SIFT it only
matches stock. The 3-line limit is not what holds Idea 2 back.
```

Files:
```bash
git add bench_prune/run_stall3.sh bench_prune/run_stall3.out \
        bench_prune/stall3_enron/ bench_prune/stall3_gist/ bench_prune/stall3_sift/ bench_prune/stall3_sift10m/
```
Look at:
- `bench_prune/hnswlib_patched/prefetch_controller.h`: `lookahead_lines`
- `bench_prune/bench_stall.cpp`: the `#ifdef FULLVEC` block
- build: `g++ -Ofast -march=native -std=c++17 -w -I. -I../pgm -DFULLVEC bench_stall.cpp -o bench_stall_full`
- `bench_prune/stall3_<ds>/stall.csv`, `bench_prune/stall3_<ds>/correctness.csv`

---

## Commit 6: motivation plots (measured) and lookahead plot

```
plots: measured motivation figures and Idea 2 lookahead sweep (new_plot/5-7)

plot_motivation.py reads bench_prune/stall2_<dataset>/stall.csv and
replaces the old hard-coded plots/cpu_idle_time_motivation.png and
plots/cpu_memory_stall_overlap.png:
  5_cpu_time_breakdown_ef{20,64,100}.png   stall/compute/overhead bars
  6_memory_stall_timeline_ef{20,64,100}.png 3 evaluations, SHG no/stock/Idea 2
  7_lookahead_sweep.png                    speedup vs L, stock marked
  cpu_time_breakdown_ef*.csv, lookahead_sweep.csv
Bars are cycles per base-layer distance evaluation; multiply by
base_evals_per_q for per-query time.
```

Files:
```bash
git add plot_motivation.py new_plot/5_*.png new_plot/6_*.png new_plot/7_*.png \
        new_plot/cpu_time_breakdown_ef*.csv new_plot/lookahead_sweep.csv bench_prune/COMMITS.md
```
Look at: `plot_motivation.py` (Figure 5 / 6 / 7 sections), `new_plot/cpu_time_breakdown_ef64.csv`
