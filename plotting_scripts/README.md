# Unified Plotting Scripts Directory

This directory centralizes all visualization and plotting scripts for the project, ensuring a clean root directory structure.

## Primary Conference Publication Plotting Script

* **`plot_conference_results.py`**:
  * The main visualization engine for academic conference submissions (PVLDB, SIGMOD, ICDE).
  * Automatically reads benchmark results (`results_autotune_*.csv` and `results_multithread_*.csv`) and generates publication-grade PDF and PNG figures:
    1. **Figure 1**: Theoretical Microarchitectural Cost Model $D^*(d)$ curve vs. dimension $d$, and empirical lookahead sweep validation across SIFT1M, GIST1M, and Enron.
    2. **Figure 2**: Multi-Threaded Scalability (Queries Per Second vs. Threads $T \in \{1, 2, 4, 8, 16, 32, 64\}$) comparing Baseline HNSW, Stock HNSW, Pf-Filter Static, and Pf-Filter Adaptive.
    3. **Figure 3**: Speedup sustained under memory bus concurrency and tail latency (p50/p99) suppression.
  * **Usage**:
    ```bash
    python3 plotting_scripts/plot_conference_results.py bench_prune/conf_runs
    ```

---

## Historical & Diagnostic Benchmark Plots

* `plot_all_3_datasets.py`: Overall comparison across Enron, SIFT1M, and GIST1M.
* `plot_all_4_datasets.py`: 4-dataset comparison (including SIFT10M).
* `plot_figure5b_exact.py`: Exact reproduction of VLDB'24 Figure 5b.
* `plot_full_breakdown.py` / `plot_full_query.py`: End-to-end latency breakdowns.
* `plot_idle_time.py` / `compute_and_plot_cpu_idle_time.py`: Memory stall and CPU idle cycle quantification.
* `plot_pf_filter_design.py` / `plot_pf_filter_steps.py`: Architectural diagrams and flowcharts of Pf_Filter.
* `plot_problem_vs_pf_filter.py`: Direct comparison of baseline memory stalls vs. Pf_Filter prefetching.
* `plot_speedup_vs_hnsw.py` / `plot_speedup_vs_shg.py`: Speedup curves over standard baselines.

*All new plotting scripts should be created and maintained in this directory.*
