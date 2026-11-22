#!/usr/bin/env python3
"""
plot_sift_and_enron_comparison.py
---------------------------------
Comprehensive multi-dataset benchmark comparison for SHG with Skip-Aware Prefetching.
Compares:
  - Enron Benchmark (High-Dimensional: d=1,369, N=95K)
  - SIFT Benchmark (Large-Scale: d=128, N=1,000,000)

Evaluates:
  1. Recall-vs-Latency Pareto Frontier (Figure 5 reproduction)
  2. Latency Reduction (% Speedup) of Step 2 across target recall levels
  3. Throughput (QPS) Scaling
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

# File paths
enron_csv = 'results/data/ef_sweep_curves_enron.csv'
if not os.path.exists(enron_csv) and os.path.exists('results/data/ef_sweep_curves.csv'):
    enron_csv = 'results/data/ef_sweep_curves.csv'

sift_csv = 'results/data/ef_sweep_curves_sift.csv'

# Paper digitized curve for Enron
paper_enron_time = [0.270, 0.280, 0.298, 0.302, 0.312, 0.320, 0.425]
paper_enron_recall = [0.520, 0.750, 0.840, 0.900, 0.940, 0.950, 0.970]

# Check available datasets
has_enron = os.path.exists(enron_csv)
has_sift = os.path.exists(sift_csv)

print(f"Data status: Enron={has_enron}, SIFT={has_sift}")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, axes = plt.subplots(3, 2, figsize=(16, 15), dpi=300)
fig.suptitle("Cross-Dataset Evaluation of Skip-Aware Cache Prefetching on SHG Index\nEnron (d = 1,369, N = 94,987) vs. SIFT (d = 128, N = 1,000,000) - Single-Threaded (Core 2 Pinned)",
             fontsize=14, fontweight='bold', y=0.99)

# -------------------------------------------------------------
# COLUMN 1: ENRON DATASET
# -------------------------------------------------------------
if has_enron:
    df_e = pd.read_csv(enron_csv)
    b_e = df_e[df_e['method'] == 'Baseline_SHG'].sort_values('recall')
    s1_e = df_e[df_e['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
    s2_e = df_e[df_e['method'] == 'Step2_SkipLookahead'].sort_values('recall')

    # Panel (1, 1): Enron Recall vs Latency
    ax1 = axes[0, 0]
    ax1.plot(paper_enron_time, paper_enron_recall, 'o-.', color='#d62728', markerfacecolor='none', 
             markeredgewidth=2, markersize=7, linewidth=1.8, label='Paper SHG (Fig. 5b, Chen et al.)')
    ax1.plot(b_e['latency_ms'], b_e['recall'], 's-', color='#333333', markerfacecolor='#555555',
             markersize=6, linewidth=2.0, label='Measured Baseline SHG')
    ax1.plot(s1_e['latency_ms'], s1_e['recall'], '^--', color='#1f77b4', markersize=6, linewidth=1.8, label='Step 1 (Skip Prefetch)')
    ax1.plot(s2_e['latency_ms'], s2_e['recall'], 'D-', color='#2ca02c', markersize=7, linewidth=2.5, label='Step 2 (Skip + Lookahead)')
    ax1.set_xlabel('Search Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Recall@20', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Enron: Search Performance Curve (Time vs Recall)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.15, 0.55)
    ax1.set_ylim(0.40, 1.00)
    ax1.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel (2, 1): Enron Speedup vs Target Recall
    ax2 = axes[1, 0]
    rec_grid_e = np.linspace(0.78, 0.93, 25)
    tb_interp = np.interp(rec_grid_e, b_e['recall'], b_e['latency_ms'])
    ts2_interp = np.interp(rec_grid_e, s2_e['recall'], s2_e['latency_ms'])
    sp_pct_e = (tb_interp - ts2_interp) / tb_interp * 100.0
    ax2.plot(rec_grid_e, sp_pct_e, 'o-', color='#2ca02c', linewidth=2.3, markersize=5)
    ax2.fill_between(rec_grid_e, 0, sp_pct_e, color='#2ca02c', alpha=0.15)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Latency Reduction vs Baseline (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(c) Enron: Step 2 Speedup across Recall Levels', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 25)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Panel (3, 1): Enron QPS vs Recall
    ax3 = axes[2, 0]
    ax3.plot(b_e['recall'], b_e['qps'], 's-', color='#333333', linewidth=2.0, markersize=6, label='Measured Baseline')
    ax3.plot(s1_e['recall'], s1_e['qps'], '^--', color='#1f77b4', linewidth=1.8, markersize=6, label='Step 1 (Skip)')
    ax3.plot(s2_e['recall'], s2_e['qps'], 'D-', color='#2ca02c', linewidth=2.5, markersize=7, label='Step 2 (Lookahead)')
    ax3.set_xlabel('Recall@20', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Throughput (Queries / sec)', fontsize=11, fontweight='bold')
    ax3.set_title('(e) Enron: Throughput (QPS) vs Recall', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax3.grid(True, linestyle='--', alpha=0.6)

# -------------------------------------------------------------
# COLUMN 2: SIFT DATASET (1M vectors)
# -------------------------------------------------------------
if has_sift:
    df_s = pd.read_csv(sift_csv)
    b_s = df_s[df_s['method'] == 'Baseline_SHG'].sort_values('recall')
    s1_s = df_s[df_s['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
    s2_s = df_s[df_s['method'] == 'Step2_SkipLookahead'].sort_values('recall')

    # Panel (1, 2): SIFT Recall vs Latency
    ax4 = axes[0, 1]
    ax4.plot(b_s['latency_ms'], b_s['recall'], 's-', color='#333333', markerfacecolor='#555555',
             markersize=6, linewidth=2.0, label='Measured Baseline SHG')
    ax4.plot(s1_s['latency_ms'], s1_s['recall'], '^--', color='#1f77b4', markersize=6, linewidth=1.8, label='Step 1 (Skip Prefetch)')
    ax4.plot(s2_s['latency_ms'], s2_s['recall'], 'D-', color='#2ca02c', markersize=7, linewidth=2.5, label='Step 2 (Skip + Lookahead)')
    ax4.set_xlabel('Search Time (ms)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Recall@20', fontsize=11, fontweight='bold')
    ax4.set_title('(b) SIFT1M: Search Performance Curve (Time vs Recall)', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax4.grid(True, linestyle='--', alpha=0.6)

    # Panel (2, 2): SIFT Speedup vs Target Recall
    ax5 = axes[1, 1]
    min_rec = max(b_s['recall'].min(), s2_s['recall'].min())
    max_rec = min(b_s['recall'].max(), s2_s['recall'].max())
    rec_grid_s = np.linspace(min_rec, max_rec, 25)
    tb_interp_s = np.interp(rec_grid_s, b_s['recall'], b_s['latency_ms'])
    ts2_interp_s = np.interp(rec_grid_s, s2_s['recall'], s2_s['latency_ms'])
    sp_pct_s = (tb_interp_s - ts2_interp_s) / tb_interp_s * 100.0
    ax5.plot(rec_grid_s, sp_pct_s, 'o-', color='#2ca02c', linewidth=2.3, markersize=5)
    ax5.fill_between(rec_grid_s, 0, sp_pct_s, color='#2ca02c', alpha=0.15)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Latency Reduction vs Baseline (%)', fontsize=11, fontweight='bold')
    ax5.set_title('(d) SIFT1M: Step 2 Speedup across Recall Levels', fontsize=12, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.6)

    # Panel (3, 2): SIFT QPS vs Recall
    ax6 = axes[2, 1]
    ax6.plot(b_s['recall'], b_s['qps'], 's-', color='#333333', linewidth=2.0, markersize=6, label='Measured Baseline')
    ax6.plot(s1_s['recall'], s1_s['qps'], '^--', color='#1f77b4', linewidth=1.8, markersize=6, label='Step 1 (Skip)')
    ax6.plot(s2_s['recall'], s2_s['qps'], 'D-', color='#2ca02c', linewidth=2.5, markersize=7, label='Step 2 (Lookahead)')
    ax6.set_xlabel('Recall@20', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Throughput (Queries / sec)', fontsize=11, fontweight='bold')
    ax6.set_title('(f) SIFT1M: Throughput (QPS) vs Recall', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax6.grid(True, linestyle='--', alpha=0.6)
else:
    axes[0, 1].text(0.5, 0.5, 'SIFT Benchmark In Progress...', ha='center', va='center', fontsize=14)

plt.tight_layout()
out_plot = 'results/plots/cross_dataset_enron_sift_comparison.png'
plt.savefig(out_plot, bbox_inches='tight', dpi=300)
print(f"Saved cross-dataset comparison plot to: {out_plot}")
