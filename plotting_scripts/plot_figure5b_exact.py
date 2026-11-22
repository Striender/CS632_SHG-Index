#!/usr/bin/env python3
"""
plot_figure5b_exact.py
----------------------
Reproduces the exact Search Performance curve (Time vs Recall) from Figure 5b of
the SHG paper (Chen et al., VLDB 2024 / Screenshot_20261028_030329.png) and compares:
  1. Paper SHG curve (digitized from Figure 5b on Enron)
  2. Measured Baseline SHG (our exact measured sweep)
  3. Measured Step 2 (Skip + Lookahead Prefetching - our technique)
  4. Measured Step 1 (Skip-Only Prefetching)

Strictly uses real benchmark data from results/data/ without synthetic external estimates.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

# 1. Exact digitized coordinates from Figure 5b (Enron) in the paper / Screenshot_20261028_030329.png
# Points on the red curve with open circles:
paper_shg_time = [0.270, 0.280, 0.298, 0.302, 0.312, 0.320, 0.425]
paper_shg_recall = [0.520, 0.750, 0.840, 0.900, 0.940, 0.950, 0.970]

# Load measured ef sweep data
csv_file = 'results/data/ef_sweep_curves.csv'
if not os.path.exists(csv_file):
    print(f"Waiting for {csv_file} to be generated...")
    exit(0)

df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} rows from {csv_file}")

base_df = df[df['method'] == 'Baseline_SHG'].sort_values('recall')
step1_df = df[df['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
step2_df = df[df['method'] == 'Step2_SkipLookahead'].sort_values('recall')

# Set figure style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

# ==========================================
# Panel 1: Exact Figure 5b Reproduction & Step 2 Comparison
# ==========================================
ax1 = axes[0, 0]

# Paper SHG curve
ax1.plot(paper_shg_time, paper_shg_recall, 'o-.', color='#d62728', markerfacecolor='none', 
         markeredgewidth=2, markersize=8, linewidth=1.8, label='Paper SHG (Fig. 5b, Chen et al.)')

# Measured Baseline SHG
ax1.plot(base_df['latency_ms'], base_df['recall'], 's-', color='#333333', markerfacecolor='#555555',
         markersize=6, linewidth=2.0, label='Measured Baseline SHG (Our Machine)')

# Measured Step 1
ax1.plot(step1_df['latency_ms'], step1_df['recall'], '^--', color='#1f77b4', markerfacecolor='#1f77b4',
         markersize=6, linewidth=1.8, label='Step 1: Skip Prefetching')

# Measured Step 2 (Our Technique)
ax1.plot(step2_df['latency_ms'], step2_df['recall'], 'D-', color='#2ca02c', markerfacecolor='#2ca02c',
         markersize=7, linewidth=2.5, label='Step 2: Skip + Lookahead Prefetching (Ours)')

ax1.set_xlabel('Search Time (ms)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recall@20', fontsize=12, fontweight='bold')
ax1.set_title('(a) Search Performance Curve on Enron (Figure 5b Reproduction)', fontsize=13, fontweight='bold')
ax1.set_xlim(0.15, 0.55)
ax1.set_ylim(0.40, 1.00)
ax1.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# Panel 2: Latency Speedup (%) vs Target Recall
# ==========================================
ax2 = axes[0, 1]

# Interpolate latency at common recall points to calculate exact speedup
common_recalls = np.linspace(0.70, 0.96, 20)
base_time_interp = np.interp(common_recalls, base_df['recall'], base_df['latency_ms'])
s2_time_interp = np.interp(common_recalls, step2_df['recall'], step2_df['latency_ms'])
speedup_pct = (base_time_interp - s2_time_interp) / base_time_interp * 100.0

ax2.plot(common_recalls, speedup_pct, 'o-', color='#2ca02c', linewidth=2.5, markersize=6)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax2.fill_between(common_recalls, 0, speedup_pct, color='#2ca02c', alpha=0.15)
ax2.set_xlabel('Target Recall@20', fontsize=12, fontweight='bold')
ax2.set_ylabel('Latency Reduction vs Baseline (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Speedup of Our Technique across Recall Levels', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 30)
ax2.grid(True, linestyle='--', alpha=0.6)

# Annotate key speedup points
mid_idx = len(common_recalls) // 2
ax2.annotate(f'+{speedup_pct[mid_idx]:.1f}% speedup\n@ Recall {common_recalls[mid_idx]:.2f}',
             xy=(common_recalls[mid_idx], speedup_pct[mid_idx]),
             xytext=(common_recalls[mid_idx]-0.08, speedup_pct[mid_idx]+5),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='#2ca02c'),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#2ca02c'))

# ==========================================
# Panel 3: Throughput (QPS) vs Recall
# ==========================================
ax3 = axes[1, 0]

ax3.plot(base_df['recall'], base_df['qps'], 's-', color='#333333', label='Measured Baseline SHG', linewidth=2.0, markersize=6)
ax3.plot(step1_df['recall'], step1_df['qps'], '^--', color='#1f77b4', label='Step 1: Skip Prefetch', linewidth=1.8, markersize=6)
ax3.plot(step2_df['recall'], step2_df['qps'], 'D-', color='#2ca02c', label='Step 2: Skip + Lookahead (Ours)', linewidth=2.5, markersize=7)

ax3.set_xlabel('Recall@20', fontsize=12, fontweight='bold')
ax3.set_ylabel('Throughput (QPS)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Throughput (QPS) vs Recall', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# Panel 4: Ordered Recall Subsets (Section 5.3 Method)
# ==========================================
ax4 = axes[1, 1]

# Load per-query baseline and step2 results to plot the ordered recall subsets
q_base_file = 'results/data/enron_heds_K20_baseline_queries.txt'
q_s2_file = 'results/data/enron_heds_K20_step2_skip_lookahead_prefetch_queries.txt'

if os.path.exists(q_base_file) and os.path.exists(q_s2_file):
    qb = pd.read_csv(q_base_file, sep='\t', comment='#', names=['qid', 'time_sec', 'time_us', 'recall']).sort_values('recall').reset_index(drop=True)
    qs2 = pd.read_csv(q_s2_file, sep='\t', comment='#', names=['qid', 'time_sec', 'time_us', 'recall']).sort_values('recall').reset_index(drop=True)

    n_bins = 7
    qb_splits = np.array_split(qb, n_bins)
    qs2_splits = np.array_split(qs2, n_bins)

    qb_times = [b['time_sec'].mean() * 1000 for b in qb_splits]
    qb_recalls = [b['recall'].mean() for b in qb_splits]

    qs2_times = [b['time_sec'].mean() * 1000 for b in qs2_splits]
    qs2_recalls = [b['recall'].mean() for b in qs2_splits]

    ax4.plot(qb_times, qb_recalls, 's-', color='#333333', linewidth=2.0, markersize=6, label='Baseline (7 Query Subsets)')
    ax4.plot(qs2_times, qs2_recalls, 'D-', color='#2ca02c', linewidth=2.5, markersize=7, label='Step 2 (7 Query Subsets)')

    ax4.set_xlabel('Search Time (ms)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Recall@20', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Query Recall Subsets (Sec. 5.3 Partitioning Protocol)', fontsize=13, fontweight='bold')
    ax4.set_xlim(0.12, 0.45)
    ax4.set_ylim(0.35, 1.00)
    ax4.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
out_plot = 'results/plots/figure5b_exact_reproduction.png'
plt.savefig(out_plot, bbox_inches='tight', dpi=300)
print(f"Saved exact reproduction multi-panel plot to: {out_plot}")
