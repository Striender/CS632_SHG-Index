#!/usr/bin/env python3
"""
plot_hnsw_vs_shg_validation.py
------------------------------
Rigorous empirical validation of the SHG paper (Chen et al., VLDB 2024):
Compares:
  1. Paper Reported Curves (HNSW & SHG from Figure 5b)
  2. Measured Vanilla HNSW (M=48, efConstruction=80, K=20)
  3. Measured Baseline SHG (M=48, efConstruction=80, K=20)
  4. Measured SHG + Step 2 (Skip + Lookahead Prefetching - Our Technique)

Directly verifies whether:
  - The codebase implementation of SHG and HNSW matches the paper's behavior.
  - SHG genuinely outperforms Vanilla HNSW under the exact parameters from the paper.
  - Our Step 2 prefetching extends the Pareto frontier even further beyond SHG.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

csv_path = 'results/data/hnsw_vs_shg_comparison.csv'
if not os.path.exists(csv_path):
    print(f"Waiting for {csv_path}...")
    exit(0)

df = pd.read_csv(csv_path)

# Digitized coordinates from Figure 5b in the paper (Chen et al., VLDB 2024)
paper_shg_time = [0.270, 0.280, 0.298, 0.302, 0.312, 0.320, 0.425]
paper_shg_recall = [0.520, 0.750, 0.840, 0.900, 0.940, 0.950, 0.970]

paper_hnsw_time = [0.325, 0.360, 0.390, 0.440, 0.490, 0.505]
paper_hnsw_recall = [0.430, 0.670, 0.800, 0.870, 0.910, 0.950]

hnsw_df = df[df['method'] == 'Vanilla_HNSW'].sort_values('recall')
shg_df = df[df['method'] == 'Baseline_SHG'].sort_values('recall')
s2_df = df[df['method'] == 'Step2_SkipLookahead'].sort_values('recall')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
fig.suptitle("Validation of Paper Claims: Vanilla HNSW vs. SHG vs. Step 2 (Prefetching)\nDataset: Enron (N = 94,987, d = 1,369, K = 20, M = 48, efConstruction = 80) - Single-Threaded Core 2",
             fontsize=13, fontweight='bold', y=0.98)

# =========================================================================
# Panel 1: Search Performance (Time vs Recall) - Complete Validation
# =========================================================================
ax1 = axes[0, 0]

# Paper reported curves (dashed with open markers)
ax1.plot(paper_hnsw_time, paper_hnsw_recall, 'd--', color='#7f8c8d', markerfacecolor='none',
         markeredgewidth=2, markersize=8, linewidth=1.8, label='Paper HNSW (Fig. 5b, Chen et al.)')
ax1.plot(paper_shg_time, paper_shg_recall, 'o-.', color='#d62728', markerfacecolor='none',
         markeredgewidth=2, markersize=8, linewidth=1.8, label='Paper SHG (Fig. 5b, Chen et al.)')

# Measured curves (solid lines with filled markers)
ax1.plot(hnsw_df['latency_ms'], hnsw_df['recall'], 'd-', color='#2c3e50', markerfacecolor='#2c3e50',
         markersize=7, linewidth=2.0, label='Measured Vanilla HNSW (Local Run)')
ax1.plot(shg_df['latency_ms'], shg_df['recall'], 's-', color='#e67e22', markerfacecolor='#e67e22',
         markersize=7, linewidth=2.2, label='Measured Baseline SHG (Local Run)')
ax1.plot(s2_df['latency_ms'], s2_df['recall'], 'D-', color='#27ae60', markerfacecolor='#27ae60',
         markersize=8, linewidth=2.6, label='Measured Step 2: SHG + Prefetching (Ours)')

ax1.set_xlabel('Search Time (ms)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Recall@20', fontsize=11, fontweight='bold')
ax1.set_title('(a) Search Performance Pareto Frontier (Time vs. Recall)', fontsize=12, fontweight='bold')
ax1.set_xlim(0.18, 0.65)
ax1.set_ylim(0.40, 1.00)
ax1.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9.5)
ax1.grid(True, linestyle='--', alpha=0.6)

# =========================================================================
# Panel 2: Speedup relative to Vanilla HNSW (%)
# =========================================================================
ax2 = axes[0, 1]

# Calculate speedup over Vanilla HNSW at common recall levels
min_r = max(hnsw_df['recall'].min(), shg_df['recall'].min(), s2_df['recall'].min())
max_r = min(hnsw_df['recall'].max(), shg_df['recall'].max(), s2_df['recall'].max())
rec_grid = np.linspace(min_r, max_r, 25)

t_hnsw_interp = np.interp(rec_grid, hnsw_df['recall'], hnsw_df['latency_ms'])
t_shg_interp = np.interp(rec_grid, shg_df['recall'], shg_df['latency_ms'])
t_s2_interp = np.interp(rec_grid, s2_df['recall'], s2_df['latency_ms'])

shg_speedup_vs_hnsw = (t_hnsw_interp - t_shg_interp) / t_hnsw_interp * 100.0
s2_speedup_vs_hnsw = (t_hnsw_interp - t_s2_interp) / t_hnsw_interp * 100.0
s2_speedup_vs_shg = (t_shg_interp - t_s2_interp) / t_shg_interp * 100.0

ax2.plot(rec_grid, shg_speedup_vs_hnsw, 's-', color='#e67e22', linewidth=2.2, markersize=5, label='SHG Speedup vs. Vanilla HNSW (Paper Claim)')
ax2.plot(rec_grid, s2_speedup_vs_hnsw, 'D-', color='#27ae60', linewidth=2.5, markersize=6, label='Step 2 Speedup vs. Vanilla HNSW (Ours)')
ax2.plot(rec_grid, s2_speedup_vs_shg, '^:', color='#2980b9', linewidth=1.8, markersize=5, label='Step 2 Speedup vs. Baseline SHG')

ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax2.fill_between(rec_grid, 0, s2_speedup_vs_hnsw, color='#27ae60', alpha=0.10)
ax2.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
ax2.set_ylabel('Speedup / Latency Reduction (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Speedup Relative to Vanilla HNSW across Recall Tiers', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9.5)
ax2.grid(True, linestyle='--', alpha=0.6)

# =========================================================================
# Panel 3: Throughput (QPS) Comparison
# =========================================================================
ax3 = axes[1, 0]

ax3.plot(hnsw_df['recall'], hnsw_df['qps'], 'd-', color='#2c3e50', linewidth=2.0, markersize=6, label='Vanilla HNSW')
ax3.plot(shg_df['recall'], shg_df['qps'], 's-', color='#e67e22', linewidth=2.2, markersize=6, label='Baseline SHG')
ax3.plot(s2_df['recall'], s2_df['qps'], 'D-', color='#27ae60', linewidth=2.5, markersize=7, label='Step 2: SHG + Prefetching (Ours)')

ax3.set_xlabel('Recall@20', fontsize=11, fontweight='bold')
ax3.set_ylabel('Throughput (Queries / sec)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Search Throughput (QPS) vs. Recall', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.6)

# =========================================================================
# Panel 4: Index Construction Time & Memory Footprint
# =========================================================================
ax4 = axes[1, 1]

methods = ['Vanilla HNSW', 'SHG (Baseline)', 'SHG + Step 2']
build_times = [
    hnsw_df['build_time_s'].iloc[0],
    shg_df['build_time_s'].iloc[0],
    shg_df['build_time_s'].iloc[0] # Step 2 uses the exact same index
]
memories = [
    hnsw_df['memory_mb'].iloc[0],
    shg_df['memory_mb'].iloc[0],
    shg_df['memory_mb'].iloc[0]
]

x = np.arange(len(methods))
w = 0.35

ax4_twin = ax4.twinx()
b1 = ax4.bar(x - w/2, build_times, width=w, color='#34495e', label='Build Time (s)', edgecolor='black')
b2 = ax4_twin.bar(x + w/2, memories, width=w, color='#3498db', label='Memory Cost (MB)', edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(methods, fontsize=10, fontweight='bold')
ax4.set_ylabel('Construction Time (seconds)', color='#34495e', fontsize=11, fontweight='bold')
ax4_twin.set_ylabel('Index Memory Size (MB)', color='#2980b9', fontsize=11, fontweight='bold')
ax4.set_title('(d) Index Construction Time & Memory Footprint', fontsize=12, fontweight='bold')

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
ax4.grid(axis='y', linestyle='--', alpha=0.5)

# Value labels on bars
for bar in b1:
    h = bar.get_height()
    ax4.annotate(f'{h:.1f}s', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax4_twin.annotate(f'{h:.1f}M', xy=(bar.get_x() + bar.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
out_plot = 'results/plots/hnsw_vs_shg_paper_validation.png'
plt.savefig(out_plot, bbox_inches='tight', dpi=300)
print(f"Saved complete validation plot to: {out_plot}")
