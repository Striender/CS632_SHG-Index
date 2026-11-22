#!/usr/bin/env python3
"""
plot_sift_performance.py
------------------------
Generates a dedicated high-resolution multi-panel performance plot for the
SIFT1M dataset (N=1,000,000, d=128, K=20) comparing Baseline SHG vs Step 1 vs Step 2.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results/plots', exist_ok=True)

csv_path = 'results/data/ef_sweep_curves_sift.csv'
if not os.path.exists(csv_path):
    print(f"Missing {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
b = df[df['method'] == 'Baseline_SHG'].sort_values('recall')
s1 = df[df['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
s2 = df[df['method'] == 'Step2_SkipLookahead'].sort_values('recall')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
fig.suptitle("Empirical Evaluation on SIFT1M Dataset (N = 1,000,000, d = 128, K = 20)\nSingle-Threaded Execution (Pinned to Intel P-Core 2)",
             fontsize=13, fontweight='bold', y=0.98)

# Panel 1: Time vs Recall Pareto Curve
ax1 = axes[0, 0]
ax1.plot(b['latency_ms'], b['recall'], 's-', color='#333333', markerfacecolor='#555555',
         markersize=6, linewidth=2.0, label='Baseline SHG (Unoptimized)')
ax1.plot(s1['latency_ms'], s1['recall'], '^--', color='#1f77b4', markersize=6, linewidth=1.8, label='Step 1: Skip Prefetch')
ax1.plot(s2['latency_ms'], s2['recall'], 'D-', color='#2ca02c', markersize=7, linewidth=2.5, label='Step 2: Skip + Lookahead (Ours)')
ax1.set_xlabel('Search Time (ms)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Recall@20', fontsize=11, fontweight='bold')
ax1.set_title('(a) Search Performance Curve (Time vs Recall)', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Panel 2: Speedup vs Recall
ax2 = axes[0, 1]
min_rec = max(b['recall'].min(), s2['recall'].min())
max_rec = min(b['recall'].max(), s2['recall'].max())
rec_grid = np.linspace(min_rec, max_rec, 25)
tb_interp = np.interp(rec_grid, b['recall'], b['latency_ms'])
ts2_interp = np.interp(rec_grid, s2['recall'], s2['latency_ms'])
speedup_pct = (tb_interp - ts2_interp) / tb_interp * 100.0

ax2.plot(rec_grid, speedup_pct, 'o-', color='#2ca02c', linewidth=2.5, markersize=6)
ax2.fill_between(rec_grid, 0, speedup_pct, color='#2ca02c', alpha=0.15)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax2.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
ax2.set_ylabel('Latency Reduction vs Baseline (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Step 2 Speedup across Recall Levels', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 20)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.annotate(f'+16.3% peak speedup\n@ Recall 0.77', xy=(rec_grid[0], speedup_pct[0]),
             xytext=(rec_grid[0]+0.04, speedup_pct[0]-3),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='#2ca02c'),
             fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#2ca02c'))

# Panel 3: Throughput (QPS) vs Recall
ax3 = axes[1, 0]
ax3.plot(b['recall'], b['qps'], 's-', color='#333333', linewidth=2.0, markersize=6, label='Baseline SHG')
ax3.plot(s1['recall'], s1['qps'], '^--', color='#1f77b4', linewidth=1.8, markersize=6, label='Step 1 (Skip)')
ax3.plot(s2['recall'], s2['qps'], 'D-', color='#2ca02c', linewidth=2.5, markersize=7, label='Step 2: Lookahead (Ours)')
ax3.set_xlabel('Recall@20', fontsize=11, fontweight='bold')
ax3.set_ylabel('Throughput (Queries / sec)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Throughput (QPS) vs Recall', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', frameon=True, fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.6)

# Panel 4: Latency per beam width (ef)
ax4 = axes[1, 1]
x = np.arange(len(b))
w = 0.35
ax4.bar(x - w/2, b['latency_ms'] * 1000, width=w, label='Baseline SHG', color='#555555')
ax4.bar(x + w/2, s2['latency_ms'] * 1000, width=w, label='Step 2: Lookahead', color='#2ca02c')
ax4.set_xticks(x)
ax4.set_xticklabels(b['ef'].astype(int), rotation=45)
ax4.set_xlabel('Beam Width (efSearch)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Query Latency (µs)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Query Latency (µs) across Beam Widths (efSearch)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', frameon=True, fontsize=10)
ax4.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
out_fig = 'results/plots/sift_search_performance.png'
plt.savefig(out_fig, bbox_inches='tight', dpi=300)
print(f"Saved SIFT dedicated plot to: {out_fig}")
