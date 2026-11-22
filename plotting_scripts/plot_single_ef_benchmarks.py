import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('plots', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

enron_csv = 'results/data/hnsw_vs_shg_comparison.csv'
sift_csv = 'results/data/hnsw_vs_shg_comparison_sift.csv'

df_enron = pd.read_csv(enron_csv)
df_sift = pd.read_csv(sift_csv)

methods = ['Vanilla_HNSW', 'Baseline_SHG', 'Step1_SkipPrefetch', 'Step2_SkipLookahead']
labels = ['HNSW', 'SHG (normal)', 'SHG with Idea 1', 'SHG + Idea 2']
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Filter for ef = 20
e20_enron = df_enron[df_enron['ef'] == 20].set_index('method').reindex(methods).reset_index()
e20_sift = df_sift[df_sift['ef'] == 20].set_index('method').reindex(methods).reset_index()

# =========================================================================
# FIGURE 1: THROUGHPUT COMPARISON AT ef = 20 (ENRON & SIFT)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

x = np.arange(len(methods))
bar_width = 0.55

# Enron Throughput
bars1 = ax1.bar(x, e20_enron['qps'], width=bar_width, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
ax1.set_title('Enron Dataset (d = 1,369, N = 94,987)\nThroughput at Default efSearch = 20', fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel('Throughput (Queries / Sec) [Higher is Better]', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax1.set_ylim(0, max(e20_enron['qps']) * 1.20)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(e20_enron['qps']) * 0.02),
             f'{yval:,.1f} QPS', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# SIFT Throughput
bars2 = ax2.bar(x, e20_sift['qps'], width=bar_width, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
ax2.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nThroughput at Default efSearch = 20', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel('Throughput (Queries / Sec) [Higher is Better]', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(e20_sift['qps']) * 1.20)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(e20_sift['qps']) * 0.02),
             f'{yval:,.1f} QPS', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig('plots/throughput_comparison.png', dpi=300)
fig.savefig('results/plots/throughput_comparison.png', dpi=300)
print("Saved plots/throughput_comparison.png and results/plots/throughput_comparison.png")
plt.close(fig)

# =========================================================================
# FIGURE 2: SPEEDUP VS. SHG BASELINE AT ef = 20 (ENRON & SIFT)
# (SHG Baseline is represented by the horizontal dotted line at 1.00x)
# =========================================================================
methods_vs_shg = ['Vanilla_HNSW', 'Step1_SkipPrefetch', 'Step2_SkipLookahead']
labels_vs_shg = ['HNSW', 'SHG with Idea 1', 'SHG + Idea 2']
colors_vs_shg = ['#1f77b4', '#ff7f0e', '#2ca02c']
x_shg = np.arange(len(methods_vs_shg))

e20_enron_shg = df_enron[df_enron['ef'] == 20].set_index('method').reindex(methods_vs_shg).reset_index()
e20_sift_shg = df_sift[df_sift['ef'] == 20].set_index('method').reindex(methods_vs_shg).reset_index()

shg_base_lat_enron = df_enron[(df_enron['ef'] == 20) & (df_enron['method'] == 'Baseline_SHG')]['latency_us'].values[0]
speedups_shg_enron = [shg_base_lat_enron / lat for lat in e20_enron_shg['latency_us']]

shg_base_lat_sift = df_sift[(df_sift['ef'] == 20) & (df_sift['method'] == 'Baseline_SHG')]['latency_us'].values[0]
speedups_shg_sift = [shg_base_lat_sift / lat for lat in e20_sift_shg['latency_us']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
bar_width_3 = 0.45

# Enron Speedup vs SHG
bars1 = ax1.bar(x_shg, speedups_shg_enron, width=bar_width_3, color=colors_vs_shg, edgecolor='black', linewidth=1.2, alpha=0.9)
ax1.axhline(1.0, color='#d62728', linestyle=':', linewidth=2.2, label='SHG Baseline (= 1.00x)')
ax1.set_title('Enron Dataset (d = 1,369, N = 94,987)\nSpeedup vs. Baseline SHG at efSearch = 20', fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel('Speedup Factor vs. SHG Baseline (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax1.set_xticks(x_shg)
ax1.set_xticklabels(labels_vs_shg, fontsize=10.5, fontweight='bold')
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.set_ylim(0.85, max(speedups_shg_enron) * 1.15)
ax1.legend(loc='upper left', frameon=True, fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar, sp in zip(bars1, speedups_shg_enron):
    yval = bar.get_height()
    pct = (sp - 1.0) * 100.0
    pct_str = f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015,
             f'{sp:.2f}x\n({pct_str})', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# SIFT Speedup vs SHG
bars2 = ax2.bar(x_shg, speedups_shg_sift, width=bar_width_3, color=colors_vs_shg, edgecolor='black', linewidth=1.2, alpha=0.9)
ax2.axhline(1.0, color='#d62728', linestyle=':', linewidth=2.2, label='SHG Baseline (= 1.00x)')
ax2.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nSpeedup vs. Baseline SHG at efSearch = 20', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel('Speedup Factor vs. SHG Baseline (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax2.set_xticks(x_shg)
ax2.set_xticklabels(labels_vs_shg, fontsize=10.5, fontweight='bold')
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_ylim(0.80, max(speedups_shg_sift) * 1.18)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar, sp in zip(bars2, speedups_shg_sift):
    yval = bar.get_height()
    pct = (sp - 1.0) * 100.0
    pct_str = f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%'
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03,
             f'{sp:.2f}x\n({pct_str})', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig('plots/speedup_vs_shg.png', dpi=300)
fig.savefig('results/plots/speedup_vs_shg.png', dpi=300)
print("Saved plots/speedup_vs_shg.png and results/plots/speedup_vs_shg.png")
plt.close(fig)

# =========================================================================
# FIGURE 3: SPEEDUP VS. HNSW BASELINE AT ef = 20 (ENRON & SIFT)
# (HNSW Baseline is represented by the horizontal dotted line at 1.00x)
# =========================================================================
methods_vs_hnsw = ['Baseline_SHG', 'Step1_SkipPrefetch', 'Step2_SkipLookahead']
labels_vs_hnsw = ['SHG (normal)', 'SHG with Idea 1', 'SHG + Idea 2']
colors_vs_hnsw = ['#d62728', '#ff7f0e', '#2ca02c']
x_hnsw = np.arange(len(methods_vs_hnsw))

e20_enron_hnsw = df_enron[df_enron['ef'] == 20].set_index('method').reindex(methods_vs_hnsw).reset_index()
e20_sift_hnsw = df_sift[df_sift['ef'] == 20].set_index('method').reindex(methods_vs_hnsw).reset_index()

hnsw_base_lat_enron = df_enron[(df_enron['ef'] == 20) & (df_enron['method'] == 'Vanilla_HNSW')]['latency_us'].values[0]
speedups_hnsw_enron = [hnsw_base_lat_enron / lat for lat in e20_enron_hnsw['latency_us']]

hnsw_base_lat_sift = df_sift[(df_sift['ef'] == 20) & (df_sift['method'] == 'Vanilla_HNSW')]['latency_us'].values[0]
speedups_hnsw_sift = [hnsw_base_lat_sift / lat for lat in e20_sift_hnsw['latency_us']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Enron Speedup vs HNSW
bars1 = ax1.bar(x_hnsw, speedups_hnsw_enron, width=bar_width_3, color=colors_vs_hnsw, edgecolor='black', linewidth=1.2, alpha=0.9)
ax1.axhline(1.0, color='#1f77b4', linestyle=':', linewidth=2.2, label='HNSW Baseline (= 1.00x)')
ax1.set_title('Enron Dataset (d = 1,369, N = 94,987)\nSpeedup vs. HNSW at efSearch = 20', fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel('Speedup Factor vs. HNSW Baseline (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax1.set_xticks(x_hnsw)
ax1.set_xticklabels(labels_vs_hnsw, fontsize=10.5, fontweight='bold')
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.set_ylim(0.80, max(speedups_hnsw_enron) * 1.15)
ax1.legend(loc='upper left', frameon=True, fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar, sp in zip(bars1, speedups_hnsw_enron):
    yval = bar.get_height()
    pct = (sp - 1.0) * 100.0
    pct_str = f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015,
             f'{sp:.2f}x\n({pct_str})', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# SIFT Speedup vs HNSW
bars2 = ax2.bar(x_hnsw, speedups_hnsw_sift, width=bar_width_3, color=colors_vs_hnsw, edgecolor='black', linewidth=1.2, alpha=0.9)
ax2.axhline(1.0, color='#1f77b4', linestyle=':', linewidth=2.2, label='HNSW Baseline (= 1.00x)')
ax2.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nSpeedup vs. HNSW at efSearch = 20', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel('Speedup Factor vs. HNSW Baseline (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax2.set_xticks(x_hnsw)
ax2.set_xticklabels(labels_vs_hnsw, fontsize=10.5, fontweight='bold')
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_ylim(0.0, 1.20)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar, sp in zip(bars2, speedups_hnsw_sift):
    yval = bar.get_height()
    pct = (sp - 1.0) * 100.0
    pct_str = f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%'
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03,
             f'{sp:.2f}x\n({pct_str})', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig('plots/speedup_vs_hnsw.png', dpi=300)
fig.savefig('results/plots/speedup_vs_hnsw.png', dpi=300)
print("Saved plots/speedup_vs_hnsw.png and results/plots/speedup_vs_hnsw.png")
plt.close(fig)
