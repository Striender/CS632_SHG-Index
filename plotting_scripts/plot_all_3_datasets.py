import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('plots', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

df_enron = pd.read_csv('results/data/hnsw_vs_shg_comparison.csv')
df_sift = pd.read_csv('results/data/hnsw_vs_shg_comparison_sift.csv')
df_gist = pd.read_csv('results/data/hnsw_vs_shg_comparison_gist.csv')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

styles = {
    'Vanilla_HNSW': {
        'label': 'HNSW',
        'color': '#1f77b4',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': 2.0,
        'markersize': 6
    },
    'Baseline_SHG': {
        'label': 'SHG (normal)',
        'color': '#d62728',
        'marker': 's',
        'linestyle': '-',
        'linewidth': 2.0,
        'markersize': 6
    },
    'Step1_SkipPrefetch': {
        'label': 'SHG with Idea 1',
        'color': '#ff7f0e',
        'marker': '^',
        'linestyle': '-.',
        'linewidth': 2.0,
        'markersize': 6
    },
    'Step2_SkipLookahead': {
        'label': 'SHG + Idea 2',
        'color': '#2ca02c',
        'marker': 'D',
        'linestyle': '-',
        'linewidth': 2.5,
        'markersize': 7
    }
}

# =========================================================================
# FIGURE 1: RECALL@20 VS. LATENCY (3-PANEL: ENRON, GIST1M, SIFT1M)
# =========================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)

datasets_info = [
    (ax1, df_enron, 'Enron (d = 1,369, N = 94,987)', 0.80, 1.01),
    (ax2, df_gist, 'GIST1M (d = 960, N = 1,000,000)', 0.35, 1.01),
    (ax3, df_sift, 'SIFT1M (d = 128, N = 1,000,000)', 0.70, 1.01),
]

for ax, df_curr, title, ymin, ymax in datasets_info:
    for method, s in styles.items():
        m_df = df_curr[df_curr['method'] == method].sort_values('latency_ms')
        if not m_df.empty:
            ax.plot(
                m_df['latency_ms'], m_df['recall'],
                label=s['label'],
                color=s['color'],
                marker=s['marker'],
                linestyle=s['linestyle'],
                linewidth=s['linewidth'],
                markersize=s['markersize']
            )
    ax.set_title(f'{title}\nRecall@20 vs. Latency (ms)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Mean Query Latency (ms) [Lower is Better]', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Recall@20 [Higher is Better]', fontsize=10.5, fontweight='bold')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(ymin, ymax)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('plots/recall_vs_latency_3datasets.png', dpi=300)
fig.savefig('results/plots/recall_vs_latency_3datasets.png', dpi=300)
fig.savefig('plots/recall_vs_latency.png', dpi=300)
fig.savefig('results/plots/recall_vs_latency.png', dpi=300)
print("Saved plots/recall_vs_latency.png (3 datasets)")
plt.close(fig)

# =========================================================================
# FIGURE 2: THROUGHPUT AT ef = 20 (3-PANEL: ENRON, GIST1M, SIFT1M)
# =========================================================================
methods = ['Vanilla_HNSW', 'Baseline_SHG', 'Step1_SkipPrefetch', 'Step2_SkipLookahead']
labels = ['HNSW', 'SHG (normal)', 'SHG with Idea 1', 'SHG + Idea 2']
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6), dpi=300)
x = np.arange(len(methods))
bar_width = 0.55

e20_enron = df_enron[df_enron['ef'] == 20].set_index('method').reindex(methods).reset_index()
e20_gist = df_gist[df_gist['ef'] == 20].set_index('method').reindex(methods).reset_index()
e20_sift = df_sift[df_sift['ef'] == 20].set_index('method').reindex(methods).reset_index()

thru_info = [
    (ax1, e20_enron, 'Enron (d = 1,369, N = 94,987)'),
    (ax2, e20_gist, 'GIST1M (d = 960, N = 1,000,000)'),
    (ax3, e20_sift, 'SIFT1M (d = 128, N = 1,000,000)')
]

for ax, e20_df, title in thru_info:
    bars = ax.bar(x, e20_df['qps'], width=bar_width, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
    ax.set_title(f'{title}\nThroughput at efSearch = 20', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Throughput (Queries / Sec) [Higher is Better]', fontsize=10.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0, max(e20_df['qps']) * 1.22)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(e20_df['qps']) * 0.02),
                f'{yval:,.1f} QPS', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig('plots/throughput_comparison.png', dpi=300)
fig.savefig('results/plots/throughput_comparison.png', dpi=300)
print("Saved plots/throughput_comparison.png (3 datasets)")
plt.close(fig)

# =========================================================================
# FIGURE 3: SPEEDUP VS SHG BASELINE AT ef = 20 (3-PANEL)
# =========================================================================
methods_vs_shg = ['Vanilla_HNSW', 'Step1_SkipPrefetch', 'Step2_SkipLookahead']
labels_vs_shg = ['HNSW', 'SHG with Idea 1', 'SHG + Idea 2']
colors_vs_shg = ['#1f77b4', '#ff7f0e', '#2ca02c']
x_shg = np.arange(len(methods_vs_shg))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6), dpi=300)

sp_shg_info = [
    (ax1, df_enron, 'Enron (d = 1,369, N = 94,987)', 0.05, 0.85, 1.25),
    (ax2, df_gist, 'GIST1M (d = 960, N = 1,000,000)', 0.05, 0.85, 1.35),
    (ax3, df_sift, 'SIFT1M (d = 128, N = 1,000,000)', 0.10, 0.80, 2.45)
]

for ax, df_curr, title, tick_step, ymin, ymax in sp_shg_info:
    e20 = df_curr[df_curr['ef'] == 20].set_index('method').reindex(methods_vs_shg).reset_index()
    base_lat = df_curr[(df_curr['ef'] == 20) & (df_curr['method'] == 'Baseline_SHG')]['latency_us'].values[0]
    speedups = [base_lat / lat for lat in e20['latency_us']]
    
    bars = ax.bar(x_shg, speedups, width=0.45, color=colors_vs_shg, edgecolor='black', linewidth=1.2, alpha=0.9)
    ax.axhline(1.0, color='#d62728', linestyle=':', linewidth=2.2, label='SHG Baseline (= 1.00x)')
    ax.set_title(f'{title}\nSpeedup vs. Baseline SHG at ef = 20', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Speedup Factor vs. SHG Baseline (x)', fontsize=10.5, fontweight='bold')
    ax.set_xticks(x_shg)
    ax.set_xticklabels(labels_vs_shg, fontsize=9.5, fontweight='bold')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(ymin, max(speedups) * 1.15)
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, sp in zip(bars, speedups):
        yval = bar.get_height()
        pct = (sp - 1.0) * 100.0
        pct_str = f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02,
                f'{sp:.2f}x\n({pct_str})', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig('plots/speedup_vs_shg.png', dpi=300)
fig.savefig('results/plots/speedup_vs_shg.png', dpi=300)
print("Saved plots/speedup_vs_shg.png (3 datasets)")
plt.close(fig)

# =========================================================================
# FIGURE 4: SPEEDUP VS HNSW BASELINE AT ef = 20 (3-PANEL)
# =========================================================================
methods_vs_hnsw = ['Baseline_SHG', 'Step1_SkipPrefetch', 'Step2_SkipLookahead']
labels_vs_hnsw = ['SHG (normal)', 'SHG with Idea 1', 'SHG + Idea 2']
colors_vs_hnsw = ['#d62728', '#ff7f0e', '#2ca02c']
x_hnsw = np.arange(len(methods_vs_hnsw))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6), dpi=300)

sp_hnsw_info = [
    (ax1, df_enron, 'Enron (d = 1,369, N = 94,987)', 0.05, 0.75, 1.15),
    (ax2, df_gist, 'GIST1M (d = 960, N = 1,000,000)', 0.05, 0.70, 1.20),
    (ax3, df_sift, 'SIFT1M (d = 128, N = 1,000,000)', 0.10, 0.00, 1.20)
]

for ax, df_curr, title, tick_step, ymin, ymax in sp_hnsw_info:
    e20 = df_curr[df_curr['ef'] == 20].set_index('method').reindex(methods_vs_hnsw).reset_index()
    base_lat = df_curr[(df_curr['ef'] == 20) & (df_curr['method'] == 'Vanilla_HNSW')]['latency_us'].values[0]
    speedups = [base_lat / lat for lat in e20['latency_us']]
    
    bars = ax.bar(x_hnsw, speedups, width=0.45, color=colors_vs_hnsw, edgecolor='black', linewidth=1.2, alpha=0.9)
    ax.axhline(1.0, color='#1f77b4', linestyle=':', linewidth=2.2, label='HNSW Baseline (= 1.00x)')
    ax.set_title(f'{title}\nSpeedup vs. HNSW at ef = 20', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Speedup Factor vs. HNSW Baseline (x)', fontsize=10.5, fontweight='bold')
    ax.set_xticks(x_hnsw)
    ax.set_xticklabels(labels_vs_hnsw, fontsize=9.5, fontweight='bold')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(ymin, max(speedups) * 1.15)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, sp in zip(bars, speedups):
        yval = bar.get_height()
        pct = (sp - 1.0) * 100.0
        pct_str = f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02,
                f'{sp:.2f}x\n({pct_str})', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig('plots/speedup_vs_hnsw.png', dpi=300)
fig.savefig('results/plots/speedup_vs_hnsw.png', dpi=300)
print("Saved plots/speedup_vs_hnsw.png (3 datasets)")
plt.close(fig)
