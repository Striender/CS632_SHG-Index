import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('plots', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# Load data
enron_csv = 'results/data/hnsw_vs_shg_comparison.csv'
sift_csv = 'results/data/hnsw_vs_shg_comparison_sift.csv'
if not os.path.exists(sift_csv):
    sift_csv = 'results/data/ef_sweep_curves_sift.csv'

df_enron = pd.read_csv(enron_csv)
df_sift = pd.read_csv(sift_csv)

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
# FIGURE 1: RECALL@20 VS. LATENCY (MS) - ENRON (LEFT) & SIFT (RIGHT)
# =========================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Left Panel: Enron Recall vs Latency (ms)
for method, s in styles.items():
    m_df = df_enron[df_enron['method'] == method].sort_values('latency_ms')
    if not m_df.empty:
        ax1.plot(
            m_df['latency_ms'], m_df['recall'],
            label=s['label'],
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=s['linewidth'],
            markersize=s['markersize']
        )

ax1.set_title('Enron Dataset (d = 1,369, N = 94,987)\nRecall@20 vs. Latency (ms)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Mean Query Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.set_ylim(0.80, 1.01)
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Right Panel: SIFT Recall vs Latency (ms)
for method, s in styles.items():
    m_df = df_sift[df_sift['method'] == method].sort_values('latency_ms')
    if not m_df.empty:
        ax2.plot(
            m_df['latency_ms'], m_df['recall'],
            label=s['label'],
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=s['linewidth'],
            markersize=s['markersize']
        )

ax2.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nRecall@20 vs. Latency (ms)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Mean Query Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_ylim(0.75, 1.01)
ax2.legend(loc='lower right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()
fig1.savefig('plots/recall_vs_latency.png', dpi=300)
fig1.savefig('results/plots/recall_vs_latency.png', dpi=300)
print("Saved plots/recall_vs_latency.png and results/plots/recall_vs_latency.png")

# =========================================================================
# FIGURE 2: RECALL@20 VS. THROUGHPUT (QPS) - ENRON (LEFT) & SIFT (RIGHT)
# =========================================================================
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Left Panel: Enron Throughput vs Recall@20
for method, s in styles.items():
    m_df = df_enron[df_enron['method'] == method].sort_values('qps')
    if not m_df.empty:
        ax3.plot(
            m_df['qps'], m_df['recall'],
            label=s['label'],
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=s['linewidth'],
            markersize=s['markersize']
        )

ax3.set_title('Enron Dataset (d = 1,369, N = 94,987)\nRecall@20 vs. Throughput (QPS)', fontsize=13, fontweight='bold', pad=12)
ax3.set_xlabel('Throughput (Queries / Sec) [Higher is Better]', fontsize=11, fontweight='bold')
ax3.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax3.set_ylim(0.80, 1.01)
ax3.legend(loc='lower right', frameon=True, fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.5)

# Right Panel: SIFT Throughput vs Recall@20
for method, s in styles.items():
    m_df = df_sift[df_sift['method'] == method].sort_values('qps')
    if not m_df.empty:
        ax4.plot(
            m_df['qps'], m_df['recall'],
            label=s['label'],
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=s['linewidth'],
            markersize=s['markersize']
        )

ax4.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nRecall@20 vs. Throughput (QPS)', fontsize=13, fontweight='bold', pad=12)
ax4.set_xlabel('Throughput (Queries / Sec) [Higher is Better]', fontsize=11, fontweight='bold')
ax4.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax4.set_ylim(0.75, 1.01)
ax4.legend(loc='lower right', frameon=True, fontsize=10)
ax4.grid(True, linestyle='--', alpha=0.5)

fig2.tight_layout()
fig2.savefig('plots/throughput_comparison.png', dpi=300)
fig2.savefig('results/plots/throughput_comparison.png', dpi=300)
print("Saved plots/throughput_comparison.png and results/plots/throughput_comparison.png")
