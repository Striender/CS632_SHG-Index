import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('plots', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

csv_path = 'results/data/hnsw_vs_shg_comparison.csv'
df = pd.read_csv(csv_path)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

styles = {
    'Vanilla_HNSW': {
        'label': 'Vanilla HNSW (M=48, efC=80)',
        'color': '#1f77b4',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': 2.2,
        'markersize': 7
    },
    'Baseline_SHG': {
        'label': 'SHG (normal / baseline)',
        'color': '#d62728',
        'marker': 's',
        'linestyle': '-',
        'linewidth': 2.2,
        'markersize': 7
    },
    'Step1_SkipPrefetch': {
        'label': 'SHG with Idea 1 (Skip Prefetch)',
        'color': '#ff7f0e',
        'marker': '^',
        'linestyle': '-.',
        'linewidth': 2.2,
        'markersize': 7
    },
    'Step2_SkipLookahead': {
        'label': 'SHG + Idea 2 (Skip + Lookahead k=2)',
        'color': '#2ca02c',
        'marker': 'D',
        'linestyle': '-',
        'linewidth': 2.6,
        'markersize': 8
    }
}

# Left plot: Latency in ms (x-axis) vs Recall@20 (y-axis)
for method, s in styles.items():
    m_df = df[df['method'] == method].sort_values('latency_ms')
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
        # Annotate select ef values
        for _, row in m_df.iterrows():
            if int(row['ef']) in [20, 48, 128, 512]:
                ax1.annotate(
                    f"ef={int(row['ef'])}",
                    (row['latency_ms'], row['recall']),
                    textcoords="offset points",
                    xytext=(8, -10 if method == 'Step2_SkipLookahead' else 5),
                    ha='left',
                    fontsize=8,
                    color=s['color'],
                    fontweight='bold'
                )

ax1.set_title('Recall@20 vs. Latency on Enron (d=1,369)\nMean Latency Across 5 Repeats per efSearch', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Mean Query Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Right plot: Throughput QPS (x-axis) vs Recall@20 (y-axis)
for method, s in styles.items():
    m_df = df[df['method'] == method].sort_values('qps')
    if not m_df.empty:
        ax2.plot(
            m_df['qps'], m_df['recall'],
            label=s['label'],
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=s['linewidth'],
            markersize=s['markersize']
        )

ax2.set_title('Recall@20 vs. Throughput (QPS) on Enron\nMean Across 5 Repeats per efSearch', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Throughput (Queries / Sec) [Higher is Better]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax2.legend(loc='lower right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save in plots/ and results/plots/
out_path1 = 'plots/recall_vs_latency.png'
out_path2 = 'results/plots/recall_vs_latency.png'
plt.savefig(out_path1, dpi=300)
plt.savefig(out_path2, dpi=300)
print(f"Successfully generated plots:\n  - {out_path1}\n  - {out_path2}")
