import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('plots', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def load_and_bin_subsets(file_path, n_bins=7):
    if not os.path.exists(file_path):
        return None, None
    df = pd.read_csv(file_path, sep='\t', comment='#', names=['qid', 'time_sec', 'time_us', 'recall'])
    df = df.sort_values('recall').reset_index(drop=True)
    splits = np.array_split(df, n_bins)
    
    mean_times_ms = [b['time_us'].mean() / 1000.0 for b in splits]
    mean_recalls = [b['recall'].mean() for b in splits]
    return mean_times_ms, mean_recalls

methods = [
    ('HNSW', 'HNSW', '#1f77b4', 'o', '--', 2.0, 6),
    ('Baseline_SHG', 'SHG (normal)', '#d62728', 's', '-', 2.0, 6),
    ('Step1_SkipPrefetch', 'SHG with Idea 1', '#ff7f0e', '^', '-.', 2.0, 6),
    ('Step2_SkipLookahead', 'SHG + Idea 2', '#2ca02c', 'D', '-', 2.5, 7)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# =========================================================================
# LEFT PANEL: ENRON (Author's 7-Subset Protocol - Original Recall-Ordered)
# =========================================================================
for m_key, m_label, col, mkr, ls, lw, ms in methods:
    fpath = f'results/data/enron_{m_key}_K20_queries.txt'
    times, recalls = load_and_bin_subsets(fpath, n_bins=7)
    if times is not None:
        ax1.plot(times, recalls, label=m_label, color=col, marker=mkr, linestyle=ls, linewidth=lw, markersize=ms)

ax1.set_title('Enron Dataset (d = 1,369, N = 94,987)\nAuthor\'s Exact Recall Subsets Protocol (k = 20, ef = 20)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Mean Query Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.set_xlim(0.12, 0.38)
ax1.set_ylim(0.40, 1.02)
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# =========================================================================
# RIGHT PANEL: SIFT1M (Author's 7-Subset Protocol - Original Recall-Ordered)
# =========================================================================
for m_key, m_label, col, mkr, ls, lw, ms in methods:
    fpath = f'results/data/sift_{m_key}_K20_queries.txt'
    times, recalls = load_and_bin_subsets(fpath, n_bins=7)
    if times is not None:
        ax2.plot(times, recalls, label=m_label, color=col, marker=mkr, linestyle=ls, linewidth=lw, markersize=ms)

ax2.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nAuthor\'s Exact Recall Subsets Protocol (k = 20, ef = 20)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Mean Query Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Recall@20 [Higher is Better]', fontsize=11, fontweight='bold')
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_xlim(0.04, 0.17)
ax2.set_ylim(0.35, 1.02)
ax2.legend(loc='lower right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('plots/author_exact_subsets_comparison.png', dpi=300)
fig.savefig('results/plots/author_exact_subsets_comparison.png', dpi=300)
print("Saved plots/author_exact_subsets_comparison.png and results/plots/author_exact_subsets_comparison.png")
