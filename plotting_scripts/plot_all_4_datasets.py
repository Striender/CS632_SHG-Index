"""Enron, GIST1M, SIFT1M, SIFT10M side by side. SIFT10M comes from bench10m/out (server run),
the other three from the pinned-core i5-13400 runs, so compare trends across panels, not absolute µs."""
import os
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker

os.makedirs('plots', exist_ok=True); os.makedirs('results/plots', exist_ok=True)
D = 'results/data/'
datasets = [
    ('Enron (d = 1,369, N = 94,987)',     pd.read_csv(D+'hnsw_vs_shg_comparison.csv'),         0.80),
    ('GIST1M (d = 960, N = 1,000,000)',   pd.read_csv(D+'hnsw_vs_shg_comparison_gist.csv'),    0.35),
    ('SIFT1M (d = 128, N = 1,000,000)',   pd.read_csv(D+'hnsw_vs_shg_comparison_sift.csv'),    0.70),
    ('SIFT10M (d = 128, N = 10,000,000)', pd.read_csv(D+'hnsw_vs_shg_comparison_sift10m.csv'), 0.70),
]
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
S = {
    'Vanilla_HNSW':        dict(label='HNSW',                         color='#1f77b4', marker='o', ls='--', lw=2.0),
    'Baseline_SHG':        dict(label='SHG (normal)',                 color='#d62728', marker='s', ls='-',  lw=2.0),
    'Step1_SkipPrefetch':  dict(label='SHG with Idea 1',              color='#ff7f0e', marker='^', ls='-.', lw=2.0),
    'Step2_SkipLookahead': dict(label='SHG + Idea 2',                 color='#2ca02c', marker='D', ls='-',  lw=2.5),
    'Step2_NoBasePrune':   dict(label='SHG, no base prune',  color='#9467bd', marker='P', ls='-',  lw=2.5),
}

def save(fig, name):
    for d in ('plots', 'results/plots'): fig.savefig(f'{d}/{name}', dpi=200)
    plt.close(fig); print('Saved plots/' + name)

def grid(): return plt.subplots(2, 2, figsize=(16, 12), dpi=200)

# 1. recall vs latency
fig, axes = grid()
for ax, (title, df, ymin) in zip(axes.flat, datasets):
    for m, s in S.items():
        d = df[df.method == m].sort_values('latency_ms')
        if len(d): ax.plot(d.latency_ms, d.recall, label=s['label'], color=s['color'], marker=s['marker'],
                           linestyle=s['ls'], linewidth=s['lw'], markersize=6)
    ax.set_title(f'{title}\nRecall@20 vs. Latency', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Query Latency (ms) [Lower is Better]', fontweight='bold')
    ax.set_ylabel('Recall@20 [Higher is Better]', fontweight='bold')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05)); ax.set_ylim(ymin, 1.01)
    ax.legend(loc='lower right', frameon=True, fontsize=9); ax.grid(True, ls='--', alpha=.5)
plt.tight_layout(); save(fig, 'recall_vs_latency_4datasets.png')

# 2-4. bar charts at ef = 20
def bars(name, value_fn, ylabel, ref_line=None, exclude=()):
    fig, axes = grid()
    for ax, (title, df, _) in zip(axes.flat, datasets):
        e = df[df.ef == 20].set_index('method')
        ms = [m for m in S if m in e.index and m not in exclude]
        vals = [value_fn(e, m) for m in ms]
        b = ax.bar(range(len(ms)), vals, width=0.55, color=[S[m]['color'] for m in ms], edgecolor='black', lw=1.2)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([S[m]['label'].replace(', ', ',\n') for m in ms], fontsize=9, fontweight='bold')
        if ref_line:
            ax.axhline(1.0, color=ref_line[1], ls=':', lw=2.2, label=ref_line[0]); ax.legend(loc='upper left', fontsize=9)
        top = max(vals + ([1.0] if ref_line else [])); ax.set_ylim(0, top * 1.25)
        for r, v in zip(b, vals):
            txt = f'{v:,.0f} QPS' if ref_line is None else f'{v:.2f}x\n({(v-1)*100:+.1f}%)'
            ax.text(r.get_x() + r.get_width()/2, v + top*0.02, txt, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(f'{title}\n{ylabel.split(" [")[0]} at efSearch = 20', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold'); ax.grid(axis='y', ls='--', alpha=.5)
    plt.tight_layout(); save(fig, name)

bars('throughput_comparison_4datasets.png', lambda e, m: e.loc[m, 'qps'], 'Throughput (QPS) [Higher is Better]')
bars('speedup_vs_shg_4datasets.png', lambda e, m: e.loc['Baseline_SHG', 'latency_us'] / e.loc[m, 'latency_us'],
     'Speedup vs. SHG Baseline (x)', ('SHG Baseline (= 1.00x)', '#d62728'), exclude=('Baseline_SHG',))
bars('speedup_vs_hnsw_4datasets.png', lambda e, m: e.loc['Vanilla_HNSW', 'latency_us'] / e.loc[m, 'latency_us'],
     'Speedup vs. HNSW (x)', ('HNSW Baseline (= 1.00x)', '#1f77b4'), exclude=('Vanilla_HNSW',))
