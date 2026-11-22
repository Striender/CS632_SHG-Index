"""All figures for the same-graph HNSW / SHG / prefetch / pruning study -> new_plot/.
Data: bench_prune/out_<dataset>/prune.csv. Every method in a dataset shares one graph and was timed
interleaved per query on the same pinned core, so comparisons within a panel are like-for-like."""
import os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker

OUT = 'new_plot'; os.makedirs(OUT, exist_ok=True)
DATASETS = [('enron', 'Enron (d=1,369, N=95K)'), ('gist', 'GIST1M (d=960, N=1M)'),
            ('sift', 'SIFT1M (d=128, N=1M)'), ('sift10m', 'SIFT10M (d=128, N=10M)')]
data = {k: pd.read_csv(f'bench_prune/out_{k}/prune.csv') for k, _ in DATASETS if os.path.exists(f'bench_prune/out_{k}/prune.csv')}
DATASETS = [(k, t) for k, t in DATASETS if k in data]
REF = 'HNSW_stock'

S = {
    'HNSW_stock':        dict(label='HNSW (stock hnswlib prefetch)', color='#1f77b4', marker='o', ls='--'),
    'HNSW_noprefetch':   dict(label='HNSW, no prefetch',             color='#7fb3e0', marker='o', ls=':'),
    'HNSW_idea2':        dict(label='HNSW + Idea 2',                 color='#17becf', marker='o', ls='-.'),
    'SHG_original':      dict(label='SHG original (as published)',   color='#d62728', marker='s', ls='-'),
    'SHG_stock':         dict(label='SHG + stock prefetch',          color='#ff7f0e', marker='^', ls='-'),
    'SHG_idea2':         dict(label='SHG + Idea 2',                  color='#2ca02c', marker='D', ls='-'),
    'SHG_noprune':       dict(label='SHG, no base prune',            color='#9467bd', marker='P', ls='-'),
    'SHG_noprune_stock': dict(label='SHG, no prune + stock prefetch', color='#e377c2', marker='X', ls='-'),
    'SHG_noprune_idea2': dict(label='SHG, no prune + Idea 2',        color='#8c564b', marker='*', ls='-'),
}
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def lat(df, m): return df[df.method == m].set_index('ef').sort_index().lat_mean_us

def figure(rows=1):
    # 2 x 2 grid (one panel per workload) instead of one long horizontal row
    nc = 2; nr = (len(DATASETS) + 1) // 2
    fig, ax = plt.subplots(nr, nc, figsize=(7.5 * nc, 5.6 * nr), dpi=200, squeeze=False)
    for a in ax.flat[len(DATASETS):]: a.set_visible(False)
    return fig, ax

def finish(fig, name, title, ax):
    for a in ax.flat: a.grid(True, ls='--', alpha=.5)
    handles, labels = {}, {}
    for a in ax.flat:
        for h, l in zip(*a.get_legend_handles_labels()): handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=min(3, len(handles)), fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0.045 * ((len(handles) + 2) // 3), 1, 0.96))
    fig.savefig(f'{OUT}/{name}', dpi=200, bbox_inches='tight'); plt.close(fig); print('saved', f'{OUT}/{name}')

def pareto(methods, name, title):
    fig, ax = figure(1)
    for c, (k, t) in enumerate(DATASETS):
        df = data[k]; a = ax.flat[c]
        for m in methods:
            d = df[df.method == m].sort_values('ef')
            s = S[m]; a.plot(d.lat_mean_us / 1000, d.recall, color=s['color'], marker=s['marker'], ls=s['ls'], lw=2, ms=5, label=s['label'])
        a.set_title(t, fontweight='bold'); a.set_xlabel('Mean latency (ms)  [lower is better]', fontweight='bold')
        a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        if c % 2 == 0: a.set_ylabel('Recall@20  [higher is better]', fontweight='bold')
    finish(fig, name, title, ax)

def speedup(methods, base, name, title, ylabel):
    fig, ax = figure(1)
    for c, (k, t) in enumerate(DATASETS):
        df = data[k]; a = ax.flat[c]; b = lat(df, base)
        for m in methods:
            l = lat(df, m); s = S[m]
            a.plot(l.index, b / l, color=s['color'], marker=s['marker'], ls=s['ls'], lw=2, ms=5, label=s['label'])
        a.axhline(1.0, color=S[base]['color'], ls=':', lw=2, label=f'{S[base]["label"]} = 1.00x')
        a.set_xscale('log'); a.set_xticks(sorted(df.ef.unique())); a.xaxis.set_major_formatter(ticker.ScalarFormatter())
        a.tick_params(axis='x', labelsize=8); a.set_title(t, fontweight='bold'); a.set_xlabel('efSearch', fontweight='bold')
        lo, hi = a.get_ylim(); a.set_ylim(min(lo, 0.9), max(hi, 1.1))
        if c % 2 == 0: a.set_ylabel(ylabel, fontweight='bold')
    finish(fig, name, title, ax)

# 1. Baseline: HNSW vs SHG as published
pareto(['HNSW_stock', 'SHG_original'], '1_baseline_hnsw_vs_shg.png', 'Baseline: HNSW vs SHG (as published), same graph')
# 2. Pruning effect (no prefetch on either side, to isolate pruning)
pareto(['HNSW_stock', 'SHG_original', 'SHG_noprune'], '2_pruning_recall_vs_latency.png',
       'Effect of SHG base-layer pruning (no prefetch)')
speedup(['SHG_original', 'SHG_noprune'], REF, '2b_pruning_speedup_vs_hnsw.png',
        'Effect of SHG base-layer pruning: speedup vs HNSW at same ef', 'Speedup vs HNSW (x)')
# 3. Prefetch ablation: does Idea 2 beat hnswlib's stock prefetch?
speedup(['SHG_stock', 'SHG_idea2', 'SHG_noprune_stock', 'SHG_noprune_idea2'], 'SHG_original', '3_prefetch_speedup_vs_shg_original.png',
        'Prefetch ablation: speedup over SHG as published', 'Speedup vs SHG original (x)')
speedup(['HNSW_noprefetch', 'HNSW_idea2'], REF, '3b_prefetch_on_hnsw.png',
        'Same prefetch policies applied to HNSW', 'Speedup vs HNSW stock (x)')
# 4. Everything vs HNSW
speedup(['SHG_original', 'SHG_stock', 'SHG_idea2', 'SHG_noprune_stock', 'SHG_noprune_idea2'], REF, '4_all_speedup_vs_hnsw.png',
        'All SHG variants: speedup vs HNSW at same ef', 'Speedup vs HNSW (x)')
pareto(['HNSW_stock', 'SHG_original', 'SHG_idea2', 'SHG_noprune_idea2'], '4b_all_recall_vs_latency.png',
       'Recall vs latency: HNSW, SHG, and best SHG variants')

# summary table at ef=20 for the write-up
rows = []
for k, t in DATASETS:
    df = data[k].set_index(['method', 'ef'])
    h = df.loc[(REF, 20), 'lat_mean_us']; o = df.loc[('SHG_original', 20), 'lat_mean_us']
    for m in S:
        if (m, 20) in df.index:
            r = df.loc[(m, 20)]
            rows.append(dict(dataset=k, method=m, recall=r.recall, lat_us=r.lat_mean_us, vs_hnsw=h / r.lat_mean_us, vs_shg_original=o / r.lat_mean_us))
pd.DataFrame(rows).round(4).to_csv(f'{OUT}/summary_ef20.csv', index=False); print('saved', f'{OUT}/summary_ef20.csv')
