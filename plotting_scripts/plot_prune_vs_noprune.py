"""HNSW vs SHG with / without base-layer pruning, all four workloads, one figure.
Each workload's three methods share one graph and were timed interleaved per query in the same run,
so within a column the comparison is like-for-like. SIFT10M ran on the server, the rest on this machine."""
import os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker

def load(path, rename):
    df = pd.read_csv(path)
    df = df[df.method.isin(rename)].copy()
    df['method'] = df.method.map(rename)
    return df

std = {'HNSW': 'HNSW', 'SHG_pruned': 'pruned', 'SHG_no_prune': 'noprune'}
workloads = [
    ('Enron (d = 1,369, N = 95K)',  'bench_prune/out_enron/prune.csv', std),
    ('GIST1M (d = 960, N = 1M)',    'bench_prune/out_gist/prune.csv',  std),
    ('SIFT1M (d = 128, N = 1M)',    'bench_prune/out_sift/prune.csv',  std),
    ('SIFT10M (d = 128, N = 10M)',  'bench10m/out/speedup10m.csv',
     {'HNSW': 'HNSW', 'SHG_original': 'pruned', 'SHG_no_base_prune': 'noprune'}),
]
workloads = [w for w in workloads if os.path.exists(w[1])]

S = {
    'HNSW':    dict(label='HNSW',                       color='#1f77b4', marker='o', ls='--'),
    'pruned':  dict(label='SHG with base pruning',      color='#d62728', marker='s', ls='-'),
    'noprune': dict(label='SHG without base pruning',   color='#9467bd', marker='P', ls='-'),
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
n = len(workloads)
fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9.5), dpi=200, squeeze=False)
for c, (title, path, ren) in enumerate(workloads):
    df = load(path, ren)
    top, bot = axes[0, c], axes[1, c]
    for m, s in S.items():
        d = df[df.method == m].sort_values('ef')
        top.plot(d.lat_mean_us / 1000, d.recall, color=s['color'], marker=s['marker'], ls=s['ls'], lw=2.2, ms=6, label=s['label'])
    top.set_title(title, fontsize=12.5, fontweight='bold')
    top.set_xlabel('Mean latency (ms)  [lower is better]', fontweight='bold')
    top.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    base = df[df.method == 'HNSW'].set_index('ef').lat_mean_us
    for m in ('pruned', 'noprune'):
        d = df[df.method == m].set_index('ef').sort_index()
        bot.plot(d.index, base.loc[d.index] / d.lat_mean_us, color=S[m]['color'], marker=S[m]['marker'], lw=2.2, ms=6, label=S[m]['label'])
    bot.axhline(1.0, color=S['HNSW']['color'], ls=':', lw=2, label='HNSW (1.00x)')
    bot.set_xscale('log'); bot.xaxis.set_major_formatter(ticker.ScalarFormatter()); bot.set_xticks(sorted(df.ef.unique()))
    bot.tick_params(axis='x', labelsize=8.5)
    bot.set_xlabel('efSearch', fontweight='bold')
    bot.set_ylim(0, max(1.15, bot.get_ylim()[1]))
    for ax in (top, bot): ax.grid(True, ls='--', alpha=.5)
    if c == 0:
        top.set_ylabel('Recall@20  [higher is better]', fontweight='bold')
        bot.set_ylabel('Speedup vs HNSW at same ef  (x)', fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=9.5, frameon=True)
axes[1, 0].legend(loc='lower right', fontsize=9.5, frameon=True)
fig.suptitle('Effect of SHG base-layer pruning vs HNSW', fontsize=15, fontweight='bold')
plt.tight_layout()
for d in ('plots', 'results/plots'):
    os.makedirs(d, exist_ok=True); fig.savefig(f'{d}/prune_vs_noprune_all_workloads.png', dpi=200)
print('Saved plots/prune_vs_noprune_all_workloads.png with', [w[0] for w in workloads])
