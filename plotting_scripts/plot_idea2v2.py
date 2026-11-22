"""Idea 2.0 figures from bench_prune/stall4_<dataset>/stall.csv (bench_stall built with -DIDEA2V2).

8_idea2v2_speedup.png          base-layer speedup vs no prefetch, every prefetch variant, SHG and HNSW
                               (bars = efSearch 64, dots = efSearch 20 and 100)
9a_idea2v2_breakdown_hnsw.png  stall / compute / overhead per distance evaluation, HNSW, efSearch 64
9b_idea2v2_breakdown_shg.png   same for SHG
idea2v2_summary.csv            all numbers used above
Speedup = mean cold base-layer time of no-prefetch / that of the variant (same graph, same queries)."""
import os
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = 'new_plot'; os.makedirs(OUT, exist_ok=True)
DATASETS = [('enron', 'Enron (d=1,369, N=95K)'), ('gist', 'GIST1M (d=960, N=1M)'),
            ('sift', 'SIFT1M (d=128, N=1M)'), ('sift10m', 'SIFT10M (d=128, N=10M)')]
data = {k: pd.read_csv(f'bench_prune/stall4_{k}/stall.csv').set_index(['method', 'ef'])
        for k, _ in DATASETS if os.path.exists(f'bench_prune/stall4_{k}/stall.csv')}
DATASETS = [(k, t) for k, t in DATASETS if k in data]
BASE = {'SHG': 'SHG_original', 'HNSW': 'HNSW_noprefetch'}
VARS = [('stock', 'stock'), ('idea2_L2', 'Idea 2'), ('v2_basic', '2.0\nbasic'), ('v2_filter', '2.0\nfilter')]
FAM_C = {'HNSW': '#1f77b4', 'SHG': '#d62728'}
C = dict(stall='#d62728', compute='#2ca02c', overhead='#1f77b4')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def grid(w=17):
    nr = (len(DATASETS) + 1) // 2
    fig, ax = plt.subplots(nr, 2, figsize=(w, 5.8 * nr), dpi=200, squeeze=False)
    for a in ax.flat[len(DATASETS):]: a.set_visible(False)
    return fig, ax.flat

rows = []
for k, _ in DATASETS:
    d = data[k]
    for fam, b in BASE.items():
        for e in sorted(d.index.get_level_values('ef').unique()):
            t0 = d.loc[(b, e), 'cold_base_us']
            for v, _ in [('noprefetch', '')] + VARS:
                m = b if v == 'noprefetch' else f'{fam}_{v}'
                r = d.loc[(m, e)]
                rows.append(dict(dataset=k, family=fam, variant=v, efSearch=e, base_us=r.cold_base_us, speedup=t0 / r.cold_base_us,
                                 stall=r.stall_cyc_per_eval, compute=r.compute_cyc_per_eval, overhead=r.overhead_cyc_per_eval,
                                 total=r.cold_cyc_per_eval))
S = pd.DataFrame(rows)
S.round(4).to_csv(f'{OUT}/idea2v2_summary.csv', index=False)

# ---------- Figure 8: speedup of every variant ----------
fig, axes = grid()
x = np.arange(len(VARS)); w = 0.38
for a, (k, title) in zip(axes, DATASETS):
    for i, fam in enumerate(('HNSW', 'SHG')):
        s = S[(S.dataset == k) & (S.family == fam)]
        v64 = [s[(s.variant == v) & (s.efSearch == 64)].speedup.iloc[0] for v, _ in VARS]
        xs = x + (i - 0.5) * w
        a.bar(xs, v64, w, color=FAM_C[fam], edgecolor='black', lw=0.7, label=f'{fam}, efSearch 64')
        for xx, val in zip(xs, v64): a.text(xx, val + 0.005, f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        for e, mk in ((20, 'v'), (100, '^')):
            ve = [s[(s.variant == v) & (s.efSearch == e)].speedup.iloc[0] for v, _ in VARS]
            a.scatter(xs, ve, marker=mk, color='black', s=18, zorder=5, label=f'efSearch {e}' if i == 0 else None)
        best = int(np.argmax(v64)); a.bar(xs[best], v64[best], w, fill=False, edgecolor='gold', lw=2.5)
    a.axhline(1, color='grey', lw=1); a.axvline(1.5, color='grey', ls=':', lw=1.2)
    a.set_xticks(x); a.set_xticklabels([l for _, l in VARS], fontsize=9, fontweight='bold')
    lo = min(0.95, S[S.dataset == k].speedup.min() - 0.03); hi = S[S.dataset == k].speedup.max() + 0.06
    a.set_ylim(lo, hi); a.set_ylabel('Base-layer speedup vs no prefetch (x)', fontweight='bold')
    a.set_title(title, fontweight='bold'); a.grid(axis='x', visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=10.5, frameon=True)
fig.suptitle('Idea 2.0 variants vs stock prefetch and Idea 2 (gold outline = best at efSearch 64)', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(f'{OUT}/8_idea2v2_speedup.png', bbox_inches='tight'); plt.close(fig)
print('saved', f'{OUT}/8_idea2v2_speedup.png')

# ---------- Figure 9: breakdown per variant, efSearch 64 ----------
for fam, tag in (('HNSW', '9a_idea2v2_breakdown_hnsw'), ('SHG', '9b_idea2v2_breakdown_shg')):
    fig, axes = grid(18)
    labs = [('noprefetch', 'no\nprefetch' if fam == 'HNSW' else 'as\npublished')] + VARS
    for a, (k, title) in zip(axes, DATASETS):
        s = S[(S.dataset == k) & (S.family == fam) & (S.efSearch == 64)].set_index('variant')
        for xi, (v, _) in enumerate(labs):
            r = s.loc[v]; bottom = 0; tot = max(r.stall, 0) + r.compute + r.overhead
            for name in ('stall', 'compute', 'overhead'):
                val = max(r[name], 0)
                a.bar(xi, val, 0.66, bottom=bottom, color=C[name], edgecolor='black', lw=0.7)
                if val / tot > 0.08: a.text(xi, bottom + val / 2, f'{val:.0f}', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
                bottom += val
            a.text(xi, bottom * 1.01, f'{tot:.0f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        a.set_xticks(range(len(labs))); a.set_xticklabels([l for _, l in labs], fontsize=8.5, fontweight='bold')
        a.set_ylabel('Core cycles per distance evaluation', fontweight='bold'); a.set_title(title, fontweight='bold')
        a.set_ylim(0, a.get_ylim()[1] * 1.06); a.grid(axis='x', visible=False)
    hh = [plt.Rectangle((0, 0), 1, 1, color=C[n], ec='black') for n in ('stall', 'compute', 'overhead')]
    fig.legend(hh, ['Memory stall (DRAM, not hidden)', 'Distance compute', 'Overhead incl. L2/L3 hits'], loc='lower center', ncol=3, fontsize=10.5, frameon=True)
    fig.suptitle(f'{fam}: where base-layer time goes for each prefetch variant (efSearch = 64)', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(f'{OUT}/{tag}.png', bbox_inches='tight'); plt.close(fig)
    print('saved', f'{OUT}/{tag}.png')
