"""Motivation figures for prefetching, from measured data (bench_prune/stall_<dataset>/stall.csv).

Per base-layer distance evaluation, in core cycles:
  memory stall = cold run - warm rerun of the same query (the warm rerun has the query's working set in cache)
  compute      = real L2 kernel on cache-resident vectors (microbenchmark)
  overhead     = warm rerun - compute   (heap / visited-list / neighbour scanning / SHG pruning checks)
Usage: python3 plot_motivation.py [efSearch]   (20, 64 or 100; default 100). efConstruction = 80, M = 48 throughout."""
import os, sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = 'new_plot'; os.makedirs(OUT, exist_ok=True)
DATASETS = [('enron', 'Enron (d=1,369, N=95K)'), ('gist', 'GIST1M (d=960, N=1M)'),
            ('sift', 'SIFT1M (d=128, N=1M)'), ('sift10m', 'SIFT10M (d=128, N=10M)')]
data = {}; full = {}
for k, _ in DATASETS:
    f, f2 = f'bench_prune/stall4_{k}/stall.csv', f'bench_prune/stall2_{k}/stall.csv'   # stall4: corrected clock calibration
    if os.path.exists(f2): full[k] = pd.read_csv(f2)   # lookahead sweep L=2..5 (speedups only, clock-independent)
    if not os.path.exists(f): f = f2
    if os.path.exists(f):
        d = pd.read_csv(f)
        d = d[d.ef == EF].copy(); d['method'] = d.method.replace({'SHG_idea2_L2': 'SHG_idea2', 'HNSW_idea2_L2': 'HNSW_idea2'})
        data[k] = d.set_index('method')
DATASETS = [(k, t) for k, t in DATASETS if k in data]
C = dict(stall='#d62728', compute='#2ca02c', overhead='#1f77b4')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def grid():
    nr = (len(DATASETS) + 1) // 2
    fig, ax = plt.subplots(nr, 2, figsize=(17, 5.8 * nr), dpi=200, squeeze=False)
    for a in ax.flat[len(DATASETS):]: a.set_visible(False)
    return fig, ax.flat

# ---------------- Figure 5: where the time per distance evaluation goes ----------------
BARS = [('HNSW_noprefetch', 'HNSW\nno prefetch'), ('HNSW_stock', 'HNSW\nstock prefetch'), ('HNSW_idea2', 'HNSW +\nIdea 2'),
        ('SHG_original', 'SHG\nas published'), ('SHG_stock', 'SHG +\nstock prefetch'), ('SHG_idea2', 'SHG +\nIdea 2')]
fig, axes = grid()
rows = []
for a, (k, title) in zip(axes, DATASETS):
    d = data[k]
    for x, (m, lab) in enumerate(BARS):
        r = d.loc[m]; parts = [('stall', r.stall_cyc_per_eval), ('compute', r.compute_cyc_per_eval), ('overhead', r.overhead_cyc_per_eval)]
        tot = sum(max(v, 0) for _, v in parts); bottom = 0
        for name, v in parts:
            v = max(v, 0)
            a.bar(x, v, 0.62, bottom=bottom, color=C[name], edgecolor='black', lw=0.8)
            if v / tot > 0.07:
                a.text(x, bottom + v / 2, f'{v:.0f}\n({100 * v / tot:.0f}%)', ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
            bottom += v
        a.text(x, bottom * 1.01, f'{tot:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        rows.append(dict(dataset=k, method=m, ef=EF, stall_cyc=r.stall_cyc_per_eval, compute_cyc=r.compute_cyc_per_eval,
                         overhead_cyc=r.overhead_cyc_per_eval, total_cyc=tot, stall_pct=100 * max(r.stall_cyc_per_eval, 0) / tot))
    a.axvline(2.5, color='grey', ls=':', lw=1.2)
    a.set_xticks(range(len(BARS))); a.set_xticklabels([b[1] for b in BARS], fontsize=9.5, fontweight='bold')
    a.set_title(f'{title}\nDRAM latency {d.dram_latency_cyc.iloc[0]:.0f} cyc, core {d.core_ghz.iloc[0]:.2f} GHz', fontweight='bold')
    a.set_ylabel('Core cycles per distance evaluation', fontweight='bold'); a.set_ylim(0, a.get_ylim()[1] * 1.08)
    a.grid(axis='x', visible=False)
h = [plt.Rectangle((0, 0), 1, 1, color=C[n], ec='black') for n in ('stall', 'compute', 'overhead')]
fig.legend(h, ['Memory stall (CPU waiting on DRAM)', 'Distance compute (AVX2)', 'Overhead incl. L2/L3 hits (heap, visited list, neighbour scan, pruning checks)'],
           loc='lower center', ncol=3, fontsize=10.5, frameon=True)
fig.suptitle(f'Where base-layer time goes, per distance evaluation (measured, efSearch = {EF})', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(f'{OUT}/5_cpu_time_breakdown_ef{EF}.png', bbox_inches='tight'); plt.close(fig)
pd.DataFrame(rows).round(1).to_csv(f'{OUT}/cpu_time_breakdown_ef{EF}.csv', index=False)
print('saved', f'{OUT}/5_cpu_time_breakdown_ef{EF}.png', f'{OUT}/cpu_time_breakdown_ef{EF}.csv')

# ---------------- Figure 6: timeline of 3 consecutive evaluations, measured averages ----------------
TL = [('SHG_original', 'SHG as published\n(no prefetch)'), ('SHG_stock', 'SHG +\nstock prefetch'), ('SHG_idea2', 'SHG +\nIdea 2')]
fig, axes = grid()
for a, (k, title) in zip(axes, DATASETS):
    d = data[k]; ys = list(range(len(TL)))[::-1]; xmax = 0
    for y, (m, lab) in zip(ys, TL):
        r = d.loc[m]; t = 0
        segs = [('stall', max(r.stall_cyc_per_eval, 0)), ('compute', r.compute_cyc_per_eval), ('overhead', r.overhead_cyc_per_eval)]
        for n in range(3):
            for name, v in segs:
                a.barh(y, v, left=t, height=0.55, color=C[name], edgecolor='black', lw=0.7)
                if name == 'stall' and v > 0.12 * sum(s[1] for s in segs):
                    a.text(t + v / 2, y, f'stall\n{v:.0f}', ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')
                t += v
        a.text(t * 1.01, y, f'{t:.0f} cyc', va='center', fontsize=9, fontweight='bold'); xmax = max(xmax, t)
    base = d.loc['SHG_original']; tot0 = base.cold_cyc_per_eval
    for y, (m, lab) in zip(ys[1:], TL[1:]):
        r = d.loc[m]; hidden = base.stall_cyc_per_eval - r.stall_cyc_per_eval
        a.text(xmax * 1.13, y, f'{100 * (tot0 - r.cold_cyc_per_eval) / tot0:.0f}% less time\nstall hidden: {hidden:.0f} of {base.stall_cyc_per_eval:.0f} cyc',
               va='center', fontsize=8, color='#1b5e20')
    a.set_yticks(ys); a.set_yticklabels([l for _, l in TL], fontsize=9.5, fontweight='bold')
    a.set_xlim(0, xmax * 1.62); a.set_xlabel('Core cycles (3 consecutive distance evaluations, measured averages)', fontweight='bold')
    a.set_title(title, fontweight='bold'); a.grid(axis='y', visible=False)
fig.legend(h, ['Memory stall', 'Distance compute', 'Search overhead'], loc='lower center', ncol=3, fontsize=10.5, frameon=True)
fig.suptitle(f'How much memory stall prefetching hides on the SHG base layer (measured, efSearch = {EF})', fontsize=15, fontweight='bold')
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(f'{OUT}/6_memory_stall_timeline_ef{EF}.png', bbox_inches='tight'); plt.close(fig)
print('saved', f'{OUT}/6_memory_stall_timeline_ef{EF}.png')

# ---------------- Figure 7: Idea 2 lookahead distance sweep (L = 2..5) ----------------
if EF == 64:
    LS = [2, 3, 4, 5]; EFS = sorted(next(iter(full.values())).ef.unique())
    fig, axes = grid(); rows = []
    for a, (k, title) in zip(axes, DATASETS):
        d = full[k].set_index(['method', 'ef'])
        for fam, base, stock, col in (('SHG', 'SHG_original', 'SHG_stock', '#d62728'), ('HNSW', 'HNSW_noprefetch', 'HNSW_stock', '#1f77b4')):
            for e, ls in zip(EFS, ['-', '--', ':']):
                b = d.loc[(base, e), 'cold_base_us']
                sp = [b / d.loc[(f'{fam}_idea2_L{L}', e), 'cold_base_us'] for L in LS]
                a.plot(LS, sp, color=col, ls=ls, marker='o', lw=2, label=f'{fam} + Idea 2, efSearch={e}')
                a.scatter([1.6], [b / d.loc[(stock, e), 'cold_base_us']], color=col, marker='*', s=90, zorder=5)
                for L, v in zip(LS, sp): rows.append(dict(dataset=k, family=fam, efSearch=e, lookahead=L, speedup_vs_noprefetch=v,
                                                           stock_speedup=b / d.loc[(stock, e), 'cold_base_us']))
        a.axhline(1, color='grey', lw=1)
        a.set_xticks([1.6] + LS); a.set_xticklabels(['stock\n(1-ahead)'] + [f'L={L}' for L in LS], fontweight='bold')
        a.set_xlabel('Idea 2 lookahead distance (prefetch neighbour j+L)', fontweight='bold')
        a.set_ylabel('Base-layer speedup vs no prefetch (x)', fontweight='bold'); a.set_title(title, fontweight='bold')
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc='lower center', ncol=3, fontsize=10, frameon=True)
    fig.suptitle('Idea 2: does prefetching further ahead help? (stars = hnswlib stock prefetch)', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0.07, 1, 0.97)); fig.savefig(f'{OUT}/7_lookahead_sweep.png', bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(rows).round(4).to_csv(f'{OUT}/lookahead_sweep.csv', index=False); print('saved', f'{OUT}/7_lookahead_sweep.png')
