"""Everything relative to vanilla HNSW (hnswlib with its own stock prefetch) = 1.0, base layer.
Bars: SHG as published, SHG + method, HNSW + method.
Reads bench_prune/stall4_<dataset>/stall.csv, writes new_plot/13_vs_vanilla_hnsw_<method>_ef<EF>.png and .csv.
Usage: python3 plot_vs_vanilla_hnsw.py [efSearch] [v2_filter|v2_basic|idea2_L2] [hnsw]   ('hnsw' = HNSW bar only)
Speedup = base-layer time of vanilla HNSW / base-layer time of the bar (same graph, same queries).
Note: base layer only (upper layers / shortcut jumps not included); SHG's base-layer pruning
also lowers its recall (see bench_prune/out_<dataset>/prune.csv)."""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = int(sys.argv[1]) if len(sys.argv) > 1 else 64
VAR = sys.argv[2] if len(sys.argv) > 2 else 'v2_filter'
NAME = {'v2_filter': 'Idea 2.0', 'v2_basic': 'Idea 2.0 without filter', 'idea2_L2': 'Idea 2'}[VAR]
OUT = 'new_plot'
HNSW_ONLY = len(sys.argv) > 3 and sys.argv[3] == 'hnsw'
if HNSW_ONLY and VAR == 'v2_filter': NAME = 'Pf_Filter'
DS = [('enron', 'Enron'), ('gist', 'GIST1M'), ('sift', 'SIFT1M'), ('sift10m', 'SIFT10M')]
BARS = [('SHG_original', 'SHG as published', '#f28b82'),
        (f'SHG_{VAR}', f'SHG + {NAME}', '#c5221f'),
        (f'HNSW_{VAR}', f'HNSW + {NAME}', '#1f5fbf')]
if HNSW_ONLY: BARS = BARS[2:]
INK, MUTED = '#202124', '#5f6368'

rows = []
for k, name in DS:
    d = pd.read_csv(f'bench_prune/stall4_{k}/stall.csv').set_index(['method', 'ef'])
    t0 = d.loc[('HNSW_stock', EF), 'cold_base_us']
    for m, lab, _ in BARS:
        t = d.loc[(m, EF), 'cold_base_us']
        rows.append(dict(dataset=name, bar=lab, efSearch=EF, vanilla_hnsw_us=t0, us=t, speedup_vs_vanilla_hnsw=t0 / t))
R = pd.DataFrame(rows)
tag = f'14_hnsw_{VAR}_vs_vanilla_hnsw_ef{EF}' if HNSW_ONLY else f'13_vs_vanilla_hnsw_{VAR}_ef{EF}'
R.round(4).to_csv(f'{OUT}/{tag}.csv', index=False)

fig, a = plt.subplots(figsize=(10 if HNSW_ONLY else 12, 5.8), dpi=200)
w = 0.45 if HNSW_ONLY else 0.26
for i, (m, lab, col) in enumerate(BARS):
    v = R[R.bar == lab].speedup_vs_vanilla_hnsw.values
    xs = [x + (0 if HNSW_ONLY else (i - 1) * (w + 0.015)) for x in range(len(DS))]
    if HNSW_ONLY:   # gain over vanilla: bars grow from the 1.0 line
        a.bar(xs, v - 1, w, bottom=1, color=col, edgecolor='white', lw=1.5, label=lab)
        for x, val in zip(xs, v):
            a.text(x, max(val, 1) + 0.002, f'{val:.3f}x  ({100 * (val - 1):+.1f}%)', ha='center', va='bottom', fontsize=10, color=INK, fontweight='bold')
    else:
        a.bar(xs, v, w, color=col, edgecolor='white', lw=1.5, label=lab)
        for x, val in zip(xs, v):
            a.text(x, val + 0.015, f'{val:.2f}x', ha='center', va='bottom', fontsize=9, color=INK, fontweight='bold')
a.axhline(1, color=INK, lw=1.3, ls='--')
a.set_xticks(range(len(DS))); a.set_xticklabels([n for _, n in DS], fontsize=11, fontweight='bold')
a.set_ylabel('Speedup over vanilla HNSW' if HNSW_ONLY else 'Base-layer speedup vs vanilla HNSW (x)', fontsize=11, fontweight='bold')
if HNSW_ONLY:
    a.set_ylim(min(0.97, R.speedup_vs_vanilla_hnsw.min() - 0.01), R.speedup_vs_vanilla_hnsw.max() + 0.02)
else:
    a.set_ylim(0, max(1.1, R.speedup_vs_vanilla_hnsw.max()) + 0.25)
a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', color='#e8eaed'); a.set_axisbelow(True)
a.legend(loc='upper left', ncol=3, fontsize=10, frameon=False)
fig.suptitle((f'HNSW + {NAME}' if HNSW_ONLY else f'SHG and HNSW with {NAME}') + f', relative to vanilla HNSW (efSearch = {EF})', fontsize=15, fontweight='bold', color=INK)
a.set_title('Dashed line = vanilla HNSW (1.0)  |  ' + ('' if HNSW_ONLY else 'base-layer time only  |  ') + '> 1 = faster than vanilla HNSW, < 1 = slower  |  M = 48, efConstruction = 80, K = 20',
            fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f'{OUT}/{tag}.png', bbox_inches='tight'); plt.close(fig)
print('saved', f'{OUT}/{tag}.png')
