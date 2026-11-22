"""One-glance result: speedup of a prefetch method over the published (vanilla) system, base layer.
  HNSW: baseline = vanilla hnswlib (its own stock prefetch)      -> HNSW_stock
  SHG : baseline = SHG as published in the paper (no prefetch)   -> SHG_original
Reads bench_prune/stall4_<dataset>/stall.csv, writes new_plot/<fig>_ef<EF>.png and .csv.
Usage: python3 plot_idea2v2_summary.py [efSearch] [v2_filter|v2_basic|idea2_L2]
Speedup = base-layer time of the vanilla system / time with the method (same graph, same queries)."""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = int(sys.argv[1]) if len(sys.argv) > 1 else 64
VAR = sys.argv[2] if len(sys.argv) > 2 else 'v2_filter'
LAB = {'v2_filter': ('Idea 2.0', 'filter + warm-up + lookahead + link-list + Idea 1', '10_idea2v2_vs_vanilla'),
       'v2_basic':  ('Idea 2.0 without filter', 'warm-up + lookahead + link-list + Idea 1', '11_idea2v2_nofilter_vs_vanilla'),
       'idea2_L2':  ('Idea 2', 'lookahead j+2, 3 lines + next-candidate link list + Idea 1', '12_idea2_vs_vanilla')}[VAR]
OUT = 'new_plot'
DS = [('enron', 'Enron'), ('gist', 'GIST1M'), ('sift', 'SIFT1M'), ('sift10m', 'SIFT10M')]
FAM = [('HNSW', 'HNSW_stock', 'HNSW + {m}  (vs vanilla HNSW)', '#1f5fbf'),
       ('SHG', 'SHG_original', 'SHG + {m}  (vs SHG as published)', '#c5221f')]
INK, MUTED = '#202124', '#5f6368'

rows = []
for k, name in DS:
    d = pd.read_csv(f'bench_prune/stall4_{k}/stall.csv').set_index(['method', 'ef'])
    for fam, base, _, _ in FAM:
        t0, t1 = d.loc[(base, EF), 'cold_base_us'], d.loc[(f'{fam}_{VAR}', EF), 'cold_base_us']
        rows.append(dict(dataset=name, family=fam, efSearch=EF, vanilla_us=t0, method_us=t1, speedup_vs_vanilla=t0 / t1))
R = pd.DataFrame(rows)
R.round(4).to_csv(f'{OUT}/{LAB[2]}_ef{EF}.csv', index=False)

fig, a = plt.subplots(figsize=(11, 5.6), dpi=200)
w = 0.36
for i, (fam, _, lab, col) in enumerate(FAM):
    r = R[R.family == fam].reset_index(drop=True)
    xs = [x + (i - 0.5) * (w + 0.02) for x in range(len(r))]
    vals = r.speedup_vs_vanilla.values
    a.bar(xs, vals - 1, w, bottom=1, color=col, edgecolor='white', lw=1.5, label=lab.format(m=LAB[0]))
    for x, v in zip(xs, vals):
        a.text(x, max(v, 1) + 0.004, f'{v:.2f}x\n({100 * (v - 1):+.1f}%)', ha='center', va='bottom',
               fontsize=9, color=INK, fontweight='bold')
a.axhline(1, color=MUTED, lw=1.2)
a.text(-0.45, 0.997, 'baseline = vanilla system (1.0)', ha='left', va='top', fontsize=9, color=MUTED)
a.set_xticks(range(len(DS))); a.set_xticklabels([n for _, n in DS], fontsize=11, fontweight='bold')
a.set_ylabel('Base-layer speedup vs vanilla (x)', fontsize=11, fontweight='bold')
lo, hi = R.speedup_vs_vanilla.min(), R.speedup_vs_vanilla.max()
a.set_ylim(min(0.95, lo - 0.03), hi + 0.10)
a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', color='#e8eaed'); a.set_axisbelow(True)
a.legend(loc='upper left', fontsize=10, frameon=False)
fig.suptitle(f'{LAB[0]} vs the published systems (efSearch = {EF})', fontsize=15, fontweight='bold', color=INK)
a.set_title(f'{LAB[0]} = {LAB[1]}   |   M = 48, efConstruction = 80, K = 20', fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f'{OUT}/{LAB[2]}_ef{EF}.png', bbox_inches='tight'); plt.close(fig)
print('saved', f'{OUT}/{LAB[2]}_ef{EF}.png')
