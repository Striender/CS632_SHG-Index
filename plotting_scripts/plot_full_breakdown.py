"""Vanilla HNSW and SHG as published, base layer at efSearch 64, from ONE run (bench_stall_v2, run_stall5.sh):
base layer only (upper layers removed), split into CPU idle (waiting on memory) / distance compute / search overhead.
Per query: median of 5 runs; then mean over queries. Base split in the ratio of the per-evaluation cycles:
  idle = cold - warm, compute = microbenchmark, overhead = warm - compute.
Reads bench_prune/stall5_<dataset>/stall.csv. Writes new_plot/21_full_breakdown_hnsw_ef64.png and .csv."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = 64
SYS = [('HNSW_stock', 'Vanilla HNSW', '21_full_breakdown_hnsw'), ('SHG_original', 'SHG as published', '21b_full_breakdown_shg')]
OUT = 'new_plot'
DS = [('enron', 'Enron\n(d=1,369, N=95K)'), ('gist', 'GIST1M\n(d=960, N=1M)'), ('sift', 'SIFT1M\n(d=128, N=1M)')]
PARTS = [('idle', 'CPU idle (waiting on memory)', '#d62728'),
         ('compute', 'Distance compute', '#2ca02c'),
         ('overhead', 'Search overhead', '#1f77b4')]
INK, MUTED = '#202124', '#5f6368'

for M, sysname, tag in SYS:
    rows = []
    for k, _ in DS:
        r = pd.read_csv(f'bench_prune/stall5_{k}/stall.csv').set_index(['method', 'ef']).loc[(M, EF)]
        c = dict(idle=max(r.stall_cyc_per_eval, 0), compute=r.compute_cyc_per_eval, overhead=r.overhead_cyc_per_eval)
        t = sum(c.values())
        row = dict(dataset=k, efSearch=EF, full_us=r.full_us, upper_us=r.upper_us, base_us=r.cold_base_us)
        for n, v in c.items(): row[f'{n}_us'] = r.cold_base_us * v / t
        for n in ('idle', 'compute', 'overhead'): row[f'{n}_pct'] = 100 * row[f'{n}_us'] / r.cold_base_us
        rows.append(row)
    R = pd.DataFrame(rows); R.round(2).to_csv(f'{OUT}/{tag}_ef{EF}.csv', index=False)

    fig, a = plt.subplots(figsize=(10, 6.4), dpi=200)
    ymax = R.base_us.max()
    for xi, r in R.iterrows():
        b = 0
        for n, lab, col in PARTS:
            v = r[f'{n}_us']
            a.bar(xi, v, 0.55, bottom=b, color=col, edgecolor='white', lw=1.5, label=lab if xi == 0 else None)
            if v > 0.08 * ymax:
                a.text(xi, b + v / 2, f'{r[f"{n}_pct"]:.0f}%\n({v:.0f} us)', ha='center', va='center', color='white', fontsize=9.5, fontweight='bold')
            elif v > 0:
                a.text(xi + 0.3, b + v / 2, f'{r[f"{n}_pct"]:.0f}% ({v:.0f} us)', ha='left', va='center', fontsize=8.5, color=INK, fontweight='bold')
            b += v
        a.text(xi, b + 0.012 * ymax, f'{b:.0f} us / query', ha='center', va='bottom', fontsize=10, color=INK, fontweight='bold')
    a.set_xticks(range(len(DS))); a.set_xticklabels([t for _, t in DS], fontsize=10.5, fontweight='bold')
    a.set_ylabel('Base-layer time per query (us)', fontsize=11, fontweight='bold')
    a.set_ylim(0, ymax * 1.1); a.set_xlim(-0.55, len(DS) - 0.25)
    a.grid(axis='y', color='#e8eaed'); a.set_axisbelow(True)
    h, l = a.get_legend_handles_labels()
    a.legend(h[::-1], l[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(); [fig.savefig(f'{OUT}/{tag}_ef{EF}.{x}', bbox_inches='tight') for x in ('png', 'pdf')]; plt.close(fig)
    print(R[['dataset', 'full_us', 'upper_us', 'base_us', 'idle_us', 'compute_us', 'overhead_us', 'idle_pct', 'compute_pct', 'overhead_pct']].round(1).to_string(index=False))
