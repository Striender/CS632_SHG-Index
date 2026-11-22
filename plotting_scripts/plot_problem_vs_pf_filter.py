"""Problem vs solution, efSearch 64: per dataset, the published system next to + Pf_Filter.
Stacked per-query base-layer time: CPU idle (waiting on memory) / distance compute / search overhead.
One figure for HNSW (vanilla hnswlib) and one for SHG (as published).
Reads bench_prune/stall5_<dataset>/stall.csv. Writes new_plot/19a_problem_vs_pf_filter_hnsw.png,
19b_problem_vs_pf_filter_shg.png and problem_vs_pf_filter.csv.
Base layer = 93-99.8% of query time (same run, figures 20/20b)."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = 64
OUT = 'new_plot'
DS = [('enron', 'Enron\n(d=1,369, N=95K)'), ('gist', 'GIST1M\n(d=960, N=1M)'),
      ('sift', 'SIFT1M\n(d=128, N=1M)')]
SYS = [('HNSW', 'HNSW_stock', 'vanilla HNSW', 'HNSW_v2_filter', '19a_problem_vs_pf_filter_hnsw'),
       ('SHG', 'SHG_original', 'SHG as published', 'SHG_v2_filter', '19b_problem_vs_pf_filter_shg')]
PARTS = [('idle', 'CPU idle (waiting on memory)', '#d62728'),
         ('compute', 'Distance compute', '#2ca02c'),
         ('overhead', 'Search overhead', '#1f77b4')]
INK, MUTED = '#202124', '#5f6368'

rows = []
for k, _ in DS:
    d = pd.read_csv(f'bench_prune/stall5_{k}/stall.csv').set_index(['method', 'ef'])
    for fam, base, blab, sol, _ in SYS:
        for role, m in (('problem', base), ('Pf_Filter', sol)):
            r = d.loc[(m, EF)]
            p = dict(idle=max(r.stall_cyc_per_eval, 0), compute=r.compute_cyc_per_eval, overhead=r.overhead_cyc_per_eval)
            tot = sum(p.values())
            rows.append(dict(dataset=k, system=fam, role=role, method=m, efSearch=EF, total_us=r.cold_base_us,
                             **{f'{n}_us': r.cold_base_us * v / tot for n, v in p.items()}))
R = pd.DataFrame(rows)
R.round(2).to_csv(f'{OUT}/problem_vs_pf_filter.csv', index=False)

for fam, base, blab, sol, tag in SYS:
    fig, a = plt.subplots(figsize=(12, 6.4), dpi=200)
    w, gap = 0.36, 0.03
    ymax = R[R.system == fam].total_us.max()
    for xi, (k, _) in enumerate(DS):
        s = R[(R.dataset == k) & (R.system == fam)].set_index('role')
        for j, role in enumerate(('problem', 'Pf_Filter')):
            x = xi + (j - 0.5) * (w + gap); r = s.loc[role]; bottom = 0
            for n, lab, col in PARTS:
                v = r[f'{n}_us']
                a.bar(x, v, w, bottom=bottom, color=col, edgecolor='white', lw=1.5,
                      label=lab if (xi == 0 and j == 0) else None)
                if v > 0.05 * ymax:
                    a.text(x, bottom + v / 2, f'{v:.0f}', ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
                bottom += v
            a.text(x, bottom + 0.012 * ymax, f'{bottom:.0f} us', ha='center', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
            a.text(x, -0.035 * ymax, blab.replace(' ', '\n', 1) if role == 'problem' else 'Pf_Filter', ha='center', va='top',
                   fontsize=8, color=MUTED, fontweight='bold')
        t0, t1 = s.loc['problem', 'total_us'], s.loc['Pf_Filter', 'total_us']
        i0, i1 = s.loc['problem', 'idle_us'], s.loc['Pf_Filter', 'idle_us']
        a.text(xi, max(t0, t1) + 0.075 * ymax, f'time {100 * (t1 / t0 - 1):+.1f}%\nidle {100 * (i1 / i0 - 1):+.0f}%',
               ha='center', va='bottom', fontsize=9, color=INK, fontweight='bold')
    a.set_xticks(range(len(DS))); a.set_xticklabels([t for _, t in DS], fontsize=10.5, fontweight='bold')
    a.tick_params(axis='x', pad=30, length=0)
    a.set_ylabel('Base-layer time per query (us)', fontsize=11, fontweight='bold')
    a.set_ylim(0, ymax * 1.25); a.set_xlim(-0.6, len(DS) - 0.4)
    a.grid(axis='y', color='#e8eaed'); a.set_axisbelow(True)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10.5, frameon=False)
    fig.tight_layout()
    for x in ('png', 'pdf'): fig.savefig(f'{OUT}/{tag}.{x}', bbox_inches='tight')
    plt.close(fig)
    print('saved', f'{OUT}/{tag}.png')
