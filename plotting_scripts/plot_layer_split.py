"""HNSW full query split into upper layers and base layer, efSearch 64, vanilla HNSW only (one bar per dataset).
Reads bench_prune/stall5_<dataset>/stall.csv (bench_stall_v2, run_stall5.sh, efSearch 64): upper_us and cold_base_us,
the same run as figure 21, so both figures show identical totals. Per query median of 5 runs, mean over queries.
Writes new_plot/20_layer_split_hnsw_ef64.png and layer_split_hnsw_ef64.csv."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = 64
OUT = 'new_plot'
DS = [('enron', 'Enron\n(d=1,369, N=95K)'), ('gist', 'GIST1M\n(d=960, N=1M)'),
      ('sift', 'SIFT1M\n(d=128, N=1M)')]
SYS = [('HNSW', 'HNSW_stock', 'vanilla HNSW', 'HNSW_v2_filter', 'Upper layers', '20_layer_split_hnsw'),
       ('SHG', 'SHG_original', 'SHG as published', 'SHG_v2_filter', 'Upper layers + shortcut jump', '20b_layer_split_shg')]
C_UP, C_BASE = '#f29900', '#1f5fbf'
INK, MUTED = '#202124', '#5f6368'

rows = []
for k, _ in DS:
    d = pd.read_csv(f'bench_prune/stall5_{k}/stall.csv').set_index(['method', 'ef'])
    for fam, base, _, sol, _, _ in SYS:
        for role, m in (('published', base),):
            r = d.loc[(m, EF)]; up = r.upper_us
            rows.append(dict(dataset=k, system=fam, role=role, method=m, efSearch=EF,
                             upper_us=up, base_us=r.cold_base_us))
R = pd.DataFrame(rows)
R['total_us'] = R.upper_us + R.base_us
R['base_pct'] = 100 * R.base_us / R.total_us
R.round(2).to_csv(f'{OUT}/layer_split_ef{EF}.csv', index=False)

for fam, base, blab, sol, uplab, tag in SYS:
    S = R[R.system == fam]
    ymax = S.total_us.max()
    fig, a = plt.subplots(figsize=(10, 6), dpi=200)
    w = 0.55
    for xi, (k, _) in enumerate(DS):
        s = S[S.dataset == k].set_index('role')
        for j, role in enumerate(('published',)):
            r = s.loc[role]; x = xi
            a.bar(x, r.base_us, w, color=C_BASE, edgecolor='white', lw=1.5, label='Base layer' if xi == j == 0 else None)
            a.bar(x, r.upper_us, w, bottom=r.base_us, color=C_UP, edgecolor='white', lw=1.5, label=uplab if xi == j == 0 else None)
            a.text(x, r.base_us / 2, f'{r.base_pct:.0f}%', ha='center', va='center', color='white', fontsize=9.5, fontweight='bold')
            a.text(x, r.total_us + 0.012 * ymax, f'{r.total_us:.0f} us\n(upper {r.upper_us:.0f} us)', ha='center', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
    a.set_xticks(range(len(DS))); a.set_xticklabels([t for _, t in DS], fontsize=10.5, fontweight='bold')
    a.tick_params(axis='x', pad=8, length=0)
    a.set_ylabel('Full query time (us)', fontsize=11, fontweight='bold')
    a.set_ylim(0, ymax * 1.18); a.set_xlim(-0.6, len(DS) - 0.4)
    a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', color='#e8eaed'); a.set_axisbelow(True)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10.5, frameon=False)
    fig.suptitle(f'{blab[0].upper() + blab[1:]}, full query at efSearch {EF}: upper layers vs base layer', fontsize=15, fontweight='bold', color=INK)
    a.set_title('% in the bar = base-layer share  |  median of 5 runs per query, mean over queries  |  single thread, core 2',
                fontsize=9.5, color=MUTED, pad=12)
    fig.tight_layout(); fig.savefig(f'{OUT}/{tag}_ef{EF}.png', bbox_inches='tight'); plt.close(fig)
    print('saved', f'{OUT}/{tag}_ef{EF}.png')
print(R[['dataset', 'system', 'role', 'upper_us', 'base_us', 'total_us', 'base_pct']].round(1).to_string(index=False))
