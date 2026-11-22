"""Motivation: where the CPU's time goes per query in the PUBLISHED systems, all datasets in one plot.
One figure for HNSW (vanilla hnswlib) and one for SHG (as published). Base layer, efSearch EF.
Reads bench_prune/stall4_<dataset>/stall.csv. Per query, cold_base_us is split in the ratio
  idle     = memory stall (cold run - warm rerun): CPU waiting on DRAM
  compute  = distance computation (microbenchmark, cache-resident)
  overhead = warm rerun - compute: heap, visited list, neighbour scan, pruning checks, L2/L3 hits
Writes new_plot/18a_idle_time_hnsw_ef<EF>.png, 18b_idle_time_shg_ef<EF>.png, idle_time_ef<EF>.csv.
Usage: python3 plot_idle_time.py [efSearch]"""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = int(sys.argv[1]) if len(sys.argv) > 1 else 64
OUT = 'new_plot'
DS = [('enron', 'Enron\n(d=1,369, N=95K)'), ('gist', 'GIST1M\n(d=960, N=1M)'),
      ('sift', 'SIFT1M\n(d=128, N=1M)'), ('sift10m', 'SIFT10M\n(d=128, N=10M)')]
SYS = [('HNSW', 'vanilla HNSW', 'HNSW_stock', '18a_idle_time_hnsw'),
       ('SHG', 'SHG as published', 'SHG_original', '18b_idle_time_shg')]
PARTS = [('idle', 'CPU idle (waiting on memory)', '#d62728'),
         ('compute', 'Distance compute', '#2ca02c'),
         ('overhead', 'Search overhead (heap, visited list, neighbour scan' + ', pruning' + ', L2/L3 hits)', '#1f77b4')]
INK, MUTED = '#202124', '#5f6368'

rows = []
for k, _ in DS:
    d = pd.read_csv(f'bench_prune/stall4_{k}/stall.csv').set_index(['method', 'ef'])
    for fam, lab, m, _ in SYS:
        r = d.loc[(m, EF)]
        parts = dict(idle=max(r.stall_cyc_per_eval, 0), compute=r.compute_cyc_per_eval, overhead=r.overhead_cyc_per_eval)
        tot = sum(parts.values())
        row = dict(dataset=k, system=fam, method=m, efSearch=EF, total_us=r.cold_base_us)
        for p, v in parts.items():
            row[f'{p}_pct'] = 100 * v / tot; row[f'{p}_us'] = r.cold_base_us * v / tot
        rows.append(row)
R = pd.DataFrame(rows)
R.round(2).to_csv(f'{OUT}/idle_time_ef{EF}.csv', index=False)

for fam, lab, m, tag in SYS:
    s = R[R.system == fam].reset_index(drop=True)
    fig, a = plt.subplots(figsize=(10, 6.2), dpi=200)
    x = range(len(s)); bottom = [0.0] * len(s)
    for p, plab, col in PARTS:
        if fam == 'HNSW': plab = plab.replace(', pruning', '')
        v = s[f'{p}_pct'].values
        a.bar(x, v, 0.6, bottom=bottom, color=col, edgecolor='white', lw=2, label=plab)
        for xi in x:
            if v[xi] > 6:
                a.text(xi, bottom[xi] + v[xi] / 2, f'{v[xi]:.0f}%\n({s[f"{p}_us"][xi]:.0f} us)', ha='center', va='center',
                       color='white', fontsize=10, fontweight='bold')
            elif v[xi] > 0:
                a.text(xi + 0.32, bottom[xi] + v[xi] / 2, f'{v[xi]:.0f}%', ha='left', va='center', fontsize=9, color=INK, fontweight='bold')
        bottom = [b + vv for b, vv in zip(bottom, v)]
    for xi in x:
        a.text(xi, 101.5, f'{s.total_us[xi]:.0f} us / query', ha='center', va='bottom', fontsize=10, color=INK, fontweight='bold')
    a.set_xticks(list(x)); a.set_xticklabels([t for _, t in DS], fontsize=10.5, fontweight='bold')
    a.set_ylabel('Share of base-layer time per query (%)', fontsize=11, fontweight='bold')
    a.set_ylim(0, 108); a.set_xlim(-0.55, len(s) - 0.3)
    a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', color='#e8eaed'); a.set_axisbelow(True)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=10, frameon=False)
    fig.suptitle(f'{lab}: where the CPU time goes per query (efSearch = {EF})', fontsize=15, fontweight='bold', color=INK)
    a.set_title('Base layer, measured: single thread pinned to core 2, cold cache; M = 48, efConstruction = 80, K = 20',
                fontsize=9.5, color=MUTED, pad=22)
    fig.tight_layout()
    fig.savefig(f'{OUT}/{tag}_ef{EF}.png', bbox_inches='tight'); plt.close(fig)
    print('saved', f'{OUT}/{tag}_ef{EF}.png')
print(R[['dataset', 'system', 'total_us', 'idle_pct', 'compute_pct', 'overhead_pct']].round(1).to_string(index=False))
