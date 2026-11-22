"""Complete picture: FULL per-query latency (upper layers + shortcuts + base layer) and recall@20.
Reads bench_prune/out2_<dataset>/prune.csv (bench_prune2: run_prune2.sh). Writes to new_plot/:
  15_full_recall_vs_latency.png      recall vs mean per-query latency, efSearch 20..300
  16_full_speedup_vs_vanilla.png     speedup over the published system at each efSearch
                                     (HNSW + Pf_Filter vs vanilla HNSW, SHG + Pf_Filter vs SHG as published)
  17_full_vs_vanilla_hnsw_equal_recall.png  speedup over vanilla HNSW at EQUAL recall (interpolated)
  full_query_summary.csv
Latency: per query, median of P=5 timed runs; then mean over queries. Pinned to core 2, single thread.
Usage: python3 plot_full_query.py [dir prefix, default bench_prune/out2_]"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRE = sys.argv[1] if len(sys.argv) > 1 else 'bench_prune/out2_'
OUT = 'new_plot'; os.makedirs(OUT, exist_ok=True)
DS = [('enron', 'Enron (d=1,369, N=95K)'), ('gist', 'GIST1M (d=960, N=1M)'),
      ('sift', 'SIFT1M (d=128, N=1M)'), ('sift10m', 'SIFT10M (d=128, N=10M)')]
data = {k: pd.read_csv(f'{PRE}{k}/prune.csv') for k, _ in DS if os.path.exists(f'{PRE}{k}/prune.csv')}
DS = [(k, t) for k, t in DS if k in data]
INK, MUTED = '#202124', '#5f6368'
# method, label, colour, linestyle
CURVES = [('HNSW_stock', 'vanilla HNSW', '#8ab4f8', '--'), ('HNSW_v2_filter', 'HNSW + Pf_Filter', '#1f5fbf', '-'),
          ('SHG_original', 'SHG as published', '#f28b82', '--'), ('SHG_v2_filter', 'SHG + Pf_Filter', '#c5221f', '-'),
          ('SHG_noprune_v2_filter', 'SHG no pruning + Pf_Filter', '#e37400', ':')]

def grid(w=16, h=5.6):
    nr = (len(DS) + 1) // 2
    fig, ax = plt.subplots(nr, 2, figsize=(w, h * nr), dpi=200, squeeze=False)
    for a in ax.flat[len(DS):]: a.set_visible(False)
    return fig, ax.flat

def style(a):
    a.spines[['top', 'right']].set_visible(False); a.grid(color='#e8eaed'); a.set_axisbelow(True)

# ---------- 15: recall vs latency ----------
fig, axes = grid()
for a, (k, title) in zip(axes, DS):
    d = data[k]
    for m, lab, col, ls in CURVES:
        s = d[d.method == m].sort_values('ef')
        if len(s): a.plot(s.lat_mean_us, s.recall, color=col, ls=ls, marker='o', ms=4, lw=2, label=lab)
    a.set_xscale('log'); a.set_xlabel('Mean per-query latency (us, log)', fontweight='bold')
    a.set_ylabel('Recall@20', fontweight='bold'); a.set_title(title, fontweight='bold'); style(a)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, fontsize=10.5, frameon=False)
fig.suptitle('Full query: recall vs latency (efSearch 20-300; up and left is better)', fontsize=15, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(f'{OUT}/15_full_recall_vs_latency.png', bbox_inches='tight'); plt.close(fig)

# ---------- 16: speedup over the published system, per efSearch ----------
rows = []
fig, axes = grid()
for a, (k, title) in zip(axes, DS):
    d = data[k].set_index(['method', 'ef'])
    efs = sorted(data[k].ef.unique())
    for fam, base, m, col in (('HNSW', 'HNSW_stock', 'HNSW_v2_filter', '#1f5fbf'), ('SHG', 'SHG_original', 'SHG_v2_filter', '#c5221f')):
        sp = [d.loc[(base, e), 'lat_mean_us'] / d.loc[(m, e), 'lat_mean_us'] for e in efs]
        a.plot(efs, sp, color=col, marker='o', lw=2, label=f'{fam} + Pf_Filter vs {"vanilla HNSW" if fam == "HNSW" else "SHG as published"}')
        for e, v in zip(efs, sp):
            rows.append(dict(dataset=k, comparison=f'{fam}_v2_filter_vs_{base}', efSearch=e, speedup=v,
                             base_us=d.loc[(base, e), 'lat_mean_us'], method_us=d.loc[(m, e), 'lat_mean_us'],
                             base_recall=d.loc[(base, e), 'recall'], method_recall=d.loc[(m, e), 'recall']))
    a.axhline(1, color=INK, ls='--', lw=1.2)
    a.set_xscale('log'); a.set_xticks(efs); a.set_xticklabels(efs)
    a.set_xlabel('efSearch', fontweight='bold'); a.set_ylabel('Full-query speedup (x)', fontweight='bold')
    a.set_title(title, fontweight='bold'); style(a)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=10.5, frameon=False)
fig.suptitle('Full query: Pf_Filter speedup over the published system (dashed = published = 1.0; same recall, results identical)',
             fontsize=14, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(f'{OUT}/16_full_speedup_vs_vanilla.png', bbox_inches='tight'); plt.close(fig)

# ---------- 17: vs vanilla HNSW at equal recall ----------
def lat_at(d, m, r):
    """latency of method m at recall r, interpolated in log-latency along its efSearch curve; NaN if out of range"""
    s = d[d.method == m].sort_values('ef')
    rec, lat = s.recall.values, np.log(s.lat_mean_us.values)
    if r > rec.max(): return np.nan              # never reaches this recall
    if r < rec.min(): return -1.0                # already above it at the lowest efSearch
    rec_m = np.maximum.accumulate(rec)          # keep monotone for interpolation
    return float(np.exp(np.interp(r, rec_m, lat)))

TARGETS = [0.85, 0.90, 0.95]
fig, axes = grid(h=5.0)
for a, (k, title) in zip(axes, DS):
    d = data[k]; w = 0.2
    bars = [('HNSW_v2_filter', 'HNSW + Pf_Filter', '#1f5fbf'), ('SHG_original', 'SHG as published', '#f28b82'),
            ('SHG_v2_filter', 'SHG + Pf_Filter', '#c5221f'), ('SHG_noprune_v2_filter', 'SHG no pruning + Pf_Filter', '#e37400')]
    for i, (m, lab, col) in enumerate(bars):
        for j, r in enumerate(TARGETS):
            t0, t = lat_at(d, 'HNSW_stock', r), lat_at(d, m, r)
            ok = np.isfinite(t0) and np.isfinite(t) and t0 > 0 and t > 0
            v = t0 / t if ok else np.nan
            rows.append(dict(dataset=k, comparison=f'{m}_vs_HNSW_stock_at_recall', efSearch=np.nan, recall_target=r, speedup=v))
            x = j + (i - 1.5) * w
            if ok:
                a.bar(x, v, w * 0.92, color=col, label=lab if j == 0 else None)
                a.text(x, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, color=INK, fontweight='bold')
            else:
                why = 'never\nreaches' if (np.isnan(t) or np.isnan(t0)) else 'n/a'
                a.bar(x, 0, w * 0.92, color=col, label=lab if j == 0 else None)
                a.text(x, 0.03, why, ha='center', va='bottom', fontsize=6.5, color=MUTED)
    a.axhline(1, color=INK, ls='--', lw=1.2)
    a.set_xticks(range(len(TARGETS))); a.set_xticklabels([f'recall {r:.2f}' for r in TARGETS], fontweight='bold')
    a.set_ylabel('Speedup vs vanilla HNSW (x)', fontweight='bold'); a.set_title(title, fontweight='bold'); style(a); a.grid(axis='x', visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=10.5, frameon=False)
fig.suptitle('Full query at EQUAL recall, relative to vanilla HNSW (dashed = 1.0)', fontsize=15, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(f'{OUT}/17_full_vs_vanilla_hnsw_equal_recall.png', bbox_inches='tight'); plt.close(fig)

pd.DataFrame(rows).round(4).to_csv(f'{OUT}/full_query_summary.csv', index=False)
print('saved 15, 16, 17 and full_query_summary.csv for', [k for k, _ in DS])
