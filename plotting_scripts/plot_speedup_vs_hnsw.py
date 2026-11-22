import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('plots', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

enron_csv = 'results/data/hnsw_vs_shg_comparison.csv'
sift_csv = 'results/data/hnsw_vs_shg_comparison_sift.csv'

df_enron = pd.read_csv(enron_csv)
df_sift = pd.read_csv(sift_csv)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

styles = {
    'Vanilla_HNSW': {
        'label': 'HNSW (Baseline = 1.00x)',
        'color': '#1f77b4',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': 2.2,
        'markersize': 6
    },
    'Baseline_SHG': {
        'label': 'SHG (normal)',
        'color': '#d62728',
        'marker': 's',
        'linestyle': '-',
        'linewidth': 2.0,
        'markersize': 6
    },
    'Step1_SkipPrefetch': {
        'label': 'SHG with Idea 1',
        'color': '#ff7f0e',
        'marker': '^',
        'linestyle': '-.',
        'linewidth': 2.0,
        'markersize': 6
    },
    'Step2_SkipLookahead': {
        'label': 'SHG + Idea 2',
        'color': '#2ca02c',
        'marker': 'D',
        'linestyle': '-',
        'linewidth': 2.5,
        'markersize': 7
    }
}

speedup_records = []

# Process Enron
hnsw_e = df_enron[df_enron['method'] == 'Vanilla_HNSW'].sort_values('recall')
shg_e = df_enron[df_enron['method'] == 'Baseline_SHG'].sort_values('recall')
s1_e = df_enron[df_enron['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
s2_e = df_enron[df_enron['method'] == 'Step2_SkipLookahead'].sort_values('recall')

recs_e = np.linspace(0.83, 0.99, 25)
ht_e = np.interp(recs_e, hnsw_e['recall'], hnsw_e['latency_ms'])
bt_e = np.interp(recs_e, shg_e['recall'], shg_e['latency_ms'])
s1t_e = np.interp(recs_e, s1_e['recall'], s1_e['latency_ms'])
s2t_e = np.interp(recs_e, s2_e['recall'], s2_e['latency_ms'])

ax1.axhline(1.0, color='#1f77b4', linestyle='--', linewidth=2.2, label='HNSW (Baseline = 1.00x)')
ax1.plot(recs_e, ht_e / bt_e, 's-', color='#d62728', label='SHG (normal)', linewidth=2.0, markersize=6)
ax1.plot(recs_e, ht_e / s1t_e, '^-.', color='#ff7f0e', label='SHG with Idea 1', linewidth=2.0, markersize=6)
ax1.plot(recs_e, ht_e / s2t_e, 'D-', color='#2ca02c', label='SHG + Idea 2', linewidth=2.5, markersize=7)

ax1.set_title('Enron Dataset (d = 1,369, N = 94,987)\nSpeedup Factor vs. HNSW Baseline across Target Recall', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
ax1.set_ylabel('Speedup Factor vs. HNSW (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.set_xlim(0.82, 1.00)
ax1.set_ylim(0.55, 1.15)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.legend(loc='lower left', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Process SIFT
hnsw_s = df_sift[df_sift['method'] == 'Vanilla_HNSW'].sort_values('recall')
shg_s = df_sift[df_sift['method'] == 'Baseline_SHG'].sort_values('recall')
s1_s = df_sift[df_sift['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
s2_s = df_sift[df_sift['method'] == 'Step2_SkipLookahead'].sort_values('recall')

recs_s = np.linspace(0.80, 0.998, 25)
ht_s = np.interp(recs_s, hnsw_s['recall'], hnsw_s['latency_ms'])
bt_s = np.interp(recs_s, shg_s['recall'], shg_s['latency_ms'])
s1t_s = np.interp(recs_s, s1_s['recall'], s1_s['latency_ms'])
s2t_s = np.interp(recs_s, s2_s['recall'], s2_s['latency_ms'])

ax2.axhline(1.0, color='#1f77b4', linestyle='--', linewidth=2.2, label='HNSW (Baseline = 1.00x)')
ax2.plot(recs_s, ht_s / bt_s, 's-', color='#d62728', label='SHG (normal)', linewidth=2.0, markersize=6)
ax2.plot(recs_s, ht_s / s1t_s, '^-.', color='#ff7f0e', label='SHG with Idea 1', linewidth=2.0, markersize=6)
ax2.plot(recs_s, ht_s / s2t_s, 'D-', color='#2ca02c', label='SHG + Idea 2', linewidth=2.5, markersize=7)

ax2.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nSpeedup Factor vs. HNSW Baseline across Target Recall', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
ax2.set_ylabel('Speedup Factor vs. HNSW (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_xlim(0.79, 1.005)
ax2.set_ylim(0.30, 1.10)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.legend(loc='lower left', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('plots/speedup_vs_hnsw.png', dpi=300)
fig.savefig('results/plots/speedup_vs_hnsw.png', dpi=300)
print("Saved plots/speedup_vs_hnsw.png and results/plots/speedup_vs_hnsw.png")

# Save speedup summary table to CSV
for ef in [20, 24, 32, 48, 64, 96, 128, 192, 256, 512]:
    he = df_enron[(df_enron['method']=='Vanilla_HNSW') & (df_enron['ef']==ef)]['latency_us'].values[0]
    be = df_enron[(df_enron['method']=='Baseline_SHG') & (df_enron['ef']==ef)]['latency_us'].values[0]
    s1e = df_enron[(df_enron['method']=='Step1_SkipPrefetch') & (df_enron['ef']==ef)]['latency_us'].values[0]
    s2e = df_enron[(df_enron['method']=='Step2_SkipLookahead') & (df_enron['ef']==ef)]['latency_us'].values[0]

    hs = df_sift[(df_sift['method']=='Vanilla_HNSW') & (df_sift['ef']==ef)]['latency_us'].values[0]
    bs = df_sift[(df_sift['method']=='Baseline_SHG') & (df_sift['ef']==ef)]['latency_us'].values[0]
    s1s = df_sift[(df_sift['method']=='Step1_SkipPrefetch') & (df_sift['ef']==ef)]['latency_us'].values[0]
    s2s = df_sift[(df_sift['method']=='Step2_SkipLookahead') & (df_sift['ef']==ef)]['latency_us'].values[0]

    speedup_records.append({
        'ef': ef,
        'Enron_HNSW_Base': 1.0,
        'Enron_SHG_Normal_Speedup': round(he / be, 3),
        'Enron_Idea1_Speedup': round(he / s1e, 3),
        'Enron_Idea2_Speedup': round(he / s2e, 3),
        'SIFT_HNSW_Base': 1.0,
        'SIFT_SHG_Normal_Speedup': round(hs / bs, 3),
        'SIFT_Idea1_Speedup': round(hs / s1s, 3),
        'SIFT_Idea2_Speedup': round(hs / s2s, 3)
    })

pd.DataFrame(speedup_records).to_csv('results/data/speedup_summary_vs_hnsw.csv', index=False)
print("Saved results/data/speedup_summary_vs_hnsw.csv")
