"""SIFT10M HNSW vs SHG plots -> c_plot/.

Usage: python3 plot_c_plots.py <results10m_hnsw.csv> <results10m_shg.csv> \
                               <perquery10m_hnsw.csv> <perquery10m_shg.csv> [out_dir]
"""
import sys, os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Fixed categorical order (validated: CVD/normal-vision pass; low-contrast slots get direct labels)
STYLE = {
    "HNSW":               dict(color="#2a78d6", marker="o", label="HNSW"),
    "SHG_original":       dict(color="#eb6834", marker="s", label="SHG (authors' code)"),
    "SHG_step2_prefetch": dict(color="#1baf7a", marker="^", label="SHG + Step 2 prefetch"),
    "SHG_no_base_prune":  dict(color="#eda100", marker="D", label="SHG, base pruning off"),
}
ORDER = list(STYLE)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})


def dodge_labels(ax, items, min_px=15):
    """items: [(x, y, text)]; place direct labels at line ends, pushed apart vertically."""
    if not items:
        return
    ax.figure.canvas.draw()
    pts = [(ax.transData.transform((x, y)), t) for x, y, t in items]
    pts.sort(key=lambda p: p[0][1])
    placed = []
    for (px, py), t in pts:
        ty = max(py, placed[-1] + min_px) if placed else py
        placed.append(ty)
        ax.annotate(t, xy=ax.transData.inverted().transform((px, py)), xytext=(7, ty - py),
                    textcoords="offset pixels", fontsize=9, color=INK2,
                    va="center", annotation_clip=False)


def lines(ax, df, x, y, label_end=True):
    ends = []
    for m in ORDER:
        d = df[df.method == m].sort_values("ef")
        if d.empty:
            continue
        s = STYLE[m]
        ax.plot(d[x], d[y], color=s["color"], marker=s["marker"], label=s["label"],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        if label_end:  # direct label at the high-ef end (text in ink, not series color)
            ends.append((d[x].iloc[-1], d[y].iloc[-1], s["label"]))
    ax._pending_labels = ends


TIMING = os.environ.get("TIMING", "1") != "0"   # TIMING=0: skip latency-based plots (noisy run)
TIMING_PLOTS = {"01_recall_vs_latency.png", "02_qps_vs_recall.png", "03_speedup_vs_hnsw_iso_recall.png",
                "04_p99_latency_vs_recall.png", "09_paper_subset_protocol.png"}


def save(fig, out, name):
    if not TIMING and name in TIMING_PLOTS:
        plt.close(fig)
        print("skipped (TIMING=0)", name)
        return
    fig.tight_layout()
    for ax in fig.axes:
        dodge_labels(ax, getattr(ax, "_pending_labels", []))
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=160)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


def iso_recall_speedup(df, targets):
    """Latency of HNSW / latency of method, both interpolated at the same recall."""
    rows = []
    h = df[df.method == "HNSW"].sort_values("recall")
    for m in ORDER[1:]:
        d = df[df.method == m].sort_values("recall")
        if d.empty:
            continue
        for r in targets:
            if not (h.recall.min() <= r <= h.recall.max() and d.recall.min() <= r <= d.recall.max()):
                continue
            lh = np.interp(r, h.recall, h.lat_mean_us)
            lm = np.interp(r, d.recall, d.lat_mean_us)
            rows.append(dict(method=m, recall=r, hnsw_us=lh, method_us=lm, speedup=lh / lm))
    return pd.DataFrame(rows, columns=["method", "recall", "hnsw_us", "method_us", "speedup"])


def main():
    a = sys.argv[1:]
    out = a[4] if len(a) > 4 else "c_plot"
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    for f in a[:4]:
        shutil.copy(f, os.path.join(out, "data", os.path.basename(f)))
    df = pd.concat([pd.read_csv(a[0]), pd.read_csv(a[1])], ignore_index=True)
    pq = pd.concat([pd.read_csv(a[2]), pd.read_csv(a[3])], ignore_index=True)
    df["upper_pct"] = df.upper_frac * 100

    # 1. Recall vs latency (Pareto)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lines(ax, df, "lat_mean_us", "recall")
    ax.set(xlabel="Mean query latency (µs, log scale)", ylabel="Recall@20",
           title="SIFT10M: recall vs latency (ef 20–300, k=20)", xscale="log")
    ax.legend(loc="lower right")
    save(fig, out, "01_recall_vs_latency.png")

    # 2. QPS vs recall (ann-benchmarks style)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lines(ax, df, "recall", "qps")
    ax.set(xlabel="Recall@20", ylabel="Queries per second (log scale, 1 core)",
           title="SIFT10M: throughput vs recall (up and to the right is better)", yscale="log")
    ax.legend(loc="lower left")
    save(fig, out, "02_qps_vs_recall.png")

    # 3. Iso-recall speedup over HNSW
    targets = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98]
    sp = iso_recall_speedup(df, targets)
    sp.to_csv(os.path.join(out, "data", "iso_recall_speedup.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in ORDER[1:]:
        d = sp[sp.method == m]
        if d.empty:
            continue
        s = STYLE[m]
        ax.plot(d.recall, d.speedup, color=s["color"], marker=s["marker"], label=s["label"],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{d.speedup.iloc[-1]:.2f}×", (d.recall.iloc[-1], d.speedup.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK2, va="center")
    ax.axhline(1.0, color=INK2, linestyle=":", linewidth=1.2)
    ax.annotate("HNSW = 1.00×", (targets[0], 1.0), xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=INK2)
    ax.axhline(1.5, color=INK2, linestyle="--", linewidth=1)
    ax.annotate("paper's claim ≈ 1.5×", (targets[0], 1.5), xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=INK2)
    ax.set(xlabel="Target Recall@20", ylabel="Speedup vs HNSW at equal recall (×)",
           title="SIFT10M: speedup over HNSW at the same recall (>1 means faster than HNSW)")
    ax.set_ylim(0, max(1.7, sp.speedup.max() * 1.15) if not sp.empty else 1.7)
    ax.legend(loc="lower right")
    save(fig, out, "03_speedup_vs_hnsw_iso_recall.png")

    # 4. Tail latency p99 vs recall
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lines(ax, df, "recall", "lat_p99_us")
    ax.set(xlabel="Recall@20", ylabel="p99 latency (µs, log scale)",
           title="SIFT10M: tail latency (p99) vs recall (lower is better)", yscale="log")
    ax.legend(loc="upper left")
    save(fig, out, "04_p99_latency_vs_recall.png")

    # 5. Distance computations per query
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lines(ax, df, "recall", "dist_per_q")
    ax.set(xlabel="Recall@20", ylabel="Full-precision distance computations per query",
           title="SIFT10M: work per query vs recall (lower is better)")
    ax.legend(loc="upper left")
    save(fig, out, "05_distance_computations_vs_recall.png")

    # 6. Share of query time in upper levels vs paper's 18–42%
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhspan(18, 42, color="#cde2fb", alpha=0.6, linewidth=0)
    ax.annotate("paper Fig. 7: HNSW upper levels = 18–42% of query time", (20, 40),
                fontsize=9, color=INK2, va="top")
    lines(ax, df, "ef", "upper_pct")
    ax.set(xlabel="efSearch (log scale)", ylabel="Time in upper levels (% of query time)",
           title="SIFT10M: how much time is there to save in the upper levels?", xscale="log")
    ax.set_xticks([20, 30, 40, 60, 80, 120, 200, 300]); ax.set_xticklabels([20, 30, 40, 60, 80, 120, 200, 300])
    ax.set_ylim(0, 45)
    ax.legend(loc="center right")
    save(fig, out, "06_upper_level_time_share.png")

    # 7 & 8. Build time and memory (two separate charts; one axis each)
    b = df.groupby("method").first().reindex([m for m in ["HNSW", "SHG_original"] if m in set(df.method)])
    names = ["HNSW", "SHG"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = ax.bar([0], [b.build_s.iloc[0] / 60], width=0.5, color=STYLE["HNSW"]["color"])
    if len(b) > 1:
        ax.bar([1], [b.build_s.iloc[1] / 60], width=0.5, color=STYLE["SHG_original"]["color"], label="graph")
        ax.bar([1], [b.shortcut_s.iloc[1] / 60], bottom=[b.build_s.iloc[1] / 60 + 0.02 * b.build_s.max() / 60],
               width=0.5, color="#f4a582", label="shortcuts")
    for i, (bs, ss) in enumerate(zip(b.build_s, b.shortcut_s)):
        ax.annotate(f"{(bs + ss) / 60:.1f} min", (i, (bs + ss) / 60), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(b))); ax.set_xticklabels(["HNSW\n(16 threads)", "SHG\n(1 thread — code not thread-safe)"][:len(b)])
    ax.set(ylabel="Build time (minutes)", title="SIFT10M: index build time")
    ax.grid(axis="x", visible=False)
    if len(b) > 1:
        ax.legend(loc="upper left")
    save(fig, out, "07_build_time.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(b.index):
        ax.bar([i], [b.index_mb.iloc[i] / 1024], width=0.5, color=STYLE[m]["color"])
        ax.annotate(f"{b.index_mb.iloc[i] / 1024:.2f} GB", (i, b.index_mb.iloc[i] / 1024), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(b))); ax.set_xticklabels(names[:len(b)])
    ax.set(ylabel="Index memory (GB)", title="SIFT10M: index memory (vectors + graph + SHG extras)")
    ax.grid(axis="x", visible=False)
    save(fig, out, "08_index_memory.png")

    # 9. Paper's protocol: one run (ef = k = 20), queries sorted by recall, 10 equal subsets
    ef0 = 20
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub_rows = []
    for m in ORDER:
        d = pq[(pq.method == m) & (pq.ef == ef0)].sort_values(["recall", "lat_us"]).reset_index(drop=True)
        if d.empty:
            continue
        d["bin"] = np.arange(len(d)) * 10 // len(d)
        g = d.groupby("bin").agg(recall=("recall", "mean"), lat_us=("lat_us", "mean")).reset_index()
        g["method"] = m
        sub_rows.append(g)
        s = STYLE[m]
        ax.plot(g.lat_us, g.recall, color=s["color"], marker=s["marker"], label=s["label"],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    if sub_rows:
        pd.concat(sub_rows).to_csv(os.path.join(out, "data", "paper_subset_protocol_ef20.csv"), index=False)
    ax.set(xlabel="Mean latency of subset (µs)", ylabel="Mean Recall@20 of subset",
           title=f"SIFT10M: paper's protocol (ef={ef0}, queries sorted by recall, 10 subsets)")
    ax.legend(loc="lower right")
    save(fig, out, "09_paper_subset_protocol.png")

    # 10. Work at equal recall: full-precision distance computations relative to HNSW
    h = df[df.method == "HNSW"].sort_values("recall")
    rows = []
    for m in ORDER[1:]:
        d = df[df.method == m].sort_values("recall")
        for r in targets:
            if len(d) and h.recall.min() <= r <= h.recall.max() and d.recall.min() <= r <= d.recall.max():
                dh, dm = np.interp(r, h.recall, h.dist_per_q), np.interp(r, d.recall, d.dist_per_q)
                rows.append(dict(method=m, recall=r, hnsw_dist=dh, method_dist=dm, ratio=dh / dm))
    wr = pd.DataFrame(rows, columns=["method", "recall", "hnsw_dist", "method_dist", "ratio"])
    wr.to_csv(os.path.join(out, "data", "iso_recall_distance_ratio.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in ["SHG_original", "SHG_no_base_prune"]:
        d = wr[wr.method == m]
        if d.empty:
            continue
        s = STYLE[m]
        ax.plot(d.recall, d.ratio, color=s["color"], marker=s["marker"], label=s["label"],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{d.ratio.iloc[-1]:.2f}×", (d.recall.iloc[-1], d.ratio.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=INK2, va="center")
    ax.axhline(1.0, color=INK2, linestyle=":", linewidth=1.2)
    ax.annotate("same work as HNSW", (targets[0], 1.0), xytext=(0, 5), textcoords="offset points", fontsize=9, color=INK2)
    ax.set(xlabel="Target Recall@20", ylabel="HNSW distance computations ÷ SHG distance computations",
           title="SIFT10M: work saved at equal recall (>1 means SHG does less work)")
    ax.set_ylim(0.6, 1.4)
    ax.legend(loc="lower right")
    save(fig, out, "10_work_ratio_vs_hnsw_iso_recall.png")

    # 11. Upper levels actually traversed per query (deterministic)
    lv = df[df.ef == df.ef.min()].set_index("method")
    ml = int(os.environ.get("MAXLEVEL", "4"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [("HNSW", ml, STYLE["HNSW"]["color"])]
    if "SHG_original" in lv.index:
        vals.append(("SHG", ml - lv.loc["SHG_original", "lvls_skipped_per_q"], STYLE["SHG_original"]["color"]))
    for i, (n_, v, c) in enumerate(vals):
        ax.bar([i], [v], width=0.5, color=c)
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([v[0] for v in vals])
    ax.set(ylabel="Upper levels searched per query", title=f"SIFT10M: upper levels searched (of {ml}) — shortcuts at work")
    ax.grid(axis="x", visible=False)
    save(fig, out, "11_upper_levels_searched.png")

    # Summary table (table view for accessibility + low-contrast relief)
    cols = ["method", "ef", "recall", "lat_mean_us", "lat_p50_us", "lat_p99_us", "qps", "dist_per_q",
            "upper_pct", "skip_q_frac", "lvls_skipped_per_q"]
    if not TIMING:
        cols = [c for c in cols if c not in ("lat_mean_us", "lat_p50_us", "lat_p99_us", "qps")]
    t = df[cols].sort_values(["method", "ef"])
    with open(os.path.join(out, "summary_table.md"), "w") as f:
        f.write("# SIFT10M HNSW vs SHG: all metrics\n\n")
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |\n")
        if not wr.empty:
            f.write("\n## Distance computations at equal recall (HNSW / SHG; >1 = SHG does less work)\n\n| method | recall | HNSW dist/q | SHG dist/q | ratio |\n|---|---|---|---|---|\n")
            for _, r in wr.iterrows():
                f.write(f"| {r.method} | {r.recall:.2f} | {r.hnsw_dist:.0f} | {r.method_dist:.0f} | {r.ratio:.2f}× |\n")
        if TIMING and not sp.empty:
            f.write("\n## Speedup vs HNSW at equal recall\n\n| method | recall | HNSW µs | method µs | speedup |\n|---|---|---|---|---|\n")
            for _, r in sp.iterrows():
                f.write(f"| {r.method} | {r.recall:.2f} | {r.hnsw_us:.1f} | {r.method_us:.1f} | {r.speedup:.2f}× |\n")
    print("wrote", os.path.join(out, "summary_table.md"))


if __name__ == "__main__":
    main()
