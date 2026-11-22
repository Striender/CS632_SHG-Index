"""Pf_Filter design as a numbered flow of one round of the HNSW base-layer search (no title: the
LaTeX caption names it). Blue boxes = prefetches Pf_Filter adds; white boxes = normal HNSW search work.
Order follows hnswlib_patched/heds.h searchBaseLayerST: pop -> on_candidate_queue_next -> v2_prologue
-> per neighbour: v2_step (flag j+4, filtered vector j+2) -> visited check + distance -> push (+ link-list prefetch).
Writes new_plot/pf_filter_design.pdf and .png."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 780, 720
INK, QUIET, EDGE = '#202124', '#5f6368', '#80868b'
PF_FILL, PF_EDGE = '#e3ecf9', '#1f5fbf'
fig = plt.figure(figsize=(W / 80, H / 80))
a = fig.add_axes([0, 0, 1, 1]); a.set_xlim(0, W); a.set_ylim(H, 0); a.axis('off')
FS = 0.78

def txt(x, y, s, size=13, bold=False, color=INK, ha='left', rot=0):
    a.text(x, y, s, fontsize=size * FS, fontweight='bold' if bold else 'normal', color=color, ha=ha,
           va='baseline' if rot == 0 else 'center', rotation=rot)
def rbox(x, y, w, h, pf, ls='-', lw=None):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=8',
                               fc=PF_FILL if pf else 'white', ec=PF_EDGE if pf else EDGE,
                               lw=lw or (1.6 if pf else 1.1), ls=ls))
def step(n, x, y, w, name, sub, pf, h=52):
    rbox(x, y, w, h, pf)
    a.add_patch(plt.Circle((x + 22, y + h / 2), 12, fc=PF_EDGE if pf else INK, ec='none'))
    a.text(x + 22, y + h / 2 + 0.5, str(n), fontsize=11 * FS, fontweight='bold', color='white', ha='center', va='center')
    txt(x + 44, y + 23, name, bold=True); txt(x + 44, y + 40, sub, 11.5, color=QUIET)
def arrow(pts, label=None, lx=0, ly=0, ha='left', rot=0):
    xs, ys = zip(*pts)
    a.plot(xs, ys, color=EDGE, lw=1.2)
    a.annotate('', xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.2, mutation_scale=11, shrinkA=0, shrinkB=0))
    if label: txt(lx, ly, label, 11.5, color=QUIET, ha=ha, rot=rot)

X, BW = 150, 440; CX = X + BW / 2
step(1, X, 20, BW, 'Pop the best candidate from the queue', 'read its neighbour list', False)
step(2, X, 96, BW, "Prefetch the next candidate's neighbour list", 'ready when this round ends', True)
step(3, X, 172, BW, 'Warm-up: prefetch flags 1-4 and vectors 1-2', 'so the first neighbours do not wait', True)
arrow([(CX, 72), (CX, 96)]); arrow([(CX, 148), (CX, 172)]); arrow([(CX, 224), (CX, 262)])

# loop over the neighbour list
rbox(110, 250, 620, 418, False, ls=(0, (4, 3)), lw=1.0)
txt(126, 244, 'For each neighbour j of the list', 11.5, bold=True, color=QUIET)
step(4, X, 262, BW, 'Prefetch the visited flag of neighbour j+4', '1 byte, cheap to fetch early', True)
step(5, X, 338, BW, 'Is neighbour j+2 already visited?', 'its flag was prefetched 2 steps ago: in cache', True)
step(6, X, 414, BW, "No: prefetch neighbour j+2's vector", 'first 3 cache lines, the filter skips visited nodes', True)
rbox(612, 338, 104, 52, True); txt(664, 361, 'Yes: skip', bold=True, ha='center'); txt(664, 378, 'no fetch', 11.5, color=QUIET, ha='center')
step(7, X, 490, BW, 'Process neighbour j: if new, compute distance', 'its vector was prefetched 2 steps ago', False)
step(8, X, 566, BW, 'If close enough, push it into the queue', "and prefetch the queue's best node's neighbour list", False)
arrow([(CX, 314), (CX, 338)]); arrow([(CX, 390), (CX, 414)], 'no', CX + 8, 406)
arrow([(X + BW, 364), (612, 364)], 'yes', 601, 357, ha='center')
arrow([(CX, 466), (CX, 490)]); arrow([(664, 390), (664, 516), (X + BW, 516)])
arrow([(CX, 542), (CX, 566)])
arrow([(X, 592), (128, 592), (128, 288), (X, 288)], 'next j', 120, 440, ha='center', rot=90)
arrow([(CX, 618), (CX, 668)])
txt(CX + 8, 650, 'after the last neighbour', 11.5, color=QUIET)
arrow([(CX, 668), (CX, 684), (752, 684), (752, 46), (X + BW, 46)], 'next round', 762, 360, ha='center', rot=270)

# legend
rbox(150, 696, 22, 14, True); txt(180, 708, 'added by Pf_Filter', 11.5, color=INK)
rbox(400, 696, 22, 14, False); txt(430, 708, 'normal HNSW search work', 11.5, color=INK)

for ext in ('pdf', 'png'):
    fig.savefig(f'new_plot/pf_filter_design.{ext}', dpi=250, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print('saved new_plot/pf_filter_design.pdf and .png')
