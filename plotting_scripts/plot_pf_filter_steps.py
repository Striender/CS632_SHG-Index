"""Pf_Filter design as a step-by-step figure in the style of a paper diagram (real structures + numbered steps).
Same steps as plot_pf_filter_design.py. Writes new_plot/pf_filter_steps.pdf and .png."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})
W, H = 100, 72
fig = plt.figure(figsize=(7.5, 7.5 * H / W))
a = fig.add_axes([0, 0, 1, 1]); a.set_xlim(0, W); a.set_ylim(0, H); a.axis('off')
FS = 10.5
DARK, PANEL, GREY = '#9e9e9e', '#ececec', '#7a7a7a'

def box(x, y, w, h, t='', fc='white', fs=FS, tc='black', lw=1.0, **k):
    a.add_patch(Rectangle((x, y), w, h, fc=fc, ec='black', lw=lw, **k))
    if t: a.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=tc)

def step(x, y, n):
    a.add_patch(Circle((x, y), 1.35, fc='black', ec='black', zorder=5))
    a.text(x, y - 0.05, str(n), ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=6)

def arrow(p, q, dashed=False, color='black', **k):
    a.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=1.0, color=color,
                                linestyle=(0, (3, 2)) if dashed else '-', shrinkA=0, shrinkB=0, **k))

def tag(x, y, t):
    a.text(x, y, t, ha='left', va='center', fontsize=FS, bbox=dict(fc='white', ec='black', lw=0.8, pad=2.2))

# ---------------- candidate queue ----------------
QX, QW, RH = 2, 12, 5
for i, (t, fc) in enumerate([('c$_1$ (best)', DARK), ('c$_2$', 'white'), ('c$_3$', 'white'), ('$\\vdots$', 'white'), ('c$_n$', 'white')]):
    box(QX, 61 - i * RH, QW, RH, t, fc=fc)
a.text(QX + QW / 2, 38.6, 'Candidate queue\n(best on top)', ha='center', va='top', fontsize=FS)

# ---------------- panel 1: start of a step ----------------
a.add_patch(Rectangle((21, 48.5), 78, 22, fc=PANEL, ec=GREY, lw=0.8))
tag(22.5, 68.4, 'Start of a step')
step(25, 63.5, 1); a.text(27.3, 63.5, 'Pop the best candidate c$_1$ and read its neighbour list', va='center', fontsize=FS)
arrow((QX + QW, 63.5), (23.6, 63.5))
step(25, 57.5, 2); a.text(27.3, 57.5, 'Prefetch the neighbour list of the next candidate c$_2$', va='center', fontsize=FS)
arrow((QX + QW, 58.5), (23.6, 57.6), dashed=True)
step(25, 51.5, 3); a.text(27.3, 51.5, 'Warm-up: prefetch the flags of neighbours 1-4 and the vectors of 1-2',
                          va='center', fontsize=FS)

# ---------------- panel 2: per neighbour j ----------------
a.add_patch(Rectangle((21, 1), 78, 45.5, fc=PANEL, ec=GREY, lw=0.8))
tag(22.5, 44.2, 'For each neighbour j  (look-ahead D = 2)')
CX, CW = 36, 8
names = ['j', 'j+1', 'j+2', 'j+3', 'j+4', 'j+5', '$\\cdots$']
flags = ['0', '1', '0', '1', '?', '', '']
for i, (n, f) in enumerate(zip(names, flags)):
    box(CX + i * CW, 32, CW, 5, f, fc='white')
    box(CX + i * CW, 27, CW, 5, n, fc=DARK if i == 0 else 'white')
a.text(CX - 1, 34.5, 'visited flag', ha='right', va='center', fontsize=FS)
a.text(CX - 1, 29.5, 'neighbour', ha='right', va='center', fontsize=FS)
cx = lambda i: CX + i * CW + CW / 2

# 4: prefetch flag of j+4
step(cx(4), 41.2, 4); a.text(cx(4) + 2.3, 41.2, 'Prefetch the flag of j+4', va='center', fontsize=FS)
arrow((cx(4), 39.85), (cx(4), 37.05), dashed=True)
# 5: is j+2 new?
step(cx(2), 20.5, 5); a.text(cx(2) + 2.3, 20.5, 'Is j+2 new?\n(its flag is in cache)', va='center', fontsize=FS)
arrow((cx(2), 27), (cx(2), 21.85))
# 6: yes -> prefetch vector of j+2
arrow((73.5, 20.5), (79.6, 20.5)); a.text(76.5, 21.4, 'yes', ha='center', va='bottom', fontsize=9.5)
step(81, 20.5, 6); a.text(83.3, 20.5, 'Prefetch its\nvector (3 lines)', va='center', fontsize=FS)
# no -> skip
arrow((cx(2), 19.15), (cx(2), 15.6)); a.text(cx(2) + 0.8, 17.3, 'no', ha='left', va='center', fontsize=9.5)
a.text(cx(2), 14.6, 'skip: no vector fetch', ha='center', va='top', fontsize=FS, color=GREY, style='italic')
# 7: distance of j
step(cx(0), 20.5, 7); a.text(cx(0) - 2.3, 20.5, 'Distance of j\n(its vector is\nalready in cache)', ha='right', va='center', fontsize=FS)
arrow((cx(0), 21.85), (cx(0), 27))
# 8: push + prefetch list of new queue top
arrow((cx(0), 19.15), (cx(0), 8.35))
step(cx(0), 7, 8); a.text(cx(0) + 2.3, 7, 'If close: push j into the queue and\nprefetch the neighbour list of the queue top',
                          va='center', fontsize=FS)
a.plot([cx(0) - 1.35, 17], [7, 7], color='black', lw=1.0); a.plot([17, 17], [7, 43.5], color='black', lw=1.0)
arrow((17, 43.5), (QX + QW, 43.5)); a.text(17.6, 25, 'push', rotation=90, ha='left', va='center', fontsize=9.5)

# legend
a.plot([76, 81], [57.5, 57.5], color='black', lw=1.0, ls=(0, (3, 2))); a.text(82, 57.5, 'prefetch', va='center', fontsize=9.5)
a.plot([76, 81], [53.5, 53.5], color='black', lw=1.0); a.text(82, 53.5, 'normal step', va='center', fontsize=9.5)

for ext in ('pdf', 'png'):
    fig.savefig(f'new_plot/pf_filter_steps.{ext}', dpi=250, bbox_inches='tight', pad_inches=0.03)
print('saved new_plot/pf_filter_steps.pdf/.png')
