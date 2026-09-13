"""Header image for the post: the CPU decode ladder, stock defaults to tuned, 1600x900, site palette."""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io\blogs\resources\blog-1'
BG = '#07090d'; PANEL = '#0f1218'; INK = '#f4f6fb'; INK2 = '#a3acbd'; INK3 = '#6e7688'; GRID = '#1c212b'
ACC = '#00d4ff'; VIO = '#7c5cff'; MUTED = '#2b3140'

bench = [json.loads(l) for l in open(os.path.join(LAB, 'results', 'bench.jsonl'))]
srv = [json.loads(l) for l in open(os.path.join(LAB, 'results', 'server.jsonl'))]
def tg(tag, **m):
    rs = [r for r in bench if r['tag'] == tag and r['n_gen'] and all(r.get(k) == v for k, v in m.items())]
    return rs[-1]['avg_ts']
def sv(tag, wl):
    return [r for r in srv if r['tag'] == tag and r['workload'] == wl][-1]['tok_s_mean']

steps = [('stock defaults\n16 threads', tg('pair-cpu-default'), MUTED),
         ('-t 8\nP-cores only', tg('pair-cpu-t8'), '#1590b0'),
         ('-t 8 + n-gram drafts\n(editing task)', sv('srv-cpu-ngram-simple-3-8', 'edit'), ACC)]

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
# left text block
fig.text(0.06, 0.80, 'One laptop. One model.\nEvery knob llama.cpp has.', color=INK, fontsize=44, fontweight='bold', va='top', linespacing=1.15, family=['Segoe UI', 'DejaVu Sans'])
fig.text(0.06, 0.47, 'Qwen3-0.6B Q8_0 on an i7-14650HX with one DDR5 stick\nand an RTX 5060. Threads, CPU DLLs, the tied LM head,\nGPU splits, KV cache, speculative decoding.\nEvery step measured for speed and for accuracy.', color=INK2, fontsize=19, va='top', linespacing=1.5, family=['Segoe UI', 'DejaVu Sans'])
fig.text(0.06, 0.13, 'inboxpraveen.github.io', color=INK3, fontsize=15, family=['Segoe UI', 'DejaVu Sans'])
fig.text(0.06, 0.09, 'CPU decode, tokens per second, same Q8_0 weights, measured under my normal working load; the last bar is an editing prompt where drafts apply', color=INK3, fontsize=13, family=['Segoe UI', 'DejaVu Sans'])

ax = fig.add_axes([0.56, 0.14, 0.40, 0.72], facecolor=BG)
x = list(range(len(steps))); vals = [s[1] for s in steps]
bars = ax.bar(x, vals, width=0.62, color=[s[2] for s in steps])
for xi, v in zip(x, vals):
    ax.text(xi, v + 1.8, f'{v:.0f}', ha='center', va='bottom', color=INK, fontsize=20, fontweight='bold', family=['Segoe UI', 'DejaVu Sans'])
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], color=INK2, fontsize=12.5, family=['Segoe UI', 'DejaVu Sans'])
ax.tick_params(axis='x', length=0, pad=10); ax.tick_params(axis='y', colors=INK3, labelsize=12)
for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
ax.set_ylim(0, max(vals) * 1.22)
ax.axhline(70.7, color=VIO, linewidth=1.5, linestyle=(0, (6, 4)))
ax.text(-0.3, 72.2, 'single-token bandwidth ceiling, 70.7 tok/s', color=VIO, fontsize=11.5, ha='left', va='bottom', family=['Segoe UI', 'DejaVu Sans'])

os.makedirs(SITE, exist_ok=True)
png = os.path.join(SITE, 'Header.png')
fig.savefig(png, dpi=100, facecolor=BG)
im = Image.open(png).convert('RGB')
im.save(os.path.join(SITE, 'Header.webp'), 'WEBP', quality=88, method=6)
print('wrote', png, im.size, os.path.getsize(os.path.join(SITE, 'Header.webp')), 'bytes webp')
