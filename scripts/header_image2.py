"""Header image for the second post: six architectures against the CPU bandwidth line, 1600x900, site palette."""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io\blogs\resources\blog-2'
BG = '#07090d'; INK = '#f4f6fb'; INK2 = '#a3acbd'; INK3 = '#6e7688'; GRID = '#1c212b'
ACC = '#00d4ff'; VIO = '#7c5cff'; AMB = '#bd8630'; GRN = '#37a06d'
FONT = ['Segoe UI', 'DejaVu Sans']

bench = [json.loads(l) for l in open(os.path.join(LAB, 'results', 'bench.jsonl'))]
meta = json.load(open(os.path.join(LAB, 'results', 'zoo_meta.json')))
def tg(fname):
    return [r for r in bench if r['tag'] == 'zoo-cpu' and r['n_gen'] and os.path.basename(r['model_filename']) == fname][-1]['avg_ts']
zoo = [('Qwen3-0.6B\ndense', 'Qwen3-0.6B-Q8_0.gguf', '../Qwen3-0.6B-Q8_0.gguf', ACC),
       ('Gemma-3-1B\nlocal/global', 'gemma-3-1b-it-Q8_0.gguf', 'gemma-3-1b-it-Q8_0.gguf', ACC),
       ('Granite MoE\n8 of 32 experts', 'granite-3.1-1b-a400m-instruct-Q8_0.gguf', 'granite-3.1-1b-a400m-instruct-Q8_0.gguf', VIO),
       ('LFM2-700M\nconv hybrid', 'LFM2-700M-Q8_0.gguf', 'LFM2-700M-Q8_0.gguf', AMB),
       ('Falcon-H1\nMamba2 hybrid', 'Falcon-H1-0.5B-Instruct-Q8_0.gguf', 'Falcon-H1-0.5B-Instruct-Q8_0.gguf', AMB),
       ('Mamba-130M\npure SSM', 'mamba-130m-Q8_0.gguf', 'mamba-130m-Q8_0.gguf', GRN)]
vals = [meta[mf]['active_bytes'] / 1e9 * tg(bf) for _, bf, mf, _ in zoo]

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
fig.text(0.06, 0.82, 'Your machine, your model.\nWhat actually carries over.', color=INK, fontsize=40, fontweight='bold', va='top', linespacing=1.15, family=FONT)
fig.text(0.06, 0.52, 'One number transfers between machines: bandwidth\ndivided by the bytes a token really reads. Six architectures,\ndense, MoE, hybrid and SSM, measured against that line.\nPlus the pre-checks, the ten-minute accuracy test\nand the order to try things in.', color=INK2, fontsize=19, va='top', linespacing=1.5, family=FONT)
fig.text(0.06, 0.13, 'inboxpraveen.github.io', color=INK3, fontsize=15, family=FONT)
fig.text(0.06, 0.09, 'Part 2 of the llama.cpp on one laptop series', color=INK3, fontsize=13, family=FONT)

ax = fig.add_axes([0.645, 0.16, 0.315, 0.70], facecolor=BG)
y = list(range(len(zoo)))[::-1]
ax.barh(y, vals, height=0.62, color=[z[3] for z in zoo])
for yi, v in zip(y, vals):
    ax.text(v + 0.8, yi, f'{v:.0f} GB/s', va='center', color=INK, fontsize=15, fontweight='bold', family=FONT)
ax.set_yticks(y); ax.set_yticklabels([z[0] for z in zoo], color=INK2, fontsize=12.5, family=FONT)
ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', colors=INK3, labelsize=12)
for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
ax.set_xlim(0, 52)
ax.axvline(34, color=INK2, linewidth=1.5, linestyle=(0, (6, 4)))
ax.axvline(44.8, color=INK3, linewidth=1.5, linestyle=(0, (2, 3)))
ax.text(34, len(zoo) - 0.3, 'measured line, 33 to 34 GB/s', color=INK2, fontsize=11, ha='right', va='bottom', family=FONT)
ax.text(44.8, len(zoo) - 0.3, ' spec, 44.8', color=INK3, fontsize=11, ha='left', va='bottom', family=FONT)
ax.set_xlabel('CPU decode: bytes read per token x tokens per second (GB/s)', color=INK3, fontsize=11.5, family=FONT)

os.makedirs(SITE, exist_ok=True)
png = os.path.join(SITE, 'Header.png')
fig.savefig(png, dpi=100, facecolor=BG)
im = Image.open(png).convert('RGB')
im.save(os.path.join(SITE, 'Header.webp'), 'WEBP', quality=88, method=6)
print('wrote', png, im.size, os.path.getsize(os.path.join(SITE, 'Header.webp')), 'bytes webp')
