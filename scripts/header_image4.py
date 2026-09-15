"""Header image for post 4: tokens per Hindi word before and after the tokenizer work, 1600x900, site palette."""
import os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io\blogs\resources\blog-4'
BG = '#07090d'; INK = '#f4f6fb'; INK2 = '#a3acbd'; INK3 = '#6e7688'; GRID = '#1c212b'
ACC = '#00d4ff'; AMB = '#bd8630'; MUTED = '#3a4150'
FONT = ['Segoe UI', 'DejaVu Sans']

if '--webp-only' in sys.argv:
    p = os.path.join(SITE, 'Header.png'); Image.open(p).convert('RGB').save(os.path.join(SITE, 'Header.webp'), 'WEBP', quality=88, method=6)
    print('wrote webp', os.path.getsize(os.path.join(SITE, 'Header.webp'))); sys.exit()
ts = {json.loads(l)['label']: json.loads(l) for l in open(os.path.join(LAB, 'results', 'adapt', 'tok_stats.jsonl'), encoding='utf-8')}
rows = [('SmolLM2, stock', ts['SmolLM2 (49k)']['hi']['tokens_per_word'], AMB),
        ('Qwen2.5', ts['Qwen2.5 (152k)']['hi']['tokens_per_word'], MUTED),
        ('Llama 3.2', ts['Llama 3.2 (128k)']['hi']['tokens_per_word'], MUTED),
        ('Gemma 3', ts['Gemma 3 (262k)']['hi']['tokens_per_word'], MUTED),
        ('SmolLM2 + 16k\nHindi tokens', ts['tok-16k']['hi']['tokens_per_word'], ACC)]
en = ts['SmolLM2 (49k)']['en']['tokens_per_word']

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
fig.text(0.06, 0.84, 'A small model\nlearns Hindi.', color=INK, fontsize=44, fontweight='bold', va='top', linespacing=1.12, family=FONT)
fig.text(0.06, 0.55, 'Taking an English-only 135M model to a new\nlanguage on one 8 GB laptop GPU: the tokenizer,\nthe new embeddings, continued pretraining, a\nsupervised warm-up and RL with rewards a\nscript can check. Every number measured.', color=INK2, fontsize=18.5, va='top', linespacing=1.5, family=FONT)
fig.text(0.06, 0.13, 'inboxpraveen.github.io', color=INK3, fontsize=15, family=FONT)
fig.text(0.06, 0.09, 'Scripts and results in the repository; the large-lab pipeline set beside it', color=INK3, fontsize=13, family=FONT)

ax = fig.add_axes([0.60, 0.18, 0.36, 0.66], facecolor=BG)
y = list(range(len(rows)))[::-1]
ax.barh(y, [r[1] for r in rows], height=0.6, color=[r[2] for r in rows], alpha=0.9)
for yi, (lab, v, col) in zip(y, rows):
    ax.text(v + 0.12, yi, f'{v:.2f}', va='center', color=INK, fontsize=15, fontweight='bold', family=FONT)
ax.axvline(en, color=INK3, linestyle=':', linewidth=1.5)
ax.text(en + 0.08, y[0] + 0.62, f'English: {en:.2f}', color=INK3, fontsize=11.5, family=FONT)
ax.set_xlim(0, 6.4)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=12.5, family=FONT)
ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', colors=INK3, labelsize=11)
for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
ax.set_xlabel('tokens per Hindi word (held-out web text)', color=INK3, fontsize=11.5, family=FONT)

os.makedirs(SITE, exist_ok=True)
png = os.path.join(SITE, 'Header.png')
fig.savefig(png, dpi=100, facecolor=BG)
try:
    Image.open(png).convert('RGB').save(os.path.join(SITE, 'Header.webp'), 'WEBP', quality=88, method=6)
except Exception as e:  # this env's Pillow has no webp codec; convert with: conda run -n adapt-lab python scripts/header_image4.py --webp-only
    print('webp skipped:', e)
print('wrote', png)
