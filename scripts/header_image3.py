"""Header image for the Cerebras post: the bandwidth line at both ends, 1600x900, site palette."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import wafer_model as wm

SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io\blogs\resources\blog-3'
BG = '#07090d'; INK = '#f4f6fb'; INK2 = '#a3acbd'; INK3 = '#6e7688'; GRID = '#1c212b'
ACC = '#00d4ff'; VIO = '#7c5cff'; AMB = '#bd8630'; GRN = '#37a06d'
FONT = ['Segoe UI', 'DejaVu Sans']

rows = [('this laptop\n34 GB/s', wm.ceiling(34e9), None, INK3),
        ('8 x H100\n27 TB/s', wm.ceiling(8 * wm.h100_bw), 128, AMB),
        ('8 x B200\n64 TB/s', wm.ceiling(8 * wm.b200_bw), None, AMB),
        ('4 x WSE-3\n84 PB/s', wm.ceiling(4 * wm.SRAM_BW), 2100, ACC)]

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
fig.text(0.06, 0.84, 'Where the bandwidth\nline ends.', color=INK, fontsize=42, fontweight='bold', va='top', linespacing=1.12, family=FONT)
fig.text(0.06, 0.55, 'How Cerebras gets a 70B model to 2,000 tokens\na second, worked out from the outside: the wafer,\nthe SRAM arithmetic, the microseconds a layer\nreally costs, and what the design gives up.', color=INK2, fontsize=19, va='top', linespacing=1.5, family=FONT)
fig.text(0.06, 0.13, 'inboxpraveen.github.io', color=INK3, fontsize=15, family=FONT)
fig.text(0.06, 0.09, 'Every number dated and sourced; the arithmetic is mine', color=INK3, fontsize=13, family=FONT)

ax = fig.add_axes([0.60, 0.16, 0.36, 0.70], facecolor=BG)
y = list(range(len(rows)))[::-1]
ax.barh(y, [r[1] for r in rows], height=0.6, color=[r[3] for r in rows], alpha=0.45)
for yi, (lab, c, a, col) in zip(y, rows):
    ax.text(c * 1.25, yi, f'{c:,.0f}' if c >= 10 else f'{c:.2g}', va='center', color=INK, fontsize=15, fontweight='bold', family=FONT)
    if a:
        ax.scatter([a], [yi], s=140, color=col, zorder=4, edgecolor=BG, linewidth=1.2)
        ax.text(a, yi + 0.42, f'reported: {a:,}', color=INK2, fontsize=11.5, ha='center', family=FONT)
ax.set_xscale('log'); ax.set_xlim(0.1, 2e7)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=12.5, family=FONT)
ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', colors=INK3, labelsize=11)
for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
ax.set_xlabel('tokens/s if only bandwidth mattered, Llama 70B at 16-bit (bars) vs reported (dots)', color=INK3, fontsize=11.5, family=FONT)

os.makedirs(SITE, exist_ok=True)
png = os.path.join(SITE, 'Header.png')
fig.savefig(png, dpi=100, facecolor=BG)
im = Image.open(png).convert('RGB')
im.save(os.path.join(SITE, 'Header.webp'), 'WEBP', quality=88, method=6)
print('wrote', png, im.size, os.path.getsize(os.path.join(SITE, 'Header.webp')), 'bytes webp')
