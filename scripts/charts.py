"""
Figures for the blog. Static SVG, dark surface matching the site tokens, one axis per chart,
thin marks, hairline grid, text in text tokens, categorical hues in fixed order (validated for the
dark surface with the dataviz palette validator: #10a3c2 #8266ee #bd8630 #37a06d).
"""
import os, json, glob, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.environ.get('CHART_OUT', os.path.join(LAB, 'results', 'charts'))
os.makedirs(OUT, exist_ok=True)

SURFACE = '#0f1218'
INK = '#f4f6fb'; INK2 = '#a3acbd'; INK3 = '#6e7688'
GRID = '#1c212b'
C1, C2, C3, C4 = '#10a3c2', '#8266ee', '#bd8630', '#37a06d'
MUTED = '#3a4150'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'grid.color': GRID, 'grid.linewidth': 1, 'axes.grid': True, 'axes.grid.axis': 'y',
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'], 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': '600', 'axes.titlelocation': 'left', 'axes.titlepad': 14,
    'legend.frameon': False, 'legend.fontsize': 10, 'svg.fonttype': 'none',
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.svg'), bbox_inches='tight', pad_inches=0.25)
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', pad_inches=0.25, dpi=160)
    plt.close(fig)
    print('wrote', name)


def bench_rows():
    rows = [json.loads(l) for l in open(os.path.join(LAB, 'results', 'bench.jsonl'))]
    return rows


def by_tag(rows, tag):
    return [r for r in rows if r['tag'] == tag]


def hbar(ax, labels, values, errs=None, color=C1, colors=None, fmt='{:.1f}', label_inside=False):
    y = np.arange(len(labels))[::-1]
    cols = colors or [color] * len(labels)
    ax.barh(y, values, height=0.58, color=cols, xerr=errs, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=0))
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)
    xmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.012, yi, fmt.format(v), va='center', ha='left', color=INK2, fontsize=10)
    ax.set_xlim(0, xmax * 1.16)


# ---------------------------------------------------------------------------------------------------
def chart_kld_vs_bytes():
    rows = json.load(open(os.path.join(LAB, 'results', 'kld_table.json')))
    rows = [r for r in rows if r['name'] != 'bf16-self']
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    groups = {'plain': (C1, 'Standard llama-quantize mixes'),
              'head': (C2, 'Head (token_embd) changed'),
              'block': (C3, 'Attention / FFN / layer-range changed'),
              'official': (C4, 'Official Qwen3-0.6B-Q8_0')}
    def grp(n):
        if n.startswith('official'): return 'official'
        if n.startswith('plain'): return 'plain'
        if n.startswith('head'): return 'head'
        return 'block'
    for g, (c, lab) in groups.items():
        xs = [r['tensor_bytes'] / 1e6 for r in rows if grp(r['name']) == g]
        ys = [max(r['kld'], 1e-4) for r in rows if grp(r['name']) == g]
        ax.scatter(xs, ys, s=64, color=c, label=lab, zorder=3, edgecolors=SURFACE, linewidths=2)
    ax.set_yscale('log')
    ax.set_xlabel('Weight bytes read per decoded token (MB)')
    ax.set_ylabel('Mean KL divergence to BF16 (log scale)')
    ax.set_title('Every variant: smaller is faster, lower is more faithful')
    ax.grid(axis='x')
    short = {'plain-Q8_0': 'Q8_0', 'plain-Q6_K': 'Q6_K', 'plain-Q5_K_M': 'Q5_K_M', 'plain-Q4_K_M': 'Q4_K_M',
             'plain-Q4_K_S': 'Q4_K_S', 'plain-IQ4_XS': 'IQ4_XS', 'plain-Q4_0': 'Q4_0', 'plain-Q3_K_M': 'Q3_K_M',
             'plain-Q2_K': 'Q2_K', 'head-Q6_K-body-Q8_0': 'head Q6_K', 'head-Q4_K-body-Q8_0': 'head Q4_K',
             'head-Q3_K-body-Q8_0': 'head Q3_K', 'head-BF16-body-Q4_K_M': 'Q4_K_M + BF16 head',
             'attn-Q4_K-ffn-Q8_0': 'attn Q4_K', 'attn-Q8_0-ffn-Q4_K': 'FFN Q4_K', 'official-Q8_0': 'official Q8_0',
             'head-Q5_K-body-Q8_0': 'head Q5_K'}
    offs = {'plain-Q8_0': (6, -14), 'official-Q8_0': (6, 6), 'head-Q6_K-body-Q8_0': (-8, -16),
            'head-Q5_K-body-Q8_0': (6, 4), 'head-Q4_K-body-Q8_0': (6, 4), 'head-Q3_K-body-Q8_0': (6, 4),
            'plain-Q6_K': (6, 4), 'plain-Q5_K_M': (6, 4), 'plain-Q4_K_M': (6, 6), 'plain-Q4_K_S': (6, -13),
            'plain-IQ4_XS': (-8, -4), 'plain-Q4_0': (6, 4), 'plain-Q3_K_M': (6, 4), 'plain-Q2_K': (6, 4),
            'head-BF16-body-Q4_K_M': (6, 4), 'attn-Q4_K-ffn-Q8_0': (6, 4), 'attn-Q8_0-ffn-Q4_K': (-8, -12)}
    for r in rows:
        if r['name'] in short:
            dx, dy = offs.get(r['name'], (6, 4))
            ha = 'right' if dx < 0 else 'left'
            ax.annotate(short[r['name']], (r['tensor_bytes'] / 1e6, max(r['kld'], 1e-4)),
                        xytext=(dx, dy), textcoords='offset points', fontsize=9, color=INK2, ha=ha)
    ax.legend(loc='upper right')
    save(fig, 'kld-vs-bytes')


def chart_head_experiments():
    rows = {r['name']: r for r in json.load(open(os.path.join(LAB, 'results', 'kld_table.json')))}
    order = ['plain-Q8_0', 'head-Q6_K-body-Q8_0', 'head-Q5_K-body-Q8_0', 'head-Q4_K-body-Q8_0',
             'head-IQ4_XS-body-Q8_0', 'head-Q3_K-body-Q8_0']
    labels = ['Q8_0 head (633 MB)', 'Q6_K head (596 MB)', 'Q5_K head (575 MB)', 'Q4_K head (556 MB)',
              'IQ4_XS head (551 MB)', 'Q3_K head (535 MB)']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(wspace=0.55))
    hbar(axes[0], labels, [rows[n]['kld'] for n in order], errs=[rows[n]['kld_err'] for n in order],
         color=C2, fmt='{:.4f}')
    axes[0].set_title('Mean KLD to BF16 (body stays Q8_0)')
    hbar(axes[1], labels, [rows[n]['same_top'] for n in order], errs=[rows[n]['same_top_err'] for n in order],
         color=C1, fmt='{:.1f}%')
    axes[1].set_title('Top-1 token agreement with BF16')
    axes[1].set_xlim(75, 100); axes[1].set_yticklabels([])
    save(fig, 'head-experiments')


def chart_protect_head():
    rows = {r['name']: r for r in json.load(open(os.path.join(LAB, 'results', 'kld_table.json')))}
    order = ['head-BF16-body-Q4_K_M', 'head-Q8_0-body-Q4_K_M', 'plain-Q4_K_M', 'head-Q4_K-body-Q4_K_M']
    labels = ['BF16 head (574 MB)', 'Q8_0 head (428 MB)', 'Q6_K head, stock Q4_K_M (391 MB)', 'Q4_K head (351 MB)']
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    hbar(ax, labels, [rows[n]['kld'] for n in order], errs=[rows[n]['kld_err'] for n in order], color=C2, fmt='{:.4f}')
    ax.set_title('Q4_K_M body: raising the head above Q6_K buys nothing')
    ax.set_xlabel('Mean KLD to BF16')
    save(fig, 'protect-head')


def chart_blocks():
    rows = {r['name']: r for r in json.load(open(os.path.join(LAB, 'results', 'kld_table.json')))}
    order = ['plain-Q8_0', 'attn-Q4_K-ffn-Q8_0', 'attn-Q8_0-ffn-Q4_K', 'edge-Q8_0-mid-Q4_K', 'ffn-down-Q8-rest-Q4_K', 'plain-Q4_K_M']
    labels = ['All Q8_0 (633 MB)', 'Attention Q4_K, rest Q8_0 (545 MB)', 'FFN Q4_K, rest Q8_0 (501 MB)',
              'Layers 4-23 Q4_K, edges Q8_0 (476 MB)', 'ffn_down + head Q8_0, rest Q4_K (461 MB)', 'Stock Q4_K_M (391 MB)']
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    hbar(ax, labels, [rows[n]['kld'] for n in order], errs=[rows[n]['kld_err'] for n in order], color=C3, fmt='{:.4f}')
    ax.set_title('Which block to quantize first')
    ax.set_xlabel('Mean KLD to BF16')
    save(fig, 'blocks')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['kld']
    if 'kld' in which:
        chart_kld_vs_bytes(); chart_head_experiments(); chart_protect_head(); chart_blocks()


def chart_threads():
    rows = bench_rows()
    tg = sorted([r for r in rows if r['tag'] == 'threads-sweep-r2' and r['n_gen']], key=lambda r: r['n_threads'])
    pp = sorted([r for r in rows if r['tag'] == 'threads-sweep-r2' and r['n_prompt']], key=lambda r: r['n_threads'])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(wspace=0.3))
    for ax, data, title, col in [(axes[0], tg, 'Decode, tg128 (tok/s)', C1), (axes[1], pp, 'Prompt processing, pp256 (tok/s)', C2)]:
        x = [r['n_threads'] for r in data]; y = [r['avg_ts'] for r in data]; e = [r['stddev_ts'] for r in data]
        ax.fill_between(x, [a - b for a, b in zip(y, e)], [a + b for a, b in zip(y, e)], color=col, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=col, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(title); ax.set_xlabel('threads (-t)')
        ax.set_xticks([1, 4, 8, 12, 16, 24])
        ax.axvspan(8.5, 24.5, color=MUTED, alpha=0.18, linewidth=0)
        ax.set_ylim(0)
    axes[0].annotate('8 P-cores', (8, tg[[r['n_threads'] for r in tg].index(8)]['avg_ts']), xytext=(-46, 10), textcoords='offset points', color=INK2, fontsize=9)
    axes[0].text(16.5, 3, 'threads land on E-cores', color=INK3, fontsize=9, ha='center')
    axes[1].text(16.5, 12, 'threads land on E-cores', color=INK3, fontsize=9, ha='center')
    save(fig, 'threads')


def chart_dll():
    rows = bench_rows()
    variants = ['alderlake', 'haswell', 'ivybridge', 'sandybridge', 'piledriver', 'sse42', 'x64']
    def get(v, fname, test):
        rs = [r for r in rows if r['tag'] == f'dll-{v}' and os.path.basename(r['model_filename']) == fname and ((r['n_gen'] > 0) if test == 'tg' else (r['n_prompt'] > 0))]
        return rs[-1]['avg_ts'] if rs else 0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(wspace=0.45))
    y = np.arange(len(variants))[::-1]
    for ax, test, title in [(axes[0], 'tg', 'Decode, tg128 (tok/s)'), (axes[1], 'pp', 'Prompt processing, pp256 (tok/s)')]:
        q8 = [get(v, 'Qwen3-0.6B-Q8_0.gguf', test) for v in variants]
        q4 = [get(v, 'plain-Q4_K_M.gguf', test) for v in variants]
        ax.barh(y + 0.19, q8, height=0.34, color=C1, label='Q8_0')
        ax.barh(y - 0.19, q4, height=0.34, color=C2, label='Q4_K_M')
        ax.set_yticks(y); ax.set_yticklabels(variants)
        ax.grid(axis='x'); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
        ax.set_title(title)
        m = max(q8 + q4)
        for yi, a, b in zip(y, q8, q4):
            ax.text(a + m * 0.012, yi + 0.19, f'{a:.0f}', va='center', fontsize=8.5, color=INK2)
            ax.text(b + m * 0.012, yi - 0.19, f'{b:.0f}', va='center', fontsize=8.5, color=INK2)
        ax.set_xlim(0, m * 1.15)
    axes[1].set_yticklabels([])
    axes[0].legend(loc='lower right')
    save(fig, 'dll-variants')


if __name__ == '__main__' and ('threads' in sys.argv or 'dll' in sys.argv):
    if 'threads' in sys.argv: chart_threads()
    if 'dll' in sys.argv: chart_dll()


def chart_ngl():
    rows = bench_rows()
    tg = sorted([r for r in rows if r['tag'] == 'ngl-sweep' and r['n_gen']], key=lambda r: r['n_gpu_layers'])
    x = [min(r['n_gpu_layers'], 29) for r in tg]; y = [r['avg_ts'] for r in tg]; e = [r['stddev_ts'] for r in tg]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.fill_between(x, [a - b for a, b in zip(y, e)], [a + b for a, b in zip(y, e)], color=C1, alpha=0.12, linewidth=0)
    ax.plot(x, y, color=C1, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 29]); ax.set_xticklabels(['0', '4', '8', '12', '16', '20', '24', '28', 'all'])
    ax.set_xlabel('layers on the GPU (-ngl); "all" also moves the tied head')
    ax.set_title('Decode speed against layers offloaded, Qwen3-0.6B Q8_0, tg128 (tok/s)')
    ax.set_ylim(0)
    for xi, yi in zip(x, y):
        if xi in (0, 12, 20, 28, 29):
            ax.annotate(f'{yi:.0f}', (xi, yi), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9, color=INK2)
    ax.plot([0, 29], [y[0], y[-1]], color=MUTED, linewidth=1, linestyle='-', zorder=1)
    ax.text(10, y[0] + (y[-1] - y[0]) * 10 / 29 + 22, 'if speed scaled with layers', color=INK3, fontsize=9, ha='center')
    save(fig, 'ngl-sweep')


def chart_depth():
    rows = bench_rows()
    series = [('GPU, f16 KV', 'depth-cuda', C1), ('GPU, q8_0 KV', 'depth-cuda-kvq8', C2),
              ('CPU -t 8, FA on', 'depth-cpu', C3), ('CPU -t 8, FA off', 'depth-cpu-fa-off', C4)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(wspace=0.3))
    for label, tag, col in series:
        rs = sorted([r for r in rows if r['tag'] == tag and r['n_gen']], key=lambda r: r['n_depth'])
        x = [r['n_depth'] for r in rs]; y = [r['avg_ts'] for r in rs]
        ax = axes[0] if 'GPU' in label else axes[1]
        ax.plot(x, y, color=col, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(f'{y[-1]:.0f}', (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK2)
    for ax, t in zip(axes, ['GPU, tg64 (tok/s)', 'CPU, tg64 (tok/s)']):
        ax.set_title(t); ax.set_xlabel('tokens already in context'); ax.set_ylim(0)
        ax.set_xticks([0, 2048, 8192, 16384]); ax.set_xticklabels(['0', '2K', '8K', '16K'])
        ax.legend(loc='upper right')
    save(fig, 'depth')


if __name__ == '__main__' and ('ngl' in sys.argv or 'depth' in sys.argv):
    if 'ngl' in sys.argv: chart_ngl()
    if 'depth' in sys.argv: chart_depth()


def chart_ngl_8b():
    rows = bench_rows()
    g = {r['tag']: r for r in rows if r['tag'].startswith('8b-') and r['n_gen']}
    xs = [0, 16, 24, 28, 30, 32, 33, 34, 35, 36, 37]
    ys = [g['8b-cpu']['avg_ts']] + [g[f'8b-ngl-{n}']['avg_ts'] for n in [16, 24, 28, 30, 32, 33, 34, 35, 36]] + [g['8b-ngl-99']['avg_ts']]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(xs, ys, color=C1, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, label='whole layers (-ngl)')
    # FFN-only placement: equivalent layers on CPU ~ 0.7 * N
    ox = [37 - 0.7 * n for n in [4, 8, 12]]; oy = [g[f'8b-otffn-{n}']['avg_ts'] for n in [4, 8, 12]]
    ax.plot(ox, oy, color=C2, linewidth=2, marker='s', markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, linestyle='none', label='FFN of first 4 / 8 / 12 layers on CPU (-ot)')
    ax.set_xticks([0, 16, 24, 28, 32, 36, 37]); ax.set_xticklabels(['0', '16', '24', '28', '32', '36', 'all'])
    ax.set_xlabel('layers on the GPU (-ngl); "all" also moves the head')
    ax.set_title('Qwen3-8B Q6_K decode against layers offloaded, tg64 (tok/s)')
    ax.set_ylim(0); ax.legend(loc='upper left')
    for xi, yi in zip(xs, ys):
        if xi in (0, 28, 32, 36, 37):
            ax.annotate(f'{yi:.1f}', (xi, yi), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9, color=INK2)
    save(fig, 'ngl-8b')


if __name__ == '__main__' and 'ngl8b' in sys.argv:
    chart_ngl_8b()


def chart_spec():
    srv = [json.loads(l) for l in open(os.path.join(LAB, 'results', 'server.jsonl'))]
    drv = [json.loads(l) for l in open(os.path.join(LAB, 'results', 'driver.jsonl'))]
    def sg(tag, wl):
        rs = [r for r in srv if r['tag'] == tag and r['workload'] == wl]; return rs[-1]['tok_s_mean'] if rs else 0
    def dg(tag, loop, wl):
        rs = [r for r in drv if r['tag'] == tag and r['loop'] == loop and r['workload'] == wl]; return rs[-1]['tok_s_mean'] if rs else 0
    wls = ['free', 'edit', 'summary_quote']; wl_lab = ['free writing', 'edit a paragraph', 'quote and explain']
    panels = [
        ('CPU, -t 8 (tok/s)', [('no speculation', C1, [sg('srv-cpu-none', w) for w in wls]),
                               ('ngram-simple, defaults', MUTED, [sg('srv-cpu-ngram-simple', w) for w in wls]),
                               ('ngram-simple, n=3 m=8', C2, [sg('srv-cpu-ngram-simple-3-8', w) for w in wls]),
                               ('driver n-gram loop, n=3 draft 8', C3, [dg('drv-cpu', 'ngram_spec_loop', w) for w in wls])]),
        ('CUDA (tok/s)', [('no speculation', C1, [sg('srv-cuda-none-rerun', w) for w in wls]),
                          ('ngram-simple, defaults', MUTED, [sg('srv-cuda-ngram-simple', w) for w in wls]),
                          ('ngram-simple, n=3 m=16', C2, [sg('srv-cuda-ngram-simple-3-16-rerun', w) for w in wls]),
                          ('driver n-gram loop, n=3 draft 16', C3, [dg('drv-cuda-ng16', 'ngram_spec_loop', w) for w in wls])]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(wspace=0.25))
    for ax, (title, series) in zip(axes, panels):
        n = len(series); w = 0.8 / n; x = np.arange(len(wls))
        m = max(max(v) for _, _, v in series)
        for i, (lab, col, vals) in enumerate(series):
            xs = x - 0.4 + w * (i + 0.5)
            ax.bar(xs, vals, width=w - 0.04, color=col, label=lab)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + m * 0.015, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(wl_lab); ax.set_title(title); ax.set_ylim(0, m * 1.42)
        ax.legend(loc='upper left', fontsize=8.5, ncol=2)
    save(fig, 'speculative')


if __name__ == '__main__' and 'spec' in sys.argv:
    chart_spec()


def chart_speed_vs_bytes():
    rows = bench_rows()
    kl = {r['name']: r for r in json.load(open(os.path.join(LAB, 'results', 'kld_table.json')))}
    def pts(tag):
        out = []
        for r in rows:
            if r['tag'] == tag and r['n_gen']:
                n = os.path.basename(r['model_filename'])[:-5]
                if n in kl: out.append((kl[n]['tensor_bytes'] / 1e6, r['avg_ts'], n))
        return out
    cpu = pts('quant-cpu-r2') + [(1192.2, [r for r in rows if r['tag'] == 'quant-cpu-r2-bf16'][-1]['avg_ts'], 'bf16')]
    gpu = pts('quant-cuda') + [(1192.2, [r for r in rows if r['tag'] == 'quant-cuda-bf16' and r['n_gen']][-1]['avg_ts'], 'bf16')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(wspace=0.25))
    for ax, data, title, col in [(axes[0], cpu, 'CPU -t 8, tg128 (tok/s)', C1), (axes[1], gpu, 'CUDA, tg128 (tok/s)', C2)]:
        xs = [d[0] for d in data]; ys = [d[1] for d in data]
        ax.scatter(xs, ys, s=52, color=col, zorder=3, edgecolors=SURFACE, linewidths=2)
        ax.set_title(title); ax.set_xlabel('weight bytes read per token (MB)'); ax.set_ylim(0); ax.grid(axis='x')
        for x, y, n in data:
            lab = {'plain-Q8_0': 'Q8_0', 'plain-Q4_K_M': 'Q4_K_M', 'plain-Q2_K': 'Q2_K', 'bf16': 'BF16', 'plain-Q6_K': 'Q6_K', 'plain-Q4_0': 'Q4_0'}.get(n)
            if lab: ax.annotate(lab, (x, y), xytext=(6, 5), textcoords='offset points', fontsize=9, color=INK2)
    # CPU bandwidth reference line: tok/s = 34 GB/s / bytes
    xs = np.linspace(250, 1250, 50); axes[0].plot(xs, 34e3 / xs, color=MUTED, linewidth=1, zorder=1)
    axes[0].text(900, 34e3 / 900 + 6, '34 GB/s / bytes', color=INK3, fontsize=9)
    save(fig, 'speed-vs-bytes')


if __name__ == '__main__' and 'svb' in sys.argv:
    chart_speed_vs_bytes()
