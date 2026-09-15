"""Figures for post 4 (Hindi adaptation of SmolLM2-135M). Same surface and palette as charts.py. Every
number comes from results/adapt/*.jsonl written by the scripts in scripts/adapt/."""
import os, sys, json, glob, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
from charts import LAB, SURFACE, INK, INK2, INK3, GRID, C1, C2, C3, C4, MUTED  # noqa

RES = os.path.join(LAB, 'results', 'adapt')
OUT = os.path.join(LAB, 'results', 'charts4')
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.svg'), bbox_inches='tight', pad_inches=0.25)
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', pad_inches=0.25, dpi=160)
    plt.close(fig)
    print('wrote', name)


def rows(name):
    p = os.path.join(RES, name + '.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def evals(tag):
    return [r for r in rows('cpt-' + tag) if r['kind'] == 'eval']


def last(name, key, val):
    rs = [r for r in rows(name) if r.get(key) == val]
    return rs[-1] if rs else None


# ------------------------------------------------------------------------------------------- fertility
def chart_fertility():
    ts = {r['label']: r for r in rows('tok_stats')}
    order = ['SmolLM2 (49k)', 'Qwen2.5 (152k)', 'Llama 3.2 (128k)', 'Gemma 3 (262k)', 'Sarvam-1 (68k)', 'tok-8k', 'tok-16k', 'tok-32k']
    names = {'tok-8k': 'SmolLM2 + 8k Hindi (57k)', 'tok-16k': 'SmolLM2 + 16k Hindi (65k)', 'tok-32k': 'SmolLM2 + 29k Hindi (79k)'}
    order = [o for o in order if o in ts]
    y = np.arange(len(order))[::-1]
    hi = [ts[o]['hi']['tokens_per_word'] for o in order]; en = [ts[o]['en']['tokens_per_word'] for o in order]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.barh(y + 0.19, hi, height=0.36, color=[C1 if o.startswith('tok-') else C3 for o in order], label='Hindi')
    ax.barh(y - 0.19, en, height=0.36, color=MUTED, label='English')
    ax.set_yticks(y); ax.set_yticklabels([names.get(o, o) for o in order])
    ax.grid(axis='x'); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    for yi, h, e in zip(y, hi, en):
        ax.text(h + 0.06, yi + 0.19, f'{h:.2f}', va='center', fontsize=9.5, color=INK2)
        ax.text(e + 0.06, yi - 0.19, f'{e:.2f}', va='center', fontsize=9.5, color=INK3)
    ax.set_xlim(0, max(hi) * 1.12)
    ax.set_xlabel('tokens per whitespace-separated word (1,000 held-out documents each)')
    ax.set_title('Tokens per word: stock tokenizers in amber, this post\'s extended SmolLM2 tokenizers in blue')
    ax.legend(loc='lower right')
    save(fig, 'fertility')


# ------------------------------------------------------------------------------------------- size vs language
def logx_params(ax):
    ax.set_xscale('log')
    ax.set_xticks([1e8, 2e8, 5e8, 1e9, 2e9, 5e9]); ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e9:.0f}B' if v >= 1e9 else f'{v/1e6:.0f}M'))
    ax.set_xlim(1e8, 4e9)


OFFS = {'Qwen2.5-0.5B': (6, 6), 'Qwen3-0.6B': (6, -12), 'SmolLM2-1.7B': (6, 6), 'Qwen3-1.7B': (6, -12), 'Qwen2.5-1.5B': (-8, 4), 'sarvam-1': (-8, 8)}  # FineWeb-2 panel
OFFS_WIKI = {'Qwen2.5-0.5B': (6, -12), 'Qwen3-0.6B': (6, 6), 'SmolLM2-1.7B': (6, 6), 'Qwen3-1.7B': (6, -12), 'Qwen2.5-1.5B': (6, -12), 'sarvam-1': (-8, 8)}


def chart_bpc_sizes():
    rs = rows('bpb')
    seen = {}
    for r in rs:
        seen[r['tag']] = r
    stock = [r for r in seen.values() if not r['model'].startswith('models/') and 'Instruct' not in r['tag']]
    ours = [r for r in seen.values() if r['model'].startswith('models/')]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    logx_params(ax)
    fam = lambda t: 'SmolLM2' if t.startswith('SmolLM2') else ('Qwen' if t.startswith('Qwen') else 'Sarvam')
    cols = {'SmolLM2': C3, 'Qwen': C2, 'Sarvam': C4}
    for r in stock:
        ax.scatter([r['params']], [r['hi']['bpc']], s=60, color=cols[fam(r['tag'])], zorder=3, edgecolor=SURFACE)
        dx, dy = OFFS.get(r['tag'], (6, 5))
        ax.annotate(r['tag'], (r['params'], r['hi']['bpc']), xytext=(dx, dy), textcoords='offset points', fontsize=8.5, color=INK2, ha='right' if dx < 0 else 'left')
    labels = {'cpt-main': 'this post: 135M after continued pretraining', 'run1200': '135M after 1,200 steps'}
    if 'cpt-main' in seen:
        labels.pop('run1200')  # both sit at the same x; label the main run only
    for r in ours:
        if r['tag'] in labels:
            ax.scatter([r['params']], [r['hi']['bpc']], s=80, color=C1, marker='D', zorder=4, edgecolor=SURFACE)
            ax.annotate(labels[r['tag']], (r['params'], r['hi']['bpc']), xytext=(8, -12), textcoords='offset points', fontsize=8.5, color=C1)
        elif r['tag'] == 'run1200':
            ax.scatter([r['params']], [r['hi']['bpc']], s=50, color=C1, marker='D', zorder=4, edgecolor=SURFACE, alpha=0.6)
    ax.set_xlabel('parameters (log scale)')
    ax.set_ylabel('Hindi bits per character, lower is better')
    ax.set_title('Hindi quality against model size: stock models (amber SmolLM2, purple Qwen, green Sarvam-1) and this post\'s 135M (blue)')
    save(fig, 'bpc-sizes')


def chart_wiki():
    """Same models, second held-out set: Hindi Wikipedia, which nothing here trained on (FineWeb-2 is web crawl)."""
    bw = {r['tag']: r for r in rows('bpb_wiki')}
    bp = {r['tag']: r for r in rows('bpb')}
    if not bw:
        return
    fam = lambda t: 'SmolLM2' if t.startswith('SmolLM2') else ('Qwen' if t.startswith('Qwen') else 'Sarvam')
    cols = {'SmolLM2': C3, 'Qwen': C2, 'Sarvam': C4}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, (title, get, offs) in zip(axes, [('held-out FineWeb-2 (same source as the training text)', lambda t: bp[t]['hi']['bpc'] if t in bp else None, OFFS),
                                            ('Hindi Wikipedia (never trained on)', lambda t: bw[t]['wiki']['bpc'] if t in bw else None, OFFS_WIKI)]):
        logx_params(ax)
        for t, r in bw.items():
            y = get(t)
            if y is None or t.startswith(('grpo', 'sft')):
                continue
            if t == 'cpt-main':
                ax.scatter([r['params']], [y], s=80, color=C1, marker='D', zorder=4, edgecolor=SURFACE)
                ax.annotate('this post: 135M after CPT', (r['params'], y), xytext=(8, -12), textcoords='offset points', fontsize=8.5, color=C1)
            elif t == 'run1200':
                ax.scatter([r['params']], [y], s=50, color=C1, marker='D', zorder=4, edgecolor=SURFACE, alpha=0.6)
                ax.annotate('after 1,200 steps', (r['params'], y), xytext=(8, 4), textcoords='offset points', fontsize=8, color=C1, alpha=0.8)
            else:
                ax.scatter([r['params']], [y], s=60, color=cols[fam(t)], zorder=3, edgecolor=SURFACE)
                dx, dy = offs.get(t, (6, 5))
                ax.annotate(t, (r['params'], y), xytext=(dx, dy), textcoords='offset points', fontsize=8.5, color=INK2, ha='right' if dx < 0 else 'left')
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel('parameters (log scale)')
    axes[0].set_ylabel('Hindi bits per character, lower is better')
    fig.suptitle('The same models on two Hindi held-out sets', fontsize=12, color=INK)
    save(fig, 'wiki')


# ------------------------------------------------------------------------------------------- ablations
def train_elapsed(tag):
    """Wall-clock at each eval, minus the time the evals themselves took (the stock tokenizer's evals are
    four times longer because the same text is four times as many tokens)."""
    rs = rows('cpt-' + tag)
    tr = {r['step']: r['elapsed_s'] for r in rs if r['kind'] == 'train'}
    out, spent = [], 0.0
    for r in rs:
        if r['kind'] != 'eval':
            continue
        t_train = tr.get(r['step'], r['elapsed_s'])
        out.append(t_train - spent)
        spent += max(0.0, r['elapsed_s'] - t_train)
    return out


def _curve(ax, tag, label, color, x='hi_chars', y='hi_bpc', ls='-', marker='o'):
    ev = evals(tag)
    if not ev:
        return
    xs = train_elapsed(tag) if x == 'elapsed_s' else [r[x] for r in ev]
    ys = [r[y] for r in ev]
    ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=5, label=label, markeredgecolor=SURFACE)


def chart_init():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for tag, lab, col in [('abl-16k-random', 'random N(0, 0.02)', C3), ('abl-16k-hf', 'transformers default (mean and covariance of old rows)', C2), ('abl-16k-mean', 'mean of the old sub-token embeddings', C1)]:
        _curve(ax, tag, lab, col)
    ax.set_xlabel('Hindi characters trained on')
    ax.set_ylabel('Hindi bits per character (held-out, log scale)')
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_title('New embedding rows: how they start decides the first few million tokens (16k Hindi tokens, 400 steps each)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M'))
    ax.legend()
    save(fig, 'init')


def chart_tokenizer():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    runs = [('abl-stock', 'stock tokenizer (49k)', MUTED), ('abl-8k', '+8k Hindi tokens', C3), ('abl-16k-mean', '+16k Hindi tokens', C1), ('abl-32k', '+29k Hindi tokens', C2)]
    for tag, lab, col in runs:
        _curve(axes[0], tag, lab, col, x='hi_chars')
        _curve(axes[1], tag, lab, col, x='elapsed_s')
    axes[0].set_xlabel('Hindi characters trained on'); axes[1].set_xlabel('wall-clock training time on the RTX 5060')
    axes[0].set_ylabel('Hindi bits per character (held-out, log scale)')
    for ax in axes:
        ax.set_yscale('log'); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M'))
    axes[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(120))
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/60:.0f} min'))
    axes[0].set_title('Same 400 steps, counted in characters'); axes[1].set_title('Same 400 steps, counted in minutes')
    axes[0].legend()
    fig.suptitle('Vocabulary extension: the stock tokenizer sees a quarter of the Hindi per step and pays for it', x=0.01, ha='left', fontsize=13, fontweight='600')
    fig.tight_layout()
    save(fig, 'tokenizer-ablation')


def chart_replay():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    runs = [('abl-noreplay', '0% English', C3), ('abl-16k-mean', '10% English', C1), ('abl-replay30', '30% English', C2)]
    for tag, lab, col in runs:
        _curve(axes[0], tag, lab, col, x='tokens', y='hi_bpc')
        _curve(axes[1], tag, lab, col, x='tokens', y='en_bpc')
    for ax in axes:
        ax.set_xlabel('tokens trained on'); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M'))
    axes[0].set_ylabel('Hindi bits per character (log scale)'); axes[1].set_ylabel('English bits per character')
    axes[0].set_yscale('log'); axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')); axes[0].yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axes[0].set_title('Hindi, held-out'); axes[1].set_title('English, held-out (FineWeb-Edu)')
    axes[0].legend()
    fig.suptitle('Replay: the share of English in the mix decides how much English the model forgets', x=0.01, ha='left', fontsize=13, fontweight='600')
    fig.tight_layout()
    save(fig, 'replay')


def chart_lr():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    runs = [('abl-lr2e-4', 'peak 2e-4', C3), ('abl-16k-mean', 'peak 5e-4', C1), ('abl-lr1e-3', 'peak 1e-3', C2)]
    for tag, lab, col in runs:
        _curve(axes[0], tag, lab, col, x='tokens', y='hi_bpc')
        _curve(axes[1], tag, lab, col, x='tokens', y='en_bpc')
    for ax in axes:
        ax.set_xlabel('tokens trained on'); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M'))
    axes[0].set_ylabel('Hindi bits per character (log scale)'); axes[1].set_ylabel('English bits per character')
    axes[0].set_yscale('log'); axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')); axes[0].yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axes[0].set_title('Hindi, held-out'); axes[1].set_title('English, held-out (FineWeb-Edu)')
    axes[0].legend()
    fig.suptitle('Peak learning rate: 400 steps, warmup 50, cosine to a tenth of the peak', x=0.01, ha='left', fontsize=13, fontweight='600')
    fig.tight_layout()
    save(fig, 'lr')


# ------------------------------------------------------------------------------------------- the main run and scaling
def chart_main():
    ev = evals('main')
    if len(ev) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot([r['tokens'] for r in ev], [r['hi_bpc'] for r in ev], color=C1, linewidth=2, marker='o', markersize=4, label='Hindi', markeredgecolor=SURFACE)
    ax.plot([r['tokens'] for r in ev], [r['en_bpc'] for r in ev], color=MUTED, linewidth=2, marker='o', markersize=4, label='English', markeredgecolor=SURFACE)
    base = last('bpb', 'tag', 'SmolLM2-135M')
    if base:
        ax.axhline(base['hi']['bpc'], color=C3, linestyle='--', linewidth=1.2)
        ax.annotate('stock 135M, Hindi', (ev[-1]['tokens'], base['hi']['bpc']), xytext=(-4, 5), textcoords='offset points', ha='right', fontsize=9, color=C3)
        ax.axhline(base['en']['bpc'], color=INK3, linestyle='--', linewidth=1.2)
        ax.annotate('stock 135M, English', (ev[-1]['tokens'], base['en']['bpc']), xytext=(-4, 5), textcoords='offset points', ha='right', fontsize=9, color=INK3)
    ax.set_xlabel('tokens trained on (90% Hindi, 10% English)'); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M'))
    ax.set_ylabel('bits per character (held-out)')
    ax.set_title('The main run: Hindi comes down, English gives some back')
    ax.legend(loc='upper right')
    save(fig, 'main-run')


def chart_scaling():
    pts = []
    for tag in ['abl-16k-mean', 'run1200', 'main']:
        ev = evals(tag)
        cfg = [r for r in rows('cpt-' + tag) if r['kind'] == 'config']
        if ev and cfg and ev[-1]['step'] == cfg[-1]['steps']:  # complete runs only
            pts.append((ev[-1]['hi_tokens'], ev[-1]['hi_bpc'], tag))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xscale('log'); ax.set_yscale('log')
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    ax.scatter(xs, ys, s=70, color=C1, zorder=4, edgecolor=SURFACE)
    for x, y, t in pts:
        ax.annotate({'abl-16k-mean': '400 steps', 'run1200': '1,200 steps', 'main': 'main run'}[t], (x, y), xytext=(8, -14), textcoords='offset points', fontsize=9, color=INK2)
    # power law fit through the full-schedule endpoints
    a, b = np.polyfit(np.log(xs), np.log(ys), 1)
    gx = np.logspace(np.log10(xs.min()), np.log10(xs.max() * 12), 100)
    ax.plot(gx, np.exp(b) * gx ** a, color=C1, linestyle=':', linewidth=1.5, label=f'fit through the {len(pts)} runs: bpc ~ tokens^{a:.2f}')
    for tag, col, side, dy in [('SmolLM2-360M', C3, 'right', 4), ('SmolLM2-1.7B', C3, 'left', 4), ('Qwen2.5-1.5B', C2, 'right', 4), ('Qwen3-1.7B', C2, 'right', -13), ('sarvam-1', C4, 'left', 4)]:
        r = last('bpb', 'tag', tag)
        if r:
            ax.axhline(r['hi']['bpc'], color=col, linestyle='--', linewidth=1)
            x_at = gx[-1] if side == 'right' else gx[0]
            ax.annotate(f"{tag} (stock): {r['hi']['bpc']:.2f}", (x_at, r['hi']['bpc']), xytext=(-4 if side == 'right' else 4, dy), textcoords='offset points', ha=side, fontsize=8.5, color=col)
    ax.set_yticks([0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('Hindi tokens trained on (log)'); ax.set_ylabel('Hindi bits per character (log)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e9:.1f}B' if v >= 1e9 else f'{v/1e6:.0f}M'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.set_title('If we keep going: three complete runs, a line through them, and where the bigger stock models sit')
    ax.legend(loc='lower left')
    save(fig, 'scaling')


# ------------------------------------------------------------------------------------------- SFT and GRPO
def chart_sft():
    """Left: answer-token loss for the run that feeds GRPO (the cold-start one when it exists). Right: what the
    stage does to the model's Hindi and English text scores, for both SFT runs."""
    runs = [t for t in ['sft-sft-cold', 'sft-sft'] if rows(t)]
    if not runs:
        return
    main = runs[0]
    rs = rows(main)
    tr = [r for r in rs if r['kind'] == 'train']; ev = [r for r in rs if r['kind'] == 'eval']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    ax.plot([r['step'] for r in tr], [r['loss_ema'] for r in tr], color=C1, linewidth=1.8, label='train loss (smoothed)')
    ax.plot([r['step'] for r in ev], [r['test_loss'] for r in ev], color=C3, linewidth=2, marker='o', markersize=5, label='held-out loss', markeredgecolor=SURFACE)
    ax.set_xlabel('optimizer steps'); ax.set_ylabel('cross-entropy per answer token (nats)')
    ax.set_title('Answer loss' + (' (with 100 cold-start examples)' if main.endswith('cold') else ''), fontsize=10.5)
    ax.legend()
    ax = axes[1]
    for t, ls in zip(runs, ['-', '--']):
        e = [r for r in rows(t) if r['kind'] == 'eval']
        lab = 'with cold start' if t.endswith('cold') else 'without'
        ax.plot([r['step'] for r in e], [r['hi_bpc'] for r in e], color=C1, linestyle=ls, linewidth=2, marker='o', markersize=4, markeredgecolor=SURFACE, label=f'Hindi, {lab}')
        ax.plot([r['step'] for r in e], [r['en_bpc'] for r in e], color=MUTED, linestyle=ls, linewidth=2, marker='o', markersize=4, markeredgecolor=SURFACE, label=f'English, {lab}')
    ax.set_xlabel('optimizer steps'); ax.set_ylabel('held-out bits per character')
    ax.set_title('What the stage costs the text model', fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.suptitle('Supervised fine-tuning on Hindi instruction pairs', fontsize=12, color=INK)
    save(fig, 'sft')


def chart_grpo(tag='grpo'):
    rs = rows('grpo-' + tag)
    tr = [r for r in rs if r['kind'] == 'train']; ev = [r for r in rs if r['kind'] == 'eval']
    if not tr:
        return
    def sm(v, k=15):
        out, buf = [], []
        for x in v:
            buf.append(x); buf = buf[-k:]
            out.append(sum(buf) / len(buf))
        return out
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    ax = axes[0]
    s = [r for r in tr if r['sent_reward'] is not None]; o = [r for r in tr if r['open_reward'] is not None]
    ax.plot([r['step'] for r in s], sm([r['sent_reward'] for r in s]), color=C1, linewidth=1.8, label='sentiment reward (train batches)')
    ax.plot([r['step'] for r in s], sm([r['sent_format'] for r in s]), color=C1, linewidth=1.2, linestyle=':', label='sentiment: valid JSON share')
    ax.plot([r['step'] for r in o], sm([r['open_reward'] for r in o]), color=C3, linewidth=1.8, label='language reward (train batches)')
    ax.set_xlabel('GRPO steps'); ax.set_ylabel('mean reward, smoothed over 15 steps'); ax.set_ylim(0, 1.02)
    ax.set_title('Training rewards'); ax.legend(loc='lower right', fontsize=9)
    ax = axes[1]
    ax.plot([r['step'] for r in ev], [r['sent_acc'] for r in ev], color=C1, linewidth=2, marker='o', markersize=5, label='sentiment accuracy (held-out)', markeredgecolor=SURFACE)
    ax.plot([r['step'] for r in ev], [r['sent_format'] for r in ev], color=C1, linewidth=1.4, linestyle=':', marker='o', markersize=4, label='valid JSON share (held-out)', markeredgecolor=SURFACE)
    ax.plot([r['step'] for r in ev], [r['lang_reward'] for r in ev], color=C3, linewidth=2, marker='o', markersize=5, label='language reward (held-out prompts)', markeredgecolor=SURFACE)
    ax.set_xlabel('GRPO steps'); ax.set_ylim(0, 1.02); ax.set_title('Held-out evaluation, greedy decoding'); ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('GRPO with two rule-based rewards', x=0.01, ha='left', fontsize=13, fontweight='600')
    fig.tight_layout()
    save(fig, 'grpo')
    # KL and length side by side (two measures, two panels, one axis each)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot([r['step'] for r in tr], sm([r['kl'] for r in tr]), color=C2, linewidth=1.8)
    ax.set_xlabel('GRPO steps'); ax.set_ylabel('nats per token'); ax.set_title('KL from the SFT policy (per token, k3 estimate)', fontsize=10.5)
    ax = axes[1]
    ax.plot([r['step'] for r in tr], sm([r['comp_len'] for r in tr]), color=INK3, linewidth=1.8)
    ax.set_xlabel('GRPO steps'); ax.set_ylabel('tokens'); ax.set_title('Mean completion length (both tasks, smoothed)', fontsize=10.5)
    fig.suptitle('How far the policy moved, and how long its answers got', fontsize=12, color=INK, y=1.03)
    save(fig, 'grpo-kl')


def chart_task():
    rs = rows('task_eval')
    seen = {}
    for r in rs:
        m = r['model'].replace(chr(92), '/')
        tag = 'sft-cold' if m.endswith('/sft-cold') else ('sft' if m.endswith('/sft') else r['tag'])
        if r.get('sent_n', 0) >= seen.get(tag, {}).get('sent_n', 0):
            seen[tag] = r
    order = [('SmolLM2-135M-Instruct', 'SmolLM2-135M-Instruct (stock)'), ('sft-cold', 'this post: 135M after CPT + SFT'), ('grpo', 'this post: + GRPO, first reward'), ('grpo-v2', 'this post: + GRPO, repaired reward'), ('grpo-v2-t07', 'this post: + repaired reward, sampled at 0.7'),
             ('SmolLM2-360M-Instruct', 'SmolLM2-360M-Instruct (stock)'), ('SmolLM2-1.7B-Instruct', 'SmolLM2-1.7B-Instruct (stock)'),
             ('Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct (stock)'), ('Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct (stock)'), ('Qwen3-1.7B', 'Qwen3-1.7B (stock, no thinking)')]
    order = [(t, l) for t, l in order if t in seen]
    if not order:
        return
    y = np.arange(len(order))[::-1]
    acc = [seen[t]['sent_acc'] for t, _ in order]; fmt = [seen[t]['sent_format'] for t, _ in order]
    v2 = all('lang_reward_v2' in seen[t] for t, _ in order)  # once every row carries the repaired reward, show that one
    lang = [seen[t]['lang_reward_v2' if v2 else 'lang_reward'] for t, _ in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    cols = [C1 if t in ('sft-cold', 'grpo', 'grpo-v2', 'grpo-v2-t07') else C3 for t, _ in order]
    ax.barh(y + 0.19, acc, height=0.36, color=cols, label='accuracy (right label in valid JSON)')
    ax.barh(y - 0.19, fmt, height=0.36, color=[MUTED] * len(order), label='valid JSON share')
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in order]); ax.tick_params(axis='y', length=0)
    ax.grid(axis='x'); ax.grid(axis='y', visible=False); ax.set_xlim(0, 1.15)
    for yi, a, f in zip(y, acc, fmt):
        ax.text(a + 0.015, yi + 0.19, f'{a:.2f}', va='center', fontsize=9, color=INK2); ax.text(f + 0.015, yi - 0.19, f'{f:.2f}', va='center', fontsize=9, color=INK3)
    ax.axvline(0.5, color=INK3, linestyle=':', linewidth=1); ax.text(0.505, y.max() + 0.55, 'coin flip', fontsize=8.5, color=INK3)
    ax.set_title('Hindi sentiment, 598 held-out reviews'); ax.legend(loc='upper right', fontsize=9)
    ax = axes[1]
    ax.barh(y, lang, height=0.5, color=cols)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis='y', length=0); ax.grid(axis='x'); ax.grid(axis='y', visible=False); ax.set_xlim(0, 1.15)
    for yi, v in zip(y, lang):
        ax.text(v + 0.015, yi, f'{v:.2f}', va='center', fontsize=9, color=INK2)
    ax.set_title(('Repaired' if v2 else 'First') + ' language reward on 100 held-out prompts, greedy')
    fig.suptitle('Where the 135M lands against stock instruct models (this post in blue)', x=0.01, ha='left', fontsize=13, fontweight='600')
    fig.tight_layout()
    save(fig, 'task')


def chart_coldstart():
    """Held-out sentiment accuracy and valid-JSON share during GRPO from three SFT checkpoints that differ only in
    how many labelled JSON examples were mixed into the instruction data (0, 20, 100)."""
    runs = [('grpo-nocold', '0 cold-start examples', C4), ('grpo-cold20', '20 examples', C3), ('grpo', '100 examples', C1)]
    have = [(t, lab, col) for t, lab, col in runs if rows('grpo-' + t)]
    if len(have) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for t, lab, col in have:
        ev = [r for r in rows('grpo-' + t) if r['kind'] == 'eval' and r['step'] <= 150]  # the two new runs are 150 steps; compare like with like
        axes[0].plot([r['step'] for r in ev], [r['sent_format'] for r in ev], color=col, linewidth=2, marker='o', markersize=5, markeredgecolor=SURFACE, label=lab)
        axes[1].plot([r['step'] for r in ev], [r['sent_acc'] for r in ev], color=col, linewidth=2, marker='o', markersize=5, markeredgecolor=SURFACE, label=lab)
    axes[0].set_title('Valid JSON share, held-out reviews', fontsize=10.5); axes[1].set_title('Sentiment accuracy, held-out reviews', fontsize=10.5)
    for ax in axes:
        ax.set_xlabel('GRPO steps'); ax.set_ylim(-0.02, 1.02); ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('The same GRPO from three starts: how much cold start the reward needs (200 held-out reviews)', fontsize=12, color=INK, y=1.03)
    save(fig, 'coldstart')


def chart_rewards():
    """What each reward taught: the SFT start, GRPO with the first language reward, GRPO with the repaired one,
    on the full held-out sets (task_eval rows that carry both reward versions)."""
    te = {}
    for r in rows('task_eval'):
        m = r['model'].replace(chr(92), '/')
        tag = 'sft-cold' if m.endswith('/sft-cold') else r['tag']
        if 'lang_reward_v2' in r and r.get('sent_n', 0) >= te.get(tag, {}).get('sent_n', 0):
            te[tag] = r
    NL = chr(10)
    want = [('sft-cold', 'after' + NL + 'SFT'), ('grpo', 'GRPO,' + NL + 'first' + NL + 'reward'), ('grpo-v2', 'GRPO,' + NL + 'repaired' + NL + 'reward'), ('grpo-v2-t07', 'repaired,' + NL + 'sampled' + NL + 'at 0.7')]
    have = [(t, lab) for t, lab in want if t in te]
    if len(have) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    x = np.arange(len(have)); w = 0.36
    labs = [lab for _, lab in have]
    ax = axes[0]
    ax.bar(x, [te[t]['sent_acc'] for t, _ in have], width=0.55, color=C1, edgecolor=SURFACE)
    for i, (t, _) in enumerate(have):
        ax.text(i, te[t]['sent_acc'] + 0.015, f"{te[t]['sent_acc']:.3f}", ha='center', fontsize=9, color=INK2)
    ax.set_ylim(0, 1.08); ax.set_title('Sentiment accuracy, 598 held-out reviews', fontsize=10.5)
    ax = axes[1]
    ax.bar(x - w / 2, [te[t]['lang_looping'] for t, _ in have], width=w, color=C3, edgecolor=SURFACE, label='greedy decoding')
    ax.bar(x + w / 2, [te[t]['lang_looping_sampled'] for t, _ in have], width=w, color=C3, alpha=0.5, edgecolor=SURFACE, label='sampled at temperature 1')
    ax.set_ylim(0, 1.0); ax.set_title('Open prompts: share of answers that loop', fontsize=10.5); ax.legend(fontsize=8.5, loc='upper right')
    ax = axes[2]
    ax.bar(x - w / 2, [te[t]['lang_reward_v2'] for t, _ in have], width=w, color=C2, edgecolor=SURFACE, label='greedy decoding')
    ax.bar(x + w / 2, [te[t]['lang_reward_v2_sampled'] for t, _ in have], width=w, color=C2, alpha=0.5, edgecolor=SURFACE, label='sampled at temperature 1')
    ax.set_ylim(0, 1.0); ax.set_title('Open prompts: repaired language reward', fontsize=10.5); ax.set_ylim(0, 1.15); ax.legend(fontsize=8.5, loc='upper left', ncol=2)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    fig.suptitle('What each reward taught, measured on the held-out sets', fontsize=12, color=INK, y=1.03)
    save(fig, 'rewards')


CHARTS = {'rewards': chart_rewards, 'coldstart': chart_coldstart, 'fertility': chart_fertility, 'sizes': chart_bpc_sizes, 'wiki': chart_wiki, 'init': chart_init, 'tokenizer': chart_tokenizer, 'replay': chart_replay,
          'lr': chart_lr, 'main': chart_main, 'scaling': chart_scaling, 'sft': chart_sft, 'grpo': chart_grpo, 'task': chart_task}

if __name__ == '__main__':
    keys = sys.argv[1:] or list(CHARTS)
    for k in keys:
        CHARTS[k]()
