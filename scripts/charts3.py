"""Figures for the Cerebras post. Same surface and palette as charts.py. Nothing here is measured on the
laptop; every chart says in its title whether it is a vendor number, an independent measurement, or
arithmetic from published specs (results/cerebras_sources.json via wafer_model.py)."""
import os, sys, datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
from charts import LAB, SURFACE, INK, INK2, INK3, GRID, C1, C2, C3, C4, MUTED  # noqa
import wafer_model as wm

OUT = os.path.join(LAB, 'results', 'charts3')
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.svg'), bbox_inches='tight', pad_inches=0.25)
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', pad_inches=0.25, dpi=160)
    plt.close(fig)
    print('wrote', name)


def d(s):
    s = s[:10]
    if len(s) == 7: s += '-15'
    if len(s) == 4: s += '-07-01'
    return dt.date.fromisoformat(s)


# ------------------------------------------------------------------------------------------- vendor + independent: timeline
def chart_timeline():
    rows = [r for r in wm.SRC['speeds'] if r['layers']]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_yscale('log')
    for r in rows:
        cer = r['who'].startswith('Cerebras')
        col = C1 if cer else C3
        mk = 'o' if r['kind'] == 'independent' else ('s' if r['kind'] == 'vendor' else 'D')
        ax.scatter([d(r['date'])], [r['tps']], s=54, color=col, marker=mk, zorder=3, edgecolor=SURFACE, linewidth=0.8)
    labels = {  # (model, who) -> (dx, dy, ha)
        ('Llama 3.1 8B', 'Cerebras'): (6, -13, 'left'), ('Llama 3.1 70B', 'Cerebras'): (6, -12, 'left'),
        ('Llama 3.1 405B', 'Cerebras'): (6, -12, 'left'), ('DeepSeek R1 Distill 70B', 'Cerebras'): (6, 6, 'left'),
        ('Llama 4 Scout', 'Cerebras'): (-6, 8, 'right'), ('Llama 4 Maverick', 'Cerebras'): (6, -13, 'left'),
        ('Llama 4 Maverick', 'NVIDIA DGX B200'): (6, -12, 'left'), ('Qwen3-235B-A22B', 'Cerebras'): (6, -13, 'left'),
        ('Qwen3 Coder 480B', 'Cerebras'): (6, -13, 'left'), ('gpt-oss-120b', 'Cerebras'): (6, 6, 'left'),
        ('gpt-oss-120b', 'NVIDIA GB200 (Baseten)'): (6, -12, 'left'), ('Kimi K2.6', 'Cerebras'): (6, -12, 'left'),
        ('gpt-oss-120b', 'Cerebras CS-4'): (-6, 6, 'right'), ('gpt-oss-120b', 'Cerebras public API'): (6, -12, 'left'),
    }
    seen = set()
    for r in rows:
        key = (r['model'], r['who'])
        if key in seen:
            key2 = key + (r['date'],)
        seen.add(key)
        dx, dy, ha = labels.get(key, (6, 6, 'left'))
        txt = r['model'] if r['who'] == 'Cerebras' else f"{r['model']} ({r['who'].replace('Cerebras ', '')})"
        if r['model'] == 'Llama 3.1 70B' and r['spec']: txt = '70B, speculative decoding on'; dx, dy, ha = 6, 8, 'left'
        ax.annotate(txt, (d(r['date']), r['tps']), xytext=(dx, dy), textcoords='offset points', fontsize=8.2, color=INK2, ha=ha)
    g = wm.SRC['gpu_baselines_2024']
    for k, lab, dy in [('h100_8b_tps', 'H100 cloud, 8B (as cited by Cerebras)', 6), ('h100_70b_tps', 'H100 cloud, 70B', -12)]:
        ax.scatter([d(g[k]['date'])], [g[k]['v']], s=54, color=C3, marker='s', zorder=3, edgecolor=SURFACE)
        ax.annotate(lab, (d(g[k]['date']), g[k]['v']), xytext=(6, dy), textcoords='offset points', fontsize=8.2, color=INK2)
    ax.set_ylim(80, 8000)
    ax.set_ylabel('published output tokens per second, one user')
    ax.set_title('Published per-user decode speed, dated (Cerebras in blue, NVIDIA GPU systems in amber)')
    ax.legend(handles=[Line2D([], [], marker='o', linestyle='', color=INK2, label='independent measurement (Artificial Analysis)'),
                       Line2D([], [], marker='s', linestyle='', color=INK2, label='vendor statement'),
                       Line2D([], [], marker='D', linestyle='', color=INK2, label='as cited by a competitor')], loc='upper left', fontsize=8.5)
    ax.grid(True, axis='y', which='both')
    fig.autofmt_xdate()
    save(fig, 'timeline')


# ------------------------------------------------------------------------------------------- computed: the line at both ends
def chart_far_end():
    mk = 'llama31_70b'
    systems = [('this laptop, one DDR5 DIMM, 34 GB/s', wm.v('gpu', 'laptop_bw_gbs') * 1e9, None, INK3),
               ('RTX 5060 Laptop, 384 GB/s', wm.v('gpu', 'rtx5060_bw_gbs') * 1e9, None, INK3),
               ('8 x H100, 3.35 TB/s each', 8 * wm.h100_bw, ('H100 cloud, as cited Aug 2024', wm.SRC['gpu_baselines_2024']['h100_70b_tps']['v']), C3),
               ('8 x B200, 8 TB/s each', 8 * wm.b200_bw, None, C3),
               ('576 x Groq LPU, 80 TB/s each', 576 * wm.v('gpu', 'groq_bw_tbs') * 1e12, ('Groq, 8-bit, Aug 2024', 300), C2),
               ('4 x WSE-3, 21 PB/s each', 4 * wm.SRAM_BW, ('Cerebras Aug 2024 / Oct 2024', 450, 2100), C1),
               ('4 x WSE-3 Turbo, 43 PB/s each', 4 * wm.T_SRAM_BW, None, C1)]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    y = np.arange(len(systems))[::-1]
    for yi, (lab, bw, ach, col) in zip(y, systems):
        c = wm.ceiling(bw, mk)
        ax.barh(yi, c, height=0.58, color=col, alpha=0.35)
        ax.text(c * 1.15, yi, f'{c:,.0f}' if c >= 10 else f'{c:.2g}', va='center', fontsize=9.5, color=INK)
        if ach:
            for a in ach[1:]:
                ax.scatter([a], [yi], s=60, color=col, zorder=4, edgecolor=SURFACE, linewidth=0.8)
            a = ach[-1]
            ax.annotate(f"{ach[0]}: {' then '.join(f'{x:,}' for x in ach[1:])} ({100 * a / c:.2g}% of the line)", (a, yi), xytext=(0, 12), textcoords='offset points', fontsize=8.5, color=INK2, ha='center')
    ax.set_xscale('log'); ax.set_xlim(0.1, 8e6)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in systems], fontsize=9.5)
    ax.tick_params(axis='y', length=0); ax.grid(True, axis='x', which='major'); ax.grid(False, axis='y')
    ax.set_xlabel('tokens per second if only bandwidth mattered: bandwidth / 141 GB (Llama 3.1 70B at 16-bit)')
    ax.set_title('The same line, at both ends. Bars are bandwidth / bytes; dots are what was actually reported (computed from spec sheets)')
    save(fig, 'far-end')


# ------------------------------------------------------------------------------------------- vendor: SRAM by generation
def chart_generations():
    gens = [('WSE-1\n2019, 16 nm', wm.v('wafer', 'wse1_sram_gb'), wm.v('wafer', 'wse1_transistors_t'), wm.v('wafer', 'wse1_cores')),
            ('WSE-2\n2021, 7 nm', wm.v('wafer', 'wse2_sram_gb'), wm.v('wafer', 'wse2_transistors_t'), wm.v('wafer', 'wse2_cores')),
            ('WSE-3\n2024, 5 nm', wm.v('wafer', 'wse3_sram_gb'), wm.v('wafer', 'wse3_transistors_t'), wm.v('wafer', 'wse3_cores')),
            ('WSE-3 Turbo\n2026, 5 nm', wm.v('wafer', 'wse3_sram_gb'), wm.v('wafer', 'wse3_transistors_t'), wm.v('wafer', 'wse3_cores'))]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(gens))
    ax.bar(x, [g[1] for g in gens], width=0.55, color=C1)
    for xi, g in zip(x, gens):
        ax.text(xi, g[1] + 1, f'{g[1]} GB', ha='center', fontsize=10.5, color=INK)
        ax.text(xi, 2, f'{g[2]:g} T transistors\n{g[3] / 1e3:.0f}K cores', ha='center', va='bottom', fontsize=8.5, color=SURFACE)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in gens], fontsize=9.5)
    ax.set_ylim(0, 56); ax.set_ylabel('on-wafer SRAM (GB)')
    ax.set_title('SRAM per wafer across generations: transistors tripled, memory grew 10% over the last node (vendor figures)')
    ax.annotate('same 44 GB;\nthe Turbo raises clocks,\nnot capacity', (3, 44), xytext=(-150, 30), textcoords='offset points', fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle='-', color=INK3, shrinkB=8))
    save(fig, 'generations')


# ------------------------------------------------------------------------------------------- computed: wafers per model
def chart_wafers():
    models = ['llama31_8b', 'llama31_70b', 'gptoss_120b', 'qwen3_235b', 'llama31_405b', 'maverick', 'kimi_k2']
    stated = {'llama31_8b': 'stated: 1', 'llama31_70b': 'stated: 4', 'kimi_k2': 'stored 4-bit (stated)'}
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(models)); w = 0.26
    for i, (bits, col) in enumerate([(16, C1), (8, C2), (4, C4)]):
        vals = [wm.wafers_needed(m, bits) for m in models]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=col, label=f'{bits}-bit weights')
        for xi, val in zip(x + (i - 1) * w, vals):
            ax.text(xi, val + 0.6, str(val), ha='center', fontsize=8.5, color=INK2)
    for xi, m in zip(x, models):
        if m in stated:
            ax.text(xi, -6.5, stated[m], ha='center', fontsize=8.2, color=INK3)
    ax.set_xticks(x); ax.set_xticklabels([wm.M[m]['label'] for m in models], fontsize=9.2)
    ax.set_ylim(0, 60); ax.set_ylabel('wafers needed (44 GB each, 85% usable)')
    ax.set_title('Wafers a model needs just to hold its weights, by precision (computed from parameter counts)')
    ax.legend(loc='upper left')
    save(fig, 'wafers-per-model')


# ------------------------------------------------------------------------------------------- computed from published: per layer
def chart_per_layer():
    rows = [r for r in wm.SRC['speeds'] if r['layers']]
    rows.sort(key=lambda r: wm.implied_us_per_layer(r['tps'], r['layers']))
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        val = wm.implied_us_per_layer(r['tps'], r['layers'])
        cer = r['who'].startswith('Cerebras')
        col = C1 if cer else C3
        ax.barh(yi, val, height=0.62, color=col, alpha=1.0 if r['spec'] is False else 0.55, hatch='' if r['spec'] is False else '////', edgecolor=SURFACE)
        who = '' if r['who'] == 'Cerebras' else f" ({r['who'].replace('Cerebras ', '')})"
        ax.text(val + 0.5, yi, f"{val:.1f} µs   {r['model']}{who}, {r['tps']:,} tok/s, {r['date'][:7]}", va='center', fontsize=8.6, color=INK)
    ax.set_yticks([]); ax.set_xlim(0, 60); ax.grid(True, axis='x'); ax.grid(False, axis='y')
    ax.set_xlabel('microseconds per layer per output token = 1e6 / (tok/s x layers)')
    ax.set_title('What each published speed implies per transformer layer (arithmetic on vendor and Artificial Analysis numbers)')
    ax.legend(handles=[Patch(color=C1, label='Cerebras, no speculative decoding'), Patch(color=C1, alpha=0.55, hatch='////', label='Cerebras, speculative decoding on'),
                       Patch(color=C3, alpha=0.55, hatch='////', label='NVIDIA Blackwell systems, speculative decoding on')], loc='upper right', fontsize=8.5)
    save(fig, 'per-layer')


# ------------------------------------------------------------------------------------------- model: where a layer's time goes
def chart_budget():
    devs = [('8 x H100 (TP8), 16-bit', 'h100'), ('8 x B200 (TP8), 16-bit', 'b200'), ('4 x WSE-3, 16-bit', 'wse3'), ('4 x WSE-3 Turbo, 16-bit', 'wse3t')]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(devs))[::-1]
    for yi, (lab, dv) in zip(y, devs):
        b, c, s = wm.layer_budget(dv)
        left = 0
        for val, col, name in [(b, C2, 'bytes'), (c, C4, 'compute'), (s, C3, 'sync and latency')]:
            ax.barh(yi, val, left=left, height=0.58, color=col, edgecolor=SURFACE, linewidth=1)
            left += val
        ax.text(left + 1, yi, f'{left:.0f} µs per layer, {1e6 / (left * 80):,.0f} tok/s at 80 layers', va='center', fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([dv[0] for dv in devs], fontsize=9.5); ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 135); ax.grid(True, axis='x'); ax.grid(False, axis='y')
    ax.set_xlabel('microseconds per layer per decode step, Llama 3.1 70B, one user, no speculative decoding')
    ax.set_title('Where a layer\'s time goes (a model, fitted to the two published points without speculative decoding)')
    ax.legend(handles=[Patch(color=C2, label='weights bytes / bandwidth'), Patch(color=C4, label='FLOPs / dense compute'), Patch(color=C3, label='synchronisation and latency (fitted)')], loc='lower right', fontsize=8.5)
    save(fig, 'layer-budget')


# ------------------------------------------------------------------------------------------- computed: KV against SRAM
def chart_kv():
    models = ['gptoss_120b', 'kimi_k2', 'llama31_8b', 'qwen3_235b', 'maverick', 'llama31_70b', 'llama31_405b']
    ctx = [(8192, C4), (32768, C2), (131072, C1)]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(models)); w = 0.26
    for i, (n, col) in enumerate(ctx):
        vals = [wm.kv_per_token(m) * n / 1e9 for m in models]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=col, label=f'{n // 1024}K context')
        for xi, val in zip(x + (i - 1) * w, vals):
            ax.text(xi, val * 1.12, f'{val:.1f}' if val < 10 else f'{val:.0f}', ha='center', fontsize=8, color=INK2)
    ax.axhline(wm.SRAM_B / 1e9, color=INK2, linewidth=1.4, linestyle=(0, (6, 4)))
    ax.text(-0.45, wm.SRAM_B / 1e9 * 1.12, 'one whole wafer, 44 GB', color=INK2, fontsize=9, ha='left')
    ax.set_yscale('log'); ax.set_ylim(0.1, 200)
    ax.set_xticks(x); ax.set_xticklabels([wm.M[m]['label'] for m in models], fontsize=9.2)
    ax.set_ylabel('KV cache per user (GB, 16-bit)')
    ax.set_title('KV cache for one user, by context length, against the wafer (computed from published model configs)')
    ax.legend(loc='upper left')
    save(fig, 'kv-vs-sram')


# ------------------------------------------------------------------------------------------- computed: break-even users
def chart_breakeven():
    prices = np.linspace(0.25, 2.0, 100)
    tps = 2100
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for capex, lab, col in [(wm.v('wafer', 'bom_per_wafer_rack_usd'), 'hardware at the estimated $450k BOM per wafer', C1),
                            (wm.v('wafer', 'list_price_per_system_usd'), 'hardware at a $2.5M list price per system', C2)]:
        u = [wm.users_to_break_even(p, tps, 4, capex) for p in prices]
        ax.plot(prices, u, color=col, linewidth=2, label=lab)
        for p in [0.60, 1.20]:
            uu = wm.users_to_break_even(p, tps, 4, capex)
            ax.scatter([p], [uu], color=col, s=40, zorder=3)
            ax.annotate(f'{uu:.0f}', (p, uu), xytext=(5, 4), textcoords='offset points', fontsize=9, color=INK2)
    ax.set_xlabel('price per million output tokens (USD)')
    ax.set_ylabel('concurrent users needed, every hour, to cover cost')
    ax.set_title('Llama 70B on four wafers at 2,100 tok/s per user: users needed to pay for the machine (arithmetic, assumptions in the text)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 180)
    save(fig, 'breakeven')


if __name__ == '__main__':
    which = sys.argv[1:] or ['timeline', 'farend', 'gens', 'wafers', 'perlayer', 'budget', 'kv', 'breakeven']
    fns = {'timeline': chart_timeline, 'farend': chart_far_end, 'gens': chart_generations, 'wafers': chart_wafers,
           'perlayer': chart_per_layer, 'budget': chart_budget, 'kv': chart_kv, 'breakeven': chart_breakeven}
    for w in which:
        fns[w]()
