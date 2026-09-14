"""Figures for the follow-up post. Same surface and palette as charts.py. Every chart says in its title
whether it is measured on this laptop, computed from GGUF metadata, or computed from published spec sheets."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(__file__))
from charts import LAB, SURFACE, INK, INK2, INK3, GRID, C1, C2, C3, C4, MUTED, bench_rows  # noqa

OUT = os.path.join(LAB, 'results', 'charts2')
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.svg'), bbox_inches='tight', pad_inches=0.25)
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', pad_inches=0.25, dpi=160)
    plt.close(fig)
    print('wrote', name)

META = json.load(open(os.path.join(LAB, 'results', 'zoo_meta.json')))
ZOO = [  # file, short label, family, colour
    ('../Qwen3-0.6B-Q8_0.gguf', 'Qwen3-0.6B', 'dense decoder', C1),
    ('gemma-3-1b-it-Q8_0.gguf', 'Gemma-3-1B', 'dense, local/global attention', C1),
    ('granite-3.1-1b-a400m-instruct-Q8_0.gguf', 'Granite-3.1 1B-A400M', 'MoE, 8 of 32 experts', C2),
    ('Falcon-H1-0.5B-Instruct-Q8_0.gguf', 'Falcon-H1-0.5B', 'hybrid Mamba2 + attention', C3),
    ('LFM2-700M-Q8_0.gguf', 'LFM2-700M', 'hybrid short-conv + attention', C3),
    ('mamba-130m-Q8_0.gguf', 'Mamba-130M', 'pure SSM', C4),
]
BENCH_NAME = {'../Qwen3-0.6B-Q8_0.gguf': 'Qwen3-0.6B-Q8_0.gguf'}


def bench_file(fn):
    return os.path.basename(BENCH_NAME.get(fn, fn))


def rows_for(rows, tag, fn, test):
    return [r for r in rows if r['tag'] == tag and os.path.basename(r['model_filename']) == bench_file(fn)
            and ((r['n_gen'] > 0) if test == 'tg' else (r['n_prompt'] > 0))]


# ------------------------------------------------------------------------------------------- computed: ceilings
def chart_ceilings():
    """tok/s ceiling = bandwidth / bytes per token, for bandwidth classes people actually own (spec sheet)."""
    classes = [('DDR3-1333, 2 ch (2010 desktop)', 21.3, INK3), ('LPDDR4X, thin client / mini PC', 34.1, INK3),
               ('DDR5-5600, 1 DIMM (this laptop)', 44.8, C1), ('DDR5-5600, 2 DIMMs', 89.6, C1),
               ('Apple M4 (unified)', 120, C4), ('Apple M4 Max', 546, C4),
               ('RTX 5060 Laptop, GDDR7', 384, C2), ('RTX 4090, GDDR6X', 1008, C2)]
    x = np.logspace(np.log10(0.1), np.log10(80), 200)  # GB per token
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for lab, bw, col in classes:
        y = bw / x
        ax.plot(x, y, color=col, linewidth=1.6 if col != INK3 else 1.2, alpha=0.95)
        xi = 76
        ax.annotate(f'{lab}, {bw:g} GB/s', (xi, bw / xi), xytext=(0, 3), textcoords='offset points', fontsize=8.5, color=col if col != INK3 else INK2, ha='right', va='bottom')
    # measured points from this laptop, both devices, bandwidth-bound cases only
    rows = bench_rows()
    def last(tag, fname, **m):
        rs = [r for r in rows if r['tag'] == tag and os.path.basename(r['model_filename']) == fname and r['n_gen'] > 0 and all(r.get(k) == v for k, v in m.items())]
        return rs[-1]['avg_ts']
    pts = [(0.6335, last('zoo-cpu', 'Qwen3-0.6B-Q8_0.gguf'), 'Qwen3-0.6B Q8_0, CPU -t 8', C1),
           (6.72, last('8b-cpu', 'Qwen3-8B-Q6_K.gguf'), 'Qwen3-8B Q6_K, CPU -t 8', C1),
           (6.72, last('zoo-cuda-probe', 'Qwen3-8B-Q6_K.gguf'), 'Qwen3-8B Q6_K, RTX 5060', C2)]
    for gb, tps, lab, col in pts:
        ax.scatter([gb], [tps], s=70, color=col, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax.annotate(lab, (gb, tps), xytext=(8, -3), textcoords='offset points', fontsize=9, color=INK)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.1, 80); ax.set_ylim(0.2, 3000)
    ax.set_xlabel('bytes read per generated token (GB): weights, plus KV cache at depth')
    ax.set_ylabel('tokens per second, upper bound')
    ax.set_title('The ceiling for any machine: bandwidth divided by bytes (spec-sheet lines, measured dots)')
    ax.grid(axis='x')
    for v, t in [(1, '1 GB'), (4, '4 GB'), (8, '8 GB'), (40, '40 GB')]:
        ax.axvline(v, color=GRID, linewidth=1)
    save(fig, 'ceilings')


# ------------------------------------------------------------------------------------------- computed: bytes by role
def chart_arch_bytes():
    """Where the bytes of one decode step live, per architecture, from the GGUF tensors."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    labels, y = [], np.arange(len(ZOO))[::-1]
    seg_cols = {'attention': C1, 'ffn / active experts': C2, 'ssm / conv': C3, 'head': C4, 'other': MUTED}
    for yi, (fn, lab, fam, _) in zip(y, ZOO):
        m = META[fn]; roles = m['roles']
        attn = sum(b for k, b in roles.items() if 'attn' in k and 'norm' not in k)
        exps = sum(b for k, b in roles.items() if '_exps' in k)
        ffn_dense = sum(b for k, b in roles.items() if 'ffn' in k and '_exps' not in k and 'norm' not in k and 'inp' not in k)
        ssm = sum(b for k, b in roles.items() if 'ssm' in k or 'shortconv' in k)
        head = m['head_bytes']
        active_exps = exps * m['n_expert_used'] / m['n_expert'] if m['n_expert'] else 0
        other = m['active_bytes'] - attn - ffn_dense - active_exps - ssm - (head if m['tied'] else head)
        segs = [('attention', attn), ('ffn / active experts', ffn_dense + active_exps), ('ssm / conv', ssm), ('head', head), ('other', max(other, 0))]
        left = 0
        for name, b in segs:
            if b <= 0: continue
            ax.barh(yi, b / 1e6, left=left, height=0.58, color=seg_cols[name], edgecolor=SURFACE, linewidth=1.5)
            left += b / 1e6
        inactive = (exps - active_exps) / 1e6
        if inactive > 0:
            ax.barh(yi, inactive, left=left, height=0.58, color=SURFACE, edgecolor=C2, hatch='///', linewidth=1)
            ax.text(left + inactive + 12, yi, f'{left:.0f} MB read of {left + inactive:.0f} in the file', va='center', color=INK2, fontsize=9)
        else:
            ax.text(left + 12, yi, f'{left:.0f} MB', va='center', color=INK2, fontsize=9)
        labels.append(f'{lab}\n{fam}')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.tick_params(axis='y', length=0); ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_xlim(0, 1950)
    ax.set_xlabel('MB read per decode step (Q8_0 files, from the tensor list)')
    ax.set_title('Where a decode step spends its bytes, by architecture (computed from GGUF metadata)')
    ax.legend(handles=[Patch(color=c, label=n) for n, c in seg_cols.items() if n != 'other'] + [Patch(facecolor=SURFACE, edgecolor=C2, hatch='///', label='experts not routed to')],
              loc='lower right', ncol=3)
    save(fig, 'arch-bytes')


# ------------------------------------------------------------------------------------------- computed: KV per token
def chart_kv():
    """KV cache or state per 1K tokens of context, f16, from metadata for the zoo and from published configs for big models."""
    items = [  # label, KB per token, growth note, colour, source
        ('Llama-2-7B (MHA, 32 kv heads)', 2 * 32 * 32 * 128 * 2 / 1024, 'grows', INK3, 'config'),
        ('Llama-3-8B (GQA, 8 kv heads)', 2 * 32 * 8 * 128 * 2 / 1024, 'grows', INK3, 'config'),
        ('Qwen3-8B (GQA, 8 kv heads)', META['../Qwen3-8B-Q6_K.gguf']['kv_bytes_per_token_f16'] / 1024, 'grows', C1, 'gguf'),
        ('Qwen3-0.6B (GQA, 8 kv heads)', META['../Qwen3-0.6B-Q8_0.gguf']['kv_bytes_per_token_f16'] / 1024, 'grows', C1, 'gguf'),
        ('Granite-3.1 1B-A400M (GQA)', META['granite-3.1-1b-a400m-instruct-Q8_0.gguf']['kv_bytes_per_token_f16'] / 1024, 'grows', C2, 'gguf'),
        ('Gemma-3-1B, 4 global layers', 2 * 4 * 1 * 256 * 2 / 1024, 'grows; the 22 local layers stop at 512 tokens', C1, 'gguf'),
        ('DeepSeek-V3 (MLA, 576-wide latent)', 61 * 576 * 2 / 1024, 'grows', INK3, 'config'),
        ('LFM2-700M, 6 attention layers', META['LFM2-700M-Q8_0.gguf']['kv_bytes_per_token_f16'] / 1024, 'grows; conv layers are fixed', C3, 'gguf'),
        ('Falcon-H1-0.5B, attention half', META['Falcon-H1-0.5B-Instruct-Q8_0.gguf']['kv_bytes_per_token_f16'] / 1024, 'grows; SSM state is fixed', C3, 'gguf'),
        ('Mamba-130M (SSM state)', 0, 'fixed size, does not grow', C4, 'gguf'),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(items))[::-1]
    for yi, (lab, kb, note, col, src) in zip(y, items):
        ax.barh(yi, kb, height=0.58, color=col)
        ax.text(max(kb, 0) + 6, yi, f'{kb:.0f} KB/token, {kb * 8192 / 1024:.0f} MB at 8K   ({note})' if kb else 'constant state, 0 per token', va='center', color=INK2, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9.5)
    ax.tick_params(axis='y', length=0); ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_xlim(0, 900); ax.set_xlabel('KV cache bytes per token of context, f16 (KB)')
    ax.set_title('What each token of context costs: the attention type decides it (grey: published configs; colour: GGUF metadata)')
    save(fig, 'kv-per-token')


# ------------------------------------------------------------------------------------------- computed: MoE totals
def chart_moe():
    models = [('Granite-3.1 1B-A400M', 1.33, 0.40), ('Qwen3-30B-A3B', 30.5, 3.3), ('gpt-oss-20b', 21, 3.6), ('Mixtral 8x7B', 46.7, 12.9),
              ('Llama 4 Scout', 109, 17), ('gpt-oss-120b', 117, 5.1), ('DeepSeek-V3', 671, 37)]
    bpw = 4.85  # Q4_K_M-class bits per weight
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    y = np.arange(len(models))[::-1]
    for yi, (lab, tot, act) in zip(y, models):
        tb, ab = tot * bpw / 8, act * bpw / 8
        ax.barh(yi, tb, height=0.58, color=SURFACE, edgecolor=C2, hatch='///', linewidth=1)
        ax.barh(yi, ab, height=0.58, color=C2)
        ax.text(tb * 1.12, yi, f'{ab:.1f} GB read per token, {tb:.0f} GB to hold  ({100 * act / tot:.0f}%)', va='center', color=INK2, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([m[0] for m in models], fontsize=9.5)
    ax.tick_params(axis='y', length=0); ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_xscale('log'); ax.set_xlim(0.1, 6000)
    ax.set_xlabel('GB at ~4.85 bits per weight (log scale)')
    ax.set_title('Mixture of experts: RAM must hold the whole file, bandwidth only pays for the active part (published parameter counts)')
    ax.legend(handles=[Patch(color=C2, label='read per token (shared + routed experts)'), Patch(facecolor=SURFACE, edgecolor=C2, hatch='///', label='must fit in memory')], loc='upper right')
    save(fig, 'moe-active')


# ------------------------------------------------------------------------------------------- measured: zoo speed
def chart_zoo_speed(tag_cpu='zoo-cpu', tag_cuda='zoo-cuda'):
    """Effective bandwidth = bytes actually read per token x tok/s, against the device's line."""
    rows = bench_rows()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw=dict(wspace=0.55))
    for ax, tag, title, bw, note, bw2 in [(axes[0], tag_cpu, 'CPU, -t 8', 34.0, 'streaming test 33 to 34, DIMM spec 44.8', 44.8),
                                          (axes[1], tag_cuda, 'RTX 5060 Laptop', 384, 'spec sheet, 384 GB/s', None)]:
        labels, vals, cols, txt = [], [], [], []
        for fn, lab, fam, col in ZOO:
            rs = rows_for(rows, tag, fn, 'tg')
            if not rs: continue
            r = rs[-1]; gb = META[fn]['active_bytes'] / 1e9
            labels.append(lab); vals.append(gb * r['avg_ts']); cols.append(col); txt.append(f"{r['avg_ts']:.0f} tok/s x {gb * 1000:.0f} MB")
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, height=0.58, color=cols)
        for yi, v, t in zip(y, vals, txt):
            ax.text((bw + bw * 0.03) if abs(v - bw) < bw * 0.2 else v + bw * 0.02, yi, f'{v:.0f} GB/s  ({t})', va='center', color=INK2, fontsize=8.5)
        ax.axvline(bw, color=INK2, linewidth=1.2, linestyle=(0, (5, 4)))
        if bw2: ax.axvline(bw2, color=INK3, linewidth=1.2, linestyle=(0, (2, 3)))
        ax.text(bw2 or bw, len(labels) - 0.35, note, color=INK2, fontsize=8.5, ha='right', va='bottom')
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5); ax.tick_params(axis='y', length=0)
        ax.grid(axis='x'); ax.grid(axis='y', visible=False); ax.set_xlim(0, bw * 1.75)
        ax.set_title(title); ax.set_xlabel('effective bandwidth: bytes read per token x tokens per second')
    fig.suptitle('Six architectures against the bandwidth line: MoE at its active bytes (tg128, measured)', x=0.01, ha='left', fontsize=12, fontweight='600', color=INK, y=1.02)
    save(fig, 'zoo-speed')


def chart_zoo_depth(tag='zoo-depth-cpu', device='CPU, -t 8'):
    rows = bench_rows()
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for fn, lab, fam, col in ZOO:
        rs = sorted(rows_for(rows, tag, fn, 'tg'), key=lambda r: r['n_depth'])
        if not rs: continue
        base = rs[0]['avg_ts']
        ax.plot([r['n_depth'] for r in rs], [100 * r['avg_ts'] / base for r in rs], color=col, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                linestyle='-' if fam.startswith('dense') else ('--' if 'MoE' in fam else '-.'))
        ax.annotate(f'{lab} ({base:.0f} tok/s at 0)', (rs[-1]['n_depth'], 100 * rs[-1]['avg_ts'] / base), xytext=(6, 0), textcoords='offset points', fontsize=8.5, color=INK2, va='center')
    ax.set_xticks([0, 2048, 8192]); ax.set_xticklabels(['0', '2K', '8K'])
    ax.set_xlim(-200, 11500); ax.set_ylim(0, 128)
    ax.set_xlabel('tokens already in context'); ax.set_ylabel('decode speed, % of the empty-context speed')
    ax.set_title(f'How each architecture slows down as context fills ({device}, measured)')
    save(fig, 'zoo-depth' if 'cpu' in tag else 'zoo-depth-cuda')


def chart_battery():
    """AC vs battery, paired: the pre-check finding."""
    rows = bench_rows()
    def get(tag, fn, test, **m):
        rs = [r for r in rows_for(rows, tag, fn, test) if all(r.get(k) == v for k, v in m.items())]
        return rs[-1]['avg_ts'] if rs else 0
    q = '../Qwen3-0.6B-Q8_0.gguf'; e = 'Qwen3-8B-Q6_K.gguf'
    pairs = [('0.6B CPU decode, -t 8', get('zoo-cpu', q, 'tg'), get('zoo-cpu-battery', q, 'tg')),
             ('0.6B CPU prompt, -t 8', get('zoo-cpu', q, 'pp'), get('zoo-cpu-battery', q, 'pp')),
             ('0.6B GPU decode', get('zoo-cuda', q, 'tg'), get('zoo-cuda-battery', q, 'tg')),
             ('8B GPU decode', get('zoo-cuda-probe', e, 'tg'), get('zoo-cuda-probe-battery', e, 'tg'))]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.6), gridspec_kw=dict(wspace=0.5))
    for ax, (lab, ac, bat) in zip(axes, pairs):
        ax.bar([0, 1], [ac, bat], width=0.62, color=[C1, MUTED])
        for xi, v in enumerate([ac, bat]):
            ax.text(xi, v * 1.03, f'{v:.0f}', ha='center', color=INK, fontsize=10, fontweight='600')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['plugged in', 'battery'], fontsize=9)
        ax.set_title(lab, fontsize=10.5); ax.set_ylim(0, max(ac, bat) * 1.22); ax.tick_params(axis='x', length=0)
    fig.suptitle('Same laptop, same commands, one cable: the pre-check that matters most (tok/s, measured)', x=0.01, ha='left', fontsize=12, fontweight='600', color=INK, y=1.04)
    save(fig, 'battery')


def chart_oldcpu():
    """Quant choice when the CPU has no AVX2: the sse42 / sandybridge kernels an old desktop would load, versus alderlake."""
    rows = bench_rows()
    quants = ['plain-Q8_0.gguf', 'plain-Q6_K.gguf', 'plain-Q5_K_M.gguf', 'plain-Q4_K_M.gguf', 'plain-IQ4_XS.gguf', 'plain-Q4_0.gguf']
    qlab = ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'IQ4_XS', 'Q4_0']
    sets = [('sse42', 'SSE4.2 only (2008 to 2010 Core 2 / Nehalem)', C3), ('sandybridge', 'AVX (2011 to 2012 Sandy / Ivy Bridge)', C2), ('alderlake', 'AVX2 + VNNI (this chip, native)', C1)]
    def get(v, q, test):
        rs = [r for r in rows if r['tag'] == f'oldcpu-{v}' and os.path.basename(r['model_filename']) == q and ((r['n_gen'] > 0) if test == 'tg' else (r['n_prompt'] > 0))]
        return rs[-1]['avg_ts'] if rs else 0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw=dict(wspace=0.3))
    x = np.arange(len(quants)); w = 0.26
    for ax, test, title in [(axes[0], 'tg', 'Decode, tg128, -t 8 (tok/s)'), (axes[1], 'pp', 'Prompt processing, pp256 (tok/s)')]:
        for i, (v, lab, col) in enumerate(sets):
            vals = [get(v, q, test) for q in quants]
            ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=col, label=lab)
            if test == 'tg':
                for xi, val in zip(x + (i - 1) * w, vals):
                    ax.text(xi, val + 1.5, f'{val:.0f}', ha='center', color=INK2, fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(qlab); ax.tick_params(axis='x', length=0)
        ax.set_title(title); ax.set_ylim(0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle('Same CPU, same files, three instruction sets: which quant is fastest depends on the kernels you get (measured)', x=0.01, ha='left', fontsize=12, fontweight='600', color=INK, y=1.02)
    save(fig, 'oldcpu')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['ceilings', 'arch', 'kv', 'moe']
    for t in todo:
        {'ceilings': chart_ceilings, 'arch': chart_arch_bytes, 'kv': chart_kv, 'moe': chart_moe, 'zoo': chart_zoo_speed,
         'depth': chart_zoo_depth, 'depthcuda': lambda: chart_zoo_depth('zoo-depth-cuda', 'RTX 5060 Laptop'), 'battery': chart_battery, 'oldcpu': chart_oldcpu}[t]()
