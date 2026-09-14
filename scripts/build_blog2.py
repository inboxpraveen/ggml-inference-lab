"""Assemble blogs/blog-2.html from blog/blog-2.src.html + blog/b2-*.html. Every {{PLACEHOLDER}} is computed
here from results/, the same discipline as fill_blog.py: no measured number is typed into the prose."""
import os, re, json, glob, shutil, sys

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io'
RES = os.path.join(LAB, 'results')

bench = [json.loads(l) for l in open(os.path.join(RES, 'bench.jsonl'))]
kld = {r['name']: r for r in json.load(open(os.path.join(RES, 'kld_table.json')))}
META = json.load(open(os.path.join(RES, 'zoo_meta.json')))
CONV = json.load(open(os.path.join(RES, 'kld_convergence.json')))
srv = [json.loads(l) for l in open(os.path.join(RES, 'server.jsonl'))]


def rows(tag, test=None, **match):
    out = []
    for r in bench:
        if r['tag'] != tag: continue
        if test == 'pp' and not r['n_prompt']: continue
        if test == 'tg' and not r['n_gen']: continue
        if any(r.get(k) != v for k, v in match.items()): continue
        out.append(r)
    return out


def one(tag, test, fname=None, **match):
    rs = rows(tag, test, **match)
    if fname: rs = [r for r in rs if os.path.basename(r['model_filename']) == fname]
    if not rs: raise KeyError(f'no row for {tag} {test} {fname} {match}')
    return rs[-1]


def ts(r, d=1):
    return f"{r['avg_ts']:.{d}f}&nbsp;&plusmn;&nbsp;{r['stddev_ts']:.{d}f}"


def srv_get(tag, wl):
    rs = [r for r in srv if r['tag'] == tag and r['workload'] == wl]
    return rs[-1]


V = {}
ZOO = [('Qwen3-0.6B-Q8_0.gguf', '../Qwen3-0.6B-Q8_0.gguf', 'Qwen3-0.6B', 'dense decoder, GQA, tied head'),
       ('gemma-3-1b-it-Q8_0.gguf', 'gemma-3-1b-it-Q8_0.gguf', 'Gemma-3-1B', 'dense, 5 local : 1 global attention, tied head'),
       ('granite-3.1-1b-a400m-instruct-Q8_0.gguf', 'granite-3.1-1b-a400m-instruct-Q8_0.gguf', 'Granite-3.1 1B-A400M', 'MoE, 8 of 32 experts per token'),
       ('Falcon-H1-0.5B-Instruct-Q8_0.gguf', 'Falcon-H1-0.5B-Instruct-Q8_0.gguf', 'Falcon-H1-0.5B', 'hybrid, Mamba2 and attention in every layer'),
       ('LFM2-700M-Q8_0.gguf', 'LFM2-700M-Q8_0.gguf', 'LFM2-700M', 'hybrid, 10 short-conv + 6 attention layers'),
       ('mamba-130m-Q8_0.gguf', 'mamba-130m-Q8_0.gguf', 'Mamba-130M', 'pure SSM, no attention')]

# ---------------------------------------------------------------- zoo table
zr = []
for bf, mf, name, fam in ZOO:
    m = META[mf]
    c = one('zoo-cpu', 'tg', bf); cp = one('zoo-cpu', 'pp', bf); g = one('zoo-cuda', 'tg', bf)
    act = m['active_bytes'] / 1e6
    kv = f"{m['kv_bytes_per_token_f16'] / 1024:.0f}" if m['kv_bytes_per_token_f16'] else 'state only'
    zr.append(f"                        <tr><td>{name}</td><td>{fam}</td><td>{m['total_bytes'] / 1e6:.0f}</td><td>{act:.0f}</td><td>{kv}</td>"
              f"<td>{ts(c)}</td><td>{act * c['avg_ts'] / 1e3:.0f}</td><td>{ts(cp, 0)}</td><td>{ts(g, 0)}</td><td>{act * g['avg_ts'] / 1e3:.0f}</td></tr>")
    key = name.split('-')[0].upper().replace('.', '') + ('06' if '0.6' in name else '')
V['ZOO_ROWS'] = '\n'.join(zr)


def eff(tag, bf, mf):
    return META[mf]['active_bytes'] / 1e9 * one(tag, 'tg', bf)['avg_ts']


for bf, mf, name, fam in ZOO:
    k = {'Qwen3-0.6B': 'Q06', 'Gemma-3-1B': 'GEM', 'Granite-3.1 1B-A400M': 'GRA', 'Falcon-H1-0.5B': 'FAL', 'LFM2-700M': 'LFM', 'Mamba-130M': 'MAM'}[name]
    V[f'{k}_CPU_TG'] = f"{one('zoo-cpu', 'tg', bf)['avg_ts']:.0f}"; V[f'{k}_CUDA_TG'] = f"{one('zoo-cuda', 'tg', bf)['avg_ts']:.0f}"
    V[f'{k}_CPU_EFF'] = f"{eff('zoo-cpu', bf, mf):.0f}"; V[f'{k}_CUDA_EFF'] = f"{eff('zoo-cuda', bf, mf):.0f}"
    V[f'{k}_CUDA_PCT'] = f"{100 * eff('zoo-cuda', bf, mf) / 384:.0f}"
    V[f'{k}_ACTIVE_MB'] = f"{META[mf]['active_bytes'] / 1e6:.0f}"; V[f'{k}_TOTAL_MB'] = f"{META[mf]['total_bytes'] / 1e6:.0f}"
    V[f'{k}_CPU_PP'] = f"{one('zoo-cpu', 'pp', bf)['avg_ts']:.0f}"

# ---------------------------------------------------------------- head share
hs = []
for mf, name in [('gemma-3-1b-it-Q8_0.gguf', 'Gemma-3-1B'), ('mamba-130m-Q8_0.gguf', 'Mamba-130M'), ('../Qwen3-0.6B-Q8_0.gguf', 'Qwen3-0.6B'),
                 ('LFM2-700M-Q8_0.gguf', 'LFM2-700M'), ('granite-3.1-1b-a400m-instruct-Q8_0.gguf', 'Granite-3.1 1B-A400M'),
                 ('../Qwen3-8B-Q6_K.gguf', 'Qwen3-8B'), ('Falcon-H1-0.5B-Instruct-Q8_0.gguf', 'Falcon-H1-0.5B')]:
    m = META[mf]; share = 100 * m['head_bytes'] / m['active_bytes']
    hs.append(f"                        <tr><td>{name}</td><td>{m['vocab']:,}</td><td>{m['n_embd']}</td><td>{'tied' if m['tied'] else 'separate'}</td><td>{m['head_bytes'] / 1e6:.0f}</td><td>{share:.0f}%</td></tr>")
    V['HEAD_' + {'Gemma-3-1B': 'GEM', 'Mamba-130M': 'MAM', 'Qwen3-0.6B': 'Q06', 'LFM2-700M': 'LFM', 'Granite-3.1 1B-A400M': 'GRA', 'Qwen3-8B': 'Q8', 'Falcon-H1-0.5B': 'FAL'}[name]] = f"{share:.0f}"
V['HEAD_ROWS'] = '\n'.join(hs)

# ---------------------------------------------------------------- depth
dr = []
for bf, mf, name, fam in ZOO:
    cells = [f"<td>{name}</td>"]
    for tag in ['zoo-depth-cpu', 'zoo-depth-cuda']:
        rs = sorted([r for r in rows(tag, 'tg') if os.path.basename(r['model_filename']) == bf], key=lambda r: r['n_depth'])
        base = rs[0]['avg_ts']
        for r in rs:
            cells.append(f"<td>{r['avg_ts']:.0f}" + (f" ({100 * r['avg_ts'] / base:.0f}%)" if r['n_depth'] else '') + "</td>")
    dr.append('                        <tr>' + ''.join(cells) + '</tr>')
V['DEPTH_ROWS'] = '\n'.join(dr)


def dpct(tag, bf, depth):
    rs = [r for r in rows(tag, 'tg') if os.path.basename(r['model_filename']) == bf]
    b = [r for r in rs if r['n_depth'] == 0][-1]['avg_ts']; d = [r for r in rs if r['n_depth'] == depth][-1]['avg_ts']
    return f"{100 * d / b:.0f}"


V['Q06_CPU_8K_PCT'] = dpct('zoo-depth-cpu', 'Qwen3-0.6B-Q8_0.gguf', 8192); V['GRA_CPU_8K_PCT'] = dpct('zoo-depth-cpu', 'granite-3.1-1b-a400m-instruct-Q8_0.gguf', 8192)
V['GEM_CPU_8K_PCT'] = dpct('zoo-depth-cpu', 'gemma-3-1b-it-Q8_0.gguf', 8192); V['FAL_CPU_8K_PCT'] = dpct('zoo-depth-cpu', 'Falcon-H1-0.5B-Instruct-Q8_0.gguf', 8192)
V['LFM_CPU_8K_PCT'] = dpct('zoo-depth-cpu', 'LFM2-700M-Q8_0.gguf', 8192); V['Q06_CUDA_8K_PCT'] = dpct('zoo-depth-cuda', 'Qwen3-0.6B-Q8_0.gguf', 8192)
V['GEM_CUDA_8K_PCT'] = dpct('zoo-depth-cuda', 'gemma-3-1b-it-Q8_0.gguf', 8192); V['GRA_CUDA_8K_PCT'] = dpct('zoo-depth-cuda', 'granite-3.1-1b-a400m-instruct-Q8_0.gguf', 8192)
V['Q06_KV_8K_MB'] = f"{META['../Qwen3-0.6B-Q8_0.gguf']['kv_bytes_per_token_f16'] * 8192 / 1e6:.0f}"
V['Q06_KV_KB'] = f"{META['../Qwen3-0.6B-Q8_0.gguf']['kv_bytes_per_token_f16'] / 1024:.0f}"
V['GRA_KV_KB'] = f"{META['granite-3.1-1b-a400m-instruct-Q8_0.gguf']['kv_bytes_per_token_f16'] / 1024:.0f}"
V['Q8B_KV_KB'] = f"{META['../Qwen3-8B-Q6_K.gguf']['kv_bytes_per_token_f16'] / 1024:.0f}"

# ---------------------------------------------------------------- battery, power, load
q = 'Qwen3-0.6B-Q8_0.gguf'; e = 'Qwen3-8B-Q6_K.gguf'
V['BAT_CPU_TG_AC'] = f"{one('zoo-cpu', 'tg', q)['avg_ts']:.1f}"; V['BAT_CPU_TG_BAT'] = f"{one('zoo-cpu-battery', 'tg', q)['avg_ts']:.1f}"
V['BAT_CPU_PP_AC'] = f"{one('zoo-cpu', 'pp', q)['avg_ts']:.0f}"; V['BAT_CPU_PP_BAT'] = f"{one('zoo-cpu-battery', 'pp', q)['avg_ts']:.0f}"
V['BAT_GPU_06_AC'] = f"{one('zoo-cuda', 'tg', q)['avg_ts']:.0f}"; V['BAT_GPU_06_BAT'] = f"{one('zoo-cuda-battery', 'tg', q)['avg_ts']:.0f}"
V['BAT_GPU_8B_AC'] = f"{one('zoo-cuda-probe', 'tg', e)['avg_ts']:.1f}"; V['BAT_GPU_8B_BAT'] = f"{one('zoo-cuda-probe-battery', 'tg', e)['avg_ts']:.1f}"
V['BAT_GPU_8B_RATIO'] = f"{float(V['BAT_GPU_8B_AC']) / float(V['BAT_GPU_8B_BAT']):.1f}"
V['BAT_GPU_06_RATIO'] = f"{float(V['BAT_GPU_06_AC']) / float(V['BAT_GPU_06_BAT']):.1f}"
pc_bat = open(os.path.join(RES, 'precheck-battery.txt'), encoding='utf-8').read(); pc_ac = open(os.path.join(RES, 'precheck-ac.txt'), encoding='utf-8').read()
V['PL_BATTERY'] = re.search(r'power limit now ([\d.]+) W', pc_bat).group(1).rstrip('0').rstrip('.')
V['PL_AC'] = re.search(r'power limit now ([\d.]+) W', pc_ac).group(1).rstrip('0').rstrip('.')
V['PL_DEFAULT'] = re.search(r'default ([\d.]+) W', pc_ac).group(1).rstrip('0').rstrip('.')
V['PRECHECK_AC'] = pc_ac.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
V['PC_STREAM_BW'] = re.search(r'streaming read, 8 process\(es\):\s+([\d.]+) GB/s', pc_ac).group(1)
V['PC_EFF_BW'] = re.search(r'= ([\d.]+) GB/s effective', pc_ac).group(1)
V['PC_STREAM_BW_BAT'] = re.search(r'streaming read, 8 process\(es\):\s+([\d.]+) GB/s', pc_bat).group(1)
V['BOOST_8B_ALONE'] = ts(one('boost-alone', 'tg', e)); V['BOOST_8B_LOAD'] = ts(one('boost-cpuload', 'tg', e))
V['BOOST_06_ALONE'] = ts(one('boost-alone', 'tg', q)); V['BOOST_06_LOAD'] = ts(one('boost-cpuload', 'tg', q))
V['B1_8B_GPU'] = f"{one('8b-ngl-99', 'tg', e)['avg_ts']:.1f}"; V['B1_06_GPU'] = f"{one('baseline-cuda-default', 'tg', q)['avg_ts']:.0f}"
V['NOW_8B_GPU'] = f"{one('zoo-cuda-probe', 'tg', e)['avg_ts']:.1f}"; V['NOW_06_GPU'] = f"{one('zoo-cuda', 'tg', q)['avg_ts']:.0f}"
V['NOW_8B_EFF'] = f"{6.72 * one('zoo-cuda-probe', 'tg', e)['avg_ts']:.0f}"; V['NOW_8B_PCT'] = f"{100 * 6.72 * one('zoo-cuda-probe', 'tg', e)['avg_ts'] / 384:.0f}"
V['B1_8B_PCT'] = f"{100 * 6.72 * one('8b-ngl-99', 'tg', e)['avg_ts'] / 384:.0f}"
V['GPU_8B_GAP_PCT'] = f"{100 * (one('zoo-cuda-probe', 'tg', e)['avg_ts'] / one('8b-ngl-99', 'tg', e)['avg_ts'] - 1):.0f}"

# ---------------------------------------------------------------- blog-1 carry-overs
V['CPU_8B_TG'] = f"{one('8b-cpu', 'tg', e)['avg_ts']:.1f}"
V['FD_99_D4K'] = f"{one('8b-fit-depth', 'tg', e, n_gpu_layers=99, n_depth=4096, type_k='f16')['avg_ts']:.1f}"
V['FD_99_D8K'] = f"{one('8b-fit-depth', 'tg', e, n_gpu_layers=99, n_depth=8192, type_k='f16')['avg_ts']:.1f}"
V['FD_Q8KV_D8K'] = f"{one('8b-fit-depth', 'tg', e, n_gpu_layers=99, n_depth=8192, type_k='q8_0')['avg_ts']:.1f}"
V['DLL_SSE42_PP'] = f"{one('dll-sse42', 'pp', q)['avg_ts']:.0f}"; V['DLL_ALDER_PP'] = f"{one('dll-alderlake', 'pp', q)['avg_ts']:.0f}"
V['DLL_SSE42_TG'] = f"{one('dll-sse42', 'tg', q)['avg_ts']:.1f}"; V['DLL_ALDER_TG'] = f"{one('dll-alderlake', 'tg', q)['avg_ts']:.1f}"
V['DLL_SSE42_PP_Q4'] = f"{one('dll-sse42', 'pp', 'plain-Q4_K_M.gguf')['avg_ts']:.0f}"; V['DLL_ALDER_PP_Q4'] = f"{one('dll-alderlake', 'pp', 'plain-Q4_K_M.gguf')['avg_ts']:.0f}"
V['DLL_SSE42_TG_Q4'] = f"{one('dll-sse42', 'tg', 'plain-Q4_K_M.gguf')['avg_ts']:.1f}"; V['DLL_ALDER_TG_Q4'] = f"{one('dll-alderlake', 'tg', 'plain-Q4_K_M.gguf')['avg_ts']:.1f}"
r2 = {r['n_threads']: r for r in rows('threads-sweep-r2', 'tg')}
V['T8_TG'] = f"{r2[8]['avg_ts']:.1f}"; V['T16_TG'] = f"{r2[16]['avg_ts']:.1f}"; V['T24_TG'] = f"{r2[24]['avg_ts']:.1f}"
V['SPEC_CPU_NONE'] = f"{srv_get('srv-cpu-none', 'edit')['tok_s_mean']:.1f}"; V['SPEC_CPU_NG'] = f"{srv_get('srv-cpu-ngram-simple-3-8', 'edit')['tok_s_mean']:.1f}"
V['SPEC_CPU_NONE_FREE'] = f"{srv_get('srv-cpu-none', 'free')['tok_s_mean']:.1f}"; V['SPEC_CPU_NG_FREE'] = f"{srv_get('srv-cpu-ngram-simple-3-8', 'free')['tok_s_mean']:.1f}"
V['SPEC_GPU_NONE'] = f"{srv_get('srv-cuda-none-rerun', 'edit')['tok_s_mean']:.0f}"; V['SPEC_GPU_NG'] = f"{srv_get('srv-cuda-ngram-simple-3-16-rerun', 'edit')['tok_s_mean']:.0f}"
V['SPEC_8B_NONE'] = f"{srv_get('srv-8b-none', 'edit')['tok_s_mean']:.1f}"; V['SPEC_8B_DRAFT'] = f"{srv_get('srv-8b-draft-0.6b', 'edit')['tok_s_mean']:.1f}"
V['SPEC_8B_NONE_FREE'] = f"{srv_get('srv-8b-none', 'free')['tok_s_mean']:.1f}"; V['SPEC_8B_DRAFT_FREE'] = f"{srv_get('srv-8b-draft-0.6b', 'free')['tok_s_mean']:.1f}"
V['KLD_Q8'] = f"{kld['plain-Q8_0']['kld']:.4f}"; V['KLD_Q6'] = f"{kld['plain-Q6_K']['kld']:.4f}"; V['KLD_Q4KM'] = f"{kld['plain-Q4_K_M']['kld']:.4f}"
V['KLD_Q3KM'] = f"{kld['plain-Q3_K_M']['kld']:.4f}"; V['KLD_Q2K'] = f"{kld['plain-Q2_K']['kld']:.3f}"; V['KLD_HEADQ6'] = f"{kld['head-Q6_K-body-Q8_0']['kld']:.4f}"
V['KLD_HEADQ4'] = f"{kld['head-Q4_K-body-Q8_0']['kld']:.4f}"; V['KLD_ATTNQ4'] = f"{kld['attn-Q4_K-ffn-Q8_0']['kld']:.4f}"; V['KLD_FFNQ4'] = f"{kld['attn-Q8_0-ffn-Q4_K']['kld']:.4f}"
V['KLD_Q4KM_Q8HEAD'] = f"{kld['head-Q8_0-body-Q4_K_M']['kld']:.4f}"; V['TOP1_Q4KM'] = f"{kld['plain-Q4_K_M']['same_top']:.0f}"; V['TOP1_Q2K'] = f"{kld['plain-Q2_K']['same_top']:.0f}"

# ---------------------------------------------------------------- KLD convergence and the short recipe
cr = []
for n, lab in [('plain-Q8_0', 'Q8_0'), ('head-Q6_K-body-Q8_0', 'Q8_0 body, Q6_K head'), ('plain-Q6_K', 'Q6_K'), ('plain-Q4_K_M', 'Q4_K_M'), ('plain-Q3_K_M', 'Q3_K_M')]:
    c = CONV[n]
    cr.append(f"                        <tr><td>{lab}</td>" + ''.join(f"<td>{c[str(k)]['kld']:.4f} &plusmn; {c[str(k)]['kld_err']:.4f}</td>" for k in [5, 20, 40, 200]) + "</tr>")
V['KLD_CONV_ROWS'] = '\n'.join(cr)


def kld20(name):
    t = open(os.path.join(RES, 'kld', f'{name}-base20.txt'), encoding='utf-8', errors='replace').read()
    return re.search(r'Mean\s+KLD:\s+([\d.]+) ±\s+([\d.]+)', t).groups()


a4, e4 = kld20('plain-Q4_K_M'); a8, e8 = kld20('plain-Q8_0')
V['KLD20_Q4KM'] = f"{float(a4):.4f}"; V['KLD20_Q4KM_ERR'] = f"{float(e4):.4f}"; V['KLD20_Q8'] = f"{float(a8):.4f}"; V['KLD20_Q8_ERR'] = f"{float(e8):.4f}"
V['KLD20_Q4KM_DIFF'] = f"{abs(100 * (float(a4) / kld['plain-Q4_K_M']['kld'] - 1)):.0f}"
V['KLD20_Q8_DIFF'] = f"{abs(100 * (float(a8) / kld['plain-Q8_0']['kld'] - 1)):.0f}"
V['KLD20_BASE_GB'] = f"{os.path.getsize(os.path.join(RES, 'kld', 'base-bf16-c512-20.kld')) / 1e9:.1f}"

# ---------------------------------------------------------------- encoders, T5
er = []
for r in rows('zoo-embed-cpu', 'pp') + rows('zoo-embed-cuda', 'pp'):
    dev = 'RTX 5060' if r['n_gpu_layers'] == 99 else f"CPU, -t {r['n_threads']}"
    er.append(f"                        <tr><td>{dev}</td><td>{r['n_prompt']}</td><td>{ts(r, 0)}</td></tr>")
V['EMBED_ROWS'] = '\n'.join(er)
V['EMB_CPU8_128'] = f"{one('zoo-embed-cpu', 'pp', n_threads=8, n_prompt=128)['avg_ts']:.0f}"; V['EMB_CPU16_128'] = f"{one('zoo-embed-cpu', 'pp', n_threads=16, n_prompt=128)['avg_ts']:.0f}"
V['EMB_CPU4_128'] = f"{one('zoo-embed-cpu', 'pp', n_threads=4, n_prompt=128)['avg_ts']:.0f}"; V['EMB_GPU_512'] = f"{one('zoo-embed-cuda', 'pp', n_prompt=512)['avg_ts']:.0f}"
V['EMB_CPU8_512'] = f"{one('zoo-embed-cpu', 'pp', n_threads=8, n_prompt=512)['avg_ts']:.0f}"


def t5(fname):
    t = open(os.path.join(RES, fname), encoding='utf-8', errors='replace').read()
    evals = [float(x) for x in re.findall(r'eval time =\s+[\d.]+ ms /\s+\d+ runs\s+\(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second', t)]
    totals = [float(x) for x in re.findall(r'total time =\s+([\d.]+) ms', t)]
    return min(evals), max(evals), min(totals), max(totals)


a, b, c, d = t5('t5-cpu.txt'); V['T5_CPU_LO'] = f"{a:.0f}"; V['T5_CPU_HI'] = f"{b:.0f}"; V['T5_CPU_TOTAL_LO'] = f"{c:.0f}"; V['T5_CPU_TOTAL_HI'] = f"{d:.0f}"
a, b, c, d = t5('t5-cuda.txt'); V['T5_GPU_LO'] = f"{a:.0f}"; V['T5_GPU_HI'] = f"{b:.0f}"
V['T5_ACTIVE_MB'] = f"{META['flan-t5-small.Q8_0.gguf']['active_bytes'] / 1e6:.0f}"; V['T5_TOTAL_MB'] = f"{META['flan-t5-small.Q8_0.gguf']['total_bytes'] / 1e6:.0f}"
V['T5_CPU_EFF'] = f"{META['flan-t5-small.Q8_0.gguf']['active_bytes'] / 1e9 * float(V['T5_CPU_HI']):.0f}"

# ---------------------------------------------------------------- old CPU kernels
def oc(v, q, test):
    return one(f'oldcpu-{v}', test, f'plain-{q}.gguf')['avg_ts']
V['OLD_SSE_Q8_TG'] = f"{oc('sse42', 'Q8_0', 'tg'):.0f}"; V['OLD_SSE_Q40_TG'] = f"{oc('sse42', 'Q4_0', 'tg'):.0f}"
V['OLD_SSE_Q4KM_TG'] = f"{oc('sse42', 'Q4_K_M', 'tg'):.0f}"; V['OLD_SSE_IQ4_TG'] = f"{oc('sse42', 'IQ4_XS', 'tg'):.0f}"
V['OLD_SSE_Q4KM_PP'] = f"{oc('sse42', 'Q4_K_M', 'pp'):.0f}"; V['OLD_SSE_Q8_PP'] = f"{oc('sse42', 'Q8_0', 'pp'):.0f}"; V['OLD_SSE_Q40_PP'] = f"{oc('sse42', 'Q4_0', 'pp'):.0f}"
V['OLD_AVX_Q4KM_TG'] = f"{oc('sandybridge', 'Q4_K_M', 'tg'):.0f}"; V['OLD_AVX_IQ4_TG'] = f"{oc('sandybridge', 'IQ4_XS', 'tg'):.0f}"
V['OLD_ALDER_Q40_TG'] = f"{oc('alderlake', 'Q4_0', 'tg'):.0f}"; V['OLD_ALDER_IQ4_TG'] = f"{oc('alderlake', 'IQ4_XS', 'tg'):.0f}"
V['MEMBW_PCT_OF_PEAK'] = f"{100 * float(V['PC_STREAM_BW']) / 44.8:.0f}"
V['OLD_AVX_Q40_TG'] = f"{oc('sandybridge', 'Q4_0', 'tg'):.0f}"
V['EMB_GPU_RATIO_128'] = f"{one('zoo-embed-cuda', 'pp', n_prompt=128)['avg_ts'] / one('zoo-embed-cpu', 'pp', n_threads=8, n_prompt=128)['avg_ts']:.0f}"
V['EMB_GPU_RATIO_512'] = f"{one('zoo-embed-cuda', 'pp', n_prompt=512)['avg_ts'] / one('zoo-embed-cpu', 'pp', n_threads=8, n_prompt=512)['avg_ts']:.0f}"

# ---------------------------------------------------------------- assemble
tpl = open(os.path.join(LAB, 'blog', 'blog-2.src.html'), encoding='utf-8').read()
body = '\n'.join(open(f, encoding='utf-8').read() for f in sorted(glob.glob(os.path.join(LAB, 'blog', 'b2-*.html'))))
page = tpl.replace('{{BODY}}', body)


def fill(s):
    return re.sub(r'\{\{([A-Z0-9_]+)\}\}', lambda m: V.get(m.group(1), m.group(0)), s)


page = fill(page)
words = len(re.sub(r'<[^>]+>', ' ', page).split())
V['READ_TIME'] = str(max(1, round(words / 230)))
page = fill(page)
out = os.path.join(SITE, 'blogs', 'blog-2.html')
open(out, 'w', encoding='utf-8').write(page)
res = os.path.join(SITE, 'blogs', 'resources', 'blog-2'); os.makedirs(res, exist_ok=True)
for f in glob.glob(os.path.join(RES, 'charts2', '*.svg')):
    shutil.copy(f, res)
for name in ['kld-vs-bytes', 'speculative']:
    shutil.copy(os.path.join(RES, 'charts', name + '.svg'), res)
print(f'wrote {out}: {words} words, ~{V["READ_TIME"]} min')
left = sorted(set(re.findall(r'\{\{([A-Z0-9_]+)\}\}', page)))
if left: print('unfilled:', left)
