"""
Assemble blogs/blog-1.html for the portfolio from blog/blog-1.src.html + blog/body-*.html, filling every
{{PLACEHOLDER}} from the results files so the prose can never disagree with the data.
"""
import os, re, json, glob, shutil, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io'
RES = os.path.join(LAB, 'results')

bench = [json.loads(l) for l in open(os.path.join(RES, 'bench.jsonl'))]
kld = {r['name']: r for r in json.load(open(os.path.join(RES, 'kld_table.json')))}


def rows(tag, test=None, **match):
    out = []
    for r in bench:
        if r['tag'] != tag: continue
        if test == 'pp' and not r['n_prompt']: continue
        if test == 'tg' and not r['n_gen']: continue
        if any(r.get(k) != v for k, v in match.items()): continue
        out.append(r)
    return out


def one(tag, test, **match):
    rs = rows(tag, test, **match)
    if not rs:
        raise KeyError(f'no row for {tag} {test} {match}')
    return rs[-1]


def ts(r, digits=1):
    return f"{r['avg_ts']:.{digits}f}&nbsp;&plusmn;&nbsp;{r['stddev_ts']:.{digits}f}"


def best(r):
    return max(r['samples_ts'])


V = {}
# --- hardware / ceilings ----------------------------------------------------------------------------
membw = 34.0
for line in open(os.path.join(RES, 'membw.txt')):
    m = re.search(r'aggregate\s+([\d.]+) GB/s', line)
    if m: membw = max(membw, float(m.group(1)))
V['MEMBW'] = f'{membw:.1f}'
V['MEMBW_PCT'] = f'{100 * membw / 44.8:.0f}'
V['CPU_CEIL'] = f'{membw / 0.6335:.1f}'

# --- baselines ----------------------------------------------------------------------------------------
b_cpu_tg = one('pair-cpu-default', 'tg'); b_cpu_pp = one('pair-cpu-default', 'pp')
b_gpu_tg = one('baseline-cuda-default', 'tg'); b_gpu_pp = one('baseline-cuda-default', 'pp')
V['BASE_CPU_PP'] = ts(b_cpu_pp); V['BASE_CPU_TG'] = ts(b_cpu_tg)
V['BASE_CPU_TG_PCT'] = f"{100 * b_cpu_tg['avg_ts'] / 70.7:.0f}"
V['BASE_CUDA_PP'] = ts(b_gpu_pp, 0); V['BASE_CUDA_TG'] = ts(b_gpu_tg)
V['BASE_CUDA_TG_PCT'] = f"{100 * b_gpu_tg['avg_ts'] / 606:.0f}"

# --- threads ------------------------------------------------------------------------------------------
tr = []
for r in rows('threads-sweep-r2', 'tg'):
    pp = [x for x in rows('threads-sweep-r2', 'pp') if x['n_threads'] == r['n_threads']][0]
    tr.append(f"                        <tr><td>{r['n_threads']}</td><td>{ts(pp)}</td><td>{ts(r)}</td></tr>")
V['THREADS_ROWS'] = '\n'.join(tr)
r2 = {r['n_threads']: r for r in rows('threads-sweep-r2', 'tg')}
V['R2_T8_TG'] = f"{r2[8]['avg_ts']:.1f}"; V['R2_T6_TG'] = f"{r2[6]['avg_ts']:.1f}"; V['R2_T16_TG'] = f"{r2[16]['avg_ts']:.1f}"
V['R2_T24_TG'] = f"{r2[24]['avg_ts']:.1f}"; V['R2_T12_SD'] = f"{r2[12]['stddev_ts']:.1f}"; V['R2_T4_TG'] = f"{r2[4]['avg_ts']:.1f}"
V['R2_T16_LOSS'] = f"{100 * (1 - r2[16]['avg_ts'] / r2[8]['avg_ts']):.0f}"; V['R2_T24_LOSS'] = f"{100 * (1 - r2[24]['avg_ts'] / r2[8]['avg_ts']):.0f}"
r2p = {r['n_threads']: r for r in rows('threads-sweep-r2', 'pp')}
V['R2_T8_PP'] = f"{r2p[8]['avg_ts']:.0f}"; V['R2_T16_PP'] = f"{r2p[16]['avg_ts']:.0f}"; V['R2_T24_PP'] = f"{r2p[24]['avg_ts']:.0f}"
# paired pp512 checks, 8 then 16 threads back to back, in the order they were run
tb = rows('tb-paired-r2', 'pp')
pairs = []
for i in range(len(tb) - 1):
    a, b = tb[i], tb[i + 1]
    if a['n_threads'] == 8 and b['n_threads'] == 16 and a['cpu_mask'] == '0x0' and b['cpu_mask'] == '0x0':
        pairs.append(f"{a['avg_ts']:.0f} against {b['avg_ts']:.0f}")
V['TB_PAIRS'] = ', '.join(pairs[:-1]) + ' and ' + pairs[-1]
V['TB_PIN8_PP'] = f"{[r for r in tb if r['n_threads'] == 8 and r['cpu_mask'] == '0x5555'][-1]['avg_ts']:.0f}"
V['TB_PIN16_PP'] = f"{[r for r in tb if r['n_threads'] == 16 and r['cpu_mask'] == '0x5555'][-1]['avg_ts']:.0f}"
V['BASE_GPU_RATIO'] = f"{b_gpu_tg['avg_ts'] / b_cpu_tg['avg_ts']:.0f}"
pq = rows('pair-quiet-r2', 'tg')
q16 = [r['avg_ts'] for r in pq if r['n_threads'] == 16]; q8 = [r['avg_ts'] for r in pq if r['n_threads'] == 8]
V['PQ_16'] = ' and '.join(f"{v:.1f}" for v in q16); V['PQ_8'] = ' and '.join(f"{v:.1f}" for v in q8)
V['PQ_LOSS'] = f"{100 * (1 - sum(q16) / len(q16) / (sum(q8) / len(q8))):.0f}"
V['PQ_24'] = f"{[r['avg_ts'] for r in pq if r['n_threads'] == 24][-1]:.1f}"
V['WORK_LOSS'] = f"{100 * (1 - b_cpu_tg['avg_ts'] / one('pair-cpu-t8', 'tg')['avg_ts']):.0f}"
pin = one('cpumask-p8', 'tg'); pinpp = one('cpumask-p8', 'pp')
V['PIN_BEST_TG'] = f"{pin['avg_ts']:.1f}"; V['PIN_BEST_PP'] = f"{pinpp['avg_ts']:.0f}"
V['PIN_BEST_PCT'] = f"{100 * pin['avg_ts'] / 70.7:.0f}"
V['SWEEP_T8_TG'] = f"{[r for r in rows('threads-sweep','tg') if r['n_threads']==8][0]['avg_ts']:.1f}"
V['QUIET_T8_TG'] = f"{one('paired-head-cpu-r1','tg', model_filename='models/variants/plain-Q8_0.gguf')['avg_ts']:.1f}"
V['QUIET_PIN_TG'] = f"{one('cpu-tb','tg', n_threads=8, cpu_mask='0x5555')['avg_ts']:.1f}"
V['QUIET_4CORE_TG'] = f"{one('cpu-tb','tg', n_threads=8, cpu_mask='0xFFFF')['avg_ts']:.1f}"
V['QUIET_8CORE_PP'] = f"{one('cpu-tb','pp', n_threads=8, cpu_mask='0x5555')['avg_ts']:.0f}"
V['QUIET_4CORE_PP'] = f"{one('cpu-tb','pp', n_threads=8, cpu_mask='0xFFFF')['avg_ts']:.0f}"
ph = {n: one('paired-head-cpu-r1','tg', model_filename=f'models/variants/{n}.gguf')['avg_ts'] for n in ['plain-Q8_0','head-Q6_K-body-Q8_0','head-Q5_K-body-Q8_0','plain-Q6_K']}
V['PH_Q8_R1'] = f"{ph['plain-Q8_0']:.1f}"; V['PH_Q6H_R1'] = f"{ph['head-Q6_K-body-Q8_0']:.1f}"; V['PH_Q5H_R1'] = f"{ph['head-Q5_K-body-Q8_0']:.1f}"; V['PH_Q6_R1'] = f"{ph['plain-Q6_K']:.1f}"
V['PH_Q6H_PCT'] = f"{100*(ph['head-Q6_K-body-Q8_0']/ph['plain-Q8_0']-1):.1f}"; V['PH_Q5H_PCT'] = f"{100*(ph['head-Q5_K-body-Q8_0']/ph['plain-Q8_0']-1):.1f}"; V['PH_Q6_PCT'] = f"{100*(ph['plain-Q6_K']/ph['plain-Q8_0']-1):.1f}"
V['PAIR_STRICT_TG'] = ts(one('pair-cpu-pinned-strict', 'tg'))
V['PAIR_T8_TG'] = ts(one('pair-cpu-t8', 'tg'))
V['ECORE_TG'] = f"{one('cpumask-e8', 'tg')['avg_ts']:.2f}"

# --- KLD table ----------------------------------------------------------------------------------------
order = ['bf16-self', 'plain-Q8_0', 'official-Q8_0', 'head-Q6_K-body-Q8_0', 'head-Q5_K-body-Q8_0', 'plain-Q6_K',
         'head-Q4_K-body-Q8_0', 'head-IQ4_XS-body-Q8_0', 'plain-Q5_K_M', 'attn-Q4_K-ffn-Q8_0', 'attn-Q8_0-ffn-Q4_K',
         'edge-Q8_0-mid-Q4_K', 'ffn-down-Q8-rest-Q4_K', 'head-BF16-body-Q4_K_M', 'head-Q8_0-body-Q4_K_M',
         'plain-Q4_K_M', 'plain-Q4_K_S', 'head-Q4_K-body-Q4_K_M', 'head-Q3_K-body-Q8_0', 'plain-IQ4_XS',
         'plain-Q4_0', 'plain-Q3_K_M', 'plain-Q2_K']
nice = {'bf16-self': 'BF16 (reference vs its own stored logits)', 'plain-Q8_0': 'Q8_0 (requantized from BF16)',
        'official-Q8_0': 'Q8_0 (official Qwen file)', 'head-Q6_K-body-Q8_0': 'Q8_0 body, Q6_K head',
        'head-Q5_K-body-Q8_0': 'Q8_0 body, Q5_K head', 'plain-Q6_K': 'Q6_K', 'head-Q4_K-body-Q8_0': 'Q8_0 body, Q4_K head',
        'head-IQ4_XS-body-Q8_0': 'Q8_0 body, IQ4_XS head', 'plain-Q5_K_M': 'Q5_K_M',
        'attn-Q4_K-ffn-Q8_0': 'Q8_0, attention at Q4_K', 'attn-Q8_0-ffn-Q4_K': 'Q8_0, FFN at Q4_K',
        'edge-Q8_0-mid-Q4_K': 'Q8_0 edges, layers 4-23 at Q4_K', 'ffn-down-Q8-rest-Q4_K': 'Q4_K_M with ffn_down and head at Q8_0',
        'head-BF16-body-Q4_K_M': 'Q4_K_M body, BF16 head', 'head-Q8_0-body-Q4_K_M': 'Q4_K_M body, Q8_0 head',
        'plain-Q4_K_M': 'Q4_K_M', 'plain-Q4_K_S': 'Q4_K_S', 'head-Q4_K-body-Q4_K_M': 'Q4_K_M body, Q4_K head',
        'head-Q3_K-body-Q8_0': 'Q8_0 body, Q3_K head', 'plain-IQ4_XS': 'IQ4_XS', 'plain-Q4_0': 'Q4_0',
        'plain-Q3_K_M': 'Q3_K_M', 'plain-Q2_K': 'Q2_K'}
kr = []
for n in order:
    r = kld[n]
    kr.append(f"                        <tr><td>{nice[n]}</td><td>{r['tensor_bytes']/1e6:.0f}</td><td>{r['head_type']}</td>"
              f"<td>{r['ppl']:.2f}</td><td>{max(r['kld'],0):.4f} &plusmn; {r['kld_err']:.4f}</td><td>{r['kld_p99']:.3f}</td>"
              f"<td>{r['dp_rms']:.1f}%</td><td>{r['same_top']:.1f}%</td></tr>")
V['KLD_ROWS'] = '\n'.join(kr)
V['KLD_Q4KM'] = f"{kld['plain-Q4_K_M']['kld']:.4f}"
V['KLD_Q4KM_Q8HEAD'] = f"{kld['head-Q8_0-body-Q4_K_M']['kld']:.4f}"
V['KLD_Q4KM_BF16HEAD'] = f"{kld['head-BF16-body-Q4_K_M']['kld']:.4f}"
V['KLD_Q8_OWN'] = f"{kld['plain-Q8_0']['kld']:.4f}"
V['KLD_Q8_OFFICIAL'] = f"{kld['official-Q8_0']['kld']:.4f}"

# speed per variant (CPU -t 8 and CUDA), matched by file name through model_filename
def speed(tag, fname, test):
    rs = [r for r in rows(tag, test) if os.path.basename(r['model_filename']) == fname]
    return rs[-1] if rs else None

hr = []
q8 = kld['plain-Q8_0']['tensor_bytes']
for n in ['plain-Q8_0', 'head-Q6_K-body-Q8_0', 'head-Q5_K-body-Q8_0', 'head-Q4_K-body-Q8_0', 'head-IQ4_XS-body-Q8_0', 'head-Q3_K-body-Q8_0']:
    r = kld[n]; f = n + '.gguf'
    c = speed('quant-cpu-r2', f, 'tg'); g = speed('quant-cuda', f, 'tg')
    hr.append(f"                        <tr><td>{r['head_type']}</td><td>{r['tensor_bytes']/1e6:.0f}</td><td>{(q8-r['tensor_bytes'])/1e6:.0f} MB ({100*(q8-r['tensor_bytes'])/q8:.0f}%)</td>"
              f"<td>{r['kld']:.4f}</td><td>{r['same_top']:.1f}%</td><td>{ts(c) if c else '?'}</td><td>{ts(g,0) if g else '?'}</td></tr>")
V['HEAD_ROWS'] = '\n'.join(hr)

br = []
for n in ['plain-Q8_0', 'attn-Q4_K-ffn-Q8_0', 'attn-Q8_0-ffn-Q4_K', 'edge-Q8_0-mid-Q4_K', 'ffn-down-Q8-rest-Q4_K', 'plain-Q4_K_M']:
    r = kld[n]; saved = (q8 - r['tensor_bytes']) / 1e6
    per = (r['kld'] - kld['plain-Q8_0']['kld']) / saved * 100 if saved > 0 else 0
    br.append(f"                        <tr><td>{nice[n]}</td><td>{r['tensor_bytes']/1e6:.0f}</td><td>{saved:.0f} MB</td><td>{r['kld']:.4f}</td><td>{per:.4f}</td></tr>")
V['BLOCK_ROWS'] = '\n'.join(br)

# --- DLL variants -------------------------------------------------------------------------------------
isa = {'alderlake': 'AVX2, FMA, F16C, BMI2, AVX-VNNI', 'haswell': 'AVX2, FMA, F16C, BMI2', 'ivybridge': 'AVX, F16C',
       'sandybridge': 'AVX', 'piledriver': 'AVX, FMA, F16C (AMD)', 'sse42': 'SSE4.2', 'x64': 'SSE2 baseline'}
dr = []
for v in ['alderlake', 'haswell', 'ivybridge', 'sandybridge', 'piledriver', 'sse42', 'x64']:
    q8pp = speed(f'dll-{v}', 'Qwen3-0.6B-Q8_0.gguf', 'pp'); q8tg = speed(f'dll-{v}', 'Qwen3-0.6B-Q8_0.gguf', 'tg')
    q4pp = speed(f'dll-{v}', 'plain-Q4_K_M.gguf', 'pp'); q4tg = speed(f'dll-{v}', 'plain-Q4_K_M.gguf', 'tg')
    dr.append(f"                        <tr><td><code>{v}</code></td><td>{isa[v]}</td><td>{ts(q8pp,0)}</td><td>{ts(q8tg)}</td><td>{ts(q4pp,0)}</td><td>{ts(q4tg)}</td></tr>")
for v in ['skylakex', 'cannonlake', 'icelake', 'cascadelake', 'cooperlake', 'sapphirerapids', 'zen4']:
    dr.append(f"                        <tr><td><code>{v}</code></td><td>AVX-512 family</td><td colspan=\"4\">does not load on this CPU</td></tr>")
V['DLL_ROWS'] = '\n'.join(dr)

# --- GPU section --------------------------------------------------------------------------------------
def q(tag, fname, test):
    r = speed(tag, fname, test)
    return r

gq8 = speed('quant-cuda', 'plain-Q8_0.gguf', 'tg'); gq4 = speed('quant-cuda', 'plain-Q4_K_M.gguf', 'tg')
gq6 = speed('quant-cuda', 'plain-Q6_K.gguf', 'tg'); gq2 = speed('quant-cuda', 'plain-Q2_K.gguf', 'tg')
cq8 = speed('quant-cpu-r2', 'plain-Q8_0.gguf', 'tg'); cq4 = speed('quant-cpu-r2', 'plain-Q4_K_M.gguf', 'tg')
V['GPU_Q8_TG'] = f"{gq8['avg_ts']:.0f}"; V['GPU_Q4KM_TG'] = f"{gq4['avg_ts']:.0f}"
V['GPU_Q6K_TG'] = f"{gq6['avg_ts']:.0f}"; V['GPU_Q2K_TG'] = f"{gq2['avg_ts']:.0f}"; V['GPU_BF16_TG'] = f"{one('quant-cuda-bf16','tg')['avg_ts']:.0f}"
V['CPU_Q4_SPEEDUP'] = f"{cq4['avg_ts'] / cq8['avg_ts']:.2f}"

qs = []
for n in ['plain-Q8_0', 'head-Q6_K-body-Q8_0', 'plain-Q6_K', 'plain-Q5_K_M', 'plain-Q4_K_M', 'plain-Q4_K_S', 'plain-IQ4_XS', 'plain-Q4_0', 'plain-Q3_K_M', 'plain-Q2_K']:
    r = kld[n]; f = n + '.gguf'
    c = speed('quant-cpu-r2', f, 'tg'); g = speed('quant-cuda', f, 'tg'); gp = speed('quant-cuda', f, 'pp')
    qs.append(f"                        <tr><td>{nice[n]}</td><td>{r['tensor_bytes']/1e6:.0f}</td><td>{ts(c) if c else '?'}</td><td>{ts(g,0) if g else '?'}</td><td>{ts(gp,0) if gp else '?'}</td><td>{r['kld']:.4f}</td></tr>")
bf_c = one('quant-cpu-r2-bf16', 'tg'); bf_g = one('quant-cuda-bf16', 'tg'); bf_gp = one('quant-cuda-bf16', 'pp')
qs.insert(0, f"                        <tr><td>BF16</td><td>1192</td><td>{ts(bf_c)}</td><td>{ts(bf_g,0)}</td><td>{ts(bf_gp,0)}</td><td>0 (reference)</td></tr>")
V['QUANT_SPEED_ROWS'] = '\n'.join(qs)

ngl = sorted(rows('ngl-sweep', 'tg'), key=lambda r: r['n_gpu_layers'])
nglpp = {r['n_gpu_layers']: r for r in rows('ngl-sweep', 'pp')}
nr = []
where = lambda n: ('all on CPU' if n == 0 else ('everything, head included' if n >= 29 else f'{n} of 28 layers on GPU'))
for r in ngl:
    n = r['n_gpu_layers']
    nr.append(f"                        <tr><td>{'all' if n >= 29 else n}</td><td>{where(n)}</td><td>{ts(nglpp[n],0)}</td><td>{ts(r)}</td></tr>")
V['NGL_ROWS'] = '\n'.join(nr)
g0 = [r for r in ngl if r['n_gpu_layers'] == 0][0]['avg_ts']; g99 = [r for r in ngl if r['n_gpu_layers'] >= 29][0]['avg_ts']
g12 = [r for r in ngl if r['n_gpu_layers'] == 12][0]['avg_ts']; g28 = [r for r in ngl if r['n_gpu_layers'] == 28][0]['avg_ts']
V['NGL12_PCT'] = f"{100 * (g12 - g0) / (g99 - g0):.0f}"; V['NGL28_TG'] = f"{g28:.0f}"; V['NGL99_TG'] = f"{g99:.0f}"

fa_on = one('fa-kv', 'tg', flash_attn=1); fa_off = one('fa-kv', 'tg', flash_attn=0)
fa_on_pp = one('fa-kv', 'pp', flash_attn=1); fa_off_pp = one('fa-kv', 'pp', flash_attn=0)
kv8 = one('kv-q8', 'tg'); kv8pp = one('kv-q8', 'pp'); kv4 = one('kv-q4', 'tg'); kv4pp = one('kv-q4', 'pp')
V['FAKV_ROWS'] = '\n'.join([
    f"                        <tr><td>Flash attention on, f16 KV (default)</td><td>{ts(fa_on_pp,0)}</td><td>{ts(fa_on)}</td></tr>",
    f"                        <tr><td>Flash attention off</td><td>{ts(fa_off_pp,0)}</td><td>{ts(fa_off)}</td></tr>",
    f"                        <tr><td>FA on, q8_0 K and V cache</td><td>{ts(kv8pp,0)}</td><td>{ts(kv8)}</td></tr>",
    f"                        <tr><td>FA on, q4_0 K and V cache</td><td>{ts(kv4pp,0)}</td><td>{ts(kv4)}</td></tr>"])
V['FA_PP_LOSS'] = f"{100 * (1 - fa_off_pp['avg_ts'] / fa_on_pp['avg_ts']):.0f}"
V['FA_TG_LOSS'] = f"{100 * (1 - fa_off['avg_ts'] / fa_on['avg_ts']):.0f}"

def depth_row(label, tag, **m):
    rs = sorted(rows(tag, 'tg', **m), key=lambda r: r['n_depth'])
    cells = {r['n_depth']: ts(r) for r in rs}
    return f"                        <tr><td>{label}</td>" + ''.join(f"<td>{cells.get(d, '')}</td>" for d in [0, 2048, 8192, 16384]) + "</tr>"
V['DEPTH_ROWS'] = '\n'.join([depth_row('GPU, f16 KV', 'depth-cuda'), depth_row('GPU, q8_0 KV', 'depth-cuda-kvq8'),
                             depth_row('CPU -t 8, FA on', 'depth-cpu'), depth_row('CPU -t 8, FA off', 'depth-cpu-fa-off')])
V['CPU_FA_ON_8K'] = f"{[r for r in rows('depth-cpu','tg') if r['n_depth']==8192][0]['avg_ts']:.1f}"
V['CPU_FA_OFF_8K'] = f"{[r for r in rows('depth-cpu-fa-off','tg') if r['n_depth']==8192][0]['avg_ts']:.1f}"

ub = {r['n_ubatch']: r for r in rows('ubatch-cuda', 'pp')}; cub = {r['n_ubatch']: r for r in rows('ubatch-cpu', 'pp')}
V['UB64_PP'] = f"{ub[64]['avg_ts']:,.0f}"; V['UB512_PP'] = f"{ub[512]['avg_ts']:,.0f}"
V['CPU_UB32_PP'] = f"{cub[32]['avg_ts']:.0f}"; V['CPU_UB512_PP'] = f"{cub[512]['avg_ts']:.0f}"

# --- 8B section ---------------------------------------------------------------------------------------
n8 = {}
for r in bench:
    if r['tag'].startswith('8b-') and r['n_gen']:
        n8[r['tag']] = r
n8pp = {r['tag']: r for r in bench if r['tag'].startswith('8b-') and r['n_prompt']}
def n8row(label, tag, cpu_share):
    return f"                        <tr><td>{label}</td><td>{cpu_share}</td><td>{ts(n8pp[tag],0)}</td><td>{ts(n8[tag])}</td></tr>"
r8 = [n8row('CPU only', '8b-cpu', '100%')]
for n in [16, 24, 28, 30, 32, 33, 34, 35, 36]:
    r8.append(n8row(f'-ngl {n}', f'8b-ngl-{n}', f'{100*(36-n)/36:.0f}% (+ head)'))
r8.append(n8row('-ngl 99 (everything)', '8b-ngl-99', '0%'))
r8.append(n8row('-ngl 99, FFN of layers 0-3 on CPU', '8b-otffn-4', '~8%'))
r8.append(n8row('-ngl 99, FFN of layers 0-7 on CPU', '8b-otffn-8', '~16%'))
r8.append(n8row('-ngl 99, FFN of layers 0-11 on CPU', '8b-otffn-12', '~24%'))
r8.append(n8row('-ngl 99, q8_0 KV cache', '8b-kvq8-ngl99', '0%'))
r8.append(n8row('-ngl 34, q8_0 KV cache', '8b-kvq8-ngl34', '6%'))
V['NGL8B_ROWS'] = '\n'.join(r8)
t99 = n8['8b-ngl-99']['avg_ts']
V['N8_99_TG'] = f"{t99:.1f}"; V['N8_BW'] = f"{t99 * 6.72:.0f}"; V['N8_BW_PCT'] = f"{100 * t99 * 6.72 / 384:.0f}"
V['N8_CPU_TG'] = f"{n8['8b-cpu']['avg_ts']:.1f}"
V['N8_36_34_PCT'] = f"{100 * (1 - n8['8b-ngl-34']['avg_ts'] / n8['8b-ngl-36']['avg_ts']):.0f}"
V['N8_36_32_PCT'] = f"{100 * (1 - n8['8b-ngl-32']['avg_ts'] / n8['8b-ngl-36']['avg_ts']):.0f}"
V['N8_OT4_TG'] = f"{n8['8b-otffn-4']['avg_ts']:.1f}"; V['N8_33_TG'] = f"{n8['8b-ngl-33']['avg_ts']:.1f}"
V['N8_KVQ8_TG'] = f"{n8['8b-kvq8-ngl99']['avg_ts']:.1f}"
fd = rows('8b-fit-depth', 'tg')
def fdt(**m):
    return [r for r in fd if all(r.get(k) == v for k, v in m.items())][-1]['avg_ts']
V['FD_99_D0'] = f"{fdt(n_gpu_layers=99, n_depth=0, type_k='f16'):.1f}"; V['FD_99_D4K'] = f"{fdt(n_gpu_layers=99, n_depth=4096, type_k='f16'):.1f}"
V['FD_99_D8K'] = f"{fdt(n_gpu_layers=99, n_depth=8192, type_k='f16'):.1f}"; V['FD_34_D8K'] = f"{fdt(n_gpu_layers=34, n_depth=8192):.1f}"
V['FD_32_D8K'] = f"{fdt(n_gpu_layers=32, n_depth=8192):.1f}"; V['FD_Q8KV_D8K'] = f"{fdt(n_gpu_layers=99, n_depth=8192, type_k='q8_0'):.1f}"


import fill_blog_part2
fill_blog_part2.compute(V, RES, dict(rows=rows, one=one, ts=ts, speed=speed, b_cpu_tg=b_cpu_tg, b_gpu_tg=b_gpu_tg))


def fill(s):
    def rep(m):
        k = m.group(1)
        if k not in V:
            print('MISSING', k); return m.group(0)
        return str(V[k])
    return re.sub(r'\{\{([A-Z0-9_]+)\}\}', rep, s)


if __name__ == '__main__':
    import sys
    tpl = open(os.path.join(LAB, 'blog', 'blog-1.src.html'), encoding='utf-8').read()
    body = ''.join(open(f, encoding='utf-8').read() for f in sorted(glob.glob(os.path.join(LAB, 'blog', 'body-*.html'))))
    page = tpl.replace('{{BODY}}', body)
    text = re.sub(r'<[^>]+>', ' ', body)
    words = len(re.findall(r"[A-Za-z0-9'&;.-]+", text))
    V['READ_TIME'] = str(max(1, round(words / 230)))
    page = fill(page)
    out = os.path.join(SITE, 'blogs', 'blog-1.html')
    open(out, 'w', encoding='utf-8').write(page)
    res = os.path.join(SITE, 'blogs', 'resources', 'blog-1'); os.makedirs(res, exist_ok=True)
    for f in glob.glob(os.path.join(RES, 'charts', '*.svg')):
        shutil.copy(f, res)
    print(f'wrote {out}: {words} words, ~{V["READ_TIME"]} min')
    left = sorted(set(re.findall(r'\{\{([A-Z0-9_]+)\}\}', page)))
    if left: print('unfilled:', left)
