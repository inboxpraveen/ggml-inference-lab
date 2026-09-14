"""Assemble blogs/blog-3.html from blog/blog-3.src.html + blog/b3-*.html. Every {{PLACEHOLDER}} comes from
results/cerebras_sources.json (dated, sourced numbers) or from wafer_model.py (arithmetic on them). No number
is typed into the prose, so a sentence cannot disagree with a chart or a source."""
import os, re, glob, shutil, json, datetime as dt, sys
sys.path.insert(0, os.path.dirname(__file__))
import wafer_model as wm

LAB = wm.LAB
SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io'
SRC = wm.SRC
V = {}


def fmt(x):
    if isinstance(x, bool): return str(x)
    if isinstance(x, int): return f'{x:,}'
    if isinstance(x, float): return f'{x:,.2f}'.rstrip('0').rstrip('.') if x != int(x) else f'{int(x):,}'
    return str(x)


# ---------------------------------------------------------------- every sourced number, verbatim
PREFIX = {'wafer': 'WAFER', 'gpu': 'GPU', 'papers': 'PAPERS', 'business': 'B', 'extra': 'X'}
for sec, pre in PREFIX.items():
    for k, e in SRC[sec].items():
        V[f'{pre}_{k.upper()}'] = fmt(e['v'])
V['SOURCE_COUNT'] = str(len({e['src'] for sec in ['wafer', 'gpu', 'papers', 'business', 'extra', 'gpu_baselines_2024'] for e in SRC[sec].values()} | {r['src'] for r in SRC['speeds']}))

# ---------------------------------------------------------------- speeds by name
def sp(model, who='Cerebras', date=None):
    rs = [r for r in SRC['speeds'] if r['model'] == model and r['who'] == who and (date is None or r['date'].startswith(date))]
    assert rs, (model, who, date)
    return rs[-1]


S = {'S_8B': sp('Llama 3.1 8B'), 'S_70B_AUG': sp('Llama 3.1 70B', date='2024-08'), 'S_70B_OCT': sp('Llama 3.1 70B', date='2024-10'),
     'S_405B': sp('Llama 3.1 405B'), 'S_MAV': sp('Llama 4 Maverick'), 'S_MAV_NV': sp('Llama 4 Maverick', 'NVIDIA DGX B200'),
     'S_OSS': sp('gpt-oss-120b', date='2025'), 'S_OSS_NV': sp('gpt-oss-120b', 'NVIDIA GB200 (Baseten)'), 'S_KIMI': sp('Kimi K2.6'),
     'S_OSS_CS4': sp('gpt-oss-120b', 'Cerebras CS-4'), 'S_OSS_AA': sp('gpt-oss-120b', 'Cerebras public API'),
     'S_SCOUT': sp('Llama 4 Scout'), 'S_QWEN': sp('Qwen3-235B-A22B'), 'S_CODER': sp('Qwen3 Coder 480B'), 'S_R1': sp('DeepSeek R1 Distill 70B')}
for k, r in S.items():
    V[k] = f"{r['tps']:,}"
V['S_H100_70B'] = fmt(SRC['gpu_baselines_2024']['h100_70b_tps']['v'])
V['S_405B_100K'] = fmt(SRC['extra']['s405b_100k_tps']['v'])
V['TTFT_405B_MS'] = fmt(SRC['extra']['ttft_405b_ms']['v'])
V['S_OSS_AA_TTFT'] = f"{SRC['extra']['aa_gptoss_ttft_s']['v']:.2f}"
V['RATIO_MAV'] = f"{S['S_MAV']['tps'] / S['S_MAV_NV']['tps']:.1f}"
V['RATIO_OSS'] = f"{S['S_OSS']['tps'] / S['S_OSS_NV']['tps']:.1f}"

# speed table rows
def kind_label(r):
    return {'vendor': 'vendor', 'independent': 'independent', 'community': 'as cited by a competitor'}[r['kind']]


assume = {
    ('Llama 3.1 8B', 'Cerebras'): '16-bit weights, one wafer, no draft model yet.',
    ('Llama 3.1 70B', 'Cerebras', '2024-08'): '16-bit, four systems, no draft model. The cleanest number in the table.',
    ('Llama 3.1 70B', 'Cerebras', '2024-10'): 'Speculative decoding switched on; the post says plus or minus 20%.',
    ('Llama 3.1 405B', 'Cerebras'): '16-bit, 1K prompt; 539 at 100K context. System count never stated.',
    ('DeepSeek R1 Distill 70B', 'Cerebras'): 'Same architecture as the 70B above; reasoning workload.',
    ('Llama 4 Scout', 'Cerebras'): '109B MoE, 17B active.',
    ('Llama 4 Maverick', 'Cerebras'): '400B MoE, 17B active; measured by Artificial Analysis.',
    ('Llama 4 Maverick', 'NVIDIA DGX B200'): 'Eight GPUs, FP8, EAGLE-3 draft model; measured by Artificial Analysis.',
    ('Qwen3-235B-A22B', 'Cerebras'): 'MoE, 22B active.',
    ('Qwen3 Coder 480B', 'Cerebras'): 'MoE, 35B active; users reported multi-second time to first token.',
    ('gpt-oss-120b', 'Cerebras'): 'MoE, 5.1B active; experts are natively 4-bit.',
    ('gpt-oss-120b', 'NVIDIA GB200 (Baseten)'): 'Eight GB200, TensorRT-LLM, EAGLE-3; the number Cerebras chose to compare against.',
    ('GPT-5.3-Codex-Spark', 'Cerebras (OpenAI)'): 'Size, layers and precision undisclosed; first OpenAI model served off NVIDIA.',
    ('Kimi K2.6', 'Cerebras'): '1T MoE, 32B active; weights stored 4-bit, computed 16-bit; private endpoint.',
    ('gpt-oss-120b', 'Cerebras CS-4'): 'New three-wafer rack, doubled clock; a demo, not a public endpoint.',
    ('gpt-oss-120b', 'Cerebras public API'): 'Public endpoint under load, long prompt, 1.70 s to first token.',
}
rows = []
for r in SRC['speeds']:
    key = (r['model'], r['who'], r['date'][:7])
    a = assume.get(key) or assume.get(key[:2]) or ''
    rows.append(f"                        <tr><td>{r['date'][:7]}</td><td>{r['model']}</td><td>{r['tps']:,}</td><td>{r['who']}</td><td>{kind_label(r)}</td><td>{a}</td></tr>")
V['SPEED_ROWS'] = '\n'.join(rows)

# ---------------------------------------------------------------- wafer arithmetic
V['SRAM_FROM_CORES'] = f'{wm.sram_from_cores_gb:.1f}'
V['BYTES_PER_CORE_CYCLE'] = f'{wm.bytes_per_core_cycle:.0f}'
V['DENSE_FROM_CORES'] = f'{wm.dense_from_cores_pf:.1f}'
V['SPARSE_TO_DENSE'] = f'{wm.sparse_to_dense:.0f}'
V['BW_RATIO_H100'] = f'{wm.bw_ratio_h100:,.0f}'
V['WAFER_SIDE'] = f'{wm.wafer_side_cores:,.0f}'
V['WAFER_CROSS_US'] = f'{wm.wafer_cross_us:.2f}'

# ---------------------------------------------------------------- the line
V['BYTES_70B_GB'] = f"{wm.bytes_total('llama31_70b') / 1e9:.0f}"
V['BYTES_405B_GB'] = f"{wm.bytes_total('llama31_405b') / 1e9:.0f}"
V['BW_70B_1000_TBS'] = f"{wm.bytes_total('llama31_70b') * 1000 / 1e12:.0f}"
V['BW_405B_1000_TBS'] = f"{wm.bytes_total('llama31_405b') * 1000 / 1e12:.0f}"
ceil_h100 = wm.ceiling(8 * wm.h100_bw); ceil_wse = wm.ceiling(4 * wm.SRAM_BW); ceil_groq = wm.ceiling(576 * wm.v('gpu', 'groq_bw_tbs') * 1e12)
V['CEIL_H100X8'] = f'{ceil_h100:,.0f}'; V['CEIL_WSE3X4'] = f'{ceil_wse:,.0f}'
V['PCT_H100'] = f"{100 * SRC['gpu_baselines_2024']['h100_70b_tps']['v'] / ceil_h100:.0f}"
V['PCT_CER_2100'] = f"{100 * S['S_70B_OCT']['tps'] / ceil_wse:.2f}"
V['PCT_GROQ'] = f"{100 * 300 / ceil_groq:.2f}"

# ---------------------------------------------------------------- wafers per model
V['HEADROOM_PCT'] = f'{wm.HEADROOM * 100:.0f}'
V['FLOOR_405B_16'] = str(wm.wafers_min('llama31_405b', 16)); V['WAFERS_405B_16'] = str(wm.wafers_needed('llama31_405b', 16))
V['TWELVE_WAFERS_GB'] = f'{12 * wm.SRAM_B / 1e9:.0f}'
V['BYTES_KIMI_16_TB'] = f"{wm.bytes_total('kimi_k2') / 1e12:.0f}"
V['WAFERS_KIMI_16'] = str(wm.wafers_needed('kimi_k2', 16)); V['WAFERS_KIMI_4'] = str(wm.wafers_needed('kimi_k2', 4))
V['KIMI_ACTIVE_B'] = f"{wm.M['kimi_k2']['active_b']:.0f}"
act_bytes = wm.M['llama31_70b']['hidden'] * 2
V['ACT_70B_KB'] = f'{act_bytes / 1024:.0f}'
V['ACT_70B_MBS'] = f"{act_bytes * S['S_70B_OCT']['tps'] / 1e6:.0f}"

# ---------------------------------------------------------------- microseconds
L70 = wm.M['llama31_70b']['layers']
V['LAYERS_70B'] = str(L70)
V['STEP_70B_AUG_US'] = f"{1e6 / S['S_70B_AUG']['tps']:,.0f}"
V['US_70B_NOSPEC'] = f"{wm.WAFER_SYNC['Llama 3.1 70B']:.0f}"; V['US_8B_NOSPEC'] = f"{wm.WAFER_SYNC['Llama 3.1 8B']:.0f}"
V['LAYER_70B_GB'] = f"{wm.bytes_active('llama31_70b') / L70 / 1e9:.1f}"
V['LAYER_70B_GFLOP'] = f"{2 * wm.M['llama31_70b']['active_b'] / L70:.1f}"
b, c, s_ = wm.layer_budget('wse3')
V['BUD_WSE_BYTES'] = f'{b:.1f}'; V['BUD_WSE_FLOPS'] = f'{c:.1f}'; V['BUD_WSE_SYNC'] = f'{s_:.0f}'
share = wm.wafer_layer_share('llama31_70b')
V['LAYERS_PER_WAFER_70B'] = f'{1 / share:.0f}'; V['WAFER_SHARE_BW_PBS'] = f'{wm.SRAM_BW * share / 1e15:.2g}'; V['WAFER_SHARE_PF'] = f'{wm.dense_from_cores_pf * share:.2g}'
V['USERS_IN_WINDOW'] = f'{s_ / c:.0f}'
V['GPU_T_SYNC'] = f'{wm.GPU_T_SYNC:.0f}'
spec_cluster = [wm.implied_us_per_layer(r['tps'], r['layers']) for r in SRC['speeds'] if r['who'] == 'Cerebras' and r['spec'] and r['layers'] and r['model'] != 'Kimi K2.6']
V['US_MIN_SPEC'] = f'{min(spec_cluster):.1f}'; V['US_MAX_SPEC'] = f'{max(spec_cluster):.1f}'
V['US_KIMI'] = f"{wm.implied_us_per_layer(S['S_KIMI']['tps'], S['S_KIMI']['layers']):.1f}"
V['US_PUBLIC'] = f"{wm.implied_us_per_layer(S['S_OSS_AA']['tps'], S['S_OSS_AA']['layers']):.1f}"
V['US_OSS'] = f"{wm.implied_us_per_layer(S['S_OSS']['tps'], S['S_OSS']['layers']):.1f}"
V['US_B200_MAV'] = f"{wm.implied_us_per_layer(S['S_MAV_NV']['tps'], S['S_MAV_NV']['layers']):.1f}"
cs5 = wm.v('wafer', 'cs5_target_tps_open')
V['US_CS5_TARGET'] = f"{wm.implied_us_per_layer(cs5, wm.M['gptoss_120b']['layers']):.1f}"
V['US_CS5_TARGET_PASS'] = f"{3 * wm.implied_us_per_layer(cs5, wm.M['gptoss_120b']['layers']):.0f}"

# ---------------------------------------------------------------- costs
kv70 = wm.kv_per_token('llama31_70b'); kvoss = wm.kv_per_token('gptoss_120b'); kvkimi = wm.kv_per_token('kimi_k2')
V['KV_70B_KB'] = f'{kv70 / 1024:.0f}'; V['KV_70B_8K_GB'] = f'{kv70 * 8192 / 1e9:.1f}'; V['KV_70B_128K_GB'] = f'{kv70 * 131072 / 1e9:.0f}'
V['KV_OSS_KB'] = f'{kvoss / 1024:.0f}'; V['KV_OSS_128K_GB'] = f'{kvoss * 131072 / 1e9:.1f}'
V['KV_KIMI_KB'] = f'{kvkimi / 1024:.0f}'; V['KV_KIMI_128K_GB'] = f'{kvkimi * 131072 / 1e9:.1f}'
V['FREE_SRAM_70B_GB'] = f"{(4 * wm.SRAM_B - wm.bytes_total('llama31_70b')) / 1e9:.0f}"
pf = 2 * wm.M['llama31_405b']['params_b'] * 1e9 * 100e3
V['PREFILL_405B_100K_PFLOP'] = f'{pf / 1e15:.0f}'
V['PREFILL_405B_100K_S'] = f"{pf / (wm.wafers_min('llama31_405b') * wm.dense_from_cores_pf * 1e15):.2f}"
bom = wm.v('wafer', 'bom_per_wafer_rack_usd'); lst = wm.v('wafer', 'list_price_per_system_usd')
V['BOM_K'] = f'{bom / 1e3:.0f}'
V['HOURLY_BOM'] = f'{wm.hourly_cost(4, bom):.0f}'
tps = S['S_70B_OCT']['tps']
V['BE_060_BOM'] = f'{wm.users_to_break_even(0.60, tps, 4, bom):.0f}'; V['BE_060_LIST'] = f'{wm.users_to_break_even(0.60, tps, 4, lst):.0f}'
V['HW_SHARE_2025_PCT'] = f"{100 * wm.v('business', 'hw_rev_2025_musd') / wm.v('business', 'rev_2025_musd'):.0f}"
V['CLOUD_SHARE_Q2_PCT'] = f"{100 * wm.v('business', 'q2_2026_cloud_musd') / wm.v('business', 'q2_2026_rev_musd'):.0f}"
V['OPENAI_WAFERS_K'] = f"{wm.v('business', 'openai_mw') * 1e3 / wm.v('wafer', 'wafer_power_kw_semi') / 1e3:.0f}"
V['B_IPO_DATE_LONG'] = dt.date.fromisoformat(wm.v('business', 'ipo_date')).strftime('%d %B %Y').lstrip('0')
V['B_PRICE_70B_OUT_USD_PER_M'] = f"{wm.v('business', 'price_70b_out_usd_per_m'):.2f}"

# ---------------------------------------------------------------- carry-overs from the laptop (blog-2 zoo on the GPU)
bench = [json.loads(l) for l in open(os.path.join(LAB, 'results', 'bench.jsonl'))]
meta = json.load(open(os.path.join(LAB, 'results', 'zoo_meta.json')))
effs = []
for mf in ['../Qwen3-0.6B-Q8_0.gguf', 'gemma-3-1b-it-Q8_0.gguf', 'granite-3.1-1b-a400m-instruct-Q8_0.gguf', 'Falcon-H1-0.5B-Instruct-Q8_0.gguf', 'LFM2-700M-Q8_0.gguf', 'mamba-130m-Q8_0.gguf']:
    rs = [r for r in bench if r['tag'] == 'zoo-cuda' and r['n_gen'] and os.path.basename(r['model_filename']) == os.path.basename(mf)]
    effs.append(meta[mf]['active_bytes'] / 1e9 * rs[-1]['avg_ts'])
V['GPU_EFF_MIN'] = f'{min(effs):.0f}'; V['GPU_EFF_MAX'] = f'{max(effs):.0f}'

# ---------------------------------------------------------------- assemble
tpl = open(os.path.join(LAB, 'blog', 'blog-3.src.html'), encoding='utf-8').read()
body = '\n'.join(open(f, encoding='utf-8').read() for f in sorted(glob.glob(os.path.join(LAB, 'blog', 'b3-*.html'))))
page = tpl.replace('{{BODY}}', body)


def fill(s):
    return re.sub(r'\{\{([A-Z0-9_]+)\}\}', lambda m: V.get(m.group(1), m.group(0)), s)


page = fill(page)
words = len(re.sub(r'<[^>]+>', ' ', page).split())
V['READ_TIME'] = str(max(1, round(words / 230)))
page = fill(page)
out = os.path.join(SITE, 'blogs', 'blog-3.html')
open(out, 'w', encoding='utf-8').write(page)
res = os.path.join(SITE, 'blogs', 'resources', 'blog-3'); os.makedirs(res, exist_ok=True)
for f in glob.glob(os.path.join(LAB, 'results', 'charts3', '*.svg')):
    shutil.copy(f, res)
print(f'wrote {out}: {words} words, ~{V["READ_TIME"]} min')
left = sorted(set(re.findall(r'\{\{([A-Z0-9_]+)\}\}', page)))
if left: print('unfilled:', left)
em = page.count('\u2014') + page.count('&mdash;') - tpl.count('&mdash;')
print('em dashes in body:', em)
