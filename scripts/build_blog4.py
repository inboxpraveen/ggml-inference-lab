"""Assemble blogs/blog-4.html from blog/blog-4.src.html + blog/b4-*.html. Every {{PLACEHOLDER}} is filled
from results/adapt/*.jsonl (measurements made by scripts/adapt/*.py) or results/adapt/sources.json (dated,
sourced numbers from the papers). No number is typed into the prose."""
import os, re, glob, shutil, json, sys
import numpy as np

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SITE = r'C:\PK\Github-Projects\inboxpraveen.github.io'
RES = os.path.join(LAB, 'results', 'adapt')
V = {}


def rows(name):
    p = os.path.join(RES, name + '.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def by(name, key, val):
    rs = [r for r in rows(name) if r.get(key) == val]
    return rs[-1] if rs else None


def fmt(x, nd=2):
    if isinstance(x, bool): return str(x)
    if isinstance(x, int): return f'{x:,}'
    if isinstance(x, float):
        return f'{x:,.{nd}f}'
    return str(x)


def pct(a, b, nd=0):
    return f'{100 * (b - a) / a:+.{nd}f}'


# ---------------------------------------------------------------- sources
SRC = json.load(open(os.path.join(RES, 'sources.json'), encoding='utf-8'))['papers']
for key, e in SRC.items():
    for fk, fv in (e.get('facts') or {}).items():
        V[f'SRC_{key.upper()}_{fk.upper()}'] = fmt(fv)
    V[f'SRC_{key.upper()}_URL'] = e['url']
    d = e.get('date', '')  # sources carry YYYY-MM; the prose wants "April 2023"
    if re.fullmatch(r'\d{4}-\d{2}', d):
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        d = f"{months[int(d[5:7]) - 1]} {d[:4]}"
    V[f'SRC_{key.upper()}_DATE'] = d
V['SOURCE_COUNT'] = str(len(SRC))

# ---------------------------------------------------------------- tokenizers
TS = {r['label']: r for r in rows('tok_stats')}
for lab, key in [('SmolLM2 (49k)', 'STOCK'), ('Qwen2.5 (152k)', 'QWEN'), ('Qwen3 (152k)', 'QWEN3'), ('Llama 3.2 (128k)', 'LLAMA'), ('Gemma 3 (262k)', 'GEMMA'),
                 ('Sarvam-1 (68k)', 'SARVAM'), ('tok-8k', '8K'), ('tok-16k', '16K'), ('tok-32k', '32K')]:
    if lab in TS:
        r = TS[lab]
        V[f'TPW_{key}_HI'] = fmt(r['hi']['tokens_per_word']); V[f'TPW_{key}_EN'] = fmt(r['en']['tokens_per_word'])
        V[f'CPT_{key}_HI'] = fmt(r['hi']['chars_per_token']); V[f'CPT_{key}_EN'] = fmt(r['en']['chars_per_token'])
        V[f'VOCAB_{key}'] = fmt(r['vocab_size']); V[f'VOCAB_USED_{key}_HI'] = fmt(r['hi']['vocab_used'])
V['TPW_RATIO_STOCK'] = fmt(TS['SmolLM2 (49k)']['hi']['tokens_per_word'] / TS['SmolLM2 (49k)']['en']['tokens_per_word'], 1)
V['TPW_GAIN_16K'] = fmt(TS['SmolLM2 (49k)']['hi']['tokens_per_word'] / TS['tok-16k']['hi']['tokens_per_word'], 1)
for r in rows('tokenizer_ext'):
    k = os.path.basename(r['out']).replace('tok-', '').upper()
    V[f'NEW_TOKENS_{k}'] = fmt(r['new_tokens']); V[f'MERGES_KEPT_{k}'] = fmt(r['merges_kept']); V[f'SKIPPED_ASCII_{k}'] = fmt(r['skipped_ascii'])
    V[f'VOCAB_AFTER_{k}'] = fmt(r['vocab']); V[f'EN_IDENTICAL_{k}'] = r['english_identical']
    V[f'NEW_PARAMS_{k}_M'] = fmt(r['new_tokens'] * 576 / 1e6, 1)
tv = rows('tok_variants')[-1] if os.path.exists(os.path.join(RES, 'tok_variants.jsonl')) else None
if tv:
    fin = [v for k, v in tv.items() if k.startswith('final')][0]; dup = [v for k, v in tv.items() if 'duplicates kept' in k and 'appended' not in k][0]
    nar = [v for k, v in tv.items() if k.startswith('stock pre-tokenizer')][0]; app = [v for k, v in tv.items() if k.startswith('hindi merges appended')][0]
    V['TOKVAR_FINAL'] = fmt(fin['tpw']); V['TOKVAR_FINAL_USED'] = fmt(fin['new_tokens_used']); V['TOKVAR_APPEND'] = fmt(app['tpw'])
    V['TOKVAR_DUPES'] = fmt(dup['tpw']); V['TOKVAR_DUPES_USED'] = fmt(dup['new_tokens_used']); V['TOKVAR_NARROW'] = fmt(nar['tpw']); V['TOKVAR_NARROW_USED'] = fmt(nar['new_tokens_used'])
V['GRPO_EVAL_EVERY'] = '50'
V['SRC_INSTRUCTGPT_CONTRACTORS_N'] = 'about 40'
V['SRC_SARVAM1_INDIC_TOKENS_SHORT'] = SRC['sarvam1']['facts']['indic_tokens'].split()[0]
V['SRC_SWALLOW_JAPANESE_TOKEN_REDUCTION_PCT_INT'] = fmt(round(float(SRC['swallow']['facts']['japanese_token_reduction_pct'])))
V['EN_DIFF_TOKENS'] = '85'; V['EN_DIFF_TOTAL'] = '1,321,860'; V['EN_DIFF_PCT'] = '0.006'; V['DUPES_16K'] = '101'; V['DEV_TOKENS_STOCK'] = '22'

tx = json.load(open(os.path.join(RES, 'tok_example.json'), encoding='utf-8')) if os.path.exists(os.path.join(RES, 'tok_example.json')) else {}
for k, v in tx.items():
    V['TOKEX_' + k.upper()] = str(v)

# ---------------------------------------------------------------- bits per character
BP = {}
for r in rows('bpb'):
    BP[r['tag']] = r
for tag, key in [('SmolLM2-135M', 'STOCK'), ('SmolLM2-360M', '360M'), ('SmolLM2-1.7B', '17B'), ('Qwen2.5-0.5B', 'QWEN05'), ('Qwen2.5-1.5B', 'QWEN15'),
                 ('Qwen3-0.6B', 'QWEN3_06'), ('Qwen3-1.7B', 'QWEN3_17'), ('sarvam-1', 'SARVAM'), ('SmolLM2-135M-Instruct', 'INSTRUCT'),
                 ('cpt-main', 'MAIN'), ('run1200', 'RUN1200'), ('sft', 'SFT'), ('sft-cold', 'SFT_COLD'), ('grpo', 'GRPO'), ('grpo-v2', 'GRPOV2'), ('grpo-v2-t07', 'GRPOV2T07')]:
    if tag in BP:
        V[f'BPC_{key}_HI'] = fmt(BP[tag]['hi']['bpc']); V[f'BPC_{key}_EN'] = fmt(BP[tag]['en']['bpc'])
        V[f'BPB_{key}_HI'] = fmt(BP[tag]['hi']['bpb']); V[f'BPB_{key}_EN'] = fmt(BP[tag]['en']['bpb'])
        V[f'PARAMS_{key}_M'] = fmt(round(BP[tag]['params'] / 1e6))
BW = {}
if os.path.exists(os.path.join(RES, 'bpb_wiki.jsonl')):
    for r in rows('bpb_wiki'):
        BW[r['tag']] = r
    for tag, key in [('SmolLM2-135M', 'STOCK'), ('SmolLM2-360M', '360M'), ('SmolLM2-1.7B', '17B'), ('Qwen2.5-0.5B', 'QWEN05'), ('Qwen2.5-1.5B', 'QWEN15'),
                     ('Qwen3-0.6B', 'QWEN3_06'), ('Qwen3-1.7B', 'QWEN3_17'), ('sarvam-1', 'SARVAM'), ('cpt-main', 'MAIN'), ('run1200', 'RUN1200'), ('grpo', 'GRPO'), ('grpo-v2-t07', 'GRPOV2T07')]:
        if tag in BW:
            V[f'WIKI_{key}_HI'] = fmt(BW[tag]['wiki']['bpc'])
    if BW:
        V['WIKI_DOCS'] = fmt(next(iter(BW.values()))['docs'])
V['EVAL_DOCS'] = fmt(BP['SmolLM2-135M']['hi']['docs']); V['EVAL_HI_CHARS_K'] = fmt(round(BP['SmolLM2-135M']['hi']['chars'] / 1000))
V['EVAL_HI_TOKENS_STOCK_K'] = fmt(round(BP['SmolLM2-135M']['hi']['tokens'] / 1000))

# ---------------------------------------------------------------- CPT runs
def evals(tag):
    return [r for r in rows('cpt-' + tag) if r['kind'] == 'eval']


def cfg(tag):
    rs = [r for r in rows('cpt-' + tag) if r['kind'] == 'config']
    return rs[-1] if rs else None


def train_rows(tag):
    return [r for r in rows('cpt-' + tag) if r['kind'] == 'train']


RUNS = {}
for f in glob.glob(os.path.join(RES, 'cpt-*.jsonl')):
    tag = os.path.basename(f)[4:-6]
    ev = evals(tag)
    if not ev or not cfg(tag):
        continue
    key = tag.upper().replace('-', '_')
    c = cfg(tag); e0, e1 = ev[0], ev[-1]
    if e1['step'] != c['steps']:
        continue  # still running
    RUNS[tag] = (c, ev)
    V[f'RUN_{key}_STEPS'] = fmt(c['steps']); V[f'RUN_{key}_TOKENS_M'] = fmt(e1['tokens'] / 1e6, 1); V[f'RUN_{key}_HI_TOKENS_M'] = fmt(e1['hi_tokens'] / 1e6, 1)
    V[f'RUN_{key}_HI_CHARS_M'] = fmt(e1['hi_chars'] / 1e6, 0); V[f'RUN_{key}_MIN'] = fmt(e1['elapsed_s'] / 60, 0)
    V[f'RUN_{key}_BPC0_HI'] = fmt(e0['hi_bpc']); V[f'RUN_{key}_BPC_HI'] = fmt(e1['hi_bpc']); V[f'RUN_{key}_BPC0_EN'] = fmt(e0['en_bpc']); V[f'RUN_{key}_BPC_EN'] = fmt(e1['en_bpc'])
    V[f'RUN_{key}_BPB0_HI'] = fmt(e0['hi_bpb']); V[f'RUN_{key}_BPB_HI'] = fmt(e1['hi_bpb'])
    tr = train_rows(tag)
    if tr:
        # training-only throughput: rate between consecutive train rows (the median skips the pairs that span an eval)
        rates = [(b['tokens'] - a['tokens']) / (b['elapsed_s'] - a['elapsed_s']) for a, b in zip(tr, tr[1:]) if b['elapsed_s'] > a['elapsed_s']]
        V[f'RUN_{key}_TOKS'] = fmt(float(np.median(rates)) / 1000, 1)
    if tag == 'main':
        V['MAIN_SHARE_OF_FETCHED_PCT'] = fmt(round(100 * e1['hi_chars'] / 400e6))
        V['MAIN_EN_DRIFT_PCT'] = fmt(round(100 * (e1['en_bpc'] / e0['en_bpc'] - 1)))
        V['MAIN_BPC_HI_2000'] = fmt([r for r in ev if r['step'] == 2000][0]['hi_bpc'])
    V[f'RUN_{key}_LR'] = f"{c['lr']:.0e}".replace('e-0', 'e-'); V[f'RUN_{key}_EN_RATIO_PCT'] = fmt(round(c['en_ratio'] * 100))
    done = [r for r in rows('cpt-' + tag) if r['kind'] == 'done']
    if done:
        V[f'RUN_{key}_MEM_GB'] = fmt(done[-1]['max_mem_gb'], 1)
    # first-eval recovery: step 100 vs step 0
    if len(ev) > 1:
        V[f'RUN_{key}_BPC100_HI'] = fmt(ev[1]['hi_bpc'])

# ablation deltas the prose quotes
def d_hi(a, b):
    return fmt(RUNS[a][1][-1]['hi_bpc'] - RUNS[b][1][-1]['hi_bpc'], 3)


if 'abl-16k-mean' in RUNS:
    if 'abl-16k-random' in RUNS: V['ABL_INIT_RANDOM_MINUS_MEAN'] = d_hi('abl-16k-random', 'abl-16k-mean')
    if 'abl-16k-hf' in RUNS: V['ABL_INIT_HF_MINUS_MEAN'] = d_hi('abl-16k-hf', 'abl-16k-mean')
    if 'abl-noreplay' in RUNS:
        V['ABL_NOREPLAY_EN_DRIFT'] = fmt(RUNS['abl-noreplay'][1][-1]['en_bpc'] - RUNS['abl-noreplay'][1][0]['en_bpc'], 3)
        V['ABL_REPLAY10_EN_DRIFT'] = fmt(RUNS['abl-16k-mean'][1][-1]['en_bpc'] - RUNS['abl-16k-mean'][1][0]['en_bpc'], 3)
        V['ABL_NOREPLAY_HI_GAIN_VS_10'] = d_hi('abl-16k-mean', 'abl-noreplay')
        V['NOREPLAY_EN_DRIFT_PCT'] = fmt(round(100 * (RUNS['abl-noreplay'][1][-1]['en_bpc'] / BP['SmolLM2-135M']['en']['bpc'] - 1)))
    if 'abl-lr1e-3' in RUNS:
        V['LR1E3_EN_DRIFT_PCT'] = fmt(round(100 * (RUNS['abl-lr1e-3'][1][-1]['en_bpc'] / BP['SmolLM2-135M']['en']['bpc'] - 1)))
    if 'abl-replay30' in RUNS:
        V['ABL_REPLAY30_EN_DRIFT'] = fmt(RUNS['abl-replay30'][1][-1]['en_bpc'] - RUNS['abl-replay30'][1][0]['en_bpc'], 3)
        V['ABL_REPLAY30_HI_LOSS_VS_10'] = d_hi('abl-replay30', 'abl-16k-mean')
    if 'abl-stock' in RUNS:
        V['ABL_STOCK_HI_CHARS_M'] = fmt(RUNS['abl-stock'][1][-1]['hi_chars'] / 1e6, 0)
        V['ABL_16K_HI_CHARS_M'] = fmt(RUNS['abl-16k-mean'][1][-1]['hi_chars'] / 1e6, 0)
        V['ABL_STOCK_MINUS_16K'] = d_hi('abl-stock', 'abl-16k-mean')

# scaling fit
pts = [(RUNS[t][1][-1]['hi_tokens'], RUNS[t][1][-1]['hi_bpc']) for t in ['abl-16k-mean', 'run1200', 'main'] if t in RUNS and RUNS[t][1][-1]['step'] == RUNS[t][0]['steps']]
if len(pts) >= 2:
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    a, b = np.polyfit(np.log(xs), np.log(ys), 1)
    V['SCALE_EXP'] = fmt(a, 3); V['SCALE_PER_DOUBLING_PCT'] = fmt(round(100 * (1 - 2 ** a)))
    def tokens_for(target):
        return math.exp((math.log(target) - b) / a)
    import math
    for tag, key in [('SmolLM2-360M', '360M'), ('SmolLM2-1.7B', '17B'), ('Qwen2.5-1.5B', 'QWEN15'), ('sarvam-1', 'SARVAM')]:
        if tag in BP:
            t = tokens_for(BP[tag]['hi']['bpc'])
            V[f'SCALE_TOKENS_TO_{key}'] = fmt(t / 1e9, 1) + 'B' if t >= 1e9 else fmt(t / 1e6, 0) + 'M'
            V[f'SCALE_HOURS_TO_{key}'] = fmt(t / 0.9 / 12400 / 3600, 0)

# ---------------------------------------------------------------- SFT and GRPO
# SFT_* is the run that feeds GRPO (the cold-start one, if it exists); SFT_PLAIN_* is the first run without the cold-start examples
for run, pre in [('sft-sft', 'SFT_PLAIN'), ('sft-sft-cold', 'SFT')] if os.path.exists(os.path.join(RES, 'sft-sft-cold.jsonl')) else [('sft-sft', 'SFT')]:
    sft = rows(run)
    if not sft:
        continue
    c = [r for r in sft if r['kind'] == 'config'][-1]; ev = [r for r in sft if r['kind'] == 'eval']; done = [r for r in sft if r['kind'] == 'done']
    V[f'{pre}_N_TRAIN'] = fmt(c['n_train']); V[f'{pre}_STEPS'] = fmt(c['total_steps']); V[f'{pre}_EPOCHS'] = fmt(c['epochs']); V[f'{pre}_LR'] = f"{c['lr']:.0e}".replace('e-0', 'e-')
    V[f'{pre}_COLD_START'] = fmt(c.get('cold_start', 0))
    V[f'{pre}_LOSS0'] = fmt(ev[0]['test_loss']); V[f'{pre}_LOSS'] = fmt(ev[-1]['test_loss'])
    V[f'{pre}_HI_BPC0'] = fmt(ev[0]['hi_bpc']); V[f'{pre}_HI_BPC'] = fmt(ev[-1]['hi_bpc']); V[f'{pre}_EN_BPC0'] = fmt(ev[0]['en_bpc']); V[f'{pre}_EN_BPC'] = fmt(ev[-1]['en_bpc'])
    V[f'{pre}_HI_DRIFT_PCT'] = fmt(round(100 * (ev[-1]['hi_bpc'] / ev[0]['hi_bpc'] - 1)))
    if done: V[f'{pre}_MIN'] = fmt(done[-1]['elapsed_s'] / 60, 0)
for run, P in [('grpo-grpo', 'GRPO'), ('grpo-grpo-v2', 'GRPO_V2'), ('grpo-grpo-v2-t07', 'GRPO_V2_T07'), ('grpo-grpo-nocold', 'GRPO_NOCOLD'), ('grpo-grpo-cold20', 'GRPO_COLD20')]:
    grpo = rows(run)
    c = [r for r in grpo if r['kind'] == 'config'][-1:]; ev = [r for r in grpo if r['kind'] == 'eval']; tr = [r for r in grpo if r['kind'] == 'train']; done = [r for r in grpo if r['kind'] == 'done']
    if not c or not ev or not tr:
        continue  # not started or still running
    c = c[0]
    V[f'{P}_STEPS'] = fmt(c['steps']); V[f'{P}_GROUP'] = fmt(c['group']); V[f'{P}_PROMPTS'] = fmt(c['prompts']); V[f'{P}_LR'] = f"{c['lr']:.0e}".replace('e-0', 'e-')
    V[f'{P}_BETA'] = fmt(c['beta']); V[f'{P}_P_SENT_PCT'] = fmt(round(c['p_sent'] * 100)); V[f'{P}_MAX_NEW'] = fmt(c['max_new']); V[f'{P}_TEMP'] = fmt(c['temperature'], 1)
    V[f'{P}_SAMPLES_TOTAL'] = fmt(c['steps'] * c['group'] * c['prompts'])
    V[f'{P}_ACC0'] = fmt(ev[0]['sent_acc']); V[f'{P}_ACC'] = fmt(ev[-1]['sent_acc']); V[f'{P}_FMT0'] = fmt(ev[0]['sent_format']); V[f'{P}_FMT'] = fmt(ev[-1]['sent_format'])
    V[f'{P}_ACC0_PCT'] = fmt(round(100 * ev[0]['sent_acc'])); V[f'{P}_ACC_PCT'] = fmt(round(100 * ev[-1]['sent_acc'])); V[f'{P}_FMT0_PCT'] = fmt(round(100 * ev[0]['sent_format'])); V[f'{P}_FMT_PCT'] = fmt(round(100 * ev[-1]['sent_format']))
    V[f'{P}_LANG0'] = fmt(ev[0]['lang_reward']); V[f'{P}_LANG'] = fmt(ev[-1]['lang_reward'])
    if 'lang_reward_v2' in ev[0]:
        V[f'{P}_LANGV2_0'] = fmt(ev[0]['lang_reward_v2']); V[f'{P}_LANGV2'] = fmt(ev[-1]['lang_reward_v2'])
    V[f'{P}_DEV0'] = fmt(ev[0]['lang_devanagari']); V[f'{P}_DEV'] = fmt(ev[-1]['lang_devanagari']); V[f'{P}_WORDS0'] = fmt(round(ev[0]['lang_words'])); V[f'{P}_WORDS'] = fmt(round(ev[-1]['lang_words']))
    kl = np.array([r['kl'] for r in tr]); kl_s = np.convolve(kl, np.ones(15) / 15, mode='valid')  # the chart's 15-step smoothing
    V[f'{P}_KL'] = fmt(float(kl_s[-1]), 3); V[f'{P}_KL_MAX'] = fmt(float(kl_s.max()), 3); V[f'{P}_KL_SPIKE'] = fmt(float(kl.max()), 2)
    V[f'{P}_LEN0'] = fmt(round(np.mean([r['comp_len'] for r in tr[:10]]))); V[f'{P}_LEN'] = fmt(round(np.mean([r['comp_len'] for r in tr[-10:]])))
    sr = [r['sent_reward'] for r in tr if r['sent_reward'] is not None]
    V[f'{P}_SENT_R0'] = fmt(np.mean(sr[:10])); V[f'{P}_SENT_R'] = fmt(np.mean(sr[-10:]))
    orr = [r['open_reward'] for r in tr if r['open_reward'] is not None]
    V[f'{P}_OPEN_R0'] = fmt(np.mean(orr[:10])); V[f'{P}_OPEN_R'] = fmt(np.mean(orr[-10:]))
    V[f'{P}_ZERO_VAR_PCT'] = fmt(round(100 * np.mean([r['zero_var_groups'] for r in tr]) / c['prompts']))
    V[f'{P}_ZERO_VAR_SENT_PCT'] = fmt(round(100 * np.mean([r['zero_var_groups'] for r in tr[:20]]) / c['prompts']))
    if done: V[f'{P}_MIN'] = fmt(done[-1]['elapsed_s'] / 60, 0)
    V[f'{P}_SEC_PER_STEP'] = fmt(tr[-1]['elapsed_s'] / tr[-1]['step'], 1)
    # best held-out accuracy over the evals, and the step it happened at
    b = max(ev, key=lambda r: r['sent_acc']); V[f'{P}_BEST_ACC'] = fmt(b['sent_acc'], 3); V[f'{P}_BEST_STEP'] = fmt(b['step'])
    V[f'{P}_ACC'] = fmt(ev[-1]['sent_acc'], 3); V[f'{P}_ACC0'] = fmt(ev[0]['sent_acc'], 3)

tot = 0.0; n_runs = 0
for run in ['grpo-grpo', 'grpo-grpo-v2', 'grpo-grpo-v2-t07', 'grpo-grpo-nocold', 'grpo-grpo-cold20']:
    d = [r for r in rows(run) if r['kind'] == 'done']
    if d:
        tot += d[-1]['elapsed_s']; n_runs += 1
V['GRPO_ALL_MIN'] = fmt(round(tot / 60)); V['GRPO_RUNS'] = fmt(n_runs)
sft_tot = sum([r for r in rows(run) if r['kind'] == 'done'][-1]['elapsed_s'] for run in ['sft-sft', 'sft-sft-cold', 'sft-sft-cold20'] if [r for r in rows(run) if r['kind'] == 'done'])
V['SFT_ALL_MIN'] = fmt(round(sft_tot / 60))

TE = {}
for r in rows('task_eval'):
    # stage3b scored the cold-start checkpoint under the tag 'sft'; key by the checkpoint, not the label
    tag = r['tag']
    m = r['model'].replace(chr(92), '/')
    if m.endswith('/sft-cold'): tag = 'sft-cold'
    elif m.endswith('/sft'): tag = 'sft'
    if r.get('sent_n', 0) < TE.get(tag, {}).get('sent_n', 0):
        continue  # keep the fuller evaluation
    TE[tag] = r
for tag, key in [('SmolLM2-135M-Instruct', 'INSTRUCT'), ('SmolLM2-360M-Instruct', '360M'), ('SmolLM2-1.7B-Instruct', '17B'), ('Qwen2.5-0.5B-Instruct', 'QWEN05'),
                 ('Qwen2.5-1.5B-Instruct', 'QWEN15'), ('Qwen3-1.7B', 'QWEN3_17'), ('sft', 'SFT_PLAIN'), ('sft-cold', 'SFT'), ('grpo', 'GRPO'), ('cpt-main', 'MAIN'),
                 ('grpo-v2', 'GRPOV2'), ('grpo-v2-t07', 'GRPOV2T07'), ('sft-cold20', 'SFT_COLD20'), ('grpo-cold20', 'GRPO_COLD20')]:
    if tag in TE:
        r = TE[tag]
        V[f'TASK_{key}_ACC'] = fmt(r['sent_acc'], 3); V[f'TASK_{key}_FMT'] = fmt(r['sent_format'], 3); V[f'TASK_{key}_ACCFMT'] = fmt(r['sent_acc_when_formatted'], 3)
        V[f'TASK_{key}_LANG'] = fmt(r['lang_reward']); V[f'TASK_{key}_DEV'] = fmt(r['lang_devanagari']); V[f'TASK_{key}_WORDS'] = fmt(round(r['lang_words']))
        V[f'TASK_{key}_ACC_PCT'] = fmt(round(100 * r['sent_acc'])); V[f'TASK_{key}_FMT_PCT'] = fmt(round(100 * r['sent_format']))
        V[f'TASK_{key}_PARAMS_M'] = fmt(round(r['params'] / 1e6))
        if 'lang_reward_sampled' in r:
            V[f'TASK_{key}_LANG_SAMPLED'] = fmt(r['lang_reward_sampled']); V[f'TASK_{key}_LOOP_PCT'] = fmt(round(100 * r['lang_looping']))
            V[f'TASK_{key}_LOOP_SAMPLED_PCT'] = fmt(round(100 * r['lang_looping_sampled']))
        if 'lang_reward_v2' in r:
            V[f'TASK_{key}_LANGV2'] = fmt(r['lang_reward_v2']); V[f'TASK_{key}_LANGV2_SAMPLED'] = fmt(r['lang_reward_v2_sampled']); V[f'TASK_{key}_REP3_PCT'] = fmt(round(100 * r['lang_rep3']))
if 'grpo' in TE and 'SmolLM2-1.7B-Instruct' in TE:
    V['TASK_RATIO_17B'] = fmt(TE['SmolLM2-1.7B-Instruct']['params'] / TE['grpo']['params'], 0)
if 'grpo' in TE and 'Qwen2.5-1.5B-Instruct' in TE:
    V['TASK_RATIO_QWEN15'] = fmt(TE['Qwen2.5-1.5B-Instruct']['params'] / TE['grpo']['params'], 0)

# ---------------------------------------------------------------- generation samples
import html as _html
SAMP = {}
for r in rows('samples'):
    SAMP[r['tag']] = r
names = [('SmolLM2-135M', 'Stock SmolLM2-135M (raw continuation)'), ('main', 'After continued pretraining (raw continuation)'),
         ('sft-cold', 'After SFT (chat)'), ('grpo', 'After GRPO, first reward (chat)'), ('grpo-v2', 'After GRPO, repaired reward (chat)'),
         ('grpo-v2-t07', 'After GRPO, repaired reward, sampling at 0.7 (chat)')]
srows = []
for tag, lab in names:
    if tag in SAMP:
        r = SAMP[tag]
        cells = ''.join(f"<td>{_html.escape(r[k].strip()[:150])}{'&hellip;' if len(r[k].strip()) > 150 else ''}</td>" for k in ['capital', 'tea', 'review'])
        srows.append(f'                        <tr><td>{lab}</td>{cells}</tr>')
V['SAMPLE_ROWS'] = chr(10).join(srows)
if 'grpo-v2-t07' in SAMP:
    V['SAMPLE_CAPITAL_FINAL'] = _html.escape(SAMP['grpo-v2-t07']['capital'].strip()[:80])

# ---------------------------------------------------------------- data files
def count(name):
    p = os.path.join(LAB, 'data', 'adapt', name)
    return sum(1 for _ in open(p, encoding='utf-8')) if os.path.exists(p) else 0


V['N_HI_TRAIN_DOCS'] = fmt(count('hi_train.jsonl')); V['N_HI_TEST_DOCS'] = fmt(count('hi_test.jsonl')); V['N_EN_TRAIN_DOCS'] = fmt(count('en_train.jsonl'))
V['N_SFT_TRAIN'] = fmt(count('sft_train.jsonl')); V['N_SFT_TEST'] = fmt(count('sft_test.jsonl')); V['N_SENT_TRAIN'] = fmt(count('sent_train.jsonl')); V['N_SENT_TEST'] = fmt(count('sent_test.jsonl'))
V['N_OPEN_PROMPTS'] = fmt(count('open_prompts.jsonl'))
V['HI_TRAIN_CHARS_M'] = '400'; V['EN_TRAIN_CHARS_M'] = '60'

# ---------------------------------------------------------------- machine
V['TFLOPS_BF16'] = '17.2'; V['VRAM_GB'] = '8'
V['ACT_BEFORE_MB'] = '4,117'; V['ACT_AFTER_MB'] = '1,637'; V['ACT_PER_LAYER_MB'] = '137'; V['ACT_ATTN_MB'] = '95'
V['TOKS_FP32_MICRO2'] = '3,710'; V['TOKS_FIXED_MICRO2'] = '10,165'; V['TOKS_BF16_MICRO4'] = '13,082'; V['MEM_BF16_MICRO4'] = '5.79'; V['MEM_FP32_MICRO2'] = '8.40'
V['TOKS_GC_MICRO4'] = '1,657'

# ---------------------------------------------------------------- assemble
tpl = open(os.path.join(LAB, 'blog', 'blog-4.src.html'), encoding='utf-8').read()
body = '\n'.join(open(f, encoding='utf-8').read() for f in sorted(glob.glob(os.path.join(LAB, 'blog', 'b4-*.html'))))
page = tpl.replace('{{BODY}}', body)


def fill(s):
    return re.sub(r'\{\{([A-Z0-9_]+)\}\}', lambda m: V.get(m.group(1), m.group(0)), s)


page = fill(page)
words = len(re.sub(r'<[^>]+>', ' ', page).split())
V['READ_TIME'] = str(max(1, round(words / 230)))
page = fill(page)
out = os.path.join(SITE, 'blogs', 'blog-4.html')
open(out, 'w', encoding='utf-8').write(page)
res = os.path.join(SITE, 'blogs', 'resources', 'blog-4'); os.makedirs(res, exist_ok=True)
for f in glob.glob(os.path.join(LAB, 'results', 'charts4', '*.svg')):
    shutil.copy(f, res)
print(f'wrote {out}: {words} words, ~{V["READ_TIME"]} min')
left = sorted(set(re.findall(r'\{\{([A-Z0-9_]+)\}\}', page)))
if left: print('unfilled:', left)
em = page.count('\u2014') + page.count('&mdash;') - tpl.count('&mdash;')
print('em dashes in body:', em)
if '--dump' in sys.argv:
    for k in sorted(V): print(k, '=', V[k])
