"""Shared paths and helpers for the language-adaptation post (post 4)."""
import os, json, time, subprocess

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(LAB, 'data', 'adapt')
RES = os.path.join(LAB, 'results', 'adapt')
MODELS = os.path.join(LAB, 'models', 'adapt')
for d in (DATA, RES, MODELS):
    os.makedirs(d, exist_ok=True)

BASE_MODEL = 'HuggingFaceTB/SmolLM2-135M'


def log_row(name, row):
    """Append one JSON row to results/adapt/<name>.jsonl with a timestamp and the GPU power state."""
    row = dict(row)
    row.setdefault('ts', time.strftime('%Y-%m-%dT%H:%M:%S'))
    row.setdefault('gpu', gpu_state())
    with open(os.path.join(RES, name + '.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def gpu_state():
    """Power limit and draw from nvidia-smi, so every row records whether the laptop was on battery."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=power.limit,power.draw,memory.used,clocks.sm',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        pl, pd, mu, ck = [x.strip() for x in out.strip().split(',')]
        return {'power_limit_w': pl, 'power_draw_w': pd, 'mem_used_mb': mu, 'sm_mhz': ck}
    except Exception:
        return {}


def read_jsonl(name):
    p = os.path.join(RES, name + '.jsonl')
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def devanagari_share(s):
    """Fraction of non-space characters in the Devanagari block (U+0900 to U+097F)."""
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if 'ऀ' <= c <= 'ॿ') / len(chars)
