"""Parse every results/kld/*.txt into one table: file size, bytes/token, PPL, KLD, top-1 agreement."""
import os, re, json, glob, sys
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def tensor_bytes(path):
    r = GGUFReader(path)
    tot = head = 0
    types = {}
    for t in r.tensors:
        n = 1
        for s in t.shape: n *= int(s)
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        b = n * ts // bs
        tot += b
        if t.name == 'token_embd.weight':
            head = b; types['head'] = t.tensor_type.name
        elif t.name == 'blk.0.ffn_down.weight':
            types['ffn'] = t.tensor_type.name
        elif t.name == 'blk.0.attn_q.weight':
            types['attn'] = t.tensor_type.name
    return tot, head, types

def parse(txt):
    s = open(txt, encoding='utf-8', errors='replace').read()
    def g(pat, cast=float):
        m = re.search(pat, s)
        return cast(m.group(1)) if m else None
    def pm(pat):
        m = re.search(pat, s)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)
    out = {}
    out['ppl'], out['ppl_err'] = pm(r'Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)')
    out['ppl_base'], out['ppl_base_err'] = pm(r'Mean PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)')
    out['ppl_ratio'], out['ppl_ratio_err'] = pm(r'Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)')
    out['kld'], out['kld_err'] = pm(r'Mean\s+KLD:\s*(-?[\d.]+)\s*±\s*([\d.]+)')
    out['kld_max'] = g(r'Maximum KLD:\s*([\d.]+)')
    out['kld_p99'] = g(r'99\.0%\s+KLD:\s*([\d.]+)')
    out['kld_median'] = g(r'Median\s+KLD:\s*([\d.]+)')
    out['dp_mean'], out['dp_mean_err'] = pm(r'Mean\s+Δp:\s*(-?[\d.]+)\s*±\s*([\d.]+)')
    out['dp_rms'], out['dp_rms_err'] = pm(r'RMS Δp\s*:\s*([\d.]+)\s*±\s*([\d.]+)')
    out['same_top'], out['same_top_err'] = pm(r'Same top p:\s*([\d.]+)\s*±\s*([\d.]+)')
    return out

rows = []
files = {'official-Q8_0': os.path.join(LAB, 'models', 'Qwen3-0.6B-Q8_0.gguf'),
         'bf16-self': os.path.join(LAB, 'models', 'Qwen3-0.6B-BF16.gguf')}
for f in glob.glob(os.path.join(LAB, 'models', 'variants', '*.gguf')):
    files[os.path.basename(f)[:-5]] = f
for txt in sorted(glob.glob(os.path.join(LAB, 'results', 'kld', '*.txt'))):
    name = os.path.basename(txt)[:-4]
    if name not in files: continue
    d = parse(txt)
    if d['kld'] is None: continue
    tot, head, types = tensor_bytes(files[name])
    d.update(name=name, file_bytes=os.path.getsize(files[name]), tensor_bytes=tot, head_bytes=head,
             head_frac=head / tot, head_type=types.get('head'), attn_type=types.get('attn'), ffn_type=types.get('ffn'))
    rows.append(d)

rows.sort(key=lambda r: r['kld'])
with open(os.path.join(LAB, 'results', 'kld_table.json'), 'w') as f:
    json.dump(rows, f, indent=1)
hdr = f"{'variant':26s} {'MB':>7s} {'head':>7s} {'head%':>6s} {'PPL':>8s} {'ratio':>7s} {'KLD':>9s} {'±':>8s} {'p99':>7s} {'RMSdp%':>7s} {'top1%':>7s}"
print(hdr)
for r in rows:
    print(f"{r['name']:26s} {r['tensor_bytes']/1e6:7.1f} {r['head_type'] or '':>7s} {100*r['head_frac']:6.1f} {r['ppl']:8.3f} {r['ppl_ratio']:7.4f} {r['kld']:9.5f} {r['kld_err']:8.5f} {r['kld_p99']:7.4f} {r['dp_rms']:7.3f} {r['same_top']:7.2f}")
