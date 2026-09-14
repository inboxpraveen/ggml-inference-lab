"""How many chunks does a KL-divergence run need? llama-perplexity prints the running mean after every chunk,
so a 200-chunk log already contains what a 10, 20 or 50 chunk run would have reported. Extract those points."""
import os, re, json, glob
LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out = {}
for f in sorted(glob.glob(os.path.join(LAB, 'results', 'kld', '*.txt'))):
    name = os.path.basename(f)[:-4]
    pts = {}
    for line in open(f, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*(\d+)\s+([\d.]+) ±\s+([\d.]+)\s+([-\d.]+) ±\s+([\d.]+)\s+([\d.]+) ±\s+([\d.]+)\s+([\d.]+) ±\s+([\d.]+) %\s+([\d.]+) ±\s+([\d.]+) %', line)
        if m:
            c = int(m.group(1))
            pts[c] = dict(ppl=float(m.group(2)), kld=float(m.group(6)), kld_err=float(m.group(7)), same_top=float(m.group(10)), same_top_err=float(m.group(11)))
    if pts:
        out[name] = {c: pts[c] for c in [5, 10, 20, 40, 100, 200] if c in pts}
json.dump(out, open(os.path.join(LAB, 'results', 'kld_convergence.json'), 'w'), indent=1)
for name in ['plain-Q8_0', 'plain-Q6_K', 'plain-Q4_K_M', 'plain-Q3_K_M', 'head-Q6_K-body-Q8_0']:
    print(name, {c: f"{v['kld']:.4f}±{v['kld_err']:.4f}" for c, v in out[name].items()})
