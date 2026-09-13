#!/bin/bash
# usage: bench.sh <tag> <build:cpu|cuda|vulkan> [llama-bench args...]
# Runs llama-bench ONCE with JSONL output, appends tagged rows to results/bench.jsonl, prints a compact table.
LAB="/c/PK/Github-Projects/ggml-inference-lab"; LABW="C:/PK/Github-Projects/ggml-inference-lab"; tag="$1"; build="$2"; shift 2
out=$("$LAB/bin/$build/llama-bench.exe" -o jsonl "$@" 2>"$LAB/results/bench-$tag.err")
echo "$out" | python -c "
import sys,json
rows=[]
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('{'): continue
    d=json.loads(line); d['tag']='$tag'; d['build']='$build'; d['cmd']='$*'
    rows.append(d)
with open('C:/PK/Github-Projects/ggml-inference-lab/results/bench.jsonl','a') as f:
    for d in rows: f.write(json.dumps(d)+'\n')
for d in rows:
    test = ('pp%d' % d['n_prompt']) if d['n_prompt'] else ('tg%d' % d['n_gen'])
    if d.get('n_depth'): test += ' @d%d' % d['n_depth']
    extra = ' t=%d' % d['n_threads']
    extra += ' ngl=%d' % d['n_gpu_layers'] if d['n_gpu_layers'] not in (0,) else ''
    extra += ' mask=%s' % d['cpu_mask'] if d['cpu_mask'] not in ('0x0', '') else ''
    extra += ' fa=%s' % d['flash_attn'] if d['flash_attn'] not in (-1, 'auto') else ''
    extra += ' ctk=%s' % d['type_k'] if d['type_k'] != 'f16' else ''
    extra += ' ub=%d' % d['n_ubatch'] if d['n_ubatch'] != 512 else ''
    extra += ' ot=%s' % d['tensor_buft_overrides'] if d.get('tensor_buft_overrides') not in (None, '', 'none') else ''
    print('| %-28s | %-46s | %9s | %8.2f +- %6.2f |' % (d['model_type'], test + extra, ('%.0f MB' % (d['model_size']/1e6)), d['avg_ts'], d['stddev_ts']))
"
