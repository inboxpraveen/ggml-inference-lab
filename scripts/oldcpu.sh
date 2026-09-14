#!/usr/bin/env bash
# Which quant is fastest when the CPU has no AVX2? Force the sse42 and sandybridge kernels (the ones a 2010 to 2012 desktop
# would load) and the alderlake kernels this chip normally gets, on five quants of the same model. Paired in one session.
cd "$(dirname "$0")/.."
M="models/variants/plain-Q8_0.gguf,models/variants/plain-Q6_K.gguf,models/variants/plain-Q5_K_M.gguf,models/variants/plain-Q4_K_M.gguf,models/variants/plain-Q4_0.gguf,models/variants/plain-IQ4_XS.gguf"
for v in sse42 sandybridge alderlake; do
  d="bin/cpu-variants/$v"
  out=$("$d/llama-bench.exe" -o jsonl -m "$M" -p 256 -n 128 -r 3 -t 8 --prio 2 -ngl 0 2>"results/bench-oldcpu-$v.err")
  echo "$out" | python -c "
import sys, json
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('{'): continue
    d=json.loads(line); d['tag']='oldcpu-$v'; d['build']='cpu-variant:$v'; d['cmd']='-p 256 -n 128 -r 3 -t 8 --prio 2 -ngl 0'
    open('C:/PK/Github-Projects/ggml-inference-lab/results/bench.jsonl','a').write(json.dumps(d)+'\n')
    print(f\"{d['tag']:20s} {d['model_filename'][-22:]:22s} {'pp' if d['n_prompt'] else 'tg':2s} {d['avg_ts']:8.1f} +- {d['stddev_ts']:5.1f}\")
"
done
