#!/bin/bash
# Speed matrix part 2: force each CPU backend DLL variant by isolating it in its own folder.
LAB="/c/PK/Github-Projects/ggml-inference-lab"; cd "$LAB"
PIN="-t 8"
M8=models/Qwen3-0.6B-Q8_0.gguf
M4=models/variants/plain-Q4_K_M.gguf
mkdir -p bin/cpu-variants
for dll in bin/cpu/ggml-cpu-*.dll; do
  v=$(basename "$dll" .dll | sed 's/ggml-cpu-//')
  d="bin/cpu-variants/$v"
  if [ ! -d "$d" ]; then
    mkdir -p "$d"
    cp bin/cpu/*.exe bin/cpu/llama*.dll bin/cpu/ggml.dll bin/cpu/ggml-base.dll bin/cpu/libomp.dll bin/cpu/mtmd.dll "$d"/ 2>/dev/null
    cp "$dll" "$d"/
  fi
  echo "### variant $v"
  "$d/llama-bench.exe" --list-devices 2>&1 | grep -E "load_backend|CPU" | head -3
  out=$("$d/llama-bench.exe" -o jsonl -m "$M8,$M4" -p 256 -n 128 -r 5 --prio 2 -ngl 0 $PIN 2>"results/bench-dll-$v.err")
  echo "$out" | python -c "
import sys,json
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('{'): continue
    d=json.loads(line); d['tag']='dll-$v'; d['build']='cpu-variant:$v'; d['cmd']='-p 256 -n 128 -r 5 --prio 2 -ngl 0 $PIN'
    print(json.dumps(d))
" >> results/bench.jsonl
  echo "$out" | python -c "
import sys,json
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('{'): continue
    d=json.loads(line); print('   ', d['model_type'], d['n_prompt'] and 'pp'+str(d['n_prompt']) or 'tg'+str(d['n_gen']), round(d['avg_ts'],2), '+-', round(d['stddev_ts'],2))
"
  grep -i -E "error|not supported|no backend" "results/bench-dll-$v.err" | head -2
done
echo "### PART2 DONE"
