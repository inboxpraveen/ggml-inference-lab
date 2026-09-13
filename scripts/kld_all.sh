#!/bin/bash
# KL-divergence + perplexity of every variant against the BF16 base logits. GPU backend for all runs.
LAB="/c/PK/Github-Projects/ggml-inference-lab"
P="$LAB/bin/cuda/llama-perplexity.exe"
BASE="$LAB/results/kld/base-bf16-c512-200.kld"
F="$LAB/data/wikitext-2-raw/wiki.test.raw"
run() { # run <label> <gguf>
  local out="$LAB/results/kld/$1.txt"
  if grep -q "Same top p" "$out" 2>/dev/null; then echo "skip $1"; return; fi
  echo "== $1"
  "$P" -m "$2" -f "$F" -c 512 --chunks 200 -ngl 99 -t 8 --kl-divergence-base "$BASE" --kl-divergence > "$out" 2>&1
  grep -E "Mean PPL\(Q\)|Mean KLD|Same top p|Mean.*Δp|RMS" "$out" | head -8
}
run official-Q8_0 "$LAB/models/Qwen3-0.6B-Q8_0.gguf"
run bf16-self "$LAB/models/Qwen3-0.6B-BF16.gguf"
for f in "$LAB"/models/variants/*.gguf; do
  n=$(basename "$f" .gguf)
  run "$n" "$f"
done
echo KLD DONE
