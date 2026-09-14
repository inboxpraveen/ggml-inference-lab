#!/usr/bin/env bash
# Architecture zoo: same three tests on every small model, one quiet session. Results append to results/bench.jsonl.
# usage: bash scripts/zoo.sh <cpu|cuda|depth-cpu|depth-cuda|embed>
cd "$(dirname "$0")/.."
B="bash scripts/bench.sh"
DEC="models/Qwen3-0.6B-Q8_0.gguf models/zoo/granite-3.1-1b-a400m-instruct-Q8_0.gguf models/zoo/gemma-3-1b-it-Q8_0.gguf \
models/zoo/Falcon-H1-0.5B-Instruct-Q8_0.gguf models/zoo/LFM2-700M-Q8_0.gguf models/zoo/mamba-130m-Q8_0.gguf"
case "$1" in
  cpu)        for m in $DEC; do $B zoo-cpu  cpu  -m $m -p 512 -n 128 -r 3 -t 8 -ngl 0 --prio 2; done ;;
  cuda)       for m in $DEC; do $B zoo-cuda cuda -m $m -p 512 -n 128 -r 3 -ngl 99 --prio 2; done ;;
  depth-cpu)  for m in $DEC; do $B zoo-depth-cpu  cpu  -m $m -p 0 -n 32 -d 0,2048,8192 -r 2 -t 8 -ngl 0 --prio 2; done ;;
  depth-cuda) for m in $DEC; do $B zoo-depth-cuda cuda -m $m -p 0 -n 32 -d 0,2048,8192 -r 2 -ngl 99 --prio 2; done ;;
  embed)      $B zoo-embed-cpu  cpu  -m models/zoo/nomic-embed-text-v1.5.Q8_0.gguf -embd 1 -p 128,512 -n 0 -r 5 -t 4,8,16 -ngl 0 --prio 2
              $B zoo-embed-cuda cuda -m models/zoo/nomic-embed-text-v1.5.Q8_0.gguf -embd 1 -p 128,512 -n 0 -r 5 -ngl 99 --prio 2 ;;
esac
