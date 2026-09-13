#!/bin/bash
# Speed matrix part 1: paired CPU baseline, quant sweep on pinned CPU and on CUDA.
LAB="/c/PK/Github-Projects/ggml-inference-lab"; cd "$LAB"
B="bash scripts/bench.sh"
M=models/Qwen3-0.6B-Q8_0.gguf
PIN="-t 8"

echo "### paired baseline (official Q8_0)"
$B pair-cpu-pinned-strict cpu -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 0 -t 8 -C 0x5555 --cpu-strict 1
$B pair-cpu-default cpu -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 0
$B pair-cpu-pinned-loose cpu -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 0 -t 8 -C 0x5555 --cpu-strict 0
$B pair-cpu-t8      cpu -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 0 -t 8
$B pair-cpu-prio0   cpu -m $M -p 512 -n 128 -r 5 --prio 0 -ngl 0 $PIN

echo "### quant sweep, CPU pinned"
V=$(ls models/variants/*.gguf | tr '\n' ',' | sed 's/,$//')
$B quant-cpu cpu -m "$V" -p 256 -n 128 -r 5 --prio 2 -ngl 0 $PIN
$B quant-cpu-bf16 cpu -m models/Qwen3-0.6B-BF16.gguf -p 256 -n 128 -r 3 --prio 2 -ngl 0 $PIN

echo "### quant sweep, CUDA full offload"
$B quant-cuda cuda -m "$V" -p 512 -n 128 -r 5 --prio 2 -ngl 99
$B quant-cuda-bf16 cuda -m models/Qwen3-0.6B-BF16.gguf -p 512 -n 128 -r 5 --prio 2 -ngl 99
$B quant-cuda-official cuda -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 99
echo "### PART1 DONE"
