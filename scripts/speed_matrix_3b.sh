#!/bin/bash
LAB="/c/PK/Github-Projects/ggml-inference-lab"; cd "$LAB"
B="bash scripts/bench.sh"
M=models/Qwen3-0.6B-Q8_0.gguf
PIN="-t 8"
echo "### tensor placement: head on CPU, FFN on CPU"
$B ot-head-cpu cuda -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 99 $PIN -ot "token_embd=CPU"
$B ot-ffn-cpu-8 cuda -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 99 $PIN -ot "blk\.([0-7])\.ffn_.*=CPU"
$B nkvo cuda -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 99 $PIN -nkvo 1
echo "### CPU threads for prompt vs decode"
$B cpu-tb cpu -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 0 -t 8,16 -C 0x5555,0xFFFF --cpu-strict 1
echo "### PART3B DONE"
