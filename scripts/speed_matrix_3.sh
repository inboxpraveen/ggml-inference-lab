#!/bin/bash
# Speed matrix part 3: GPU knobs on the 0.6B, offload splits, KV cache types, batch sizes, tensor placement.
LAB="/c/PK/Github-Projects/ggml-inference-lab"; cd "$LAB"
B="bash scripts/bench.sh"
M=models/Qwen3-0.6B-Q8_0.gguf
PIN="-t 8"

echo "### ngl sweep (0.6B has 28 layers; 29 = +output)"
$B ngl-sweep cuda -m $M -p 256 -n 128 -r 3 --prio 2 -ngl 0,4,8,12,16,20,24,26,27,28,99 $PIN

echo "### flash attention and KV cache types (full offload)"
$B fa-kv cuda -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 99 -fa on,off
$B kv-q8 cuda -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0
$B kv-q4 cuda -m $M -p 512 -n 128 -r 5 --prio 2 -ngl 99 -fa on -ctk q4_0 -ctv q4_0

echo "### depth: decode at 0 / 2k / 8k / 16k context"
$B depth-cuda cuda -m $M -p 0 -n 64 -r 3 --prio 2 -ngl 99 -fa on -d 0,2048,8192,16384
$B depth-cuda-kvq8 cuda -m $M -p 0 -n 64 -r 3 --prio 2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -d 0,2048,8192,16384
$B depth-cpu cpu -m $M -p 0 -n 64 -r 3 --prio 2 -ngl 0 $PIN -fa on -d 0,2048,8192
$B depth-cpu-fa-off cpu -m $M -p 0 -n 64 -r 3 --prio 2 -ngl 0 $PIN -fa off -d 0,2048,8192

echo "### ubatch for prompt processing"
$B ubatch-cuda cuda -m $M -p 2048 -n 0 -r 3 --prio 2 -ngl 99 -ub 64,128,256,512,1024,2048
$B ubatch-cpu cpu -m $M -p 1024 -n 0 -r 3 --prio 2 -ngl 0 $PIN -ub 32,64,128,256,512

echo "### tensor placement: head on CPU, FFN on CPU"
$B ot-head-cpu cuda -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 99 $PIN -ot "token_embd=CPU"
$B ot-ffn-cpu-8 cuda -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 99 $PIN -ot "blk\.([0-7])\.ffn_.*=CPU"
$B nkvo cuda -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 99 $PIN -nkvo 1

echo "### CPU threads for prompt vs decode (pp likes more threads)"
$B cpu-tb cpu -m $M -p 512 -n 128 -r 3 --prio 2 -ngl 0 -t 8,16 -C 0x5555,0xFFFF --cpu-strict 1
echo "### PART3 DONE"
