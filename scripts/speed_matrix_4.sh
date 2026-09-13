#!/bin/bash
# Speed matrix part 4: Qwen3-8B Q6_K, a model that does not quite fit the 8 GB card.
LAB="/c/PK/Github-Projects/ggml-inference-lab"; cd "$LAB"
B="bash scripts/bench.sh"
M=models/Qwen3-8B-Q6_K.gguf
PIN="-t 8"

echo "### 8B: CPU only"
$B 8b-cpu cpu -m $M -p 128 -n 32 -r 3 --prio 2 -ngl 0 $PIN

echo "### 8B: ngl sweep (36 layers)"
for n in 16 24 28 30 32 33 34 35 36 99; do
  $B 8b-ngl-$n cuda -m $M -p 256 -n 64 -r 3 --prio 2 -ngl $n $PIN -fa on
done

echo "### 8B: keep FFN of first N layers on CPU instead of whole layers (llama-bench has -ot, not -ncffn)"
$B 8b-otffn-4  cuda -m $M -p 256 -n 64 -r 3 --prio 2 -ngl 99 $PIN -fa on -ot "blk\.[0-3]\.ffn_.*=CPU"
$B 8b-otffn-8  cuda -m $M -p 256 -n 64 -r 3 --prio 2 -ngl 99 $PIN -fa on -ot "blk\.[0-7]\.ffn_.*=CPU"
$B 8b-otffn-12 cuda -m $M -p 256 -n 64 -r 3 --prio 2 -ngl 99 $PIN -fa on -ot "blk\.([0-9]|1[01])\.ffn_.*=CPU"

echo "### 8B: KV cache q8_0 to buy VRAM"
$B 8b-kvq8-ngl99 cuda -m $M -p 256 -n 64 -r 3 --prio 2 -ngl 99 $PIN -fa on -ctk q8_0 -ctv q8_0
$B 8b-kvq8-ngl34 cuda -m $M -p 256 -n 64 -r 3 --prio 2 -ngl 34 $PIN -fa on -ctk q8_0 -ctv q8_0
echo "### PART4 DONE"
