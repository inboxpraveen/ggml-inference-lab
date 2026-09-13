#!/bin/bash
set -e
LAB="/c/PK/Github-Projects/ggml-inference-lab"
cd "$LAB/bin"
B=b10941
for z in llama-$B-bin-win-cpu-x64.zip llama-$B-bin-win-cuda-13.3-x64.zip cudart-llama-bin-win-cuda-13.3-x64.zip llama-$B-bin-win-vulkan-x64.zip; do
  [ -f "$z" ] || curl -sL -o "$z" "https://github.com/ggml-org/llama.cpp/releases/download/$B/$z"
  echo "got $z $(stat -c %s $z)"
done
cd "$LAB/models"
[ -f Qwen3-0.6B-Q8_0.gguf ] || curl -sL -o Qwen3-0.6B-Q8_0.gguf "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true"
echo "got Q8 $(stat -c %s Qwen3-0.6B-Q8_0.gguf)"
[ -f Qwen3-0.6B-BF16.gguf ] || curl -sL -o Qwen3-0.6B-BF16.gguf "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf?download=true"
echo "got BF16 $(stat -c %s Qwen3-0.6B-BF16.gguf)"
[ -f imatrix_unsloth.dat ] || curl -sL -o imatrix_unsloth.dat "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/imatrix_unsloth.dat?download=true"
echo "got imatrix $(stat -c %s imatrix_unsloth.dat)"
echo DONE
