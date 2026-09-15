#!/usr/bin/env bash
# Stage 3b: GRPO from the SFT checkpoint, then every task and text evaluation the last section quotes.
# Usage: bash scripts/adapt/stage3b.sh [sft_checkpoint] [grpo_steps]
cd "$(dirname "$0")/../.."
if [ -e results/adapt/stage3b.lock ]; then echo "stage3b already running"; exit 1; fi
touch results/adapt/stage3b.lock; trap "rm -f results/adapt/stage3b.lock" EXIT
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
SFT=${1:-models/adapt/sft}; STEPS=${2:-300}
log=results/adapt/stage3b.log
{
echo "== grpo from $SFT $(date)"
$PY scripts/adapt/grpo.py --model $SFT --tag grpo --steps $STEPS
echo "== task eval, full test set $(date)"
$PY scripts/adapt/eval_task.py models/adapt/grpo --tag grpo
$PY scripts/adapt/eval_task.py $SFT --tag sft
$PY scripts/adapt/eval_task.py HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-1.7B --template
echo "== text scores of the post-trained checkpoints $(date)"
$PY scripts/adapt/eval_bpb.py $SFT --tag sft
$PY scripts/adapt/eval_bpb.py models/adapt/grpo --tag grpo
echo "== samples $(date)"
$PY scripts/adapt/samples.py $SFT models/adapt/grpo
echo "== STAGE3B-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
