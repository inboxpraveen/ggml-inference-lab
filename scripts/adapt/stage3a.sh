#!/usr/bin/env bash
# Stage 3a: raw samples from the stock and CPT checkpoints, and the out-of-distribution Hindi score (Wikipedia)
# for every model in the size chart. Runs after stage 2; GRPO is launched separately once the SFT eval is read.
cd "$(dirname "$0")/../.."
if [ -e results/adapt/stage3a.lock ]; then echo "stage3a already running"; exit 1; fi
touch results/adapt/stage3a.lock; trap "rm -f results/adapt/stage3a.lock" EXIT
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
log=results/adapt/stage3a.log
{
echo "== samples $(date)"
$PY scripts/adapt/samples.py HuggingFaceTB/SmolLM2-135M --raw
$PY scripts/adapt/samples.py models/adapt/main --raw
echo "== wikipedia bpb $(date)"
$PY scripts/adapt/eval_ood.py HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B sarvamai/sarvam-1
$PY scripts/adapt/eval_ood.py models/adapt/run1200 --tag run1200
$PY scripts/adapt/eval_ood.py models/adapt/main --tag cpt-main
echo "== STAGE3A-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
