#!/usr/bin/env bash
# Stage 3c: the Wikipedia (out-of-distribution) Hindi score for every model in the size chart. Waits for stage 3b.
cd "$(dirname "$0")/../.."
until grep -q STAGE3B-DONE results/adapt/stage3b.log 2>/dev/null; do sleep 20; done
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
log=results/adapt/stage3c.log
{
echo "== wikipedia bpb $(date)"
$PY scripts/adapt/eval_ood.py HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B sarvamai/sarvam-1
$PY scripts/adapt/eval_ood.py models/adapt/run1200 --tag run1200
$PY scripts/adapt/eval_ood.py models/adapt/main --tag cpt-main
$PY scripts/adapt/eval_ood.py models/adapt/grpo --tag grpo
echo "== STAGE3C-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
