#!/usr/bin/env bash
# Stage 3g: text score and samples of the checkpoint the post ships (repaired reward, sampling at 0.7). Waits for stage 3e.
cd "$(dirname "$0")/../.."
until grep -q STAGE3E-DONE results/adapt/stage3e.log 2>/dev/null; do sleep 20; done
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
log=results/adapt/stage3g.log
{
$PY scripts/adapt/eval_bpb.py models/adapt/grpo-v2-t07 --tag grpo-v2-t07
$PY scripts/adapt/eval_ood.py models/adapt/grpo-v2-t07 --tag grpo-v2-t07
$PY scripts/adapt/samples.py models/adapt/grpo-v2-t07
echo "== STAGE3G-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
