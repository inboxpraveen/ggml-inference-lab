#!/usr/bin/env bash
# Stage 3d: GRPO from the SFT checkpoint that had no cold-start examples (format rate 0), to show what the
# group-normalised advantage does when no sample ever scores. Waits for stage 3c.
cd "$(dirname "$0")/../.."
until grep -q STAGE3C-DONE results/adapt/stage3c.log 2>/dev/null; do sleep 20; done
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
log=results/adapt/stage3d.log
{
echo "== grpo from the plain sft $(date)"
$PY scripts/adapt/grpo.py --model models/adapt/sft --tag grpo-nocold --steps 150 --eval-every 50
echo "== full task eval of the plain sft $(date)"
$PY scripts/adapt/eval_task.py models/adapt/sft --tag sft-plain
echo "== STAGE3D-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
