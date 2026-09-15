#!/usr/bin/env bash
# Stage 3e: two more GRPO questions, after stage 3f. (1) Does sampling closer to greedy let the repaired reward
# see the looping mode? Same run at temperature 0.7. (2) How much cold start does the reward need? SFT with 20
# labelled examples, then GRPO from it.
cd "$(dirname "$0")/../.."
until grep -q STAGE3F-DONE results/adapt/stage3f.log 2>/dev/null; do sleep 20; done
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
log=results/adapt/stage3e.log
{
echo "== grpo, v2 reward, temperature 0.7 $(date)"
$PY scripts/adapt/grpo.py --model models/adapt/sft-cold --tag grpo-v2-t07 --steps 150 --reward v2 --temperature 0.7
$PY scripts/adapt/eval_task.py models/adapt/grpo-v2-t07 --tag grpo-v2-t07
echo "== sft with 20 cold-start examples $(date)"
$PY scripts/adapt/sft.py --model models/adapt/main --tag sft-cold20 --epochs 2 --lr 1e-4 --cold-start 20
$PY scripts/adapt/eval_task.py models/adapt/sft-cold20 --tag sft-cold20 --n 200
echo "== grpo from it $(date)"
$PY scripts/adapt/grpo.py --model models/adapt/sft-cold20 --tag grpo-cold20 --steps 150 --eval-every 50
$PY scripts/adapt/eval_task.py models/adapt/grpo-cold20 --tag grpo-cold20 --n 200
echo "== STAGE3E-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
