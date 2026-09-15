#!/usr/bin/env bash
# Stage 2: the main continued-pretraining run, its bits-per-character score, then SFT and a first task evaluation.
# Usage: bash scripts/adapt/stage2.sh [main_steps] [lr] [min_lr]
cd "$(dirname "$0")/../.."
if [ -e results/adapt/stage2.lock ]; then echo "stage2 already running"; exit 1; fi
touch results/adapt/stage2.lock; trap "rm -f results/adapt/stage2.lock" EXIT
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
STEPS=${1:-3000}; LR=${2:-5e-4}; MINLR=${3:-5e-5}
log=results/adapt/stage2.log
{
echo "== main $(date)"
$PY scripts/adapt/cpt.py --tag main --tokenizer models/adapt/tok-16k --init mean --steps $STEPS --warmup 100 --eval-every 100 --lr $LR --min-lr $MINLR --save
echo "== bpb of the checkpoints $(date)"
$PY scripts/adapt/eval_bpb.py models/adapt/main --tag cpt-main
$PY scripts/adapt/eval_bpb.py models/adapt/run1200 --tag run1200
echo "== sft $(date)"
$PY scripts/adapt/sft.py --model models/adapt/main --tag sft --epochs 2 --lr 1e-4
echo "== task eval of sft $(date)"
$PY scripts/adapt/eval_task.py models/adapt/sft --tag sft --n 200
echo "== STAGE2-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
