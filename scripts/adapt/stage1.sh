#!/usr/bin/env bash
# Stage 1: reference-model baselines, the CPT ablations (400 steps each), and the 1200-step point.
cd "$(dirname "$0")/../.."
# one instance at a time
if [ -e results/adapt/stage1.lock ]; then echo "stage1 already running"; exit 1; fi
touch results/adapt/stage1.lock; trap "rm -f results/adapt/stage1.lock" EXIT
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
F="grep -vE Warning|warn|symlink|Developer|indexing|Loading|return\ t.to"
log=results/adapt/stage1.log
{
A="--steps 400 --warmup 50 --eval-every 100 --max-hi-chars 150000000"
for spec in \
  "abl-stock --tokenizer HuggingFaceTB/SmolLM2-135M" \
  "abl-16k-mean --tokenizer models/adapt/tok-16k --init mean" \
  "abl-16k-random --tokenizer models/adapt/tok-16k --init random" \
  "abl-16k-hf --tokenizer models/adapt/tok-16k --init hf" \
  "abl-noreplay --tokenizer models/adapt/tok-16k --init mean --en-ratio 0" \
  "abl-replay30 --tokenizer models/adapt/tok-16k --init mean --en-ratio 0.3" \
  "abl-lr2e-4 --tokenizer models/adapt/tok-16k --init mean --lr 2e-4 --min-lr 2e-5" \
  "abl-lr1e-3 --tokenizer models/adapt/tok-16k --init mean --lr 1e-3 --min-lr 1e-4" \
  "abl-8k --tokenizer models/adapt/tok-8k --init mean" \
  "abl-32k --tokenizer models/adapt/tok-32k --init mean" ; do
  set -- $spec; tag=$1; shift
  echo "== $tag $(date)"
  $PY scripts/adapt/cpt.py --tag $tag $A "$@"
done
echo "== run1200 $(date)"
$PY scripts/adapt/cpt.py --tag run1200 --tokenizer models/adapt/tok-16k --init mean --steps 1200 --warmup 100 --eval-every 100 --save
echo "== baselines $(date)"
$PY scripts/adapt/eval_bpb.py Qwen/Qwen2.5-1.5B Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B sarvamai/sarvam-1 HuggingFaceTB/SmolLM2-135M-Instruct
echo "== STAGE1-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
