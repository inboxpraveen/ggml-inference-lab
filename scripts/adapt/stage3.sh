#!/usr/bin/env bash
# Stage 3: the SFT rerun with the cold-start examples (the plain SFT put out no valid JSON), its task eval,
# then stage 3b (GRPO and every final evaluation) from that checkpoint.
cd "$(dirname "$0")/../.."
while [ -e results/adapt/stage3a.lock ]; do sleep 20; done
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
log=results/adapt/stage3.log
{
echo "== sft with cold start $(date)"
$PY scripts/adapt/sft.py --model models/adapt/main --tag sft-cold --epochs 2 --lr 1e-4 --cold-start 100
echo "== task eval of sft-cold $(date)"
$PY scripts/adapt/eval_task.py models/adapt/sft-cold --tag sft-cold --n 200
echo "== STAGE3-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
bash scripts/adapt/stage3b.sh models/adapt/sft-cold 300
