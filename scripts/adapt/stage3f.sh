#!/usr/bin/env bash
# Stage 3f: GRPO again from the cold-start SFT with the repaired language reward (v2, loop penalty), then the
# task evaluation re-run for every model so each row carries both reward versions. Waits for stage 3e.
cd "$(dirname "$0")/../.."
until grep -q STAGE3D-DONE results/adapt/stage3d.log 2>/dev/null; do sleep 20; done
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="conda run -n adapt-lab --no-capture-output python"
log=results/adapt/stage3f.log
{
echo "== grpo with the v2 language reward $(date)"
$PY scripts/adapt/grpo.py --model models/adapt/sft-cold --tag grpo-v2 --steps 200 --reward v2
echo "== task eval, full test set, both rewards $(date)"
$PY scripts/adapt/eval_task.py models/adapt/grpo-v2 --tag grpo-v2
$PY scripts/adapt/eval_task.py models/adapt/grpo --tag grpo
$PY scripts/adapt/eval_task.py models/adapt/sft-cold --tag sft-cold
$PY scripts/adapt/eval_task.py HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-1.7B --template
$PY scripts/adapt/eval_bpb.py models/adapt/grpo-v2 --tag grpo-v2
$PY scripts/adapt/samples.py models/adapt/grpo-v2
echo "== STAGE3F-DONE $(date)"
} 2>&1 | grep --line-buffered -vE "Warning|warn|symlink|Developer|indexing errors|Loading weights|return t.to" > $log
