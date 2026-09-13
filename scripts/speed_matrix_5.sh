#!/bin/bash
# Speed matrix part 5: end-to-end runs. llama-server with and without built-in speculative modes,
# the cffi driver's three loops, and llama-completion as the stock CLI reference.
LAB="/c/PK/Github-Projects/ggml-inference-lab"; cd "$LAB"
PY="conda run -n infer-lab --no-capture-output python"
M=models/Qwen3-0.6B-Q8_0.gguf
M8=models/Qwen3-8B-Q6_K.gguf

echo "### server, 0.6B on CPU"
$PY scripts/bench_server.py --bin cpu --tag srv-cpu-none        -- -m $M -ngl 0 -t 8 -tb 16
$PY scripts/bench_server.py --bin cpu --tag srv-cpu-ngram-simple -- -m $M -ngl 0 -t 8 -tb 16 --spec-type ngram-simple
$PY scripts/bench_server.py --bin cpu --tag srv-cpu-ngram-mod    -- -m $M -ngl 0 -t 8 -tb 16 --spec-type ngram-mod
$PY scripts/bench_server.py --bin cpu --tag srv-cpu-ngram-cache  -- -m $M -ngl 0 -t 8 -tb 16 --spec-type ngram-cache

echo "### server, 0.6B on CUDA"
$PY scripts/bench_server.py --bin cuda --tag srv-cuda-none         -- -m $M -ngl 99 -t 8
$PY scripts/bench_server.py --bin cuda --tag srv-cuda-ngram-simple -- -m $M -ngl 99 -t 8 --spec-type ngram-simple
$PY scripts/bench_server.py --bin cuda --tag srv-cuda-ngram-mod    -- -m $M -ngl 99 -t 8 --spec-type ngram-mod

echo "### driver, 0.6B"
$PY scripts/bench_driver.py --model $M --bin bin/cpu  --ngl 0  --threads 8 --n-predict 256 --reps 3 --tag drv-cpu
$PY scripts/bench_driver.py --model $M --bin bin/cuda --ngl 99 --threads 8 --n-predict 256 --reps 3 --tag drv-cuda
$PY scripts/bench_driver.py --model $M --bin bin/cuda --ngl 99 --threads 8 --n-predict 256 --reps 3 --tag drv-cuda-ng16 --n-draft 16 --loops ngram_spec_loop
$PY scripts/bench_driver.py --model $M --bin bin/cpu  --ngl 0  --threads 8 --n-predict 256 --reps 3 --tag drv-cpu-ng16 --n-draft 16 --loops ngram_spec_loop
$PY scripts/bench_driver.py --model $M --bin bin/cpu  --ngl 0  --threads 8 --n-predict 256 --reps 3 --tag drv-cpu-ng2 --n-gram 2 --n-draft 8 --loops ngram_spec_loop

echo "### server, 8B on CUDA (best split from part 4 is filled in by hand)"
NGL8B=${NGL8B:-34}
$PY scripts/bench_server.py --bin cuda --tag srv-8b-none      --n-predict 128 --reps 2 -- -m $M8 -ngl $NGL8B -t 8 -fa on
$PY scripts/bench_server.py --bin cuda --tag srv-8b-ngram-mod --n-predict 128 --reps 2 -- -m $M8 -ngl $NGL8B -t 8 -fa on --spec-type ngram-mod
$PY scripts/bench_server.py --bin cuda --tag srv-8b-draft-0.6b --n-predict 128 --reps 2 -- -m $M8 -ngl $NGL8B -t 8 -fa on --spec-type draft-simple -md $M -ngld 99 --spec-draft-n-max 6
echo "### PART5 DONE"
