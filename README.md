# ggml-inference-lab

Scripts, raw results and blog source for two posts:

- [Getting the most tokens per second out of one laptop with llama.cpp](https://inboxpraveen.github.io/blogs/blog-1.html):
  Qwen3-0.6B (and Qwen3-8B) on an i7-14650HX + RTX 5060 Laptop, every llama.cpp setting measured for
  speed and for accuracy (KL divergence against BF16).
- [Before you tune anything: a field guide](https://inboxpraveen.github.io/blogs/blog-2.html): what carries
  over to other machines and other architectures. Six small models (dense, MoE, sliding-window, two hybrids,
  pure SSM) plus T5 and an embedding model on the same tests, a pre-check script, a ten-minute KLD recipe,
  and the instruction-set versus quant experiment.

Nothing here needs a compiler. The prebuilt llama.cpp release zips and a conda env with
`numpy`, `cffi`, `matplotlib`, `pillow` and `gguf` are enough.

## Layout

```
scripts/        every measurement, one script each (bench.sh wraps llama-bench and tags rows)
                post 2: precheck.py, zoo.sh, zoo_meta.py, oldcpu.sh, kld_convergence.py, charts2.py, build_blog2.py
src/            llamabind.py (cffi ABI binding to llama.dll), driver.py (Engine + decode loops)
results/        bench.jsonl, server.jsonl, driver.jsonl, kld_table.json, load_modes.json,
                membw.txt, charts/ and charts2/ (SVG+PNG), kld/*.txt (per-variant llama-perplexity output),
                zoo_meta.json, kld_convergence.json, precheck-*.txt, t5-*.txt
research/       notes on DLL selection, runtime options, hardware ceilings, KLD methodology
blog/           post sources: blog-1.src.html + body-*.html (scripts/fill_blog.py), blog-2.src.html + b2-*.html (scripts/build_blog2.py)
```

Ignored by git because of size (see `.gitignore`): `models/` (18 GB of GGUF variants),
`bin/` (the llama.cpp b10941 zips, unpacked), `data/` (wikitext-2), `results/kld/*.kld`
(15.5 GB of stored BF16 logits) and `research/tmp/`.

## Reproduce

```bash
# 1. binaries: llama.cpp b10941 win-cpu-x64 and win-cuda-13.3-x64 zips, unpacked to bin/cpu and bin/cuda
# 2. models and data
bash scripts/download.sh            # official Qwen3-0.6B Q8_0, unsloth BF16 + imatrix, wikitext-2
# 3. quantization variants (21 files, from BF16 with the unsloth imatrix)
bash scripts/make_quants.sh
# 4. speed matrix (each script is one session; results append to results/bench.jsonl)
bash scripts/speed_matrix_1.sh      # ... through speed_matrix_5.sh
# 5. accuracy: base logits once (15.5 GB), then every variant
bash scripts/kld_all.sh && python scripts/collect_kld.py
# 6. server / driver comparisons
python scripts/bench_server.py --help
python scripts/bench_driver.py --help
# 7. charts and the post
python scripts/charts.py kld threads dll ngl depth ngl8b spec svb
python scripts/fill_blog.py

# post 2: the architecture zoo, pre-checks and the old-CPU experiment
python scripts/precheck.py                 # run this first on any machine
bash scripts/download_zoo.sh               # seven small GGUFs, one per architecture
llama-quantize models/zoo/mamba-130m-hf.F16.gguf models/zoo/mamba-130m-Q8_0.gguf Q8_0   # the published Q8_0 fails to load
python scripts/zoo_meta.py                 # bytes per token, KV per token, head share from the files
bash scripts/zoo.sh cpu; bash scripts/zoo.sh cuda; bash scripts/zoo.sh depth-cpu; bash scripts/zoo.sh depth-cuda; bash scripts/zoo.sh embed
bash scripts/oldcpu.sh                     # six quants under sse42 / sandybridge / alderlake kernels
python scripts/kld_convergence.py          # running KLD means at 5/20/40/200 chunks from the existing logs
python scripts/charts2.py ceilings arch kv moe zoo depth depthcuda battery oldcpu
python scripts/build_blog2.py
```

Traps met on the way: `llama-bench` asserts on encoder-decoder models (use `llama-completion`); `--chunks` is
ignored by `llama-perplexity` when a KLD base is supplied (the base decides); the QuantFactory mamba-130m
Q8_0 fails both the gguf reader and llama.cpp (requantize from F16); and a laptop on battery caps the GPU at
a fraction of its power limit without any warning in llama.cpp's output.

Every number in the post is looked up from `results/*.jsonl` by tag; `scripts/fill_blog.py` is the
map from placeholder to tag. If you rerun a script, the post's tables follow.

## Caveats

All measurements were taken on one laptop that was doing other work at the time. Paired,
back-to-back comparisons are the trustworthy ones; single numbers from different sessions can differ
by 30% or more, and prompt-processing numbers are the most sensitive. The post says which is which.
