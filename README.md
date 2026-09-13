# ggml-inference-lab

Scripts, raw results and blog source for the post
["One laptop, one model, every knob llama.cpp has"](https://inboxpraveen.github.io/blogs/blog-1.html):
Qwen3-0.6B (and Qwen3-8B) on an i7-14650HX + RTX 5060 Laptop, every llama.cpp setting measured for
speed and for accuracy (KL divergence against BF16).

Nothing here needs a compiler. The prebuilt llama.cpp release zips and a conda env with
`numpy`, `cffi`, `matplotlib`, `pillow` and `gguf` are enough.

## Layout

```
scripts/        every measurement, one script each (bench.sh wraps llama-bench and tags rows)
src/            llamabind.py (cffi ABI binding to llama.dll), driver.py (Engine + decode loops)
results/        bench.jsonl, server.jsonl, driver.jsonl, kld_table.json, load_modes.json,
                membw.txt, charts/ (SVG+PNG), kld/*.txt (per-variant llama-perplexity output)
research/       notes on DLL selection, runtime options, hardware ceilings, KLD methodology
blog/           the post's source: blog-1.src.html + body-*.html, assembled by scripts/fill_blog.py
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
```

Every number in the post is looked up from `results/*.jsonl` by tag; `scripts/fill_blog.py` is the
map from placeholder to tag. If you rerun a script, the post's tables follow.

## Caveats

All measurements were taken on one laptop that was doing other work at the time. Paired,
back-to-back comparisons are the trustworthy ones; single numbers from different sessions can differ
by 30% or more, and prompt-processing numbers are the most sensitive. The post says which is which.
