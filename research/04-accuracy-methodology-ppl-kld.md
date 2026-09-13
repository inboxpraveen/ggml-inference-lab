# 04 — Measuring quantization accuracy rigorously with llama.cpp (PPL + KL divergence)

Research date: 2026-09-13. Target build: **llama.cpp b10941** (`version: 0.4.0-dev (build 10941, commit 4a8993735)`, built with Clang 20.1.8 for Windows x86_64 — from `llama-perplexity.exe --version` on the local prebuilt binaries).

**Source-freshness check.** Commit `4a8993735` ("tests : reduce FA test sizes (#28842)", committed 2026-09-13T10:05:28Z) is the tip of `master` today. I diffed every file cited below between `raw.githubusercontent.com/ggml-org/llama.cpp/master/...` and `.../4a8993735/...`: `tools/perplexity/perplexity.cpp`, `tools/perplexity/README.md`, `tools/imatrix/README.md`, `common/imatrix-loader.cpp`, `scripts/get-wikitext-2.sh`, `models/templates/Qwen-Qwen3-0.6B.jinja` are **byte-identical**. So "master, Sept 2026" and "b10941" are the same code for this topic.

Machine context (from the brief): i7-14650HX (8P+8E, 24 threads, AVX2/AVX-VNNI, no AVX-512), one 16 GB DDR5-5600 DIMM, RTX 5060 Laptop 8 GB (CC 12.0, driver 616.56), NVMe, Windows 11. Nothing was benchmarked for this report; only `--help`/`--version` were run from a copy in `research/tmp/cpu/`.

---

## 1. Verification status table

| Claim | Status | Source |
|---|---|---|
| CLI flags `--kl-divergence`, `--kl-divergence-base`/`--save-all-logits`, `--chunks`, `-c`, `-b`, `--ppl-stride`, `--ppl-output-type`, `--hellaswag`, `--winogrande`, `--multiple-choice` exist in b10941 | VERIFIED | `research/tmp/perplexity-help.txt` (local `--help`) + `common/arg.cpp` lines 1745-1749, 2460-2529 |
| Base logits are stored as **uint16 (affine-quantized log-probs), not fp16** | VERIFIED | `tools/perplexity/perplexity.cpp` `log_softmax(int, const float*, uint16_t*, int)` (lines 79-107) |
| File-size formula (exact, see §3.3) | VERIFIED by reading the writer + arithmetic | `perplexity.cpp` lines 461-470, 521-527, 172 |
| `wiki.test.raw` URL and file | VERIFIED | `scripts/get-wikitext-2.sh` |
| `wiki.test.raw` = 1,290,590 bytes, sha256 `173c87a5…dd08`, **299,078 tokens with the Qwen3 tokenizer** (HF `tokenizers`, no BOS) | VERIFIED locally (HF tokenizer; llama.cpp's tokenizer may differ by a handful of tokens — UNVERIFIED to the token) | scratchpad run |
| Qwen3-0.6B GGUF: `n_vocab = 151936`, `tokenizer.ggml.add_bos_token = false`, `tokenizer.ggml.pre = qwen2`, context 40960 | VERIFIED | GGUF header of `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf` (range-fetched, parsed) |
| `llama-quantize` still reads legacy `.dat` imatrix (unsloth's `imatrix_unsloth.dat`) without conversion | VERIFIED | `common/imatrix-loader.cpp` (GGUF-then-legacy fallback) + `tools/quantize/quantize.cpp` `load_imatrix()` |
| `imatrix_unsloth.dat` (unsloth) and `Qwen_Qwen3-0.6B.imatrix` (bartowski) are **legacy format** (no `GGUF` magic; first int32 = 196 entries, then `blk.27.ffn_down.weight`) | VERIFIED | range-fetch of first 64 bytes of each |
| `convert_legacy_imatrix_to_gguf.py` does **not** exist in the repo | VERIFIED (404 on raw + absent from full tree listing, 3,957 paths) | GitHub trees API |
| Conversion is `llama-imatrix --in-file x.dat -o x.gguf` | VERIFIED | `tools/imatrix/README.md` |
| `examples/llama-eval/llama-eval.py` exists (gsm8k/aime/aime2025/aime2026/gpqa; **no MMLU**) | VERIFIED | tree listing + script argparse |
| In-tree MMLU = `llama-perplexity --multiple-choice -f mmlu-validation.bin` | VERIFIED | `perplexity.cpp` lines 1395-1405; HF dataset `ikawrakow/validation-datasets-for-llama.cpp` (last modified 2024-03-11) |
| Qwen3 GGUF chat template defaults to thinking **ON** | VERIFIED | `models/templates/Qwen-Qwen3-0.6B.jinja` lines 80-85 + GGUF-embedded template |
| `--reasoning [on|off|auto]`, `--reasoning-budget N`, `--reasoning-effort`, per-request `chat_template_kwargs {"enable_thinking": false}` exist in b10941 llama-server | VERIFIED | `research/tmp/server-help.txt`, `common/arg.cpp` 3658-3730, `tools/server/README.md` |
| "KLD < 0.01 = indistinguishable / 0.01-0.05 small / > 0.1 noticeable" is stated in discussions #2875 or #5263 | **NOT TRUE — those tiers are not in either thread** (see §5) | fetched both |
| Runtime estimates | UNVERIFIED (derived from token counts; no run allowed) | §7 |
| Inferred details: VRAM output-buffer size, meaning of legacy imatrix `ncall` (688 / 137) for 2025-era files, whether the GGUF loader logs before falling back to legacy | UNVERIFIED (flagged inline) | §7, §9 |

---

## 2. Perplexity in llama.cpp: what it measures and how

Source: `tools/perplexity/README.md` (master == b10941).

> "Perplexity measures how well the model can predict the next token with lower values being better. Note that perplexity is **not** directly comparable between models, especially if they use different tokenizers. Also note that finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases."
>
> "Within llama.cpp the perplexity of base models is used primarily to judge the quality loss from e.g. quantized models vs. FP16. The convention among contributors is to use the Wikitext-2 test set for testing unless noted otherwise (can be obtained with `scripts/get-wikitext-2.sh`)."
>
> "llama.cpp numbers are **not** directly comparable to those of other projects because the exact values depend strongly on the implementation details."

### 2.1 The chunking rule (from `perplexity.cpp`, function `perplexity()`)

```cpp
// We get the logits for all the tokens in the context window (params.n_ctx)
// from llama_decode below.  Now, based on https://huggingface.co/docs/transformers/perplexity,
// calculate the perplexity over the last half of the window (so the model always has
// some context to predict the token).
...
// Example, we have a context window of 512, we will compute perplexity for each of the
// last 256 tokens.  Then, we split the input up into context window size chunks to
// process the entire prompt.
const int first = n_ctx/2;
```

- `n_chunk_max = tokens.size() / n_ctx`; `n_chunk = params.n_chunks < 0 ? n_chunk_max : min(params.n_chunks, n_chunk_max)` (lines 493-495).
- Scored tokens per chunk = `n_ctx - 1 - first` = `n_ctx - 1 - n_ctx/2` (255 at `-c 512`, 1023 at `-c 2048`; line 620/629).
- Minimum input: `tokens.size() >= 2*n_ctx` or it errors (line 480).
- BOS: "add BOS token for the first batch of each chunk" **only if** `llama_vocab_get_add_bos(vocab)`; Qwen3 GGUF has `add_bos_token=false`, so nothing is prepended. `GGML_ASSERT(!llama_vocab_get_add_eos(vocab))`.
- Default context for the tool is **512** (`params.n_ctx = 512;` in `llama_perplexity()`; `--help` prints `-c, --ctx-size N  size of the prompt context (default: 512, 0 = loaded from model)`).
- Batch/sequence coupling (lines 504-508 and main): `n_seq = max(1, n_batch / n_ctx)`, `GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0)`, and main sets `params.n_parallel = max(1, n_batch/n_ctx); params.n_ctx = n_parallel * n_ctx; params.n_batch = min(n_batch, n_ctx)`. With defaults `-b 2048 -c 512` you get **4 chunks decoded in parallel** inside one 2048-token context. Choose `-b` as a multiple of `-c` (or smaller than it).
- Uncertainty: "The uncertainty is determined empirically by assuming a Gaussian distribution of the 'correct' logits per and then applying error propagation" (README). Code: `mean_and_uncertainty()` = sample std of per-token NLL / sqrt(count-1), then `ppl_unc = ppl * log_ppl.second`.

### 2.2 `--ppl-stride` and `--ppl-output-type`

- `--ppl-stride N` (default 0): if > 0, `perplexity()` dispatches to `perplexity_v2()`, a sliding-window variant (`n_chunk_max = (tokens - calc_chunk + stride - 1)/stride`, scores the last `stride` tokens of each window) and main bumps `params.n_ctx += params.ppl_stride/2`. **`kl_divergence()` never uses it** — do not combine `--ppl-stride` with `--kl-divergence`.
- `--ppl-output-type <0|1>` (default 0): 0 = the usual `[chunk/n] ppl` running print; 1 = `"%8d  %.4lf\n"` i.e. `num_tokens  ppl` one line per chunk (`common.h`: "= 1 -> ppl output is num_tokens, ppl, one per line"). Useful for plotting PPL vs. position.

### 2.3 Wikitext-2 download (exact)

`scripts/get-wikitext-2.sh` (verbatim, key lines):

```sh
ZIP="wikitext-2-raw-v1.zip"
FILE="wikitext-2-raw/wiki.test.raw"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/$ZIP"
...
  llama-perplexity -m model.gguf -f $FILE [other params]
```

Verified download: zip = 4,721,645 bytes (sha256 `ef7edb56…35a11`); `wiki.test.raw` = 1,290,590 bytes (sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`), plus `wiki.train.raw` (10.9 MB) and `wiki.valid.raw` (1.1 MB) in the same zip. On Windows: `curl -L -o wikitext-2-raw-v1.zip <URL>` then `tar -xf wikitext-2-raw-v1.zip` (Windows 11 ships bsdtar).

Tokenized with the Qwen3 tokenizer (`Qwen/Qwen3-0.6B/tokenizer.json`, HF `tokenizers`, no special tokens): **299,078 tokens** → `n_chunk_max` = 584 @512, 292 @1024, 146 @2048, 73 @4096. (llama.cpp's own BPE may differ by a few tokens — treat 584/292/146 as ±1.)

---

## 3. KL divergence mode: flags, file format, size

### 3.1 The two-pass workflow (README + `examples/model-conversion/scripts/utils/perplexity-gen.sh` / `perplexity-run.sh`)

(Command examples use cmd.exe `^` line continuation; in PowerShell use a backtick or put each command on one line.)

Pass 1 — record base logits (BF16 or F16 model):

```
llama-perplexity -m Qwen3-0.6B-BF16.gguf -f wikitext-2-raw/wiki.test.raw ^
   -c 512 -b 2048 --chunks 55 -ngl 99 -fa on --kl-divergence-base base-c512-ch55.kld
```

Pass 2 — score a quant against it (note: **`-f` is not needed**; the token stream is read back from the `.kld` file, exactly as `perplexity-run.sh` does):

```
llama-perplexity -m Qwen3-0.6B-Q4_K_M.gguf -c 512 -b 2048 -ngl 99 -fa on ^
   --kl-divergence-base base-c512-ch55.kld --kl-divergence
```

Flag semantics (from `common/arg.cpp`):

| Flag | Meaning | Applies to |
|---|---|---|
| `--kl-divergence-base FNAME` (alias `--save-all-logits`) | "set logits file" — **written** when `--kl-divergence` is absent, **read** when it is present | PERPLEXITY example only |
| `--kl-divergence` | "computes KL-divergence to logits provided via --kl-divergence-base" | PERPLEXITY |
| `--chunks N` | "max number of chunks to process (default: -1, -1 = all)"; shared with imatrix/retrieval | PERPLEXITY, IMATRIX |
| `-c N` | tool default 512; must match the base file (see 3.4) | all |
| `-b N` | logical batch, default 2048; `n_seq = n_batch / n_ctx` parallel chunks | all |

### 3.2 What is actually stored — uint16 affine-quantized log-probs, not fp16

README (verbatim): "In order to save space this file does **not** contain the exact same FP32 logits but instead casts them to 16 bit unsigned integers (with some scaling). So the 'f16' results are to be understood as the difference resulting only from this downcast."

Code (`perplexity.cpp` lines 79-107), per scored position:

```cpp
min_logit = std::max(min_logit, max_logit - 16);          // clamp dynamic range to e^-16
const float log_sum_exp = log(sum_exp);
const float min_log_prob = min_logit - max_logit - log_sum_exp;
const float scale = (max_logit - min_logit)/65535.f;
float * d = (float *)log_prob;  d[0] = scale;  d[1] = min_log_prob;   // 2 floats = 4 uint16 slots
log_prob += 4;
log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
```

Reader reconstructs `p_log_base = scale*base_log_prob[i] + min_log_prob` and only accumulates KLD terms where `p_log_base > -16` (line ~226). ikawrakow in PR #5076 (merged 2024-01-22): "I have decided to limit the probability range to e^(-16) to slightly improve the precision of the 16-bit values being stored"; it "reduc[es] the storage from 20 GB to 10 GB for wiki.test.run". He measured the encoding floor as "a mean KL-divergence of 4e-6" for fp16 vs its own stored logits (LLaMA-2 7B, 32K vocab). The README LLaMA-3-8B table shows the f16-vs-own-file row at KLD 0.000551 ± 0.000002 and RMS Δp 0.787 %; the BF16-base/FP16-model row shows 0.00002515. **Do a self-test (base model vs its own file) to measure your floor on your setup** — see protocol step 3.

### 3.3 Exact file layout and size

Writer (lines 468-469, 523-526, 172):

```
"_logits_"                       8 bytes
int32 n_ctx                      4
int32 n_vocab                    4
int32 n_chunk                    4
llama_token tokens[n_chunk*n_ctx]           4 * n_chunk * n_ctx
per chunk: uint16 log_probs[(n_ctx-1-n_ctx/2) * nv],  nv = 2*((n_vocab+1)/2) + 4
```

So:

`size ≈ 20 + 4·n_chunk·n_ctx + 2·n_chunk·(n_ctx − 1 − n_ctx/2)·(n_vocab_even + 4)`

The brief's approximation `n_chunks · n_ctx/2 · vocab · 2 bytes` is correct to ~0.5 % (it omits the "−1" token, the +4 header slots per position, and the token table).

For Qwen3 (`n_vocab = 151936` → `nv = 151940` → **303,880 bytes per scored token**):

| `-c` | scored tok/chunk | MiB/chunk | all chunks (584/292/146) | chunks that fit in 4 GiB | scored tokens in that budget |
|---|---|---|---|---|---|
| 512 | 255 | 73.9 | **42.1 GiB** | **55** (3.97 GiB) | 14,025 |
| 1024 | 511 | 148.1 | 42.2 GiB | 27 (3.90 GiB) | 13,797 |
| 2048 | 1023 | 296.5 | 42.3 GiB | 13 (3.76 GiB) | 13,299 |

Key insight: a disk budget buys a fixed number of **scored tokens** (~14 K per 4 GiB with a 152 K vocab) regardless of `-c`. The full-wikitext file is 42 GiB for this vocab — larger than the README's "37 GiB for LLaMA 3" (128 K vocab) — and would not fit comfortably on a laptop NVMe next to models.

### 3.4 Reader sanity checks that only warn (do not abort)

`kl_divergence()` (lines 1695-1735): checks magic `_logits_` (aborts if wrong), reads `n_ctx`, and **only `LOG_ERR`s** — does not return — if `n_ctx > llama_n_ctx(ctx)` ("has been computed with %u, while the current context is %d. Increase it with -c and retry") or if `n_vocab != llama_vocab_n_tokens(vocab)` ("inconsistent vocabulary"). Practical rule: always pass the **same `-c`** as the base run and only compare quants of the **same model** (same vocab). `n_seq` is derived from `n_batch / file_n_ctx` and capped at `llama_n_seq_max(ctx)` (with a warning).

The summary block is suppressed when fewer than 100 scored tokens: `if (kld.count < 100) return; // we do not wish to do statistics on so few values`.

---

## 4. Every statistic `--kl-divergence` prints, and how to read it

Per-chunk progress line (cumulative running values): `chunk  PPL  ln(PPL(Q)/PPL(base))  KL Divergence  Δp RMS  Same top p`.

Final block, in print order (`perplexity.cpp` lines 1912-2005; README bullet list):

**====== Perplexity statistics ======**

| Line | Definition (code) | One-sentence interpretation |
|---|---|---|
| `Mean PPL(Q)` | `exp(mean NLL_Q)` ± `ppl · std(NLL)/√(n−1)` | Perplexity of the quant on the scored tokens; lower is better, only meaningful relative to `PPL(base)` on the same tokens. |
| `Mean PPL(base)` | same from the stored base log-probs | The reference; if this drifts from your standalone base PPL run by more than the uint16 floor, something is wrong with `-c`/vocab. |
| `Cor(ln(PPL(Q)), ln(PPL(base)))` | Pearson correlation of per-token NLLs | How linearly the quant's per-token surprisal tracks the base's; 99.9 %+ for Q8/Q6, drops toward ~90 % for 2-bit (README L3-8B: q8_0 99.96 %, q2_K 89.62 %). |
| `Mean ln(PPL(Q)/PPL(base))` | `mean NLL_Q − mean NLL_base` ± propagated with covariance | Log PPL ratio; "it is 0 if the logit distributions are the same" (README); the most statistically efficient PPL comparison because the covariance term removes shared per-token noise. |
| `Mean PPL(Q)/PPL(base)` | `exp` of the above | PPL ratio; 1.000 = no loss; e.g. 1.0004 for q8_0 vs 1.028 for q4_K_M on L3-8B. |
| `Mean PPL(Q)-PPL(base)` | `ppl_Q − ppl_base` ± propagated | ΔPPL in absolute perplexity units — the number people quote as "+0.03 PPL"; depends on the base PPL so is less portable than the ratio. |

**====== KL divergence statistics ======**

| Line | Definition | Interpretation |
|---|---|---|
| `Mean KLD` | mean over scored tokens of `Σ_i p_base(i)·(ln p_base(i) − ln p_Q(i))`, restricted to base tokens with log-prob > −16; ± = std/√(n−1) | Average information lost (nats/token) by using the quant's next-token distribution instead of the base's; 0 = identical; this is the headline number. |
| `Maximum KLD` | largest per-token KLD | Worst single position; tail behaviour matters for sampling-based use, and very low-bit quants show large maxima even when the mean looks fine. |
| `99.9% / 99.0% / 95.0% / 90.0% KLD` | linear-interpolated percentiles of sorted per-token KLDs | Tail shape; ubergarm's guide and Artefact2's #5263 stats quote `KLD_99` specifically because the mean is dominated by a minority of badly-hit tokens. |
| `Median KLD` | median | "Typical" token; much smaller than the mean (heavy right tail). |
| `10.0% / 5.0% / 1.0% / 0.1% / Minimum KLD` | low percentiles | Should be ≈0; slightly **negative** values are possible because the base is uint16-quantized (README BF16/FP16 table shows Minimum KLD −0.000059) — that is encoding noise, not a bug. |

**====== Token probability statistics ======** — all in % of probability of the *correct* (actual next) token, `Δp = p_Q(correct) − p_base(correct)`:

| Line | Interpretation |
|---|---|
| `Mean Δp` | Average change in the probability assigned to the right token; "Positive values mean the model gets better at prediction, negative values mean it gets worse" (README); typically −0.02 % (q8_0) to −9 % (q2_K) on L3-8B. |
| `Maximum Δp … Minimum Δp` and percentiles (99.9, 99, 95, 90, 75, median, 25, 10, 5, 1, 0.1) | README: "If the percentiles are symmetric then the quantization is essentially just adding noise. If the negative values are significantly larger than the positive values then this indicates that the model is actually becoming worse from the quantization." |
| `RMS Δp` | `sqrt(mean Δp²)` ± `0.5/rms · std(Δp²)/√(n−1)`; "If you were to assume that the quantization simply causes Gaussian noise on the token probabilities then this would be the standard deviation of said noise" (README; origin: discussion #2875 by JohannesGaessler, who proposed RMS_p because it "is sensitive to changes of the probability in the medium range rather than changes to very high or low probabilities"). |
| `Same top p` | `n_same_top / count`, where `imax == imax_base` (argmax of Q equals argmax of base; ± from binomial approximation) | **Top-1 (greedy) agreement rate**, not nucleus-p: the fraction of positions where greedy decoding from the quant would emit the same token as greedy decoding from the base; 97.7 % (q8_0) → 91.9 % (q4_K_M) → 71.1 % (q2_K) on L3-8B. |

Practical notes on the ± values: they assume per-token Gaussian noise and scale as 1/√N. With the 4-GiB protocol (N ≈ 14 K scored tokens) the uncertainties will be roughly √(290K/14K) ≈ 4.5× wider than the README's full-wikitext numbers, so two quants whose `Mean KLD` differ by less than ~2σ are **not** separable by this protocol.

---

## 5. Thresholds — what practitioners actually say (and what they don't)

The commonly repeated tiers "KLD < 0.01 indistinguishable, 0.01-0.05 small, > 0.1 noticeable" are folklore. I could not find them stated in any of the canonical llama.cpp threads:

- **Discussion #2875** ("Root mean square of token probability differences as new quantization quality metric", JohannesGaessler, with ikawrakow): proposes RMS Δp; JG initially declined KLD ("I would prefer a way of comparison that is easily interpretable and the KL divergence I think is comparatively difficult to understand"); ikawrakow notes "PPL and the proposed RMS_p are strongly correlated". **No numeric tiers.**
- **Discussion #4110** (kalomaze, 2023-11-17, "Perplexity / PPL, as a quantization loss benchmark, is inaccurate - KL divergence seem to be a better data point"): the motivating thread; **no numeric tiers**.
- **PR #5076** (ikawrakow, merged 2024-01-22): implementation; the only number is the 4e-6 encoding floor.
- **Discussion #5263** (Artefact2, 2024-02-01, "About imatrix overfitting, and importance of input text"): reports per-dataset `Mean KLD` ≈ 0.27 and `KLD_99` ≈ 3.0-3.4 for one low-bit quant (exact type UNVERIFIED from the fetch) with different calibration texts; conclusion is that small KLD improvements don't necessarily show up on HellaSwag/MMLU. **No tiers.**
- **bartowski** ("Comparing sub 50GB Llama 4 Scout quants (KLD/Top P)", HF blog): "KLD, RMS, and Top P are all relevant regardless of the PPL, simply because they tell you how similarly a quantization performs to the full model weights" but "DON'T read too much into them, it's purely informational"; used "the first 300 chunks of wiki text". **Explicitly refuses to give thresholds.**
- **ikawrakow** (ik_llama.cpp discussion #8): "KL divergence and PPL are closely related, and PPL is more convenient to calculate with llama.cpp, so PPL it is" — he reports *quantization error* as PPL ratio − 1 (e.g. Q6_K 0.65 % on LLaMA-3.1-70B, described as unusually high).
- **ubergarm** ("Quant Cookers Basic Guide", ik_llama.cpp discussion #434, 2025-05-18): full command lines (`llama-perplexity … --ctx-size 512 --ubatch-size 512 -f wiki.test.raw -fa -ngl 99 --seed 1337`; `--kl-divergence-base … -f [corpus]`), notes "output kld base file can be quite large, this case it is ~55GiB" and "you could use Q8_0 as your baseline if necessary" when BF16 can't be run; quotes median and 99th-percentile KLD; no tiers. (His `--layer-similarity` flag is ik_llama.cpp-only; it is not in b10941's `llama-imatrix --help`.)
- **Unsloth** (Dynamic 3.0 docs, updated through 2026-04-20): "using perplexity is incorrect" because "output token values can cancel out"; recommends KLD or harder benchmarks; "the closer the KL Divergence is to 0, the better (ie 0 means identical to the full precision model)"; warns that Wikipedia calibration + Wikipedia evaluation overfits, hence their Calibration_v3/v5 sets. The "flips" argument is on the Dynamic 2.0 blog (HTTP 403 today) — UNVERIFIED wording.

What *can* be cited as scale anchors:

1. **The in-tree README LLaMA-3-8B scoreboard** (CUDA, RTX 4090, revision f364eb6f, full wikitext-2): q8_0 **0.00136**, q6_K **0.00545**, q5_K_M **0.0108**, q4_K_M **0.0313** (0.0282 with imatrix), iq4_XS 0.0363, q3_K_M **0.102**, iq3_XXS 0.184, q2_K **0.445** (0.33 with imatrix), iq1_S 2.2; `Same top p` 97.7 % → 91.9 % → 71.1 % for q8_0 / q4_K_M / q2_K. The README's own comment: "K-quants score better on mean Δp than the legacy quants than e.g. KL divergence would suggest."
2. **Artefact2's gist** (Mistral-7B, 2024-02-27; the table is labelled median KL-D): Q6_K 0.0032, Q5_K_M 0.0043, Q4_K_M 0.0075, IQ4_XS 0.0088, Q3_K_M 0.0171, IQ3_XXS 0.0330, Q2_K 0.0588, IQ2_XXS 0.175, IQ1_S 0.55.
3. **Sam McLeod, "Measuring Model Quantisation Quality with KL Divergence" (2026-04-28)** — a mean-KLD scale for Qwen-class instruct models: "< 1e-4: distributions essentially identical; 1e-3 to 5e-3: very close, typical of well-made 6-bit; 1e-2 to 5e-2: measurably larger drift, typical 4-bit territory; > 1e-1: substantial divergence, sampled outputs likely to differ obviously." Caveat: measured with `mlx-kld`, not llama.cpp, on "WikiText-2's raw test split in 32 chunks of n_ctx=2048" scoring the second half of each chunk.
4. **arXiv 2605.02404 "Statistically-Lossless Quantization of LLMs"**: uses Expected Acceptance Rate ≥ 0.99 as the distribution-lossless target and states "EAR ≥ 0.99 generally implies D_KL ≪ 0.01"; their lossless configs sit at KL ≈ 0.002-0.004. (Not llama.cpp; integer group quantization.)
5. **arXiv 2609.07664 "Accuracy is Not Enough" (Sept 2026)**: llama.cpp-based, 5 models × 4 benchmarks × 6 schemes; deliberately prefers bounded TVD/JSD over KLD because KLD "is unbounded and can be dominated by low-probability mismatches"; headline finding is that accuracy can stay flat while 56 % of probability mass moves.
6. **arXiv 2601.14277 "Which Quantization Should I Use?" (Kurt, 2026-01)**: llama.cpp PPL on wikitext-2 vs benchmarks for Llama-3.1-8B-Instruct; Q3_K_S PPL 7.32→8.96 with GSM8K −9.3 pts, Q5_0 slightly *improved* the benchmark average; "quantization format matters, not just nominal bit-width".

Recommended wording for the blog: treat mean KLD ≲ 0.005 as "Q6/Q8-class, within noise of the base for greedy use" (Same-top-p ≳ 96 %), 0.01-0.05 as "Q4/Q5-class, measurable but usually benchmark-neutral", ≳ 0.1 as "3-bit-and-below-class, expect visible behaviour changes" — **and label it as a rule of thumb anchored on the README table, not as a community standard**.

---

## 6. Why wikitext PPL on Qwen3-0.6B (instruct + thinking) is a *relative* measure

- The README itself: "finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases" and "perplexity is not directly comparable between models".
- `llama-perplexity` feeds **raw text with no chat template** (`common_tokenize(ctx, params.prompt, true)`; `--reasoning`/`--jinja` are not in the PERPLEXITY example set in `arg.cpp`). An instruct/thinking model is scored on continuing Wikipedia prose it was never asked to produce, so the absolute PPL (and even ΔPPL) says nothing about instruction following, math, or thinking quality.
- What it *does* measure well: how much the quant's next-token distribution deviates from the base's on the same text — i.e. **KLD, Δp, Same top p are the metrics, PPL ratio is the sanity check**. Unsloth's overfitting warning applies: if the imatrix was calibrated on Wikipedia-like text, wikitext KLD will flatter it.
- Cheap secondary checks (in order of cost):
  1. **`Same top p` is already top-1 agreement** — it answers "how often would greedy output diverge?" for free.
  2. **In-tree multiple-choice (no generation, thinking irrelevant):** `llama-perplexity -m Q.gguf -f mmlu-validation.bin --multiple-choice [--multiple-choice-tasks 500]`; binaries for ARC/HellaSwag/MMLU/TruthfulQA at `huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp` (files: `mmlu-validation.bin`, `mmlu-test.bin`, `hellaswag-validation.bin`, `arc-*.bin`, `truthful-qa-validation.bin`; last modified 2024-03-11). Also `--hellaswag -f hellaswag_val_full.txt --hellaswag-tasks 400` and `--winogrande`. These use log-likelihood scoring of fixed continuations, so a 0.6B model's absolute score is low but the base-vs-quant delta is meaningful.
  3. **Generation-based via `llama-server`:** `examples/llama-eval/llama-eval.py` (PR #21152) — `--dataset {aime,aime2025,aime2026,gsm8k,gpqa} --n_cases N --grader-type {regex,cli,llm} --temperature/--top-k/--top-p/--min-p --threads --resume`; posts to `/v1/chat/completions`. **No MMLU dataset in it.** For Qwen3 you must control thinking (§8) or budget it, or GSM8K runs will be dominated by long `<think>` traces.
  4. A held-out prompt set with `n_probs`/`logprobs` from `llama-server` for top-k overlap — possible but redundant given (1) and (2).

---

## 7. Recommended reproducible protocol (Qwen3-0.6B, 152 K vocab, 16 GB RAM / 8 GB VRAM)

Model files: `unsloth/Qwen3-0.6B-GGUF` (`Qwen3-0.6B-BF16.gguf` 1.20 GB, quants Q2_K…Q8_0 and UD-* variants, `imatrix_unsloth.dat` 1.15 MB, updated 2025-06-23) or `bartowski/Qwen_Qwen3-0.6B-GGUF` (`Qwen_Qwen3-0.6B-bf16.gguf` 1.51 GB, `Qwen_Qwen3-0.6B.imatrix` 1.15 MB, 2025-04-28). Base = BF16 (the model's native dtype; `config.json` `torch_dtype: bfloat16`, `tie_word_embeddings: true`). If BF16 runs into a backend issue, F16 or Q8_0 as base is the accepted fallback (ubergarm), but then say so.

**Step 0 — fixed settings for every run.** `-c 512 -b 2048 -ub 512 -fa on -ngl 99 --seed 1337 -t 8` (the default `-c 512` is the community convention; `-b 2048` gives 4 parallel chunks; `-fa on` for CUDA). Keep `-c`, `-b`, `--chunks`, the base file, backend and build fixed across quants. Record the `--version` string. (Whether `-t` matters when everything is offloaded is a speed question, not an accuracy one.)

**Step 1 — data.** `wiki.test.raw` from the URL in §2.3; 299 K Qwen3 tokens → 584 chunks @512.

**Step 2 — base logits, budgeted.** `--chunks 55` at `-c 512` → 3.97 GiB, 14,025 scored tokens. (If disk allows, `--chunks 110` ≈ 7.9 GiB doubles the token count and halves the variance; the full run is 42 GiB.) Alternative `-c 2048 --chunks 13` gives the same token budget with longer context — the 0.6B model's PPL will be lower at 2048 but the *ratios* are what you compare; pick one and never mix. Memory (estimate, UNVERIFIED — upper bound; output-buffer sizing code not read, and `batch.logits` is only set for `pos >= n_ctx/2`): the logits output buffer is at most ~`n_batch × n_vocab × 4 B` ≈ 1.2 GB plus the 1.2 GB BF16 weights and a tiny KV cache — fits in 8 GB; host RAM holds one chunk of uint16 log-probs (~74 MiB) plus the whole token stream.

```
llama-perplexity -m Qwen3-0.6B-BF16.gguf -f wiki.test.raw -c 512 -b 2048 -ub 512 -fa on -ngl 99 --seed 1337 --chunks 55 --kl-divergence-base qwen3-0.6b-bf16-c512-ch55.kld
```

**Step 3 — noise-floor control.** Run the BF16 model *against its own file* with `--kl-divergence`. Expect Mean KLD ≈ 1e-5 … 1e-3 and Same top p ≈ 99.7 %+ (README rows). Anything a quant scores below this floor is indistinguishable from the base by construction.

**Step 4 — score each quant.**

```
llama-perplexity -m Qwen3-0.6B-Q4_K_M.gguf -c 512 -b 2048 -ub 512 -fa on -ngl 99 --seed 1337 --kl-divergence-base qwen3-0.6b-bf16-c512-ch55.kld --kl-divergence > q4_k_m.kld.txt
```

Report per quant: `Mean KLD ±`, `99.0% KLD`, `Median KLD`, `Mean PPL(Q)/PPL(base) ±`, `Mean Δp ±`, `RMS Δp`, `Same top p ±`, file size. Rank by Mean KLD; use ± to decide ties.

**Step 5 — optional imatrix of your own** (only if quantizing yourself): `llama-imatrix -m Qwen3-0.6B-BF16.gguf -f calibration.txt -c 512 -ngl 99 --chunks 200 -o qwen3-0.6b.imatrix.gguf` (`--chunks` = max chunks; default all; `--output-frequency 10` saves every 10 chunks; GGUF output by default; `--parse-special` if the calibration text contains `<|im_start|>` you want tokenized as specials). Use a *different* text than the evaluation set (e.g. `wiki.train.raw` is the classic but overfits wikitext KLD; bartowski-style `calibration_datav3` or unsloth's sets are the practitioner choice — UNVERIFIED which exact file bartowski used for this model; the raw int32 `ncall` header field of the first entry reads 137 in his file vs 688 in unsloth's — whether that equals "chunks of 512" for files written by 2025-era builds is UNVERIFIED, since today's `save_imatrix_legacy` semantics may not match the writer that produced them). Then `llama-quantize --imatrix qwen3-0.6b.imatrix.gguf Qwen3-0.6B-BF16.gguf out-Q4_K_M.gguf Q4_K_M`.

**Step 6 — one task check.** `llama-perplexity -m Q.gguf -f mmlu-validation.bin --multiple-choice --multiple-choice-tasks 1000 -c 512 -ngl 99` for base and each quant (thinking never engages; pure log-likelihood).

**Expected runtime (UNVERIFIED — estimated from token counts, no run performed).** Each KLD run decodes 55 × 512 ≈ 28 K tokens of prompt through a 0.6B model (seconds on the RTX 5060; tens of seconds on the CPU), computes 14 K full-vocab softmaxes on 23 worker threads (`std::thread::hardware_concurrency() - 1`), and streams ~4 GB from NVMe: **order of 1-2 minutes per quant on GPU, ~5-10 minutes CPU-only**. The base run adds a 4 GB write. A full 42 GiB run would be NVMe-write-bound (minutes of pure I/O) and eat the free disk. Twelve quants ≈ under an hour on GPU.

---

## 8. Qwen3 thinking-mode caveat

**Default is thinking ON.** The GGUF-embedded template (identical in `models/templates/Qwen-Qwen3-0.6B.jinja`) ends with:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

i.e. the empty think block that disables reasoning is only emitted when `enable_thinking` is *explicitly* false; absent → the model opens `<think>` itself. Qwen's card: "By default, Qwen3 has thinking capabilities enabled"; recommended thinking sampling `temperature=0.6, top_p=0.95, top_k=20, min_p=0` ("DO NOT use greedy decoding"); non-thinking `temperature=0.7, top_p=0.8`. `/think` and `/no_think` soft switches in the prompt are also honoured by the model.

**Server/CLI controls in b10941** (`common/arg.cpp` 3658-3730, `research/tmp/server-help.txt`, `tools/server/README.md`):

| Control | Effect |
|---|---|
| `-rea, --reasoning [on|off|auto]` (env `LLAMA_ARG_REASONING`; default `auto` = "detect from template") | `off` sets `params.default_template_kwargs["enable_thinking"] = "false"` — the supported way to disable thinking globally; `auto` sets nothing, so Qwen3 thinks. |
| `--reasoning-budget N` (env `LLAMA_ARG_THINK_BUDGET`) | "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)"; `--reasoning-budget-message` injects text before the forced end tag. |
| `--reasoning-effort LEVEL` | passes `reasoning_effort` to the template (`minimal`…`max`); Qwen3's template ignores it (verified: zero occurrences of `reasoning_effort` in `Qwen-Qwen3-0.6B.jinja`). |
| `--reasoning-format {auto,none,deepseek,deepseek-legacy}` (env `LLAMA_ARG_THINK`) | parsing/placement of thoughts (`message.reasoning_content`), not on/off. |
| `--chat-template-kwargs '{"enable_thinking": false}'` | still works but prints: "Setting 'enable_thinking' via --chat-template-kwargs is deprecated. Use --reasoning on / --reasoning off instead." |
| Per request (`/v1/chat/completions`) | `chat_template_kwargs: {"enable_thinking": false}`; `reasoning_effort: "none"` ("If `none`, reasoning/thinking is disabled"); `reasoning_control: true` + `POST /v1/chat/completions/control {"action":"reasoning_end"}` to cut thinking live. |
| `--jinja` | default **enabled** in b10941 (`params.use_jinja = true`; `--no-jinja` to disable) — required for template kwargs to take effect. |

Interaction with the accuracy protocol: none of this affects `llama-perplexity` (raw text, no template). It matters only for generation-based checks (`llama-eval.py`, custom prompt sets): decide up front whether you evaluate thinking-on (use Qwen's non-greedy sampling, expect long outputs and high variance) or thinking-off (`-rea off`, closer to a deterministic instruct eval), and keep it identical across quants.

---

## 9. imatrix: format, legacy `.dat`, and what `llama-quantize` accepts

- `tools/imatrix/README.md`: "Recent versions of `llama-imatrix` store data in GGUF format by default. For the legacy format, use an extension other than `.gguf` when saving the output file" (PR #9400); `--output-format {gguf,dat}`; conversions: `llama-imatrix --in-file legacy.dat -o new.gguf` and `llama-imatrix --in-file new.gguf --output-format dat -o legacy.dat`; merging: `--in-file a.gguf --in-file b.gguf -o combined.gguf`; `--show-statistics` for per-tensor Σ(Act²)/entropy/ZD/CosSim (computed on squared activations).
- `--chunks N` = maximum chunks (default −1 = all); `--chunk N`/`--from-chunk` skips the first N; chunk size = `n_ctx / n_parallel` (`imatrix.cpp` line 241; default `-c 512`). Legacy output stores `ncall` = chunk count per entry; GGUF stores `imatrix.chunk_count` and `imatrix.chunk_size`.
- **Legacy `.dat` is read directly by `llama-quantize` in b10941**: `common_imatrix_load()` tries `gguf_init_from_file` and on failure calls `common_imatrix_load_legacy()` (`common/imatrix-loader.cpp` lines 84-90); `quantize.cpp` `load_imatrix()` handles `loaded.is_legacy` ("sums contain (raw/count)*ncall, divide by ncall") and skips the metadata requirement for legacy files. `gguf_init_from_file` may log a GGUF magic/parse error before the fallback (logging behaviour not read — UNVERIFIED); harmless either way. Both `unsloth/Qwen3-0.6B-GGUF/imatrix_unsloth.dat` and `bartowski/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B.imatrix` are legacy (verified magic: `c4 00 00 00 16 00 00 00 "blk.27.ffn_down.weight"` = 196 entries, first name length 22). `convert_legacy_imatrix_to_gguf.py` does **not** exist anywhere in the tree; conversion, if wanted for `--show-statistics`/merging, is via `llama-imatrix --in-file`.

---

## 10. Sources (all fetched 2026-09-13)

- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md (== 4a8993735)
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/perplexity.cpp (== 4a8993735)
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/main.cpp
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/scripts/get-wikitext-2.sh
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/imatrix/README.md ; .../tools/imatrix/imatrix.cpp
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/imatrix-loader.cpp ; .../common/imatrix-loader.h
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/quantize.cpp ; .../tools/quantize/README.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp ; .../common/common.h
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/templates/Qwen-Qwen3-0.6B.jinja
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/llama-eval/README.md ; .../examples/llama-eval/llama-eval.py
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/model-conversion/scripts/utils/perplexity-gen.sh ; perplexity-run.sh
- https://api.github.com/repos/ggml-org/llama.cpp/git/trees/master?recursive=1 ; https://api.github.com/repos/ggml-org/llama.cpp/commits/4a8993735
- https://github.com/ggml-org/llama.cpp/pull/5076 ; /discussions/2875 ; /discussions/4110 ; /discussions/5263
- https://github.com/ikawrakow/ik_llama.cpp/discussions/8 ; /discussions/434
- https://huggingface.co/blog/bartowski/llama4-scout-off
- https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9
- https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/
- https://unsloth.ai/docs/basics/dynamic-3.0-ggufs (Dynamic 2.0 blog/doc returned 403/404)
- https://arxiv.org/html/2605.02404 ; https://arxiv.org/html/2609.07664 ; https://arxiv.org/html/2601.14277v1 ; https://arxiv.org/abs/2505.15353
- https://huggingface.co/api/models/unsloth/Qwen3-0.6B-GGUF ; .../bartowski/Qwen_Qwen3-0.6B-GGUF ; .../Qwen/Qwen3-0.6B-GGUF ; Qwen/Qwen3-0.6B config.json + generation_config.json + tokenizer.json
- https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp ; https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
- https://huggingface.co/JohannesGaessler/llama.cpp_wikitext_logits (existence only — people do share base `.kld` files)
- Local: `C:/PK/Github-Projects/ggml-inference-lab/research/tmp/{perplexity,imatrix,quantize,server}-help.txt` from b10941 binaries.
