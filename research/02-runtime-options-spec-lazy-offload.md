# llama.cpp runtime options (b10941 / v0.4.0-dev): speculative types, lazy mode, load mode, offload, KV types, batching

Research date: 2026-09-13. Scope: exact semantics of the runtime flags listed in the task, read from primary sources.

## Provenance

- Binaries probed: `C:/PK/Github-Projects/ggml-inference-lab/bin/llama-b10941-bin-win-cpu-x64.zip` and `llama-b10941-bin-win-cuda-13.3-x64.zip` (+ `cudart-llama-bin-win-cuda-13.3-x64.zip`), unzipped to `C:/PK/Github-Projects/ggml-inference-lab/research/tmp/{cpu,cuda}`. `llama-server.exe --version` reports `version: 0.4.0-dev (build 10941, commit 4a8993735)`, `built with Clang 20.1.8 for Windows x86_64`. Only `--help`, `--version`, `--list-devices` and argument-parse probes were run; no model was loaded and no benchmark was run.
- Help captures: `research/tmp/server-help.txt`, `completion-help.txt`, `cli-help.txt`, `bench-help.txt`, `fit-help.txt`, `llama-help.txt`.
- Sources: fetched from `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/<path>` on 2026-09-13 AND from the commit-pinned URL `https://raw.githubusercontent.com/ggml-org/llama.cpp/4a8993735/<path>`. `cmp` showed the following files are byte-identical between master and 4a8993735: `common/arg.cpp`, `common/common.h`, `common/speculative.cpp`, `common/fit.cpp`, `src/llama-model-loader.cpp`, `src/llama-mmap.cpp`, `src/llama-context.cpp`, `src/llama-model.cpp`, `include/llama.h`, `docs/speculative.md`, `src/models/gemma3n.cpp`, `src/models/gemma4.cpp`, `src/models/qwen4exp.cpp`. So every `file:line` below applies to both the shipped binary's commit and today's master. Line numbers are from the fetched copies in `research/tmp/src/`.
- PR/release pages were read through WebFetch (a summarizer). Numbers quoted from them are labelled "per PR page via WebFetch, not independently verified".

---

## 0. Which binary supports what (probed)

| Binary | `--spec-type` etc. | `--load-mode` / `--lazy-mode` | `-fit` | `-ncffn` | notes |
|---|---|---|---|---|---|
| `llama-server.exe` | yes (full "speculative params" section) | yes | yes (on by default) | yes | also `-lcs/--lookup-cache-static`, `-lcd` for `ngram-cache` (server-help.txt:413) |
| `llama-cli.exe` (new TUI in `tools/cli`) | yes (identical speculative section, `cli-help.txt:305-408`) | yes | yes | yes | `--spec-default` present (cli-help.txt:555) |
| `llama-completion.exe` (old `main`, `tools/completion`) | **NO**. `grep -c "speculative params" completion-help.txt` = 0; probe `llama-completion.exe --spec-type ngram-mod --version` -> `error: invalid argument: --spec-type` | yes (`completion-help.txt:93-111`) | yes | yes | only the draft-model KV-type flags `-ctkd/-ctvd` appear (they live in "common params") |
| `llama-bench.exe` | none | `-lm <auto|none|mmap|mlock|mmap+mlock|dio>`, `-lzm <on|auto|off>` | `-fitt` (default **off**), `-fitc` | **absent** (`-ncmoe` only) | `-ot` separator is `;` not `,`; `-t` default 16; `-nkvo <0|1>`, `-nopo <0|1>` |
| `llama-fit-params.exe` | n/a | yes | yes + `-fitp/--fit-print` | yes | prints estimated memory |
| `llama.exe` | new unified launcher: `llama serve`, `llama cli`, `llama download`, ... (`llama-help.txt`) | | | | |

Removed-flag probes on `llama-server.exe` (CPU build):

```
--draft 8              -> error while handling argument "--draft": the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max
--spec-ngram-size-n 4  -> the argument has been removed. use the respective --spec-ngram-*-size-n
--no-mmap / --mmap / --mlock / --direct-io / --no-direct-io -> error: invalid argument: <flag>   (hard removed, no hint)
--tensor-read-lazy on  -> error: invalid argument: --tensor-read-lazy   (renamed to --lazy-mode/-lzm, release notes cite #27969)
--spec-draft-conf-min  -> error: invalid argument   (mentioned in docs/speculative.md DSpark section but not in common/arg.cpp; docs drift)
```

Accepted spellings probed OK: `-lzm on`, `-lm dio`, `-lm mmap+mlock`, `-ngl auto`, `-ngl all`, `-ncffn 4`, `-ncmoe 4`, `-ot blk.*=CPU`, `--spec-type ngram-mod`, `--spec-type ngram-simple,draft-simple`, `-fa on`, `-c 0`, `--no-host`, `-nr`, `--backend-sampling`.

---

## 1. `--spec-type` and the speculative-decoding subsystem

### 1.1 Value list and defaults

`common/arg.cpp:4236-4244`: `--spec-type` is a comma-separated list; each call **appends** to `params.speculative.types` (`insert(end, ...)`), default `{none}`. Env: `LLAMA_ARG_SPEC_TYPE`. Valid names (`common/speculative.cpp:33-45`): `none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache`.

Auto-selection when `--spec-type` is omitted (`common/arg.cpp:544-571`): if a draft repo/file is given, the type is inferred from sidecar files (mtp > dspark > dflash > eagle3) or from the draft GGUF metadata. `common_speculative_types_from_gguf` (`common/speculative.cpp:2290-2325`) only returns `draft-mtp` (if `blk.<last>.nextn.eh_proj.weight` exists), `draft-dflash`/`draft-dspark` (arch `dflash`, +`markov_w1.weight`), otherwise `{}`. **A vanilla small draft GGUF given with `-md` and no `--spec-type` therefore leaves `types = {none}`**; the server still loads the draft model (`has_dft()`), but `common_speculative_init` builds zero implementations and returns `nullptr` ("no implementations specified for speculative decoding", `speculative.cpp:2724-2727`), so no drafting happens. No code path in `common/arg.cpp`, `common/common.cpp`, `common/speculative.cpp` or `tools/server/server-context.cpp` pushes `DRAFT_SIMPLE` implicitly (grep). Upstream's llama.app CLI docs say the same ("`--spec-type` defaults to none ... you must explicitly specify which type", https://llama.app/docs/cli). Always pass `--spec-type draft-simple` with `-md`. (Observation: the `--fim-qwen-7b-spec` preset at `arg.cpp:4605-4617` sets the draft repo but not a type; not run.)

`--spec-default` (`common/arg.cpp:4709-4724`, server+cli only) = `types.push_back(NGRAM_MOD)` with `n_match=24, n_min=48, n_max=64` (a commented-out k4v config is left as a TODO).

### 1.2 Parameter structs (`common/common.h:324-385`)

| struct | fields (defaults) | flags |
|---|---|---|
| `common_params_speculative_draft` | `n_max=3`, `n_min=0`, `p_split=0.1`, `p_min=0.0`, `backend_sampling=true`, `n_gpu_layers=-1`, `cache_type_k/v=F16` | `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-split`, `--spec-draft-p-min`, `--[no-]spec-draft-backend-sampling`, `--spec-draft-ngl/-ngld`, `-ctkd/-ctvd`, `-md/--spec-draft-model`, `-hfd`, `-devd`, `-otd`, `-cmoed`, `-ncmoed`, `-td`, `-tbd`, ... |
| `common_params_speculative_ngram_mod` | `n_match=24`, `n_max=64`, `n_min=48` | `--spec-ngram-mod-n-match`, `--spec-ngram-mod-n-max`, `--spec-ngram-mod-n-min` (arg.cpp:4246-4274; n-min/n-max range 0..1024, n-match 1..1024) |
| `common_params_speculative_ngram_map` (x3: `ngram_simple`, `ngram_map_k`, `ngram_map_k4v`) | `size_n=12`, `size_m=48`, `min_hits=1` | `--spec-ngram-{simple,map-k,map-k4v}-size-n`, `...-size-m`, `...-min-hits` (arg.cpp:4277-4365) |
| `common_params_speculative_ngram_cache` | `lookup_cache_static`, `lookup_cache_dynamic` paths | `-lcs`, `-lcd` (server example params) |

Binary help confirms the defaults: `server-help.txt:342-346` (`n-max` 3, `n-min` 0), `:377-396` (mod 48/64/24; simple/map-k/k4v 12/48/1).

### 1.3 What each type does (from the implementations)

**Common dispatch** (`common/speculative.cpp:2619-2648`, `common_speculative_init`): the enabled types are instantiated in a **fixed priority order regardless of the order on the command line**:

```
ngram-simple -> ngram-map-k -> ngram-map-k4v -> ngram-mod -> ngram-cache -> draft-simple -> draft-eagle3 -> draft-mtp -> draft-dflash -> draft-dspark
```

The draft-* types after `draft-simple` are only added when a draft context exists. `common_speculative_draft` (`speculative.cpp:2807-2880`) calls each impl in that order; the **first impl that returns a non-empty draft for a sequence wins** for that sequence (`dp.drafting=false`), and later impls are skipped. `docs/speculative.md` states the same: "If a draft model is combined with a draftless decoding the draftless decoding has higher precedence." Acceptance feedback goes to the winning impl with `is_other=false` and to all others with `is_other=true` (`speculative.cpp:2892-2926`).

**Draft length bound.** `common_speculative_n_max(const common_params_speculative*)` (`speculative.cpp:2335-2368`) returns `draft.n_max` for draft-* types, `size_m` for ngram-simple/map-k/map-k4v, `ngram_mod.n_max` for ngram-mod, and a hard-coded `8` for ngram-cache (`create_state_ngram_cache`, `speculative.cpp:2206-2218`, "TODO get from config?"). Consequently **`--spec-draft-n-max` does not bound the n-gram types**; each n-gram type is bounded by its own `size-m` / `mod-n-max`. The per-sequence `dp.n_max` truncation in `common_speculative_draft` comes from the server's `slot.get_n_draft_max()` (`tools/server/server-context.cpp:483-502`: `n_ctx - prompt.n_tokens() - 2`, and `n_remaining - 1`), i.e. remaining context room, not from the flag. The `docs/speculative.md` examples `--spec-type ngram-simple --spec-draft-n-max 64` are therefore misleading on this commit (the `-n-max 64` is a no-op for ngram-simple). Not run; conclusion is from code reading.

**`draft-simple`** (`speculative.cpp:179-390`): requires `ctx_dft` (a second, vocab-compatible model; compatibility check `speculative.cpp:67-131`: same vocab type, same BOS/EOS handling, vocab size difference <= 128, token texts identical from id 5). Drafting loop: decode `id_last` on the draft model, then repeatedly sample with a fixed top-k=10 sampler (`speculative.cpp:227-236`), stop when `p(top1) < p_min` (`--spec-draft-p-min`, default 0 = never stop early), or when `n_max` (`--spec-draft-n-max`, default 3) or the per-seq context bound is reached; drafts shorter than `n_min` are discarded. The draft model uses `-ngld` (default auto), `-devd`, `-ctkd/-ctvd`, `-otd`, `-td/-tbd` etc. Requires loading a second model; with `--fit on` the draft model's memory is fitted alongside the main model (`common/common.cpp:1295-1326`).

**`ngram-simple`** (`speculative.cpp:1770-1812`, `common/ngram-map.cpp:49-113`): no draft model. On every draft call it takes the last `size_n` tokens (n-1 from history + the just-sampled token), scans the whole token history **backwards, linearly** for the most recent earlier occurrence of that n-gram, and proposes up to `size_m` tokens that followed it (needs at least `size_n + size_m + 1` tokens of history and at least `size_n` copyable tokens after the match). `process()` is a no-op ("TODO: implement"). **`--spec-ngram-simple-min-hits` is parsed but never used**: `common_ngram_simple_config` has only `size_ngram/size_mgram` (`ngram-map.h:24-27`; `speculative.cpp:2682-2690`). Cost is O(history) per generated token.

**`ngram-map-k`** (`speculative.cpp:1815-1866`, `ngram-map.cpp:221-390`): same idea but keyed through a 2^18-entry hash map from n-gram hash to history index (`COMMON_NGRAM_HASH_MAP_SIZE 262144`, `ngram-map.h:34`), rebuilt/extended incrementally and re-initialised on `begin()` for a new prompt (needed because reasoning blocks are removed from history, `ngram-map.h:72-73`). In key-only mode it proposes the `m` tokens following the matched key, and **adapts the proposed length to the number of tokens accepted last time this key was used** (`n_draft_tokens = min(m, values[0].n_accepted)`, `ngram-map.cpp:375-390`; `n_accepted` starts at `m` and is updated by `common_ngram_map_accept`, `ngram-map.cpp:518-536`). **On this commit `--spec-ngram-map-k-min-hits` has no effect**: the `key_num < min_hits` check (`ngram-map.cpp:393`) is placed after the `if (map.key_only) {... return;}` block, so it only runs for k4v. (`docs/speculative.md` claims min-hits applies to map-k; the code disagrees.)

**`ngram-map-k4v`** (same impl with `key_only=false`): for a matched key it tracks up to 4 distinct following m-grams with occurrence counts (`COMMON_NGRAM_MAX_VALUES 4`, `ngram-map.h:31`), requires `key_num >= min_hits`, and only drafts when the most frequent value is at least twice as frequent as all other values combined (`if (sum_occur > 0 && max_occur < 2 * sum_occur) return;`, `ngram-map.cpp:480-484`). Documented as "experimental"; the docs' suggested setting for repetitive text is `--spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2`.

**`ngram-mod`** (`speculative.cpp:1869-2042`, `common/ngram-mod.{h,cpp}`, PR #19164 per header comment): a single fixed-size hash table of `4*1024*1024` int32 entries (16 MiB, `speculative.cpp:1895`) **shared across all server slots**. Key = LCG hash of the last `n_match` tokens (`ngram-mod.cpp:14-22`), value = the single next token. Drafting (`draft_one`, `speculative.cpp:1953-1996`): starting from the current n-gram, repeatedly look up the next token and append it (rolling), up to `n_max` tokens; if a lookup misses before `n_min` tokens the whole draft is dropped, otherwise the draft is truncated at the miss. New n-grams are inserted in chunks of 32 (`if (sinfo.i_last + 32 < cur_len)`). Self-protection: table reset if occupancy > 25 % at `begin()` (`f_thold = 0.25`) or after 5 consecutive drafts with acceptance fraction < 0.25 (`accept()`, `speculative.cpp:2010-2040`). `n_match < 16` logs a warning "too small - poor quality is possible". Variable-length drafts (m not fixed). `docs/speculative.md`: "MoEs require long drafts", "dense models: can reduce --spec-ngram-mod-n-min and --spec-ngram-mod-n-max"; applications listed: iterating over a block of text/code, reasoning models repeating their thinking, summarization.

**`ngram-cache`** (`speculative.cpp:2044-2180`): the old "lookup decoding" (#5479/#6828/#6848 per docs): three n-gram statistics caches (context, dynamic, static) with probabilistic draft; draft length hard-coded to 8; `-lcs/-lcd` load static/dynamic cache files (server example-specific params, `server-help.txt:413`). Saving is disabled (`save_static=save_dynamic=false`, `speculative.cpp:2211-2213`).

**When each wins (documented guidance + code facts; no measurements taken):**
- `draft-simple`: wins when a small vocab-compatible draft model exists and its per-token cost is much lower than the target's (docs: "most used approach"). On this machine (8 GB VRAM, single-channel DDR5) it costs VRAM for a second model and KV cache, so it competes with `-ngl`.
- `ngram-*`: cost ~zero, win only when output repeats text already in context (code rewriting, summarising, reasoning-then-answer). `docs/speculative.md` positions `ngram-simple` as "simplest ... minimal overhead", `ngram-map-k` as the hashed version, `ngram-mod` as the shared-pool variant that "can generate variable draft lengths". `--spec-default` picks `ngram-mod` (24/48/64), i.e. upstream's current recommended default. General-chat speed-up from any n-gram mode is UNVERIFIED (expected near zero when nothing repeats).
- Multiple types can be combined (`--spec-type ngram-mod,ngram-map-k4v`); the n-gram impls always get first try, the draft model only runs if all n-gram impls produced nothing for that sequence.

### 1.4 Note on release-note item "n-gram history lookup (#28040)"

PR #28040 ("kv-cells: resolve get_prev_tokens in O(log n) from the sequence position index", per PR page via WebFetch) is **not** about `--spec-type ngram-*`. It optimises `llama_kv_cache::get_prev_tokens` (`src/llama-kv-cache.cpp:1836-1887`) / `llama_kv_cells::seq_pos_tok_le` (`src/llama-kv-cells.h:324-325`, "used by n-gram input embeddings to recover the tokens preceding a ubatch"), which feeds the **PLE n-gram hash embeddings** of Qwen3.8-Flash-Next (`src/models/qwen4exp.cpp:64-140`). Reported gain (per PR page): 73.8 -> 76.7 t/s at 55k ctx, 50.4 -> 56.4 t/s at 132k ctx on an RTX PRO 6000. Irrelevant to speculative decoding and to non-PLE models.

---

## 2. `--lazy-mode` / `-lzm` (on-demand tensor reading)

- Flag: `-lzm, --lazy-mode MODE` with `on | auto | off`, default `auto`, env `LLAMA_ARG_LAZY_MODE` (`common/arg.cpp:2705-2717`; `common/common.h:486`). API enum `llama_lazy_mode` (`include/llama.h:217-221`): `OFF=0` "always read the whole tensor up front", `AUTO=1` "lazy only for marked tensors larger than 4 GiB (requires mmap)", `ON=2` "read the rows of tensors marked by the arch on demand (requires mmap)". Default in `llama_model_default_params` is `LLAMA_LAZY_MODE_AUTO` (`src/llama-model.cpp:2767`).
- Mechanism (`src/llama-model-loader.cpp:1081-1109`, `lazy_read::add`): only tensors created with the `TENSOR_READ_LAZY` flag (`llama-model-loader.h:72`, "read rows on demand instead of loading whole tensor; requires mmap for now") are candidates. In `auto`, tensors with `ggml_nbytes(t) <= 4 GiB` are skipped with the comment "do not lazy-read small tensors, it has significant overhead and is not worth it" (`auto_min_size = 4ull*1024*1024*1024`, line 1086-1090). If `llama_mmap::SUPPORTED` is false the tensor is loaded fully with a warning. Lazy tensors always live in the plain CPU buffer type (`lazy_read::buft()`, line 1073-1079: "lazy tensors are gathered on the host, so no offload setting applies to them"), in a dedicated ggml context (`ctx_key{buft, is_lazy}`), and their file ranges are excluded from prefetch (`llama-mmap.cpp:599-618`, `ranges_complement(lazy_ranges, ...)`) and from mlock (`loader.cpp:1653-1657`: "locking a lazy tensor would fault all of it in, which is what lazy avoids").
- Interaction with `--load-mode`: `init_mappings` maps the file whenever `use_mmap || lazy.any()` (`loader.cpp:1403-1405`, comment: "read_lazy also requires mmap; this condition make sure it's usable even when --load-mode is not set to mmap"), so lazy works even with `-lm none`/`dio`; the `auto` lazy mode is resolved to `off` when any selected device lacks `mmap_support` (`src/llama-model.cpp:1429-1437`, "e.g. iGPUs", #28160).
- **Which tensors are flagged** (grep of all files fetched from `src/models/` on master, 154 files): `TENSOR_READ_LAZY` appears **only** in `src/models/gemma4.cpp:58` (`per_layer_tok_embd`, i.e. `per_layer_token_embd.weight` of Gemma 4 E-series when `n_embd_per_layer > 0`) and `src/models/qwen4exp.cpp:187-188` (Qwen3.8-Flash-Next PLE n-gram hash embedding table). **Gemma 3n is NOT flagged**: `src/models/gemma3n.cpp:41` creates `per_layer_tok_embd` with flags `0`. This contradicts the PR #27794 description (which lists Gemma-3N as a target, per PR page via WebFetch) and the task prompt's example; on this commit `--lazy-mode` does nothing for Gemma 3n.
- Why `auto` uses 4 GiB: the loader comment above (overhead), plus PR #27794's reported cost (per PR page via WebFetch, not independently verified): Gemma-4 Q5_K prefill 514-563 t/s and generation 94.2-97.3 t/s, "-8 % to -10.7 % vs baseline", peak RSS 6.16 GB vs 7.37 GB; "small models like gemma 4 ... will have a significant impact", "bigger models like qwen4 ... the effect will be minor". The PR page names the flag `--tensor-read-lazy`; the binary only accepts `--lazy-mode` (renamed in #27969 per v0.4.0 release notes).
- Practical implication for a 16 GB single-channel machine: `-lzm on` trades RAM residency for page faults on the NVMe during every token (rows are fetched on demand); it only matters for Gemma 4 E-series / Qwen3.8-Flash-Next GGUFs. Leave at `auto` otherwise.

---

## 3. `--load-mode` / `-lm` (replaces `--mmap/--no-mmap`, `--mlock`, `--direct-io`)

- Flag (`common/arg.cpp:2686-2704`): `-lm, --load-mode MODE`, values `auto | none | mmap | mlock | mmap+mlock | dio`, default `auto`, env `LLAMA_ARG_LOAD_MODE`. Enum `llama_load_mode` (`include/llama.h:205-212`): `AUTO=-1`, `NONE=0`, `MMAP=1`, `MLOCK=2`, `MMAP_MLOCK=3`, `DIRECT_IO=4`; string mapping in `src/llama.cpp:68-75`.
- Loader mapping (`src/llama-model-loader.cpp:559-560`): `use_mmap = (MMAP || MMAP_MLOCK || AUTO)`, `use_direct_io = (DIRECT_IO)`. `mlock` alone therefore means **no mmap + VirtualLock the malloc'd host buffers**; `mmap+mlock` means mmap + lock the mapping. `auto` = mmap unless a selected device reports `!caps.mmap_support` (`src/llama-model.cpp:1417-1425`); the CUDA backend reports `mmap_support = (type != GGML_BACKEND_DEVICE_TYPE_IGPU)` (`ggml/src/ggml-cuda/ggml-cuda.cu:5035`), so on the discrete RTX 5060 `auto` resolves to `mmap`; the log line prints the resolved mode: `loading model tensors, this can take a while... (load_mode = mmap|none|...)` (line 1444).
- Old spellings are hard-removed: `--no-mmap`, `--mmap`, `--mlock`, `--direct-io`, `--no-direct-io` -> `error: invalid argument` (probed). The v0.4.0 release notes (via WebFetch) do not mention the rename; the only in-tree trace is the help text.
- **Windows support** (`src/llama-mmap.cpp`):
  - mmap: `CreateFileMappingA` + `MapViewOfFile` (lines 573-597); prefetch uses `PrefetchVirtualMemory` (Win8+) over the non-lazy ranges (599-618). `llama_mmap::SUPPORTED = true` on `_WIN32` (676).
  - mlock: `VirtualLock` with automatic working-set growth via `SetProcessWorkingSetSize` (729-760). `llama_mlock::SUPPORTED = true` on `_WIN32` (819).
  - **Direct I/O is Linux-only.** `O_DIRECT` open is under `#ifdef __linux__` (185-215); the Win32 `llama_file::impl` constructor takes `use_direct_io` as `[[maybe_unused]]` and always opens with `ggml_fopen` + `ReadFile` in 64 MiB chunks (87-96, 130-146). No "falling back" warning is emitted on Windows (the Linux path logs `Failed to open file ... Falling back to buffered I/O`). The Win32 `has_direct_io()` returns `true` (174-176) but has no callers outside `llama-mmap.cpp`; `read_alignment()` returns the shared default `alignment = 1` (388-392). Net effect of `-lm dio` on Windows: identical to `-lm none` (no mmap, buffered reads).
- Loading path when mmap is off (`none`, `mlock`, `dio`): CPU-resident tensors are `read_raw` into malloc'd buffers; GPU tensors go through the **async pinned-memory upload** path (`upload_backend`, `loader.cpp:1519-1600`), which is only enabled when `!use_mmap && !check_tensors` and the device has `async + host_buffer + events` (CUDA does). With mmap on, CUDA tensors are copied from the mapping with `ggml_backend_tensor_set` (1660) and CPU tensors are used in place from the mapping (`ggml_backend_dev_buffer_from_host_ptr`, `src/llama-model.cpp:1746-1765`).
- **mmap + CPU overrides warning** (`loader.cpp:1235-1241`), emitted once when any `-ot`/`-ncffn`/`-ncmoe` override targets CPU while mmap is on: `"llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance"`. Mechanism not stated in the code comment; the observable difference is that with mmap the CPU-side tensors stay in the file mapping's page cache (and cannot be repacked, see 4.2), whereas `-lm none` copies them into backend-allocated host buffers (repack-eligible) and uses the pinned async path for the GPU part.
- `mmap` help text: "if mmap disabled, slower load but may reduce pageouts if not using mlock". On a 16 GB machine with a model that fits, `auto` (mmap) is the default; `mlock`/`mmap+mlock` prevent Windows from paging the weights out under memory pressure but require the working set to grow.

---

## 4. Offload controls

### 4.1 `-ngl` / `--gpu-layers` / `--n-gpu-layers`

`common/arg.cpp:2784-2801`: value is `auto` (-1), `all` (-2) or an integer; default `auto`; env `LLAMA_ARG_N_GPU_LAYERS`. At the model level both `-1` and `-2` mean "all": `llama_model::n_gpu_layers()` returns `params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1` (`src/llama-model.cpp:1897-1900`; the +1 is the output layer). The difference is only for `--fit`: fit refuses to touch layer placement unless `n_gpu_layers` still equals the library default `-1` (`common/fit.cpp:463-465`, "n_gpu_layers already set by user ... abort"), so `-ngl all` = "put everything on the GPU and do not let fit reduce it", `-ngl auto` = "let fit decide".

### 4.2 `-ot` / `--override-tensor`, buffer-type names, `-ncffn`, `-ncmoe`, `-cmoe`

- Syntax (`common/arg.cpp:252-283`, `parse_tensor_buffer_overrides`): `<regex>=<buffer type>[,<regex>=<buffer type>...]`, split on `,`; the pattern is a `std::regex` applied with `regex_search` (substring match, ECMAScript syntax) against the GGUF tensor name (`loader.cpp:1231-1232`); first matching override wins. Env `LLAMA_ARG_OVERRIDE_TENSOR`. Multiple `-ot` flags accumulate. In `llama-bench` the separator is `;`.
- **Accepted buffer-type names on this CUDA build** (probed: `llama-server.exe -ot blk.0.ffn_up=BOGUS --version` in `research/tmp/cuda` prints `Available buffer types: CPU, CUDA0`). Only each device's *default* buffer type is enumerated (`arg.cpp:255-262`), so `CUDA_Host` and `CPU_REPACK` are **not** valid `-ot` targets (`-ot blk.0.ffn_up=CUDA_Host` -> `unknown buffer type`, probed). `--list-devices` on the CUDA build: `CUDA0: NVIDIA GeForce RTX 5060 Laptop GPU (8123 MiB, 7043 MiB free)`.
- What `=CPU` really selects (`loader.cpp:1231-1245`): when the override target is the CPU buffer type, the loader re-runs `select_weight_buft` over the CPU buffer-type list, which `make_cpu_buft_list` (`src/llama-model.cpp:1034-1094`) orders as `[ACCEL bufts] -> [host buffer of the first GPU device, e.g. CUDA_Host, unless --no-host] -> [CPU extra bufts, e.g. CPU_REPACK, unless --no-repack/-nr] -> [CPU]`, first-supported wins (`select_weight_buft`, `loader.cpp:1060-1071`; support is tested by `ggml_backend_dev_supports_op` on a dummy tensor, 920-1058). CUDA's `supports_op` only rejects sources that live on a *different* CUDA device (`ggml/src/ggml-cuda/ggml-cuda.cu:5056-5067`), so on this machine CPU-overridden matmul weights are expected to land in **`CUDA_Host` (pinned host memory) rather than `CPU_REPACK`**, i.e. no AVX2 repacking for them. `--no-host` ("bypass host buffer allowing extra buffers to be used", `arg.cpp:2426-2432`, env `LLAMA_ARG_NO_HOST`) removes the host buffer from the list so `CPU_REPACK` becomes eligible. The trade-off is spelled out in the `make_cpu_buft_list` comment (`llama-model.cpp:1048-1053`): the host buffer is added because "storing the tensors in a host buffer is useful when the processing of large batches is offloaded to a GPU device, since it reduces the time spent on data transfers". So for CPU-resident `-ot`/`-ncffn` weights: `CUDA_Host` (default) favours prefill (cheaper op-offload copies, 4.4), `--no-host` favours single-token decode (AVX2 `CPU_REPACK` matmuls). Inferred from source, not measured. Verify at runtime from the `load_tensors: CUDA_Host model buffer size = ...` / `CPU_REPACK model buffer size` log lines. Inferred from source, not run.
- AVX2 repack coverage (`ggml/src/ggml-cpu/repack.cpp:4573-4720`, `ggml_repack_get_optimal_repack_type`): on a CPU with AVX2 (this i7-14650HX; no AVX-512) repacking exists for `Q4_0` (8x8), `Q4_K` (8x8), `IQ4_NL` (8x8) and `MXFP4` (8x8). `Q2_K` needs AVX-512; `Q5_K`, `Q6_K`, `Q8_0` repack only on NEON/RISC-V. Repacked matmuls require the activation to be F32 and host-resident (`repack.cpp:4780-4795`). Repacking also needs the weights in a backend-allocated buffer (not the mmap page cache), which is the second reason for the "consider --load-mode none" warning.
- `-ncffn N` / `--n-cpu-ffn N` (`arg.cpp:2772-2783`, env `LLAMA_ARG_N_CPU_FFN`; PR #26622 "llama : add --n-cpu-ffn option"): appends N overrides `blk\.<i>\.ffn_(up|down|gate)\.` = CPU for `i = 0..N-1` (`common/common.h:1133`, `1136-1138`, `1144-1152`). Targets the dense FFN projections of the **first** N layers; attention and norms of those layers stay on the GPU. PR-page numbers (via WebFetch, not independently verified): Qwen 3.8-27B IQ3_XXS on RTX 4060 Ti 16 GB, `-ngl 52`: 550 t/s pp, 6.36 t/s tg; `-ncffn 32`: 553 t/s pp, 7.66 t/s tg ("+20 %" tg at equal VRAM). The unchanged pp is explained by op-offload (4.4).
- `-ncmoe N` / `--n-cpu-moe N` (`arg.cpp:2762-2771`): same but with `\.ffn_(up|down|gate|gate_up)_(ch|)exps` (`common.h:1131`), i.e. expert tensors of the first N layers. `-cmoe`/`--cpu-moe` (`arg.cpp:2755-2761`) adds one override `\.ffn_(up|down|gate|gate_up)_(ch|)exps` = CPU for all layers. The draft-model equivalents are `-otd`, `-cmoed`, `-ncmoed` (`arg.cpp:4103-4125`); there is no `-ncffnd`.
- Any of `-ot`, `-ncffn`, `-ncmoe`, `-cmoe` makes `--fit` abort its layer-placement step (4.3).

### 4.3 `-fit` / `--fit-target` / `--fit-ctx` (defaults ON)

- Flags (`arg.cpp:2864-2921`): `-fit, --fit [on|off]` default **on** (`common.h:476`), env `LLAMA_ARG_FIT`; `-fitt, --fit-target MiB0,MiB1,...` per-device margin to leave free, default 1024 MiB (`common.h:481`, single value broadcast), env `LLAMA_ARG_FIT_TARGET`; `-fitc, --fit-ctx N` minimum context fit may shrink to, default 4096 (`common.h:478`), env `LLAMA_ARG_FIT_CTX`; `-fitp/--fit-print` only in `llama-fit-params`. `llama-bench`: `-fitt` default off, `-fitc` 4096.
- Where it runs: `common_init_result` (`common/common.cpp:1289-1330`) calls `common_fit_params` **before** `llama_model_load_from_file`, passing the draft/MTP model as an "extra model" that is fitted together. Contract (`common/fit.h:24-28`): "only parameters that have the same value as in llama_default_model_params are modified with the exception of the context size which is modified if and only if equal to 0". Memory is measured with a `no_alloc` dry-run load (`common_get_device_memory_data`).
- Algorithm (`common/fit.cpp`, `common_params_fit_impl`):
  1. If `-c` is 0 (default), context is first set to `n_ctx_train * n_streams` (`fit.cpp:270-272`), i.e. **the model's full trained context**, then memory is measured. If the projected free memory on the device is >= the margin, return with no changes (300-371).
  2. Step 2 (373-457): only if context is still auto (`-c 0`) and the deficit is global, shrink `n_ctx` by linear interpolation between full and `--fit-ctx`, never below `--fit-ctx`. A user-set `-c` is never changed ("context size set by user ... no change").
  3. Abort conditions (461-486, thrown as `common_params_fit_exception`, caught at 879-898 -> `LOG_WRN("failed to fit params to free device memory: ...")`, status FAILURE, **the model still loads with whatever was already changed, including a step-2 context reduction**): `n_gpu_layers != -1` ("n_gpu_layers already set by user to N, abort"); any user `-ot`/`-ncffn`/`-ncmoe` ("model_params::tensor_buft_overrides already set by user, abort"); user `-ts` or `-sm row` on multi-GPU.
  4. Step 3 (488-733): fill devices **back to front** with whole "dense" layers (for MoE: all expert tensors kept in system RAM), using false-position search on measured memory; the last layer may be partial via generated overrides of the form `blk\.<i>\.ffn_(gate|up|gate_up|down).*`, `blk\.<i>\.ffn_(gate|gate_up|down).*`, `blk\.<i>\.ffn_down.*`, or `blk\.<i>\.ffn_(up|down|gate_up|gate)_(ch|)exps` (492-543). Step 4 (735-866): for MoE models, convert dense-only layers to full layers front-to-back while they fit.
- What fit **changes**: `n_ctx` (only if `-c 0`), `n_gpu_layers`, `tensor_split` (multi-GPU), and `tensor_buft_overrides` (auto-generated `-ot` patterns for partial layers / CPU experts). What it **never** touches: `-b`, `-ub`, `-ctk/-ctv`, `-fa`, `-nkvo`, `-t`, `--load-mode`, `--lazy-mode`.
- On this machine: `--list-devices` reported 7043 MiB free of 8123 MiB; with the default 1024 MiB margin fit aims to leave >= 1 GiB free, i.e. roughly 6 GB for weights + KV + compute. `--fit-target 512` etc. trades margin for offload. Note fit measures *free* memory at start-up, so anything else using the GPU shrinks what it will offload.

### 4.4 `-nkvo`, `--no-op-offload`, `--no-repack`, `-sm`

- `-kvo/--kv-offload`, `-nkvo/--no-kv-offload` (`arg.cpp:2411-2417`, env `LLAMA_ARG_KV_OFFLOAD`; default enabled): sets `llama_context_params.offload_kqv` ("offload the KQV ops (including the KV cache) to GPU", `include/llama.h:400`). With `-nkvo` the KV cache and the KQV ops are placed on the host (`llama.h:400`); exact scheduling of attention under op-offload was not traced; a comment in `llama-context.cpp:530-531` notes the fused-op probe "is still wrong for cases like --no-kv-offload". `llama-bench` spells it `-nkvo <0|1>`.
- `--op-offload` / `--no-op-offload` (`arg.cpp:2943-2949`; `llama.h:402` "offload host tensor operations to device"; default true): controls `ggml_backend_sched` "1.off" scheduling (`ggml/src/ggml-backend.cpp:966-975`): when an op's weight lives in a host buffer on the lowest-priority (CPU) backend, a higher-priority backend may claim the op if `supports_op && offload_op`. CUDA's `offload_op` returns `get_op_batch_size(op) >= 32` (`ggml-cuda.cu:5521-5540`; `MUL_MAT` batch size = `ne[1]` = tokens; overridable via env `GGML_OP_OFFLOAD_MIN_BATCH`, `ggml-cuda.cu:5710`). Consequence: CPU-resident weights (`-ot`/`-ncffn`/`-ncmoe`) are copied to the GPU and multiplied there during prompt processing (ubatch >= 32 tokens), but computed on the CPU for single-token decode. This is why `-ncffn` barely changes pp t/s in PR #26622. `-ub` below 32 defeats it. `llama-bench` spells it `-nopo <0|1>`.
- `--repack` / `-nr, --no-repack` (`arg.cpp:2418-2425`, env `LLAMA_ARG_REPACK`, default enabled): `use_extra_bufts`; disables `CPU_REPACK`. `--no-host` (`arg.cpp:2426-2432`) removes the GPU host buffer type from the CPU list (see 4.2).
- `-sm, --split-mode {none,layer,row,tensor}` (`arg.cpp:2803-2823`, env `LLAMA_ARG_SPLIT_MODE`, default `layer`): `none` = one GPU (`-mg`), `layer` = layers/KV split across GPUs (pipelined), `row` = weights split by rows, `tensor` = weights and KV split ("EXPERIMENTAL"; forces flash attention on, `llama-context.cpp:3685-3697`, and warns when used with a single device). Irrelevant with one GPU; `layer` is fine. `-ts` and `-mg` also multi-GPU only.

---

## 5. KV cache types `-ctk` / `-ctv` and flash attention

- Flags (`arg.cpp` around line 2433; help `server-help.txt:74-82`): `-ctk, --cache-type-k TYPE`, `-ctv, --cache-type-v TYPE`, allowed `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` (`arg.cpp:304-314`, `kv_cache_types`), default `f16`, env `LLAMA_ARG_CACHE_TYPE_K/V`. Draft model: `-ctkd/-ctvd` (`--spec-draft-type-k/-v`). `-fa, --flash-attn [on|off|auto]` default `auto`, env `LLAMA_ARG_FLASH_ATTN`.
- Rules in `llama_init_from_model` (`src/llama-context.cpp:3680-3733`):
  - **Quantized V requires flash attention.** `auto` -> automatically enabled with log `enabling flash_attn since it is required for quantized V cache`; `off` -> error `quantized V cache requires flash_attn to be enabled` and context creation fails. Quantized K alone (`-ctk q8_0 -ctv f16`) does not require FA (also enforced in `llama-context.cpp:464-465`).
  - MLA models and DeepSeek-V4 require `type_k == type_v`.
  - With FA on, the K and V block sizes must divide `n_embd_head_k/v` (3715-3733); `q8_0/q4_0` have block size 32, so head dims that are not multiples of 32 fail.
  - `-fa auto` is resolved by a device probe (`resolve(llm_fused_op_flash_attn_probe, ...)`, 555-557): FA is disabled if the FLASH_ATTN_EXT op of any layer would be scheduled on a different device than the layer (i.e. unsupported by the layer's backend).
- **CUDA support for quantized KV with FA** (`ggml/src/ggml-cuda/fattn.cu:501-515`): vector-kernel instances exist for `f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0`; `iq4_nl` is **not** in that list although `-ctk iq4_nl` is accepted by the arg parser -> UNVERIFIED behaviour on CUDA (expect FA to be unsupported/fallback to CPU; avoid). Only combinations compiled in are fast: build option `GGML_CUDA_FA_QUANTS` defaults to `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16` (`ggml/CMakeLists.txt:207`; `docs/build.md:303`: "Combinations that were not compiled fall back to f16-f16 kernel with a warning"). The release workflow's `windows-cuda` job does not override this (`.github/workflows/release.yml:1012-1023`, only `-DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_CUDA=ON -DLLAMA_BUILD_BORINGSSL=ON`), so in the prebuilt CUDA DLL: `-ctk q8_0 -ctv q8_0` and `-ctk q4_0 -ctv q4_0` have native FA kernels; mixed pairs such as `-ctk q8_0 -ctv q4_0` fall back to the f16 kernel with a warning. (The workflow file was read from master, not from the b10941 tag: mark as "as of master's release.yml".) The CUDA 13.3 build compiles `120a-real` (`ggml/src/ggml-cuda/CMakeLists.txt:40-52`), i.e. native Blackwell SM 12.0 code for the RTX 5060; on Ada/Blackwell (`cc >= ADA_LOVELACE`) the quantized-KV vector kernel is used for batches of <= 2 query tokens, else the MMA F16 kernel (`fattn.cu:612-632`).
- Memory: KV bytes per token = 2 * n_layer * n_head_kv * head_dim * bytes(type) (+ block scales); `q8_0` halves f16, `q4_0` quarters it. Not measured here.

---

## 6. `-b` / `-ub` (batch and micro-batch) and prompt processing

- Flags: `-b, --batch-size N` "logical maximum batch size" default **2048**, env `LLAMA_ARG_BATCH`; `-ub, --ubatch-size N` "physical maximum batch size" default **512**, env `LLAMA_ARG_UBATCH` (`server-help.txt:30-33`; `llama.h:361-362`; library defaults `llama-context.cpp:3621-3622`).
- Clamping (`src/llama-context.cpp:245-247`): `n_batch = min(n_ctx, -b)` for causal models; `n_ubatch = min(n_batch, -ub)` (`-ub 0` -> `n_batch`). The graph/compute buffers are reserved for `min(n_ctx, n_ubatch)` tokens (`llama-context.cpp:596, 837`), so `-ub` sets the size of the CUDA compute buffer and the width of every prompt-processing matmul; `-b` only sets how many tokens one `llama_decode` call may contain (the server chunks prompts into `n_batch`). The n-ubatch is the unit split by the memory module (`memory->init_batch(*balloc, cparams.n_ubatch, ...)`, 1752).
- Effects relevant here: larger `-ub` = fewer, larger GEMMs during prefill (better GPU utilisation, larger compute buffer in VRAM); the CPU op-offload threshold of 32 tokens (4.4) means `-ub < 32` forces CPU-resident weights to be computed on the CPU during prefill. Encoder/non-causal models require `n_ubatch >= n_tokens` (1437, 1726). Pipeline parallelism (n_batch > n_ubatch across multiple GPUs) is multi-GPU only. No numbers measured here.

---

## 7. Renamed / removed / deprecated flag summary (b10941)

| Old | Status on b10941 | New |
|---|---|---|
| `--mmap`, `--no-mmap` | hard removed: `error: invalid argument` | `--load-mode mmap` / `--load-mode none` (`-lm`) |
| `--mlock` | hard removed | `--load-mode mlock` or `mmap+mlock` |
| `--direct-io`, `--no-direct-io` | hard removed | `--load-mode dio` (Linux-only effect; no-op equivalent to `none` on Windows) |
| `--tensor-read-lazy` (PR #27794 spelling) | hard removed / never shipped | `--lazy-mode` / `-lzm` (#27969 per release notes) |
| `--draft`, `--draft-n`, `--draft-max` | soft removed with hint | `--spec-draft-n-max` (draft models) or `--spec-ngram-mod-n-max` |
| `--draft-min`, `--draft-n-min` | soft removed with hint | `--spec-draft-n-min` / `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | soft removed with hint | per-type `--spec-ngram-{simple,map-k,map-k4v}-size-n/-size-m/-min-hits`, `--spec-ngram-mod-n-match` |
| `-dt, --defrag-thold` | still accepted, marked `(DEPRECATED)` in help | none |
| `--draft-p-split`, `--draft-p-min` | still accepted as aliases | `--spec-draft-p-split`, `--spec-draft-p-min` |
| `-md/--model-draft`, `-ngld`, `-td`, `-tbd`, `-devd`, `-otd`, `-ctkd/-ctvd`, `-cmoed`, `-ncmoed` | still accepted as aliases | `--spec-draft-*` spellings |

---

## 8. Consolidated implications for this machine (i7-14650HX, 16 GB single-channel DDR5-5600, RTX 5060 Laptop 8 GB, Windows 11)

1. Fully-offloaded models (weights + KV + compute <= ~6 GB with default `--fit-target 1024`): leave `-ngl auto -fit on`; fit will first try the model's full trained context and shrink it toward 4096 only if needed; set `-c` explicitly to keep fit from touching context.
2. Partially-offloaded dense models: `-ncffn N` keeps attention on the GPU and moves only the FFN of the first N layers to the CPU (PR #26622 reports +20 % tg vs `-ngl` at the same VRAM, unverified). Use `-lm none` to silence and act on the loader warning, and consider `--no-host` so CPU-side Q4_0/Q4_K/IQ4_NL weights get AVX2-repacked instead of sitting in pinned `CUDA_Host` memory (faster tg, possibly slower pp; inferred, not measured); check the `load_tensors: ... model buffer size` lines to confirm placement. Any `-ncffn`/`-ot` disables fit's layer placement, so also give an explicit `-ngl` (or accept fit's `FAILURE` warning).
3. Prompt processing with CPU-resident weights still runs on the GPU when `-ub >= 32` (op-offload); keep `-ub` at 512 (default) or higher if VRAM allows.
4. KV: `-ctk q8_0 -ctv q8_0` (or `q4_0/q4_0`) with `-fa auto/on` are the compiled-in fast combinations in the prebuilt CUDA DLL; mixed K/V quant types fall back to f16 kernels; `iq4_nl` KV on CUDA is unverified. Quantised V alone forces FA on.
5. Speculative decoding: only `llama-server`/`llama-cli` support `--spec-type` (not `llama-completion`). `--spec-default` = `ngram-mod` 24/48/64 and costs 16 MiB RAM, nothing on the GPU; it only helps when output repeats context. `--spec-draft-n-max` does not limit n-gram drafts. Draft-model speculation on 8 GB VRAM must fit a second model + KV; fit accounts for it.
6. `--lazy-mode` is a no-op for everything except Gemma 4 E-series and Qwen3.8-Flash-Next (only `gemma4.cpp` and `qwen4exp.cpp` flag tensors), and in `auto` only for a PLE tensor > 4 GiB.
7. `--load-mode dio` does nothing different from `none` on Windows (direct I/O is `#ifdef __linux__`).

---

## 9. UNVERIFIED / not run

- All throughput numbers quoted from PR #27794, #26622, #28040 pages came through WebFetch summaries.
- Runtime confirmation that `-md` without `--spec-type` produces no drafts (conclusion from code only; not run).
- Runtime placement of `-ot ...=CPU` tensors in `CUDA_Host` vs `CPU_REPACK` on this machine (inferred from `make_cpu_buft_list` order and CUDA `supports_op`; confirm with load logs).
- `iq4_nl` KV cache behaviour with CUDA flash attention (not in `ggml_cuda_fattn_kv_type_supported`).
- The release workflow flags were read from master's `.github/workflows/release.yml`, not the b10941 tag.
- `tools/server/README.md` `/slots` example JSON still shows `speculative.n_max/n_min/p_min` keys (lines 891-893); no parser for those keys was found in `tools/server/server-task.cpp` (grep) -> possibly stale doc, per-request speculative overrides UNVERIFIED.
- `docs/speculative.md` mentions `--spec-draft-conf-min` (DSpark), which the binary rejects.
