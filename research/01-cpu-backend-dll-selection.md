# How llama.cpp's prebuilt Windows build picks a `ggml-cpu-*.dll` variant

Research note for the "max token-generation speed on an i7-14650HX + RTX 5060 Laptop" blog.
Target release: **llama.cpp `b10941`** (Windows zips `llama-b10941-bin-win-{cpu,cuda-13.3,vulkan}-x64.zip`).
Every source claim below is cited to a file fetched at tag `b10941` (line numbers are for that tag). Five of them (`ggml-backend-reg.cpp`, both CPU CMakeLists, `common.cpp`, `arch/x86/cpu-feats.cpp`) were diffed against `master` on 2026-09-13 and are byte-identical; the others were only fetched at the tag.

Base URL for all source cites: `https://github.com/ggml-org/llama.cpp/blob/b10941/`

Machine facts verified locally with WMI (`Win32_Processor`, `Win32_PhysicalMemory`): `Intel(R) Core(TM) i7-14650HX`, NumberOfCores = 16, NumberOfLogicalProcessors = 24; one 16 GiB DIMM in `Controller0-ChannelA-DIMM0` at 5600 MT/s (single channel). Logical-CPU layout verified with `GetSystemCpuSetInformation` (EfficiencyClass per LP): **LP 0-15 = P-cores (EfficiencyClass 1), SMT siblings paired as (0,1), (2,3), ..., (14,15); LP 16-23 = E-cores (EfficiencyClass 0), one LP per core.** Intel ARK confirms 8 P-cores + 8 E-cores, 24 threads, "Instruction Set Extensions: Intel SSE4.1, SSE4.2, AVX2", "Intel DL Boost on CPU: Yes" ([ARK 235996](https://www.intel.com/content/www/us/en/products/sku/235996/intel-core-i7-processor-14650hx-30m-cache-up-to-5-20-ghz/specifications.html)). Wikipedia's Raptor Lake page lists the ISA as "AVX2, FMA3, AVX-VNNI, SHA, ..." ([Raptor Lake](https://en.wikipedia.org/wiki/Raptor_Lake)), and its AVX-512 page states Intel does not support AVX-512 on Alder Lake and later hybrid client parts ([AVX-512](https://en.wikipedia.org/wiki/AVX-512)).

---

## TL;DR

| Question | Answer |
|---|---|
| Which DLL does an i7-14650HX load? | **`ggml-cpu-alderlake.dll`** (score 128). Verified empirically on this machine with `llama-bench.exe --list-devices`. |
| Why? | It is the highest-scoring variant whose *every* required CPUID bit is present. All AVX-512 variants score 0 because `AVX512F` is absent; `alderlake` beats `haswell` (64) because AVX-VNNI (CPUID.7.1:EAX[4]) is present. |
| Can I force a variant by leaving only one DLL? | **Yes.** The loader scans the exe directory and the cwd for `ggml-cpu-*.dll`, scores each, and loads the best. With one DLL present that scores > 0 it is loaded. Verified. If the lone DLL scores 0 you get **no CPU backend at all** (silently in release builds). Verified with `ggml-cpu-zen4.dll`. |
| Is there an env var to force a variant? | **No.** `GGML_BACKEND_PATH` loads *one extra* backend by file path *after* auto-selection; it does not replace the auto-picked CPU DLL. Verified: both `alderlake` and `haswell` end up registered, and llama uses the first CPU device found (the auto-picked one). `GGML_BACKEND_DIR` is compile-time only. No CLI flag exists. |
| What does AVX-VNNI buy vs plain AVX2? | `vpdpbusd` replaces `vpmaddubsw`+`vpmaddwd` in the shared 8-bit dot helpers. That helper is used by **Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4** kernels (vec_dot and the 8x8 repack GEMV/GEMM). **Q4_K, Q5_K, Q6_K, Q2_K, Q3_K kernels never call it** on x86; they use raw `maddubs`/`madd` with per-sub-block scales, so `alderlake` vs `haswell` is expected to be a wash for K-quants. |
| Default `-t` on Windows hybrid CPUs? | `common_cpu_get_num_math()` on Windows just returns the physical-core count from `GetLogicalProcessorInformationEx(RelationProcessorCore)` = **16 on the 14650HX (8P + 8E, E-cores included)**. The E-core exclusion code exists but is compiled only for `x86_64 && __linux__`. |
| `--poll` | **Ignored** by the shipped Windows build, which is an OpenMP build (`libomp.dll`). The polling loop is compiled only when `GGML_USE_OPENMP` is *not* defined. Use `KMP_BLOCKTIME` / `OMP_WAIT_POLICY` instead. |
| `-C/--cpu-mask`, `--cpu-strict`, `--prio` | All three **do** work under OpenMP: applied per thread inside the `omp parallel` region on every graph compute. |

---

## 1. The loader: `ggml_backend_load_best()`

File: [`ggml/src/ggml-backend-reg.cpp`](https://github.com/ggml-org/llama.cpp/blob/b10941/ggml/src/ggml-backend-reg.cpp)

### 1.1 Entry point

The unconditional call every tool hits is in `llama_backend_init()` (`src/llama.cpp` L132-134: `if (!ggml_backend_reg_count()) ggml_backend_load_all();`), which `llama-completion`, `llama-server`, etc. call from `main()`. `common/arg.cpp` additionally calls `ggml_backend_load_all()` from the option handlers that need devices early (`-ot` L253, `--device` L1125, `--list-devices` L1142, `--rpc` L1168), and `llama-bench` calls it from its own `main()` as well (which is why `llama-bench --list-devices` prints every `loaded ... backend` line twice — the registry dedupes, the log does not). Nothing calls `ggml_backend_load_all_from_path(dir)`; a string grep of the shipped `llama.dll`, `llama-common.dll` and `llama-bench-impl.dll` finds only `ggml_backend_load_all`. `ggml_backend_load_all()` (L574-576) calls `ggml_backend_load_all_from_path(nullptr)` (L578-605), which does, in order:

```cpp
ggml_backend_load_best("blas", ...); "zendnn"; "cann"; "cuda"; "hip"; "metal"; "rpc"; "sycl";
"vulkan"; "virtgpu"; "opencl"; "hexagon"; "musa"; "openvino";
ggml_backend_load_best("cpu", silent, dir_path);                 // L599
const char * backend_path = std::getenv("GGML_BACKEND_PATH");    // L601
if (backend_path) { ggml_backend_load(backend_path); }           // L603
```

`silent` is `true` in `NDEBUG` (release) builds (L579-583), so failures to load a candidate are *not* printed in the shipped binaries.

### 1.2 Search paths (L486-502)

When `user_search_path == nullptr` (always, for the llama tools):

1. `GGML_BACKEND_DIR` — only if defined at compile time (`ggml/src/CMakeLists.txt` L368-372; it is an empty cache var by default in `ggml/CMakeLists.txt` L87, so **not defined in the release**).
2. The **executable's directory** (`get_executable_path()`, Windows branch L446-458 via `GetModuleFileNameW`).
3. The **current working directory** (`fs::current_path()`, L493-499).

Both 2 and 3 are scanned and the best score across *both* wins. A stray `ggml-cpu-*.dll` in your shell's cwd participates in the contest.

### 1.3 Candidate matching and scoring (L504-553)

For every regular file in each search path whose filename starts with `ggml-cpu-` (`backend_filename_prefix()` = `ggml-` on Windows, L464-470, plus name plus `-`, L483) and whose extension is `.dll` (L472-478):

```cpp
dl_handle_ptr handle { dl_load_library(entry) };                              // L529  (LoadLibraryW, full path)
auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");  // L534
if (score_fn) {
    int s = score_fn();                                                        // L536
    if (s > best_score) { best_score = s; best_path = entry.path(); }          // L540-543
}
```

Notes:
- Every candidate is **actually `LoadLibrary`'d** to call its exported `ggml_backend_score()`, then freed (the `dl_handle_ptr` deleter calls `FreeLibrary`, `ggml-backend-dl.h` L22-26). The DLL's imports (`ggml-base.dll`, `libomp.dll` — confirmed by grepping the binary) must therefore be resolvable, or the candidate silently drops out.
- `dl_load_library` on Windows = `LoadLibraryW(path)` wrapped in `SetErrorMode(SEM_FAILCRITICALERRORS)` (`ggml-backend-dl.cpp` L5-15) so a missing-dependency dialog is suppressed.
- Strict `>`: the first file enumerated wins a tie, but no two shipped variants share a score (see table in §2.4).
- The per-candidate `score:` debug line (L537-539) is compiled only in non-`NDEBUG` builds; the release prints nothing about scoring.

### 1.4 Fallback and final load (L555-571)

If `best_score == 0` (nothing supported), it looks for the **untagged** `ggml-cpu.dll` in each search path (L558-561) and loads that. **The Windows release does not ship `ggml-cpu.dll`** (verified: the zip has only `ggml-cpu-<variant>.dll` files), so in that case there is no CPU backend, and `llama-model.cpp` throws `"no CPU backend found"` (L1069, L1458, L1736 of `src/llama-model.cpp`).

Otherwise `load_backend(best_path)` (L220-264) re-loads the DLL, re-checks `ggml_backend_score() != 0` (L229-235), calls `ggml_backend_init()`, checks `GGML_BACKEND_API_VERSION` (L246), and logs at **INFO** level — unconditionally, even in release:

```
load_backend: loaded CPU backend from C:\...\ggml-cpu-alderlake.dll      // L259
```

That line is the practical way to see which variant you got.

### 1.5 Registration and the "two CPU backends" trap (L186-218)

`register_backend()` dedupes only by `reg` pointer identity (L191-195). Two *different* CPU DLLs produce two different `reg` pointers, both named `"CPU"`, and both get registered with their devices appended in order. `ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)` (L355-363) returns the **first** CPU device, i.e. the one from `load_best("cpu")`, which ran *before* `GGML_BACKEND_PATH`. `src/llama-model.cpp` uses exactly that call to pick the CPU device (L1067, L1456, L1734) and `src/llama-context.cpp` L353 uses `ggml_backend_init_by_type(CPU)` which goes through the same lookup.

---

## 2. How each variant is built

### 2.1 The variant list

File: [`ggml/src/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/b10941/ggml/src/CMakeLists.txt) L487-520 (x86 branch of `GGML_CPU_ALL_VARIANTS`):

```cmake
ggml_add_cpu_backend_variant(x64)
ggml_add_cpu_backend_variant(sse42              SSE42)
ggml_add_cpu_backend_variant(sandybridge        SSE42 AVX)
if (NOT MSVC)
    ggml_add_cpu_backend_variant(ivybridge      SSE42 AVX F16C)
    ggml_add_cpu_backend_variant(piledriver     SSE42 AVX F16C FMA)
endif()
ggml_add_cpu_backend_variant(haswell            SSE42 AVX F16C FMA AVX2 BMI2)
ggml_add_cpu_backend_variant(skylakex           SSE42 AVX F16C FMA AVX2 BMI2 AVX512)
ggml_add_cpu_backend_variant(cannonlake         SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI)
ggml_add_cpu_backend_variant(cascadelake        SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI)
ggml_add_cpu_backend_variant(icelake            SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI)
if (NOT MSVC)
    ggml_add_cpu_backend_variant(cooperlake     SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI AVX512_BF16)
    ggml_add_cpu_backend_variant(zen4           SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
endif()
ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
if (NOT MSVC)
    ggml_add_cpu_backend_variant(sapphirerapids SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
endif()
```

That is **14 x86 variants** with a non-MSVC compiler, 9 with MSVC. The task brief said 15; the shipped zip contains exactly 14 `ggml-cpu-*.dll` files (listed in `bin/cpu`). `ggml_add_cpu_backend_variant()` (L441-483) resets every `GGML_<feat>` to OFF and turns on only the listed ones; `GGML_CPU_ALL_VARIANTS` requires `GGML_BACKEND_DL` (L488-489).

The release workflow [`.github/workflows/release.yml`](https://github.com/ggml-org/llama.cpp/blob/b10941/.github/workflows/release.yml) `windows-cpu` job (L649-712) builds with

```
-D CMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=ON -DGGML_OPENMP_FETCH=ON  + CMAKE_ARGS
```

where `cmake/x64-windows-llvm.cmake` sets `CMAKE_C_COMPILER clang` / `CMAKE_CXX_COMPILER clang++` (the GNU-style driver, so CMake's `MSVC` is false and all 14 variants build — consistent with the zip). `CMAKE_ARGS` (L33) is `-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON` — nothing that alters CPU codegen. The CUDA/Vulkan/HIP/SYCL zips get the same `ggml-cpu-*.dll` set injected from the CPU zip in the "Merge artifacts" step (L1605-1626), which is why `bin/cuda` and `bin/vulkan` contain identical CPU DLLs.

### 2.2 Compiler flags per feature (clang/GCC path)

File: [`ggml/src/ggml-cpu/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/b10941/ggml/src/ggml-cpu/CMakeLists.txt) L307-371 (the `else()` of `if (MSVC)`):

| CMake feature | Compiler flag(s) | Define passed to the score object |
|---|---|---|
| `SSE42` | `-msse4.2` | `GGML_SSE42` |
| `F16C` | `-mf16c` | `GGML_F16C` |
| `FMA` | `-mfma` | `GGML_FMA` |
| `BMI2` | `-mbmi2` | `GGML_BMI2` |
| `AVX` | `-mavx` | `GGML_AVX` |
| `AVX2` | `-mavx2` | `GGML_AVX2` |
| `AVX_VNNI` | `-mavxvnni` | `GGML_AVX_VNNI` |
| `AVX512` | `-mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw` | `GGML_AVX512` |
| `AVX512_VBMI` | `-mavx512vbmi` | `GGML_AVX512_VBMI` |
| `AVX512_VNNI` | `-mavx512vnni` | `GGML_AVX512_VNNI` |
| `AVX512_BF16` | `-mavx512bf16` | `GGML_AVX512_BF16` |
| `AMX_TILE` | `-mamx-tile` | `GGML_AMX_TILE` |
| `AMX_INT8` | `-mamx-int8` | `GGML_AMX_INT8` |
| `AMX_BF16` | `-mamx-bf16` (not used by any shipped variant) | `GGML_AMX_BF16` |

There is **no `-march=`** for variants; each is baseline x86-64 plus exactly these `-m` flags. The `x64` variant gets no flags at all (x86-64 baseline = SSE2), so it does not even reach the `__SSSE3__` SIMD helpers in `arch/x86/quants.c` L28 — it runs generic C kernels. No shipped variant passes `-mavxvnniint8`, so the `__AVXVNNIINT8__` (`vpdpbssd`) paths in the kernels are dead code in every release DLL.

### 2.3 The score object

`ggml_add_cpu_backend_features()` (`ggml-cpu/CMakeLists.txt` L1-18) compiles `ggml-cpu/arch/x86/cpu-feats.cpp` as a separate OBJECT library **without** the arch flags, with only the `GGML_<feat>` defines, and with `-fno-lto` (L15) so no AVX-512 instruction can be inlined into the score function and SIGILL before the check runs. (`cpu-feats-x86.cpp` at the old path is gone — 404 — it is now `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`.)

File: [`ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`](https://github.com/ggml-org/llama.cpp/blob/b10941/ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp) L263-323:

```cpp
static int ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support
    int score = 1;
    cpuid_x86 is;
#ifdef GGML_FMA          if (!is.FMA())        { return 0; } score += 1;      #endif
#ifdef GGML_F16C         if (!is.F16C())       { return 0; } score += 1<<1;   #endif
#ifdef GGML_SSE42        if (!is.SSE42())      { return 0; } score += 1<<2;   #endif
#ifdef GGML_BMI2         if (!is.BMI2())       { return 0; } score += 1<<3;   #endif
#ifdef GGML_AVX          if (!is.AVX())        { return 0; } score += 1<<4;   #endif
#ifdef GGML_AVX2         if (!is.AVX2())       { return 0; } score += 1<<5;   #endif
#ifdef GGML_AVX_VNNI     if (!is.AVX_VNNI())   { return 0; } score += 1<<6;   #endif
#ifdef GGML_AVX512       if (!F && !CD && !VL && !DQ && !BW) { return 0; } score += 1<<7; #endif
#ifdef GGML_AVX512_VBMI  if (!is.AVX512_VBMI()){ return 0; } score += 1<<8;   #endif
#ifdef GGML_AVX512_BF16  if (!is.AVX512_BF16()){ return 0; } score += 1<<9;   #endif
#ifdef GGML_AVX512_VNNI  if (!is.AVX512_VNNI()){ return 0; } score += 1<<10;  #endif
#ifdef GGML_AMX_INT8     if (!is.AMX_INT8())   { return 0; } score += 1<<11;  #endif
    return score;
}
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)   // exports "ggml_backend_score"
```

(Condensed; the AVX512 block checks each of F/CD/VL/DQ/BW individually, L298-302.) The CPUID bits used (L17-88): `AVX_VNNI` = leaf 7 subleaf 1 EAX[4] (L83), `AVX512_VNNI` = leaf 7 ECX[11] (L80), `AVX512_BF16` = leaf 7.1 EAX[5] (L82), `AMX_INT8` = leaf 7 EDX[25] (L86). `AMX_TILE` has no score weight — it is only a compile flag. The score is a pure "all required bits present, else 0" gate; it never checks XGETBV/OS state (the FIXME), so on a machine where the OS has disabled AVX-512 state an AVX-512 variant would still be picked (not relevant on Windows 11 with this CPU).

The DLL's `ggml_cpu_has_*()` functions (`ggml-cpu/ggml-cpu.c` L3643-3713) are **compile-time** `#if defined(__AVXVNNI__)` etc., and the backend's feature list (`ggml-cpu/ggml-cpu.cpp` L534-645) is built from them. So the `system_info` line printed at startup (`AVX_VNNI = 1`, `AVX512 = 0`, `OPENMP = 1`, `REPACK = 1`, ...) describes **the DLL that was chosen**, not the CPU. `AVX_VNNI = 1` in the log is the fingerprint of `alderlake`.

### 2.4 Score table, and what the i7-14650HX sees

Weights: base 1, FMA 1, F16C 2, SSE42 4, BMI2 8, AVX 16, AVX2 32, AVX_VNNI 64, AVX512 128, VBMI 256, BF16 512, AVX512_VNNI 1024, AMX_INT8 2048.

| Variant | Requires (score bits) | Max score | On i7-14650HX |
|---|---|---|---|
| `x64` | nothing | 1 | 1 |
| `sse42` | SSE4.2 | 5 | 5 |
| `sandybridge` | + AVX | 21 | 21 |
| `ivybridge` | + F16C | 23 | 23 |
| `piledriver` | + FMA | 24 | 24 |
| `haswell` | + AVX2, BMI2 | 64 | 64 |
| **`alderlake`** | haswell + AVX-VNNI | **128** | **128 — chosen** |
| `skylakex` | haswell + AVX512 F/CD/VL/DQ/BW | 192 | 0 (no AVX512F) |
| `cannonlake` | skylakex + VBMI | 448 | 0 |
| `cascadelake` | skylakex + AVX512_VNNI | 1216 | 0 |
| `icelake` | skylakex + VBMI + AVX512_VNNI | 1472 | 0 |
| `cooperlake` | skylakex + AVX512_VNNI + BF16 | 1728 | 0 |
| `zen4` | skylakex + VBMI + AVX512_VNNI + BF16 | 1984 | 0 |
| `sapphirerapids` | zen4 + AMX_INT8 (AMX_TILE compiled in, unscored) | 4032 | 0 |

Because Raptor Lake exposes AVX2 + FMA + F16C + BMI2 + AVX-VNNI and no AVX-512 (Gracemont E-cores and Raptor Cove P-cores advertise the same ISA, so CPUID gives the same answer whichever core the loader thread lands on), the winner is `alderlake` with 128.

**Empirical confirmation on this machine** (copy of the CPU-only zip in `research/tmp/cpu-full`, no model loaded):

```
> llama-bench.exe --list-devices
load_backend: loaded RPC backend from ...\cpu-full\ggml-rpc.dll
load_backend: loaded CPU backend from ...\cpu-full\ggml-cpu-alderlake.dll
```

---

## 3. Forcing a variant

### 3.1 Leave only one DLL (works)

Verified in `research/tmp/cpu-haswell-only` (exes + `llama*.dll` + `ggml.dll` + `ggml-base.dll` + `libomp.dll` + only `ggml-cpu-haswell.dll`):

```
load_backend: loaded CPU backend from ...\cpu-haswell-only\ggml-cpu-haswell.dll
```

Same for `ggml-cpu-x64.dll` alone. This is the **only** supported way to A/B `haswell` vs `alderlake` on this CPU. Keep `ggml-base.dll` and `libomp.dll` — every variant imports both.

Caveats:
- The chosen file must still score > 0. With only `ggml-cpu-zen4.dll` present the run printed **no** `loaded CPU backend` line at all (silent in release), and a model load would throw `no CPU backend found`.
- The search also covers the cwd (§1.2). Run from the binary directory or make sure no other `ggml-cpu-*.dll` is in your shell's cwd.
- Renaming a variant to the untagged `ggml-cpu.dll` also works, but only via the fallback path (§1.4), i.e. only when no `ggml-cpu-*.dll` scores > 0.

### 3.2 `GGML_BACKEND_PATH` (does not force; adds a second CPU backend)

Verified in `cpu-full` with `GGML_BACKEND_PATH=<full path>\ggml-cpu-haswell.dll`:

```
load_backend: loaded CPU backend from ...\ggml-cpu-alderlake.dll
load_backend: loaded CPU backend from ...\ggml-cpu-haswell.dll
```

Both are registered; per §1.5 llama uses the first CPU device, i.e. `alderlake`. So `GGML_BACKEND_PATH` is for out-of-tree backends (L600 comment), not for variant selection. If the DLL lives outside the exe directory, its own imports (`ggml-base.dll`, `libomp.dll`) are resolved by the normal Windows `LoadLibrary` search order (application directory first) — a copy elsewhere still resolves them from the exe dir. UNVERIFIED by test, standard Windows loader behaviour.

### 3.3 Other knobs that do not exist

- `GGML_BACKEND_DIR` is a CMake cache variable baked in at build time (`ggml/src/CMakeLists.txt` L368-372); the release does not set it.
- `ggml_backend_load_all_from_path(dir)` exists as a C API, but no llama tool or `LLAMA_ARG_*` env var calls it (`common/arg.cpp` only calls `ggml_backend_load_all()`).
- `--device` / `LLAMA_ARG_DEVICE` explicitly rejects CPU devices (`common/arg.cpp` L1128-1129, `"invalid device"`).
- `GGML_DISABLE_VULKAN` exists (L131) but only for a statically linked Vulkan; nothing analogous for CPU variants.
- Verified against the shipped binaries, not just source: grepping `ggml.dll`, `ggml-base.dll`, `llama.dll`, `llama-common.dll` and the `*-impl.dll` files for `GGML_BACKEND*` strings finds exactly one environment-variable name, `GGML_BACKEND_PATH` (in `ggml.dll`); the other hits are enum names (`GGML_BACKEND_DEVICE_TYPE_CPU`, `GGML_BACKEND_SPLIT_AXIS_*`).

---

## 4. What each ISA level changes for Q8_0 and Q4_K

### 4.1 The shared 8-bit dot helpers

File: [`ggml/src/ggml-cpu/arch/x86/quants.c`](https://github.com/ggml-org/llama.cpp/blob/b10941/ggml/src/ggml-cpu/arch/x86/quants.c) L105-131:

```cpp
static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);          // AVX512-VNNI (VL form)
#elif defined(__AVXVNNI__)
    summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);      // AVX-VNNI  <-- alderlake DLL
#else
    dot = _mm256_maddubs_epi16(ax, sy); return sum_i16_pairs_float(dot);   // AVX2: vpmaddubsw + vpmaddwd
#endif
}
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__   /* never compiled in any release variant */
    _mm256_dpbssd_epi32(zero, x, y);
#else
    ax = _mm256_sign_epi8(x, x); sy = _mm256_sign_epi8(y, x); return mul_sum_us8_pairs_float(ax, sy);
#endif
}
```

The identical pattern exists for the repack GEMV/GEMM helpers in [`arch/x86/repack.cpp`](https://github.com/ggml-org/llama.cpp/blob/b10941/ggml/src/ggml-cpu/arch/x86/repack.cpp) L148-173 (`mul_sum_us8_pairs_acc_int32x8` / `mul_sum_i8_pairs_acc_int32x8`, with `_mm512_dpbusd_epi32` 512-bit variants at L118-130 for AVX-512 builds) and in llamafile's tinyBLAS `updot()` (`ggml-cpu/llamafile/sgemm.cpp` L1754-1764).

So AVX-VNNI folds `vpmaddubsw` + `vpmaddwd(ones)` (two uops, with 16-bit intermediate saturation) into one `vpdpbusd` with 32-bit accumulation. Note that the vec_dot helper passes a **zero** accumulator and immediately converts to float (`_mm256_cvtepi32_ps`) so that the per-block scale can be applied in FP — only the repack `*_acc_int32x8` helpers (L148-173) actually exploit the accumulate operand of `vpdpbusd`. Expect a smaller gain on the Q8_0 vec_dot than "one instruction replaces two" suggests. Whether that is measurable in token generation (which is memory-bound on this single-channel DDR5 box) is exactly what the blog should test; the code only tells you *where* it can matter.

### 4.2 Which kernels reach those helpers (single-token generation path, n = 1)

| Weight type | Path at n = 1 | Uses VNNI helper? | Cite |
|---|---|---|---|
| **Q8_0** | `ggml_vec_dot_q8_0_q8_0` (AVX2 branch). Not repacked on x86 (repack for Q8_0 is NEON/RVV only). llamafile sgemm refuses n < 2. | **Yes** (`mul_sum_i8_pairs_float`) | quants.c L1325-1342; repack.cpp L4704-4720; sgemm.cpp L3818-3822 |
| Q4_0 | Repacked to `Q4_0x8` (`GGML_CPU_REPACK` ON by default) → `ggml_gemv_q4_0_8x8_q8_0` → `gemv_q4_b32_8x8_q8_0_lut_avx` | **Yes** (8 helper calls) | repack.cpp L4573-4577; arch/x86/repack.cpp L522, L1448-1458 |
| Q4_1, Q5_0, Q5_1 | vec_dot | Yes | quants.c L898 and neighbours |
| IQ4_NL, MXFP4 | Repacked 8x8 → same `_lut_avx` template | Yes | arch/x86/repack.cpp L1692, L1705 |
| **Q4_K** | Repacked to `Q4_Kx8` when `ne[1] % 8 == 0` → `ggml_gemv_q4_K_8x8_q8_K`; otherwise `ggml_vec_dot_q4_K_q8_K` | **No** — 16 raw `_mm256_maddubs_epi16` + `_mm256_madd_epi16(scale, …)` per block, 0 helper calls in the GEMV, 0 in the GEMM | repack.cpp L4605-4609; arch/x86/repack.cpp L1464-1687 (gemv), L2042-3496 (gemm); quants.c L2058-2115 (vec_dot) |
| Q2_K | Repacked only when `ggml_cpu_has_avx512()`; else vec_dot | No | repack.cpp L4632-4636 |
| Q5_K, Q6_K, Q3_K, IQ*_K | vec_dot | No (raw maddubs/madd) | quants.c L2216+, L2426+, L1766+ |

Why K-quants cannot use VNNI: they multiply the 16-bit `maddubs` product by a per-sub-block int16 scale *before* widening to 32 bits (`p16l = _mm256_madd_epi16(scale_l, p16l)`, quants.c L2102), so there is no unscaled u8×s8→s32 dot to hand to `vpdpbusd`.

Prompt processing (n ≥ 2): Q8_0 / Q4_0 / Q5_0 / IQ4_NL / F16 / BF16 / F32 go through llamafile tinyBLAS (`GGML_LLAMAFILE` defaults ON via top-level `CMakeLists.txt` L165-166; sgemm.cpp L4041-4140), whose `updot` is VNNI-aware; repacked Q4_0 / Q4_K use the 8x8 GEMM kernels. The Q4_K 8x8 GEMM has a dedicated `__AVX512F__` 512-bit body (arch/x86/repack.cpp, `_mm512_maddubs_epi16` block inside L2042-3496) — an AVX-512 machine gets a wider Q4_K GEMM, but no AVX-512 variant is loadable on this CPU.

To A/B the repack layer: `-nr` / `--no-repack` (`common/arg.cpp` L2419-2425, env `LLAMA_ARG_REPACK`).

### 4.3 What the lower variants lose

- `haswell` vs `alderlake`: only the helper above (Q8_0/Q4_0/Q4_1/Q5_x/IQ4_NL/MXFP4 dots). Everything else identical.
- `piledriver`/`ivybridge`/`sandybridge` (no AVX2): the `#elif defined(__AVX__)` 128-bit-lane paths (e.g. quants.c L742-765 for Q4_0, L1343+ for Q8_0). Half the vector width, `mul_add_epi8_sse`.
- `sse42`: SSSE3/SSE4 128-bit paths.
- `x64`: generic C (no `__SSSE3__`), F16 conversion via bit-twiddling instead of F16C.

---

## 5. Default thread count on Windows hybrid CPUs

File: [`common/common.cpp`](https://github.com/ggml-org/llama.cpp/blob/b10941/common/common.cpp)

`postprocess_cpu_params()` (L288-310) sets `n_threads = common_cpu_get_num_math()` when `-t` was not given (default `n_threads = -1`, `common/common.h` L69). `common_params_parse` calls it for gen, batch, draft and draft-batch params (`common/arg.cpp` L880-884); batch defaults to a copy of the gen params ("role model").

`common_cpu_get_num_math()` (L202-228):

```cpp
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    ... if (is_hybrid_cpu()) { count cores, skipping E-cores and HT siblings } ...
#elif defined(__powerpc64__) ...
#endif
    return common_cpu_get_num_physical_cores();
```

The hybrid-aware code — `is_hybrid_cpu()` (CPUID.7.0:EDX[15], L168-172), `is_running_on_efficiency_core()` (CPUID.0x1A core-type == 0x20 "Atom", L174-180), `cpu_count_math_cpus()` (pins to each CPU, skips E-cores, `++cpu` to skip HT siblings, L182-195) — is inside `#if defined(__x86_64__) && defined(__linux__)` (L149, L198). **None of it is compiled on Windows.**

On Windows, `common_cpu_get_num_physical_cores()` (L116-143) calls `GetLogicalProcessorInformationEx(RelationProcessorCore, …)` and counts one per `RelationProcessorCore` record (`num_physical_cores += info->Processor.GroupCount`, L136 — GroupCount is 1 per core on a single-group system). E-cores are physical cores, so:

> **i7-14650HX on Windows: default `-t 16`** (8 P + 8 E). On Linux the same CPU would default to 8.

Verified without loading a model: `llama-bench.exe --help` in `research/tmp/cpu-full` prints `-t, --threads <n> (default: 16)`; llama-bench's default is `common_cpu_get_num_math()` (`tools/llama-bench/llama-bench.cpp` L394), the same function `postprocess_cpu_params` uses. (`llama-completion --help` prints `default: -1` because the value is resolved after parsing.) This also confirms the `_WIN32_WINNT >= 0x0601 && !__MINGW64__` gate (L116) fired for the clang build — the generic fallthrough at L145-146 would have printed 12.

Also: `-t 0` or negative → `std::thread::hardware_concurrency()` = 24 (`common/arg.cpp` L1517-1520). Env var: `LLAMA_ARG_THREADS` (L1522). `-tb/--threads-batch` defaults to the same as `-t`.

Startup log (`common/common.cpp` L430-433) prints `GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)` / `hardware_concurrency()` next to the system-info line.

Whether 16 threads spanning E-cores beats 8 P-core threads for memory-bound decode is a benchmark question. The code gives you the levers to test it: `-t 8 -C <P-core mask>` (§6). On this laptop the layout is verified (see header): LP 0-15 are the P-cores as SMT pairs (0,1), (2,3), ..., (14,15); LP 16-23 are the E-cores. So:

| Intent | Mask | Notes |
|---|---|---|
| P-cores only, let Windows place | `-t 8 -C 0xFFFF` | non-strict; all 8 threads may share the 16 P-core LPs |
| One thread per physical P-core, pinned | `-t 8 --cpu-strict 1 -C 0x5555` | bits 0,2,4,...,14 = first sibling of each P-core |
| P-cores incl. both hyperthreads, pinned | `-t 16 --cpu-strict 1 -C 0xFFFF` | |
| E-cores only | `-t 8 -C 0xFF0000` | bits 16-23 |
| Everything (default) | `-t 16` (no mask) | |

The snapshot also showed several LPs with the CPU-set `Parked` flag set at the time of the query — the exact core-parking behaviour the `ThreadPowerThrottling` opt-out in §6.2 is aimed at.

---

## 6. `-C/--cpu-mask`, `--cpu-strict`, `--poll`, `--prio`

Definitions: [`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/b10941/common/arg.cpp) L1513-1615. Defaults: `common/common.h` L68-75 — `priority = NORMAL`, `strict_cpu = false`, `poll = 50`, mask empty (`mask_valid = false` → any CPU).

Plumbing: `ggml_threadpool_params_from_cpu_params()` (`common/common.cpp` L1762-1776) copies `cpumask`, `prio`, `poll`, `strict_cpu` into `ggml_threadpool_params`; `common_threadpools::init` (L1798-1819) creates the pool(s) through `ggml_threadpool_new` obtained from the CPU backend registry by `get_proc_address` (so it lands in whichever `ggml-cpu-*.dll` was loaded).

### 6.1 The shipped Windows build is an OpenMP build — this changes what applies

`release.yml` L696-697: `-DGGML_OPENMP=ON -DGGML_OPENMP_FETCH=ON` (LLVM libomp 20.1.8 fetched, `ggml/src/CMakeLists.txt` L227-338). `libomp.dll` + `LICENSE-LLVM-OpenMP` are in the zip and every `ggml-cpu-*.dll` imports `libomp.dll` (verified by grep). The `system_info` line will show `OPENMP = 1`.

In [`ggml/src/ggml-cpu/ggml-cpu.c`](https://github.com/ggml-org/llama.cpp/blob/b10941/ggml/src/ggml-cpu/ggml-cpu.c):

- `ggml_graph_compute()` OpenMP branch (L3419-3436): `#pragma omp parallel num_threads(n_threads)`; inside, **each OMP thread** calls `ggml_thread_apply_priority(threadpool->prio)` and, if the mask is non-empty, `ggml_thread_apply_affinity(threadpool->workers[ith].cpumask)`, then runs the graph. Per-thread masks were precomputed in `ggml_threadpool_new_impl` (L3350-3356).
- `ggml_barrier()` is `#pragma omp barrier` under OpenMP (L582-583).
- **The whole poll/sleep machinery** — `ggml_graph_compute_poll_for_work` (L3208-3221, `n_rounds = 1024 * 128 * threadpool->poll` spins of `_mm_pause`), `ggml_graph_compute_check_for_work`, the secondary-thread loop — sits under `#ifndef GGML_USE_OPENMP` (L3176-3312). `threadpool->poll` is stored (L3333) but **never read** in the OpenMP path.

### 6.2 Flag-by-flag

| Flag | Parsing | What it does in the b10941 Windows (OpenMP) build |
|---|---|---|
| `-C / --cpu-mask M` | hex string, optional `0x`, up to 128 hex digits = 512 CPUs; **bit n = logical CPU n** (`parse_cpu_mask`, `common/common.cpp` L349-383). `-Cr lo-hi` is the range form (L312-347). Batch variants `-Cb`, `-Crb`. | Works. Each OMP thread calls `SetThreadAffinityMask` with a 64-bit mask built from its per-thread bool array (L2523-2555; > 64 CPUs unsupported on Windows, L2543-2548). Non-strict: every thread gets the **whole** mask (Windows scheduler picks within it). Warning if fewer set bits than threads (L305-308). Example: `-C 0xFFFF` = CPUs 0-15. |
| `--cpu-strict <0\|1>` | `strict_cpu` | Works. With `1`, `ggml_thread_cpumask_next()` (L2715-2735) hands each thread **one** CPU from the mask in round-robin order, so thread *i* is pinned to the *i*-th set bit; with `0` all threads share the full mask. Under OpenMP the per-thread mask table is built at pool creation (L3353-3356) and applied inside every parallel region. **SMT gotcha:** the bits are consumed consecutively, so `-t 8 --cpu-strict 1 -C 0xFFFF` pins 8 threads onto LPs 0-7 = both hyperthreads of only **4** physical P-cores. For one thread per physical core the mask must contain one bit per core (`0x5555` on this machine). |
| `--poll <0..100>` | `poll` | **No effect** in this build (see 6.1). Thread wait behaviour is libomp's: `KMP_BLOCKTIME` (default 200 ms of spinning after a parallel region before sleeping) and `OMP_WAIT_POLICY` (default `passive`, but with that KMP_BLOCKTIME spin window) per [LLVM OpenMP runtime docs](https://openmp.llvm.org/design/Runtimes.html). Inside a graph the `omp barrier` spin/yield is also libomp's. To emulate `--poll 100` set `KMP_BLOCKTIME=infinite`; to emulate `--poll 0` set `KMP_BLOCKTIME=0`. UNVERIFIED that these measurably change decode t/s here. |
| `--prio N` | -1 low, 0 normal, 1 medium, 2 high, 3 realtime (`GGML_SCHED_PRIO_*`) | Two effects. (a) Process class: `set_process_priority()` (`common/common.cpp` L234-256) calls `SetPriorityClass` with BELOW_NORMAL / NORMAL / ABOVE_NORMAL / HIGH / REALTIME (called from `common_init_result` after context creation, L1406). (b) Per thread inside every parallel region (`ggml-cpu.c` L2557-2601): `SetThreadPriority` with BELOW_NORMAL / NORMAL(no-op) / ABOVE_NORMAL / HIGHEST / TIME_CRITICAL, **and** for any prio other than LOW — including the default NORMAL — `SetThreadInformation(ThreadPowerThrottling, StateMask=0)` to opt the thread out of Windows 11 EcoQoS/power throttling (comment L2573-2576: "Newer Windows 11 versions aggressively park CPU cores and often place all our threads onto the first 4 cores"). REALTIME_PRIORITY_CLASS normally requires an elevated process; Windows silently caps it otherwise — UNVERIFIED here. |
| `--prio-batch`, `--poll-batch`, `--cpu-strict-batch` | same, for the prompt-processing pool | Same behaviour; separate pool only if params differ (`common/common.cpp` L1806-1809). |

Non-OpenMP builds (e.g. a self-built `-DGGML_OPENMP=OFF`, or the Linux `ubuntu-vulkan` release job at `release.yml` L378) use ggml's own threadpool where `--poll` is live: workers spin `n_rounds` `_mm_pause` iterations looking for a new graph, then fall back to a condvar wait (L3208-3240); the main thread is placed last in the mask order (L3362-3373).

---

## 7. Which variant each CPU family gets

Scores from §2.4. Feature data from the [Wikipedia AVX-512 support table](https://en.wikipedia.org/wiki/AVX-512) and [Raptor Lake](https://en.wikipedia.org/wiki/Raptor_Lake) / [Alder Lake](https://en.wikipedia.org/wiki/Alder_Lake) ISA lists. "Windows/Linux prebuilt" = the `GGML_CPU_ALL_VARIANTS` zips; both the Windows x64 and `ubuntu-x64` jobs (release.yml L198-202) build all 14 (Linux with GCC 14, so also not MSVC-restricted; file names are `libggml-cpu-<variant>.so`).

| CPU family | Key ISA bits present | Variant loaded (score) | Notes |
|---|---|---|---|
| Intel Sandy Bridge (2nd gen) | AVX | `sandybridge` (21) | |
| Intel Ivy Bridge (3rd gen) | AVX, F16C | `ivybridge` (23) | |
| AMD Piledriver / Steamroller (FX, A-series) | AVX, F16C, FMA, no AVX2 | `piledriver` (24) | |
| Intel Haswell → Coffee Lake / Comet Lake (4th-10th gen desktop, 8th-10th gen U/H mobile) | AVX2, FMA, F16C, BMI2 | `haswell` (64) | 6th-10th gen client parts (Skylake, Kaby, Coffee, Comet) have no AVX-512 → same as Haswell. |
| Intel Ice Lake (10th gen mobile), Tiger Lake (11th mobile), Rocket Lake (11th desktop) | AVX-512 F/CD/BW/DQ/VL + VBMI + VNNI, no BF16 | `icelake` (1472) | |
| Intel Cannon Lake (rare 8th gen i3-8121U) | AVX-512 + VBMI, no VNNI | `cannonlake` (448) | |
| Intel 12th-14th gen Core (Alder/Raptor Lake, incl. **i7-14650HX**) and Core Ultra (Meteor/Arrow/Lunar Lake) | AVX2, FMA, F16C, BMI2, **AVX-VNNI**; AVX-512 fused off | **`alderlake` (128)** | Same DLL for all hybrid client parts. Newer parts' AVX-VNNI-INT8 is never used (no variant compiled with it). Meteor/Arrow/Lunar ISA — UNVERIFIED for this note beyond "no AVX-512", all documented as AVX-VNNI-capable. |
| Intel Skylake-X / Skylake-SP Xeon (2017 HEDT & Xeon Scalable gen 1) | AVX-512 F/CD/BW/DQ/VL | `skylakex` (192) | |
| Intel Cascade Lake Xeon (gen 2) | + AVX512_VNNI | `cascadelake` (1216) | |
| Intel Cooper Lake Xeon | + AVX512_VNNI + BF16 | `cooperlake` (1728) | |
| Intel Ice Lake-SP Xeon (gen 3) | + VBMI + VNNI | `icelake` (1472) | |
| Intel Sapphire Rapids / Emerald Rapids Xeon (gen 4/5), Xeon W-2400/3400 | + BF16 + **AMX-TILE/INT8** | `sapphirerapids` (4032) | Only variant with AMX (`ggml-cpu/amx/`). Granite Rapids — UNVERIFIED but has AMX, so same. |
| AMD Zen 1 / Zen+ / **Zen 2 / Zen 3** (Ryzen 1000-5000, EPYC Naples-Milan) | AVX2, FMA, F16C, BMI2; no AVX-VNNI, no AVX-512 | `haswell` (64) | |
| AMD **Zen 4 / Zen 5** (Ryzen 7000/8000/9000, EPYC Genoa/Turin) | AVX-512 F/CD/BW/DQ/VL + VBMI + VNNI + BF16, no AMX | `zen4` (1984) | Beats `sapphirerapids` only because that one demands AMX_INT8. |
| Windows on ARM (Snapdragon X) | — | single `ggml-cpu.dll` | `release.yml` L695: `GGML_CPU_ALL_VARIANTS=OFF` for arm64; no scoring. |
| Linux aarch64 prebuilt | — | `libggml-cpu-armv8.x_y.so` family | `ggml/src/CMakeLists.txt` L524-531; ARM scoring lives in `arch/arm/cpu-feats.cpp` (not covered here). |
| **Apple Silicon (macOS prebuilt)** | — | **no variants at all** | The `macos-cpu` job (`release.yml` L64-121) passes neither `GGML_BACKEND_DL` nor `GGML_CPU_ALL_VARIANTS`, so `GGML_NATIVE` defaults ON (`ggml/CMakeLists.txt` L105-110) and the CPU backend is one natively compiled `libggml-cpu` for the CI runner's chip (`macos-26` runner; exact M-series chip UNVERIFIED), compiled into/alongside `libggml`. The `apple_m1` / `apple_m2_m3` / `apple_m4` variants defined at `ggml/src/CMakeLists.txt` L542-544 exist only for self-builds with `GGML_CPU_ALL_VARIANTS=ON`. Metal is the usual compute path anyway. |

---

## 8. Practical checklist for the blog's measurements

1. Look for `load_backend: loaded CPU backend from ...\ggml-cpu-alderlake.dll` and `AVX_VNNI = 1` in the system_info line; that confirms the variant.
2. To A/B `alderlake` vs `haswell`: make two copies of the bin folder, delete every other `ggml-cpu-*.dll`, run each from its own directory (cwd is also searched). Expect a difference only on Q8_0 / Q4_0 / Q5_x / IQ4_NL / MXFP4, not Q4_K_M / Q5_K_M / Q6_K.
3. Default is `-t 16` (E-cores included; verified via `llama-bench --help`). Try `-t 8 --cpu-strict 1 -C 0x5555` (one thread per physical P-core) vs `-t 8 -C 0xFFFF` (non-strict, P-cores only) vs `-t 16` vs `-t 24`. Do **not** combine `--cpu-strict 1` with `-C 0xFFFF -t 8` — that pins onto 4 physical cores (§6.2). Layout verified: LP 0-15 P-cores in SMT pairs, LP 16-23 E-cores.
4. `--poll` does nothing in this build; sweep `KMP_BLOCKTIME` (e.g. `0`, `200` default, `infinite`) instead. Note `--prio 0` already disables Windows power throttling per thread; `--prio 2` (HIGH) additionally raises the process class.
5. `-nr` toggles the Q4_0/Q4_K repack GEMV/GEMM path; worth an A/B for Q4_K_M.
6. Never trust "8 (8) cores" from summarised pages — Intel ARK and WMI both say 16 cores / 24 threads.

## Files and paths used

- Local binaries: `C:/PK/Github-Projects/ggml-inference-lab/bin/cpu/` (14 `ggml-cpu-*.dll`; identical copies in `bin/cuda`, `bin/vulkan`).
- Test folders created for this note (safe to delete): `C:/PK/Github-Projects/ggml-inference-lab/research/tmp/cpu-full`, `cpu-haswell-only`, `cpu-zen4-only`, `cpu-x64-only`.
- Sources at `b10941`: `src/llama.cpp`, `tools/llama-bench/llama-bench.cpp`, `ggml/src/ggml-backend-reg.cpp`, `ggml/src/ggml-backend-dl.{h,cpp}`, `ggml/src/ggml-backend-impl.h`, `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, `ggml/src/ggml-cpu/CMakeLists.txt`, `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`, `ggml/src/ggml-cpu/arch/x86/quants.c`, `ggml/src/ggml-cpu/arch/x86/repack.cpp`, `ggml/src/ggml-cpu/repack.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ggml-cpu.cpp`, `ggml/src/ggml-cpu/llamafile/sgemm.cpp`, `common/common.{h,cpp}`, `common/arg.cpp`, `src/llama-model.cpp`, `src/llama-context.cpp`, `CMakeLists.txt`, `cmake/x64-windows-llvm.cmake`, `.github/workflows/release.yml`.
