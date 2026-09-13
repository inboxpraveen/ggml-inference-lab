# 03 — Hardware ceilings and the Qwen3-0.6B model (primary-source research)

Research date: 2026-09-13. Target software: llama.cpp release **b10941** (published 2026-09-13T10:34:11Z, per
`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b10941`). Every llama.cpp file quoted below was
re-fetched from the `b10941` tag URL (`https://raw.githubusercontent.com/ggml-org/llama.cpp/b10941/<path>`) and
compared byte-for-byte with `master` at the time of research. All 26 files cited with line numbers were
**identical** between `master` and `b10941` (`src/llama-quant.cpp`, `src/llama-model.cpp`, `src/llama-model-loader.cpp`,
`src/llama-arch.cpp`, `src/llama-graph.cpp`, `src/llama-hparams.cpp`, `src/llama-kv-cache.cpp`, `src/llama-context.cpp`,
`src/models/qwen3.cpp`, `conversion/qwen.py`, `conversion/base.py`, `conversion/gemma.py`, `convert_hf_to_gguf.py`,
`gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py`, `common/arg.cpp`, `common/common.h`, `common/common.cpp`,
`CMakeLists.txt`, `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, `ggml/src/ggml-common.h`, `ggml/src/ggml-cpu/CMakeLists.txt`,
`ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`, `ggml/src/ggml-cpu/arch/x86/quants.c`, `ggml/src/ggml-cuda/CMakeLists.txt`,
`ggml/src/ggml-cuda/common.cuh`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `.github/workflows/release.yml`), so the line
numbers below are valid for both.

Convention: **VERIFIED** = read directly from the cited primary source or measured on this machine with a
non-benchmark query (nvidia-smi, WMI, CPUID). **UNVERIFIED** = plausible/derived but not confirmed by a primary
source; measure it in the lab rather than quoting it.

No model benchmark was run. The only local commands executed were `nvidia-smi --query-gpu`, `Get-CimInstance`,
a 20-byte CPUID stub, `unzip -l`, and a Python string scan of `ggml-cuda.dll` from the release zip.

---

## Part A — Hardware ceilings

### A.1 System memory: one DDR5-5600 SO-DIMM, single channel

**Measured on this machine (VERIFIED, WMI `Win32_PhysicalMemory`):**

```
DeviceLocator        : Controller0-ChannelA-DIMM0      <- only one DIMM populated, Channel A only
Capacity             : 17179869184                     <- 16 GiB
Speed / Configured   : 5600 / 5600 MT/s
DataWidth/TotalWidth : 64 / 64                         <- no ECC
FormFactor           : 12 (SODIMM)
Manufacturer/Part    : A-DATA Technology / AO1V56WCSV1-BIPS
```

Peak bandwidth arithmetic (VERIFIED by definition of the JEDEC DDR5 bus; confirmed by Intel ARK below):

```
single channel : 64 bit x 5600 MT/s / 8 = 44.8 GB/s   (= "PC5-44800")
dual channel   : 2 x 44.8               = 89.6 GB/s
```

- Intel ARK for the i7-14650HX lists **"Max Memory Bandwidth 89.6 GB/s"** with **"Max # of Memory Channels 2"** and
  "Up to DDR5 5600 MT/s" — i.e. 44.8 GB/s per channel, which is exactly what one populated channel gives.
  Source: https://www.intel.com/content/www/us/en/products/sku/235996/intel-core-i7-processor-14650hx-30m-cache-up-to-5-20-ghz/specifications.html
- A DDR5 DIMM is internally two independent 32-bit sub-channels; that does **not** add bandwidth over the 64-bit
  total (see e.g. Crucial "Everything about DDR5", https://www.crucial.com/articles/about-memory/everything-about-ddr5-ram).
  A second DIMM in Channel B is what doubles it.
- **Achievable fraction: UNVERIFIED — measure it.** Rule-of-thumb for a single-rank client DDR5 channel with a
  streaming read pattern is roughly 75–90 % of peak (~34–40 GB/s), but the only search hit that claimed a measured
  "single-channel DDR5-5600" number (58 GB/s) exceeds the physical 44.8 GB/s ceiling and was discarded as
  mis-reported. The lab should measure this with a simple multi-threaded read-bandwidth test (or infer it from the
  Q8_0 decode rate: `BW_effective = bytes_per_token x tok/s`).

**Consequence:** this laptop's CPU decode ceiling is set by **44.8 GB/s peak**, half of what the same CPU
would have with two SO-DIMMs. Adding a second 16 GB DDR5-5600 SO-DIMM is the single largest CPU-side speedup
available (up to 2x on bandwidth-bound decode), and is worth stating in the blog.

### A.2 CPU: Intel Core i7-14650HX (Raptor Lake-HX refresh)

**Intel ARK (VERIFIED, same URL as above):**

| Item | Value |
|---|---|
| Cores / threads | 16 (8 P + 8 E) / 24 |
| P-core base / max turbo | 2.2 GHz / 5.2 GHz |
| E-core base / max turbo | 1.6 GHz / 3.7 GHz |
| L3 | 30 MB Intel Smart Cache |
| Base / max turbo power | 55 W / 157 W |
| Memory | up to DDR5-5600 or DDR4-3200, 2 channels, 89.6 GB/s, 192 GB max |
| PCIe | "5.0 and 4.0", configurations "Up to 1x16+4, 2x8+4", **20 lanes** |
| ISA listed by ARK | SSE4.1, SSE4.2, AVX2; "Intel Deep Learning Boost: Yes" (ARK does not spell out AVX-VNNI) |
| Lithography / launch | Intel 7 / Q1'24, "Products formerly Raptor Lake" |

**Measured on this machine (VERIFIED):**

- WMI `Win32_Processor`: `L2CacheSize = 24576 KB`, `L3CacheSize = 30720 KB`. 24 MB L2 = 8 P-cores x 2 MB + 2
  E-core clusters x 4 MB, matching press coverage of Raptor Lake-HX ("2 MB of L2 per P-core and 4 MB per E-core
  cluster", TechPowerUp https://www.techpowerup.com/317474/intel-releases-14th-gen-core-hx-raptor-lake-refresh-mobile-processors).
- CPUID read directly (a 20-byte `cpuid` stub called through ctypes; `cpuid.py` in the session scratchpad, output below):

```
leaf1  : FMA True  F16C True  AVX True  SSE4.2 True
leaf7.0: AVX2 True BMI2 True  AVX512F False AVX512BW False AVX512VL False AVX512_VNNI False AVX512_VBMI False AMX_TILE False AVX512_FP16 False
leaf7.1: AVX_VNNI True  AVX512_BF16 False  AVX_VNNI_INT8 False  AVX_NE_CONVERT False  AVX_IFMA False
brand  : Intel(R) Core(TM) i7-14650HX     (CPUID.1.EAX = 0xB0671)
```

So: **AVX2 + FMA + F16C + BMI2 + AVX-VNNI = yes; AVX-512 (any flavour), AVX-VNNI-INT8, AMX = no.**

**Which ggml CPU backend DLL this CPU will load (VERIFIED from source):**

- The release builds ship one DLL per x86 variant. The variant table is in
  `ggml/src/CMakeLists.txt` lines 494–517 (b10941). The relevant line:
  `ggml_add_cpu_backend_variant(alderlake SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)` (line 514). Every variant
  above it in capability (`skylakex`, `cannonlake`, `cascadelake`, `icelake`, `cooperlake`, `zen4`,
  `sapphirerapids`) requires AVX-512.
- Runtime selection is a score function in `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp` lines 263–307:
  each `#ifdef GGML_<FEATURE>` block returns score 0 if the CPU lacks the feature, otherwise adds a power of two;
  `GGML_AVX_VNNI` adds `1<<6` (line 293–295) and `GGML_AVX512` returns 0 without AVX512F/CD/VL/DQ/BW (297–303).
  The highest non-zero score wins, so on this CPU it is **`ggml-cpu-alderlake.dll`**.
- Both zips in `C:/PK/Github-Projects/ggml-inference-lab/bin` contain that DLL:
  `llama-b10941-bin-win-cpu-x64.zip` (51 files, 15 `ggml-cpu-*.dll`) and
  `llama-b10941-bin-win-cuda-13.3-x64.zip` (52 files: the same 15 CPU DLLs + `ggml-cuda.dll`), verified with `unzip -l`.
- What AVX-VNNI buys: `ggml/src/ggml-cpu/arch/x86/quants.c` line 110–113 uses
  `_mm256_dpbusd_avx_epi32` under `#elif defined(__AVXVNNI__)` for the u8·s8 dot products that all the Q4/Q5/Q8
  integer kernels are built on (instead of the `_mm256_maddubs_epi16` + `_mm256_madd_epi16` pair used on plain AVX2).
  `ggml/src/ggml-cpu/CMakeLists.txt` line 299–300 defines `__AVXVNNI__` and `GGML_AVX_VNNI` for that variant.

**CPU compute peak (for the roofline, derived, UNVERIFIED clocks):**

- Raptor Cove P-core: two 256-bit FMA ports -> 16 FP32 FMA/cycle = 32 FLOP/cycle. 8 P-cores x 32 x f_allcore.
  At an assumed sustained AVX2 all-core clock of ~4.0 GHz (UNVERIFIED; max single-core turbo is 5.2 GHz) that is
  **~1.0 TFLOP/s FP32** from the P-cores.
- Gracemont E-core: press analysis describes dual 256-bit-capable FP pipes with FMA3 (Tom's Hardware
  https://www.tomshardware.com/news/intels-upcoming-gracemont-microarchitecture-to-support-avx-avx2-and-avx-vnni,
  HWCooling https://www.hwcooling.net/en/gracemont-the-not-so-little-alder-lake-core-microarch-analysis/3/). Treat the
  E-core FMA throughput as UNVERIFIED; upper bound 8 x 32 x 3.7 GHz ~ 0.95 TFLOP/s, realistically well below that.
- Integer (VNNI) dot-product throughput is higher per byte; ggml's quantized kernels are int8-based, so the practical
  compute ceiling for Q8_0/Q4 kernels is above the FP32 number. None of this matters for batch-1 decode (see A.5).

### A.3 GPU: NVIDIA GeForce RTX 5060 Laptop GPU (GB206, Blackwell, CC 12.0)

**NVIDIA official (VERIFIED, https://www.nvidia.com/en-us/geforce/laptops/compare/ and
https://www.nvidia.com/en-us/geforce/laptops/50-series/):**

| Item | Value |
|---|---|
| CUDA cores | 3328 |
| Boost clock | 1455 – 2497 MHz (OEM-dependent) |
| AI TOPS | 572 |
| Memory | 8 GB GDDR7, **128-bit** interface |
| Memory bandwidth | **384 GB/s** (50-series page) |
| GPU subsystem power | **45 – 100 W** |
| Bus | "PCI Express Gen 5" |
| Architecture / cores | Blackwell, 5th-gen Tensor, 4th-gen RT, DLSS 5 (as listed on the compare page at research time) |

**Notebookcheck (corroborating, https://www.notebookcheck.net/Nvidia-GeForce-RTX-5060-Laptop-Benchmarks-and-Specs.934941.0.html):**
chip GB206, 3328 CUDA / 104 Tensor / 26 RT cores, core 952–1455 MHz, memory 24 000 MHz effective (24 Gbps),
128-bit, **384 GB/s**, TGP 45–100 W, PCIe Gen 5, 21.9 B transistors, 181 mm².

**Beware the desktop number:** several search hits quote 448 GB/s and 3840 CUDA cores — that is the *desktop*
RTX 5060 (28 Gbps GDDR7). The laptop part is 24 Gbps x 128 bit / 8 = **384 GB/s**.

**SM count:** NVIDIA does not publish it for laptops. Derived (VERIFIED arithmetic): 3328 CUDA cores / 128 per
Blackwell SM = **26 SMs**, consistent with 26 RT cores (1 per SM) and 104 tensor cores (4 per SM).

**Measured on this machine (VERIFIED, `nvidia-smi --query-gpu=...`):**

```
name                        : NVIDIA GeForce RTX 5060 Laptop GPU
driver_version              : 616.56
memory.total                : 8151 MiB
compute_cap                 : 12.0
clocks.max.sm / graphics    : 3090 MHz          <- reported max clock on this unit (above NVIDIA's 1455-2497 MHz boost range)
clocks.max.memory           : 12001 MHz         <- 12 GHz x 2 = 24 Gbps GDDR7 -> 24e9 x 128/8 = 384 GB/s
power.default_limit         : 55 W              <- default TGP on this unit
power.max_limit             : 115 W             <- 100 W + 15 W Dynamic Boost headroom
power.min_limit             : 5 W
pcie.link.gen.gpumax        : 5
pcie.link.gen.hostmax       : 5
pcie.link.gen.current       : 1                 <- idle power management, not the link ceiling
pcie.link.width.current     : 8
pcie.link.width.max         : 16
temperature.gpu             : 52 C (idle)
```

- **PCIe link:** both host and GPU report Gen 5 capability and the link is wired **x8** (width.current = 8 while the
  GPU idles at Gen 1). Expect **PCIe 5.0 x8 = 32 GT/s x 8 x 128/130 / 8 ≈ 31.5 GB/s per direction** under load.
  The task brief's "PCIe 4.0 x8 => ~16 GB/s" is the pessimistic case (15.75 GB/s/direction). The negotiated
  generation *under load* is **UNVERIFIED** here (no benchmark was run); the blog's PCIe test should read
  `pcie.link.gen.current` during a run. Encoding/bandwidth arithmetic per e.g. Techgage
  https://techgage.com/news/what-can-we-expect-from-pcie-4-0-and-5-0-bandwidth/.
- The CPU exposes 20 lanes as "1x16+4" or "2x8+4" (ARK), so an x8 dGPU link is one of Intel's reference configurations.
- **TGP matters for decode less than you think** (decode is bandwidth-bound, A.5) but matters a lot for prefill and for
  sustained clocks. Default limit 55 W on this unit; whether the OEM's Dynamic Boost raises it under load is UNVERIFIED.

**GPU compute peak (derived):** FP32 = 3328 x 2 FLOP x clock. At NVIDIA's 2497 MHz: **16.6 TFLOP/s**; at this
unit's 3090 MHz max: 20.6 TFLOP/s. Tensor-core INT8/FP16 throughput is several times higher; NVIDIA publishes only the
"572 AI TOPS" headline for this SKU and does not state the precision/sparsity it assumes, so dense FP16/INT8
peaks are UNVERIFIED.

### A.4 Blackwell (CC 12.0) and the prebuilt CUDA binaries

**Which zip is which (VERIFIED, `.github/workflows/release.yml` b10941 lines 974–1030 and the release asset list):**

- The `windows-cuda` job builds `ggml-cuda.dll` only, with
  `-DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_CUDA=ON` for CUDA `12.4` (x64, plus
  `-DGGML_CUDA_CUB_3DOT2=ON`), `13.3` (x64) and `13.4` (arm64). The published zip then bundles that DLL with the CPU
  build's executables and CPU DLLs (52 files in `llama-b10941-bin-win-cuda-13.3-x64.zip`).
- Runtime DLLs (`cudart64_*.dll`, `cublas64_*.dll`, `cublasLt64_*.dll`) come separately in
  `cudart-llama-bin-win-cuda-13.3-x64.zip` (390 MB), which is also in `bin/`.

**Does the cuda-13.3 build contain native sm_120 code, or rely on PTX JIT? -> Native SASS for `sm_120a`, no JIT
needed on this GPU (VERIFIED from CMake + DLL scan).**

`ggml/src/ggml-cuda/CMakeLists.txt` (b10941) lines 8–55, because `GGML_NATIVE=OFF` and `CMAKE_CUDA_ARCHITECTURES`
is not set by the workflow:

```cmake
# 120    == Blackwell, needs CUDA v12.8, FP4 tensor cores
# XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
# XX-real    == compile CUDA code as device code for this specific architecture
...
if (CUDAToolkit_VERSION VERSION_LESS "13")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
endif ()
list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real 90-virtual)
endif()
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    # ... 120f-virtual would in principle work ... However, the architectures 120a-real and 121a-real should work
    # with basically any CMake version and until the release of e.g. Rubin there is no benefit to shipping
    # virtual architectures for Blackwell.
    list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
endif()
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
endif()
```

So the **CUDA 13.3** DLL is built for `75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real`.
The **CUDA 12.4** DLL (12.4 < 12.8) has no 120 entry at all and would run on this GPU only by JIT-compiling the
`90-virtual` PTX (slow first start, and no Blackwell-specific code paths). Corroboration: a string scan of the
extracted `ggml-cuda.dll` (144,910,336 bytes) found the arch identifiers `sm_86` (21x), `sm_89`, `sm_120a`,
`sm_121a` (28x each) and **no** `.target`/`compute_XX` PTX text — expected, because the fatbin is compressed
(`-compress-mode=${GGML_CUDA_COMPRESSION_MODE}`, default `"size"`, `ggml/CMakeLists.txt` line 211 and
`ggml-cuda/CMakeLists.txt` line 199), so the CMake list is the authoritative statement and the DLL strings are
consistent with it.

Caveats on the `a` suffix: `sm_120a` is *architecture-specific* SASS — it runs only on CC 12.0 devices (exactly this
GPU) and is not forward-compatible with future architectures; that is why llama.cpp also keeps `90-virtual` PTX as
the generic fallback. Inside the code, `ggml/src/ggml-cuda/common.cuh` line 60 defines
`GGML_CUDA_CC_BLACKWELL 1200` with the comment "While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core
instructions available to 1200 family", i.e. consumer Blackwell gets its own tensor-core code paths.

Driver: CUDA 13.x minor-version compatibility requires driver >= 580, and the CUDA 13.3 GA release notes list the
toolkit driver as >= 610.43 (Linux column; the Windows column is "N/A" since 13.1)
(https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html). Driver **616.56** on this
machine satisfies both. The same notes confirm CUDA 13.0 dropped Maxwell/Pascal/Volta (sm_50–70), which is why the
13.x CMake branch omits `50/61/70-virtual`. Exact Windows minimum for 13.3: UNVERIFIED.

CUDA graphs: the top-level `CMakeLists.txt` (b10941) lines 169–171 set `GGML_CUDA_GRAPHS_DEFAULT ON` when
building llama.cpp, so the release DLL has `GGML_CUDA_USE_GRAPHS` compiled in (`ggml-cuda/CMakeLists.txt` 123–124).
At runtime they are disabled by setting the env var `GGML_CUDA_DISABLE_GRAPHS` (`common.cuh` line 1292) — useful
for an A/B in the launch-overhead section below.

### A.5 The decode roofline: `tok/s_max = bandwidth / bytes_read_per_token`

Batch-1 decode multiplies every weight matrix by **one** activation vector. For each weight byte streamed from
memory the hardware performs ~2 FLOP per weight (multiply + add), so the arithmetic intensity is

```
I_decode  = 2 FLOP / bytes_per_weight
          = 2 / 1.0625  = 1.88 FLOP/byte   at Q8_0  (8.5 bits/weight)
          = 2 / 0.5625  = 3.56 FLOP/byte   at Q4_K  (4.5 bits/weight)
```

The machine's "ridge point" (where it stops being bandwidth-bound) is `peak FLOP/s ÷ peak bytes/s`:

| Device | Peak compute (derived) | Peak BW | Ridge (FLOP/byte) | Q8_0 decode is bandwidth-bound by |
|---|---|---|---|---|
| i7-14650HX, P-cores FP32 @ ~4 GHz (UNVERIFIED clock) | ~1.0 TFLOP/s | 44.8 GB/s | ~23 | ~12x |
| RTX 5060 Laptop, FP32 CUDA cores @ 2497 MHz | 16.6 TFLOP/s | 384 GB/s | ~43 | ~23x |
| RTX 5060 Laptop, tensor cores (INT8, UNVERIFIED peak) | >> 16.6 | 384 GB/s | >> 43 | >> 23x |

So at batch 1 the weights stream is the bottleneck on both devices and the ceiling is simply:

```
t_token   >= bytes_read_per_token / BW
tok/s_max  = BW / bytes_read_per_token
```

**Prefill** processes `n_ubatch` tokens (default 512, `common/common.h` line 452) against the *same* weight
stream, so intensity scales with the batch: `I_prefill ≈ 1.88 x n_tokens` FLOP/byte at Q8_0. It crosses the CPU
ridge at n ≈ 12 tokens and the GPU FP32 ridge at n ≈ 23 tokens (later for tensor-core paths). Beyond that, prefill
is **compute-bound** — its rate depends on FLOP/s (clocks, TGP, tensor cores, MMQ vs cuBLAS), not on memory.

**Numbers for the 0.64 GB Q8_0 Qwen3-0.6B file** (633,495,552 bytes of tensor data per token, see Part B; the file is
639,446,688 bytes including ~5.95 MB of GGUF metadata which is not read per token):

| Device | Peak BW | Weights-only ceiling | at 80 % of peak (UNVERIFIED efficiency) |
|---|---|---|---|
| DDR5-5600, one channel (this laptop) | 44.8 GB/s | **70.7 tok/s** (14.1 ms/token) | 56.6 tok/s |
| DDR5-5600, two channels (if a 2nd DIMM were added) | 89.6 GB/s | 141.4 tok/s | 113 tok/s |
| RTX 5060 Laptop GDDR7 | 384 GB/s | **606 tok/s** (1.65 ms/token) | 485 tok/s |

Two corrections the weights-only number misses:

1. **KV-cache reads grow with context.** Every decode step also reads the whole K and V cache for all previous
   positions: 114,688 bytes per position at f16 for this model (B.7). Bytes/token = weights + n_past x 114,688:
   at n_past = 1024 that is +117 MB (+19 %), at n_past = 4096 it is +470 MB — nearly doubling the traffic and
   halving the ceiling (CPU: ~40 tok/s; GPU: ~350 tok/s). `q8_0` KV (60,928 B/position) roughly halves this term.
2. **On the GPU a 0.6B model is small enough that per-kernel launch/latency overhead competes with bandwidth.** The
   Qwen3 graph (`src/models/qwen3.cpp`) issues on the order of 12–15 ops per layer (norm, Q/K/V matmuls or fused QKV,
   q_norm, k_norm, 2x RoPE, attention, output matmul, add, norm, gate/up matmuls, GLU, down matmul, add) x 28 layers
   ≈ 350–450 kernels per token before fusion. At a few microseconds each that is ~1–2 ms/token — the same order as
   the 1.65 ms bandwidth time. This is why CUDA graphs (A.4) and kernel fusion (`GGML_CUDA_DISABLE_FUSION`,
   `ggml-cuda.cu` lines 3426/4499) matter far more for this model than for a 7B one. The per-launch cost on this
   driver/GPU is UNVERIFIED — measure with and without `GGML_CUDA_DISABLE_GRAPHS=1`.

**Prefill numbers (illustrative, derived):** the matmul work per prompt token is ~2 x 440.5 M (per-layer weights)
= 0.88 GFLOP, plus the LM head only for the positions whose logits are requested (llama.cpp gathers only the
needed rows with `inp_out_ids` before the head: `qwen3.cpp` `build_inp_out_ids()` / `ggml_get_rows(ctx0, cur,
inp_out_ids)`), plus attention (negligible at these lengths: 4 x n_ctx x 2048 FLOP per token per layer). A 512-token
prompt is ~450 GFLOP: ~0.45 s at 1 TFLOP/s CPU (~1,100 tok/s) and ~27 ms at 16.6 TFLOP/s GPU FP32 (~19,000 tok/s
if the kernels hit peak — they will not; expect a fraction). UNVERIFIED — these are the ceilings the measurements
should be compared against, not predictions.

**Where the PCIe link enters:** once all 28 layers plus the output layer are in VRAM (`-ngl all`, or `-ngl 29`+;
the default is `-ngl auto` = `n_gpu_layers = -1`, `common/common.h` line 473 and `common/arg.cpp` 2785–2794, which
together with the default `-fit on` — `fit_params = true`, `common.h` 476; `common_fit_params()` called from
`common_init_result`, `common/common.cpp` 1295–1325 — sizes unset parameters to free device memory with a 1 GiB
per-device reserve, `fit_params_target`, line 481; for a 0.64 GB model on an 8 GB card that should mean full offload,
but the exact fit algorithm is UNVERIFIED here and belongs in the CLI-flags research note), the only per-token PCIe traffic is the token id in and the logits vector out (151,936 x 4 B = 608 KB per
decoded token, ~19 µs at 31.5 GB/s) — negligible. It becomes the bottleneck only when layers are split between
CPU and GPU (activations cross the link each layer boundary) or when weights are left in host memory.

---

## Part B — The model: Qwen/Qwen3-0.6B

### B.1 HF `config.json` (VERIFIED, https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)

| Key | Value |
|---|---|
| `architectures` | `["Qwen3ForCausalLM"]`, `model_type: "qwen3"` |
| `hidden_size` | **1024** |
| `intermediate_size` | **3072** |
| `num_hidden_layers` | **28** |
| `num_attention_heads` | **16** |
| `num_key_value_heads` | **8** (GQA ratio n_gqa = 2) |
| `head_dim` | **128** |
| `vocab_size` | **151936** |
| `tie_word_embeddings` | **true** |
| `max_position_embeddings` | **40960** |
| `rope_theta` / `rms_norm_eps` | 1,000,000 / 1e-6 |
| `attention_bias` | false (no Q/K/V biases, unlike Qwen2) |
| `torch_dtype` | bfloat16 |
| `bos_token_id` / `eos_token_id` | 151643 / 151645 |

Note the trap: `num_attention_heads x head_dim = 16 x 128 = 2048 ≠ hidden_size = 1024`. Q projects 1024 -> 2048 and
O projects 2048 -> 1024; K and V project 1024 -> 8 x 128 = 1024.

### B.2 Exact parameter count by tensor group (VERIFIED against the safetensors header)

Per layer (shapes read from the safetensors header of `model.safetensors`, HTTP range request, first 40 KB):

| Tensor (HF name) | Shape | Params |
|---|---|---|
| `self_attn.q_proj.weight` | [2048, 1024] | 2,097,152 |
| `self_attn.k_proj.weight` | [1024, 1024] | 1,048,576 |
| `self_attn.v_proj.weight` | [1024, 1024] | 1,048,576 |
| `self_attn.o_proj.weight` | [1024, 2048] | 2,097,152 |
| `self_attn.q_norm.weight`, `k_norm.weight` | [128] each | 256 |
| **attention subtotal** | | **6,291,712** |
| `mlp.gate_proj.weight` | [3072, 1024] | 3,145,728 |
| `mlp.up_proj.weight` | [3072, 1024] | 3,145,728 |
| `mlp.down_proj.weight` | [1024, 3072] | 3,145,728 |
| **MLP subtotal** | | **9,437,184** |
| `input_layernorm`, `post_attention_layernorm` | [1024] each | 2,048 |
| **per layer** | | **15,730,944** |

Whole model:

| Group | Params | Share (tied) |
|---|---|---|
| 28 layers x 15,730,944 | 440,466,432 | 73.90 % |
| `model.norm.weight` (final RMSNorm) | 1,024 | ~0 |
| `model.embed_tokens.weight` [151936, 1024] | **155,582,464** | **26.10 %** |
| **Total unique parameters (tied head)** | **596,049,920** | |
| + separate `lm_head.weight` if stored | +155,582,464 -> 751,632,384 | (embedding then = 20.70 % of stored) |

The 2-D matrix weights per layer are exactly 15,728,640 (attention 6,291,456 + MLP 9,437,184); the 1-D norms are
2,304 per layer + 1,024 final = 65,536 total, kept as F32 in every GGUF.

**Surprise in the checkpoint (VERIFIED):** despite `tie_word_embeddings: true`, `model.safetensors`
(1,503,300,328 bytes) contains **both** `lm_head.weight [151936, 1024]` and `model.embed_tokens.weight
[151936, 1024]` as separate BF16 tensors; the header's total is 751,632,384 parameters, and
751,632,384 x 2 B + 35,552 B JSON header + 8 B length field = 1,503,300,328 B, exactly the file size. This matters for conversion (B.5).

### B.3 Bytes per weight of the quant types (VERIFIED, `ggml/src/ggml-common.h` and `gguf-py/gguf/constants.py`)

Block structs (`ggml-common.h`; `QK4_0 = QK8_0 = QK4_NL = 32`, `QK_K = 256`, `K_SCALE_SIZE = 12`, `ggml_half` = 2 B):

| Type | Block (weights) | Bytes/block | Layout (from the struct) | bits/weight |
|---|---|---|---|---|
| F32 | 1 | 4 | | 32 |
| F16 / BF16 | 1 | 2 | | 16 |
| Q8_0 | 32 | 34 | `half d; int8 qs[32]` | **8.5** |
| Q6_K | 256 | 210 | `ql[128] qh[64] int8 scales[16] half d` | **6.5625** |
| Q5_K | 256 | 176 | `half d, dmin; scales[12]; qh[32]; qs[128]` | **5.5** |
| Q5_0 | 32 | 22 | `half d; qh[4]; qs[16]` | 5.5 |
| Q4_K | 256 | 144 | `half d, dmin; scales[12]; qs[128]` | **4.5** |
| Q4_0 | 32 | 18 | `half d; qs[16]` | 4.5 |
| Q4_1 | 32 | 20 | `half d, m; qs[16]` | 5.0 |
| IQ4_NL | 32 | 18 | `half d; qs[16]` (non-linear codebook) | 4.5 |
| IQ4_XS | 256 | 136 | `half d; u16 scales_h; scales_l[4]; qs[128]` | **4.25** |
| Q3_K | 256 | 110 | `hmask[32] qs[64] scales[12] half d` | 3.4375 |
| Q2_K | 256 | 84 | `scales[16] qs[64] half d, dmin` | 2.625 |
| Q8_K (activation-side only) | 256 | 292 | `float d; int8 qs[256]; int16 bsums[16]` | 9.125 |

(`GGML_QUANT_SIZES` in `constants.py` lines 5829–5858 gives the same `(block, bytes)` pairs.)

The task brief's numbers check out: Q8_0 = 34 bytes per 32 weights = 8.5 bpw; Q6_K 6.5625; Q5_K 5.5; Q4_K 4.5.

### B.4 Embedding/head matrix as a fraction of bytes, per type (computed from B.2 + B.3)

For a *tied* GGUF (one matrix, 155,582,464 weights) with all 2-D tensors in a single type and norms in F32:

| Type | Embedding/head bytes | 28-layer 2-D bytes | Total tensor bytes | Embedding share |
|---|---|---|---|---|
| BF16 | 311,164,928 (311.2 MB) | 880,803,840 | 1,192,230,912 | 26.10 % |
| Q8_0 | 165,306,368 (165.3 MB) | 467,927,040 | **633,495,552** | 26.09 % |
| Q6_K | 127,626,240 | 361,267,200 | 489,155,584 | 26.09 % |
| Q5_K | 106,962,944 | 302,776,320 | 410,001,408 | 26.09 % |
| Q4_K | 87,515,136 | 247,726,080 | 335,503,360 | 26.08 % |
| Q4_0 / IQ4_NL | 87,515,136 | 247,726,080 | 335,503,360 | 26.08 % |
| IQ4_XS | 82,653,184 | 233,963,520 | 316,878,848 | 26.08 % |
| Q3_K | 66,851,840 | 189,235,200 | 256,349,184 | 26.08 % |
| Q2_K | 51,050,496 | 144,506,880 | 195,819,520 | 26.07 % |

(The share is ~26.1 % regardless of type because all 2-D tensor dimensions here are multiples of 256, so every
tensor is quantized at the same bpw. In real `_M`/`_S` mixes the head is kept at Q6_K while layers go lower, so its
byte share *rises*: in Q4_K_M it is 127.6 MB of 390.8 MB = **32.7 %**; in Q2_K it is 127.6 of 290.3 MB = **44 %**.)

Sanity check against the official file: Q8_0 tensor bytes 633,495,552 + 5,951,136 B metadata = 639,446,688 B =
the exact size of `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (HF tree API). The "0.64 GB model" is therefore
**633.5 MB of weights streamed per decode token.**

### B.5 Why `get_rows` reads one row but the tied LM head reads the whole matrix

Both operations use the same tensor object when weights are tied, but they are different ggml ops on it:

- **Input embedding** — `src/llama-graph.cpp` `llm_graph_context::build_inp_embd()`:
  `cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);` with `inp->tokens` an I32 vector of `ubatch.n_tokens` ids.
  `GET_ROWS` is a gather: for each token it copies one row (`n_embd` = 1024 weights, 1088 bytes at Q8_0) and
  dequantizes it. Cost per decoded token: **~1 KB**, independent of vocab size.
- **LM head** — `src/models/qwen3.cpp` graph tail:
  `cur = build_lora_mm(model.output, cur, model.output_s);` -> `ggml_mul_mat(ctx0, w, cur)` (`llama-graph.cpp`,
  `build_lora_mm`). To produce logits for all 151,936 vocabulary entries it must compute the dot product of the
  hidden state with **every** row of the [1024 x 151936] matrix, i.e. read all 165.3 MB (Q8_0) — every decode step.
  There is no way to know which vocabulary rows will score highest without reading them all.

So for this model **26 % of every decode step's memory traffic is the LM head**, and the per-layer transformer is
the other 74 %. That is unusually head-heavy (a 7B Llama-2's head is ~2 %: 32000 x 4096 of 6.7 B) and is the main reason Qwen3-0.6B does not
decode 12x faster than a 7B model despite having 12x fewer parameters per layer — plus the launch-overhead
effect in A.5.

### B.6 GGUF tensor names, and what the converter does with `tie_word_embeddings`

**Naming (VERIFIED, `gguf-py/gguf/constants.py` `TENSOR_NAMES` lines 1403–1446, and `MODEL_TENSORS[MODEL_ARCH.QWEN3]`
lines 2724–2741):**

```
token_embd.weight          [1024, 151936]   (GGUF stores ne[0]=n_embd first)
output_norm.weight         [1024]
output.weight              [1024, 151936]   optional, see below
blk.N.attn_norm.weight     [1024]
blk.N.attn_q.weight        [1024, 2048]
blk.N.attn_k.weight        [1024, 1024]
blk.N.attn_v.weight        [1024, 1024]
blk.N.attn_output.weight   [2048, 1024]
blk.N.attn_q_norm.weight   [128]
blk.N.attn_k_norm.weight   [128]
blk.N.ffn_norm.weight      [1024]
blk.N.ffn_gate.weight      [1024, 3072]
blk.N.ffn_up.weight        [1024, 3072]
blk.N.ffn_down.weight      [3072, 1024]
```

`MODEL_TENSORS[QWEN3]` also lists `ATTN_QKV` (`blk.N.attn_qkv`) and `ROPE_FREQS`; a fused QKV tensor is written
only if the converter is run with `--fuse-qkv` (`convert_hf_to_gguf.py` b10941 line 161; `conversion/base.py`
`fuse_qkv: bool = False`, line 134). The loader accepts either form (`llama-model.cpp` `create_tensor_qkv`, tries
`attn_qkv` with `TENSOR_NOT_REQUIRED` first, then separate Q/K/V). All public Qwen3-0.6B GGUFs examined use separate Q/K/V.

The HF -> GGUF name mapping comes from `gguf-py/gguf/tensor_mapping.py`: `model.embed_tokens` -> `token_embd`,
`lm_head` -> `output`, `model.norm` -> `output_norm`, `model.layers.{bid}.self_attn.q_norm` -> `attn_q_norm`, etc.

**Converter behaviour (VERIFIED, b10941 `conversion/qwen.py` and `conversion/base.py`):**

- `Qwen3Model(Qwen2Model)` (`conversion/qwen.py` lines 161–256) sets `model_arch = gguf.MODEL_ARCH.QWEN3` and reads
  `tie_word_embeddings` **only** inside `_find_rerank_config()` (line 216) for the Qwen3-Reranker special case; its
  `modify_tensors()` (241–256) otherwise just `yield from super().modify_tensors(...)`.
- `conversion/base.py` contains **no** generic "skip `lm_head.weight` when tied" logic (grep for
  `tie_word_embeddings` / `lm_head` / `tied`: zero hits). Contrast `conversion/gemma.py` lines 56–58, which does
  `if name == "lm_head.weight": ... skip` with the comment "lm_head is not used in llama.cpp".
- Therefore: **if the checkpoint contains `lm_head.weight`, the converter writes `output.weight`; if it does not,
  no `output.weight` is written.** Since Qwen's checkpoint *does* contain it (B.2), a straight conversion of
  `Qwen/Qwen3-0.6B` produces a GGUF with a redundant 155.6 M-parameter `output.weight`.

**Both variants exist in the wild (VERIFIED by parsing GGUF headers via HTTP range requests):**

| Repo / file | Tensors | `output.weight`? | `token_embd` type | File size |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (official, `general.file_type = 7`) | **310** | no | Q8_0 | 639,446,688 |
| `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` | 310 | no | Q8_0 | 639,447,744 |
| `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` (`file_type = 15`) | 310 | no | **Q6_K** | 396,705,472 |
| `ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (`general.size_label = 752M`) | **311** | **yes, Q8_0 [1024,151936]** | Q8_0 | **804,753,632** |
| `ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf` | — | (yes, by size) | — | 1,509,347,552 (= 751.6 M x 2) |
| `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf` | — | (no, by size) | — | 1,198,182,848 (= 596.0 M x 2) |
| `bartowski/Qwen_Qwen3-0.6B-GGUF/*-Q4_K_M.gguf` | — | (yes, by size) | — | 484,220,320 vs unsloth 396,705,472 |

310 = 2 global + 28 x 11; 311 adds `output.weight`. The ggml-org Q8_0 is 165.3 MB larger — exactly one Q8_0 copy
of the [1024 x 151936] matrix.

**Runtime fallback (VERIFIED, `src/models/qwen3.cpp` `load_arch_tensors`):**

```cpp
tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
// if output is NULL, init from the input tok embed
if (output == NULL) {
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
}
```

and in `src/llama-model-loader.cpp` lines 1156–1161 the duplicated request is re-classified as the output tensor
("we assume that it is being loaded as the output tensor": `if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && (flags &
TENSOR_DUPLICATED)) tn_tensor = LLM_TENSOR_OUTPUT;`), which makes it use the OUTPUT tensor-info
(`llama-arch.cpp` line 717: `{LLM_TENSOR_OUTPUT, {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL_MAT}}` vs line 713
`{LLM_TENSOR_TOKEN_EMBD, {LLM_TENSOR_LAYER_INPUT, GGML_OP_GET_ROWS}}`) and therefore the **output layer's buffer
list** (`llama-model-loader.cpp` 1207–1218).

**Consequences for the blog:**

1. *Bytes per decode token are identical either way*: the head matmul reads one [1024 x 151936] matrix
   (165.3 MB at Q8_0) whether it is called `output.weight` or the duplicated `token_embd.weight`. The 311-tensor
   file only costs disk/RAM/VRAM footprint (+165 MB), not decode bandwidth.
2. *Placement differs with tied weights and GPU offload* (VERIFIED, `src/llama-model.cpp` lines 1508–1519 and
   `llama-model-loader.cpp` 1361–1383): the input layer is pinned to CPU — "there is very little benefit to
   offloading the input layer, so always keep it on the CPU" — while `dev_output = get_layer_buft_list(n_layer_all)`
   goes to the GPU when the output layer is offloaded, i.e. `-ngl >= n_layer + 1` (= 29 for this model, or `-ngl all`;
   with the default `-ngl auto` + `-fit on` the fit step picks the value — see A.5). `TENSOR_DUPLICATED` reuses the existing tensor only "if the original tensor was
   allocated in the same buffer type context"; otherwise `ggml_dup_tensor` creates a second copy and
   `size_data += ggml_nbytes(&t_meta)`. So with full offload a *tied* GGUF ends up with `token_embd` in host RAM
   (for `get_rows`) **and** a second copy in VRAM (for the head matmul) — 165 MB each at Q8_0. With CPU-only
   inference both live in the same CPU buffer and there is one copy.
3. *Quantization mix differs*: `src/llama-quant.cpp` line 185 `bool has_tied_embeddings = true; // assume tied until
   we see output.weight` (cleared at line 903 when `output.weight` is present). In `llama_tensor_get_type_impl`
   (lines 456–477): `if (category == OUTPUT || (qs.has_tied_embeddings && category == TOKEN_EMBD))` -> the head
   gets `--output-tensor-type` if given, else Q8_0 if `arch == FALCON || nx % qk_k != 0`, else Q5_K for the IQ1/IQ2
   family, **else Q6_K unless the default type is already Q8_0**. For the 311-tensor (untied) file the same rule
   applies to `output.weight` while `token_embd.weight` falls through to the plain `TOKEN_EMBD` branch (lines
   490–506) and stays at the file's default type (Q4_K in a Q4_K_M). That is why unsloth's tied Q4_K_M has
   `token_embd = Q6_K` (header parsed above) and why bartowski's untied Q4_K_M is larger than unsloth's by
   484,220,320 − 396,705,472 = 87,514,848 B ≈ 87,515,136 B, which is exactly one Q4_K copy of the [1024 x 151936]
   matrix (a Q4_K `token_embd` + a Q6_K `output.weight` vs one Q6_K matrix) — an independent confirmation of the rule.

### B.7 KV cache bytes per token (VERIFIED, `src/llama-hparams.cpp` 156–166 and `src/llama-kv-cache.cpp` 233–234)

```
n_embd_k_gqa = n_embd_head_k * n_head_kv = 128 * 8 = 1024   (same for V)
k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);  // per layer
v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
```

Per cached position, all 28 layers, K + V:

| KV type | bytes per position | 4,096 ctx | 40,960 ctx (model max) |
|---|---|---|---|
| f16 (default, `common/common.h` 587–588 in `common_params`: `cache_type_k = cache_type_v = GGML_TYPE_F16`; 341–342 are the draft-model copies) | 28 x 2 x 1024 x 2 = **114,688 B (112 KiB)** | 448 MiB | 4,480 MiB |
| q8_0 (`-ctk q8_0 -ctv q8_0`) | 28 x 2 x (1024/32 x 34) = **60,928 B (59.5 KiB)** | 238 MiB | 2,380 MiB |

Rules that apply (VERIFIED, `src/llama-context.cpp` 3704–3737): a quantized **V** cache requires Flash Attention —
with the default `-fa auto` (`common/common.h` line 499) llama.cpp logs "enabling flash_attn since it is required for
quantized V cache" and turns it on; with `-fa off` it errors out. A quantized K or V type also needs its block size
to divide `n_embd_head` (128 % 32 == 0 for q8_0/q4_0 — fine here; 128 % 256 != 0 rules out K-quant KV types).

### B.8 Quantization mixes: what `_S/_M/_L` change for *this* model (VERIFIED, `src/llama-quant.cpp`)

Inputs the selector looks at for Qwen3-0.6B: `arch = LLM_ARCH_QWEN3` (not Falcon), `n_expert = 0`,
`n_gqa() = 16/8 = 2` (so every `n_gqa() >= 4` branch is **false**), `type = LLM_TYPE_0_6B` (`qwen3.cpp`
`load_arch_hparams`: 28 layers and `n_embd == 1024`), 28 layers, every 2-D `ne[0]` in {1024, 2048, 3072} is a multiple
of 256 so `tensor_type_fallback()` never demotes anything, and `has_tied_embeddings = true` for the 310-tensor files.

Defaults per ftype: `llama_ftype_get_default_type()` lines 847–890 (Q4_K_S and Q4_K_M -> Q4_K; Q5_K_S/M -> Q5_K;
Q3_K_S/M/L -> Q3_K; Q2_K -> Q2_K; IQ4_XS -> IQ4_XS; IQ4_NL -> IQ4_NL; Q6_K, Q8_0, Q4_0 -> themselves).
`--pure` skips the mix logic entirely; `--tensor-type <regex>=<type>` overrides it per tensor (lines 707–729).

Layer-position helper (line 434–436):

```cpp
auto use_more_bits = [](int i_layer, int n_layers) -> bool {
    return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
};
```

For n = 28 this is true for layers **{0,1,2, 5,8,11,14,17,20,23, 24,25,26,27}** — 14 of 28.

Resulting per-tensor types (head = `token_embd` when tied):

| ftype | head | attn_q, attn_k, ffn_gate, ffn_up | attn_v | attn_output | ffn_down |
|---|---|---|---|---|---|
| Q8_0 | Q8_0 (`new_type != Q8_0` guard, line 474–476) | Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| Q6_K | Q6_K | Q6_K | Q6_K | Q6_K | Q6_K |
| Q5_K_M | Q6_K | Q5_K | **Q6_K** in the 14 `use_more_bits` layers, else Q5_K (line 556–557) | Q5_K | **Q6_K** in those 14 layers (line 623) |
| Q5_K_S | Q6_K | Q5_K | Q5_K | Q5_K | Q5_K |
| Q4_K_M | Q6_K | Q4_K | **Q6_K** in the 14 layers (556–557) | Q4_K | **Q6_K** in the 14 layers (612–618) |
| Q4_K_S | Q6_K | Q4_K | **Q5_K** for `i_attention_wv < 4` (layers 0–3, line 558) | Q4_K | **Q5_K** for `i_layer < n_layer/8` (layers 0–2, line 624–626) |
| Q4_0 | Q6_K | Q4_0 | Q4_0 | Q4_0 | Q4_0; **Q4_1** in layers 0–2 only when an imatrix is supplied (627–633) |
| IQ4_XS | Q6_K | IQ4_XS | IQ4_XS (Q5_K bump needs n_gqa>=4 — not here, line 553–555) | IQ4_XS | **Q5_K** in layers 0–2 only *without* imatrix (620–622) |
| IQ4_NL | Q6_K | IQ4_NL | IQ4_NL | IQ4_NL | same rule as IQ4_XS |
| Q3_K_L | Q6_K | Q3_K | **Q5_K** (line 552) | **Q5_K** (line 648) | **Q5_K** (line 609–611) |
| Q3_K_M | Q6_K | Q3_K | **Q5_K** for the first 2 layers, else **Q4_K** (549–551) | **Q4_K** (647) | **Q5_K** for `i_layer < n_layer/16` (layer 0), else **Q4_K** (600–604) |
| Q3_K_S | Q6_K | Q3_K | Q3_K | Q3_K | Q3_K |
| Q2_K | Q6_K | Q2_K | **Q3_K** (`n_gqa >= 4` false, line 534–536) | **Q3_K** (645) | **Q3_K** (593) |

(A fused `attn_qkv`, if present, gets Q4_K under Q3_K_M/L, Q5_K under Q4_K_M and Q6_K under Q5_K_M, lines 655–661 — not applicable to the split Q/K/V files examined.)
The `ATTENTION_K`/`ATTENTION_Q` branches only touch IQ3 ftypes and 8-expert MoE; `FFN_GATE`/`FFN_UP` only IQ3_XS.

**Validation of this table against real files (VERIFIED):** computing the tensor bytes from the table and the
block sizes in B.3 and comparing with the unsloth file sizes gives a constant **5,952,192-byte** difference
(the GGUF header + alignment padding) for **every** K-quant mix — Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S,
Q3_K_M, Q3_K_S, Q2_K:

| ftype | predicted tensor bytes | unsloth file | delta |
|---|---|---|---|
| Q8_0 | 633,495,552 | 639,447,744 | 5,952,192 |
| Q6_K | 489,155,584 | 495,107,776 | 5,952,192 |
| Q5_K_M | 438,463,488 | 444,415,680 | 5,952,192 |
| Q5_K_S | 430,664,704 | 436,616,896 | 5,952,192 |
| Q4_K_M | 390,753,280 | 396,705,472 | 5,952,192 |
| Q4_K_S | 377,318,400 | 383,270,592 | 5,952,192 |
| Q3_K_M | 341,175,296 | 347,127,488 | 5,952,192 |
| Q3_K_S | 317,123,584 | 323,075,776 | 5,952,192 |
| Q2_K | 290,286,592 | 296,238,784 | 5,952,192 |
| Q4_0 (no imatrix) | 375,614,464 | 382,156,480 | 6,542,016 = 5,952,192 + **589,824** |
| IQ4_XS (no imatrix) | 363,326,464 | 367,804,096 | 4,477,632 = 5,952,192 − **1,474,560** |
| IQ4_NL (no imatrix) | 376,794,112 | 381,566,656 | 4,772,544 = 5,952,192 − **1,179,648** |

The three outliers are exactly the imatrix-dependent branches, proving unsloth quantized with an imatrix:
Q4_0 +589,824 = 3 layers x 3,145,728 x (20 − 18)/32 (`ffn_down` 0–2 -> Q4_1); IQ4_XS −1,474,560 =
3 x 3,145,728 x (176 − 136)/256 and IQ4_NL −1,179,648 = 3 x 3,145,728 x (176/256 − 18/32) (the `!qs.has_imatrix`
Q5_K bump did not fire). The unsloth Q4_K_M header independently confirms `token_embd = Q6_K`, `attn_v`/`ffn_down`
layer 0 = Q6_K, everything else Q4_K, and its per-type totals (Q6_K 214,302,720 = 155,582,464 + 14 x 4,194,304;
Q4_K 381,681,664) match the table exactly.

Practical reading for the blog: in this model `_M` vs `_S` at 4-bit changes only `attn_v` and `ffn_down` in half
the layers (+13.4 MB, +3.6 % bytes per token for Q4_K_M over Q4_K_S), and every 4-bit-class mix still carries a
127.6 MB Q6_K head — **a third of the traffic**. `--output-tensor-type q4_k` (or `--tensor-type token_embd=q4_k`)
on a tied file drops that to 87.5 MB and is the one quantization knob with an outsized bandwidth effect here;
the perplexity cost is what the lab should measure.

---

## Appendix — commands and scripts used (all non-benchmark)

- Source fetch/compare: `curl -sL https://raw.githubusercontent.com/ggml-org/llama.cpp/{master,b10941}/<path>` + `cmp`.
- Safetensors header: `curl -r 0-40000 https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors` then
  parse the 8-byte length + JSON header.
- GGUF headers: `curl -r 0-12000000 https://huggingface.co/<repo>/resolve/main/<file>.gguf` then a ~40-line GGUF v3
  header parser (magic, n_tensors, KV pairs, tensor infos) — script `ggufhdr.py` in the session scratchpad.
- Byte model / roofline: `calc.py` in the session scratchpad (numbers reproduced in B.2, B.4, B.7, B.8, A.5).
- Hardware: `nvidia-smi --query-gpu=...`, `Get-CimInstance Win32_PhysicalMemory / Win32_Processor`, `cpuid.py`
  (ctypes + VirtualAlloc executable stub, leaves 1, 7.0, 7.1, 0x80000002–4).
- Release zips: `unzip -l bin/*.zip`; `ggml-cuda.dll` extracted to `research/tmp/b10941-cuda/` and scanned with a
  Python regex for `sm_\d+[af]?`, `\.target`, `compute_\d+`.
