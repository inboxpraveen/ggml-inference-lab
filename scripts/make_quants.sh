#!/bin/bash
# Build every quantized variant used in the study from the BF16 GGUF.
# Naming: <family>-<detail>.gguf. All K-quants below Q8 use the unsloth importance matrix (legacy .dat, read natively).
set -e
LAB="/c/PK/Github-Projects/ggml-inference-lab"
Q="$LAB/bin/cpu/llama-quantize.exe"
SRC="$LAB/models/Qwen3-0.6B-BF16.gguf"
IM="$LAB/models/imatrix_unsloth.dat"
OUT="$LAB/models/variants"; mkdir -p "$OUT"
T=8
LOG="$LAB/results/quantize.log"; : > "$LOG"

mk() {  # mk <name> <type> [extra llama-quantize args...]
  local name="$1"; local type="$2"; shift 2
  local f="$OUT/$name.gguf"
  if [ -f "$f" ]; then echo "skip $name"; return; fi
  echo "== $name  ($type $*)" | tee -a "$LOG"
  "$Q" "$@" "$SRC" "$f" "$type" $T >> "$LOG" 2>&1
  ls -l "$f" | awk '{print "   ", $5, "bytes"}'
}

# ---- A. plain sweep (default mixes) ---------------------------------------------------------
mk plain-Q8_0    Q8_0
mk plain-Q6_K    Q6_K   --imatrix "$IM"
mk plain-Q5_K_M  Q5_K_M --imatrix "$IM"
mk plain-Q4_K_M  Q4_K_M --imatrix "$IM"
mk plain-Q4_K_S  Q4_K_S --imatrix "$IM"
mk plain-IQ4_XS  IQ4_XS --imatrix "$IM"
mk plain-Q4_0    Q4_0   --imatrix "$IM"
mk plain-Q3_K_M  Q3_K_M --imatrix "$IM"
mk plain-Q2_K    Q2_K   --imatrix "$IM"

# ---- B. head-only experiments: body stays Q8_0, the tied head (token_embd) changes ------------
mk head-Q6_K-body-Q8_0   Q8_0 --imatrix "$IM" --output-tensor-type q6_k
mk head-Q5_K-body-Q8_0   Q8_0 --imatrix "$IM" --output-tensor-type q5_k
mk head-Q4_K-body-Q8_0   Q8_0 --imatrix "$IM" --output-tensor-type q4_k
mk head-IQ4_XS-body-Q8_0 Q8_0 --imatrix "$IM" --output-tensor-type iq4_xs
mk head-Q3_K-body-Q8_0   Q8_0 --imatrix "$IM" --output-tensor-type q3_k

# ---- C. protect-the-head experiments: body Q4_K_M, head raised --------------------------------
mk head-Q8_0-body-Q4_K_M Q4_K_M --imatrix "$IM" --output-tensor-type q8_0
mk head-BF16-body-Q4_K_M Q4_K_M --imatrix "$IM" --output-tensor-type bf16
mk head-Q4_K-body-Q4_K_M Q4_K_M --imatrix "$IM" --output-tensor-type q4_k   # default mix would give Q6_K head

# ---- D. which block matters: attention vs FFN, early vs late layers ----------------------------
mk attn-Q8_0-ffn-Q4_K    Q8_0 --imatrix "$IM" --tensor-type "ffn_(up|down|gate)=q4_k"
mk attn-Q4_K-ffn-Q8_0    Q8_0 --imatrix "$IM" --tensor-type "attn_(q|k|v|output)=q4_k"
mk ffn-down-Q8-rest-Q4_K Q4_K_M --imatrix "$IM" --output-tensor-type q8_0 --tensor-type "ffn_down=q8_0"
mk edge-Q8_0-mid-Q4_K    Q8_0 --imatrix "$IM" --tensor-type "blk\.([4-9]|1[0-9]|2[0-3])\.(attn_(q|k|v|output)|ffn_(up|down|gate))=q4_k"
echo ALL DONE
