#!/usr/bin/env bash
# Small GGUFs, one per architecture, for the follow-up post. All Q8_0 so bytes-per-token comparisons share a format.
set -e
cd "$(dirname "$0")/../models/zoo"
get() { [ -f "$2" ] || curl -L --retry 3 -o "$2" "https://huggingface.co/$1/resolve/main/$2"; }
get bartowski/granite-3.1-1b-a400m-instruct-GGUF granite-3.1-1b-a400m-instruct-Q8_0.gguf   # MoE, 32 experts, 8 active
get ggml-org/gemma-3-1b-it-GGUF                    gemma-3-1b-it-Q8_0.gguf                    # sliding-window local + global attention
get tiiuae/Falcon-H1-0.5B-Instruct-GGUF            Falcon-H1-0.5B-Instruct-Q8_0.gguf          # hybrid Mamba2 + attention
get LiquidAI/LFM2-700M-GGUF                        LFM2-700M-Q8_0.gguf                        # hybrid short-conv + attention
get QuantFactory/mamba-130m-hf-GGUF                mamba-130m-hf.Q8_0.gguf                    # pure SSM
get Felladrin/gguf-flan-t5-small                   flan-t5-small.Q8_0.gguf                    # encoder-decoder
get nomic-ai/nomic-embed-text-v1.5-GGUF            nomic-embed-text-v1.5.Q8_0.gguf            # encoder-only, embeddings
ls -la
