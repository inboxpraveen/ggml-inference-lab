"""
A small inference driver on top of llama.dll, written to measure (not to replace llama-cli).

Three generation loops share one model/context:

  greedy_loop      one llama_decode per token, argmax over a zero-copy numpy view of the logits
  sampler_loop     same, but sampling done in C by llama.cpp's own sampler chain (control for Python overhead)
  ngram_spec_loop  self-speculative decoding: draft the next K tokens from the sequence's own n-gram history
                   (prompt lookup), verify all K+1 in ONE llama_decode, keep the accepted prefix, roll the
                   KV cache back with llama_memory_seq_rm. No draft model, no extra weights read.

Every loop reports end-to-end tok/s measured with a wall clock around the whole generation, including
tokenization of the output pieces, so the numbers are comparable with llama-completion's "eval time"
rather than llama-bench (which excludes sampling).
"""
import os, sys, time, ctypes
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llamabind import ffi, load

_quiet = True


@ffi.callback("void(int, const char *, void *)")
def _log_cb(level, text, ud):
    if not _quiet:
        sys.stderr.write(ffi.string(text).decode('utf-8', 'replace'))


class Engine:
    def __init__(self, model_path, bin_dir, n_gpu_layers=0, n_threads=8, n_threads_batch=None,
                 n_ctx=4096, n_batch=2048, flash_attn='auto', type_k=None, type_v=None, load_mode=None,
                 verbose=False):
        global _quiet
        _quiet = not verbose
        self.lib = lib = load(bin_dir)
        lib.llama_log_set(_log_cb, ffi.NULL)
        lib.llama_backend_init()

        mp = lib.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        if load_mode is not None:
            mp.load_mode = {'auto': -1, 'none': 0, 'mmap': 1, 'mlock': 2, 'mmap+mlock': 3}[load_mode]
        self.model = lib.llama_model_load_from_file(model_path.encode(), mp)
        if self.model == ffi.NULL:
            raise RuntimeError(f'failed to load {model_path}')
        self.vocab = lib.llama_model_get_vocab(self.model)
        self.n_vocab = lib.llama_vocab_n_tokens(self.vocab)

        cp = lib.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = n_batch
        cp.n_ubatch = min(512, n_batch)
        cp.n_threads = n_threads
        cp.n_threads_batch = n_threads_batch or n_threads
        cp.flash_attn_type = {'auto': -1, 'off': 0, 'on': 1}[flash_attn]
        if type_k: cp.type_k = {'f16': 1, 'q8_0': 8, 'q4_0': 2}[type_k]
        if type_v: cp.type_v = {'f16': 1, 'q8_0': 8, 'q4_0': 2}[type_v]
        cp.no_perf = False
        self.ctx = lib.llama_init_from_model(self.model, cp)
        if self.ctx == ffi.NULL:
            raise RuntimeError('failed to create context')
        self.mem = lib.llama_get_memory(self.ctx)
        self.n_ctx = lib.llama_n_ctx(self.ctx)
        # one reusable batch big enough for a prompt chunk or a draft
        self.batch = lib.llama_batch_init(n_batch, 0, 1)
        self._piece_buf = ffi.new('char[]', 256)

    # ---- tokenizer ------------------------------------------------------------------------------
    def tokenize(self, text, add_special=True, parse_special=True):
        b = text.encode('utf-8')
        n = len(b) + 16
        out = ffi.new('llama_token[]', n)
        r = self.lib.llama_tokenize(self.vocab, b, len(b), out, n, add_special, parse_special)
        if r < 0:
            out = ffi.new('llama_token[]', -r)
            r = self.lib.llama_tokenize(self.vocab, b, len(b), out, -r, add_special, parse_special)
        return [out[i] for i in range(r)]

    def piece(self, tok):
        n = self.lib.llama_token_to_piece(self.vocab, tok, self._piece_buf, 256, 0, True)
        return ffi.unpack(self._piece_buf, n) if n > 0 else b''

    def detokenize(self, toks):
        return b''.join(self.piece(t) for t in toks).decode('utf-8', 'replace')

    def apply_chat_template(self, user_text, enable_thinking=False):
        # Qwen3 ChatML; the empty <think> block is what the template emits when enable_thinking=false
        s = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        if not enable_thinking:
            s += "<think>\n\n</think>\n\n"
        return s

    # ---- low-level decode helpers ---------------------------------------------------------------
    def _fill(self, toks, pos0, want_logits):
        """want_logits: 'last' or 'all'"""
        b = self.batch
        n = len(toks)
        b.n_tokens = n
        for i, t in enumerate(toks):
            b.token[i] = t
            b.pos[i] = pos0 + i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = 1 if (want_logits == 'all' or i == n - 1) else 0
        return b

    def _decode(self, toks, pos0, want_logits='last'):
        nb = int(self.lib.llama_n_batch(self.ctx))
        for i in range(0, len(toks), nb):
            chunk = toks[i:i + nb]
            wl = want_logits if i + nb >= len(toks) else ('all' if want_logits == 'all' else 'last')
            r = self.lib.llama_decode(self.ctx, self._fill(chunk, pos0 + i, wl))
            if r != 0:
                raise RuntimeError(f'llama_decode returned {r}')

    def logits_view(self, i):
        """Zero-copy float32 view of the logits for output row i. Valid until the next llama_decode."""
        p = self.lib.llama_get_logits_ith(self.ctx, i)
        return np.frombuffer(ffi.buffer(p, self.n_vocab * 4), dtype=np.float32)

    def clear(self):
        self.lib.llama_memory_clear(self.mem, True)

    def is_eog(self, t):
        return self.lib.llama_vocab_is_eog(self.vocab, t)

    # ---- loop 1: plain greedy ---------------------------------------------------------------------
    def greedy_loop(self, prompt_toks, n_predict, stop_at_eog=True):
        self.clear()
        t0 = time.perf_counter()
        self._decode(prompt_toks, 0, 'last')
        t1 = time.perf_counter()
        out = []
        n_past = len(prompt_toks)
        tok = int(np.argmax(self.logits_view(-1)))
        while len(out) < n_predict:
            out.append(tok)
            if stop_at_eog and self.is_eog(tok):
                break
            self._decode([tok], n_past, 'last')
            n_past += 1
            tok = int(np.argmax(self.logits_view(-1)))
        t2 = time.perf_counter()
        return dict(tokens=out, n_prompt=len(prompt_toks), prompt_s=t1 - t0, gen_s=t2 - t1,
                    n_gen=len(out), decode_calls=len(out), tok_s=len(out) / (t2 - t1))

    # ---- loop 2: llama.cpp's own sampler (C-side argmax) -------------------------------------------
    def sampler_loop(self, prompt_toks, n_predict, stop_at_eog=True):
        lib = self.lib
        sp = lib.llama_sampler_chain_default_params(); sp.no_perf = True
        smpl = lib.llama_sampler_chain_init(sp)
        lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy())
        self.clear()
        t0 = time.perf_counter()
        self._decode(prompt_toks, 0, 'last')
        t1 = time.perf_counter()
        out = []; n_past = len(prompt_toks)
        tok = lib.llama_sampler_sample(smpl, self.ctx, -1)
        while len(out) < n_predict:
            out.append(tok)
            if stop_at_eog and self.is_eog(tok):
                break
            self._decode([tok], n_past, 'last'); n_past += 1
            tok = lib.llama_sampler_sample(smpl, self.ctx, -1)
        t2 = time.perf_counter()
        lib.llama_sampler_free(smpl)
        return dict(tokens=out, n_prompt=len(prompt_toks), prompt_s=t1 - t0, gen_s=t2 - t1,
                    n_gen=len(out), decode_calls=len(out), tok_s=len(out) / (t2 - t1))

    # ---- loop 3: self-speculative n-gram (prompt lookup) decoding ------------------------------------
    def ngram_spec_loop(self, prompt_toks, n_predict, n_gram=3, n_draft=8, min_seq=0, stop_at_eog=True):
        """
        Draft source is the sequence itself (prompt + everything generated so far). After each accepted
        token we look for the last `n_gram` tokens earlier in the sequence; if found, the tokens that
        followed that earlier occurrence become the draft (up to n_draft). The batch [tok, d1..dK] is
        decoded once with logits for every row; row i's argmax is the model's answer given d1..di.
        Accept while the argmax agrees with the draft. Greedy target means the output is bit-identical
        to greedy_loop, which is the property we verify in the benchmark script.
        """
        self.clear()
        t0 = time.perf_counter()
        self._decode(prompt_toks, 0, 'last')
        t1 = time.perf_counter()
        seq = list(prompt_toks)
        out = []
        n_past = len(prompt_toks)
        tok = int(np.argmax(self.logits_view(-1)))
        calls = 0; drafted = 0; accepted = 0
        # index: n-gram -> list of positions where it occurs (position of the token AFTER the n-gram)
        index = {}
        def add_index(upto):
            # index n-grams ending before `upto`
            # i is the position AFTER an n-gram; i == upto is the n-gram that ends at the sequence end
            for i in range(max(n_gram, add_index.done), upto + 1):
                key = tuple(seq[i - n_gram:i])
                index.setdefault(key, []).append(i)
            add_index.done = max(add_index.done, upto + 1)
        add_index.done = n_gram
        add_index(len(seq))

        while len(out) < n_predict:
            out.append(tok); seq.append(tok)
            if stop_at_eog and self.is_eog(tok):
                break
            add_index(len(seq))
            # draft from the most recent earlier occurrence of the last n_gram tokens
            key = tuple(seq[-n_gram:])
            draft = []
            cands = index.get(key)
            if cands and len(cands) > 1:
                # cands[-1] is the current position itself (just indexed); use the previous occurrence
                j = cands[-2]
                draft = seq[j:j + n_draft]
            if not draft:
                self._decode([tok], n_past, 'last'); n_past += 1; calls += 1
                tok = int(np.argmax(self.logits_view(-1)))
                continue
            batch = [tok] + draft
            self._decode(batch, n_past, 'all'); calls += 1; drafted += len(draft)
            # row i logits -> prediction for position after batch[i]
            n_acc = 0
            nxt = None
            for i in range(len(batch)):
                pred = int(np.argmax(self.logits_view(i)))
                if i < len(draft) and pred == draft[i] and len(out) + n_acc + 1 < n_predict:
                    n_acc += 1
                    if stop_at_eog and self.is_eog(pred):
                        nxt = pred; break
                else:
                    nxt = pred; break
            # accepted draft tokens become real output tokens
            for i in range(n_acc):
                out.append(draft[i]); seq.append(draft[i])
            accepted += n_acc
            add_index(len(seq))
            keep = n_past + 1 + n_acc                 # tok + accepted drafts are valid KV entries
            self.lib.llama_memory_seq_rm(self.mem, 0, keep, -1)
            n_past = keep
            tok = nxt
            if stop_at_eog and n_acc and self.is_eog(seq[-1]):
                break
        t2 = time.perf_counter()
        return dict(tokens=out, n_prompt=len(prompt_toks), prompt_s=t1 - t0, gen_s=t2 - t1,
                    n_gen=len(out), decode_calls=calls, drafted=drafted, accepted=accepted,
                    accept_rate=(accepted / drafted if drafted else 0.0), tok_s=len(out) / (t2 - t1))

    def close(self):
        self.lib.llama_batch_free(self.batch)
        self.lib.llama_free(self.ctx)
        self.lib.llama_model_free(self.model)
