"""Bits per byte (and per character) of a causal LM on the Hindi and English held-out sets.

Per-token perplexity is not comparable across tokenizers, and this post changes the tokenizer halfway
through, so every quality number is total negative log-likelihood divided by the UTF-8 bytes it covered.
Each document is cut to its first --chars characters, a BOS token is prepended so every text token is
scored, and the token sequence is capped at --max-tokens (the byte count is taken from the decoded tokens
that were actually scored).

  python scripts/adapt/eval_bpb.py HuggingFaceTB/SmolLM2-135M [more models...] [--docs 400] [--tag name]
  python scripts/adapt/eval_bpb.py models/adapt/cpt-main --tokenizer models/adapt/tok-16k
"""
import os, sys, json, math, argparse, time
import torch
from common import DATA, RES, log_row
import fastattn; fastattn.install()
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_docs(name, n, chars):
    docs = []
    with open(os.path.join(DATA, name), encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line)['text'][:chars])
            if len(docs) >= n:
                break
    return docs


@torch.no_grad()
def bpb(model, tok, docs, max_tokens, device='cuda', chunk=1024):
    """One document at a time, logits upcast in chunks of 1024 positions, so a 150k-entry vocabulary at
    4096 tokens does not need gigabytes of fp32 logits."""
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    nll = 0.0; n_bytes = 0; n_chars = 0; n_tok = 0
    for d in docs:
        ids = tok(d, add_special_tokens=False)['input_ids'][:max_tokens - 1]
        text = tok.decode(ids)
        x = torch.tensor([[bos] + ids], device=device)
        hidden = model.model(input_ids=x).last_hidden_state[0, :-1]
        tgt = x[0, 1:]
        head = model.get_output_embeddings()
        for i in range(0, hidden.shape[0], chunk):
            logits = head(hidden[i:i + chunk]).float()
            nll += torch.nn.functional.cross_entropy(logits, tgt[i:i + chunk], reduction='sum').item()
        n_bytes += len(text.encode('utf-8')); n_chars += len(text); n_tok += len(ids)
    return {'bpb': nll / math.log(2) / n_bytes, 'bpc': nll / math.log(2) / n_chars,
            'nll_per_token': nll / n_tok, 'tokens': n_tok, 'bytes': n_bytes, 'chars': n_chars, 'docs': len(docs)}


def evaluate(model, tok, docs_hi, docs_en, max_tokens):
    return {'hi': bpb(model, tok, docs_hi, max_tokens), 'en': bpb(model, tok, docs_en, max_tokens)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--docs', type=int, default=400)
    ap.add_argument('--chars', type=int, default=1500)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--dtype', default='bf16')
    args = ap.parse_args()
    hi = load_docs('hi_test.jsonl', args.docs, args.chars); en = load_docs('en_test.jsonl', args.docs, args.chars)
    dt = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    for path in args.models:
        tok = AutoTokenizer.from_pretrained(args.tokenizer or path)
        t = time.time()
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).cuda().eval()
        n_params = sum(p.numel() for p in model.parameters())
        r = evaluate(model, tok, hi, en, args.max_tokens)
        row = {'model': path, 'tag': args.tag or os.path.basename(path.rstrip('/\\')), 'params': n_params,
               'tokenizer': args.tokenizer or path, 'vocab': len(tok), 'chars_per_doc': args.chars, **r,
               'seconds': time.time() - t}
        print(f"{row['tag']:28s} {n_params/1e6:7.0f}M  hi bpb {r['hi']['bpb']:.3f} (bpc {r['hi']['bpc']:.3f}, "
              f"{r['hi']['tokens']} tok)  en bpb {r['en']['bpb']:.3f} (bpc {r['en']['bpc']:.3f})  {row['seconds']:.0f}s")
        log_row('bpb', row)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
