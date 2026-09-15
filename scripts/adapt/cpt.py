"""Continued pretraining of SmolLM2-135M on Hindi (plus an English replay share) on one GPU.

Plain PyTorch loop: AdamW, linear warmup then cosine, bf16 weights and activations with an fp32 master copy
that the optimizer updates (the 8 GB card cannot hold fp32 activations at a useful batch), packed sequences,
and the cross-entropy computed in 512-position chunks that are recomputed in the backward pass so the
65k-wide logits never exist all at once. Every --eval-every steps it scores bits per byte on fixed Hindi and English held-out samples and
appends a row to results/adapt/cpt-<tag>.jsonl; the model and tokenizer land in models/adapt/<tag>/.

  python scripts/adapt/cpt.py --tag main --tokenizer models/adapt/tok-16k --init mean --steps 3600 --en-ratio 0.1
  python scripts/adapt/cpt.py --tag base-tok --steps 400            # stock tokenizer, no new rows
  python scripts/adapt/cpt.py --tag init-random --tokenizer models/adapt/tok-16k --init random --steps 400
"""
import os, sys, json, math, time, argparse, random
import numpy as np
import torch
from common import DATA, RES, MODELS, BASE_MODEL, log_row, gpu_state
from train_utils import MasterWeights, chunked_ce
import fastattn; fastattn.install()
from eval_bpb import load_docs, evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--tokenizer', default=BASE_MODEL)
ap.add_argument('--model', default=BASE_MODEL)
ap.add_argument('--init', default='mean', choices=['mean', 'random', 'hf'])
ap.add_argument('--steps', type=int, default=400)
ap.add_argument('--seq', type=int, default=1024)
ap.add_argument('--micro', type=int, default=4)
ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--min-lr', type=float, default=5e-5)
ap.add_argument('--warmup', type=int, default=50)
ap.add_argument('--wd', type=float, default=0.1)
ap.add_argument('--en-ratio', type=float, default=0.1)
ap.add_argument('--eval-every', type=int, default=100)
ap.add_argument('--eval-docs', type=int, default=150)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--save', action='store_true')
ap.add_argument('--max-hi-chars', type=int, default=0, help='cap the Hindi text used (0 = all of hi_train.jsonl)')
args = ap.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(args.tokenizer)
tok_name = os.path.basename(args.tokenizer.rstrip('/\\'))
eos = tok.eos_token_id


def token_cache(name, max_chars=0):
    """Tokenize data/adapt/<name>.jsonl once per tokenizer into a flat uint32 array (docs joined by EOS)."""
    cache_dir = os.path.join(DATA, 'cache'); os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f'{tok_name}-{name}{"-" + str(max_chars) if max_chars else ""}.npy')
    if os.path.exists(p):
        return np.load(p, mmap_mode='r')
    ids, chars, buf, t = [], 0, [], time.time()
    with open(os.path.join(DATA, name + '.jsonl'), encoding='utf-8') as f:
        for line in f:
            txt = json.loads(line)['text']
            buf.append(txt); chars += len(txt)
            if len(buf) == 512:
                for enc in tok(buf, add_special_tokens=False)['input_ids']:
                    ids.extend(enc); ids.append(eos)
                buf = []
            if max_chars and chars >= max_chars:
                break
    for enc in tok(buf, add_special_tokens=False)['input_ids']:
        ids.extend(enc); ids.append(eos)
    arr = np.array(ids, dtype=np.uint32)
    np.save(p, arr)
    print(f'tokenized {name}: {chars/1e6:.0f}M chars -> {len(arr)/1e6:.1f}M tokens ({chars/len(arr):.2f} chars/token) in {time.time()-t:.0f}s')
    with open(p + '.json', 'w') as f:
        json.dump({'chars': chars, 'tokens': int(len(arr))}, f)
    return arr


hi = token_cache('hi_train', args.max_hi_chars)
en = token_cache('en_train')
hi_meta = json.load(open(os.path.join(DATA, 'cache', f'{tok_name}-hi_train{"-" + str(args.max_hi_chars) if args.max_hi_chars else ""}.npy.json')))
hi_cpt = hi_meta['chars'] / hi_meta['tokens']  # chars per token, to report characters seen
print(f'hindi {len(hi)/1e6:.1f}M tokens, english {len(en)/1e6:.1f}M tokens; tokens per step {args.micro*args.accum*args.seq}')


class Packed:
    """Sequential windows over a flat token array, shuffled once per epoch."""
    def __init__(self, arr, seq, seed):
        self.arr, self.seq = arr, seq
        self.n = (len(arr) - 1) // seq
        self.rng = random.Random(seed)
        self.order, self.i = [], 0

    def next(self):
        if self.i >= len(self.order):
            self.order = list(range(self.n)); self.rng.shuffle(self.order); self.i = 0
        k = self.order[self.i]; self.i += 1
        w = self.arr[k * self.seq: k * self.seq + self.seq + 1]
        return torch.from_numpy(w.astype(np.int64))


hi_stream, en_stream = Packed(hi, args.seq, args.seed), Packed(en, args.seq, args.seed + 1)
mix_rng = random.Random(args.seed + 2)


def batch():
    rows, n_hi = [], 0
    for _ in range(args.micro):
        if mix_rng.random() < args.en_ratio:
            rows.append(en_stream.next())
        else:
            rows.append(hi_stream.next()); n_hi += 1
    x = torch.stack(rows).to(dev)
    return x[:, :-1], x[:, 1:], n_hi


# model, with new embedding rows if the tokenizer grew
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation='sdpa')
old_vocab = model.get_input_embeddings().weight.shape[0]
n_new = len(tok) - old_vocab
if n_new > 0:
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    E = model.get_input_embeddings().weight.data
    added = json.load(open(os.path.join(args.tokenizer, 'added_tokens_list.json'), encoding='utf-8'))
    assert len(added) == n_new, (len(added), n_new)
    if args.init == 'hf':
        model.resize_token_embeddings(len(tok), mean_resizing=True)
    else:
        model.resize_token_embeddings(len(tok), mean_resizing=False)
        E = model.get_input_embeddings().weight.data
        if args.init == 'random':
            E[old_vocab:].normal_(0, 0.02)
        else:  # mean of the stock sub-token embeddings of each new token
            # Tokenize the byte-level string itself (not its decoded text: byte-level merges can cross UTF-8
            # character boundaries, and those tokens decode to U+FFFD).
            bpe = base_tok.backend_tokenizer.model
            n_multi = 0
            for i, s in enumerate(added):
                ids = [t.id for t in bpe.tokenize(s)]
                n_multi += len(ids) > 1
                E[old_vocab + i] = E[ids].mean(0)
            print(f'mean init: {n_multi} of {len(added)} new tokens average more than one stock token')
    print(f'added {n_new} embedding rows, init={args.init}, tied={model.config.tie_word_embeddings}')
model.config.use_cache = False
model.to(dev).train()
n_params = sum(p.numel() for p in model.parameters())

# fp32 master copy for the optimizer; the model itself runs in bf16
mw = MasterWeights(model, lr=args.lr, wd=args.wd)
head = model.get_output_embeddings()


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))


hi_eval = load_docs('hi_test.jsonl', args.eval_docs, 1500)
en_eval = load_docs('en_test.jsonl', args.eval_docs, 1500)
run = f'cpt-{args.tag}'
cfg = {k: v for k, v in vars(args).items()}
cfg.update({'params': n_params, 'vocab': len(tok), 'tokens_per_step': args.micro * args.accum * args.seq,
            'hi_chars_per_token': hi_cpt, 'hi_tokens_available': int(len(hi)), 'en_tokens_available': int(len(en))})


def do_eval(step, tokens_seen, hi_tokens_seen, extra=None):
    model.eval()
    r = evaluate(model, tok, hi_eval, en_eval, 4096)
    model.train()
    row = {'kind': 'eval', 'step': step, 'tokens': tokens_seen, 'hi_tokens': hi_tokens_seen,
           'hi_chars': hi_tokens_seen * hi_cpt, 'hi_bpb': r['hi']['bpb'], 'en_bpb': r['en']['bpb'],
           'hi_bpc': r['hi']['bpc'], 'en_bpc': r['en']['bpc'], 'elapsed_s': time.time() - t0}
    if extra:
        row.update(extra)
    log_row(run, row)
    print(f"  eval step {step}: hi bpb {r['hi']['bpb']:.3f}  en bpb {r['en']['bpb']:.3f}  ({tokens_seen/1e6:.1f}M tokens, {row['elapsed_s']/60:.1f} min)", flush=True)
    return r


log_row(run, {'kind': 'config', **cfg})
t0 = time.time()
tokens_seen = hi_tokens_seen = 0
do_eval(0, 0, 0)
loss_ema = None
for step in range(args.steps):
    lr = lr_at(step)
    mw.set_lr(lr)
    total = 0.0; total_hi = 0
    for _ in range(args.accum):
        x, y, n_hi = batch()
        h = model.model(input_ids=x).last_hidden_state
        loss = chunked_ce(head, h, y)
        (loss / args.accum).backward()
        total += loss.item() / args.accum; total_hi += n_hi
    gn = mw.step()
    tokens_seen += args.micro * args.accum * args.seq
    hi_tokens_seen += total_hi * args.seq
    loss_ema = total if loss_ema is None else 0.98 * loss_ema + 0.02 * total
    if step % 10 == 0 or step == args.steps - 1:
        el = time.time() - t0
        log_row(run, {'kind': 'train', 'step': step + 1, 'loss': total, 'loss_ema': loss_ema, 'lr': lr, 'grad_norm': gn,
                      'tokens': tokens_seen, 'hi_tokens': hi_tokens_seen, 'elapsed_s': el, 'tok_per_s': tokens_seen / el})
        if step % 50 == 0:
            print(f'step {step+1}/{args.steps} loss {total:.3f} ema {loss_ema:.3f} lr {lr:.2e} gn {gn:.2f} '
                  f'{tokens_seen/el:.0f} tok/s  {el/60:.1f} min  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB', flush=True)
    if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
        do_eval(step + 1, tokens_seen, hi_tokens_seen, {'loss_ema': loss_ema})

if args.save:
    out = os.path.join(MODELS, args.tag)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print('saved', out)
log_row(run, {'kind': 'done', 'steps': args.steps, 'tokens': tokens_seen, 'hi_tokens': hi_tokens_seen,
              'elapsed_s': time.time() - t0, 'max_mem_gb': torch.cuda.max_memory_allocated() / 1e9})
