"""Supervised fine-tuning on Hindi instruction pairs in ChatML, loss on the assistant tokens only.

  python scripts/adapt/sft.py --model models/adapt/main --tag sft --epochs 2 --lr 1e-4
"""
import os, json, math, time, argparse, random
import torch
from common import DATA, RES, MODELS, log_row
from chat import chat_prompt, chat_full
from eval_bpb import load_docs, evaluate
from train_utils import MasterWeights, chunked_ce
import fastattn; fastattn.install()
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--epochs', type=float, default=2)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--min-lr', type=float, default=1e-5)
ap.add_argument('--micro', type=int, default=8)
ap.add_argument('--accum', type=int, default=2)
ap.add_argument('--max-len', type=int, default=640)
ap.add_argument('--n-train', type=int, default=12000)
ap.add_argument('--warmup', type=int, default=30)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--cold-start', type=int, default=0, help='mix in N sentiment examples in the JSON answer format')
args = ap.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed)
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation='sdpa').to(dev)
model.config.use_cache = False
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
mw = MasterWeights(model, lr=args.lr, wd=0.1)
head = model.get_output_embeddings()


def encode(rows):
    out = []
    for r in rows:
        p_ids = tok(chat_prompt(r['prompt']), add_special_tokens=False)['input_ids']
        f_ids = tok(chat_full(r['prompt'], r['response']), add_special_tokens=False)['input_ids']
        if len(f_ids) > args.max_len:
            continue
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        out.append((f_ids, labels))
    return out


rows = [json.loads(l) for l in open(os.path.join(DATA, 'sft_train.jsonl'), encoding='utf-8')][:args.n_train]
if args.cold_start:
    from chat import SENT_INSTR
    sent = [json.loads(l) for l in open(os.path.join(DATA, 'sent_train.jsonl'), encoding='utf-8')][:args.cold_start]
    rows += [{'prompt': SENT_INSTR + r['review'], 'response': json.dumps({'label': r['label']})} for r in sent]
train = encode(rows)
test = encode([json.loads(l) for l in open(os.path.join(DATA, 'sft_test.jsonl'), encoding='utf-8')])
print(f'{len(train)} train, {len(test)} test examples after length filter')


def collate(batch):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return ids.to(dev), lab.to(dev), att.to(dev)


def loss_on(ids, lab, att):
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    return chunked_ce(head, h, lab[:, 1:])


@torch.no_grad()
def eval_loss():
    model.eval(); tot = n = 0
    for i in range(0, len(test), 16):
        ids, lab, att = collate(test[i:i + 16])
        k = (lab[:, 1:] != -100).sum().item()
        tot += loss_on(ids, lab, att).item() * k; n += k
    model.train()
    return tot / n


hi_eval = load_docs('hi_test.jsonl', 150, 1500); en_eval = load_docs('en_test.jsonl', 150, 1500)


def drift():
    model.eval()
    r = evaluate(model, tok, hi_eval, en_eval, 4096)
    model.train()
    return {'hi_bpc': r['hi']['bpc'], 'en_bpc': r['en']['bpc']}


run = f'sft-{args.tag}'
steps_per_epoch = len(train) // (args.micro * args.accum)
total_steps = int(steps_per_epoch * args.epochs)


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    p = (s - args.warmup) / max(1, total_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))


log_row(run, {'kind': 'config', **vars(args), 'n_train': len(train), 'total_steps': total_steps})
t0 = time.time()
e0 = eval_loss(); d0 = drift()
log_row(run, {'kind': 'eval', 'step': 0, 'test_loss': e0, **d0})
print(f'step 0: test loss {e0:.3f} hi bpc {d0["hi_bpc"]:.3f} en bpc {d0["en_bpc"]:.3f}')
model.train(); order = []; ema = None
for step in range(total_steps):
    lr = lr_at(step)
    mw.set_lr(lr)
    tot = 0.0
    for _ in range(args.accum):
        if len(order) < args.micro:
            order = list(range(len(train))); random.shuffle(order)
        idx = [order.pop() for _ in range(args.micro)]
        loss = loss_on(*collate([train[i] for i in idx]))
        (loss / args.accum).backward(); tot += loss.item() / args.accum
    gn = mw.step()
    ema = tot if ema is None else 0.95 * ema + 0.05 * tot
    if step % 10 == 0:
        log_row(run, {'kind': 'train', 'step': step + 1, 'loss': tot, 'loss_ema': ema, 'lr': lr, 'grad_norm': gn, 'elapsed_s': time.time() - t0})
    if step % 50 == 0:
        print(f'step {step+1}/{total_steps} loss {tot:.3f} ema {ema:.3f} lr {lr:.1e} {(time.time()-t0)/60:.1f} min', flush=True)
    if (step + 1) % (steps_per_epoch // 2) == 0 or step == total_steps - 1:
        e = eval_loss(); d = drift()
        log_row(run, {'kind': 'eval', 'step': step + 1, 'test_loss': e, **d, 'loss_ema': ema})
        print(f'  eval step {step+1}: test loss {e:.3f} hi bpc {d["hi_bpc"]:.3f} en bpc {d["en_bpc"]:.3f}', flush=True)

out = os.path.join(MODELS, args.tag)
model.config.use_cache = True
model.save_pretrained(out); tok.save_pretrained(out)
log_row(run, {'kind': 'done', 'steps': total_steps, 'elapsed_s': time.time() - t0, 'out': out})
print('saved', out)
