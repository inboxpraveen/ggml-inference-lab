"""GRPO with verifiable rewards, written out in full so every line can be read.

Per step: sample B prompts (sentiment reviews with a label, or open Hindi prompts with no reference), draw G
completions each from the current policy, score them with the rule-based rewards in chat.py, normalise the
rewards inside each group into advantages, and take one gradient step on
    loss = -(1/N_tokens) * sum_over_completion_tokens( advantage * log pi(token) )  [+ beta * KL(pi || ref)]
One optimizer step per batch of samples means the policy that generated the samples is the policy being
updated, so no probability ratio or clipping is needed (that machinery exists for reusing samples).

  python scripts/adapt/grpo.py --model models/adapt/sft --tag grpo --steps 300
"""
import os, json, math, time, argparse, random
import torch
from common import DATA, RES, MODELS, log_row
from chat import chat_prompt, sentiment_prompt, reward_sentiment, reward_language, reward_language_v2, parse_label
from eval_task import eval_sentiment, eval_language
from train_utils import MasterWeights, chunked_logprobs
import fastattn; fastattn.install()
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--prompts', type=int, default=4, help='prompts per step (B)')
ap.add_argument('--group', type=int, default=8, help='completions per prompt (G)')
ap.add_argument('--p-sent', type=float, default=0.7, help='share of prompts that are sentiment tasks')
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--beta', type=float, default=0.0, help='KL penalty to the starting policy')
ap.add_argument('--reward', default='v1', choices=['v1', 'v2'], help='language reward: v1 (share x length x distinct) or v2 (v1 x loop penalty)')
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--max-new', type=int, default=96)
ap.add_argument('--eval-every', type=int, default=50)
ap.add_argument('--eval-n', type=int, default=200)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
lang_reward = reward_language_v2 if args.reward == 'v2' else reward_language
torch.manual_seed(args.seed); random.seed(args.seed)
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = 'left'
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
im_end = tok.convert_tokens_to_ids('<|im_end|>')
policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation='sdpa').to(dev)
policy.config.use_cache = False
mw = MasterWeights(policy, lr=args.lr, wd=0.0, betas=(0.9, 0.99))
# the starting policy is always loaded so the KL from it can be logged, even when it is not penalised
ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
for p in ref.parameters():
    p.requires_grad_(False)

sent_train = [json.loads(l) for l in open(os.path.join(DATA, 'sent_train.jsonl'), encoding='utf-8')]
open_train = [json.loads(l)['prompt'] for l in open(os.path.join(DATA, 'open_prompts.jsonl'), encoding='utf-8')]
sent_test = [json.loads(l) for l in open(os.path.join(DATA, 'sent_test.jsonl'), encoding='utf-8')]
open_test = [json.loads(l)['prompt'] for l in open(os.path.join(DATA, 'sft_test.jsonl'), encoding='utf-8')][:100]


def sample_prompts():
    items = []
    for _ in range(args.prompts):
        if random.random() < args.p_sent:
            r = random.choice(sent_train)
            items.append({'task': 'sent', 'text': sentiment_prompt(r['review']), 'label': r['label']})
        else:
            items.append({'task': 'open', 'text': chat_prompt(random.choice(open_train))})
    return items


@torch.no_grad()
def generate(items):
    """G completions per prompt, all in one batched generate call. Returns token ids and decoded text."""
    policy.eval()
    texts = [it['text'] for it in items for _ in range(args.group)]
    enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(dev)
    out = policy.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                          max_new_tokens=args.max_new, eos_token_id=[im_end, tok.eos_token_id],
                          pad_token_id=tok.pad_token_id, use_cache=True)
    policy.train()
    P = enc['input_ids'].shape[1]
    comps = out[:, P:]
    comp_ids, comp_text = [], []
    for row in comps:
        ids = row.tolist()
        cut = len(ids)
        for j, t in enumerate(ids):
            if t in (im_end, tok.eos_token_id, tok.pad_token_id):
                cut = j + 1 if t == im_end else j  # keep <|im_end|> as a learned token, drop padding
                break
        ids = ids[:cut]
        comp_ids.append(ids); comp_text.append(tok.decode(ids, skip_special_tokens=True))
    return enc['input_ids'], enc['attention_mask'], comp_ids, comp_text


def logprobs_of(model, prompt_ids, prompt_mask, comp_ids):
    """Per-token log-probabilities of the completion tokens under model, batched with right padding."""
    B = prompt_ids.shape[0]
    L = max(len(c) for c in comp_ids)
    full = torch.full((B, prompt_ids.shape[1] + L), tok.pad_token_id, dtype=torch.long, device=dev)
    mask = torch.zeros_like(full)
    cmask = torch.zeros((B, L), dtype=torch.float32, device=dev)
    full[:, :prompt_ids.shape[1]] = prompt_ids; mask[:, :prompt_ids.shape[1]] = prompt_mask
    for i, c in enumerate(comp_ids):
        if c:
            full[i, prompt_ids.shape[1]:prompt_ids.shape[1] + len(c)] = torch.tensor(c, device=dev)
            mask[i, prompt_ids.shape[1]:prompt_ids.shape[1] + len(c)] = 1; cmask[i, :len(c)] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    h = model.model(input_ids=full, attention_mask=mask, position_ids=pos).last_hidden_state[:, prompt_ids.shape[1] - 1:-1]
    tgt = full[:, prompt_ids.shape[1]:]
    lp = chunked_logprobs(model.get_output_embeddings(), h, tgt)
    return lp, cmask


run = f'grpo-{args.tag}'
log_row(run, {'kind': 'config', **vars(args)})
t0 = time.time()


def do_eval(step):
    policy.eval()
    s = eval_sentiment(policy, tok, sent_test[:args.eval_n], chat_fn=sentiment_prompt)
    l = eval_language(policy, tok, open_test, chat_fn=chat_prompt)
    policy.train()
    row = {'kind': 'eval', 'step': step, 'sent_acc': s['accuracy'], 'sent_format': s['format_rate'],
           'lang_reward': l['mean_reward'], 'lang_reward_v2': l['mean_reward_v2'], 'lang_rep3': l['mean_rep3'],
           'lang_devanagari': l['mean_devanagari'], 'lang_words': l['mean_words'],
           'elapsed_s': time.time() - t0}
    log_row(run, row)
    print(f"  eval step {step}: sentiment acc {s['accuracy']:.3f} format {s['format_rate']:.3f} | language reward {l['mean_reward']:.3f} v2 {l['mean_reward_v2']:.3f} "
          f"devanagari {l['mean_devanagari']:.3f} words {l['mean_words']:.0f}  ({row['elapsed_s']/60:.1f} min)", flush=True)


do_eval(0)
policy.train()
for step in range(args.steps):
    items = sample_prompts()
    p_ids, p_mask, comp_ids, comp_text = generate(items)
    rewards, tasks = [], []
    for i, it in enumerate(items):
        for g in range(args.group):
            c = comp_text[i * args.group + g]
            r = reward_sentiment(c, it['label']) if it['task'] == 'sent' else lang_reward(c)
            rewards.append(r); tasks.append(it['task'])
    R = torch.tensor(rewards, device=dev).view(args.prompts, args.group)
    adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)
    adv = adv.view(-1)
    lp, cmask = logprobs_of(policy, p_ids, p_mask, comp_ids)
    n_tok = cmask.sum().clamp(min=1)
    pg = -(adv.unsqueeze(1) * lp * cmask).sum() / n_tok
    loss = pg
    with torch.no_grad():
        lp_ref, _ = logprobs_of(ref, p_ids, p_mask, comp_ids)
    d = lp_ref - lp
    kl = (torch.exp(d) - d - 1)  # k3 estimator, always >= 0
    kl_val = ((kl * cmask).sum() / n_tok).item()
    if args.beta > 0:
        loss = loss + args.beta * (kl * cmask).sum() / n_tok
    loss.backward()
    gn = mw.step()
    sent_r = [r for r, t in zip(rewards, tasks) if t == 'sent']; open_r = [r for r, t in zip(rewards, tasks) if t == 'open']
    fmt = [parse_label(c) is not None for c, t in zip(comp_text, tasks) if t == 'sent']
    row = {'kind': 'train', 'step': step + 1, 'loss': pg.item(), 'kl': kl_val, 'grad_norm': gn,
           'reward_mean': float(sum(rewards) / len(rewards)),
           'sent_reward': float(sum(sent_r) / len(sent_r)) if sent_r else None,
           'sent_format': float(sum(fmt) / len(fmt)) if fmt else None,
           'open_reward': float(sum(open_r) / len(open_r)) if open_r else None,
           'zero_var_groups': int((R.std(1) < 1e-6).sum().item()),
           'comp_len': float(sum(len(c) for c in comp_ids) / len(comp_ids)), 'elapsed_s': time.time() - t0}
    log_row(run, row)
    if step % 10 == 0:
        print(f"step {step+1}/{args.steps} reward {row['reward_mean']:.3f} sent {row['sent_reward']} fmt {row['sent_format']} open {row['open_reward']} "
              f"len {row['comp_len']:.0f} kl {kl_val:.4f} gn {gn:.2f} {(time.time()-t0)/60:.1f} min", flush=True)
    if step == 0 or (step + 1) % 100 == 0:
        with open(os.path.join(RES, f'{run}-samples.jsonl'), 'a', encoding='utf-8') as f:
            for i, it in enumerate(items[:2]):
                for g in range(min(3, args.group)):
                    k = i * args.group + g
                    f.write(json.dumps({'step': step + 1, 'task': it['task'], 'label': it.get('label'), 'prompt': it['text'][-200:],
                                        'completion': comp_text[k], 'reward': rewards[k]}, ensure_ascii=False) + '\n')
    if (step + 1) % args.eval_every == 0:
        do_eval(step + 1)

out = os.path.join(MODELS, args.tag)
policy.config.use_cache = True
policy.save_pretrained(out); tok.save_pretrained(out)
log_row(run, {'kind': 'done', 'steps': args.steps, 'elapsed_s': time.time() - t0, 'out': out})
print('saved', out)
