"""Task evaluation: Hindi sentiment (accuracy and JSON format rate on the held-out reviews) and the
language reward on held-out open prompts. Works for our checkpoints (ChatML) and for stock instruct models
(their own chat template).

  python scripts/adapt/eval_task.py models/adapt/grpo --tag grpo
  python scripts/adapt/eval_task.py HuggingFaceTB/SmolLM2-1.7B-Instruct Qwen/Qwen2.5-1.5B-Instruct --template
"""
import os, sys, json, argparse, time
import torch
from common import DATA, RES, log_row, devanagari_share
from chat import reward_language_v2, repeated_trigram_share, SENT_INSTR, chat_prompt, sentiment_prompt, parse_label, reward_language
import fastattn; fastattn.install()
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def generate_greedy(model, tok, prompts, max_new=96, batch=16, sample=False):
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    stops = [tok.eos_token_id]
    for t in ('<|im_end|>', '<end_of_turn>', '<|eot_id|>'):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.append(i)
    outs = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if sample:  # the distribution GRPO actually optimised: temperature 1, fixed seed
                torch.manual_seed(0)
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=max_new, eos_token_id=stops, pad_token_id=tok.pad_token_id)
            else:
                out = model.generate(**enc, do_sample=False, max_new_tokens=max_new, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        for row in out[:, enc['input_ids'].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


def eval_sentiment(model, tok, rows, chat_fn, max_new=48):
    prompts = [chat_fn(r['review']) for r in rows]
    outs = generate_greedy(model, tok, prompts, max_new=max_new)
    got = [parse_label(o) for o in outs]
    fmt = sum(g is not None for g in got) / len(got)
    acc = sum(g == r['label'] for g, r in zip(got, rows)) / len(rows)
    acc_when_formatted = (sum(g == r['label'] for g, r in zip(got, rows) if g is not None) / max(1, sum(g is not None for g in got)))
    return {'accuracy': acc, 'format_rate': fmt, 'accuracy_when_formatted': acc_when_formatted, 'n': len(rows), 'samples': list(zip([r['review'][:80] for r in rows[:5]], outs[:5]))}


def distinct_ratio(o):
    w = o.split()
    return len(set(w)) / len(w) if w else 0.0


def eval_language(model, tok, prompts, chat_fn, max_new=160, sample=False):
    outs = generate_greedy(model, tok, [chat_fn(p) for p in prompts], max_new=max_new, sample=sample)
    rs = [reward_language(o) for o in outs]
    r2 = [reward_language_v2(o) for o in outs]
    return {'mean_reward': sum(rs) / len(rs), 'mean_reward_v2': sum(r2) / len(r2), 'mean_rep3': sum(repeated_trigram_share(o) for o in outs) / len(outs),
            'mean_devanagari': sum(devanagari_share(o) for o in outs) / len(outs),
            'mean_words': sum(len(o.split()) for o in outs) / len(outs), 'mean_distinct': sum(distinct_ratio(o) for o in outs) / len(outs),
            'looping_share': sum(distinct_ratio(o) < 0.6 for o in outs) / len(outs),
            'n': len(outs), 'samples': list(zip(prompts[:5], outs[:5]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--template', action='store_true', help="use the model's own chat template (stock instruct models)")
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n', type=int, default=0, help='0 = all held-out reviews')
    ap.add_argument('--n-open', type=int, default=100)
    args = ap.parse_args()
    sent_test = [json.loads(l) for l in open(os.path.join(DATA, 'sent_test.jsonl'), encoding='utf-8')]
    if args.n:
        sent_test = sent_test[:args.n]
    open_test = [json.loads(l)['prompt'] for l in open(os.path.join(DATA, 'sft_test.jsonl'), encoding='utf-8')][:args.n_open]
    for path in args.models:
        t = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        if args.template:
            def s_fn(review):
                return tok.apply_chat_template([{'role': 'user', 'content': SENT_INSTR + review}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            def o_fn(p):
                return tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            s_fn, o_fn = sentiment_prompt, chat_prompt
        s = eval_sentiment(model, tok, sent_test, s_fn)
        l = eval_language(model, tok, open_test, o_fn)
        ls = eval_language(model, tok, open_test, o_fn, sample=True)
        n_params = sum(p.numel() for p in model.parameters())
        row = {'model': path, 'tag': args.tag or os.path.basename(path.rstrip('/\\')), 'params': n_params, 'template': args.template,
               'sent_acc': s['accuracy'], 'sent_format': s['format_rate'], 'sent_acc_when_formatted': s['accuracy_when_formatted'], 'sent_n': s['n'],
               'lang_reward': l['mean_reward'], 'lang_devanagari': l['mean_devanagari'], 'lang_words': l['mean_words'],
               'lang_distinct': l['mean_distinct'], 'lang_looping': l['looping_share'], 'lang_reward_v2': l['mean_reward_v2'], 'lang_rep3': l['mean_rep3'],
               'lang_reward_v2_sampled': ls['mean_reward_v2'], 'lang_rep3_sampled': ls['mean_rep3'],
               'lang_reward_sampled': ls['mean_reward'], 'lang_devanagari_sampled': ls['mean_devanagari'], 'lang_words_sampled': ls['mean_words'],
               'lang_looping_sampled': ls['looping_share'], 'lang_samples_sampled': ls['samples'],
               'sent_samples': s['samples'], 'lang_samples': l['samples'], 'seconds': time.time() - t}
        print(f"{row['tag']:28s} {n_params/1e6:6.0f}M  sentiment acc {s['accuracy']:.3f} format {s['format_rate']:.3f} "
              f"(acc|fmt {s['accuracy_when_formatted']:.3f})  language reward {l['mean_reward']:.3f} devanagari {l['mean_devanagari']:.3f} words {l['mean_words']:.0f} "
              f"v2 {l['mean_reward_v2']:.3f} looping {l['looping_share']:.2f} | sampled: reward {ls['mean_reward']:.3f} v2 {ls['mean_reward_v2']:.3f} looping {ls['looping_share']:.2f}  {row['seconds']:.0f}s")
        log_row('task_eval', row)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
