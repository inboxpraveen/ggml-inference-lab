"""Greedy generations for a few fixed Hindi prompts from each checkpoint, saved to results/adapt/samples.jsonl
and rendered as an HTML table for the post.

  python scripts/adapt/samples.py HuggingFaceTB/SmolLM2-135M models/adapt/main models/adapt/sft models/adapt/grpo
"""
import os, sys, json, html, argparse
import torch
from common import RES, log_row
from chat import chat_prompt, sentiment_prompt
import fastattn; fastattn.install()
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    ('capital', 'भारत की राजधानी क्या है?'),  # what is the capital of India?
    ('tea', 'चाय बनाने के तीन चरण बताओ।'),  # give three steps to make tea
    ('review', None),  # a sentiment prompt, filled below
]
REVIEW = 'फोन की बैटरी दो घंटे में खत्म हो जाती है और कैमरा भी धुंधला है।'
# "the phone's battery dies in two hours and the camera is blurry too"


@torch.no_grad()
def gen(model, tok, prompt, max_new=80):
    enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    stops = [tok.eos_token_id]
    i = tok.convert_tokens_to_ids('<|im_end|>')
    if isinstance(i, int) and i >= 0:
        stops.append(i)
    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new, eos_token_id=stops, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--raw', action='store_true', help='no chat wrapping (for base models)')
    args = ap.parse_args()
    for path in args.models:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        tag = os.path.basename(path.rstrip('/\\'))
        row = {'model': path, 'tag': tag, 'raw': args.raw}
        for key, p in PROMPTS:
            if key == 'review':
                text = sentiment_prompt(REVIEW) if not args.raw else REVIEW + ' '
            else:
                text = chat_prompt(p) if not args.raw else p + ' '
            row[key] = gen(model, tok, text)
            print(tag, key, repr(row[key][:120]))
        log_row('samples', row)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
