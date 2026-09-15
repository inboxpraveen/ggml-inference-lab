"""Tokenizer fertility on Hindi and English held-out text.

For each tokenizer: tokens per whitespace word, characters per token, UTF-8 bytes per token, and how many
distinct vocabulary entries the Hindi text actually uses. Rows go to results/adapt/tok_stats.jsonl.

  python scripts/adapt/tok_stats.py                       # the stock tokenizers
  python scripts/adapt/tok_stats.py models/adapt/tok-16k  # plus one or more local tokenizers
"""
import os, sys, json
from common import DATA, RES, log_row, read_jsonl
from transformers import AutoTokenizer

STOCK = [
    ('HuggingFaceTB/SmolLM2-135M', 'SmolLM2 (49k)'),
    ('Qwen/Qwen2.5-0.5B', 'Qwen2.5 (152k)'),
    ('Qwen/Qwen3-0.6B', 'Qwen3 (152k)'),
    ('unsloth/Llama-3.2-1B', 'Llama 3.2 (128k)'),
    ('unsloth/gemma-3-270m', 'Gemma 3 (262k)'),
    ('sarvamai/sarvam-1', 'Sarvam-1 (68k)'),
    ('ai4bharat/Airavata', 'Airavata (48k)'),
]


def load_docs(name, n=1000):
    p = os.path.join(DATA, name)
    docs = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line)['text'])
            if len(docs) >= n:
                break
    return docs


def stats(tok, docs):
    n_tok = n_word = n_char = n_byte = 0
    used = set()
    for d in docs:
        ids = tok(d, add_special_tokens=False)['input_ids']
        n_tok += len(ids); n_word += len(d.split()); n_char += len(d); n_byte += len(d.encode('utf-8'))
        used.update(ids)
    return {'tokens_per_word': n_tok / n_word, 'chars_per_token': n_char / n_tok,
            'bytes_per_token': n_byte / n_tok, 'vocab_used': len(used), 'n_tokens': n_tok, 'n_words': n_word}


def main():
    hi = load_docs('hi_test.jsonl'); en = load_docs('en_test.jsonl')
    todo = list(STOCK) + [(p, os.path.basename(p.rstrip('/\\'))) for p in sys.argv[1:]]
    for path, label in todo:
        try:
            tok = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            print('skip', path, type(e).__name__, str(e)[:100]); continue
        row = {'tokenizer': path, 'label': label, 'vocab_size': len(tok),
               'hi': stats(tok, hi), 'en': stats(tok, en)}
        print(f"{label:18s} vocab {len(tok):7d}  hi {row['hi']['tokens_per_word']:.2f} tok/word "
              f"{row['hi']['chars_per_token']:.2f} ch/tok  en {row['en']['tokens_per_word']:.2f} tok/word "
              f"{row['en']['chars_per_token']:.2f} ch/tok   hi vocab used {row['hi']['vocab_used']}")
        log_row('tok_stats', row)


if __name__ == '__main__':
    main()
