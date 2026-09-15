"""Build the post-training data files in data/adapt/:

  sft_train.jsonl / sft_test.jsonl   Hindi instruction pairs from FreedomIntelligence/alpaca-gpt4-hindi,
                                     kept only if prompt and answer are mostly Devanagari and short enough
  sent_train.jsonl / sent_test.jsonl Hindi product reviews with a positive/negative label from
                                     ai4bharat/IndicSentiment (validation + first 400 test rows for RL
                                     training; the remaining test rows are held out and never trained on)
  open_prompts.jsonl                 Hindi prompts with no reference answer, for the language reward

Run: python scripts/adapt/prep_post.py
"""
import os, json, random, urllib.request
from common import DATA, devanagari_share
from datasets import load_dataset

random.seed(0)


def write(name, rows):
    with open(os.path.join(DATA, name), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{name}: {len(rows)} rows')


# instruction pairs
if not os.path.exists(os.path.join(DATA, 'sft_train.jsonl')):
    ds = load_dataset('FreedomIntelligence/alpaca-gpt4-hindi', split='train')
    rows = []
    for ex in ds:
        conv = ex['conversations']
        if len(conv) != 2 or conv[0]['from'] != 'human':
            continue
        q, a = conv[0]['value'].strip(), conv[1]['value'].strip()
        if not (20 <= len(q) <= 600 and 20 <= len(a) <= 1200):
            continue
        if devanagari_share(q) < 0.7 or devanagari_share(a) < 0.7:
            continue
        rows.append({'prompt': q, 'response': a})
    random.shuffle(rows)
    write('sft_test.jsonl', rows[:500])
    write('sft_train.jsonl', rows[500:12500])
    write('open_prompts.jsonl', [{'prompt': r['prompt']} for r in rows[12500:14500]])

# sentiment
if not os.path.exists(os.path.join(DATA, 'sent_train.jsonl')):
    base = 'https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/{}/hi.json'
    def fetch(split):
        raw = urllib.request.urlopen(base.format(split)).read().decode('utf-8')
        out = []
        for line in raw.strip().split('\n'):
            d = json.loads(line)
            if not d.get('LABEL') or not d.get('INDIC REVIEW'):
                continue
            out.append({'review': d['INDIC REVIEW'].strip(), 'label': d['LABEL'].strip().lower(),
                        'english': d['ENGLISH REVIEW'].strip(), 'category': d.get('CATEGORY')})
        return out
    val, test = fetch('validation'), fetch('test')
    random.shuffle(test)
    write('sent_train.jsonl', val + test[:400])
    write('sent_test.jsonl', test[400:])
    from collections import Counter
    print('labels train', Counter(r['label'] for r in val + test[:400]), 'test', Counter(r['label'] for r in test[400:]))
