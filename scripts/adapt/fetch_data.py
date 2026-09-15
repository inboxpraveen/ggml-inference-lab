"""Fetch the text this post trains and evaluates on. Everything lands in data/adapt/ (git-ignored).

  hi_train.jsonl   Hindi web text, streamed from FineWeb-2 hin_Deva train (first N documents)
  hi_test.jsonl    Hindi held-out text from FineWeb-2 hin_Deva test (fixed sample, never trained on)
  en_train.jsonl   English replay text from FineWeb-Edu sample-10BT (the model's own pretraining mix)
  en_test.jsonl    English held-out text: FineWeb-Edu documents after the training cut (fixed sample)
  en_wikitext.jsonl wikitext-2 test articles (already in data/), used to check English tokenization is unchanged

Run: python scripts/adapt/fetch_data.py [--hi-chars 400000000] [--en-chars 60000000]
"""
import os, sys, json, argparse, time
from common import DATA, LAB
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument('--hi-chars', type=int, default=400_000_000)
ap.add_argument('--en-chars', type=int, default=60_000_000)
ap.add_argument('--hi-test-docs', type=int, default=3000)
ap.add_argument('--min-chars', type=int, default=200)
args = ap.parse_args()


def stream_to(path, ds, budget_chars, min_chars, key='text', extra=None):
    if os.path.exists(path):
        print('exists, skipping', path); return
    n = chars = 0; t = time.time()
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        for ex in ds:
            txt = ex[key]
            if len(txt) < min_chars:
                continue
            row = {'text': txt}
            if extra:
                row.update({k: ex.get(k) for k in extra})
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1; chars += len(txt)
            if n % 5000 == 0:
                print(f'  {n} docs, {chars/1e6:.0f}M chars, {time.time()-t:.0f}s', flush=True)
            if chars >= budget_chars:
                break
    os.replace(path + '.tmp', path)
    print(f'wrote {path}: {n} docs, {chars/1e6:.1f}M chars')


# Hindi held-out: the test split is one 88 MB parquet; take a fixed prefix so the sample is reproducible.
hi_test = load_dataset('HuggingFaceFW/fineweb-2', name='hin_Deva', split='test', streaming=True)
stream_to(os.path.join(DATA, 'hi_test.jsonl'), hi_test.take(args.hi_test_docs), 10**12, args.min_chars,
          extra=['language_score'])

# Hindi training text: stream the train split in file order. FineWeb-2 is already deduplicated and
# language-filtered, so the first N documents are as good a sample as any.
hi_train = load_dataset('HuggingFaceFW/fineweb-2', name='hin_Deva', split='train', streaming=True)
stream_to(os.path.join(DATA, 'hi_train.jsonl'), hi_train, args.hi_chars, args.min_chars)

# English replay: FineWeb-Edu is the largest part of SmolLM2's own training mix.
en_train = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
stream_to(os.path.join(DATA, 'en_train.jsonl'), en_train, args.en_chars, args.min_chars)
# English held-out: the next documents of the same stream, after the training cut, so it matches the
# distribution the model was pretrained on.
if not os.path.exists(os.path.join(DATA, 'en_test.jsonl')):
    n_train = sum(1 for _ in open(os.path.join(DATA, 'en_train.jsonl'), encoding='utf-8'))
    stream_to(os.path.join(DATA, 'en_test.jsonl'), en_train.skip(n_train + 200).take(args.hi_test_docs), 10**12, args.min_chars)

# English held-out: wikitext-2 test, split into articles at the " = Title = " headings.
wt = os.path.join(LAB, 'data', 'wikitext-2-raw', 'wiki.test.raw')
out = os.path.join(DATA, 'en_wikitext.jsonl')
if not os.path.exists(out):
    docs, cur = [], []
    for line in open(wt, encoding='utf-8'):
        if line.startswith(' = ') and not line.startswith(' = = ') and cur:
            docs.append(''.join(cur)); cur = []
        cur.append(line)
    if cur:
        docs.append(''.join(cur))
    with open(out, 'w', encoding='utf-8') as f:
        for d in docs:
            if len(d) >= args.min_chars:
                f.write(json.dumps({'text': d}, ensure_ascii=False) + '\n')
    print('wrote', out, len(docs), 'articles')
