"""Out-of-distribution Hindi held-out set: Hindi Wikipedia articles (wikimedia/wikipedia, 20231101.hi).
Nothing in this post trains on Wikipedia; FineWeb-2 is web crawl, so this is a second opinion on the Hindi score."""
import os, json, time, random
from datasets import load_dataset
from common import DATA

path = os.path.join(DATA, 'hi_wiki.jsonl')
if os.path.exists(path):
    print('exists', path); raise SystemExit
ds = load_dataset('wikimedia/wikipedia', '20231101.hi', split='train', streaming=True)
cand = []; t = time.time()
for ex in ds.skip(2000):  # skip the front of the dump (very short stubs and lists)
    if len(ex['text']) >= 1500:
        cand.append({'text': ex['text'], 'title': ex['title']})
    if len(cand) >= 8000:
        break
random.Random(0).shuffle(cand)  # the dump is ordered by topic (runs of district articles); a fixed shuffle spreads it
with open(path + '.tmp', 'w', encoding='utf-8') as f:
    for r in cand[:600]:
        f.write(json.dumps(r, ensure_ascii=False) + chr(10))
os.replace(path + '.tmp', path)
print(f'wrote {path}: 600 of {len(cand)} candidate docs in {time.time()-t:.0f}s')
