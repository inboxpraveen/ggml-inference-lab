"""Measure what each tokenizer-extension detail is worth.

Starts from the finished tok-16k (Hindi merges first, stock duplicates dropped, wide pre-tokenizer,
generic loader) and undoes one detail at a time, scoring Hindi tokens per word on the held-out set.
"""
import os, json, copy, tempfile
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from common import DATA, MODELS, RES, BASE_MODEL, log_row

src = os.path.join(MODELS, 'tok-16k')
ext = json.load(open(os.path.join(src, 'tokenizer.json'), encoding='utf-8'))
row16 = [json.loads(l) for l in open(os.path.join(RES, 'tokenizer_ext.jsonl'), encoding='utf-8')
         if 'tok-16k' in l][-1]
n_kept = row16['merges_kept']
merges = [m if isinstance(m, str) else ' '.join(m) for m in ext['model']['merges']]
hi_merges, stock_rest = merges[:n_kept], merges[n_kept:]
stock = json.loads(AutoTokenizer.from_pretrained(BASE_MODEL).backend_tokenizer.to_str())
stock_merges = [m if isinstance(m, str) else ' '.join(m) for m in stock['model']['merges']]
hi_set = set(hi_merges)
dupes = [m for m in stock_merges if m in hi_set]
print('hindi merges', len(hi_merges), 'stock merges', len(stock_merges), 'shared pairs', len(dupes))

hi = [json.loads(l)['text'] for l in open(os.path.join(DATA, 'hi_test.jsonl'), encoding='utf-8')][:500]
words = sum(len(d.split()) for d in hi)


def tpw(tok):
    return sum(len(tok.encode(d, add_special_tokens=False).ids) for d in hi) / words


def build(merge_list, pretok=None):
    j = copy.deepcopy(ext)
    j['model']['merges'] = merge_list
    if pretok is not None:
        j['pre_tokenizer'] = pretok
    return Tokenizer.from_str(json.dumps(j, ensure_ascii=False))


variants = {
    'final (hindi merges first, duplicates dropped, wide pre-tokenizer)': build(hi_merges + stock_rest),
    'hindi merges appended after the stock list': build(stock_rest + hi_merges),
    'hindi merges first but stock duplicates kept': build(hi_merges + stock_merges),
    'stock pre-tokenizer (splits at matras)': build(hi_merges + stock_rest, stock['pre_tokenizer']),
    'appended, duplicates kept, stock pre-tokenizer': build(stock_merges + hi_merges, stock['pre_tokenizer']),
}
base_tpw = tpw(Tokenizer.from_str(json.dumps(stock, ensure_ascii=False)))
print(f'stock tokenizer: {base_tpw:.2f} tokens/word')
out = {'stock': base_tpw}
for name, t in variants.items():
    v = tpw(t)
    used = len({i for d in hi[:200] for i in t.encode(d, add_special_tokens=False).ids if i >= 49152})
    print(f'{v:.2f} tokens/word, {used:5d} of the new tokens used  {name}')
    out[name] = {'tpw': v, 'new_tokens_used': used}
log_row('tok_variants', out)
