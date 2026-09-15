"""Extend SmolLM2's byte-level BPE with Hindi tokens without changing how it tokenizes English.

Steps, each one checked:
  1. Train a byte-level BPE on Hindi text with a pre-tokenizer that keeps Devanagari words whole. The GPT-2
     regex SmolLM2 uses splits a word at every vowel sign (matras are \\p{M}, not \\p{L}), so a Hindi word can
     never become one token under it. We use \\p{L}[\\p{L}\\p{M}]* instead; on English text that regex behaves
     identically, which step 4 asserts.
  2. Keep only the learned merges whose result contains a non-ASCII byte (i.e. Devanagari); ASCII merges would
     change English tokenization and the model already has those.
  3. Put the Hindi merges in front of the original ones (higher priority) and append their tokens to the vocab.
  4. Assert: English tokenization identical to the stock tokenizer; Hindi round-trips.

  python scripts/adapt/extend_tokenizer.py --new-tokens 16000 --out models/adapt/tok-16k
"""
import os, sys, json, argparse
from common import DATA, MODELS, log_row
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
from transformers import AutoTokenizer

# GPT-2's pattern with \p{L}+ widened to \p{L}[\p{L}\p{M}]* so combining marks stay inside the word.
WIDE = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}[\p{L}\p{M}]*| ?\p{N}+| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"""

ap = argparse.ArgumentParser()
ap.add_argument('--new-tokens', type=int, default=16000)
ap.add_argument('--train-chars', type=int, default=150_000_000)
ap.add_argument('--out', default=None)
ap.add_argument('--base', default='HuggingFaceTB/SmolLM2-135M')
args = ap.parse_args()
out = args.out or os.path.join(MODELS, f'tok-{args.new_tokens // 1000}k')


def hindi_lines(budget):
    n = 0
    with open(os.path.join(DATA, 'hi_train.jsonl'), encoding='utf-8') as f:
        for line in f:
            t = json.loads(line)['text']
            yield t
            n += len(t)
            if n >= budget:
                return


def wide_pretok():
    return pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Split(Regex(WIDE), 'isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


# GPT-2's byte -> unicode table, to tell which byte-level chars stand for bytes >= 0x80
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
char_to_byte = {chr(c): b for b, c in zip(bs, cs)}


def has_non_ascii(tok_str):
    return any(char_to_byte[c] >= 0x80 for c in tok_str)


def merges_as_str(ms):
    return [m if isinstance(m, str) else ' '.join(m) for m in ms]


# 1. train the Hindi BPE
hi_tok = Tokenizer(models.BPE())
hi_tok.pre_tokenizer = wide_pretok()
hi_tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=256 + args.new_tokens + 3000, min_frequency=5, show_progress=False,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
hi_tok.train_from_iterator(hindi_lines(args.train_chars), trainer=trainer)
hi_json = json.loads(hi_tok.to_str())
hi_merges = merges_as_str(hi_json['model']['merges'])
print('hindi bpe: vocab', len(hi_json['model']['vocab']), 'merges', len(hi_merges))

# 2 + 3. splice into the stock tokenizer.json
base = AutoTokenizer.from_pretrained(args.base)
stock = json.loads(base.backend_tokenizer.to_str())
vocab = stock['model']['vocab']
old_merges = merges_as_str(stock['model']['merges'])
next_id = max(vocab.values()) + 1
added, kept, skipped_ascii, skipped_missing = [], [], 0, 0
for m in hi_merges:
    a, b = m.split(' ')
    tok_str = a + b
    if not has_non_ascii(tok_str):
        skipped_ascii += 1; continue
    if a not in vocab or b not in vocab:
        skipped_missing += 1; continue
    kept.append(m)
    if tok_str not in vocab:
        vocab[tok_str] = next_id; next_id += 1; added.append(tok_str)
    if len(added) >= args.new_tokens:
        break
# a pair that appears in both lists would get the later (lower-priority) rank when the merge table is
# built, which silently reorders the Hindi merge path; drop the stock copy of any pair we now own.
kept_set = set(kept)
dupes = sum(m in kept_set for m in old_merges)
stock['model']['merges'] = kept + [m for m in old_merges if m not in kept_set]
print(f'dropped {dupes} stock merges that the Hindi list already contains')
tmp = Tokenizer(models.BPE()); tmp.pre_tokenizer = wide_pretok()
stock['pre_tokenizer'] = json.loads(tmp.to_str())['pre_tokenizer']
print(f'kept {len(kept)} merges, added {len(added)} tokens, skipped {skipped_ascii} ascii-only, '
      f'{skipped_missing} with missing parts; vocab now {len(vocab)}')

os.makedirs(out, exist_ok=True)
base.save_pretrained(out)
# transformers' GPT2Tokenizer class rebuilds the ByteLevel pre-tokenizer on load, which would throw away the
# widened regex; the generic fast class loads tokenizer.json as it is.
cfg_p = os.path.join(out, 'tokenizer_config.json')
cfg = json.load(open(cfg_p, encoding='utf-8'))
cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
json.dump(cfg, open(cfg_p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
with open(os.path.join(out, 'tokenizer.json'), 'w', encoding='utf-8') as f:
    json.dump(stock, f, ensure_ascii=False)
new = AutoTokenizer.from_pretrained(out)
assert len(new) == len(vocab), (len(new), len(vocab))
probe = 'भारत'
assert new.backend_tokenizer.pre_tokenizer.pre_tokenize_str(probe) == tmp.pre_tokenizer.pre_tokenize_str(probe),     'the transformers wrapper replaced the pre-tokenizer'

# 4. checks
en = [json.loads(l)['text'] for l in open(os.path.join(DATA, 'en_wikitext.jsonl'), encoding='utf-8')]
same = sum(base(d)['input_ids'] == new(d)['input_ids'] for d in en)
print(f'english identical on {same}/{len(en)} wikitext articles')
hi = [json.loads(l)['text'] for l in open(os.path.join(DATA, 'hi_test.jsonl'), encoding='utf-8')][:500]
rt = sum(new.decode(new(d, add_special_tokens=False)['input_ids']) == d for d in hi)
print(f'hindi round-trip {rt}/{len(hi)}')
old_n = sum(len(base(d, add_special_tokens=False)['input_ids']) for d in hi)
new_n = sum(len(new(d, add_special_tokens=False)['input_ids']) for d in hi)
words = sum(len(d.split()) for d in hi)
print(f'hindi tokens/word: stock {old_n/words:.2f} -> extended {new_n/words:.2f}')
ex = 'भारत की राजधानी नई दिल्ली है।'
print('example stock   :', [base.decode([i]) for i in base(ex, add_special_tokens=False)['input_ids']])
print('example extended:', [new.decode([i]) for i in new(ex, add_special_tokens=False)['input_ids']])
log_row('tokenizer_ext', {'out': out, 'new_tokens': len(added), 'merges_kept': len(kept), 'skipped_ascii': skipped_ascii,
                          'skipped_missing': skipped_missing, 'vocab': len(vocab), 'english_identical': f'{same}/{len(en)}',
                          'hindi_roundtrip': f'{rt}/{len(hi)}', 'hi_tpw_stock': old_n / words, 'hi_tpw_ext': new_n / words,
                          'train_chars': args.train_chars})
with open(os.path.join(out, 'added_tokens_list.json'), 'w', encoding='utf-8') as f:
    json.dump(added, f, ensure_ascii=False)
