"""One sentence through the stock and the extended tokenizer, as HTML for the post (results/adapt/tok_example.json)."""
import os, json, html
from common import RES, MODELS
from transformers import AutoTokenizer

SENT = 'भारत की राजधानी नई दिल्ली है।'  # Bharat ki rajdhani Nai Dilli hai.
EN = 'The capital of India is New Delhi.'
out = {}
for name, path in [('stock', 'HuggingFaceTB/SmolLM2-135M'), ('ext', os.path.join(MODELS, 'tok-16k'))]:
    tok = AutoTokenizer.from_pretrained(path)
    for lang, s in [('hi', SENT), ('en', EN)]:
        ids = tok(s, add_special_tokens=False)['input_ids']
        pieces = []
        for i in ids:
            t = tok.decode([i])
            pieces.append(t if '�' not in t else '░')  # partial UTF-8 byte shown as a shaded block
        out[f'{name}_{lang}_n'] = len(ids)
        out[f'{name}_{lang}_html'] = ' '.join(f'<code>{html.escape(p).replace(" ", "&#9251;")}</code>' for p in pieces)
        print(name, lang, len(ids), pieces)
json.dump(out, open(os.path.join(RES, 'tok_example.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# how many pieces the stock pre-tokenizer cuts one word into before BPE starts
stock = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
word = 'राजधानी'  # rajdhani, capital
pieces = stock.backend_tokenizer.pre_tokenizer.pre_tokenize_str(word)
out['stock_pretok_pieces_rajdhani'] = len(pieces)
print('pre-tokenizer pieces for', word, len(pieces), [p[0] for p in pieces])
json.dump(out, open(os.path.join(RES, 'tok_example.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
