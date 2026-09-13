"""Compare two GGUFs tensor by tensor: which tensors differ byte-wise, and the metadata that differs."""
import sys, hashlib
import numpy as np
from gguf import GGUFReader

a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: t for t in a.tensors}
tb = {t.name: t for t in b.tensors}
print('tensors', len(ta), len(tb), 'only-in-a', sorted(set(ta) - set(tb)), 'only-in-b', sorted(set(tb) - set(ta)))
diff = []
for n in sorted(set(ta) & set(tb)):
    x, y = ta[n], tb[n]
    if x.tensor_type != y.tensor_type:
        diff.append((n, 'type', x.tensor_type.name, y.tensor_type.name)); continue
    da, db = np.asarray(x.data).view(np.uint8), np.asarray(y.data).view(np.uint8)
    if da.shape != db.shape or not np.array_equal(da, db):
        nd = int(np.count_nonzero(da != db)) if da.shape == db.shape else -1
        diff.append((n, 'bytes', nd, da.size))
print('differing tensors:', len(diff))
for d in diff[:12]:
    print('  ', d)
def meta(r):
    out = {}
    for k, f in r.fields.items():
        if k.startswith('tokenizer.ggml.') and k not in ('tokenizer.ggml.model', 'tokenizer.ggml.pre', 'tokenizer.ggml.add_bos_token', 'tokenizer.ggml.eos_token_id', 'tokenizer.ggml.padding_token_id', 'tokenizer.ggml.bos_token_id'):
            continue
        try:
            v = f.parts[f.data[0]]
            v = v.tobytes().decode() if f.types and f.types[0].name == 'STRING' else v.tolist()
        except Exception:
            v = '?'
        out[k] = v
    return out
ma, mb = meta(a), meta(b)
for k in sorted(set(ma) | set(mb)):
    if ma.get(k) != mb.get(k):
        print(f'  meta {k}: {str(ma.get(k))[:60]!r} vs {str(mb.get(k))[:60]!r}')
