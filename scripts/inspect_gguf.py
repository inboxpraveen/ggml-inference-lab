"""Dump tensor names, shapes, types and byte sizes of a GGUF, grouped by role."""
import sys, collections
from gguf import GGUFReader, GGMLQuantizationType as T
from gguf.constants import GGML_QUANT_SIZES

r = GGUFReader(sys.argv[1])
def kv(k):
    f = r.fields.get(k)
    if f is None: return None
    v = f.parts[f.data[0]]
    return v.tobytes().decode() if f.types and f.types[0].name == 'STRING' else v.tolist()[0]
for k in ['general.architecture','general.name','general.file_type','qwen3.embedding_length','qwen3.block_count',
          'qwen3.feed_forward_length','qwen3.attention.head_count','qwen3.attention.head_count_kv',
          'qwen3.attention.key_length','qwen3.context_length','tokenizer.ggml.model']:
    print(f'{k:40s} {kv(k)}')
vocab = len(r.fields['tokenizer.ggml.tokens'].data)
print(f'{"vocab (token count)":40s} {vocab}')
groups = collections.OrderedDict()
total_bytes = total_params = 0
print()
print(f'{"tensor":32s} {"shape":22s} {"type":8s} {"bpw":6s} {"MB":>9s}')
for t in r.tensors:
    n = t.name; shape = [int(x) for x in t.shape]
    nparams = 1
    for s in shape: nparams *= s
    bs, ts = GGML_QUANT_SIZES[t.tensor_type]
    nbytes = nparams * ts // bs
    bpw = 8*nbytes/nparams
    total_bytes += nbytes; total_params += nparams
    if n.startswith('blk.'):
        role = 'blk.*.' + n.split('.',2)[2]
    else:
        role = n
    g = groups.setdefault(role, [0,0,set()])
    g[0]+=nparams; g[1]+=nbytes; g[2].add(t.tensor_type.name)
    if not n.startswith('blk.') or n.startswith('blk.0.'):
        print(f'{n:32s} {str(shape):22s} {t.tensor_type.name:8s} {bpw:5.2f} {nbytes/1e6:9.2f}')
print()
print(f'{"role":32s} {"params(M)":>10s} {"MB":>9s} {"%params":>8s} {"%bytes":>7s}  types')
for role,(p,b,ty) in groups.items():
    print(f'{role:32s} {p/1e6:10.2f} {b/1e6:9.2f} {100*p/total_params:8.2f} {100*b/total_bytes:7.2f}  {",".join(sorted(ty))}')
print(f'\nTOTAL params {total_params/1e6:.2f} M, bytes {total_bytes/1e6:.2f} MB, mean bpw {8*total_bytes/total_params:.3f}')
