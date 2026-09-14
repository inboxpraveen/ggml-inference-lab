"""Architecture facts for the zoo models, read from the GGUF metadata rather than from memory.
Writes results/zoo_meta.json: per file, the arch, layer/head/expert counts, total bytes, the bytes a decode
step actually reads (dense: all; MoE: shared + active experts), KV or state bytes per token, head tie."""
import os, json, sys, collections
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ZOO = os.path.join(LAB, 'models', 'zoo')
files = sorted(f for f in os.listdir(ZOO) if f.endswith('.gguf')) + ['../Qwen3-0.6B-Q8_0.gguf', '../Qwen3-8B-Q6_K.gguf']


def field(r, k):
    f = r.fields.get(k)
    if f is None:
        return None
    if f.types and f.types[0].name == 'STRING':
        return f.parts[f.data[0]].tobytes().decode()
    if f.types and f.types[0].name == 'ARRAY':
        return [f.parts[i].tolist()[0] for i in f.data]
    return f.parts[f.data[0]].tolist()[0]


out = {}
for fn in files:
    path = os.path.join(ZOO, fn)
    try:
        r = GGUFReader(path)
    except Exception as e:  # the mamba-130m Q8_0 file trips the reader's block-size check on a 48-wide conv tensor
        out[fn] = dict(error=str(e)); print(fn, 'READER FAILED:', e); continue
    arch = field(r, 'general.architecture')
    g = lambda k, d=None: (field(r, f'{arch}.{k}') if field(r, f'{arch}.{k}') is not None else d)
    n_layer = g('block_count'); n_embd = g('embedding_length')
    n_head = g('attention.head_count'); n_head_kv = g('attention.head_count_kv', n_head)
    head_dim = g('attention.key_length') or (n_embd // n_head if n_head else None)
    n_expert = g('expert_count', 0) or 0; n_expert_used = g('expert_used_count', 0) or 0
    swa = g('attention.sliding_window', 0) or 0
    ssm_d_state = g('ssm.state_size'); ssm_d_inner = g('ssm.inner_size'); ssm_d_conv = g('ssm.conv_kernel'); ssm_n_group = g('ssm.group_count')
    layer_types = None
    roles = collections.OrderedDict()
    total = 0; head_bytes = 0; embd_bytes = 0; tied = True
    # per-layer kv heads may be a list (hybrids give 0 for non-attention layers)
    per_layer_kv = n_head_kv if isinstance(n_head_kv, list) else None
    for t in r.tensors:
        shape = [int(x) for x in t.shape]
        n = 1
        for s in shape: n *= s
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        nb = n * ts // bs
        total += nb
        name = t.name
        role = 'blk.*.' + name.split('.', 2)[2] if name.startswith('blk.') else name
        roles[role] = roles.get(role, 0) + nb
        if name == 'output.weight': head_bytes = nb; tied = False
        if name == 'token_embd.weight': embd_bytes = nb
    if tied: head_bytes = embd_bytes
    # bytes actually read by one decode step: everything except inactive experts (and the input embedding row lookup)
    expert_bytes = sum(b for k, b in roles.items() if '_exps' in k)
    active = total - expert_bytes + (expert_bytes * n_expert_used / n_expert if n_expert else 0)
    if not tied:
        active -= embd_bytes  # input lookup reads one row, not the table
    # attention layers: count layers that have attn_k
    attn_layers = sum(1 for t in r.tensors if t.name.endswith('attn_k.weight'))
    kv_per_tok = None
    if attn_layers and head_dim:
        if per_layer_kv:
            kv_per_tok = sum(2 * kvh * head_dim * 2 for kvh in per_layer_kv)  # f16
        else:
            kv_per_tok = 2 * attn_layers * (n_head_kv or n_head) * head_dim * 2
    out[fn] = dict(arch=arch, name=field(r, 'general.name'), n_layer=n_layer, attn_layers=attn_layers, n_embd=n_embd,
                   n_head=n_head, n_head_kv=n_head_kv, head_dim=head_dim, n_expert=n_expert, n_expert_used=n_expert_used,
                   sliding_window=swa, ssm=dict(d_state=ssm_d_state, d_inner=ssm_d_inner, d_conv=ssm_d_conv, n_group=ssm_n_group),
                   n_ctx_train=g('context_length'), vocab=len(r.fields['tokenizer.ggml.tokens'].data) if 'tokenizer.ggml.tokens' in r.fields else None,
                   total_bytes=total, active_bytes=active, expert_bytes=expert_bytes, head_bytes=head_bytes, tied=tied,
                   kv_bytes_per_token_f16=kv_per_tok, roles={k: v for k, v in roles.items()})
    print(f"{fn:45s} {arch:10s} L={n_layer} attnL={attn_layers} kv={n_head_kv} hd={head_dim} exp={n_expert}/{n_expert_used} swa={swa} "
          f"total={total/1e6:.0f}MB active={active/1e6:.0f}MB head={head_bytes/1e6:.0f}MB tied={tied} kv/tok={kv_per_tok}")
json.dump(out, open(os.path.join(LAB, 'results', 'zoo_meta.json'), 'w'), indent=1)
