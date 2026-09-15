"""Make HF's SDPA path use a fused attention kernel on this GPU.

transformers passes enable_gqa=True to torch's scaled_dot_product_attention when the model has grouped-query
attention and no padding mask. On the RTX 5060 (sm_120) with torch 2.11 that combination dispatches to the
math kernel, which materialises the full attention matrix for the backward pass: about 95 MB per layer per
2k tokens on a model whose whole activation footprint should be a third of that. Repeating K and V to the
query head count first (a few MB) lets the memory-efficient kernel run. Measured in this repo: activations
per 2k tokens fall from 4.1 GB to about 1.3 GB and throughput roughly doubles.
"""
import transformers.integrations.sdpa_attention as sa

_orig_use_gqa = sa.use_gqa_in_sdpa


def _never_gqa(attention_mask, key, value):
    return False


def install():
    sa.use_gqa_in_sdpa = _never_gqa
