"""Shared training plumbing for the 8 GB card: bf16 weights with an fp32 master copy, and chunked head
computations so the vocabulary-wide logits never exist all at once."""
import torch
from torch.utils.checkpoint import checkpoint


class MasterWeights:
    """Model runs in bf16; the optimizer updates an fp32 copy and writes it back after every step."""

    def __init__(self, model, lr, wd=0.1, betas=(0.9, 0.95)):
        self.params = list(model.parameters())
        self.master = [p.detach().clone().float() for p in self.params]
        model.to(torch.bfloat16)
        decay = [m for p, m in zip(self.params, self.master) if p.dim() >= 2]
        no_decay = [m for p, m in zip(self.params, self.master) if p.dim() < 2]
        self.opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}],
                                     lr=lr, betas=betas, eps=1e-8, fused=True)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'] = lr

    def step(self, clip=1.0):
        for p, m in zip(self.params, self.master):
            m.grad = None if p.grad is None else p.grad.float(); p.grad = None
        gn = torch.nn.utils.clip_grad_norm_(self.master, clip).item()
        self.opt.step(); self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, m in zip(self.params, self.master):
                p.copy_(m)
        return gn


def chunked_ce(head, h, y, chunk=512, ignore_index=-100):
    """Mean cross-entropy over targets != ignore_index; logits built 512 positions at a time and rebuilt in
    the backward pass."""
    hf, yf = h.reshape(-1, h.shape[-1]), y.reshape(-1)

    def f(hc, yc):
        return torch.nn.functional.cross_entropy(head(hc).float(), yc, reduction='sum', ignore_index=ignore_index)
    total = 0.0
    for i in range(0, hf.shape[0], chunk):
        total = total + checkpoint(f, hf[i:i + chunk], yf[i:i + chunk], use_reentrant=False)
    return total / (yf != ignore_index).sum().clamp(min=1)


def chunked_logprobs(head, h, y, chunk=512):
    """Per-position log p(y | h) with the same chunking; returns a tensor shaped like y."""
    hf, yf = h.reshape(-1, h.shape[-1]), y.reshape(-1)

    def f(hc, yc):
        return -torch.nn.functional.cross_entropy(head(hc).float(), yc, reduction='none')
    outs = [checkpoint(f, hf[i:i + chunk], yf[i:i + chunk], use_reentrant=False) for i in range(0, hf.shape[0], chunk)]
    return torch.cat(outs).view(y.shape)
