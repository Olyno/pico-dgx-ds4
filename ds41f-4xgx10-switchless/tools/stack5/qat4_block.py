"""qat4_block.py: the quantisation-aware pilot of docs/stack5.md (lever C), as the block added to our drafter trainer.

Not runnable on its own (see body4_sweep.py): `m` is the torch mirror of the DSpark drafter, `dsm.STE` a straight-through
function (forward: the quantised value; backward: the gradient to the latent), `dsm.BF` torch.bfloat16. The block rounds the
chosen dense latents to MXFP4 at every refresh, and first moves each latent halfway to its MXFP4 value, because FP8 latents
sit exactly on MXFP4 rounding ties (see the doc).
"""
import torch

QAT4 = opt("--qat4", "", str)
if QAT4:
    # stack5 lever C: quantisation-aware training on the MXFP4 grid. The chosen dense tensors' fp32 latents are rounded to
    # e2m1 values on one power-of-two scale per 32 values of a row at every refresh, with a straight-through gradient.
    def mx4f(w):
        o, i = w.shape; wf = w.reshape(o, i // 32, 32)
        s_ = torch.exp2(torch.ceil(torch.log2(wf.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 6.0)))
        x = (wf.abs() / s_).clamp(max=6.0)
        q_ = torch.where(x < 2, torch.round(x * 2) / 2, torch.where(x < 4, torch.round(x), torch.round(x / 2) * 2))
        return (q_ * s_ * wf.sign()).reshape(o, i)
    QPICK = [k for k in m.dense_names if QAT4 == "all" or any(t_ in k for t_ in QAT4.split(","))]
    # The FP8 latents are binary fractions, so about a quarter of them sit EXACTLY on an MXFP4 rounding tie; the first
    # optimizer step, however small, would flip all of those at once in the gradient's direction and wreck the model
    # (measured: train d1 .83 -> .09 within 10 steps at any rate, AdamW or SGD). Start each latent halfway between its
    # FP8 value and its MXFP4 value instead: at least a quarter of a grid step away from every threshold.
    with torch.no_grad():
        for k in QPICK:
            w = m.p(k); q0 = mx4f(w); w.copy_(q0 + 0.5 * (w - q0))
    _refresh0 = m.refresh
    def _refresh_q4():
        _refresh0()
        for k in QPICK:
            w = m.p(k)
            with torch.no_grad(): qv = mx4f(w)
            m.q[k] = dsm.STE.apply(w, qv).to(dsm.BF) if w.requires_grad else qv.to(dsm.BF)
    m.refresh = _refresh_q4
