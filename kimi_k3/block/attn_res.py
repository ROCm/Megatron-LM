"""Attention-residual mixer — the FP32 eager oracle.

Transcribed from the released `KimiLinearModel._apply_attn_res`
(HF moonshotai/Kimi-K3, revision a590ce09). Every parity test for AttnRes is
against this function; the fused kernel in P11 must reproduce it.

The mechanism, per decoder layer (develop/architecture §6):

    prefix_sum      [S, B, H]      running sum inside the current block
    block_residual  [S, B, K, H]   K frozen block outputs, K = ceil(l / block_size)

and each mix is a softmax over the K+1 candidates of a score formed by dotting an
RMS-normalised candidate with `norm.weight * proj.weight`.

Two facts the release's own code makes obvious and that matter later:

* the RMSNorm gain and the `[1, H]` projection only ever appear multiplied
  together, so they collapse to a single `[H]` score vector -- a free fold for
  the P11 fused kernel;
* the mix upcasts the whole `[T, K+1, H]` stack to fp32, twice, which is the
  dominant non-GEMM memory cost in the model (see `tools/attn_res_probe.py`).
"""

from typing import Optional

import torch

from ..numerics import hi_dtype as _hi


def score_vector(norm_weight: torch.Tensor, proj_weight: torch.Tensor) -> torch.Tensor:
    """``norm.weight * proj.weight.squeeze(0)``, in the mix dtype. Shape [H]."""
    hi = _hi(norm_weight.dtype)
    return norm_weight.to(hi) * proj_weight.squeeze(0).to(hi)


def attn_res_mix(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
    fp32: bool = True,
) -> torch.Tensor:
    """One AttnRes mix.

    Args:
        prefix_sum: ``[T, H]`` (tokens flattened).
        block_residual: ``[T, K, H]``; ``K == 0`` is legal and means "no slots yet",
            in which case the mix is a no-op that returns ``prefix_sum``.
        norm_weight: RMSNorm gain, ``[H]``.
        proj_weight: the per-site projection, ``[1, H]``.
        eps: the norm's ``variance_epsilon``.
        fp32: run the mix in fp32, as the release does. False is for measuring
            what a lower-precision variant would cost, never for parity.
    """
    if block_residual.shape[1] == 0:
        return prefix_sum

    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # [T, K+1, H]
    v_hi = v.to(_hi(v.dtype)) if fp32 else v
    variance = v_hi.pow(2).mean(-1, keepdim=True)
    k = v_hi * torch.rsqrt(variance + eps)
    weight = score_vector(norm_weight, proj_weight)
    if not fp32:
        weight = weight.to(v_hi.dtype)
    scores = (k * weight).sum(-1)  # [T, K+1]
    probs = scores.softmax(-1).unsqueeze(1)  # [T, 1, K+1]
    out = torch.matmul(probs, v_hi).squeeze(1)
    return out.to(v.dtype)


class AttnResMixer(torch.nn.Module):
    """Module wrapper owning one site's norm gain and projection.

    Two instances per decoder layer (attention site and MLP site) plus one at the
    model output, matching the 187 `*_res_proj` tensors in the released
    checkpoint (93 x 2 + 1).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        fp32: bool = True,
        fused: bool = False,
        chunk: int = 4096,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.proj = torch.nn.Parameter(torch.zeros(1, hidden_size))
        self.eps = eps
        self.fp32 = fp32
        #: `--k3-attn-res-fused`. Off by default: the eager path is the oracle,
        #: and the chunked one turns on once G44 records a measured win (R5.3).
        self.fused = fused
        self.chunk = chunk

    def forward(
        self, prefix_sum: torch.Tensor, block_residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if block_residual is None:
            return prefix_sum
        if self.fused and self.fp32:
            return attn_res_mix_fused(
                prefix_sum, block_residual, self.weight, self.proj, self.eps, self.chunk
            )
        return attn_res_mix(
            prefix_sum, block_residual, self.weight, self.proj, self.eps, self.fp32
        )


#: Rows per chunk in the fused mixer.
#:
#: **Not purely a memory knob, and the exception was measured rather than
#: assumed.** The mix is independent per token, so the *forward* is bit-identical
#: at any chunk size, at both the fast and release tiers. The *backward* is not:
#: at production width a small chunk makes the batched matmul a different shape,
#: rocBLAS picks a different kernel, and the per-token gradients change in the
#: last bits. Measured at T=8192, K=8, H=7168:
#:
#:     chunk 32    -> per-token gradients differ, rel 4e-04 (prefix) / 7e-04 (slots)
#:     chunk 1024  -> bit-identical
#:     chunk 4096  -> bit-identical
#:
#: So the default sits in the region where the guarantee holds. Below ~1024 rows
#: at release width the chunking is still *correct*, but it is no longer free, and
#: G43 says so instead of claiming a bound it does not have.
ATTN_RES_CHUNK = 4096

#: Below this many rows the backward is no longer bit-identical at release width.
ATTN_RES_BITWISE_MIN_CHUNK = 1024


def attn_res_mix_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
    chunk: int = ATTN_RES_CHUNK,
) -> torch.Tensor:
    """The same mix, without ever holding a full `[T, K+1, H]` fp32 stack.

    The eager mixer materialises that stack twice -- once for the concatenation
    and once for the normalised copy `k` -- and at production shape that is the
    dominant non-GEMM memory cost in the model: **109.6 GiB per forward**
    (`results/attn_res.md`). Every token's mix is independent of every other, so
    the whole thing can be done a block of rows at a time, and the peak fp32
    temporary drops by `T / chunk`.

    The arithmetic is untouched: each chunk runs exactly the eager op sequence on
    exactly the same values in the same order, so the **forward is bit-identical**
    at any chunk size, at both tiers.

    Two things about the backward are not, and both are stated rather than
    discovered later:

    * `norm_weight` is `[H]`, shared by every token, so its gradient is a sum over
      all `T` rows and chunking reorders that sum. At release width the error is
      1.9e-07 of the gradient's magnitude at 4 chunks and 7.7e-07 at 256 -- it
      *does* grow, but sub-linearly, 64x the chunks for 4x the error, which is the
      signature of a reordered sum rather than lost precision. One chunk is
      bitwise, as it must be.
    * The **per-token** gradients are bit-identical only while the chunk is large
      enough that the batched matmul keeps the same rocBLAS kernel. At release
      width that threshold is around `ATTN_RES_BITWISE_MIN_CHUNK`; at `chunk=32`
      they differ by 4e-04 relative. The default is well above it.

    ponytail: the next rung is dropping the concatenation as well -- computing
    each candidate's score and output contribution from `block_residual` and
    `prefix_sum` separately, so the `[T, K+1, H]` stack never exists at any size.
    It saves roughly another half, and it is *not* bit-identical, because the
    output reduction stops being a single matmul over `K+1`. Worth doing only if
    a trace says this row still matters after chunking.
    """
    if block_residual.shape[1] == 0:
        return prefix_sum
    if chunk <= 0 or prefix_sum.shape[0] <= chunk:
        return attn_res_mix(prefix_sum, block_residual, norm_weight, proj_weight, eps)

    return torch.cat(
        [
            attn_res_mix(
                prefix_sum[start : start + chunk],
                block_residual[start : start + chunk],
                norm_weight, proj_weight, eps,
            )
            for start in range(0, prefix_sum.shape[0], chunk)
        ],
        dim=0,
    )
