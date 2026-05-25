"""
Mamba-style selective state-space model (SSM) for protein language modelling.

Architecture (per block, Pre-LN):

::

    x -> RMSNorm -> in_proj -> [x_inner, z_gate]
              x_inner -> conv1d (depthwise, causal) -> SiLU -> selective_scan
              z_gate  -> SiLU
              y = scan_out * z_gate -> out_proj -> + residual

The selective SSM follows Mamba (Gu & Dao, 2023):

::

    h_t = exp(\\Delta_t \\odot A) \\odot h_{t-1}  +  (\\Delta_t \\odot B_t) x_t
    y_t = C_t^\\top h_t

with ``A`` a learned ``(d_inner, d_state)`` diagonal selectivity, and
``\\Delta_t``, ``B_t``, ``C_t`` input-dependent (linear projections of
``x_inner``). A learned skip ``D \\odot x_inner`` is added to the scan
output for stability.

The scan is implemented in pure PyTorch (``selective_scan_ref``) so this
file is portable across CPU / CUDA / MPS without depending on the
``mamba_ssm`` CUDA kernels. The reference scan is O(L) sequential and is
fine for the CPU smoke tests + tiny ablations; users targeting full-scale
training on Hopper should drop in the official kernel via a small adapter.

The model exposes the same surface as :class:`nanoprot.models.gpt2.GPT`:

- ``forward(idx, targets=None)`` -> loss (AR cross-entropy) when targets
  is given, logits otherwise
- ``init_weights()``
- ``setup_optimizer(...)`` -> mixed Muon (matrix params) + AdamW (the rest)
- ``num_scaling_params()`` / ``estimate_flops()``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoprot.optim import DistMuonAdamW, MuonAdamW
from nanoprot.runtime import COMPUTE_DTYPE, get_dist_info, print0


# ---------------------------------------------------------------------------
# Config + utility layers
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    """Compile-time config consumed by :class:`Mamba`."""

    sequence_len: int = 1024
    vocab_size: int = 50_256
    n_layer: int = 6
    n_embd: int = 384       # d_model
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 24       # default = max(n_embd // 16, 1)
    layer_norm: str = "rmsnorm"
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4


class Linear(nn.Linear):
    """nn.Linear that casts its weights to match input dtype in forward.

    Mirrors the custom Linear used in :mod:`nanoprot.models.gpt2` and
    :mod:`nanoprot.models.esm2` so master weights stay in fp32 for the
    optimizer while matmuls run in the activation dtype.
    """

    def forward(self, x):  # noqa: D401
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias if self.bias is None else self.bias.to(dtype=x.dtype))


def _build_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=1e-5)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=1e-5)
    raise ValueError(f"unknown norm kind {kind!r}; use 'rmsnorm' or 'layernorm'")


# ---------------------------------------------------------------------------
# Selective SSM scan (pure PyTorch reference)
# ---------------------------------------------------------------------------

def selective_scan_ref(
    x: torch.Tensor,           # (B, L, D_inner)
    dt: torch.Tensor,          # (B, L, D_inner)
    A: torch.Tensor,           # (D_inner, N)            — learned, negative
    B_t: torch.Tensor,         # (B, L, N)
    C_t: torch.Tensor,         # (B, L, N)
    D_skip: torch.Tensor,      # (D_inner,)              — learned skip
) -> torch.Tensor:
    """Reference selective-scan implementation.

    Computes the recurrence

    ::

        h_t = exp(dt_t \\odot A) \\odot h_{t-1} + (dt_t \\odot B_t) x_t
        y_t = C_t^\\top h_t  +  D_skip \\odot x_t

    in a serial Python loop. Correct on any device (CPU / GPU / MPS) but
    O(L) sequential; replace with a fused CUDA kernel for full-scale
    training on Hopper.

    All internal arithmetic runs in fp32 even when inputs are bf16/fp16.
    Recurrences accumulate round-off step-by-step, so the recurrent state
    ``h`` and its inputs (``dt * A``, ``exp(dt*A)``, ``dt * B``) must be
    held at full precision to avoid drift over the sequence length. This
    matches the upcasting performed by the official Mamba reference
    implementation. The output is cast back to the input dtype before
    return so callers see the same dtype they passed in.
    """
    in_dtype = x.dtype
    # Upcast everything to fp32 for the recurrence.
    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B_t.float()
    C_f = C_t.float()
    D_f = D_skip.float()

    B, L, D = x_f.shape
    N = A_f.shape[1]

    # Discretize.
    dt_A = dt_f.unsqueeze(-1) * A_f                # (B, L, D, N)
    dA = torch.exp(dt_A)                           # (B, L, D, N)
    dB = dt_f.unsqueeze(-1) * B_f.unsqueeze(2)     # (B, L, D, N)

    h = torch.zeros(B, D, N, device=x_f.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * x_f[:, t].unsqueeze(-1)   # (B, D, N)
        y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)          # (B, D)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                                  # (B, L, D)
    y = y + D_f.unsqueeze(0).unsqueeze(0) * x_f                 # skip
    return y.to(in_dtype)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """One selective-SSM block with gating + depthwise causal conv."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        D = config.n_embd
        D_inner = config.expand * D
        N = config.d_state
        dt_rank = config.dt_rank
        K = config.d_conv

        self.in_proj = Linear(D, 2 * D_inner, bias=False)

        # Depthwise causal conv: shape (D_inner, 1, K). We pad-left by K-1
        # then call conv1d with padding=0 so the kernel only sees past + self.
        self.conv1d = nn.Conv1d(
            in_channels=D_inner, out_channels=D_inner, kernel_size=K,
            padding=0, groups=D_inner, bias=True,
        )

        # Selectivity projections.
        self.x_proj = Linear(D_inner, dt_rank + 2 * N, bias=False)
        self.dt_proj = Linear(dt_rank, D_inner, bias=True)

        # Learned SSM parameters.
        # A is stored as A_log = log(-A) so A = -exp(A_log) is always negative
        # (decay) and A_log can be a free real parameter.
        # We use torch.zeros (not torch.empty) so that direct construction
        # without going through meta + to_empty + init_weights() yields a
        # mathematically valid (if unconverged) model rather than NaN-prone
        # uninitialized memory. init_weights() overwrites both regardless.
        self.A_log = nn.Parameter(torch.zeros(D_inner, N))
        self.D = nn.Parameter(torch.zeros(D_inner))

        self.out_proj = Linear(D_inner, D, bias=False)
        self.K = K
        self.D_inner = D_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        xz = self.in_proj(x)                                    # (B, L, 2*D_inner)
        x_inner, z_gate = xz.chunk(2, dim=-1)                   # each (B, L, D_inner)

        # Depthwise causal conv: (B, D_inner, L) with left-padding K-1.
        # We call F.conv1d directly so we can cast weight + bias to match the
        # activation dtype (mirrors the dtype-casting we do in Linear). The
        # canonical nn.Conv1d would error on bf16 input + fp32 master weights.
        x_inner = x_inner.transpose(1, 2)                       # (B, D_inner, L)
        x_inner = F.pad(x_inner, (self.K - 1, 0))               # (B, D_inner, L + K-1)
        w = self.conv1d.weight.to(x_inner.dtype)
        b = self.conv1d.bias.to(x_inner.dtype) if self.conv1d.bias is not None else None
        x_inner = F.conv1d(x_inner, w, b, stride=1, padding=0, groups=self.D_inner)
        x_inner = x_inner.transpose(1, 2)                       # (B, L, D_inner)
        x_inner = F.silu(x_inner)

        # Compute Δ, B, C from x_inner.
        x_dbl = self.x_proj(x_inner)                            # (B, L, dt_rank + 2N)
        dt_rank = self.config.dt_rank
        N = self.config.d_state
        dt = x_dbl[..., :dt_rank]                               # (B, L, dt_rank)
        B_t = x_dbl[..., dt_rank: dt_rank + N]                  # (B, L, N)
        C_t = x_dbl[..., dt_rank + N:]                          # (B, L, N)
        dt = F.softplus(self.dt_proj(dt))                       # (B, L, D_inner)

        A = -torch.exp(self.A_log.to(x_inner.dtype))            # (D_inner, N) negative

        y = selective_scan_ref(x_inner, dt, A, B_t, C_t, self.D.to(x_inner.dtype))
        y = y * F.silu(z_gate)
        return self.out_proj(y)


class MambaResidualBlock(nn.Module):
    """Pre-LN wrapper: x + MambaBlock(Norm(x))."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.norm = _build_norm(config.layer_norm, config.n_embd)
        self.mixer = MambaBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class Mamba(nn.Module):
    """Mamba-style selective SSM language model.

    Designed for the meta-device + ``to_empty(...) + init_weights()`` build
    pattern. Causal by construction (the SSM scan only reads past tokens),
    so the AR training objective dispatches directly to this model.
    """

    def __init__(self, config: MambaConfig, pad_vocab_size_to: int = 64) -> None:
        super().__init__()
        self.config = config
        padded = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded} for efficiency")
        self.padded_vocab_size = padded

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded, config.n_embd),
                "h": nn.ModuleList(
                    [MambaResidualBlock(config) for _ in range(config.n_layer)]
                ),
                "ln_f": _build_norm(config.layer_norm, config.n_embd),
            }
        )
        self.lm_head = Linear(config.n_embd, padded, bias=False)

    # -- weight init --------------------------------------------------------

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialise on the current device."""
        cfg = self.config
        std = 0.02
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

        for resblock in self.transformer.h:
            blk: MambaBlock = resblock.mixer
            nn.init.normal_(blk.in_proj.weight, mean=0.0, std=std)
            nn.init.normal_(blk.x_proj.weight, mean=0.0, std=std)
            nn.init.normal_(blk.out_proj.weight, mean=0.0, std=std)

            # Δ-projection bias initialised so softplus(bias) is in [dt_min, dt_max],
            # matching the Mamba paper recipe. Allocate scratch on the param's
            # own device so we don't cross PCIe for a single bias init.
            nn.init.normal_(blk.dt_proj.weight, mean=0.0, std=std)
            dev = blk.dt_proj.bias.device
            dt_min_log = math.log(cfg.dt_min)
            dt_max_log = math.log(cfg.dt_max)
            dt_init = torch.exp(
                torch.empty(blk.D_inner, device=dev).uniform_() * (dt_max_log - dt_min_log) + dt_min_log
            ).clamp(min=cfg.dt_init_floor)
            # softplus^{-1}(x) = x + log(-expm1(-x))  (numerically stable for positive x)
            dt_inv = dt_init + torch.log(-torch.expm1(-dt_init))
            blk.dt_proj.bias.copy_(dt_inv)

            # Conv1d weight: small normal init; bias zero.
            nn.init.normal_(blk.conv1d.weight, mean=0.0, std=std)
            if blk.conv1d.bias is not None:
                nn.init.zeros_(blk.conv1d.bias)

            # A_log: A = -exp(A_log). Initialise A to -1..-d_state per channel
            # (standard S4D-Real / Mamba init: A_n = -n for n=1..d_state).
            n = torch.arange(1, cfg.d_state + 1, dtype=torch.float32, device=blk.A_log.device)
            A_init = torch.log(n.unsqueeze(0).expand(blk.D_inner, -1))
            blk.A_log.copy_(A_init)

            # D (skip): start at 1 so the early-training pass-through dominates.
            nn.init.ones_(blk.D)

            # Norm.
            if hasattr(resblock.norm, "weight") and resblock.norm.weight is not None:
                nn.init.ones_(resblock.norm.weight)
            if hasattr(resblock.norm, "bias") and resblock.norm.bias is not None:
                nn.init.zeros_(resblock.norm.bias)

        if hasattr(self.transformer.ln_f, "weight") and self.transformer.ln_f.weight is not None:
            nn.init.ones_(self.transformer.ln_f.weight)
        if hasattr(self.transformer.ln_f, "bias") and self.transformer.ln_f.bias is not None:
            nn.init.zeros_(self.transformer.ln_f.bias)

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    # -- scaling-law accounting ---------------------------------------------

    def num_scaling_params(self) -> dict:
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        matrices = 0
        scalars = 0
        for resblock in self.transformer.h:
            blk: MambaBlock = resblock.mixer
            for mod in (blk.in_proj, blk.x_proj, blk.dt_proj, blk.out_proj, blk.conv1d):
                matrices += sum(p.numel() for p in mod.parameters())
            scalars += blk.A_log.numel() + blk.D.numel()
            scalars += sum(p.numel() for p in resblock.norm.parameters())
        scalars += sum(p.numel() for p in self.transformer.ln_f.parameters())
        total = wte + lm_head + matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "param count mismatch"
        return {
            "wte": wte,
            "value_embeds": 0,
            "lm_head": lm_head,
            "transformer_matrices": matrices,
            "scalars": scalars,
            "total": total,
        }

    def estimate_flops(self) -> int:
        nparams = sum(p.numel() for p in self.parameters())
        non_matmul = (
            self.transformer.wte.weight.numel()
            + sum(p.numel() for p in self.transformer.ln_f.parameters())
        )
        for resblock in self.transformer.h:
            blk = resblock.mixer
            non_matmul += blk.A_log.numel() + blk.D.numel()
            non_matmul += sum(p.numel() for p in resblock.norm.parameters())
        cfg = self.config
        # Selective SSM scan: per token O(D_inner * N) for h update + C dot,
        # times sequence length.
        D_inner = cfg.expand * cfg.n_embd
        scan_flops = 6 * D_inner * cfg.d_state * cfg.sequence_len * cfg.n_layer
        return 6 * (nparams - non_matmul) + scan_flops

    # -- forward ------------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets=None, loss_reduction: str = "mean"):
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        for resblock in self.transformer.h:
            x = resblock(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size].float()

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction=loss_reduction,
            )
            if loss_reduction == "none":
                loss = loss.view(targets.shape)
            return loss
        return logits

    # -- optimizer setup ----------------------------------------------------

    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        scalar_lr: float = 0.5,
    ):
        """Mixed Muon (linear weights) + AdamW (everything else).

        Conv1d weights are 3-D (``(D_inner, 1, K)``) so they go to AdamW;
        Mamba's small selectivity tensors (``A_log``, ``D``) and the
        LayerNorm scalars also go to AdamW. Matrix-shaped projection
        weights (``in_proj``, ``x_proj``, ``dt_proj``, ``out_proj``) take
        the Muon path, grouped by shape for stacking.
        """
        model_dim = self.config.n_embd
        ddp, *_ = get_dist_info()

        matrix_params: list[nn.Parameter] = []
        adam_misc: list[nn.Parameter] = []
        scalar_params: list[nn.Parameter] = []
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        for resblock in self.transformer.h:
            blk: MambaBlock = resblock.mixer
            for mod in (blk.in_proj, blk.x_proj, blk.dt_proj, blk.out_proj):
                matrix_params.extend(p for p in mod.parameters() if p.ndim == 2)
                adam_misc.extend(p for p in mod.parameters() if p.ndim != 2)
            adam_misc.extend(blk.conv1d.parameters())   # 3-D weight + 1-D bias
            scalar_params.extend([blk.A_log, blk.D])
            scalar_params.extend(resblock.norm.parameters())
        scalar_params.extend(self.transformer.ln_f.parameters())

        total = (
            len(matrix_params) + len(adam_misc) + len(scalar_params)
            + len(embedding_params) + len(lm_head_params)
        )
        assert total == len(list(self.parameters())), "mamba: optimizer param-group mismatch"

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(
                kind="adamw", params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01,
            ),
            dict(
                kind="adamw", params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001,
            ),
        ]
        if adam_misc:
            param_groups.append(dict(
                kind="adamw", params=adam_misc,
                lr=matrix_lr * dmodel_lr_scale,
                betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0,
            ))
        if scalar_params:
            param_groups.append(dict(
                kind="adamw", params=scalar_params,
                lr=scalar_lr * 0.01,
                betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
            ))
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind="muon", params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer


# Public alias (mirrors GPT / ESM2 naming).
MAMBA = Mamba
MAMBAConfig = MambaConfig
