"""
ESM-2-style encoder-only masked-LM transformer.

Notable features:
- Bidirectional self-attention (no causal mask)
- Pre-LayerNorm (LayerNorm before each sub-block)
- RoPE positional encoding (same as GPT-2 path)
- Group-Query Attention support
- GeLU activation in MLP
- Simple linear LM head over the full vocabulary
- BERT-style init (Normal(0, 0.02))

The model exposes the same surface as :class:`nanoprot.models.gpt2.GPT`:

- ``forward(idx, targets=None)`` -> loss when targets is given, logits otherwise
- ``init_weights()``
- ``setup_optimizer(...)`` -> mixed Muon (matrix params) + AdamW (embeddings, scalars)
- ``num_scaling_params()``
- ``estimate_flops()``

so the training loop can drive it through the same interface regardless of
which architecture the user picked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoprot.attention import flash_attn
from nanoprot.optim import DistMuonAdamW, MuonAdamW
from nanoprot.runtime import COMPUTE_DTYPE, get_dist_info, print0


# ---------------------------------------------------------------------------
# Config + utility layers
# ---------------------------------------------------------------------------

@dataclass
class Esm2Config:
    """Compile-time config consumed by :class:`Esm2`."""

    sequence_len: int = 1024
    vocab_size: int = 33
    n_layer: int = 6
    n_head: int = 20
    n_kv_head: int = 20
    n_embd: int = 320
    layer_norm: str = "layernorm"  # "layernorm" | "rmsnorm"
    mlp_activation: str = "gelu"


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.

    Mirrors the custom Linear used by the gpt2 model: master weights stay
    fp32 for the optimizer, matmuls run in the activation dtype.
    """

    def forward(self, x):  # noqa: D401
        return F.linear(x, self.weight.to(dtype=x.dtype))


def _build_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=1e-5)
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=1e-5)
    raise ValueError(f"unknown norm kind {kind!r}; use 'layernorm' or 'rmsnorm'")


def _apply_rotary_emb(x, cos, sin):
    """Apply RoPE to (B, T, H, D) tensor."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BidirectionalAttention(nn.Module):
    """Multi-head self-attention without causal mask.

    Uses RoPE on Q/K, supports Group-Query Attention, dispatches to
    FlashAttention-3 when available with PyTorch SDPA fallback.
    """

    def __init__(self, config: Esm2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin) -> torch.Tensor:
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        # Bidirectional: causal=False, no sliding window.
        y = flash_attn.flash_attn_func(q, k, v, causal=False, window_size=(-1, -1))
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class Mlp(nn.Module):
    """Two-layer MLP with GeLU activation."""

    def __init__(self, config: Esm2Config) -> None:
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Esm2Block(nn.Module):
    """One Pre-LN transformer block: x + attn(LN(x)); x + mlp(LN(x))."""

    def __init__(self, config: Esm2Config) -> None:
        super().__init__()
        self.ln1 = _build_norm(config.layer_norm, config.n_embd)
        self.attn = BidirectionalAttention(config)
        self.ln2 = _build_norm(config.layer_norm, config.n_embd)
        self.mlp = Mlp(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(self.ln1(x), cos_sin)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class Esm2(nn.Module):
    """ESM-2-style encoder-only masked-LM transformer.

    Designed to be instantiated under ``torch.device("meta")`` (so the
    constructor only allocates shape information) and then materialised via
    ``model.to_empty(device=...)`` + :meth:`init_weights`.
    """

    def __init__(self, config: Esm2Config, pad_vocab_size_to: int = 64) -> None:
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
                    [Esm2Block(config) for _ in range(config.n_layer)]
                ),
                "ln_f": _build_norm(config.layer_norm, config.n_embd),
            }
        )
        self.lm_head = Linear(config.n_embd, padded, bias=False)

        # Precompute RoPE table on meta so init_weights() can rematerialize on device.
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, config.n_embd // config.n_head
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # -- weight init --------------------------------------------------------

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialise on the current (real) device.

        BERT-style init: Normal(0, 0.02) for linear + embedding weights;
        LayerNorm weights -> 1, biases -> 0 (RMSNorm has no learnable bias).
        """
        cfg = self.config
        std = 0.02
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        for block in self.transformer.h:
            for lin in (
                block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.c_proj,
                block.mlp.c_fc, block.mlp.c_proj,
            ):
                nn.init.normal_(lin.weight, mean=0.0, std=std)
            for norm in (block.ln1, block.ln2):
                if hasattr(norm, "weight") and norm.weight is not None:
                    nn.init.ones_(norm.weight)
                if hasattr(norm, "bias") and norm.bias is not None:
                    nn.init.zeros_(norm.bias)
        if hasattr(self.transformer.ln_f, "weight") and self.transformer.ln_f.weight is not None:
            nn.init.ones_(self.transformer.ln_f.weight)
        if hasattr(self.transformer.ln_f, "bias") and self.transformer.ln_f.bias is not None:
            nn.init.zeros_(self.transformer.ln_f.bias)

        # RoPE tables on the real device + compute dtype.
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, cfg.n_embd // cfg.n_head
        )
        self.cos, self.sin = cos, sin

        # Cast token embedding to COMPUTE_DTYPE for memory savings.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: float = 10_000.0):
        device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.to(COMPUTE_DTYPE)[None, :, None, :]
        sin = sin.to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    # -- scaling-law accounting ---------------------------------------------

    def num_scaling_params(self) -> dict:
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = 0
        layernorm_params = 0
        for block in self.transformer.h:
            for lin in (
                block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.c_proj,
                block.mlp.c_fc, block.mlp.c_proj,
            ):
                transformer_matrices += sum(p.numel() for p in lin.parameters())
            for norm in (block.ln1, block.ln2):
                layernorm_params += sum(p.numel() for p in norm.parameters())
        layernorm_params += sum(p.numel() for p in self.transformer.ln_f.parameters())
        total = wte + lm_head + transformer_matrices + layernorm_params
        assert total == sum(p.numel() for p in self.parameters()), "param count mismatch"
        return {
            "wte": wte,
            "value_embeds": 0,  # ESM-2 has none; field kept for cross-arch logging parity
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": layernorm_params,
            "total": total,
        }

    def estimate_flops(self) -> int:
        nparams = sum(p.numel() for p in self.parameters())
        non_matmul = (
            self.transformer.wte.weight.numel()
            + sum(p.numel() for p in self.transformer.ln_f.parameters())
        )
        for block in self.transformer.h:
            for norm in (block.ln1, block.ln2):
                non_matmul += sum(p.numel() for p in norm.parameters())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # Bidirectional attention is O(T^2) per layer per head; same flop
        # accounting as causal for the matmul part (causal saves only on the
        # softmax half; PaLM-style estimate ignores that).
        attn_flops = 12 * h * q * t * self.config.n_layer
        return 6 * (nparams - non_matmul) + attn_flops

    # -- forward ------------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets=None, loss_reduction: str = "mean"):
        """Forward pass.

        Parameters
        ----------
        idx : (B, T) int64
            Input token ids. For MLM, masked positions hold the [MASK] token.
        targets : (B, T) int64, optional
            If given, original token ids at masked positions and the
            ``-100`` ignore_index elsewhere. The returned loss is averaged
            over the non-ignore positions only.
        loss_reduction : str
            Passed through to ``F.cross_entropy``.
        """
        B, T = idx.size()
        assert T <= self.cos.size(1), f"sequence length {T} exceeds rotary cache {self.cos.size(1)}"

        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size]
        logits = logits.float()

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
        """Mixed Muon + AdamW optimizer, matching :meth:`Gpt2.setup_optimizer`."""
        model_dim = self.config.n_embd
        ddp, *_ = get_dist_info()

        matrix_params: list[nn.Parameter] = []
        for block in self.transformer.h:
            for lin in (
                block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.c_proj,
                block.mlp.c_fc, block.mlp.c_proj,
            ):
                matrix_params.extend(lin.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        scalar_params: list[nn.Parameter] = []
        for block in self.transformer.h:
            for norm in (block.ln1, block.ln2):
                scalar_params.extend(norm.parameters())
        scalar_params.extend(self.transformer.ln_f.parameters())

        total_param_count = (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(scalar_params)
        )
        assert total_param_count == len(list(self.parameters())), "esm2: optimizer param-group mismatch"

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


# ---------------------------------------------------------------------------
# Public alias (mirrors GPT class name from gpt2 module)
# ---------------------------------------------------------------------------

ESM2Config = Esm2Config  # alias for consistency with the public registry
ESM2 = Esm2
