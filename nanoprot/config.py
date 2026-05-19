"""
nanoprot.config — typed, YAML-driven configuration for nanoprot training runs.

A nanoprot run is fully specified by a single YAML file that the user passes
to the training entry point. The schema is divided into sub-blocks
(``model``, ``tokenizer``, ``data``, ``optimizer``, ``training``, ``eval``,
``logging``, ``checkpointing``); each block is a Pydantic model with
sensible defaults so users only need to write the fields they want to
override.

The ``model`` block is a discriminated union over architecture:

- ``arch: gpt2``  -> :class:`Gpt2ModelConfig` (decoder-only, autoregressive)
- ``arch: esm2``  -> :class:`Esm2ModelConfig` (encoder-only, masked LM)

Adding a new architecture is a matter of writing a new ``ModelConfig``
subclass and appending it to the union. Each model config has a ``derive()``
method that fills in optional fields (e.g.\ ``d_model``, ``n_heads``) from a
single complexity dial (``depth``).

The ``training.objective`` field selects the training objective and which
data loader the training loop dispatches to:

- ``objective: ar``  -> autoregressive next-residue prediction (default; gpt2)
- ``objective: mlm`` -> BERT-style 15%/80/10/10 masked language modelling (esm2)

Environment-variable substitution is supported in any string field via the
standard ``$VAR`` and ``${VAR}`` syntax (resolved with ``os.path.expandvars``).

Usage
-----

>>> from nanoprot.config import load_config
>>> cfg = load_config("configs/gpt2_d20_uniref50.yaml")
>>> cfg.model.d_model
1280
>>> cfg.training.objective
'ar'
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ===========================================================================
# Model config: discriminated union by ``arch``
# ===========================================================================

class _ModelConfigBase(BaseModel):
    """Common base for architecture-specific model configs.

    Each subclass sets ``arch: Literal["xxx"]`` and implements ``derive()``
    to fill in any optional / derived fields after the YAML is loaded.
    """

    model_config = ConfigDict(extra="forbid")

    depth: int = Field(..., gt=0, le=128, description="Number of transformer layers.")
    d_model: Optional[int] = Field(
        default=None,
        description=(
            "Hidden dimension. If null, derived from depth + head_dim per the "
            "arch's rule."
        ),
    )
    n_heads: Optional[int] = Field(
        default=None,
        description="Number of attention heads. If null, derived from d_model / head_dim.",
    )
    n_kv_heads: Optional[int] = Field(
        default=None,
        description="Number of KV heads (Group-Query Attention). If null, equals n_heads.",
    )
    head_dim: int = Field(default=128, gt=0, description="Per-head dimension.")
    max_seq_len: int = Field(default=512, gt=0, description="Maximum context length.")
    vocab_size: int = Field(default=50_256, gt=0, description="Tokenizer vocabulary size.")
    rope: bool = Field(default=True, description="Use Rotary Position Embeddings.")

    def derive(self) -> "_ModelConfigBase":
        """Fill in any derived / optional fields. Subclasses override as needed."""
        return self

    def _derive_width_and_heads(self, multiplier: int) -> None:
        """Shared rule: ``d_model = multiplier * depth`` rounded up to head_dim."""
        if self.d_model is None:
            raw = self.depth * multiplier
            self.d_model = ((raw + self.head_dim - 1) // self.head_dim) * self.head_dim
        if self.n_heads is None:
            if self.d_model % self.head_dim != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) is not divisible by head_dim "
                    f"({self.head_dim})"
                )
            self.n_heads = self.d_model // self.head_dim
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads})"
            )


# ---------------------------------------------------------------------------
# gpt2: decoder-only autoregressive transformer
# ---------------------------------------------------------------------------

class Gpt2ModelConfig(_ModelConfigBase):
    """Configuration for a decoder-only GPT-2-style protein language model.

    Uses RoPE positional encoding, QK-RMSNorm, squared-ReLU MLPs, Group-Query
    Attention, optional sliding-window attention, and ResFormer-style value
    embeddings on alternating layers.

    Width rule: ``d_model = 64 * depth`` rounded up to a multiple of ``head_dim``.
    """

    arch: Literal["gpt2"] = "gpt2"
    qk_norm: bool = Field(
        default=True,
        description="Apply parameter-free RMSNorm to queries and keys before attention.",
    )
    mlp_activation: Literal["relu_squared", "gelu", "swiglu"] = "relu_squared"
    logit_softcap: Optional[float] = Field(
        default=15.0,
        description="If set, cap logits via softcap * tanh(logit / softcap). null disables.",
    )
    window_pattern: str = Field(
        default="L",
        description=(
            "Per-layer attention pattern as a string of 'S' (sliding) and 'L' (full); "
            "tiled across all layers. 'L' = full attention everywhere (SDPA-safe)."
        ),
    )

    @field_validator("window_pattern")
    @classmethod
    def _validate_window_pattern(cls, v: str) -> str:
        if not v:
            raise ValueError("window_pattern must be non-empty")
        if any(c not in "SL" for c in v.upper()):
            raise ValueError(f"window_pattern must contain only 'S' or 'L', got {v!r}")
        return v.upper()

    def derive(self) -> "Gpt2ModelConfig":
        self._derive_width_and_heads(multiplier=64)
        return self


# ---------------------------------------------------------------------------
# esm2: encoder-only masked-LM transformer
# ---------------------------------------------------------------------------

class Esm2ModelConfig(_ModelConfigBase):
    """Configuration for an ESM-2-style encoder-only masked-LM transformer.

    Bidirectional self-attention (no causal mask), pre-LayerNorm, GeLU MLP,
    RoPE positional encoding. Trained with BERT-style 15%/80/10/10 masking
    (see :class:`TrainingConfig.objective` = ``"mlm"``).

    Width rule: ``d_model = 40 * depth`` rounded up to head_dim (close to
    the ESM-2 size grid: 6L/320d, 12L/480d, 30L/640d, 33L/1280d).
    """

    arch: Literal["esm2"] = "esm2"
    head_dim: int = Field(default=64, gt=0, description="Per-head dimension (ESM-2 default 64).")
    vocab_size: int = Field(default=33, gt=0, description="ESM-2 vocab: 20 AAs + special tokens.")
    max_seq_len: int = Field(default=1024, gt=0)
    mlp_activation: Literal["gelu", "swiglu"] = "gelu"
    layer_norm: Literal["layernorm", "rmsnorm"] = Field(
        default="layernorm",
        description="ESM-2 originally uses LayerNorm; RMSNorm option for ablations.",
    )

    def derive(self) -> "Esm2ModelConfig":
        # ESM-2 uses smaller width-per-depth than GPT-2 in our scale grid.
        self._derive_width_and_heads(multiplier=40)
        return self


# ---------------------------------------------------------------------------
# mamba: selective state-space model
# ---------------------------------------------------------------------------

class MambaModelConfig(_ModelConfigBase):
    """Configuration for a Mamba-style selective state-space model.

    Pre-RMSNorm trunk. Each block: input projection -> depthwise 1-D conv
    -> SiLU -> selective SSM (input-dependent ``B``, ``C``, ``\\Delta``;
    diagonal ``A``) gated by a sigmoid-of-projection branch -> output
    projection. Causal by construction (the SSM scan is left-to-right),
    so it pairs with the autoregressive training objective.

    Width rule: ``d_model = 64 * depth`` rounded up to ``head_dim``
    (head_dim has no architectural meaning for Mamba, but is kept for
    the rounding-to-multiple convenience and for parity with the gpt2
    width grid).
    """

    arch: Literal["mamba"] = "mamba"
    head_dim: int = Field(default=64, gt=0, description="Rounding granularity for d_model.")
    n_kv_heads: Optional[int] = Field(
        default=1,
        description="Not used by Mamba; kept for schema compatibility.",
    )
    n_heads: Optional[int] = Field(default=1, description="Not used by Mamba.")
    d_state: int = Field(
        default=16,
        gt=0,
        description="Per-channel SSM hidden-state size (the 'N' in Mamba).",
    )
    d_conv: int = Field(
        default=4,
        gt=0,
        description="Kernel size of the depthwise causal 1-D convolution.",
    )
    expand: int = Field(
        default=2,
        gt=0,
        description="Inner-dimension expansion factor; ``d_inner = expand * d_model``.",
    )
    dt_rank: Optional[int] = Field(
        default=None,
        description=(
            "Rank of the input -> Delta projection. If null, set to "
            "``max(d_model // 16, 1)`` (matches the public Mamba defaults)."
        ),
    )
    layer_norm: Literal["rmsnorm", "layernorm"] = Field(
        default="rmsnorm",
        description="Mamba's reference implementation uses RMSNorm.",
    )

    def derive(self) -> "MambaModelConfig":
        # Width: like gpt2 (64 per depth), rounded to head_dim multiple.
        self._derive_width_and_heads(multiplier=64)
        if self.dt_rank is None:
            assert self.d_model is not None
            self.dt_rank = max(self.d_model // 16, 1)
        return self


# ---------------------------------------------------------------------------
# The discriminated union the rest of the schema uses
# ---------------------------------------------------------------------------

ModelConfig = Annotated[
    Union[Gpt2ModelConfig, Esm2ModelConfig, MambaModelConfig],
    Field(discriminator="arch"),
]


# ===========================================================================
# Tokenizer
# ===========================================================================

class TokenizerConfig(BaseModel):
    """Which tokenizer to use.

    - ``bpe``  : Byte-Pair Encoding trained on UniRef50, V=50,256 (gpt2).
    - ``esm2`` : 33-token ESM-2 alphabet (20 AAs + special tokens) (esm2).
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["bpe", "esm2"] = "bpe"
    vocab_size: Optional[int] = Field(
        default=None,
        description="Tokenizer vocabulary size. If null, uses the tokenizer's default.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Path to a trained tokenizer JSON. Only applies to ``bpe``.",
    )


# ===========================================================================
# Data
# ===========================================================================

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Literal["uniref50"] = "uniref50"
    shard_dir: str = Field(
        ...,
        description="Directory containing tokenized parquet shards. Supports $VAR.",
    )
    packing: Literal["bos_aligned_best_fit", "fixed_length"] = "bos_aligned_best_fit"
    val_shard: Literal["last", "first"] = Field(
        default="last",
        description="Which shard to hold out for validation.",
    )


# ===========================================================================
# Optimizer
# ===========================================================================

class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["muon_adamw", "adamw"] = "muon_adamw"
    matrix_lr: float = Field(default=0.02, gt=0, description="Muon LR for matrix params.")
    embedding_lr: float = Field(default=0.3, gt=0, description="AdamW LR for token embeddings.")
    unembedding_lr: float = Field(default=0.008, gt=0, description="AdamW LR for lm_head.")
    scalar_lr: float = Field(default=0.5, gt=0, description="AdamW LR for scalar params.")
    weight_decay: float = Field(default=0.28, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-10, gt=0)


# ===========================================================================
# Training
# ===========================================================================

class TrainingConfig(BaseModel):
    """Training-loop configuration: budget, batch sizes, schedule, objective."""

    model_config = ConfigDict(extra="forbid")

    objective: Literal["ar", "mlm"] = Field(
        default="ar",
        description=(
            "Training objective. ``ar`` (autoregressive next-residue prediction; "
            "for gpt2) or ``mlm`` (BERT-style masked LM; for esm2). Selects the "
            "data loader the training loop dispatches to."
        ),
    )
    mlm_probability: float = Field(
        default=0.15,
        gt=0,
        lt=1.0,
        description=(
            "Per-residue probability of selecting a token for masking under the "
            "MLM objective (BERT default 0.15). Of the selected tokens, 80% are "
            "replaced with [MASK], 10% with a random token, 10% left unchanged."
        ),
    )
    total_residues: Optional[float] = Field(
        default=None,
        description=(
            "Total training residues. If null, derived as "
            "param_data_ratio * n_params (Chinchilla-style)."
        ),
    )
    param_data_ratio: float = Field(
        default=12.0,
        gt=0,
        description="Residues per parameter when total_residues is auto-derived.",
    )
    total_batch_size: int = Field(
        default=524_288,
        gt=0,
        description="Residues per optimizer step. Held fixed across the entire sweep.",
    )
    device_batch_size: int = Field(default=32, gt=0)
    warmup_steps: int = Field(default=40, ge=0)
    warmdown_ratio: float = Field(default=0.65, ge=0, le=1)
    final_lr_frac: float = Field(default=0.05, gt=0, le=1)
    precision: Literal["fp32", "bf16", "fp8"] = "bf16"
    flash_attention: bool = Field(
        default=True,
        description="Use FlashAttention-3 when available. Falls back to PyTorch SDPA.",
    )
    seed: Optional[int] = Field(
        default=None,
        description=(
            "RNG seed for model init + data shuffling. If null, inherits from the "
            "top-level ``seed`` field."
        ),
    )


# ===========================================================================
# Eval, logging, checkpointing
# ===========================================================================

class EvalConfig(BaseModel):
    """Validation-pass cadence + metric."""

    model_config = ConfigDict(extra="forbid")

    eval_every: int = Field(
        default=250,
        description="Steps between validation passes. -1 disables eval entirely.",
    )
    eval_tokens: int = Field(
        default=80 * 524_288,
        gt=0,
        description=(
            "Approximate number of tokens to evaluate per pass. The training loop "
            "converts this to a batch count via the world tokens-per-batch."
        ),
    )
    metric: Literal["bpr", "loss"] = Field(
        default="bpr",
        description="Primary intrinsic metric. ``bpr`` = bits per residue.",
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = Field(default="default")
    wandb_project: str = Field(default="nanoprot")
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(
        ...,
        description="Directory where model_*.pt + meta_*.json land. Supports $VAR.",
    )
    save_every: int = Field(
        default=-1,
        description="Steps between intermediate checkpoint dumps. -1 = save only at end.",
    )
    save_steps: list[int] = Field(
        default_factory=list,
        description="Additional explicit step numbers at which to dump a checkpoint.",
    )


# ===========================================================================
# Top-level schema
# ===========================================================================

class NanoprotConfig(BaseModel):
    """The top-level config object for a nanoprot training run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="unnamed", description="Human-readable run identifier.")
    seed: int = Field(default=42, description="Global RNG seed.")

    model: ModelConfig
    tokenizer: TokenizerConfig = Field(default_factory=lambda: TokenizerConfig())
    data: DataConfig
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig())
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig())
    eval: EvalConfig = Field(default_factory=lambda: EvalConfig())
    logging: LoggingConfig = Field(default_factory=lambda: LoggingConfig())
    checkpointing: CheckpointConfig

    # -- pre-validation: default model.arch to "gpt2" for back-compat --------

    @model_validator(mode="before")
    @classmethod
    def _default_arch(cls, data: Any) -> Any:
        """If ``model.arch`` is omitted, default it to ``"gpt2"`` so older
        v0.2-style configs (written before the discriminated union) keep
        working without a manual edit."""
        if isinstance(data, dict):
            m = data.get("model")
            if isinstance(m, dict) and "arch" not in m:
                m["arch"] = "gpt2"
        return data

    # -- post-load derivations ------------------------------------------------

    @model_validator(mode="after")
    def _derive(self) -> "NanoprotConfig":
        # Per-arch derivations (d_model, n_heads, n_kv_heads, validation)
        self.model.derive()

        # Resolve environment variables in path-shaped string fields.
        self.data.shard_dir = os.path.expandvars(self.data.shard_dir)
        self.checkpointing.output_dir = os.path.expandvars(self.checkpointing.output_dir)
        if self.tokenizer.path is not None:
            self.tokenizer.path = os.path.expandvars(self.tokenizer.path)

        # If the user did not set training.seed explicitly, inherit from cfg.seed.
        if self.training.seed is None:
            self.training.seed = self.seed

        # Default tokenizer to the right one for the architecture if the user
        # didn't pick one explicitly. (We only override when name == "bpe"
        # AND arch != "gpt2", because "bpe" is the schema default.)
        # Users who explicitly set tokenizer.name keep their choice.

        return self

    # -- convenience ----------------------------------------------------------

    def estimate_params(self) -> int:
        """Closed-form lower-bound estimate of total parameter count.

        Used to derive ``training.total_residues`` when the user leaves it
        ``null`` (Chinchilla-style data budget). Counts:

        - transformer matrices: ``12 d^2 per layer``;
        - token embedding + untied lm_head: ``2 V d``;
        - for ``arch="gpt2"`` only, ResFormer value embeddings on alternating
          layers: ``ceil(n_layer / 2) * V * d_kv``.

        The real parameter count is logged by the training loop after the
        model is instantiated.
        """
        m = self.model
        assert m.d_model is not None and m.n_kv_heads is not None  # derived
        d = m.d_model
        n_layer = m.depth
        V = m.vocab_size
        d_kv = m.n_kv_heads * m.head_dim
        n = 12 * d * d * n_layer + 2 * V * d
        if m.arch == "gpt2":
            n_ve_layers = (n_layer + 1) // 2
            n += n_ve_layers * V * d_kv
        return n

    def total_residues(self) -> int:
        """Total residues to train on (explicit or Chinchilla-derived)."""
        if self.training.total_residues is not None:
            return int(self.training.total_residues)
        return int(self.training.param_data_ratio * self.estimate_params())


# ===========================================================================
# Loaders
# ===========================================================================

def load_config(path: Union[str, Path]) -> NanoprotConfig:
    """Load and validate a nanoprot YAML config.

    Default the ``model.arch`` field to ``"gpt2"`` if missing, so older v0.2
    configs (which omitted the discriminator) keep working.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No config file at {path}")
    with path.open("r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    # Back-compat: if model.arch is missing, default it to "gpt2".
    if "model" in raw and isinstance(raw["model"], dict):
        raw["model"].setdefault("arch", "gpt2")
    return NanoprotConfig.model_validate(raw)


def dump_config(cfg: NanoprotConfig, path: Union[str, Path]) -> None:
    """Round-trip a config back to YAML (post-derivation values included)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(cfg.model_dump(), fh, sort_keys=False, default_flow_style=False)
