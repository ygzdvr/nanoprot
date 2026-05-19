"""
nanoprot.config — typed, YAML-driven configuration for nanoprot training runs.

A nanoprot run is fully specified by a single YAML file that the user passes
to the training entry point. The schema is divided into sub-blocks
(``model``, ``tokenizer``, ``data``, ``optimizer``, ``training``, ``eval``,
``logging``, ``checkpointing``); each block is a Pydantic model with
sensible defaults so users only need to write the fields they want to
override.

Several fields are intentionally optional (``model.d_model``,
``model.n_heads``, ``training.total_residues``); they are filled in by
``NanoprotConfig.derive()`` after the YAML is loaded, so the user can either
specify them explicitly or let nanoprot derive them from a single complexity
dial (``model.depth``).

Environment-variable substitution is supported in any string field via the
standard ``$VAR`` and ``${VAR}`` syntax (resolved with ``os.path.expandvars``).

Usage
-----

>>> from nanoprot.config import load_config
>>> cfg = load_config("configs/gpt2_d20_uniref50.yaml")
>>> cfg.model.d_model
1280
>>> cfg.training.total_residues
5489000000.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-block schemas
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Architectural choices.

    Only ``depth`` is required at YAML-write time; ``d_model`` and ``n_heads``
    are derived from ``depth`` and ``head_dim`` if left ``null``.
    """

    model_config = ConfigDict(extra="forbid")

    arch: Literal["gpt2"] = Field(
        default="gpt2",
        description=(
            "Model architecture family. v0.1 supports 'gpt2' (decoder-only). "
            "v0.2 will add 'esm2' (masked encoder); v0.3 will add 'mamba'."
        ),
    )
    depth: int = Field(..., gt=0, le=128, description="Number of transformer layers.")
    d_model: Optional[int] = Field(
        default=None,
        description=(
            "Hidden dimension. If null, derived as depth * 64 rounded up to a "
            "multiple of head_dim."
        ),
    )
    n_heads: Optional[int] = Field(
        default=None,
        description="Number of attention heads. If null, derived as d_model / head_dim.",
    )
    n_kv_heads: Optional[int] = Field(
        default=None,
        description="Number of KV heads (Group-Query Attention). If null, equal to n_heads.",
    )
    head_dim: int = Field(default=128, gt=0, description="Per-head dimension.")
    max_seq_len: int = Field(default=512, gt=0, description="Maximum context length.")
    vocab_size: int = Field(
        default=50_256,
        gt=0,
        description="Tokenizer vocabulary size (padded to multiple of 64 for kernel efficiency).",
    )
    rope: bool = Field(default=True, description="Use Rotary Position Embeddings.")
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


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["bpe"] = "bpe"
    vocab_size: int = Field(default=50_256, gt=0)
    path: Optional[str] = Field(
        default=None,
        description="Path to a trained tokenizer JSON. If null, train one from the data.",
    )


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Literal["uniref50"] = "uniref50"
    shard_dir: str = Field(
        ...,
        description="Directory containing tokenized parquet shards. Supports $VAR substitution.",
    )
    packing: Literal["bos_aligned_best_fit", "fixed_length"] = "bos_aligned_best_fit"
    val_shard: Literal["last", "first"] = Field(
        default="last",
        description="Which shard to hold out for validation.",
    )


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


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
        description="Residues per parameter when total_residues is auto-derived (Chinchilla-optimal ≈ 20).",
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
        description="Use FlashAttention-3 when available (Hopper GPUs). Falls back to PyTorch SDPA.",
    )
    seed: int = Field(default=42, description="RNG seed for model init + data shuffling.")


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_every: int = Field(default=250, description="Steps between validation passes. -1 disables.")
    eval_tokens: int = Field(default=80 * 524_288, gt=0)
    metric: Literal["bpr", "loss"] = "bpr"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = Field(default="default")
    wandb_project: str = Field(default="nanoprot")
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(
        ...,
        description="Directory where model_*.pt + meta_*.json land. Supports $VAR substitution.",
    )
    save_every: int = Field(
        default=-1,
        description="Steps between intermediate checkpoint dumps. -1 = save only at end.",
    )
    save_steps: list[int] = Field(
        default_factory=list,
        description="Additional explicit step numbers at which to dump a checkpoint.",
    )


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------

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

    # -- post-load derivations ------------------------------------------------

    @model_validator(mode="after")
    def _derive(self) -> "NanoprotConfig":
        # Derive d_model and n_heads from depth if left null.
        m = self.model
        if m.d_model is None:
            raw = m.depth * 64
            # round up to a multiple of head_dim for kernel efficiency
            m.d_model = ((raw + m.head_dim - 1) // m.head_dim) * m.head_dim
        if m.n_heads is None:
            if m.d_model % m.head_dim != 0:
                raise ValueError(
                    f"d_model ({m.d_model}) is not divisible by head_dim ({m.head_dim})"
                )
            m.n_heads = m.d_model // m.head_dim
        if m.n_kv_heads is None:
            m.n_kv_heads = m.n_heads
        if m.n_heads % m.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({m.n_heads}) must be divisible by n_kv_heads ({m.n_kv_heads})"
            )

        # Resolve environment variables in path-shaped string fields.
        self.data.shard_dir = os.path.expandvars(self.data.shard_dir)
        self.checkpointing.output_dir = os.path.expandvars(self.checkpointing.output_dir)
        if self.tokenizer.path is not None:
            self.tokenizer.path = os.path.expandvars(self.tokenizer.path)

        # Sync per-run seeds if the user only set the global one.
        if self.training.seed == 42 and self.seed != 42:
            self.training.seed = self.seed

        return self

    # -- convenience ----------------------------------------------------------

    def estimate_params(self) -> int:
        """Cheap closed-form estimate of total parameter count.

        Used to derive ``training.total_residues`` when the user leaves it
        ``null`` (Chinchilla-style data budget). Not exact — assumes a
        standard transformer with tied embeddings and 12 d_model^2 per layer.
        The real parameter count is logged by the training script after the
        model is instantiated.
        """
        d = self.model.d_model
        assert d is not None  # derived in @model_validator
        n_layer = self.model.depth
        V = self.model.vocab_size
        # 12 * d^2 per layer (4d^2 attention + 8d^2 MLP), embedding + lm_head
        return 12 * d * d * n_layer + 2 * V * d

    def total_residues(self) -> int:
        """Total residues to train on (explicit or Chinchilla-derived)."""
        if self.training.total_residues is not None:
            return int(self.training.total_residues)
        return int(self.training.param_data_ratio * self.estimate_params())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> NanoprotConfig:
    """Load and validate a nanoprot YAML config.

    Parameters
    ----------
    path : str or Path
        Path to a YAML file matching the :class:`NanoprotConfig` schema.

    Returns
    -------
    NanoprotConfig
        The validated, derivation-filled config object.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    pydantic.ValidationError
        If the YAML structure does not match the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No config file at {path}")
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    return NanoprotConfig.model_validate(raw)


def dump_config(cfg: NanoprotConfig, path: Union[str, Path]) -> None:
    """Round-trip a config back to YAML (post-derivation values included)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(cfg.model_dump(), fh, sort_keys=False, default_flow_style=False)
