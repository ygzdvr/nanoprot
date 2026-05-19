"""Model architectures.

v0.2 shipped GPT-2-style decoder-only transformers; v0.3 adds ESM-2-style
masked-LM encoders. v0.4 will add Mamba / SSM. New architectures register
themselves via :func:`register_model` so they show up in :func:`build_model`.
"""

from __future__ import annotations

from typing import Callable, Dict, Union

from nanoprot.config import (
    Esm2ModelConfig,
    Gpt2ModelConfig,
    _ModelConfigBase,
)
from nanoprot.models.esm2 import ESM2, ESM2Config, Esm2
from nanoprot.models.gpt2 import GPT, GPTConfig

__all__ = [
    "GPT",
    "GPTConfig",
    "Esm2",
    "ESM2",
    "ESM2Config",
    "build_model",
    "register_model",
    "list_archs",
]


# Architecture registry: name -> factory(model_cfg) -> nn.Module
_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str, factory: Callable) -> None:
    """Register a model factory under ``name``.

    The factory must accept a config (subclass of ``_ModelConfigBase``) and
    return an instantiated ``nn.Module``.
    """
    if name in _REGISTRY:
        raise ValueError(f"model arch {name!r} already registered")
    _REGISTRY[name] = factory


def list_archs() -> list[str]:
    """Return the list of registered model architectures."""
    return sorted(_REGISTRY)


def build_model(cfg: Union[Gpt2ModelConfig, Esm2ModelConfig, _ModelConfigBase]):
    """Build a model from a per-arch config.

    The returned ``nn.Module`` is on whatever device the caller's
    ``torch.device(...)`` context was set to; for the lazy-init pattern,
    call inside ``with torch.device("meta"):`` then ``model.to_empty(...)``
    and ``model.init_weights()``.
    """
    arch = cfg.arch
    if arch not in _REGISTRY:
        raise ValueError(
            f"unknown model arch {arch!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[arch](cfg)


# -- gpt2 factory ----------------------------------------------------------

def _build_gpt2(cfg: Gpt2ModelConfig) -> GPT:
    legacy = GPTConfig(
        sequence_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.depth,
        n_head=cfg.n_heads,  # type: ignore[arg-type]
        n_kv_head=cfg.n_kv_heads,  # type: ignore[arg-type]
        n_embd=cfg.d_model,  # type: ignore[arg-type]
        window_pattern=cfg.window_pattern,
    )
    return GPT(legacy)


register_model("gpt2", _build_gpt2)


# -- esm2 factory ----------------------------------------------------------

def _build_esm2(cfg: Esm2ModelConfig) -> Esm2:
    legacy = ESM2Config(
        sequence_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.depth,
        n_head=cfg.n_heads,  # type: ignore[arg-type]
        n_kv_head=cfg.n_kv_heads,  # type: ignore[arg-type]
        n_embd=cfg.d_model,  # type: ignore[arg-type]
        layer_norm=cfg.layer_norm,
        mlp_activation=cfg.mlp_activation,
    )
    return Esm2(legacy)


register_model("esm2", _build_esm2)
