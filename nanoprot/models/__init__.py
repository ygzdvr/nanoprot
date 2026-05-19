"""Model architectures.

v0.2 supports GPT-2-style decoder-only transformers. v0.3 adds ESM-2-style
masked encoders; v0.4 adds Mamba/SSM. New architectures register themselves
via :func:`register_model` so they show up in :func:`build_model`.
"""

from __future__ import annotations

from typing import Callable, Dict

from nanoprot.config import ModelConfig
from nanoprot.models.gpt2 import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig", "build_model", "register_model", "list_archs"]


# Architecture registry: name -> factory(model_cfg) -> nn.Module
_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str, factory: Callable) -> None:
    """Register a model factory under ``name``.

    The factory must accept a :class:`nanoprot.config.ModelConfig` and return
    an instantiated ``nn.Module``.
    """
    if name in _REGISTRY:
        raise ValueError(f"model arch {name!r} already registered")
    _REGISTRY[name] = factory


def list_archs() -> list[str]:
    """Return the list of registered model architectures."""
    return sorted(_REGISTRY)


def build_model(cfg: ModelConfig):
    """Build a model from a :class:`ModelConfig`.

    The returned ``nn.Module`` is on meta device (per the legacy GPT init);
    call ``model.to_empty(device=...)`` and ``model.init_weights()`` to
    materialize parameters.
    """
    if cfg.arch not in _REGISTRY:
        raise ValueError(
            f"unknown model arch {cfg.arch!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[cfg.arch](cfg)


# -- gpt2 factory ----------------------------------------------------------

def _build_gpt2(cfg: ModelConfig) -> GPT:
    """Build a GPT model from a nanoprot :class:`ModelConfig`."""
    legacy = GPTConfig(
        sequence_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.depth,
        n_head=cfg.n_heads,  # type: ignore[arg-type]  # derived in config validator
        n_kv_head=cfg.n_kv_heads,  # type: ignore[arg-type]
        n_embd=cfg.d_model,  # type: ignore[arg-type]
        window_pattern=cfg.window_pattern,
    )
    return GPT(legacy)


register_model("gpt2", _build_gpt2)
