"""Shared pytest helpers."""

from __future__ import annotations

from typing import Any

from nanoprot.config import NanoprotConfig


# Fields that are populated by ``_derive_width_and_heads`` /
# ``MambaModelConfig.derive``. If we round-trip a config through
# ``model_dump`` -> ``model_validate`` to re-trigger derivation, we need to
# strip these first; otherwise the second pass sees them as already set
# and skips derivation, silently producing an incorrect config.
_DERIVED_MODEL_FIELDS = ("d_model", "n_heads", "n_kv_heads", "dt_rank")


def re_derive_model_with(cfg: NanoprotConfig, **model_overrides: Any) -> NanoprotConfig:
    """Mutate ``cfg.model`` fields, drop derived fields, then revalidate.

    Use this instead of writing::

        cfg.model.depth = 6
        cfg = NanoprotConfig.model_validate(cfg.model_dump())

    which is a silent no-op for width because ``model_dump`` keeps the
    already-derived ``d_model``, so the second ``model_validate`` skips
    derivation and the new depth never propagates to width.
    """
    d = cfg.model_dump()
    d["model"] = {**d["model"], **model_overrides}
    for k in _DERIVED_MODEL_FIELDS:
        d["model"].pop(k, None)
    return NanoprotConfig.model_validate(d)
