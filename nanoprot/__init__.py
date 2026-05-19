"""nanoprot — a minimal, config-driven training framework for protein language models."""

__version__ = "0.4.0"

from nanoprot.config import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    Esm2ModelConfig,
    Gpt2ModelConfig,
    LoggingConfig,
    MambaModelConfig,
    ModelConfig,
    NanoprotConfig,
    OptimizerConfig,
    TokenizerConfig,
    TrainingConfig,
    dump_config,
    load_config,
)
from nanoprot.models import GPT, GPTConfig, build_model, list_archs, register_model

__all__ = [
    "__version__",
    # config (re-exported)
    "NanoprotConfig",
    "ModelConfig",
    "Gpt2ModelConfig",
    "Esm2ModelConfig",
    "MambaModelConfig",
    "TokenizerConfig",
    "DataConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "EvalConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "load_config",
    "dump_config",
    # model registry
    "GPT",
    "GPTConfig",
    "build_model",
    "list_archs",
    "register_model",
]
