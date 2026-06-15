"""Cross-architecture probing harness (Pillars 2 & 3).

See ``docs/probing_harness_plan.md`` for the design. The offline core:

- :mod:`nanoprot.eval.probe.extract` — non-invasive forward-hook extraction of the
  residual stream at ``n_layer + 1`` points, uniform across gpt2 / esm2 / mamba.
- :mod:`nanoprot.eval.probe.linear` — standardized linear probes with a
  validation-selected best layer (reported on test).
"""

from nanoprot.eval.probe.extract import (  # noqa: F401
    extract_layers,
    get_blocks,
    n_probe_points,
    relative_depths,
    token_identity_features,
)
from nanoprot.eval.probe.linear import (  # noqa: F401
    LinearProbe,
    accuracy,
    evaluate_probe,
    fit_linear_probe,
    flatten_residues,
    macro_f1,
    probe_layer_sweep,
)
from nanoprot.eval.probe.labels import (  # noqa: F401
    ProbeDataset,
    ProbeProtein,
    assign_splits,
    default_tokenizer,
    encode_protein,
    load_probe_dataset,
    tokenize_and_align,
)
