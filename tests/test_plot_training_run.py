"""Training-run viz: parse history.jsonl and reconstruct from a release log."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_training_run import _from_log, _load_history

_LOG = """\
Loaded config: nanoprot-esm2-L-s0  (/x/nanoprot-esm2-L-s0.yaml)
step      0/14857 | loss 3.6600 | smooth 3.6600 | lr_mult 0.025 | tok/s 2.6e+04
step     10/14857 | loss 2.9000 | smooth 3.1000 | lr_mult 0.275 | tok/s 1.5e+05
  [eval] step 250: val_loss=2.6031 val_bpr=3.7554 (best=3.7554)
  [eval] step 500: val_loss=2.5579 val_bpr=3.6902 ppl=12.91 acc=0.31 (best=3.6902)
"""


def test_reconstruct_from_log(tmp_path: Path) -> None:
    train, ev, cell = _from_log_text(tmp_path, _LOG, total_batch=524288)
    assert cell == "nanoprot-esm2-L-s0"
    assert len(train) == 2
    assert train[1]["step"] == 10 and abs(train[1]["loss"] - 2.9) < 1e-9
    assert train[1]["tokens"] == 10 * 524288        # tokens derived from step
    assert len(ev) == 2
    assert ev[0]["val_ppl"] is None and ev[0]["val_accuracy"] is None   # old-format eval
    assert ev[1]["val_ppl"] == 12.91 and ev[1]["val_accuracy"] == 0.31  # full-logging eval


def _from_log_text(tmp_path: Path, text: str, total_batch: int):
    p = tmp_path / "run.out"
    p.write_text(text)
    return _from_log(p, total_batch)


def test_load_history_splits_train_and_eval(tmp_path: Path) -> None:
    p = tmp_path / "history.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"type": "train", "step": 0, "loss": 3.6, "tokens": 100},
        {"type": "train", "step": 1, "loss": 3.5, "tokens": 200},
        {"type": "eval", "step": 1, "val_bpr": 5.0, "val_ppl": 30.0, "val_accuracy": 0.1},
    ]))
    train, ev = _load_history(p)
    assert len(train) == 2 and len(ev) == 1
    assert ev[0]["val_accuracy"] == 0.1
