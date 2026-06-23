"""Tests for the probe report (scripts/probe_report.py).

The load-bearing invariant is that a "task" is (concept, source): ss3/netsurfp,
ss8/netsurfp and rsa/netsurfp share source="netsurfp" but must never be merged
(different class sets, and rsa is r2 not macro_f1). Also covers the within-seed paired
gpt2-mamba delta.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_report import (  # noqa: E402
    load_per_class, paired_gpt2_mamba, summary, _tasks,
)


def _row(concept, source, arch, scale, seed, learned, baseline):
    return {"concept": concept, "source": source, "arch": arch, "scale": scale,
            "seed": str(seed), "metric": "r2" if concept == "rsa" else "macro_f1",
            "learned_test": str(learned), "baseline_test": str(baseline),
            "learned_minus_baseline": str(round(learned - baseline, 6))}


def test_concept_and_source_not_conflated():
    # same source, different concept -> must stay two distinct tasks
    rows = [_row("ss3", "netsurfp", "gpt2", "M", 0, 0.63, 0.44),
            _row("ss8", "netsurfp", "gpt2", "M", 0, 0.27, 0.17),
            _row("rsa", "netsurfp", "gpt2", "M", 0, 0.29, 0.19)]
    assert set(_tasks(rows)) == {("ss3", "netsurfp"), ("ss8", "netsurfp"), ("rsa", "netsurfp")}
    g, metric = summary(rows)
    assert metric[("ss3", "netsurfp")] == "macro_f1"
    assert metric[("rsa", "netsurfp")] == "r2"
    assert (("ss3", "netsurfp"), "gpt2", "M") in g
    assert (("ss8", "netsurfp"), "gpt2", "M") in g


def test_paired_delta_within_seed_and_sign_flag():
    rows = []
    for s, (g, m) in enumerate([(0.15, 0.10), (0.16, 0.11), (0.14, 0.10)]):   # gpt2 wins all 3
        rows += [_row("ss3", "netsurfp", "gpt2", "S", s, 0.5 + g, 0.5),
                 _row("ss3", "netsurfp", "mamba", "S", s, 0.5 + m, 0.5)]
    for s, (g, m) in enumerate([(0.09, 0.10), (0.11, 0.10), (0.10, 0.09)]):   # gpt2 loses seed 0
        rows += [_row("ss3", "netsurfp", "gpt2", "XS", s, 0.5 + g, 0.5),
                 _row("ss3", "netsurfp", "mamba", "XS", s, 0.5 + m, 0.5)]
    p = paired_gpt2_mamba(rows)
    muS, sdS, nS, winS = p[(("ss3", "netsurfp"), "S")]
    assert nS == 3 and winS is True and muS > 0
    _, _, _, winXS = p[(("ss3", "netsurfp"), "XS")]
    assert winXS is False                       # gpt2 lost one seed at XS -> not unanimous


def test_load_per_class_keyed_by_concept(tmp_path):
    for concept, classes, pcf in [("ss3", ["helix", "strand", "coil"], [0.7, 0.5, 0.6]),
                                  ("ss8", list("GHIBESTC"), [0.1] * 8)]:
        side = {"row": {"concept": concept, "source": "netsurfp", "arch": "gpt2",
                        "scale": "M", "seed": "0"},
                "class_names": classes,
                "best": {"layer": 5, "learned_test": {"per_class_f1": pcf}}}
        (tmp_path / f"gpt2-M-s0_netsurfp_{concept}.json").write_text(json.dumps(side))
    pc, names = load_per_class(tmp_path)
    assert names[("ss3", "netsurfp")] == ["helix", "strand", "coil"]    # not conflated w/ ss8
    assert names[("ss8", "netsurfp")] == list("GHIBESTC")
    assert pc[("ss3", "netsurfp", "gpt2", "M", "0")] == [0.7, 0.5, 0.6]
