#!/usr/bin/env python
"""
Generate a HuggingFace model card (README.md) for a nanoprot release checkpoint.

The card is built from the checkpoint's own self-describing metadata
(``meta_{step}.json`` — see nanoprot.training.loop._save), so it never drifts
from the artifact. When sibling seed directories exist
(``nanoprot-{arch}-{scale}-s{0,1,2}``), the headline metric is reported as
mean +/- std across seeds, with seed 0 as the default download.

It also writes a self-contained ``config.yaml`` next to the checkpoint (from the
embedded config) so the "how to load" snippet is runnable as-is.

Usage:
  # one model (auto-aggregates sibling seeds in the parent dir):
  python -m scripts.make_model_card --checkpoint-dir $NANOPROT_BASE_DIR/release/nanoprot-gpt2-M-s0

  # every model under a release root:
  python -m scripts.make_model_card --release-root $NANOPROT_BASE_DIR/release

  # custom output path / disable seed aggregation:
  python -m scripts.make_model_card --checkpoint-dir DIR --out CARD.md --no-siblings
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAME_RE = re.compile(r"^(?P<base>nanoprot-(?P<arch>gpt2|esm2|mamba)-(?P<scale>XS|S|M|L))-s(?P<seed>\d+)$")

_OBJECTIVE_TASK = {
    "ar": ("causal-language-modeling", "text-generation"),
    "mlm": ("masked-language-modeling", "fill-mask"),
}


# ---------------------------------------------------------------------------
# Reading checkpoints
# ---------------------------------------------------------------------------

def _final_meta(ckpt_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the meta dict for the highest-step checkpoint in ``ckpt_dir``."""
    metas = sorted(ckpt_dir.glob("meta_*.json"))
    if not metas:
        return None
    best = None
    for m in metas:
        try:
            d = json.loads(m.read_text())
        except Exception:
            continue
        if best is None or int(d.get("step", -1)) > int(best.get("step", -1)):
            best = d
            best["_path"] = str(m)
    return best


def _param_count(meta: Dict[str, Any]) -> Optional[int]:
    """Real param count: prefer meta; fall back to a registry build if absent."""
    n = meta.get("n_params")
    if isinstance(n, int) and n > 0:
        return n
    # Fallback for checkpoints written before the self-describing meta (v0.5).
    cfg_dict = meta.get("config")
    if not cfg_dict:
        return None
    try:
        import torch
        from nanoprot.config import NanoprotConfig
        from nanoprot.models import build_model
        cfg = NanoprotConfig.model_validate(cfg_dict)
        with torch.device("meta"):
            model = build_model(cfg.model)
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _find_sibling_seeds(ckpt_dir: Path) -> List[Path]:
    """All ``...-s{N}`` dirs sharing this checkpoint's arch+scale (incl. itself)."""
    m = NAME_RE.match(ckpt_dir.name)
    if not m:
        return [ckpt_dir]
    pattern = str(ckpt_dir.parent / f"{m['base']}-s*")
    sibs = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    return sibs or [ckpt_dir]


def _load_provenance(meta: Dict[str, Any], ckpt_dir: Path) -> Optional[Dict[str, Any]]:
    """Find the corpus provenance.json: prefer one already beside the checkpoint,
    else read it from the training run's shard_dir (recorded in the embedded
    config). Returns None if absent."""
    candidates = [ckpt_dir / "provenance.json"]
    shard_dir = ((meta.get("config") or {}).get("data") or {}).get("shard_dir")
    if shard_dir:
        candidates.append(Path(shard_dir) / "provenance.json")
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_params(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    return f"{n/1e9:.2f}B ({n:,})" if n >= 1e9 else f"{n/1e6:.1f}M ({n:,})"


def _fmt_int(n: Optional[float]) -> str:
    return f"{int(n):,}" if n is not None else "unknown"


def _fmt_residues(n: Optional[float]) -> str:
    if n is None:
        return "unknown"
    n = float(n)
    return f"{n/1e9:.2f}B ({int(n):,})" if n >= 1e9 else f"{n/1e6:.1f}M ({int(n):,})"


def _fmt_hours(sec: Optional[float]) -> str:
    if sec is None:
        return "unknown"
    h = sec / 3600.0
    return f"{h:.2f} h ({sec/60:.0f} min)"


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


# ---------------------------------------------------------------------------
# Card construction
# ---------------------------------------------------------------------------

def build_card(meta: Dict[str, Any], sibling_metas: List[Dict[str, Any]], *,
               provenance: Optional[Dict[str, Any]] = None,
               hf_org: str = "yagizdevre") -> str:
    cfg = meta.get("config", {})
    model = cfg.get("model", meta.get("model_config", {}))
    train = cfg.get("training", meta.get("training_config", {}))
    arch = model.get("arch", "?")
    name = meta.get("name", "nanoprot-model")
    nm = NAME_RE.match(name)
    base = nm["base"] if nm else name
    scale = nm["scale"] if nm else "?"
    seed = nm["seed"] if nm else str(train.get("seed", "?"))

    objective = train.get("objective", "ar")
    task_type, task_pipeline = _OBJECTIVE_TASK.get(objective, ("language-modeling", "text-generation"))
    n_params = _param_count(meta)

    # Headline metric: mean +/- std across seeds.
    bprs = [m.get("last_val_bpr") for m in sibling_metas]
    mu, sd = _mean_std([b for b in bprs if b is not None])
    n_seeds = len([b for b in bprs if b is not None])
    if objective == "ar":
        metric_name = "bits-per-residue"
        metric_label = "Validation bits-per-residue (lower is better)"
        compare_note = (
            "Directly comparable to other **autoregressive** nanoprot models "
            "(`gpt2`, `mamba`) — same 33-token vocabulary, same AR objective, "
            "same data budget. **Not** comparable to the `esm2` (masked-LM) "
            "models, whose metric is a different quantity."
        )
    else:
        metric_name = "masked-cross-entropy-bits"
        metric_label = "Validation masked cross-entropy (bits/residue over masked positions)"
        compare_note = (
            "This is a masked-LM pseudo-perplexity over masked positions — a "
            "**different quantity** from the autoregressive models' bits-per-"
            "residue. Compare `esm2` models to each other and on downstream / "
            "probing tasks, not via this number against `gpt2`/`mamba`."
        )

    metric_value = f"{mu:.4f}" if mu is not None else "n/a"
    metric_pm = f" ± {sd:.4f} (n={n_seeds} seeds)" if (sd is not None and n_seeds > 1) else ""

    # Training-data provenance (from the corpus's provenance.json, if present).
    if provenance:
        rv = provenance.get("release_version")
        rel = f"UniRef50 release {rv}" if rv else "UniRef50 (release unspecified)"
        rd = provenance.get("release_date")
        if rd:
            rel += f" ({rd})"
        nseq = provenance.get("num_sequences")
        data_desc = rel + (f", {nseq:,} sequences" if nseq else "") + \
            "; held-out final shard for validation"
        prepared = provenance.get("prepared_utc")
    else:
        data_desc = "UniRef50; held-out final shard for validation"
        prepared = None
    prepared_note = f" (corpus prepared {prepared})" if prepared else ""

    # ---- YAML front-matter ----
    front = {
        "license": "mit",
        "library_name": "nanoprot",
        "tags": ["protein-language-model", "proteins", "uniref50", "nanoprot", arch],
        "datasets": ["uniref50"],
        "pipeline_tag": task_pipeline,
        "metrics": [metric_name],
    }
    if mu is not None:
        front["model-index"] = [{
            "name": base,
            "results": [{
                "task": {"type": task_type},
                "dataset": {"name": "UniRef50 (held-out shard)", "type": "uniref50"},
                "metrics": [{"type": metric_name, "value": round(mu, 4)}],
            }],
        }]
    fm = "---\n" + yaml.safe_dump(front, sort_keys=False).strip() + "\n---\n"

    # ---- Body ----
    depth = model.get("depth")
    d_model = model.get("d_model")
    n_heads = model.get("n_heads")
    arch_extra = ""
    if arch == "mamba":
        arch_extra = (
            f"| d_state | {model.get('d_state')} |\n"
            f"| d_conv | {model.get('d_conv')} |\n"
            f"| expand | {model.get('expand')} |\n"
            f"| dt_rank | {model.get('dt_rank')} |\n"
        )
    elif arch == "gpt2":
        arch_extra = (
            f"| MLP activation | {model.get('mlp_activation')} |\n"
            f"| logit softcap | {model.get('logit_softcap')} |\n"
            f"| window pattern | {model.get('window_pattern')} |\n"
        )
    elif arch == "esm2":
        arch_extra = (
            f"| norm | {model.get('layer_norm')} |\n"
            f"| MLP activation | {model.get('mlp_activation')} |\n"
        )

    n_iter = meta.get("num_iterations")
    total_residues = meta.get("total_residues")
    total_flops = meta.get("total_flops")
    train_time = meta.get("train_time_sec")
    world_size = meta.get("world_size")
    version = meta.get("version", "unknown")
    final_step = meta.get("step")

    sibling_links = "\n".join(
        f"- `{base}-s{m_['_seed']}` — final {metric_name} "
        f"{m_.get('last_val_bpr'):.4f}" if m_.get("last_val_bpr") is not None
        else f"- `{base}-s{m_['_seed']}`"
        for m_ in sibling_metas
    )

    flops_str = f"{total_flops:.3e}" if total_flops is not None else "unknown"

    body = f"""
# {base}

**A protein language model (`{arch}` architecture) — {_fmt_params(n_params)} parameters, \
trained on {_fmt_residues(total_residues)} UniRef50 residues.**

`{base}` is part of the **nanoprot** suite: a Pythia-style matrix of protein
language models spanning three architectures (`gpt2`, `esm2`, `mamba`) and four
scales (XS/S/M/L), each trained from scratch on UniRef50 under a matched,
Chinchilla-style data budget. The suite is built for *controlled* comparison —
same data, same tokenizer, one variable at a time.

## Headline result

**{metric_label}: {metric_value}{metric_pm}**

> {compare_note}

## Model details

| | |
|---|---|
| Architecture | `{arch}` |
| Objective | {objective.upper()} ({task_type}) |
| Scale rung | {scale} |
| Parameters | {_fmt_params(n_params)} |
| Layers (depth) | {depth} |
| Hidden size (d_model) | {d_model} |
| Attention heads | {n_heads} |
| Max sequence length | {model.get('max_seq_len')} |
| Vocabulary | 33-token residue alphabet (ESM-2) |
{arch_extra}| Precision | {train.get('precision')} |

## Training

| | |
|---|---|
| Data | {data_desc} |
| Tokenizer | `esm2` — 33-token residue alphabet (shared across the whole suite) |
| Optimizer | Muon (matrices) + AdamW (embeddings/scalars), weight_decay={cfg.get('optimizer', {}).get('weight_decay')} |
| Batch size | {_fmt_int(train.get('total_batch_size'))} residues/step |
| Optimizer steps | {_fmt_int(n_iter)} |
| Residues seen | {_fmt_residues(total_residues)} |
| Param/data ratio | {train.get('param_data_ratio')} (Chinchilla-style) |
| Total FLOPs | {flops_str} |
| Wall-clock | {_fmt_hours(train_time)} on {world_size} GPU(s) |
| Seed | {seed} (siblings: see below) |
| nanoprot version | {version} |

## Evaluation

Evaluated on a held-out UniRef50 shard. **{metric_label}: {metric_value}{metric_pm}.**

{compare_note}

## Intended use & limitations

Research use: learning protein representations, extracting residual-stream
features, mechanistic-interpretability probing, and architecture comparison.
Trained only on UniRef50 sequences — **not** for clinical or diagnostic use, and
not aligned to any downstream task out of the box.

## How to load

Install nanoprot (`pip install nanoprot`), download this repo, and point the
arch-aware loader at the folder — it works for **any** nanoprot architecture
(gpt2 / esm2 / mamba), reading the embedded config and selecting the right
tokenizer automatically.

```python
from nanoprot.training.checkpoint import load_pretrained

model, cfg, meta, tokenizer = load_pretrained(
    "path/to/this/repo", device="cpu", return_tokenizer=True,
)
model.eval()
# meta carries the trained-artifact facts (params, FLOPs, val metric, ...)
```

## The nanoprot suite

Hub: `{hf_org}/{base}` (seed 0 is the default; siblings on branches `seed1`,
`seed2`). This model's sibling seeds:

{sibling_links}

The full suite spans `{{gpt2, esm2, mamba}} x {{XS, S, M, L}} x {{seed 0,1,2}}`.
See the [nanoprot repository](https://github.com/ygzdvr/nanoprot) for the
complete grid and the scaling-curve comparisons.

## Citation

```bibtex
@software{{nanoprot,
  author  = {{Devre, H. Yagiz}},
  title   = {{nanoprot: a minimal training framework for protein language models}},
  year    = {{2026}},
  url     = {{https://github.com/ygzdvr/nanoprot}}
}}
```

## Reproducibility

Trained with nanoprot v{version}{prepared_note}. The exact, complete training
config is in `config.yaml` (also embedded in `meta_{final_step:06d}.json`).
Re-train with:

```bash
python -m scripts.train --config config.yaml
```
"""
    return fm + body


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_for_dir(ckpt_dir: Path, *, out: Optional[Path], siblings: bool,
                 write_config: bool, hf_org: str = "yagizdevre") -> Optional[Path]:
    meta = _final_meta(ckpt_dir)
    if meta is None:
        print(f"  [skip] no meta_*.json in {ckpt_dir}", file=sys.stderr)
        return None

    if siblings:
        sib_dirs = _find_sibling_seeds(ckpt_dir)
    else:
        sib_dirs = [ckpt_dir]
    sib_metas: List[Dict[str, Any]] = []
    for d in sib_dirs:
        sm = _final_meta(d)
        if sm is not None:
            m = NAME_RE.match(d.name)
            sm["_seed"] = m["seed"] if m else "?"
            sib_metas.append(sm)
    if not sib_metas:
        sib_metas = [dict(meta, _seed=NAME_RE.match(ckpt_dir.name)["seed"]
                          if NAME_RE.match(ckpt_dir.name) else "?")]

    # Write a self-contained config.yaml from the embedded config.
    if write_config and meta.get("config"):
        (ckpt_dir / "config.yaml").write_text(
            yaml.safe_dump(meta["config"], sort_keys=False)
        )

    # Corpus provenance: read from the data dir, and ship a copy next to the
    # weights so the released artifact records exactly which UniRef50 it saw.
    provenance = _load_provenance(meta, ckpt_dir)
    if provenance and not (ckpt_dir / "provenance.json").exists():
        (ckpt_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    card = build_card(meta, sib_metas, provenance=provenance, hf_org=hf_org)
    out_path = out or (ckpt_dir / "README.md")
    out_path.write_text(card)
    rel = provenance.get("release_version") if provenance else None
    print(f"  wrote {out_path}  ({len(sib_metas)} seed(s); "
          f"UniRef50 {rel or 'release unknown'})")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint-dir", type=Path, help="One model directory.")
    g.add_argument("--release-root", type=Path, help="Generate cards for every model under here.")
    ap.add_argument("--out", type=Path, default=None, help="Output path (single-dir mode).")
    ap.add_argument("--no-siblings", action="store_true", help="Do not aggregate seeds.")
    ap.add_argument("--no-write-config", action="store_true",
                    help="Do not write config.yaml next to the checkpoint.")
    ap.add_argument("--hf-org", type=str, default="yagizdevre",
                    help="HuggingFace namespace for the model repos (default: yagizdevre).")
    args = ap.parse_args()

    siblings = not args.no_siblings
    write_config = not args.no_write_config

    if args.checkpoint_dir:
        make_for_dir(args.checkpoint_dir, out=args.out, siblings=siblings,
                     write_config=write_config, hf_org=args.hf_org)
        return 0

    # release-root mode: one card per arch-scale (seed 0 is the headline dir);
    # if seed-0 is missing, fall back to the lowest available seed.
    root = args.release_root
    by_base: Dict[str, List[Path]] = {}
    for d in sorted(root.glob("nanoprot-*-s*")):
        if not d.is_dir():
            continue
        m = NAME_RE.match(d.name)
        if m:
            by_base.setdefault(m["base"], []).append(d)
    if not by_base:
        sys.exit(f"no nanoprot-*-s* model dirs found under {root}")
    print(f"Generating cards for {len(by_base)} models under {root}/")
    for base, dirs in sorted(by_base.items()):
        dirs.sort(key=lambda p: int(NAME_RE.match(p.name)["seed"]))
        head = dirs[0]  # lowest seed = headline
        make_for_dir(head, out=None, siblings=siblings, write_config=write_config,
                     hf_org=args.hf_org)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
