#!/usr/bin/env python
"""
Upload the nanoprot release suite to the HuggingFace Hub.

One repo per (arch, scale): ``{org}/nanoprot-{arch}-{scale}``. Seed 0 (the
headline, with the model card) goes to the repo root; the other seeds' weights
go to ``seed1/`` and ``seed2/`` subfolders. **Optimizer state (`optim_*.pt`) is
never uploaded** — it is multi-GB resume state of no use to downloaders.

Ships per repo: ``model_*.pt``, ``meta_*.json``, ``config.yaml``,
``provenance.json``, ``README.md`` (the auto-generated card). Run
``scripts.make_model_card --release-root ...`` first so the cards exist.

SAFE BY DEFAULT: prints the upload plan and uploads nothing unless you pass
``--push``. The real upload needs HuggingFace auth (`huggingface-cli login` or
HF_TOKEN) and write access to the org.

Usage:
  python -m scripts.upload_release --release-root $NANOPROT_BASE_DIR/release   # dry-run
  python -m scripts.upload_release --release-root DIR --hf-org yagizdevre --push   # real
  python -m scripts.upload_release --release-root DIR --only esm2-L --push
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAME_RE = re.compile(r"^nanoprot-(?P<arch>gpt2|esm2|mamba)-(?P<scale>XS|S|M|L)-s(?P<seed>\d+)$")
# What ships to the Hub. Everything else (notably optim_*.pt) is excluded.
SHIP_PATTERNS = ("model_*.pt", "meta_*.json", "config.yaml", "provenance.json", "README.md")


def _ship_files(cell_dir: Path) -> List[Path]:
    out: List[Path] = []
    for pat in SHIP_PATTERNS:
        out.extend(sorted(cell_dir.glob(pat)))
    # Defensive: never ship optimizer shards even if a pattern ever matched them.
    return [f for f in out if not f.name.startswith("optim_")]


def _is_done(cell_dir: Path) -> bool:
    import json
    metas = sorted(cell_dir.glob("meta_*.json"))
    if not metas:
        return False
    try:
        m = json.loads(metas[-1].read_text())
        return m.get("step") == m.get("num_iterations")
    except Exception:
        return False


def _human(nbytes: int) -> str:
    x = float(nbytes)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or u == "TB":
            return f"{x:.1f}{u}"
        x /= 1024


def plan(release_root: Path, hf_org: str, only: Optional[str]) -> List[Tuple[str, List[Tuple[Path, str]]]]:
    """Return [(repo_id, [(local_file, path_in_repo), ...]), ...]."""
    bases: Dict[str, Dict[int, Path]] = {}
    for d in sorted(release_root.glob("nanoprot-*-s*")):
        m = NAME_RE.match(d.name)
        if not (d.is_dir() and m and _is_done(d)):
            continue
        base = f"{m['arch']}-{m['scale']}"
        if only and base != only:
            continue
        bases.setdefault(base, {})[int(m["seed"])] = d

    out: List[Tuple[str, List[Tuple[Path, str]]]] = []
    for base, seeds in sorted(bases.items()):
        repo = f"{hf_org}/nanoprot-{base}"
        files: List[Tuple[Path, str]] = []
        head = seeds.get(min(seeds))  # lowest available seed is the headline
        for f in _ship_files(head):
            files.append((f, f.name))                       # root
        for s in sorted(seeds):
            if seeds[s] is head:
                continue
            for f in seeds[s].glob("model_*.pt"):
                files.append((f, f"seed{s}/{f.name}"))
            for f in seeds[s].glob("meta_*.json"):
                files.append((f, f"seed{s}/{f.name}"))
        out.append((repo, files))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--hf-org", type=str, default="yagizdevre")
    ap.add_argument("--only", type=str, default=None, help="Just one base, e.g. esm2-L.")
    ap.add_argument("--private", action="store_true", help="Create private repos.")
    ap.add_argument("--push", action="store_true",
                    help="Actually upload. Without this, prints the plan and exits.")
    args = ap.parse_args()

    repos = plan(args.release_root, args.hf_org, args.only)
    if not repos:
        print(f"No completed (carded) cells under {args.release_root} to upload.")
        return 0

    total = 0
    missing_card = []
    print(f"\n  Upload plan ({'PUSH' if args.push else 'DRY-RUN'}) — {len(repos)} repos\n")
    for repo, files in repos:
        size = sum(f.stat().st_size for f, _ in files)
        total += size
        has_card = any(p == "README.md" for _, p in files)
        if not has_card:
            missing_card.append(repo)
        print(f"  {repo}   ({len(files)} files, {_human(size)}{'' if has_card else '  [!] no README.md card'})")
        for f, p in files:
            print(f"        {p:28s} {_human(f.stat().st_size):>9}")
    excluded = sum(
        f.stat().st_size
        for d in args.release_root.glob("nanoprot-*-s*")
        for f in d.glob("optim_*.pt")
    )
    print(f"\n  TOTAL to upload: {_human(total)}   |   excluded optimizer state: {_human(excluded)}")
    if missing_card:
        print(f"  [!] {len(missing_card)} repo(s) missing README.md — run "
              f"`python -m scripts.make_model_card --release-root {args.release_root}` first.")

    if not args.push:
        print("\n  DRY-RUN — nothing uploaded. Re-run with --push to upload.\n")
        return 0

    # ---- real upload ----
    from huggingface_hub import HfApi
    api = HfApi()
    me = api.whoami()  # fails clearly if not authenticated
    print(f"\n  Authenticated as: {me.get('name')}. Uploading...\n")
    for repo, files in repos:
        api.create_repo(repo, repo_type="model", exist_ok=True, private=args.private)
        for f, p in files:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=p, repo_id=repo, repo_type="model")
        print(f"  pushed {repo}  ({len(files)} files)")
    print("\n  Done.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
