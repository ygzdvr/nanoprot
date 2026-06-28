#!/usr/bin/env python3
"""Option A — cross-domain (NLP) capability trajectories from public Pythia checkpoints.

Computes, per (Pythia size, training step): a downstream CAPABILITY (LAMBADA-openai last-word
greedy accuracy) and a held-out LOSS (mean LM cross-entropy on the same passages), by loading the
public checkpoint at revision `step{N}`. Forward passes only — no training. Writes a CSV in the
same (arch, scale, seed, step, ..., capability, loss) shape the forecasting analysis consumes, so
`forecast_capability`-style extrapolation can be run on NLP to show the method generalizes beyond
proteins (Pythia is one architecture family → method-generality, not arch-reversal).
"""
import argparse
import csv
import math
from pathlib import Path

import torch

# Pythia checkpoint revisions that exist (log-spaced early + linear later).
DEFAULT_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000,
                 16000, 32000, 64000, 143000]
TOKENS_PER_STEP = 2_097_152  # Pythia batch: 1024 seq × 2048 tok


@torch.no_grad()
def eval_checkpoint(size, step, examples, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = f"EleutherAI/pythia-{size}"
    rev = f"step{step}"
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    correct = 0; total = 0; loss_sum = 0.0; loss_tok = 0
    for text in examples:
        text = text.strip()
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        full_ids = enc.input_ids.to(device)
        offsets = enc["offset_mapping"][0].tolist()
        if full_ids.shape[1] < 2:
            continue
        # robust last-word target span via char offsets (no separate-tokenization prefix bug)
        last = text.split(" ")[-1]
        start_char = text.rfind(last)
        # a token belongs to the last word if its char span OVERLAPS [start_char, end); byte-level
        # BPE folds the leading space into the first word token, which straddles the boundary.
        tgt_pos = [i for i, (a, b) in enumerate(offsets) if b > start_char and b > a]
        if not tgt_pos or tgt_pos[0] == 0:
            continue
        n_ctx = tgt_pos[0]
        logits = model(full_ids).logits[0]            # (T, V)
        # next-token loss over the whole passage (held-out LM loss proxy)
        tgt = full_ids[0, 1:]
        ce = torch.nn.functional.cross_entropy(logits[:-1].float(), tgt, reduction="sum")
        loss_sum += float(ce); loss_tok += int(tgt.numel())
        # LAMBADA: greedy argmax must match every target-word token
        pred = logits[n_ctx - 1: -1].argmax(-1)        # predictions for positions n_ctx..end
        gold = full_ids[0, n_ctx:]
        if pred.shape[0] == gold.shape[0] and torch.equal(pred, gold):
            correct += 1
        total += 1
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(n_params=n_params, lambada_acc=correct / max(1, total),
               loss=loss_sum / max(1, loss_tok), n_eval=total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["70m", "160m", "410m"])
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    ap.add_argument("--n-examples", type=int, default=800)
    ap.add_argument("--out", type=Path, default=Path("docs/pythia_capability.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    # fp32 throughout: these are small models, and bf16 corrupts the CE/argmax of confident
    # (late-checkpoint) models, which inverted loss and crushed accuracy in validation.
    dtype = torch.float32

    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    examples = [ds[i]["text"] for i in range(min(args.n_examples, len(ds)))]
    print(f"[pythia] {len(examples)} LAMBADA examples; sizes={args.sizes}; {len(args.steps)} steps; {args.device}/{dtype}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    cols = ["arch", "scale", "seed", "step", "n_params", "tokens", "flops", "lambada_acc", "loss", "n_eval"]
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if is_new:
            w.writeheader()
        for size in args.sizes:
            for step in args.steps:
                try:
                    r = eval_checkpoint(size, step, examples, args.device, dtype)
                except Exception as e:
                    print(f"  pythia-{size} step{step}: FAILED {repr(e)[:120]}")
                    continue
                tokens = step * TOKENS_PER_STEP
                row = dict(arch="pythia", scale=size, seed=0, step=step, n_params=r["n_params"],
                           tokens=tokens, flops=6 * r["n_params"] * tokens,
                           lambada_acc=round(r["lambada_acc"], 5), loss=round(r["loss"], 5),
                           n_eval=r["n_eval"])
                w.writerow(row); f.flush()
                print(f"  pythia-{size} step{step:>6}: acc={row['lambada_acc']:.3f} loss={row['loss']:.3f} (n={r['n_eval']})")
    print(f"[pythia] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
