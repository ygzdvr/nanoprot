#!/usr/bin/env python
"""Minimal faithful models + train/probe loop for the synthetic selectivity experiment.

Two minimal instances of the two architecture families the paper compares:
  AttnModel : causal multi-head attention (RoPE) + ReLU^2 MLP, RMSNorm pre-norm.  Averages -> native to
              order-INVARIANT / compositional structure early.
  SSMModel  : a minimal SELECTIVE diagonal state-space block (input-dependent dt + gate, sequential
              scan) + ReLU^2 MLP.  Processes in order with a running state -> native to ORDERED-local
              and PERIODIC structure early.

We train each SUPERVISED on the per-position labelling function and log per-position macro-F1 over
training (iso-FLOP). This measures the RATE at which each architecture learns to COMPUTE the label --
the computational inductive bias -- the controlled complement to the natural-data setup where the
property emerges as an LM byproduct. Matched strength is by construction (identical Bayes ceiling), and
verified empirically (converged F1 ~ equal across tasks).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.synth_selectivity import (
    C_CLASSES, EPS, IGNORE, N_TOK, SEQ_LEN, TASKS, make_dataset, omega_of, pi_of, _rng,
)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope(x, base=10000.0):
    # x: [B, H, L, hd]
    B, H, L, hd = x.shape
    half = hd // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=x.device).float() / half))
    t = torch.arange(L, device=x.device).float()
    ang = torch.outer(t, freqs)                       # [L, half]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class MLP(nn.Module):
    def __init__(self, d, ratio=3):
        super().__init__(); self.fc = nn.Linear(d, ratio * d); self.pr = nn.Linear(ratio * d, d)

    def forward(self, x):
        return self.pr(F.relu(self.fc(x)) ** 2)


class AttnBlock(nn.Module):
    def __init__(self, d, n_head=4, ratio=3):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.h, self.hd = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d); self.mlp = MLP(d, ratio)

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).split(d, -1)
        q = q.view(B, L, self.h, self.hd).transpose(1, 2)
        k = k.view(B, L, self.h, self.hd).transpose(1, 2)
        v = v.view(B, L, self.h, self.hd).transpose(1, 2)
        q, k = rope(q), rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(o.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.n2(x))


class SSMBlock(nn.Module):
    """Minimal SELECTIVE diagonal SSM: input-dependent step size dt and output gate (the 'selective'
    essence of Mamba), diagonal state transition, causal sequential scan. State size = n_state per
    channel-group; we keep it channel-diagonal (one scalar state per model dim) for minimality."""
    def __init__(self, d, ratio=3, n_state=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.ns = n_state
        self.in_proj = nn.Linear(d, 2 * d)                 # x_in, gate z
        self.dt_proj = nn.Linear(d, d)                     # input-dependent log-step
        self.B_proj = nn.Linear(d, n_state)                # input-dependent input matrix
        self.C_proj = nn.Linear(d, n_state)                # input-dependent output matrix
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_state + 1).float()).repeat(d, 1))  # [d,ns] S4D-real
        self.D = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d); self.mlp = MLP(d, ratio)
        # --- Mamba/S4 dt initialization (THE fix): without this dt~0.7 collapses the state memory to
        # ~1 step and the recurrence cannot count/track phase. Init dt_proj.bias so softplus(bias) is
        # log-uniform in [dt_min, dt_max], and keep dt_proj.weight tiny so early dt ~= that bias.
        nn.init.uniform_(self.dt_proj.weight, -1e-3, 1e-3)
        dt = torch.exp(torch.rand(d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))   # inverse softplus so softplus(bias)=dt

    def _scan(self, xin, dt, Bm, Cm):
        # xin,[B,L,d]; dt,[B,L,d]; Bm,Cm,[B,L,ns]
        Bsz, L, d = xin.shape
        A = -torch.exp(self.A_log)                          # [d, ns]
        dA = torch.exp(dt[..., None] * A[None, None])       # [B,L,d,ns]
        dBx = dt[..., None] * Bm[:, :, None, :] * xin[..., None]   # [B,L,d,ns]
        h = torch.zeros(Bsz, d, self.ns, device=xin.device, dtype=xin.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]                    # [B,d,ns]
            ys.append((h * Cm[:, t, None, :]).sum(-1))      # [B,d]
        return torch.stack(ys, 1) + xin * self.D            # [B,L,d]

    def forward(self, x):
        B, L, d = x.shape
        xin, z = self.in_proj(self.n1(x)).split(d, -1)
        dt = F.softplus(self.dt_proj(self.n1(x)))
        Bm, Cm = self.B_proj(self.n1(x)), self.C_proj(self.n1(x))
        y = self._scan(F.silu(xin), dt, Bm, Cm)
        x = x + self.out(y * F.silu(z))
        return x + self.mlp(self.n2(x))


def _init_mamba_block(blk, cfg, std=0.02):
    """Apply the real Mamba init recipe to a bare MambaBlock (the class defaults A_log/D to zeros and
    dt_proj.bias to nn.Linear default -> softplus~0.7 -> 1-step memory; without this the faithful block
    would be crippled exactly like the minimal SSM was)."""
    nn.init.normal_(blk.in_proj.weight, std=std); nn.init.normal_(blk.x_proj.weight, std=std)
    nn.init.normal_(blk.out_proj.weight, std=std); nn.init.normal_(blk.dt_proj.weight, std=std)
    lo, hi = math.log(cfg.dt_min), math.log(cfg.dt_max)
    dt = torch.exp(torch.empty(blk.D_inner).uniform_() * (hi - lo) + lo).clamp(min=cfg.dt_init_floor)
    blk.dt_proj.bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))            # softplus^-1
    nn.init.normal_(blk.conv1d.weight, std=std); nn.init.zeros_(blk.conv1d.bias)
    n = torch.arange(1, cfg.d_state + 1, dtype=torch.float32)                  # S4D-real A = -(1..N)
    blk.A_log.data.copy_(torch.log(n.unsqueeze(0).expand(blk.D_inner, -1)).contiguous())
    nn.init.ones_(blk.D)


class MambaMixerBlock(nn.Module):
    """The FAITHFUL real-Mamba mixer (nanoprot.models.mamba.MambaBlock: depthwise causal conv1d +
    expand=2 + input-dependent dt/B/C, properly initialized) wrapped with the same ReLU^2 MLP as
    AttnBlock, so the attn-vs-mamba comparison isolates the MIXER's inductive bias. This is the
    real-Mamba diagnostic: does a faithful selective SSM (with the conv1d + expand the minimal SSMBlock
    lacked) win early on ordered/periodic tasks, or does attention still lead (-> mechanism is
    pretraining-specific, not a supervised-learning-speed effect)?"""
    def __init__(self, d, ratio=3, n_state=16, d_conv=4, expand=2):
        super().__init__()
        from nanoprot.models.mamba import MambaBlock, MambaConfig
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        cfg = MambaConfig(n_embd=d, d_state=n_state, d_conv=d_conv, expand=expand,
                          dt_rank=max(d // 16, 1), sequence_len=SEQ_LEN, vocab_size=N_TOK)
        self.mixer = MambaBlock(cfg); _init_mamba_block(self.mixer, cfg)
        self.mlp = MLP(d, ratio)

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.mlp(self.n2(x))


class SeqModel(nn.Module):
    def __init__(self, arch, d=96, n_layer=3, n_head=4, n_state=16):
        super().__init__()
        self.emb = nn.Embedding(N_TOK, d)
        blk = {"attn": AttnBlock, "ssm": SSMBlock, "mamba": MambaMixerBlock}[arch]
        kw = {"n_head": n_head} if arch == "attn" else {"n_state": n_state}
        self.blocks = nn.ModuleList([blk(d, **kw) for _ in range(n_layer)])
        self.nf = RMSNorm(d); self.head = nn.Linear(d, C_CLASSES)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.nf(h))


def flops_per_token(arch, d=96, n_layer=3, L=SEQ_LEN, ratio=3, n_state=16):
    mlp = 4 * ratio * d * d
    if arch == "attn":
        core = 4 * d * d + 2 * L * d           # qkvo proj + context (scores+AV) amortized per token
    elif arch == "mamba":
        di = 2 * d                             # expand=2: in/out proj + conv + per-token scan
        core = 6 * d * d + di * 4 + 6 * di * n_state
    else:
        core = (2 * d * d) + (2 * d * n_state) + (d * d) + 4 * d * n_state  # in/out proj + B/C + scan
    return n_layer * (core + mlp) * 2          # *2 for MAC->FLOP; fwd; training ~3x but constant cancels


# ---------------------------------------------------------------------------
# train + evaluate
# ---------------------------------------------------------------------------

def macro_f1(pred, y):
    f1 = []
    for c in range(C_CLASSES):
        m = y != IGNORE
        tp = int(((pred == c) & (y == c) & m).sum()); fp = int(((pred == c) & (y != c) & m).sum())
        fn = int(((pred != c) & (y == c) & m).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1))


def eval_steps(total):
    """Dense early (to resolve the 1%-compute region), sparse late."""
    s = sorted(set([0, 2, 5, 10, 15, 20, 30, 45, 65, 90, 130, 180, 250, 350, 500, 700, 1000,
                    1400, 1900] + [total - 1]))
    return [x for x in s if x < total]


def train_curve(arch, task, seed, *, steps=2500, batch=64, d=96, device="cpu", eps=EPS):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    model = SeqModel(arch, d=d).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.05)
    # fixed val set
    xv, yv = make_dataset(task, 512, SEQ_LEN, seed + 10007, eps=eps)
    xv_t = torch.tensor(xv, device=dev); yv_t = torch.tensor(yv, device=dev)
    fpt = flops_per_token(arch, d=d)
    evs = set(eval_steps(steps)); rows = []
    gen = _rng(seed + 1)

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            pred = model(xv_t).argmax(-1).cpu().numpy()
        model.train()
        f1 = macro_f1(pred, yv)
        flop = fpt * batch * SEQ_LEN * step
        rows.append({"arch": arch, "task": task, "seed": seed, "step": step, "flop": flop,
                     "val_f1": round(f1, 5), "n_param": n_param})
        return f1

    if 0 in evs:
        evaluate(0)
    for step in range(1, steps):
        xb, yb = make_dataset(task, batch, SEQ_LEN, int(gen.integers(1 << 30)), eps=eps)
        xb_t = torch.tensor(xb, device=dev); yb_t = torch.tensor(yb, device=dev)
        logits = model(xb_t)
        loss = F.cross_entropy(logits.reshape(-1, C_CLASSES), yb_t.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step in evs:
            evaluate(step)
    return rows, n_param


def run_cell(spec, args):
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows, n_param = train_curve(args.arch, args.task, args.seed, steps=args.steps,
                                batch=args.batch, d=args.d_model, device=dev)
    om, pi = spec[args.task]["omega"], spec[args.task]["pi"]
    for r in rows:
        r["omega"] = om; r["pi"] = pi
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arch", "task", "seed", "step", "flop", "val_f1",
                                           "n_param", "omega", "pi"])
        if new:
            w.writeheader()
        w.writerows(rows)
    fin = rows[-1]["val_f1"]
    print(f"  {args.arch:4s} {args.task:10s} s{args.seed}: {len(rows)} evals, "
          f"final F1={fin:.3f}, params={n_param/1e3:.0f}k, dev={dev}")
    return 0


def smoke():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  smoke on {dev}")
    for arch in ("attn", "ssm"):
        m = SeqModel(arch).to(dev)
        x, y = make_dataset("motif_R2", 8, SEQ_LEN, 0)
        xt = torch.tensor(x, device=dev); yt = torch.tensor(y, device=dev)
        logit = m(xt)
        loss = F.cross_entropy(logit.reshape(-1, C_CLASSES), yt.reshape(-1), ignore_index=IGNORE)
        loss.backward()
        gnorm = math.sqrt(sum(p.grad.pow(2).sum().item() for p in m.parameters() if p.grad is not None))
        print(f"    {arch}: out {tuple(logit.shape)}, params {sum(p.numel() for p in m.parameters())/1e3:.0f}k, "
              f"loss {loss.item():.3f}, gradnorm {gnorm:.2f}, flops/tok {flops_per_token(arch):.2e}")
    return 0
