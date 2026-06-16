# nanoprot — project status

_Snapshot: 2026-06-16 · version 0.5.0 · HEAD `66d5fc1` · 156 fast + 28 slow tests green · working tree clean & pushed_

This document explains, end to end: **what nanoprot is, what we set out to do, what we
have accomplished, and what we want to do next.** It is a self-contained handoff — no
prior context required.

---

## 1. The goal

**nanoprot** is a clean, config-driven training framework for **protein language models
(PLMs)** — a *"Pythia for proteins."* It is a **separate public repo** (GitHub
[`ygzdvr/nanoprot`](https://github.com/ygzdvr/nanoprot); models on Hugging Face under
[`yagizdevre`](https://huggingface.co/yagizdevre)) from the senior-thesis mech-interp
work that lives in the parent directory.

There are **two repos and two papers**:

| repo | what | target venue |
|---|---|---|
| **nanoprot** (this dir) | a **Pythia-for-proteins resource / methods paper** — the codebase + the released model suite + the analyses the suite unlocks | **MLSB** (or a structural-biology venue) |
| the thesis (parent dir) | SAE / probing **mechanistic interpretability** of PLMs | Nature Methods (separate track) |

Like Pythia (Biderman et al., ICML 2023 oral), the nanoprot paper's value is **not a new
model** — it is a clean, reproducible **suite of models across scale, seed, and
architecture**, plus the scientific analyses it makes routine. **nanoprot's novelty over
Pythia** is the controlled **cross-architecture** axis (a GPT-2 autoregressive decoder, an
ESM-2 masked-LM encoder, and a Mamba selective-SSM — at matched scale and data budget),
together with multiple seeds and the protein domain (where ESM/ProGen ship only a handful
of large *final* checkpoints).

### The paper pillars

- **Pillar 0 — the suite + framework** (the released artifact).
- **Pillar 1 — cross-architecture scaling laws** (which architecture scales best?).
- **Pillar 2 — developmental mech-interp** (*when during training* do biological concepts
  emerge, across scale and architecture?).
- **Pillar 3 — cross-architecture downstream probing** (which architecture produces the
  most biologically useful representations?).
- **Pillar 4 (optional)** — PLM-specific studies (generation, long context, memorization).

---

## 2. The framework we built (v0.1 → v0.5)

A single YAML file fully specifies a run; the only complexity dial is `--depth` (model
dim = `64·depth`, all other hyperparameters derived). Three architectures live in one
Pydantic discriminated-union `ModelConfig`:

- **`gpt2`** — decoder-only autoregressive transformer (RoPE, QK-norm, ReLU² MLP, GQA,
  ResFormer value embeddings, per-layer residual/skip scaling, smear + backout).
- **`esm2`** — ESM-2-style **encoder, masked-LM** (bidirectional, pre-LN).
- **`mamba`** — **selective state-space model** (causal; fused `mamba-ssm` CUDA kernel,
  ~100× over the portable reference scan, with a pure-PyTorch fallback).

All three share the **33-token ESM-2 residue alphabet** (a deliberate choice so
per-residue probe labels align exactly), a mixed **Muon + AdamW** optimizer with muP-style
LR scaling, and a self-describing checkpoint format (meta carries params, FLOPs, residues,
wall-clock, and the full resolved config).

**Version arc** (all shipped): v0.1 config schema → v0.2 gpt2 + loader + optimizer + loop
+ DDP → v0.3 esm2 + MLM → v0.4 mamba → v0.4.1 audit patches → **v0.5 the model release**.

---

## 3. What we accomplished

### Pillar 0 — the suite is trained and **publicly released**

The full grid — **3 architectures × 4 scales (XS/S/M/L ≈ 8/35/150/650 M) × 3 seeds = 36
models** — was trained from scratch on **UniRef50 (release 2026_01, 60.25 M sequences)**
under a matched compute-optimal budget (residues = 12 × params), 512-residue context.
**All 36 cells trained, zero failures.**

All 12 `(arch, scale)` models (each carrying its 3 seeds) are **public on the Hugging Face
Hub**: `yagizdevre/nanoprot-{gpt2,mamba,esm2}-{XS,S,M,L}`, grouped in a
[collection](https://huggingface.co/collections/yagizdevre/nanoprot-v05-protein-lm-scaling-suite-6a2ad647b6cc80fa1b846cf4).
Optimizer state is excluded; each repo carries the model + meta + config + provenance +
an auto-generated model card (mean±std over seeds). Verified live by a
download → `load_pretrained` → forward-pass round-trip.

### Pillar 1 — cross-architecture scaling laws (DONE, statistically hardened)

**Headline: transformers out-scale Mamba SSMs on proteins.** For the directly comparable
autoregressive models (validation bits-per-residue), gpt2 beats mamba at every scale and
the absolute gap widens (XS +0.026 → L +0.109). Compute-optimal scaling-law fits
`L(N) = E + A·N^(−α)`:

| arch | α (scaling exponent) | 95% CI (bootstrap) | val metric @ L |
|---|---:|---:|---:|
| **gpt2** (AR, bpr) | **0.0309** | **[0.0307, 0.0312]** | 3.536 bpr |
| mamba (AR, bpr) | 0.0228 | [0.0228, 0.0229] | 3.645 bpr |
| esm2 (MLM, mCE) | 0.0278 | [0.0277, 0.0280] | 3.307 mCE |

The **Tier-A rigor pass** (`scripts/scaling_rigor.py`, stratified seed bootstrap, 4000
resamples) makes this bulletproof: the gpt2 and mamba α intervals are **non-overlapping**
(Δα = +0.0081 [+0.0078, +0.0084], P(gpt2 steeper) = 1.000), the per-scale loss gap is
significantly positive at every scale, and the win holds at matched **compute** (iso-FLOP),
not just matched parameters. gpt2 has both the lower loss *and* the steeper exponent →
the attention inductive bias scales better than the selective-SSM one on next-residue
prediction. _(esm2 is a different, MLM objective and is never crowned in an AR-vs-MLM gap.)_

**A methodological note we keep:** a 120-step sub-sampled *calibration* run had ranked
mamba *above* gpt2; the fully-converged runs reverse it. Under-trained comparisons mislead.

### Pillar 3 — cross-architecture probing (DONE, triangulated)

We asked the question the scaling result does *not* answer: does the architecture that
scales best at next-residue prediction also produce the most **biologically decodable**
representations? We built a complete probing harness and ran it on real data.

**Method.** Frozen models; **linear probes** on the residual stream at **every layer**
(extracted non-invasively via forward hooks, compared by *relative depth*); the best layer
is **selected on validation and reported on test** as **`learned − baseline`** (the same
probe on a random-init model of the identical architecture — the Pythia-style control). The
target concept is **3-state secondary structure (SS3)**. Crucially, esm2 is **bidirectional**
(sees both neighbours) so it is reported as a separate **"encoder reference"**, never as a
head-to-head architecture win; the rigorous comparison is **gpt2 vs mamba** (both causal).

**Result (seed 0, NetSurfP/CB513), `learned − baseline` macro-F1:**

| arch | XS | S | M | L |
|---|---:|---:|---:|---:|
| **gpt2** | +0.091 | +0.141 | +0.191 | **+0.232** |
| mamba | +0.090 | +0.102 | +0.136 | +0.186 |
| esm2 (encoder ref.) | +0.039 | +0.189 | +0.242 | +0.282 |

**This cross-validates the scaling headline on a downstream biological task:** gpt2 beats
mamba on SS decodability at S/M/L, and its *absolute* score overtakes mamba at M/L
(gpt2-L 0.671 vs mamba-L 0.658). esm2 wins absolute decodability (bidirectional), all three
scale monotonically, and structure is most decodable in the **late layers**, deepening with
scale.

**§6 triangulation — the finding is source-robust.** We re-ran the probe against **three
independent SS3 label sources** with deliberately different label conventions and protein
sets:

| source | basis | SS3 dist (helix/strand/coil) |
|---|---|---|
| NetSurfP-2.0 / CB513 | DSSP on **experimental** structure | 35 / 22 / 43 |
| DSSP-from-AlphaFold (biotite P-SEA) | SS on **predicted** structure, pLDDT-masked | 39 / 15 / 45 |
| Swiss-Prot | HELIX/STRAND **annotation** (else→coil) | 26 / 15 / 60 |

**Verdict: gpt2 > mamba replicates at 11 of 12 source×scale cells, and at S, M, L on all
three sources** (the lone exception is XS-on-DSSP, a near-tie 0.144 vs 0.153). The result is
robust to **dataset *and* to experimental-vs-predicted structure** — not an artifact of one
benchmark. A per-residue **label-agreement check** (`scripts/probe_agreement.py`) puts
Swiss-Prot↔AF concordance at **68%** (515 k residues), with the disagreement traced to
Swiss-Prot's "else→coil" convention (36% of its "coil" is real AF helix/strand) — an honest,
quantified source difference, not noise. Figure: `docs/figures/probes_triangulation.png`
(2×2, one clean 3-arch scaling panel per source, shared y-axis).

### Supporting infrastructure built along the way

- **Full per-step logging + interval checkpointing harness** (for the data-budget sweep):
  `history.jsonl` records loss/bpr/ppl/accuracy, grad-norm, LR, MFU, throughput, GPU mem
  per step; **intermediate checkpoints are model-only** at a configurable interval (this is
  the *enabler* for Pillar 2). A 3×2 per-run dashboard plots it.
- **Analysis + release tooling:** `aggregate_results`, `scaling_laws`, `scaling_rigor`,
  `make_model_card`, `upload_release`, `plot_scaling_curves`, `plot_training_*`.

---

## 4. Current codebase state (exact)

```
nanoprot/eval/probe/    extract.py  labels.py  linear.py  run.py  __init__.py   # probing harness
scripts/                run_probes.py  prepare_probe_data.py  plot_probes.py
                        probe_agreement.py  scaling_laws.py  scaling_rigor.py
                        aggregate_results.py  make_model_card.py  upload_release.py
                        gen_release_configs.py  gen_sweep_configs.py  prepare_uniref50.py
                        plot_scaling_curves.py  plot_training_curves.py  plot_training_run.py
                        train.py  show_config.py
runs/                   train_release.slurm  train_sweep.slurm  probe_sweep.slurm
                        calibrate_throughput.slurm  prepare_uniref50.slurm
docs/figures/           scaling_laws  scaling_rigor  scaling_curves  training_curves
                        probes  probes_triangulation          (.png + .pdf each)
docs/                   probing_harness_plan.md   v0.5_release_scope.md
results/                results.csv  probe_ss3_{netsurfp,swissprot,dssp}.csv
RESULTS.md  README.md  status.md  pyproject.toml
```

- **Tests:** 156 fast + 28 slow (model-building / end-to-end), all green. Run model-building
  tests with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (see gotchas).
- **Data on disk** (`$NANOPROT_BASE_DIR/.cache/nanoprot`, git-ignored): the 36 release
  checkpoints (`release/`), all three probe caches (`probes/ss3_{netsurfp,swissprot,dssp}`),
  raw downloads (`probes/raw/`: NetSurfP npz, Swiss-Prot XML, AlphaFold PDBs), and the
  per-source probe-result CSVs + per-layer JSON sidecars (`probe_results/`).
- **Roadmap (README):** v0.1–v0.5 shipped/released; v0.6 "next" = the cross-architecture
  benchmark suite (probing) — which is now in fact **largely built and run** (this status
  doc is ahead of that table).

---

## 5. What we want to do next

Prioritized. The first is cheap and finishes Pillar 3; the second is the big, most-novel
unexplored science.

1. **Pillar 3 → publication-complete: 3-seed error bars.** The probes were run on **seed 0
   only**. Re-probe seeds 1–2 across the cells (login GPU, or `sbatch runs/probe_sweep.slurm`
   which loops all 36 cells × all cached sources) so the scaling-transfer curves get proper
   confidence bands. _Small effort; the harness is ready._

2. **Pillar 2 — emergence of capabilities *during training* (the Pythia signature).** This
   is the most compelling unexplored axis. We have shown emergence **with scale** (probing
   decodability grows XS→L) and **with depth** (the layer-wise curves), but **not with
   training time** — because the released models are **end-only** (one checkpoint each).
   *Both halves of the machine already exist:* the **interval-checkpoint logging harness**
   captures training trajectories, and the **probing harness** consumes any checkpoint. The
   experiment: train a representative subset (e.g. gpt2 + mamba at one or two scales) with
   `save_every` on (~15 checkpoints across training), then probe each checkpoint → a
   **SS-decodability-vs-training-step curve** → "does the helix capability emerge earlier in
   gpt2 than mamba, and earlier at larger scale?". **Cost = a re-train** (the keystone we
   deferred); probing is cheap. The data-budget sweep is a two-birds path (it produces
   interval checkpoints *and* the L(N,D) surface).

3. **Data-budget scaling surface (`L(N, D)`).** `runs/train_sweep.slurm` + the 18 sweep
   configs are built (fixed N, varied data budget, full logging + interval checkpoints on)
   but **not launched** — gives the isolated data-exponent β and decouples N from D.

4. **Draft the MLSB paper** — framework + suite + Pillars 1 & 3 (and 2 if the trajectory
   re-train happens). The narrative is strong: *a clean cross-architecture PLM suite, plus
   a source-robust finding that the attention inductive bias both trains and represents
   proteins better than the SSM one, and the harness to keep asking such questions.*

5. **Optional Pillar 4 / extensions:** generation (sample novel proteins), long-context
   behavior (does attention's lead grow with sequence length?), memorization, and the other
   probe concepts (functional sites, family) that the harness already supports.

---

## 6. Key operational facts & gotchas (for whoever continues)

- **Cluster:** della. **Login node = A100 (sm80), has internet *and* a usable GPU** — that
  is where downloads and the probe forward-passes ran. Compute partition = `ailab` (H200,
  sm90), 8 cores/GPU cap, **no internet on compute nodes** (FA3 kernel download DNS-fails →
  SDPA fallback; set `NANOPROT_DISABLE_FA3=1`).
- **Login-node CPU watchdog SIGKILLs (exit 137) multi-threaded torch under pytest.** Always
  run model-building tests and login-node probes with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- **Namespaces:** code repo is **GitHub `ygzdvr/nanoprot`**; models are **HF `yagizdevre`**
  (different — do not conflate).
- **External data quirks learned the hard way:** NetSurfP-2.0 npz = `(N,L,F)`, AA one-hot
  channels 0–19 ordered `ACDEFGHIKLMNPQRSTVWY`, Q8 `GHIBESTC` at 57–64, CB513 eval-mask at
  52. UniProt XML namespace is now **`https`**://uniprot.org/uniprot (was http — detect it
  from the root). AlphaFold download is **`model_v6`** (v4 is gone/404). DSSP-from-AF uses
  **biotite** `annotate_sse` (P-SEA, no external DSSP binary) — installed as the `[probe]`
  extra.
- **`$NANOPROT_BASE_DIR`** defaults to `<repo>/.cache/nanoprot`. The harness cannot submit
  Slurm jobs itself for training; provide the `sbatch` command.

---

## 7. One-paragraph summary

The nanoprot framework is built and at v0.5: three architectures (gpt2 / esm2 / mamba),
one config dial, reproducible. The **36-model suite is trained and public on Hugging Face.**
Two of the four science pillars are effectively done: **cross-architecture scaling laws**
(gpt2 out-scales mamba, statistically bulletproof) and **cross-architecture probing** (gpt2
also produces more biologically decodable representations — a finding that **replicates
across three independent label sources**). The probing harness (8 modules, 3 label sources)
is complete and proven on real data. The two open threads are **3-seed error bars** (cheap,
finishes Pillar 3) and **emergence-during-training / Pillar 2** (the most novel remaining
science — needs a trajectory re-train, for which both the checkpointing and probing
machinery already exist). Then: the data-budget sweep and the MLSB paper.
