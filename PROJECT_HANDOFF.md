# Project Handoff Document — nanoprot Two-Paper Program

> **Status date:** 2026-07-06 · **Repo:** `nanoprot` (nested git repo inside the outer `nanochat` repo) · **Branch:** `main` · **HEAD:** `4b919fc`
> **Scope:** the *architecture rank-reversal* (Paper B, ICLR) and the *developmental atlas / matched-loss inductive bias* (Paper A, Nature-family), computed from a training-time sweep of protein and genome language models. Self-contained: a reader is assumed to know nothing from the originating conversation.
> **Revision:** v2 (audited against the full conversation + re-verified against on-disk data). Corrections in v2 are flagged inline with **[v2 FIX]**.

Epistemic status legend:
- **[PROVEN]** — mathematically shown or verified byte-identical in code this session.
- **[VERIFIED]** — checked against on-disk data/meta this session.
- **[EMPIRICAL]** — measured from data with the stated uncertainty.
- **[HYPOTHESIS]** — proposed explanation, not established.
- **[INFERRED]** — reconstructed from indirect evidence; not directly stated.
- **[UNKNOWN]** — not available; must be checked.

---

## 1. Executive Summary

**Plain language.** We train small autoregressive language models on protein sequences (UniRef50) and on DNA (human genome hg38), in two architecture families — a **Transformer** (`gpt2`, attention) and a **state-space model** (`mamba`, linear recurrence) — at several sizes and seeds, saving many checkpoints across training. At each checkpoint we ask: *how much biological structure (protein secondary structure, DNA reading frame, …) can a simple linear probe read out of the model's internal activations?* The central finding (Paper B) is a **rank-reversal**: early in training the state-space model exposes more of this structure, but by convergence the Transformer wins — even though the *language-model loss* never reverses. So a naive "train a little, pick the better architecture" heuristic **mis-ranks** architectures; we build a **forecaster** that fixes the ranking cheaply (≈90% compute saved at the 10%-budget operating point). Paper A is a broader "developmental atlas" whose headline is a **matched-loss inductive-bias coefficient γ**: at *equal loss*, the Transformer still exposes more decodable structure.

**Technical language.** Let `score(a,s,r,C)` be linear-probe decodability of a concept from the residual stream of arch `a∈{gpt2,mamba}`, scale `s`, seed `r`, at compute `C`. Define the seed-paired margin `Δ_r(C)=score(gpt2)-score(mamba)`. A **reversal** at scale `s` is `d_early<0 ∧ d_final>0` (sign flip between an early iso-compute budget `C_e=0.01·C_f` and convergence `C_f`). We quantify `P(reversal)` with a **hierarchical partial-pooling posterior** (Gibbs) that replaced an over-confident 3-seed bootstrap. Two accounting corrections were required to make `C` a valid cross-architecture axis: **(a) [v2 FIX]** a FLOPs-per-token bug that inflated mamba's per-token cost by a **scale-dependent 2.40×–6.61×** (3.24× at scale M), and **(b)** a genome-context axis bug. On the corrected axis with 10 seeds, the reversal is robust for protein secondary structure (`ss3/M P=0.92`, `ss8/M P=0.75`) and replicates cross-domain on DNA reading-frame (`frame/M P=0.93`) but not on surface composition (`gc P=0.00`, a clean negative control). The forecaster and γ are **[PROVEN] FLOP-invariant** (byte-identical under the correction), so they survive untouched.

**Current stage.** All GPU compute is finished (Slurm queue empty). The corrected, seed-powered empirical core is committed. Remaining work is CPU analysis + **rewriting `paper_B.tex`** (still on pre-correction numbers) and **regenerating the stale protein/genome crossover + rank-reversal figures**.

---

## 2. Project Goal

**Original objective (the standing `/goal`).** Two publishable papers from nanoprot: an **ICLR spotlight** (Paper B) and a **cited Nature-family acceptance** (Paper A). Operator framing: "full professor at MIT; reason at the highest effort; be technical and mathematical; do not skip a step; do not make mistakes."

**Paper B (ICLR spotlight) — architecture rank-reversal.** An *early* biological-capability proxy mis-ranks `gpt2` vs `mamba` while *loss does not*; a calibrated forecaster recovers the converged ranking cheaply; the effect replicates cross-domain (protein → genome).

**Paper A (Nature-family) — developmental atlas + matched-loss inductive bias.** Capabilities emerge in a reproducible order across arch/scale/objective (function last), and at *matched loss* the Transformer exposes more decodable structure (coefficient **γ>0**).

**Objective changes (documented so they are not re-litigated):**
1. **[Pivot, pre-session]** The thesis originally targeted mechanistic interpretability via Top-K sparse autoencoders (see the SAE pipeline in `CLAUDE.md`). It pivoted (2026-04-28) away from a Cagnetta-style γ_lang/β/α_D language-statistics scaling framework; **do not reintroduce Cagnetta / γ_lang / β / α_D**. The "γ" here is the *inductive-bias coefficient*, a different object (§3.7).
2. **[This session]** The Paper-B "spotlight lever" was reconsidered: a synthetic "selectivity-law" centerpiece was attempted and **abandoned** (Exp. S); the evidence bar was raised (bootstrap → hierarchical posterior); and two FLOP-accounting bugs forced a recomputation that **overturned** two earlier claims (fold/pfam reversal; a "20–25× scale-law").

**Operating constraints (in force).** No subagents / no autonomous loops — main-loop only, small reviewed steps. Never launch training/installs without explicit direction; cluster is Princeton della (Slurm, partition `ailab`). Figures: no bar plots, sans-serif 7–9 pt, no rainbow, view the PNG before committing. Trajectory-result CSVs are **append-only, no dedup** — always guard re-probes with `MODEL_FILTER`.

---

## 3. Mathematical Formulation

### 3.1 Notation
| Symbol | Meaning |
|---|---|
| `a` | architecture `∈{gpt2, mamba}` (the "AR pair"); `esm2` = a third (encoder) arch, Paper A only |
| `s`/`N` | scale ∈ {XS, S, M, L} (param counts in §6) |
| `r` | seed, integer 0…9 |
| `k` | biological concept (`ss3`, `ss8`, `frame`, `gc`, …) |
| `t` | training step (gradient update) |
| `L` | context length (protein 512, genome 1024) |
| `fpt` | FLOPs per token (`flops_per_token` in meta) |
| cell | `(concept, arch, scale)` or `(arch, scale, seed)` |

### 3.2 Decodability score and Δ
A **linear probe** (ridge/logistic; **[INFERRED]** exact regularizer — confirm in `run_trajectory_probes.py`) is fit on residual-stream activations at the best layer, on a homology-safe split.
- `learned_test` = probe test score on the **trained** model; `metric` = **macro-F1** (classification) / **R²** (regression); higher = better.
- `random_init_baseline` = same probe on a **random-init** model.
- **Δ (`delta`)** ` = learned_test − random_init_baseline`.

> **Design decision (why Δ).** [EMPIRICAL] Absolute decodability conflates the baseline: mamba's *random-init* reps are already more decodable, which **flipped the sign** of the inductive-bias coefficient in an early attempt. Δ isolates *learned* structure. The reversal margin (§3.4) uses `learned_test`; the forecaster and γ use `Δ`.

Loss channel: score `= −val_bpr`, `val_bpr` = validation **bits per residue**.

### 3.3 Compute axis `C(t)` (where the FLOP bugs live)
`train_flops`, `train_residues` are per-**run totals** repeated on every checkpoint row; `train_flops = fpt·total_residues`, `train_residues = total_residues`, so
```
C(t) = train_flops · step / train_residues = fpt · step.                          (Eq. 1)
```
**[PROVEN]** `fpt` enters `C` linearly ⇒ an error in `fpt` **rescales that architecture's compute axis** and shifts its curve in `log C`. The reversal compares archs at a *shared* `C`, so only the **ratio** `fpt_mamba/fpt_gpt2` matters (a common factor cancels). `C_f = fpt · num_iterations` (the run's final compute).

### 3.4 Rank-reversal (Paper B anchor) — `scripts/rank_reversal.py`
Per scale, over seeds present for both archs, interpolate in `log C`; margin `Δ_r(C)=score(gpt2)−score(mamba)`.
- Shared range `[LO,HI]` in `log C`; `C_f=exp(HI)`.
- Early budget `log C_e = max(LO, HI + log(early_frac))`, `early_frac=0.01`.
- `d_early = mean_r Δ_r(C_e)`, `d_final = mean_r Δ_r(C_f)`.
- **Reversal** ` := (d_early<0) ∧ (d_final>0)`.
- **Crossover** `C×/C_f`, `C× = inf{C : Δ̄(C')>0 ∀C'≥C}` — the compute fraction below which mamba leads (scale-comparable).
- `P(reversal)`: seed bootstrap, `B=2000` resamples of the seed index.
- `interp` = piecewise-linear in `log C`, endpoint-clamped.

### 3.5 iso-X robustness — `scripts/iso_x_robustness.py`
Reversal under three axes (`AXES`): `flop = fpt·step` (==rank_reversal), `step = step`, `token = (train_residues/max_step)·step`. Plus `own-convergence d_final` (each arch at its own last checkpoint) and `mamba%@flopEnd`. **[EMPIRICAL]** the reversal is **axis-sensitive** — robust for `ss3/ss8`, fragile for others.

### 3.6 Hierarchical reversal posterior — `scripts/hierarchical_reversal.py`
Per modality, separately for early/final margins, over cells `c` (= `concept/scale`) with per-seed observations `y_{c,r}`:
```
y_{c,r} ~ N(θ_c, σ²);  θ_c ~ N(μ, τ²);  μ ~ N(0,100);  σ²,τ² ~ InvGamma(1, 0.01).
P(reversal|data) = E_post[ 1(θ_early[c]<0 ∧ θ_final[c]>0) ].                       (Eq. 2)
```
`θ_early`, `θ_final` a-posteriori independent → AND over paired independent draws. Gibbs, `n_iter=8000`, `burn=2000`, closed-form full conditionals. A side-by-side old bootstrap (`B=4000`) is printed for comparison. Partial pooling shrinks noisy 3-seed cells toward the modality mean; 10-seed cells shrink little.

> **[EMPIRICAL] Conservatism.** The genome hierarchy pools `frame/M` with `gc` (large *positive* early margin ≈+0.17). Pooling shrinks `frame/M` toward positive, so `frame/M P=0.93` is a *lower bound*.

### 3.7 Matched-loss inductive-bias γ (Paper A) — `fit_capability_scaling.py`, `bayes_crosscheck.py`
Per concept:
```
Δ_i = β0 + β_L·(−bpr)_i + β_M·1[M]_i + β_LL·1[L]_i + γ·1[gpt2]_i + ε_i.               (Eq. 3)
```
`γ>0` ⇒ at matched measured loss & scale, gpt2 exposes more learned structure. CI: cluster bootstrap over `(scale,seed)`; model comparison ΔAIC(`H_full` vs `H_loss`). Bayesian cross-check: same LMM + per-cell random intercept `b_cell~N(0,σ_b²)`, exact Gibbs, 4 dispersed chains, split-R̂ + bulk-ESS (self-tested on iid/AR(1)).

> **[HISTORY] Old→final.** An early **logistic** fit for γ **failed** (asymptote unidentifiable — protein curves still rising). Final uses robust observed quantities (`t50`, `Δ_final`) driven by *measured* loss. The logistic survives only for *forecasting* saturating NLP (`forecast.py`, Exp. P).

### 3.8 Forecaster / allocation / calibration — `scripts/forecast_protocol.py`
Unit = protein cell `(k,a,N)`; response = converged `Δ_final` (seed mean). Forecast from ckpts up to `C_e=f·C_f` under strict **leave-one-cell-out (LOCO)**.
- Budgets `f∈{0.01,0.03,0.1,0.3}`; FLOP savings `=1−f`.
- **B-ALLOC** (selection: pick the better *arch* at fixed scale) vs **B-FCAST** (value: predict `Δ_final`).
- Forecasters: `early-probe`, `loss-rank`/`loss-extrap`, `scaling-pop` (`Δ_final~logN`), `cap-linear`, `cap-sat` (bounded logistic — saturating domains), `cap-hier` (`w·cap-linear+(1−w)·scaling-pop`, `w=clip(f/f0)`, `f0=0.1`).
- `LOGN = {S:log 36e6, M:log 143e6, L:log 1e9}` — **[VERIFIED] nominal anchors** for the logN regression, *not* exact param counts (actual M≈135–152M, L≈599–630M; §6).
- Calibration: split-conformal (distribution-free) + Gaussian PI.

**[EMPIRICAL] Key result:** `loss-rank` selection accuracy is **below chance** → selecting by *loss* mis-picks the arch (the reversal as a decision failure). `cap-*` beat chance; `scaling-pop` predicts magnitude well but cannot rank same-scale archs; `cap-hier` is the best value forecaster; conformal ~90% coverage.

### 3.9 FLOP-invariance results
- **[PROVEN] Forecaster.** All quantities use the *relative* budget `f=C_e/C_f`; from Eq. 1 the `fpt` cancels ⇒ invariant. Verified byte-identical (buggy vs corrected; `d13aa6d`, `d8437ef`).
- **[PROVEN] γ.** Identified at matched *measured* loss; a constant per-cell rescale of `C` preserves the loss→Δ relation at matched loss. Byte-identical.
- **[PROVEN] Emergence ordering.** A constant per-cell `log C` shift preserves the *order*.
- **[PROVEN] FLOP-dependent (only):** the cross-architecture matched-compute comparison — reversal margins `d_early`,`d_final` and the crossover — because they compare archs at the same `C` and depend on `fpt_mamba/fpt_gpt2`.
- **[v2 caveat]** These invariance proofs assume a *uniform per-cell* rescale. They are verified on the `_corrected` dirs (uniform `fpt`). The **raw** `trajectory_results/` dir now has *mixed* per-seed `fpt` (§5, §9 Bug #2) and must not be used directly.

### 3.10 Explicit assumptions [v2 ADD]
1. **Seed-pairing** `gpt2-seed-r ↔ mamba-seed-r` is a meaningful pairing (shared seed → comparable init RNG stream). [ASSUMPTION]
2. Linear-probe decodability is a valid *capability proxy*. [ASSUMPTION]
3. Piecewise-linear interpolation in `log C` is adequate between the ~13–21 log-spaced checkpoints. [ASSUMPTION]
4. `early_frac=0.01` (1% of `C_f`) is a sensible "early" budget above the noisy step-1 checkpoint. [DESIGN CHOICE]
5. Batch size is set by *scale*, not *architecture*, so the `batch_tokens` factor cancels in the iso-`C` cross-arch comparison. [INFERRED — verify in configs]
6. `num_iterations ≈ target_param_data_ratio · n_params / batch_tokens` with `target_param_data_ratio ≈ 12`. [INFERRED from `CLAUDE.md`]

---

## 4. Repository / File Structure

**Two nested git repos:**
- **Outer:** `/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat` — branch `submission` @ `77b14c3`. Papers under `brain/papers/` (`*.md/*.tex` force-added, gitignored). `nanoprot/` is *untracked* here.
- **Inner (the code, all results):** `…/nanochat/nanoprot` — branch `main` @ `4b919fc`. GitHub `ygzdvr/nanoprot`; models on HF `yagizdevre` (**not** `ygzdvr`).

### 4.1 Package `nanoprot/`
| Path | Purpose | Status |
|---|---|---|
| `nanoprot/models/gpt2.py` | Transformer LM; `estimate_flops()` (attention `L`-factor legitimate). | Active |
| `nanoprot/models/mamba.py` | State-space LM; `estimate_flops()` — **FLOP bug fixed here** (§9). | Active |
| `nanoprot/models/esm2.py` | Encoder (BERT-style); Paper A only. | Active |
| `nanoprot/optim.py` | `MuonAdamW`/`DistMuonAdamW`. | Active |
| `nanoprot/data/{builder,dataloader,dataset,mlm}.py` | Data pipeline. | Active |
| `nanoprot/tokenizers/{residue,bpe,genome,esm2}.py` | Tokenizers (protein V=33, genome V=10). | Active |
| `nanoprot/training/{loop,checkpoint}.py` | Training loop + checkpoint I/O. | Active |
| `nanoprot/eval/{loss,protein}.py`, `nanoprot/{config,runtime,attention}.py` | Eval/config/runtime/attention. | Active |

### 4.2 Analysis / plotting `scripts/` (49 files; load-bearing)
| Script | Lines | Role | Status |
|---|---|---|---|
| `rank_reversal.py` | 209 | Paper B anchor; margins + crossover + bootstrap → `docs/rank_reversal.csv`. | Active |
| `hierarchical_reversal.py` | 168 | Hierarchical posterior (Eq. 2); forest figs. | Active (edited this session) |
| `iso_x_robustness.py` | 188 | Reversal under flop/step/token; defines `AR,interp,load_raw,AXES`. | Active |
| `reversal_corrected_flops.py` | 163 | Corrected-FLOP recompute; `corrected_fpt_lookup()` (gate-checked). | **Active (edited: genome-aware + guard)** |
| `make_corrected_trajectory.py` | 88 | Corrected-FLOP CSV copies (OVERWRITE `train_flops`). | **Active (edited: per-domain cfpt)** |
| `forecast_protocol.py` | 292 | B-ALLOC/FCAST/CAL (LOCO). | Active |
| `fit_capability_scaling.py` | 197 | γ (Eq. 3) + emergence N-scaling. | Active |
| `bayes_crosscheck.py` | 205 | γ Bayesian cross-check (Gibbs LMM). | Active |
| `run_trajectory_probes.py` | 225 | Runs linear probes on every checkpoint → concept CSVs. | Active |
| `prepare_genome_probes.py` | 340 | Genome labels (exon/frame/gc/splice). | Active |
| `analyze_emergence.py`, `plot_emergence.py` | 195/375 | Emergence order (Paper A). | Active |
| `pythia_capability.py`, `pythia_forecast.py` | 118/129 | Forecaster NLP cross-domain validation (Exp. P). | Active |
| `synth_selectivity.py`, `synth_train_impl.py` | 352/399 | **Abandoned** synthetic selectivity (Exp. S). | Diagnostic only |
| `plot_rank_reversal.py`, `plot_crossover_scaling.py` | 98/60 | Reversal + crossover figures. | Active (crossover reads `docs/rank_reversal.csv`) |
| `gen_release_configs.py`, `gen_genome_configs.py`, `gen_sweep_configs.py` | — | Config generators. | Active |
| `train.py` | 99 | Training entry. | Active |

*(Also: `aggregate_results, fit_lnd_sweep, forecast, forecast_capability, make_model_card, plot_*, posthoc_val_loss_eval, prepare_*, probe_*, run_probes, scaling_*, selectivity_*, show_config, upload_release`.)*

### 4.3 Slurm / shell `runs/`
`genome_train.slurm` (**gpu:1**), `genome_probe.slurm` (`TASKS` default `genome_exon` → override to `genome_frame`), `train_trajectory.slurm` (protein train, gpu:2), `trajectory_probe.slurm` (`MODEL_FILTER`), `train_release.slurm`, `train_sweep.slurm`, `genome_valloss.slurm`, `posthoc_valloss.slurm`, `prepare_{uniref50,genome,genome_probes}.slurm`, `probe_sweep.slurm`, `synth_selectivity.slurm`, `pythia_capability.slurm`, `regen_figures.sh`, `install_mamba_ssm.sh`, `calibrate_throughput.slurm`, `genome_smoke.slurm`.

### 4.4 Docs `docs/` (tracked)
`hierarchical_reversal_genome10_corrected.txt` (**headline, this session**), `hierarchical_reversal_10seed_corrected.txt` (protein, `3669a2e`), `rank_reversal{,_protein,_genome}.csv`, `reversal_corrected_flops.txt`, `iso_x_robustness.txt`, `forecast_protocol{,_genome}_report.md`, `capability_scaling_report.md`, `bayes_crosscheck_report.md`, `emergence_report.{md,csv}`, `pythia_forecast_report.md`, atlas notes, `_bc/_cs/_fc_corrected.md`. Figures in `docs/figures/` (§5).

### 4.5 Papers (outer repo)
`../brain/papers/paper_A.tex`, `../brain/papers/paper_B.tex` (+PDFs). **`paper_B.tex` is stale** (§13).

---

## 5. Data Inventory

All under `NANOPROT_BASE_DIR = …/nanoprot/.cache/nanoprot` (**gitignored** — regenerable).

| Name | Path (`.cache/nanoprot/`) | Type | Shape / Schema | How Generated | Used By | Status | Notes |
|---|---|---|---|---|---|---|---|
| UniRef50 shards | `uniref50_parquet/` (17G) | Input | parquet, protein seqs | `prepare_uniref50` | protein train | Keep | ctx 512, V=33 |
| hg38 shards | `hg38_parquet/` (1.3G) | Input | parquet, DNA char | `prepare_genome` | genome train | Keep | ctx 1024, V=10 |
| v0.5 release ckpts | `release/` (58G, **36 cells**) | Model | `model_/meta_/optimizer_*` | `train_release.slurm` | γ, cfpt lookup | Keep | gpt2/mamba/esm2 × XS/S/M/L × s0–2 (all pre-fix ⇒ mamba metas buggy) |
| protein trajectory ckpts | `release_traj/` (579G) | Model | **21** log-spaced ckpts/cell | `train_trajectory.slurm` | protein probes | Keep | source of `trajectory_results` |
| genome trajectory ckpts | `genome_traj/` (161G, **26 cells**) | Model | **13** ckpts/cell | `genome_train.slurm` | genome probes, genome cfpt | Keep | gpt2/mamba×M s0–9, ×S s0–2 |
| sweep ckpts | `sweep/` (113G) | Model | L(N,D) sweep | `train_sweep.slurm` | scaling-law scripts | Keep (secondary) | off the reversal path |
| probe caches | `probes/` (20G) | Intermediate | 18 concept label dirs (+features **[INFERRED]**) | `prepare_*_probe*` | all probes | Keep | see list below |
| **protein probe results** | `trajectory_results/` (53M) | Generated | concept CSVs + `traj_*.json` | `run_trajectory_probes.py` | reversal/forecaster/γ | Keep | **[v2 FIX] MIXED `fpt`** (mamba s0–2 buggy 2.944e9, s3–9 corrected 9.097e8) ⇒ raw axis invalid; use `_corrected` |
| **protein corrected** | `trajectory_results_corrected/` (49M) | Generated | corrected `train_flops` (uniform mamba 9.097e8) | `make_corrected_trajectory.py` | corrected reversal/hier | **Keep — the valid protein dir** | 10 seeds at ss3/ss8/M |
| **genome probe results** | `trajectory_results_genome/` (13M) | Generated | `genome_{exon,frame,gc,splice}.csv`+`val_loss.csv`+json | `genome_probe.slurm` | genome reversal/hier | Keep | **also MIXED `fpt`**; `genome_frame.csv` now 10 seeds |
| **genome corrected** | `trajectory_results_genome_corrected/` (4.6M) | Generated | corrected `train_flops` (genome axis, gpt2 9.637e8) | `make_corrected_trajectory.py` | genome hier | **Keep — the valid genome dir (this session)** | |
| trajectory backup | `trajectory_results.bak_preL/` (9.9M) | Backup | old protein results | manual | — | Can delete | pre-L snapshot |
| synth results | `synth_results/` (113K) | Generated | synthetic outputs | `synth_selectivity.py` | Exp. S | Keep (diagnostic) | negative control |
| misc | `demo/`, `genome_smoke/` (276M ea), `probe_results*/` | Intermediate | smoke/demo | various | — | Can delete | scratch |

**Probe concept dirs** (`probes/`): protein — `ss3_{netsurfp,dssp,swissprot}`, `ss8_netsurfp`, `rsa_netsurfp`, `active_site_swissprot`, `disorder_swissprot`, `transmembrane_swissprot`, `signal_swissprot`, `subcellular_swissprot`, `ec_swissprot`, `pfam_swissprot`, `fold_cath`; genome — `genome_{exon,frame,gc,splice}`; plus `raw/`.

**Concept CSV schema** (`arch,scale,seed,step,train_residues,train_flops,concept,source,metric,best_layer,learned_test,delta`):
- `concept` = clean name; **filename ≠ concept value** (`active_site_swissprot.csv`→`active`; `ec_swissprot.csv`→`ec_class`; `pfam_swissprot.csv`→`pfam_family`; `fold_cath.csv`→`fold`; `ss3_netsurfp.csv`→`ss3`).
- Implied `fpt = train_flops/train_residues`. **[v2 FIX] Raw dirs are MIXED** (per-seed `fpt` differs: buggy s0–2, corrected s3–9). Only `_corrected` dirs OVERWRITE to a single `fpt` per `(arch,scale)`.
- **Append-only, no dedup, keyed by columns not filename** → dedup by `(arch,scale,seed,step)`; always re-probe with `MODEL_FILTER`.

**`val_loss.csv`:** `arch,scale,seed,step,val_loss,val_bpr,train_residues`.
**`docs/rank_reversal.csv`:** `task,scale,n_seed,C_f,early_frac,d_early,d_early_lo,d_early_hi,cross_over_frac,d_final,d_final_lo,d_final_hi,reversal,p_reversal`. **[v2 FIX] the committed copy is STALE** (§8/§9).
**`meta_*.json` keys:** `best_val_bpr, config, flops_per_token, last_val_bpr, last_val_loss, model_config, n_params, name, num_iterations, smooth_loss, step, total_flops, total_residues, train_time_sec, training_config, version, world_size`. `model_config`: `depth, d_model, n_heads, expand, d_state, max_seq_len, vocab_size, …`.

**[v2 FIX] Checkpoint schedule.** Protein trajectory = **21** log-spaced steps/cell: `1,2,3,5,7,10,15,21,31,46,67,99,145,212,311,456,669,981,1439,2109,3098`. Final step = `num_iterations`, which **scales ∝ n_params**: gpt2-M **3098**, mamba-M **3473** (151.7M/135.4M = 1.12 = 3473/3098). Genome trajectory = **13** ckpts/cell over a similar range (gpt2-M→3098, mamba-M→3473, S→~800). *(The earlier "20 steps ending 2109" was wrong — it omitted the final ckpt(s).)*

**Figures (`docs/figures/`, tracked):**
- `hier_reversal_{protein,genome}.png` — **current headline** (edited this session; per-cell `n` labels).
- `rank_reversal.png`, `rank_reversal_genome.png` — **[v2 FIX] BOTH STALE**: `rank_reversal_genome.png` predates the axis fix (`mtime 2026-07-02`); `rank_reversal.png` + the CSV predate the 10-seed + FLOP fix. Regenerate on `_corrected`.
- `crossover_scaling.png` — **[v2 FIX] STALE** (reads the stale `docs/rank_reversal.csv`; crossover is FLOP-dependent — see §8). Regenerate.
- `forecast_protocol.png`, `inductive_bias_forest.png`, `emergence_*`, `pythia_timing.png`, `synth_selectivity_law.png`, `scaling_*`, `where_layers.png`, `probes*.png`.

---

## 6. Model Inventory

Single-dial design (`--depth` sets everything). `gpt2`: `d_model=depth×64`, `n_heads=depth` (head_dim 64), ReLU² MLP, RoPE, QK-norm. `mamba`: `expand=2`, `d_state=16`, `n_heads=1`. Optimizer `MuonAdamW`; causal LM cross-entropy.

### 6.1 Protein models (`L=512`, `V=33`) — [VERIFIED from meta]
| arch | scale | depth | d_model | n_heads | n_params | `fpt` buggy | `fpt` corrected | inflation buggy/corr |
|---|---|---|---|---|---|---|---|---|
| gpt2 | XS | 6 | 384 | 6 | 10,739,966 | 7.8006e7 | 7.8006e7 | 1.00× |
| gpt2 | S | 9 | 576 | 9 | 36,090,440 | 2.4707e8 | 2.4707e8 | 1.00× |
| gpt2 | M | 14 | 896 | 14 | 135,390,414 | 8.8666e8 | 8.8666e8 | 1.00× |
| gpt2 | L | 23 | 1472 | 23 | 599,354,680 | 3.7968e9 | 3.7968e9 | 1.00× |
| mamba | XS | 7 | 448 | 1 | 9,186,688 | 3.6257e8 | **5.4889e7** | **6.61×** |
| mamba | S | 11 | 704 | 1 | 35,024,000 | 9.6951e8 | **2.0973e8** | **4.62×** |
| mamba | M | 18 | 1152 | 1 | 151,749,504 | 2.9441e9 | **9.0967e8** | **3.24×** |
| mamba | L | 29 | 1856 | 1 | 630,572,288 | 9.0625e9 | **3.7817e9** | **2.40×** |

Corrected `fpt_mamba/fpt_gpt2` at M (protein) = 9.0967e8/8.8666e8 = **1.026** (near-equal, param-matched). **[v2 FIX] The inflation is scale-dependent** (scan share of the buggy total: XS 85%, S 79%, M 69%, L 58%); the commit/docstring "~2.6×" is a rough single figure (closest to L 2.40×). The separate "3.3× more FLOP-expensive than gpt2" = buggy mamba-M/gpt2-M = 2.944e9/8.867e8 = 3.32× (a *comparison*, not the self-inflation).

### 6.2 Genome models (`L=1024`, `V=10`) — [VERIFIED]
| arch | scale | depth | d_model | n_params | `fpt` logged (s0) | `fpt` corrected | final step |
|---|---|---|---|---|---|---|---|
| gpt2 | S | 9 | 576 | 36,090,440 | 2.7892e8 | 2.7892e8 | 826 |
| gpt2 | M | 14 | 896 | 135,390,414 | 9.6373e8 | 9.6373e8 | 3098 |
| mamba | S | 11 | 704 | 35,024,000 | 1.7308e9 (buggy) | 2.0973e8 | 801 |
| mamba | M | 18 | 1152 | 151,749,504 | 4.9826e9 (buggy) | **9.0967e8** | 3473 |

**Two pinned facts** [PROVEN]: (1) mamba corrected `fpt` is **context-invariant** (genome mamba-M `9.0967e8` = protein mamba-M `9.0967e8`; scan O(1)/token). (2) gpt2 `fpt` is **context-dependent** (genome gpt2-M `9.6373e8` = protein `8.8666e8 ×1.087`; attention over 1024 vs 512). ⇒ corrected genome ratio `fpt_mamba/fpt_gpt2 = 0.944` (mamba ~5.6% *cheaper*/token). Applying the *protein* ratio (1.026) to genome understates gpt2 by ~8.7% — Bug #2.

### 6.3 Third arch: esm2 (Paper A only)
Encoder / masked-LM. **[v2 ADD] `fpt = 1.003e9`** (from `trajectory_results` esm2 rows). Dims/hparams **[UNKNOWN]** — see `nanoprot/models/esm2.py`, `release/nanoprot-esm2-*`. Used for the atlas ("encoder out-decodes AR on structure"); stays off the AR matched-loss axis. Status: kept, not on the reversal path.

### 6.4 Training hyperparameters [v2 ADD]
- `num_iterations` per cell **[VERIFIED]** = 3098 (gpt2-M) / 3473 (mamba-M), ∝ n_params; `C_f = fpt·num_iterations`.
- `target_param_data_ratio ≈ 12` **[INFERRED, `CLAUDE.md`]**.
- LR schedule, batch size, weight decay, warmup **[UNKNOWN]** — see `nanoprot/training/loop.py`, `nanoprot/optim.py`, the generated configs.

### 6.5 Seed coverage
- Protein trajectory: gpt2/mamba × S/M × s0–2 for most concepts; **ss3, ss8, rsa at M extended to s0–9** (esm2 present at S/M s0–2).
- Genome trajectory: gpt2/mamba × M × s0–9 (**only `frame` probed for s3–9**); × S × s0–2. `exon/gc/splice` remain n=3.
- Release: 36 cells = gpt2/mamba/esm2 × XS/S/M/L × s0–2.

### 6.6 Reference model (`CLAUDE.md`)
`prot_d20`: 1.17 B params, T=512, ratio 12, 5.49 B residues — the *separate* SAE thesis component; **not** used by these results. Kept, tangential.

---

## 7. Experiment Log

### Experiment A — Protein rank-reversal (Paper B core)
- **Goal/Hypothesis:** early-proxy mis-ranking; `d_early<0 ∧ d_final>0` for structural concepts.
- **Setup/Data:** `release_traj/` → `run_trajectory_probes.py` → `trajectory_results/` → `make_corrected_trajectory` → hierarchical posterior on `_corrected`. Scales S,M; `early_frac=0.01`; `n_boot=2000`.
- **Command:** `python -m scripts.hierarchical_reversal --protein-dir .cache/nanoprot/trajectory_results_corrected --genome-dir .cache/nanoprot/trajectory_results_genome_corrected`.
- **Results [EMPIRICAL, corrected, n=10]:** `ss3/M P=0.92` (CrI [−0.038,−0.012], excl. 0); `ss8/M P=0.75` (CrI [−0.032,−0.007], excl. 0); population `P(μ_early<0)=1.00` (CrI [−0.017,−0.006]).
- **Problems:** originally on buggy FLOPs + 3-seed bootstrap. **Status:** ✅ confirmed core. **Follow-up:** paper rewrite; regenerate `rank_reversal.png`/crossover on `_corrected`.

### Experiment B — Genome cross-domain replication (B-GEN)
- **Goal/Hypothesis:** reversal in DNA on a *structural/periodic* concept.
- **Setup:** `genome_traj/` → `genome_probe.slurm` (`TASKS=genome_frame`, `MODEL_FILTER`) → corrected genome dir → hierarchical.
- **Command (this session):** `TASKS=genome_frame MODEL_FILTER='(gpt2|mamba)-M-s[3-9]' sbatch --partition=ailab --time=02:00:00 --array=0-13%8 --export=ALL,TASKS=genome_frame,MODEL_FILTER='(gpt2|mamba)-M-s[3-9]' runs/genome_probe.slurm` → `python -m scripts.make_corrected_trajectory` → hierarchical.
- **Results [EMPIRICAL, n=10, corrected genome axis]:** `frame/M P=0.93` (θ_early −0.016, CrI [−0.035,+0.005]); `gc/M P=0.00`, `gc/S P=0.00` (negative control); `splice/M P=0.24`, `exon/M P=0.57` (n=3). **[v2 ADD] Effect size:** frame mamba-M-s9 @step 3473 `learned_test=0.4658` (macro-F1), `Δ=+0.0946` (chance ≈0.33).
- **[v2 FIX] Three-step evolution of the genome frame number:** `P=1.0`, crossover 0.20 **(naive seed-bootstrap, buggy FLOPs, `c43a967`, written into `paper_B`)** → `0.77` **[INFERRED, prior-session record: hierarchical n=3 on corrected FLOPs but the WRONG (protein-cfpt) genome axis; not reproducible after `e1dbb9e`]** → **`0.93`** (hierarchical n=10, correct genome axis).
- **Interpretation:** reversal replicates on reading-frame (periodic/structural analog of secondary structure), not on composition (GC) → tracks learned structure. **Status:** ✅ confirmed. **Follow-up:** recompute genome crossover on `_corrected`; regenerate `rank_reversal_genome.png`.

### Experiment C — Forecaster / allocation / calibration (B-ALLOC/FCAST/CAL)
- **Goal:** turn the reversal into a cheap, calibrated decision rule (LOCO).
- **Command:** `python -m scripts.forecast_protocol` (protein); `--concepts exon` for genome.
- **Results [EMPIRICAL]:** `loss-rank` selection **< chance**; `cap-*` beat chance; `cap-hier` best value; conformal ~90%; savings `1−f`. **[PROVEN] FLOP-invariant** (`d13aa6d`, `d8437ef`). **Status:** ✅ survives the bug. **Follow-up:** more seeds/scales; per-cluster bootstrap; NUTS.

### Experiment D — Matched-loss inductive bias γ (Paper A)
- **Method:** Eq. 3 OLS + cluster bootstrap + AIC (`fit_capability_scaling.py`); Gibbs LMM (`bayes_crosscheck.py`).
- **Results [EMPIRICAL]:** `γ>0`, `P(γ>0)≈1` for ss3/ss8/disorder; Bayes brackets OLS (R̂<1.01, ESS>400). **[PROVEN] FLOP-invariant.** **Status:** ✅ Paper A crown jewel intact.

### Experiment E — iso-X robustness
- **Method:** flop/step/token axes + own-convergence (`iso_x_robustness.py`).
- **Results [EMPIRICAL]:** axis-sensitive; robust core = secondary structure; fragile cells fail under iso-step. **Status:** ✅ motivated dropping fragile cells + the FLOP-bug hunt. **Output:** `docs/iso_x_robustness.txt`.

### Experiment F — Emergence / developmental atlas (Paper A)
- **Method:** `analyze_emergence.py`/`plot_emergence.py`; `t50` = log-compute at ½·Δ_final.
- **Results [EMPIRICAL]:** function (active-site) last & weakest, reproducible across arch/scale/objective; 4 breadth axes distribute across the timeline. **[PROVEN]** ordering FLOP-invariant. **Status:** ✅ atlas core. **Outputs:** `docs/emergence_report.{md,csv}`, `docs/figures/emergence_*`.

### Experiment P — Pythia forecaster NLP validation [v2 ADD]
- **Goal:** show the forecaster/allocation generalizes beyond proteins to a well-known NLP suite (Pythia), where capability curves *saturate* (the bounded `cap-sat`/logistic form applies).
- **Method:** `scripts/pythia_capability.py`, `scripts/pythia_forecast.py`; `runs/pythia_capability.slurm`.
- **Outputs:** `docs/pythia_capability.csv`, `docs/pythia_forecast_report.md`, `docs/figures/pythia_timing.png`.
- **Results [EMPIRICAL — details not re-verified this session; see the report]:** the bounded forecaster extrapolates saturating NLP capability curves (cross-domain generality for Paper B). **Status:** ✅ supporting. **Follow-up:** confirm the exact numbers in `pythia_forecast_report.md` before citing.

### Experiment S — Synthetic "selectivity-law" (ABANDONED)
- **Goal:** a controlled dial isolating *why* SSMs win early (spotlight centerpiece attempt).
- **Method:** `synth_selectivity.py`/`synth_train_impl.py` — synthetic tasks with a tunable selectivity/periodicity dial; minimal attn vs SSM (real MambaBlock); frozen-probe decodability.
- **Failures (confounds, in order):** (1) crippled SSM `dt`-init → 1-step memory → **wrong-sign** (`55a3da3`); (2) iso-FLOP artifact; (3) noise-floor ~1% measurement; (4) generic AdamW not `MuonAdamW` (`ed02d33`); (5) end-task accuracy vs frozen-probe decodability.
- **Result:** after fixes, the reversal did **not** reproduce as a supervised-learning-speed effect → **pretraining-specific**.
- **Lesson:** *audit the instrument before interpreting a surprising result.* **Status:** ❌ abandoned as centerpiece; kept as negative control (`synth_results/`, real-mamba diagnostic `d7cecb8`). **Do not** revive without a new hypothesis.

---

## 8. Results Summary

**Confirmed — the defensible core:**
| Result | Value | Status |
|---|---|---|
| Protein reversal `ss3/M` | `P=0.92`, n=10, CrI excl. 0 | robust |
| Protein reversal `ss8/M` | `P=0.75`, n=10, CrI excl. 0 | robust |
| Genome reversal `frame/M` | `P=0.93`, n=10 | robust (conservative) |
| Negative control `gc` | `P=0.00` | clean |
| Protein population `P(μ_early<0)` | `1.00`, CrI [−0.017,−0.006] | robust |
| Forecaster / savings | `loss-rank<chance`; `cap-*`>chance; ~90% @ f=0.1 | **FLOP-invariant [PROVEN]** |
| Inductive bias γ (Paper A) | `γ>0`, `P(γ>0)≈1` | **FLOP-invariant [PROVEN]** |
| Emergence: function-last | reproducible across arch/scale/objective | **FLOP-invariant [PROVEN]** |

**Overturned / dropped (do NOT reuse):**
- `fold/pfam/ec` "reversal (rev=1)" — bug-manufactured; now `fold/M P≈0.59`, `pfam/M P≈0.60` (n=3). Dropped from the reversal headline.
- "mis-ranking window shrinks ~20–25× with scale" — largely a bug artifact. **[v2 VERIFIED]** on corrected data the S→M crossover ratio is ~4.8× for ss3 (`ss3/S 0.848 → ss3/M 0.178`), not 20–25×. Restate honestly.
- naive `P(reversal)=1.0` (3-seed bootstrap) — replaced by the hierarchical posterior.

**[v2 FIX] Stale artifacts (must regenerate on `_corrected`):** `docs/rank_reversal.csv`, `docs/figures/rank_reversal.png`, `docs/figures/rank_reversal_genome.png`, `docs/figures/crossover_scaling.png`. **Verified:** the committed `rank_reversal.csv` ss3/M crossover 0.034 matches *neither* the current buggy recompute (0.089) nor the corrected recompute (0.178). Crossover is FLOP-dependent and shifted materially (ss3/M ×2, ss8/M 0.162→0.218).

**Uncertain / not done:** corrected crossover numbers for the paper; fold/pfam/genome-exon/gc/splice at 10 seeds (n=3 now); L-scale reversal (gated on probing); esm2 dims; Exp. P exact numbers.

---

## 9. Bugs, Debugging, and Fixes

### Bug #1 — mamba FLOPs/token inflated (scale-dependent) — FIXED `28206cc`
- **Symptom:** mamba appeared far more FLOP-expensive/token than gpt2; iso-FLOP axis handicapped mamba early, flattered gpt2 at convergence.
- **Cause:** `nanoprot/models/mamba.py::estimate_flops` multiplied the selective-scan term by `cfg.sequence_len`. But `estimate_flops` returns FLOPs **per token**, and the scan does O(1)/token (no `L`).
- **Old:** `scan_flops = 6·D_inner·d_state·n_layer·L`. **Final:** `scan_flops = 6·D_inner·d_state·n_layer`, return `6·(nparams − non_matmul) + scan_flops` (`D_inner = expand·d_model`).
- **[v2 FIX] Impact is scale-dependent** (inflation buggy/corr): **XS 6.61×, S 4.62×, M 3.24×, L 2.40×** (scan share of buggy total 85/79/69/58%). The commit/docstring "~2.6×" ≈ L; the "3.3× vs gpt2" is a comparison at M.
- **Verification [PROVEN]:** three ways agree on mamba-M corrected `fpt=9.0967e8`: analytic recompute; gate `6·n_params+scan_bug ≈ logged` within 0.23%; fresh training with fixed code logs `9.0967e8` directly. (mamba-M buggy scan `= 3.98e6·512 = 2.038e9`, 69% of buggy total 2.944e9.)
- **Downstream:** `037bcf5`, `d13aa6d`, `d8437ef`, `a125101`, `3669a2e`.

### Bug #2 — genome axis used protein cfpt + mixed-meta double-correction — FIXED `e1dbb9e` (this session)
- **Symptom:** genome frame reversal mis-registered; genome gpt2 axis understated.
- **Cause A (context):** `make_corrected_trajectory.py` derived corrected `fpt` **only from the protein release** and applied it to genome rows. Genome ctx 1024 ⇒ genome gpt2 `fpt` +8.7%; mamba corrected `fpt` context-invariant.
- **Cause B (mixed metas):** **[v2 FIX] this affects BOTH modalities.** After Bug #1's fix, newly-trained seeds log the *already-corrected* `fpt`, while pre-fix seeds log the *buggy* `fpt`. Verified: protein `trajectory_results` mamba/M has s0–2 = 2.944e9 (buggy), s3–9 = 9.097e8 (corrected); genome mamba/M has s0–2 = 4.98e9 (buggy), s3–9 = 9.097e8 (corrected).
- **Fix:** (i) **per-domain cfpt** — genome rows ← `genome_traj` configs, protein rows ← `release`, routed by dir name; (ii) a **guard** `fpt > 1.5·6·n_params` in `corrected_fpt_lookup` (⇔ O(L) scan still present) so mixed metas are not re-corrected; (iii) `make_corrected_trajectory` **OVERWRITES** `train_flops := cfpt·train_residues` (not a ratio-multiply), collapsing mixed per-seed `fpt` to one value per `(arch,scale)`. Division of labor: the **guard** handles mixed metas *read from `genome_traj`* for the cfpt; the **OVERWRITE** handles the mixed rows in *both* result dirs.
- **Verification [PROVEN]:** protein ratio 1.026 (unchanged), genome ratio 0.944; gates PASS (protein 0.23%, genome 0.11%). `frame/M` 0.77 (n=3, wrong axis) → **0.93** (n=10, correct axis).

### Bug #3 — genome training hung at Step 0 (2-GPU DDP) — FIXED `1289f50`
- **Cause:** genome multi-GPU DDP data-loader split bug. **Fix:** single-GPU (`gpu:1`); `cpus-per-task 16→8` (`aee3ec9`).

### Bug #4 — genome training TIMEOUT (this session) — FIXED (relaunch)
- **Cause:** genome mamba-M ~2× slower (ctx 1024, gpu:1); timed out at `--time=04:00:00`. **Fix:** `--time=10:00:00`; all reached step 3473. (della QOS: keep `--time ≥ 02:00:00`.)

### Bug #5 — synthetic wrong-sign result — see Exp. S (five confounds; demoted to negative control).

### Figure honesty fix (this session)
`hierarchical_reversal.py` title showed `int(mean seeds)=3`, hiding `frame/M`'s 10 seeds → per-cell `n` labels + seed range (`4b919fc`).

**Unresolved / to watch:** `rank_reversal{,_genome}.csv/png` + `crossover_scaling.png` are **stale** (§8); `reversal_corrected_flops.py::main` still points `--release` at the protein dir for its *diagnostic* genome section (the *paper* path `make_corrected_trajectory` is fixed) — low-priority.

---

## 10. Current Working Pipeline (end-to-end)

1. **Inputs:** `uniref50_parquet/` (ctx 512, V=33); `hg38_parquet/` (ctx 1024, V=10).
2. **Configs:** `gen_release_configs.py` / `gen_genome_configs.py` → `configs/{trajectory,…}/*.yaml`.
3. **Train (GPU):** `train_trajectory.slurm` (protein, gpu:2) `torchrun --standalone --nproc_per_node=$NPROC -m scripts.train -- --config $CFG`; `genome_train.slurm` (genome, **gpu:1**). 21 (protein) / 13 (genome) log-spaced ckpts/cell → `release_traj/`, `genome_traj/`. Skip-if-complete via `step==num_iterations`. `WANDB_MODE=offline`, `NANOPROT_DISABLE_FA3=1`.
4. **Probe (GPU):** `trajectory_probe.slurm` (protein; `MODEL_FILTER`), `genome_probe.slurm` (`TASKS=genome_frame`, `MODEL_FILTER`) → concept CSVs (**mixed `fpt`**).
5. **FLOP correction (CPU):** `python -m scripts.make_corrected_trajectory` → `trajectory_results_corrected/`, `trajectory_results_genome_corrected/` (per-domain cfpt; gates must PASS <2%). **Always analyze the `_corrected` dirs.**
6. **Analysis (CPU):**
   - Reversal posterior: `python -m scripts.hierarchical_reversal --protein-dir …_corrected --genome-dir …_genome_corrected`.
   - Reversal/crossover figs: `python -m scripts.rank_reversal --results-dir …_corrected` then `plot_rank_reversal.py` / `plot_crossover_scaling.py` **(regenerate — currently stale)**.
   - Forecaster: `python -m scripts.forecast_protocol` (+`--concepts exon`). γ: `fit_capability_scaling.py`, `bayes_crosscheck.py`. Emergence: `analyze_emergence.py`.
7. **Outputs:** committed `docs/*` + `docs/figures/*`; papers in `../brain/papers/*.tex`.

**Environment:** `source starter.sh` (loads `gcc/11`, `cudatoolkit/12.8`, Cargo, `.venv`, caches, `NANOPROT_BASE_DIR`). Package manager **uv** (`uv sync --extra gpu --dev`). Slurm partition `ailab`. Libs: PyTorch, numpy, scipy (`least_squares`), matplotlib; `mamba_ssm` via `runs/install_mamba_ssm.sh`. Exact versions **[UNKNOWN]** — see `pyproject.toml`/`uv.lock`.

---

## 11. Important Implementation Details

- **`rank_reversal.load_meta / load_task_curves`:** `(arch,scale,seed)→sorted[(log C, score)]`, `C=fpt·step` (Eq. 1); `val_bpr` as `−bpr`; `interp` piecewise-linear in `log C`, clamped.
- **`reversal_corrected_flops.corrected_fpt_lookup(release_dir)`:** globs both `nanoprot-*` and `genome-*`; **guard** `fpt>1.5·6·n_params` picks the correction branch (pre-fix) vs use-as-is (post-fix); gate `|6·n_params+scan_bug−fpt|/fpt<2%`.
- **`make_corrected_trajectory.cfpt_from_release` + per-dir routing:** `cfpt[(arch,scale)]=mean_seed fpt_corr`; genome dirs←`genome_traj`, protein←`release`; **OVERWRITE** `train_flops:=cfpt·train_residues`; copies `val_loss.csv`+`*.json` verbatim.
- **`hierarchical_reversal.gibbs_hier`:** conjugate Gibbs (§3.6); separate early/final chains; reversal AND-ed over paired draws.
- **`forecast_protocol`:** `cap_sat` bounded logistic via `scipy.least_squares` (soft_l1); `cap_hier` blends cap-linear + scaling-pop with `w=clip(f/f0)`; LOCO; split-conformal.
- **`gpt2.estimate_flops`:** `6·(nparams−nparams_exclude) + Σ_layer 12·h·q·effective_seq` (attention `L`-factor legitimate; excludes embeddings/scalars/value-embeds).
- **Design:** depth is the only dial; d12/M is the muP reference; keep SDPA/non-Hopper paths; preserve checkpoint back-compat.

---

## 12. Open Questions and Unknowns

1. **Corrected crossover numbers** (protein + genome) for the paper — old values stale/FLOP-dependent. [Actionable, CPU]
2. **10-seed extension** for fold/pfam + genome exon/gc/splice (n=3 now). [Optional GPU]
3. **L-scale reversal** — gated on L probing. [UNKNOWN status]
4. **esm2 dims/hparams** — [UNKNOWN].
5. **Exact probe regularizer** — [INFERRED linear]; confirm in `run_trajectory_probes.py`.
6. **Training hparams** (LR/batch/wd/warmup) and **library versions** — [UNKNOWN]; `training/loop.py`, `optim.py`, `uv.lock`.
7. **Mechanism of the reversal** — [HYPOTHESIS] SSM recurrence favors periodic/structural features early (codon period-3 ≙ helix/strand periodicity); the synthetic could not isolate it (Exp. S).
8. **`probes/` (20G) exact contents** — [INFERRED] labels + cached features; confirm.
9. **Exp. P (Pythia) exact numbers** — not re-verified this session; read `docs/pythia_forecast_report.md`.

---

## 13. Recommended Next Steps (ordered checklist)

1. **Regenerate the stale figures on the corrected axis:** `python -m scripts.rank_reversal --results-dir .cache/nanoprot/trajectory_results_corrected --out docs/rank_reversal.csv`; genome equivalent → `docs/rank_reversal_genome.csv`; then `plot_rank_reversal.py` and `plot_crossover_scaling.py`. Record the **corrected** protein *and* genome crossover `C×/C_f`.
2. **Rewrite `../brain/papers/paper_B.tex`** around the corrected core:
   - Replace every `P(reversal)=1.0` (lines ~29, 45, 84, 102, 154, 156, 179) with the hierarchical posterior (`ss3/M 0.92`, `ss8/M 0.75`, `frame/M 0.93`) + the uncertainty model.
   - Replace the genome crossover `0.20` (lines ~156–157, 180) with the recomputed value; swap the stale `rank_reversal_genome.png` (line 176).
   - **Drop** fold/pfam from the reversal headline; restate the "scale-law" honestly (~4.8× S→M for ss3, not 20–25×).
   - Re-lead abstract/intro on the three defensible legs: (i) reproducible reversal with honest uncertainty, (ii) FLOP-invariant *practical* forecaster (loss mis-picks; ~90% savings), (iii) cross-domain replication + clean GC negative control.
3. **Sync Paper A** (γ, emergence — already FLOP-invariant); confirm no stale FLOP-dependent panels.
4. **Full top-venue reviewer read** once (1)–(3) land.
5. *(Optional GPU)* extend fold/pfam + genome exon/gc/splice to 10 seeds if reviewers want breadth.

**Verify before asserting:** re-run `make_corrected_trajectory` (gates PASS) and `hierarchical_reversal` to reproduce `ss3/M 0.92`, `ss8/M 0.75`, `frame/M 0.93`; confirm regenerated figures read the `_corrected` dirs; check figure mtimes post-regen.

---

## 14. Things Not To Repeat

- **Do not** apply protein `cfpt` to genome rows (Bug #2A) — genome gpt2 `fpt` is context-dependent.
- **Do not** multiply `train_flops` by a ratio — new seeds already log corrected `fpt`; **OVERWRITE** (ratio-multiply double-corrects → negative `fpt`).
- **Do not** analyze the **raw** `trajectory_results{,_genome}/` dirs — they are **mixed-`fpt`**; only `_corrected` is a valid axis.
- **Do not** cite `docs/rank_reversal.csv` or `crossover_scaling.png`/`rank_reversal*.png` without regenerating — they are **stale** (§8).
- **Do not** re-probe without `MODEL_FILTER` — CSVs are append-only, no dedup.
- **Do not** revive the synthetic "selectivity-law" as a centerpiece (Exp. S — pretraining-specific; five confounds; wrong-sign artifacts).
- **Do not** report the naive `P(reversal)=1.0`, the "20–25× scale-law", or fold/pfam reversals — all overturned.
- **Do not** reintroduce Cagnetta / γ_lang / β / α_D framing (pivoted 2026-04-28).
- **Do not** run genome training multi-GPU (Bug #3) or with `--time<02:00:00` (della QOS routing).
- **Do not** interpret a surprising result before auditing the instrument (the session's hard-won lesson).

---

## 15. Glossary

- **AR pair** — the two autoregressive archs: `gpt2` (attention), `mamba` (state-space). Margin = gpt2 − mamba.
- **`fpt`** — FLOPs per token; enters `C` linearly (Eq. 1). Scale-dependent buggy inflation (§6.1).
- **`C(t)`** — compute axis `= fpt·step`; `C_f = fpt·num_iterations`.
- **`Δ`/`delta`** — learned_test − random_init_baseline (learned structure).
- **`d_early`, `d_final`** — mean seed-paired margin at 1% budget / convergence.
- **Reversal** — `d_early<0 ∧ d_final>0`.
- **`C×/C_f`** — crossover: compute fraction below which mamba leads (FLOP-dependent; regenerate).
- **`P(reversal)`** — hierarchical posterior of the reversal indicator (Eq. 2); *not* the old bootstrap.
- **γ** — inductive-bias coefficient (Eq. 3); gpt2's learned-Δ advantage at matched loss & scale. **Not** Cagnetta γ_lang.
- **cfpt** — corrected FLOPs/token per `(arch,scale)`; per-domain (protein vs genome).
- **LOCO** — leave-one-cell-out.
- **B-ALLOC/FCAST/CAL** — selection / value-forecast / calibration (`forecast_protocol.py`).
- **B-GEN** — genome cross-domain replication. **A-BREADTH** — Paper A breadth concepts (pfam_family, ec_class, subcellular, fold).
- **mixed-`fpt`** — raw result dirs where per-seed `fpt` differs (buggy s0–2, corrected s3–9); invalid axis until OVERWRITTEN.
- **cell** — `(concept, arch, scale)` or `(arch, scale, seed)`.
- **`val_bpr`** — validation bits per residue (LM loss).
- **Scales XS/S/M/L** — sizes; §6 for params.
- **`release/` vs `release_traj/` vs `genome_traj/`** — v0.5 release ckpts / protein trajectory ckpts / genome trajectory ckpts.
- **`trajectory_results{,_genome}{,_corrected}/`** — probe result CSVs (raw mixed / corrected).

---

## 16. Minimal Context Prompt for a New Conversation

> You are a full professor at MIT continuing the **nanoprot two-paper program**. Read the attached `PROJECT_HANDOFF.md` (v2) in full before acting; it is the ground truth for goals, math (Eqs. 1–3, §3.10 assumptions), file/data/model inventories, experiments, bugs, and what has been overturned.
>
> **Working dir:** `/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot` (git `main`, HEAD `4b919fc`). Cluster: Princeton della, Slurm partition `ailab`. **Constraints:** no subagents / no autonomous loops — main-loop only, small reviewed steps; do not launch training/installs without explicit direction; Nature-Comms figure aesthetic (no bar plots, sans-serif 7–9pt, no rainbow, view the PNG before committing); trajectory CSVs are append-only — always guard re-probes with `MODEL_FILTER`; **always analyze the `_corrected` dirs, never the raw mixed-`fpt` dirs.**
>
> **State:** all GPU compute is done (queue empty). The corrected, seed-powered core is committed: protein reversal `ss3/M P=0.92`, `ss8/M P=0.75`; genome `frame/M P=0.93` (with `gc P=0.00` as a clean negative control); the forecaster and Paper A's γ are **proven FLOP-invariant**. Two FLOP-accounting bugs were fixed (§9). **Overturned — do not reuse:** fold/pfam reversal, the "20–25× scale-law", and the naive `P=1.0` bootstrap. **Stale — regenerate before citing:** `docs/rank_reversal{,_genome}.csv`, `rank_reversal{,_genome}.png`, `crossover_scaling.png`.
>
> **Your first task:** execute §13 in order — (1) regenerate `rank_reversal{,_genome}.csv` and the reversal/crossover figures from the `_corrected` dirs and record the corrected protein+genome crossover; then (2) rewrite `../brain/papers/paper_B.tex` around the corrected core (replace every `P(reversal)=1.0` and the genome crossover `0.20`, swap the stale figure at line ~176, drop fold/pfam from the reversal headline, re-lead the abstract on the three defensible legs). Verify by re-running `python -m scripts.make_corrected_trajectory` (gates must PASS) and `python -m scripts.hierarchical_reversal --protein-dir .cache/nanoprot/trajectory_results_corrected --genome-dir .cache/nanoprot/trajectory_results_genome_corrected` to reproduce the headline before writing. Honor §14. Goal: an ICLR spotlight (Paper B) and a cited Nature-family acceptance (Paper A).

---

*End of PROJECT_HANDOFF.md (v2) — generated 2026-07-06 at HEAD `4b919fc`. v2 corrects the FLOP-inflation figure (scale-dependent), the mixed-`fpt` status of the raw result dirs, the checkpoint schedule, and the stale-figure status; adds Exp. P (Pythia), §3.10 assumptions, training-hparam and esm2-`fpt` details. Values marked [INFERRED]/[UNKNOWN] are not verified — check before treating as fact.*
