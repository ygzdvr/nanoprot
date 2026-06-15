# Cross-architecture probing / eval harness — design plan

Status: **proposed, v2** (planning only — no code, no compute yet). v2 hardened after
review: §3 adds a baseline + val-selected best layer + contamination control, §4 notes
AR representations are predictive-not-descriptive, §5 pins the hook point + makes
per-layer streaming explicit (feasibility), §6 reframes the three sources as *not*
independent (all structure-derived). This is the shared infrastructure for two paper
pillars:

- **Pillar 3 — "which architecture wins which task?"** Run on the *existing* 36
  released final checkpoints. No retraining.
- **Pillar 2 — "*when* during training does a concept emerge?"** The same harness,
  re-run across training-trajectory checkpoints (needs the trajectory re-train,
  deferred). Pillar 2 is unlocked for free once Pillar 3's harness exists.

## 1. The scientific question

The release shows gpt2 *scales best at next-residue prediction*. That is an
intrinsic-loss claim. The probing harness asks the **downstream** question it does
not answer:

> Does the architecture that scales best at next-residue prediction also produce the
> most **biologically decodable** representations — or does the bidirectional MLM
> encoder (esm2) win on per-residue structure despite its separate pretraining
> objective and metric?

This is genuinely open and worth a section: it is exactly the "what is the right
architecture family for a given downstream task?" question the README leads with,
and it lets esm2 — which is *not* bpr-comparable — finally enter the comparison on a
level field (a shared downstream metric, not its own pretraining loss).

## 2. Probe targets (concepts)

Per-residue and per-protein labels from **Swiss-Prot** (reviewed UniProtKB, CC-BY).
Start with the standard PLM-probing trio (same tasks ESM/ProtTrans report), all
alignable to our unified 33-token residue alphabet 1:1:

| concept | granularity | classes | metric | source field |
|---|---|---|---|---|
| **secondary structure (SS3)** | per-residue | 3 (helix/strand/coil) | macro-F1 / acc | **three sources — see §6** |
| **functional site** | per-residue | 2 (site / not) | AUROC (imbalanced) | Swiss-Prot `ACT_SITE` + `BINDING` + `METAL` |
| **family / fold** | per-protein | top-K (e.g. 20) | acc / macro-F1 | Pfam / family cross-ref |

SS3 (Phase 1) is drawn from the three triangulated sources of §6; the per-residue
functional-site and per-protein family/fold concepts come from Swiss-Prot (CC-BY).
Phase 1 ships **SS3 only** (canonical, balanced, easiest to validate the pipeline);
sites + family follow once the pipeline is proven.

## 3. Probe methodology

- **Frozen model, trained probe.** The model weights are never updated — we train a
  probe *on top of* frozen residual-stream activations. This measures
  representation quality (linear decodability), not fine-tuning capacity.
- **Linear probe is the headline** (logistic regression / single `nn.Linear`).
  Optionally a 1-hidden-layer MLP as a "nonlinear decodability" companion, but linear
  is the standard and the cleanest cross-arch comparison. **Standardize** (z-score)
  features per layer before fitting — layers/archs differ in activation scale.
- **Probe every layer, but pick the best layer on VALIDATION.** The residual stream at
  each layer is a probe input; the *layer-wise* curve (where structure lives) is itself
  a finding. We headline the best layer per (arch, scale, concept) — **selected on the
  validation split and reported on the held-out test split**, never argmax-ed on test
  (sweeping ~all layers and reporting the test max is multiple-comparisons peeking).
- **Baseline is mandatory** — absolute probe scores are meaningless without a floor.
  For every (arch, scale) we run the identical probe on **(a) a random-initialized
  model of the same architecture** (untrained weights, same extraction) and **(b) a
  trivial baseline** (amino-acid identity / positional features). The reported quantity
  is **learned − baseline**: how much *structure the pretraining actually added*. (This
  is the Pythia-style control; without it "gpt2-L = 0.72 F1" is uninterpretable.)
- **Contamination control.** The probe proteins (e.g. CB513) are old and well-studied
  and almost certainly appear in UniRef50 pretraining, so representations may be partly
  *memorized* — possibly asymmetrically across archs. We filter the probe set against
  the pretraining corpus by sequence identity (e.g. drop probe proteins > ~50% identity
  to any training sequence) and, at minimum, report the overlap we could not remove.
- **Protein-level train/val/test split** (never split residues of the same protein
  across folds — that leaks). Fixed split, seeded, shared across all models so the
  comparison is controlled; for the published benchmark we use *its* canonical split.

## 4. The key confounder — encoder vs decoder context (read this first)

esm2 is **bidirectional**: residue *i*'s representation sees the whole protein. gpt2
and mamba are **causal**: residue *i* sees only the prefix ≤ *i*. For per-residue
structure (which depends on both neighbours), this hands esm2 an inherent advantage
that has nothing to do with "better representations."

This is a real, known asymmetry in encoder-vs-decoder probing. We do **not** try to
erase it; we **report it honestly and turn it into an axis of the study**:

- Compare **AR-vs-AR (gpt2 vs mamba)** as the clean, matched comparison (both causal,
  same metric) — this is the rigorous head-to-head.
- Report esm2 **separately**, explicitly flagged as bidirectional, as "the encoder
  reference." Any gpt2/mamba-vs-esm2 gap is discussed as *objective + directionality*,
  not crowned as an architecture win.
- Optionally probe AR models with a **whole-sequence forward** at probe time (they
  still attend only left — that is their nature — but it removes any train/eval
  context mismatch).
- **AR states are predictive, not descriptive.** The causal hidden state at residue *i*
  is optimized to *predict residue i+1*, not to *describe residue i*. So probing it for
  residue-*i* structure is slightly off-target by construction — another reason esm2
  (whose MLM objective makes states descriptive of the masked position) is a separate
  reference, not a like-for-like competitor. Worth a sentence in the paper.

Stating this up front is what keeps the cross-arch claim defensible.

## 5. Hidden-state extraction

Grounded fact: **no model exposes hidden states** (forwards return only loss/logits).
So extract via **PyTorch forward hooks** on each transformer block / Mamba block —
non-invasive, requires *no* change to model code, and works on the **released
checkpoints as-is** (load with `load_pretrained`, register hooks, run forward).

- **Pin the hook point — "residual stream" is not self-evident here.** gpt2 carries
  `resid_lambdas`, `x0_lambdas`, value embeddings, a smear gate and backout; mamba's
  block differs; esm2 is pre-LN. We hook **one consistent, documented point: the
  residual-stream tensor entering block *k*** (= block *k−1*'s output before any final
  norm), for *k = 0 … n_layers*, giving *n_layers + 1* probe points per model. Whatever
  point we choose, it must be the *same definition* across archs or the layer-wise
  comparison is not apples-to-apples. The choice is encoded once in `extract.py`.
- **Cross-arch alignment:** archs differ in depth and width (gpt2/mamba `d=64·depth`,
  esm2 `d=40·depth`), so compare by **relative depth** (fraction through the network),
  not absolute layer index. Token alignment is exact because every arch shares the
  33-token alphabet (a deliberate v0.5 choice) — so per-residue labels line up 1:1.
- **Per-layer streaming is mandatory, not optional — caching all layers is infeasible.**
  At L scale one model's full activations are hundreds of GB (~10⁶ residues × d_model ×
  n_layers). So the contract is: **forward once, and for each layer fit (or
  incrementally update) that layer's probe, then free it** — never hold all layers at
  once. Size the probe set so a *single* layer's activations fit in RAM (a few-thousand
  proteins × one layer ≈ a few GB), and use incremental / SGD probe fitting if even one
  layer is tight. eval mode, bf16-or-fp32 per device, proteins batched by length.

## 6. Data pipeline — three SS3 label sources, triangulated (decided 2026-06-15)

SS3 labels come from **three sources**. Up-front honest caveat: these are **not three
independent label *definitions*** — *all* SS3 is ultimately DSSP-on-a-structure. What
actually varies is **(a) the protein set** and **(b) experimental vs predicted
structure**. So the triangulation tests robustness to *dataset* and to *structure
source*, not to the definition of "helix"; we word the claim that way. nanoprot ships
its own prep (standalone public repo; the thesis pickles are gone and live elsewhere):

1. **Published benchmark** (NetSurfP-2.0 train + CB513 test) — standard sequences and a
   *canonical* split; numbers directly comparable to ESM-2 / prior PLM work on the same
   test set. Labels are DSSP on *experimental* PDB structures.
2. **Custom Swiss-Prot subset** — pinned Swiss-Prot release (provenance like
   `prepare_uniref50.py`), `HELIX`/`STRAND`/`TURN` feature annotations → SS3,
   protein-level split, fully in-repo. Caveat: these annotations are **sparse** (only
   structurally-characterized residues) and **experimental-structure-derived**, so the
   set skews to well-studied proteins and is *not* independent of source 3.
3. **DSSP from AlphaFold** — download AF PDBs for the probe proteins, run **`mkdssp`**
   (external binary, cluster install), reduce DSSP-8 → SS3. *Predicted*-structure
   labels; **filter low-confidence residues by pLDDT** (don't trust SS where AF is
   unsure).

Two agreement analyses fall out:

- **Conclusion-level (primary):** is the architecture ordering / scaling trend /
  best-layer the *same* across the three datasets? Replication across protein sets and
  across experimental-vs-predicted structure makes the finding **robust to dataset and
  structure source** — the accurate claim, *not* "source-independent."
- **Label-level (sanity check):** on proteins covered by *both* experimental
  (Swiss-Prot/PDB) and predicted (AF) structure, per-residue SS3 agreement is largely a
  measure of **AlphaFold-vs-experimental concordance** (already known to be high in
  confident regions) — a pipeline sanity check, not "our labels are clean."

`scripts/prepare_probe_data.py --source {netsurfp,swissprot,dssp}` writes each cached
label set + split under `$NANOPROT_BASE_DIR/probes/ss3_{source}/`. Keep each modest
first (a few thousand proteins). **Build order: stand up the pipeline on ONE source
first (the published benchmark — cleanest download), prove extraction + probe
end-to-end, then add the other two and run the triangulation.**

## 7. Metrics & outputs

- **Results CSV:** one row per `(arch, scale, seed, concept, layer, source)` → val
  metric, test metric, **and the matched random-init baseline metric**. The headline
  aggregation per `(arch, scale, concept, source)` is the **best layer selected on val,
  reported on test, as `learned − baseline`**, mean±std over seeds.
- **Figures (Nature-aesthetic, no bar plots):**
  - probe score (`learned − baseline`) **vs scale**, per arch & concept — does gpt2's
    scaling edge transfer to biology?
  - **layer-wise** curve (score vs relative depth) per arch — where structure lives.
  - **triangulation panel:** the arch ordering / scaling across the three sources, side
    by side — does the conclusion replicate?
- Drops straight into `RESULTS.md` as the Pillar-3 section.

## 8. Module layout (proposed)

```
nanoprot/eval/probe/
  extract.py     # forward-hook residual-stream extraction at the pinned point (§5),
                 #   all 3 archs; also serves random-init baselines (untrained weights)
  labels.py      # load cached SS3 labels per source + token alignment + splits;
                 #   trivial baseline features (AA identity / position)
  linear.py      # standardized linear/MLP probe, incremental per-layer fit, val-select
scripts/
  prepare_probe_data.py   # --source {netsurfp,swissprot,dssp} -> cached labels + provenance
  run_probes.py           # (arch,scale,seed) x layer x source -> results CSV (+ baselines)
  probe_agreement.py      # cross-source conclusion + label-agreement analysis (§6)
  plot_probes.py          # the figures
tests/
  test_probe_extract.py   # pinned hook point + shapes, per-arch, relative-depth map
  test_probe_linear.py    # standardized fit/eval + val-selected best layer + baseline
  test_prepare_probe_data.py
```

## 9. Phasing

1. **Phase 1 (no *training* compute; forward passes only):** extraction hooks + SS3
   labels (one source first) + standardized linear probe **+ random-init baseline**, on
   the 36 *final* checkpoints, all layers, best-layer-on-val → first cross-arch probe
   result + the AR-vs-AR head-to-head. **This is the Pillar-3 MVP.**
2. **Phase 2:** add functional-site + family concepts; the MLP companion probe.
3. **Phase 3 (after the trajectory re-train):** re-run across checkpoints → the
   "*when* does the helix feature emerge, across scale × arch" developmental result =
   **Pillar 2**.

## 10. Compute & risks

- **Compute:** no *training* — forward passes (incl. random-init baselines) over a
  few-thousand-protein probe set × 36 models, **one layer streamed at a time** (§5).
  Modest GPU; linear probes are CPU-cheap.
- **Hardened in v2:** baseline (§3), val-selected best layer (§3), contamination filter
  (§3), per-layer streaming (§5), pinned hook point (§5), three-source independence
  reframe (§6).
- **Remaining risks:**
  - **Encoder/decoder asymmetry** (§4) — the central scientific care-point; handled by
    making AR-vs-AR the head-to-head and esm2 a separate reference.
  - **`mkdssp` + pLDDT** for the AF source (§6) — external binary + a confidence
    threshold to choose.
  - **Swiss-Prot SS sparsity** (§6) — source 2 is small/biased; fine as one of three,
    not as the sole source.
  - **Probe-set size vs RAM** — the per-layer streaming bound (§5) sets the max probe-set
    size at L scale; pick it from the largest model's one-layer footprint.
  - **Self-containment** — re-implement Swiss-Prot parsing cleanly in nanoprot, not
    imported from the thesis.

## 11. Decisions (resolved 2026-06-15)

1. **Concept set for Phase 1:** SS3 only — start narrow; sites + family in Phase 2.
2. **SS3 label source:** **all three** — published benchmark + custom Swiss-Prot +
   DSSP-from-AlphaFold — **triangulated** (§6). Build on the published benchmark first,
   then add the other two and run the agreement analysis.
3. **esm2 handling:** reported **separately as the "encoder reference"** (never crowned
   on a cross-objective gap; the rigorous head-to-head is gpt2-vs-mamba, §4).
4. **Probe set:** each source brings its own set + split — the published benchmark for
   prior-work comparability, Swiss-Prot for in-repo reproducibility, DSSP for
   structural ground truth.
