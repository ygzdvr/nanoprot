# Cross-architecture probing / eval harness — design plan

Status: **proposed** (planning only — no code, no compute yet). This is the shared
infrastructure for two paper pillars:

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
| **secondary structure (SS3)** | per-residue | 3 (helix/strand/coil) | macro-F1 / acc | DSSP via structure or `Helix`/`Strand`/`Turn` features |
| **functional site** | per-residue | 2 (site / not) | AUROC (imbalanced) | `ACT_SITE` + `BINDING` + `METAL` |
| **family / fold** | per-protein | top-K (e.g. 20) | acc / macro-F1 | Pfam / family cross-ref |

Phase 1 ships **SS3 only** (canonical, balanced, easiest to validate the pipeline);
sites + family follow once the pipeline is proven.

## 3. Probe methodology

- **Frozen model, trained probe.** The model weights are never updated — we train a
  probe *on top of* frozen residual-stream activations. This measures
  representation quality (linear decodability), not fine-tuning capacity.
- **Linear probe is the headline** (logistic regression / single `nn.Linear`).
  Optionally a 1-hidden-layer MLP as a "nonlinear decodability" companion, but linear
  is the standard and the cleanest cross-arch comparison.
- **Probe every layer.** The residual stream after each block is a probe input;
  reporting the *layer-wise* curve is itself a finding ("where in the network does
  structure live"), and we headline the **best layer** per (arch, scale, concept).
- **Protein-level train/val/test split** (never split residues of the same protein
  across folds — that leaks). Fixed split, seeded, shared across all models so the
  comparison is controlled.

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

Stating this up front is what keeps the cross-arch claim defensible.

## 5. Hidden-state extraction

Grounded fact: **no model exposes hidden states** (forwards return only loss/logits).
So extract via **PyTorch forward hooks** on each transformer block / Mamba block —
non-invasive, requires *no* change to model code, and works on the **released
checkpoints as-is** (load with `load_pretrained`, register hooks, run forward).

- Hook each layer module's output → capture the per-residue residual stream
  `(batch, seq, d_model)` for every layer.
- **Cross-arch alignment:** archs differ in depth and width (gpt2/mamba `d=64·depth`,
  esm2 `d=40·depth`), so compare by **relative depth** (fraction through the network),
  not absolute layer index. Token alignment is exact because every arch shares the
  33-token alphabet (a deliberate v0.5 choice) — so per-residue labels line up 1:1.
- Run in eval / bf16-or-fp32 per device; batch proteins by length; cache activations
  to disk only if needed (prefer streaming: extract → probe-fit incrementally → free).

## 6. Data pipeline — three SS3 label sources, triangulated (decided 2026-06-15)

SS3 labels come from **all three sources**, and we **check whether the cross-arch
conclusion is consistent across them** (robustness) and, where proteins overlap,
whether the *labels themselves* agree (validation). nanoprot ships its own prep (it is
a standalone public repo; the thesis pickles are gone and live elsewhere):

1. **Published benchmark** (NetSurfP-2.0 / CB513) — standard sequences + a *fixed*
   train/test split; gives numbers directly comparable to ESM-2 / prior PLM work on
   the same test set.
2. **Custom Swiss-Prot subset** — pinned Swiss-Prot release (provenance recorded like
   `prepare_uniref50.py`), helix/strand/turn feature annotations → SS3, protein-level
   split. Fully in-repo and reproducible.
3. **DSSP from AlphaFold** — download AF PDBs for the probe proteins, run DSSP
   (`mkdssp`), reduce DSSP-8 → SS3. Gold-standard *structural* labels.

Two agreement analyses fall out — this is the point of doing all three:

- **Conclusion-level (primary):** is the architecture ordering / scaling trend /
  best-layer the *same* across all three datasets? If gpt2-vs-mamba replicates on
  published **and** Swiss-Prot **and** DSSP, the finding is source-independent — a far
  stronger claim than any single dataset.
- **Label-level (validation):** on proteins covered by *both* Swiss-Prot annotations
  and DSSP-from-AF, how often do the per-residue SS3 labels agree? Quantifies how noisy
  annotation-based labels are vs structural ground truth.

`scripts/prepare_probe_data.py --source {netsurfp,swissprot,dssp}` writes each cached
label set + split under `$NANOPROT_BASE_DIR/probes/ss3_{source}/`. Keep each modest
first (a few thousand proteins). **Build order: stand up the pipeline on ONE source
first (the published benchmark — cleanest download), prove extraction + probe
end-to-end, then add the other two and run the triangulation.**

## 7. Metrics & outputs

- **Results CSV:** one row per `(arch, scale, seed, concept, layer)` → metric, plus a
  best-layer aggregation per `(arch, scale, concept)` with mean±std over seeds.
- **Figures (Nature-aesthetic, no bar plots):**
  - probe metric **vs scale**, per arch & concept — does gpt2's scaling edge transfer
    to biology?
  - **layer-wise** probe curve (metric vs relative depth) per arch — where concepts live.
- Drops straight into `RESULTS.md` as the Pillar-3 section.

## 8. Module layout (proposed)

```
nanoprot/eval/probe/
  extract.py     # forward-hook per-layer activation extraction (all 3 archs)
  labels.py      # load cached Swiss-Prot labels + token alignment + splits
  linear.py      # linear / MLP probe fit + eval, layer sweep
scripts/
  prepare_probe_data.py   # Swiss-Prot -> cached labels (+ provenance)
  run_probes.py           # (arch,scale,seed) x concept -> results CSV
  plot_probes.py          # the figures
tests/
  test_probe_extract.py   # hook extraction shapes, per-arch, relative-depth map
  test_probe_linear.py    # probe fit/eval on synthetic separable data
  test_prepare_probe_data.py
```

## 9. Phasing

1. **Phase 1 (no compute beyond forward passes):** extraction hooks + SS3 labels +
   linear probe, on the 36 *final* checkpoints, all layers → first cross-arch probe
   result + the AR-vs-AR head-to-head. **This is the Pillar-3 MVP.**
2. **Phase 2:** add functional-site + family concepts; the MLP companion probe.
3. **Phase 3 (after the trajectory re-train):** re-run across checkpoints → the
   "*when* does the helix feature emerge, across scale × arch" developmental result =
   **Pillar 2**.

## 10. Compute & risks

- **Compute:** no training — forward passes over a few-thousand-protein probe set ×
  36 models × all layers. Modest GPU time; the linear probes are CPU-cheap.
- **Risks / decisions to settle:**
  - **Encoder/decoder asymmetry** (§4) — the central scientific care-point.
  - **Label quality / coverage** — Swiss-Prot SS coverage varies; may lean on DSSP
    from AlphaFold structures for SS3 if feature annotations are sparse.
  - **Probe overfitting** — fixed protein splits, report test not val, regularize.
  - **Self-containment** — re-implement Swiss-Prot parsing cleanly in nanoprot rather
    than importing the thesis code.

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
