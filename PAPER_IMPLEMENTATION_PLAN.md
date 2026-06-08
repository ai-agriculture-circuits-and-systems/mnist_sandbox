# Paper Implementation Plan (APCCAS)

This document is the **single source of truth** for running experiments and producing paper artifacts from this repository. It is written for **human authors and automated agents** (Cursor, CI agents, etc.) who need to execute the plan end-to-end without re-deriving context from chat history.

**Last updated:** 2026-05-19  
**Target venue:** IEEE APCCAS (4 pages content + 1 page references)  
**Paper framing:** Automated image **classification** framework — user provides a dataset in a standard layout; the system trains many architectures under a fixed protocol and outputs a **ranked list** of models (top-K). Case studies: **four registered datasets** (1 dev + 3 agriculture).

---

## 1. Goals and non-goals

### Goals

1. Validate the regression pipeline on **all four datasets** (quick test first, then full runs).
2. Produce reproducible artifacts: `results.json`, `REPORT_*.md`, paper **Table I**, **Fig. 2–3**.
3. Support APCCAS narrative: **system methodology** + **accuracy/cost trade-offs**, not only raw accuracy.

### Non-goals (for the main paper)

- Autoencoder and GAN benchmarks (skip unless paper scope changes).
- Publishing full top-50 tables in the 4-page main text (use appendix / supplementary / repo).
- Exhaustive grid search over all hyperparameters (the suite uses **NAS-style random sampling**).

---

## 2. Repository context (read this first)

### 2.1 What “regression” means here

**Not** statistical linear regression. In this repo, **regression** = **regression testing / benchmark sweep**:

- Train many models under the **same protocol** (splits, early stopping, search space).
- Rank by test metric; write `REPORT.md` and `outputs/regression/.../results.json`.

Implemented in:

- `utils/regression.py` — core suite
- `scripts/regression.sh` — CLI wrapper

### 2.2 Datasets (registry)

Defined in `utils/dataset_config.py` → `DATASETS` (4 canonical keys):

| Registry key | Classes | Task | Default `data_root` |
|--------------|--------:|------|---------------------|
| `mnist` | 10 | Digit classification (dev/sanity) | `data/` (`.mat` files) |
| `strawberry` | 6 | Ripeness stages | `data/Strawberry/strawberries` |
| `plant_village_raspberry` | 2 | Healthy vs background | `data/Plant_Village_Raspberry/raspberries` |
| `plant_village_orange` | 2 | Citrus greening vs background | `data/Plant_Village_Orange/oranges` |

**Aliases:** `raspberry` → `plant_village_raspberry`, `orange` → `plant_village_orange`.

**Git submodules:** Strawberry, Plant_Village_Raspberry, Plant_Village_Orange under `data/`. Initialize with `./scripts/init_submodules.sh`.

### 2.3 Models

- **~50 classification models:** `CLASSIFICATION_MODELS` in `models/model_factory.py`
- **4 autoencoders + 4 GANs:** excluded from paper classification runs unless requested
- Default `ModelFactory.create_model(..., num_classes=10)`; regression passes `num_classes` from `DatasetSpec`.

### 2.4 Search protocol (classification)

Per model, the suite runs **every** valid **loss × optimizer** pair, and for each pair runs **`nas_trials`** random hyperparameter samples (keeps the best).

| Dimension | Values |
|-----------|--------|
| Losses | `cross_entropy`, `label_smoothing`, `focal_loss` |
| Optimizers | `adam`, `sgd`, `adamw`, `rmsprop` |
| Hyperparameters (sampled) | `lr` ∈ {1e-4, 1e-3, 1e-2}, `batch_size` ∈ {8, 16, 32}, `weight_decay` ∈ {0, 1e-4, 1e-3} |

**Quick-test mode** (`-q` / `--quick-test`):

- Caps `nas_trials` to **2**, `max_epochs` to **10** (often reports 1 epoch in past runs).
- Agriculture: subsamples **~60 images** per split via `quick_max_samples` in `utils/data_loader.py`.
- MNIST quick: uses `data/test_data/test_images.npy` (create with `python3 utils/create_test_data.py` if missing).

**Full mode** (`-f` / `--full`):

- Uses full splits; `nas_trials` default **20**; `max_epochs` default **200** with early stopping (`patience` default 5).

### 2.5 Parallel workers and GPUs

- `scripts/regression.sh` default **`WORKERS=8`**.
- `utils/regression.py` caps workers at **GPU count** (`available_parallel_slots`).
- **Recommendation for this lab:** **10× Tesla V100-SXM2-32GB** → use **`-j 10`** to use all GPUs.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in `regression.sh`).

### 2.6 Outputs (do not overwrite across datasets)

Always set **per-dataset** paths:

| Artifact | Recommended path |
|----------|------------------|
| Markdown report | `reports/REPORT_<dataset>_quick.md` or `reports/REPORT_<dataset>_full.md` |
| JSON results | `outputs/regression/<dataset>_quick/results.json` |
| Partial worker outputs | `outputs/regression/<dataset>_quick/partials/` |

The default `REPORT.md` at repo root is **overwritten** each run — avoid for multi-dataset work.

### 2.7 Metrics currently logged (`TrialResult`)

From `utils/regression.py`:

| Field | Classification meaning |
|-------|------------------------|
| `metric_name` | `test_accuracy` |
| `metric_value` | Test accuracy (%) |
| `test_loss` | Test loss |
| `epochs_run` | Epochs executed |
| `epochs_to_convergence` | Best epoch (early stopping) |
| `hyperparameters` | `lr`, `batch_size`, `weight_decay` |
| `loss`, `optimizer` | Best combo for that (model, loss, opt) block |

**Not yet in regression output (paper gaps — see Phase 3):**

- Macro-F1, per-class recall
- Training wall time per trial
- Parameter count, inference latency

Confusion matrices: `utils/evaluator.py` + `main.py --plot-confusion` (manual step for winners).

---

## 3. Paper figures and table (target layout)

Agents generating plots should match this layout (4-page APCCAS limit).

| Asset | Type | Content |
|-------|------|---------|
| **Fig. 1** | Block diagram | System architecture & methodology only (no accuracy numbers) |
| **Fig. 2** | 1×3 horizontal bar charts | Top-5 test accuracy per **agriculture** dataset |
| **Fig. 3** | Scatter | Test accuracy vs `epochs_to_convergence`; color = dataset; one point per model (best per model) |
| **Table I** | Table | Top-5 per ag dataset: Acc, Macro-F1, Epochs, Loss, Optimizer, key hyperparams |

**MNIST:** Pipeline validation; mention in text or appendix — do not consume main figure space unless required.

**Full top-50:** Supplementary material or repository link, not main paper.

---

## 4. Implementation phases

### Phase 0 — Preflight

**Objective:** Environment and data ready.

```bash
cd /path/to/mnist_sandbox
./scripts/init_submodules.sh    # if agriculture submodules missing
pytest tests/ -q
nvidia-smi
mkdir -p reports outputs/regression
```

**Checks:**

| Check | Command / path |
|-------|----------------|
| MNIST mats | `data/MNISTtrain.mat`, `data/MNISTtest.mat` |
| MNIST quick subset | `data/test_data/test_images.npy` |
| Strawberry splits | `data/Strawberry/strawberries/<class>/sets/train.txt` (6 classes) |
| Raspberry splits | `.../healthy/color/sets/train.txt`, `.../background_without_leaves/without_augmentation/sets/train.txt` |
| Orange splits | `.../huanglongbing_citrus_greening/color/sets/train.txt`, `.../background_without_leaves/without_augmentation/sets/train.txt` |
| Unit tests | `pytest tests/ -q` → all pass |

**Exit criteria:** All paths exist; `pytest` green; GPUs show sufficient free memory.

---

### Phase 1 — Smoke test (3 models × 4 datasets)

**Objective:** Catch dataset, shape, and training bugs cheaply (~30–60 minutes total on 10× V100).

```bash
SMOKE="lenet,resnet,mobilenetv2"

for ds in mnist strawberry plant_village_raspberry plant_village_orange; do
  ./scripts/regression.sh --dataset "$ds" -q -j 10 \
    -m "$SMOKE" \
    -r "reports/smoke_${ds}.md" \
    -o "outputs/regression/smoke_${ds}"
done
```

**Inspect failures:**

```bash
python3 <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
bad = [r for r in data if r.get("status") != "ok"]
print(f"{len(bad)} failed / {len(data)} total")
for r in bad[:15]:
    print(r["model"], r.get("error", "")[:200])
PY
outputs/regression/smoke_strawberry/results.json
```

**Exit criteria:** All four smoke runs complete; majority of trials `status: ok`; no systematic failure (e.g. all models OOM on one dataset).

**Common fixes:**

- Missing submodule → `init_submodules.sh`
- MNIST quick npy missing → `python3 utils/create_test_data.py`
- OOM → note model names; full run may need `-m` subset or lower `--max-batch-size`

---

### Phase 2 — Quick regression (all classifiers × 4 datasets)

**Objective:** Full **50-model** screen per dataset; shortlist for full runs; draft Fig. 2 data.

```bash
CLASSIFIERS=$(python3 -c "from models.model_factory import CLASSIFICATION_MODELS; print(','.join(CLASSIFICATION_MODELS))")

for ds in mnist strawberry plant_village_raspberry plant_village_orange; do
  ./scripts/regression.sh --dataset "$ds" -q -j 10 \
    -m "$CLASSIFIERS" \
    -n 2 \
    -r "reports/REPORT_${ds}_quick.md" \
    -o "outputs/regression/${ds}_quick"
done
```

**Estimated wall time (10× V100, `-j 10`):**

| Dataset | Approx. time |
|---------|----------------|
| MNIST | 1–2 h |
| Strawberry | ~1.5–2 h |
| Raspberry | ~1–1.5 h |
| Orange | ~1–1.5 h |
| **Total** | **~5–8 h** |

**Exit criteria:**

- [ ] Four files: `reports/REPORT_<ds>_quick.md`
- [ ] Four files: `outputs/regression/<ds>_quick/results.json`
- [ ] Classification section has ~600 rows per dataset (50 models × 12 loss/optimizer combos)
- [ ] Rankings plausible (not all 0%, not all identical)
- [ ] Failure rate acceptable (investigate if >20% trials failed)

**Agent action after Phase 2:** Read each report’s “best per model” table; record **top 10–15 models per dataset** for Phase 4.

---

### Phase 3 — Paper tooling (code gaps)

**Objective:** Turn `results.json` into Table I and Fig. 2–3; add macro-F1 for reviewers.

| Priority | Task | Files | Status |
|----------|------|-------|--------|
| **P0** | Export top-5 LaTeX/CSV per dataset | `scripts/export_paper_table.py` (to create) | Not implemented |
| **P0** | Plot Fig. 2 + Fig. 3 from JSON | `scripts/plot_paper_figures.py` (to create) | Not implemented |
| **P0** | Wrapper: smoke + quick all datasets | `scripts/run_paper_smoke.sh`, `scripts/run_paper_quick_all.sh` (to create) | Not implemented |
| **P1** | Macro-F1 in evaluation + regression | `utils/evaluator.py`, `utils/regression.py` | Not implemented |
| **P1** | Params + inference ms for top-5 | small script | Not implemented |
| **P2** | Confusion matrix for winners | `python3 -m utils.main --plot-confusion` | Exists |
| **P3** | Remove duplicate `GANTrainer` in `utils/trainer.py` | cleanup | Optional |

**Agent implementing P0 scripts should:**

1. Parse `outputs/regression/<dataset>_*/results.json`.
2. Filter `metric_name == "test_accuracy"` and `status == "ok"`.
3. Apply `best_per_model` logic (max accuracy per `model`) — mirror `utils/regression.py:best_per_model`.
4. Export top-5 per dataset for Table I; plot top-5 bars (Fig. 2) and accuracy vs `epochs_to_convergence` (Fig. 3).

Use matplotlib skill / repo style if `skills/matplotlib-plot-style` applies.

---

### Phase 4 — Full regression (paper numbers)

**Objective:** Final accuracy and hyperparameters on **full data** with **NAS trials = 20**.

**Do not** run all 50 models × 4 datasets × full search without Phase 2 shortlist (weeks of GPU).

#### 4A — Build shortlist

From each `reports/REPORT_<ds>_quick.md`, pick **top 10–15** models by “Classification — best per model” rank.

Example default shortlist (adjust per dataset after Phase 2):

```
resnet,efficientnet,mobilenetv2,lenet,densenet,convnext,vit,swin_tiny,alexnet,simple_cnn
```

#### 4B — Full runs (agriculture, classification only)

```bash
SHORTLIST="resnet,efficientnet,mobilenetv2,lenet,densenet,convnext,vit,swin_tiny,alexnet,simple_cnn"

for ds in strawberry plant_village_raspberry plant_village_orange; do
  ./scripts/regression.sh --dataset "$ds" -f -j 10 -n 20 \
    -m "$SHORTLIST" \
    -r "reports/REPORT_${ds}_full.md" \
    -o "outputs/regression/${ds}_full"
done
```

**Optional MNIST full** (sanity / appendix):

```bash
./scripts/regression.sh --dataset mnist -f -j 10 -n 20 \
  -m "$SHORTLIST" \
  -r reports/REPORT_mnist_full.md \
  -o outputs/regression/mnist_full
```

**Estimated wall time (10× V100, ~10 models, 20 NAS, full data):**

| Scope | Approx. time |
|-------|----------------|
| One ag dataset | 1–2 days |
| Three ag datasets | 3–6 days |
| + MNIST | +0.5–1 day |

#### 4C — Optional exhaustive full (supplementary only)

All `CLASSIFICATION_MODELS` on all datasets with `-f -n 20`: **~2–3 weeks** sequential on one 10-GPU node. Only if required for supplementary ranking.

**Exit criteria:**

- [ ] `reports/REPORT_<ds>_full.md` for three ag datasets (and MNIST if run)
- [ ] `outputs/regression/<ds>_full/results.json` complete
- [ ] Table I and Fig. 2–3 regenerated from **full** JSON

---

### Phase 5 — Paper assets (figures & tables)

| Deliverable | Source |
|-------------|--------|
| Fig. 1 | Draw from `ARCHITECTURE.md` (TikZ, draw.io, PowerPoint) |
| Fig. 2 | `plot_paper_figures.py` → top-5 bars, 3 panels |
| Fig. 3 | `plot_paper_figures.py` → accuracy vs epochs scatter |
| Table I | `export_paper_table.py` → LaTeX or CSV |
| Confusion matrices | `main.py` for each dataset’s winner |

**Winner evaluation example:**

```bash
python3 -m utils.main --dataset plant_village_orange -f \
  --model <WINNER_FROM_REPORT> \
  --loss <LOSS> --optimizer <OPT> \
  --lr <LR> --batch-size <BS> \
  --plot-confusion --epochs <EPOCHS_FROM_REPORT>
```

(Confirm `main.py` exposes the same hyperparameter flags as the winning trial.)

---

## 5. Go / no-go checklist

Before starting **Phase 4** (full runs):

- [ ] Phase 1 smoke: 4/4 datasets complete
- [ ] Phase 2 quick: 4/4 datasets complete
- [ ] `results.json` valid JSON; export/plot scripts run without error
- [ ] Failure rate investigated; blocking bugs fixed
- [ ] Shortlist documented (per dataset) in this file or `reports/shortlist.txt`
- [ ] GPUs free (`nvidia-smi`); `reports/` and `outputs/regression/` paths unique per run

Before **submitting paper**:

- [ ] Table I populated from **full** runs (ag datasets)
- [ ] Fig. 2–3 match final JSON
- [ ] Text states: dataset sizes, class counts, NAS trials, early-stop settings, hardware
- [ ] Reproducibility: seed (`--seed 42` default), commands in appendix

---

## 6. Timeline (suggested)

| Week | Activities |
|------|------------|
| 1 | Phase 0–2; fix bugs; implement P0 scripts |
| 2–3 | Phase 4 full runs (3 ag datasets, shortlist) |
| 3 | Phase 5 figures/table; draft Results section |
| 4 | Buffer: re-runs, macro-F1, MNIST full, confusion matrices |

---

## 7. Hardware reference (this lab)

- **GPUs:** 10× NVIDIA Tesla V100-SXM2-32GB
- **Use:** `./scripts/regression.sh ... -j 10`
- **Note:** Default script uses 8 workers → 2 GPUs idle unless `-j 10`

**Rough scaling from measured quick strawberry run (~5,875 s, 8 workers, 1 epoch, nas=2, quick data):**

- Full + 20 NAS + full data + 10 workers: order of **days per dataset** for 50 models; **shorter** for 10-model shortlist.

---

## 8. Agent instructions (summary)

When assigned to “execute the paper plan”:

1. Read this file and `ARCHITECTURE.md`.
2. Run **Phase 0 → 1 → 2** in order; do not skip smoke.
3. Never use bare `REPORT.md` for multi-dataset work — use `reports/REPORT_<dataset>_<mode>.md`.
4. Use **`CLASSIFICATION_MODELS` only** unless paper scope includes generative models.
5. After Phase 2, write shortlist to `reports/shortlist.txt` (create if missing).
6. Implement or run **Phase 3** scripts before Phase 4 if tables/figures are required.
7. Phase 4: full runs only for shortlist on agriculture datasets.
8. Report back: paths to JSON/MD, failure counts, top-1 model per dataset, blockers.

**Do not:**

- Commit secrets or `.pth` blobs to git unless repo policy allows.
- Run `utils/test.py` as the primary validator (legacy; use `pytest tests/` + regression smoke).
- Assume `test_accuracy` alone is sufficient for the paper — implement or note macro-F1 gap.

---

## 9. Related files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | System design and dataset layout |
| `utils/dataset_config.py` | Dataset registry |
| `utils/regression.py` | Regression / NAS suite |
| `scripts/regression.sh` | Shell entry point |
| `models/model_factory.py` | `CLASSIFICATION_MODELS`, `MODEL_REGISTRY` |
| `REPORT.md` | Example report (overwritten by default runs) |

---

## 10. Changelog

| Date | Change |
|------|--------|
| 2026-05-19 | Initial plan: APCCAS 4-page layout, 4 datasets, phased smoke → quick → full, agent section |
