# PlantSetDelta


PlantSetDelta (psd) is a command-line toolkit for identifying regulatory sequence features that distinguish **any two (or multiple) gene sets** across plant species. It builds a feature matrix from regulatory regions around **TSS** and **TTS**, trains classical ML models via **PyCaret**, and exports interpretable **top features**.

Current release: **v0.2.0**.

This repository provides:
- **k-mer** features (fast C++ backend; optional Python fallback)
- **TF-binding** features:
  - For *Arabidopsis thaliana* (**ath**): BED-peak–based TF-bin features (strict binning) and an additional **sliding-window** version (precomputed)
  - For other species: TF-bin features predicted by a DeeperDeepSEA model trained on Arabidopsis TF targets

> Design principle: `psd build` is **offline**. All required models/resources must be downloaded in advance using `psd download` (on a login node with internet), or supplied explicitly via paths.

---

<div align="center">
  <img src="./images/plantsetdelta_fig.png" width="80%" />
</div>

---

## Contents

- [Installation](#installation)
- [Data download and offline cache](#data-download-and-offline-cache)
- [Concepts: regulatory windows, bins, TF modes](#concepts-regulatory-windows-bins-tf-modes)
- [Quick start](#quick-start)
- [Build: generate feature matrix](#build-generate-feature-matrix)
- [Train: AutoML with PyCaret](#train-automl-with-pycaret)
- [Top: interpret model and export top features](#top-interpret-model-and-export-top-features)
- [Input formats and requirements](#input-formats-and-requirements)
- [Outputs](#outputs)
- [Advanced recipes](#advanced-recipes)
- [Troubleshooting](#troubleshooting)

---

## Installation

### 1) Recommended environment

- Python **3.9–3.10**
- Linux recommended (C++ k-mer accelerator compiles automatically on first run)
- Conda is strongly recommended for a clean environment

```bash
conda create -n psd python=3.10 -y
conda activate psd
```

### 2) Install PlantSetDelta

From pip:
```bash
pip install plantsetdelta
```

From source:

```bash
git clone https://github.com/bwang889/plantsetdelta.git
cd plantsetdelta
pip install .
```
### 3) Troubleshooting Installation

Due to the complex dependency stack in the Python ecosystem, installation conflicts may occasionally occur in some local Linux/Conda environments. Below are solutions to the most common issues:

#### Issue 1: `pkg_resources` or `selene_sdk` error during installation
This is typically caused by recent updates in the Python `setuptools` ecosystem that break compatibility with older packages like `selene_sdk`. 
* **Solution:** Downgrade `setuptools` to a compatible version before installing PlantSetDelta:

```bash
pip install "setuptools<60.0"
```

#### Issue 2: `libstdc++.so.6: version 'GLIBCXX_X.X.XX' not found` error
This is a known Conda/Linux bug where the environment gets confused and tries to use the older system-level C++ compiler instead of the new one provided by Conda upon installation.
* **Solution:** Force Conda to use its own modern C++ standard library by installing `libstdcxx-ng`:

```bash
conda install -c conda-forge libstdcxx-ng
```
*(Alternatively, you can manually export your conda environment's library path to `LD_LIBRARY_PATH` prior to execution).*

#### The Ultimate Fallback: Preconfigured Conda Environment
If you still encounter stubborn dependency/version conflicts or fail to install PlantSetDelta successfully, we provide a fully preconfigured Conda environment as a fallback. 

A packaged environment is available at: https://zenodo.org/records/19183728

After downloading the environment archive (`psd_env.tar.gz`), you can unpack and activate it on compatible Linux systems as follows:

```bash
mkdir -p psd_env
tar -xzf psd_env.tar.gz -C psd_env
source psd_env/bin/activate
conda-unpack
```

### 4) External dependency (important)

**bedtools** is required if you use Arabidopsis strict TF features computed from BED peaks (or any workflow relying on `pybedtools` intersection).

Install via conda:

```bash
conda install -c bioconda bedtools -y
```

---

## Data download and offline cache

PlantSetDelta uses a local cache directory for large files (parquet features, deep models, TF BED peaks). You control the cache directory via:

- Environment variable: `PSD_DATA_DIR`

If not set, the package uses its internal `plantsetdelta/data` directory (not recommended for cluster usage).

### Recommended on HPC

Use a shared filesystem path so compute nodes can access the same files:

```bash
export PSD_DATA_DIR=/path/to/shared/psd_cache
```

### Download precomputed data and models (online only)

Run these on a node with internet access:

```bash
# Precomputed features for built-in species
psd download --species ath
psd download --species bna
psd download --species osa
psd download --species zma

# DeeperDeepSEA models (for cross-species prediction / "other")
# Recommended: download all model variants needed for sliding/strict TF prediction
psd download --species other

# Arabidopsis TF BED peaks resource (only needed for BED-based strict TF recompute)
psd download --resource ath_tf_bed
```

---

## Concepts: regulatory windows, bins, TF modes

### Regulatory windows (default)

PlantSetDelta extracts regulatory regions around:

- **TSS**: `-1 kb` upstream to `+0.5 kb` downstream  
- **TTS**: `-0.5 kb` upstream to `+1 kb` downstream  

These are configurable in `psd build` via:

- `--tss-upstream`, `--tss-downstream`
- `--tts-upstream`, `--tts-downstream`

All values must be multiples of `--BIN_SIZE` (fixed at **100 bp** in this package).

### Binning

Each regulatory region is split into **100 bp bins**. For the default 1500 bp windows, you get **15 bins** per region.

TF features are expressed as:

- `TFNAME_bin_1`, `TFNAME_bin_2`, ..., `TFNAME_bin_N`
- Then suffixed by region during merge:
  - `..._tss` and `..._tts`

### TF feature modes: strict vs sliding-window

This package supports two TF feature generation paradigms:

#### 1) Strict bin (non-sliding)
Each bin corresponds exactly to the genomic 100 bp interval.

- Arabidopsis strict TF can be derived from BED peaks (true strict) or from a center=100 DeeperDeepSEA model (for non-ath species strict prediction).

#### 2) Sliding-window
The model output represents a window larger than the 100 bp input bin (e.g. centered prediction window), leading to a *sliding* effect across bins.

### `--sliding-window` switch (build time)

`psd build` provides:

- `--sliding-window yes` (default): use **sliding-window TF features**
- `--sliding-window no`: use **strict-bin TF features**

Behavior by species:

| TF mode | ath | bna/osa/zma | other |
|---|---|---|---|
| sliding-window = yes | use precomputed **ath sliding** TF parquet | use precomputed TF parquet (already sliding) | compute TF with **center=200** model (requires genome+gtf) |
| sliding-window = no | use precomputed **ath strict** TF parquet (or recompute via BED if needed) | compute TF with **center=100** model (requires genome+gtf) | compute TF with **center=100** model (requires genome+gtf) |

> Note: k-mer features are independent of this TF mode switch.

---

## Quick start

### Scenario A (simplest): built-in species, default settings, sliding-window TF

1) Download precomputed data (once):

```bash
export PSD_DATA_DIR=/path/to/shared/psd_cache
psd download --species ath
```

2) Prepare label file `labels.csv`:

```csv
gene_id,label
AT1G01010,0
AT1G01020,1
AT1G01030,1
AT1G01040,0
```

3) Build features:

```bash
psd build --species ath --label labels.csv -o out_ath
```

4) Train:

```bash
psd train -d out_ath/train_data.csv -o out_ath --n-runs 10
```

5) Export top features:

```bash
psd top -m out_ath/best_model.pkl -d out_ath/train_data.csv -o out_ath
```

---

## Build: generate feature matrix

### Command

```bash
psd build --species <ath|bna|osa|zma|other> --label labels.csv -o out_dir [options...]
```

### Required inputs

- `--label`: CSV with columns `gene_id` and `label`
- If `--features` is not supplied, you must pass `--species`

### Key options

#### TF mode switch (most important)

```bash
--sliding-window yes|no    # default: yes
```

#### Use precomputed data (built-in species)

If **intervals are default** and `species` is one of `ath/bna/osa/zma`, PlantSetDelta will try to use precomputed parquet data.

If you change any interval (`--tss-upstream`, etc.), it triggers recompute (requires genome+gtf).

#### Recompute from genome + annotation

For recompute (or for `species other`), you must provide:

```bash
--genome-fa genome.fa
--gtf annotation.gff3   # accepts GFF3 or GTF but must contain gene features
```

#### Custom intervals

All must be multiples of 100:

```bash
--tss-upstream 1000 --tss-downstream 500
--tts-upstream 500  --tts-downstream 1000
```

#### k-mer size

```bash
--k 7   # allowed: 5, 6, 7
```

### Build examples (from simple to complex)

#### Example 1: Built-in species, default (sliding-window TF)
```bash
psd build --species osa --label labels.csv -o out_osa
```

#### Example 2: Arabidopsis strict TF (precomputed strict parquet)
```bash
psd build --species ath --label labels.csv -o out_ath_strict --sliding-window no
```

#### Example 3: Non-ath strict TF (forces center=100 model prediction)
This requires genome + annotation:

```bash
psd download --species other
psd build \
  --species bna \
  --label labels.csv \
  --sliding-window no \
  --genome-fa Brassica_napus.fa \
  --gtf zs11.gff3 \
  -o out_bna_strict
```

#### Example 4: Other species, default sliding-window TF (center=200 model)
```bash
psd download --species other
psd build \
  --species other \
  --label labels.csv \
  --genome-fa genome.fa \
  --gtf annotation.gff3 \
  -o out_other_sliding
```

#### Example 5: Other species, strict TF (center=100 model)
```bash
psd download --species other
psd build \
  --species other \
  --label labels.csv \
  --sliding-window no \
  --genome-fa genome.fa \
  --gtf annotation.gff3 \
  -o out_other_strict
```

#### Example 6: Custom regulatory windows (forces recompute)
```bash
psd build \
  --species ath \
  --label labels.csv \
  --tss-upstream 2000 --tss-downstream 500 \
  --tts-upstream 500 --tts-downstream 1500 \
  -o out_custom_intervals \
  --sliding-window yes \
  --genome-fa Arabidopsis.fa \
  --gtf Arabidopsis.gff3
```

> Any custom interval requires genome+gtf because precomputed parquets only cover the default 1500 bp windows.

#### Example 7: Provide your own feature matrix (skip all automatic feature engineering)
If you already have a feature CSV with `gene_id` plus feature columns:

```bash
psd build --label labels.csv --features my_features.csv -o out_custom_features
```

Notes:
- Your feature CSV **must** contain `gene_id`
- The build step will merge labels + features, export `raw_all_features*`, then select top features (default logic)

When TF-binding signal intensity is available (e.g., peak intensity values), we recommend constructing and providing a quantitative feature matrix for training via `--features`, rather than relying only on the default binary encoding. Quantitative features may provide better predictive performance when signal information is available.

---

## Train: AutoML with PyCaret

### Command

```bash
psd train -d out_dir/train_data.csv -o out_dir [options...]
```

### What `psd train` does

For each run (seed), PlantSetDelta performs two evaluations:

1. **Training-set cross-validation (CV) comparison**  
   PyCaret runs k-fold cross-validation on the training partition and produces a model comparison table.  
   The comparison is sorted by **Balanced Accuracy** (custom metric added via `add_metric`).

2. **Holdout test evaluation (PyCaret default split)**  
   Using PyCaret's default train/holdout split inside `setup()`, each candidate model is evaluated on the holdout set
   via `predict_model()`. A second comparison table is saved for the holdout test metrics.

This gives you both:
- a **CV-based view** (robust to data splits), and
- a **holdout-based view** (a quick sanity check on unseen data within the same dataset).

> Note: The "test" results here refer to PyCaret's internal holdout set created by `setup()`, not an external test file.

### Key options

- `--n-runs`: number of training runs (seeds 0..n_runs-1; default 10)
- `--n-jobs`: number of CPU cores used by PyCaret (default 4)
- `--group-column`: optional gene-family or sequence-cluster column. When supplied, both the outer holdout and inner cross-validation keep each group entirely within one partition.
- `--ml-models`: restrict the model set (repeatable). Example:

```bash
psd train -d out/train_data.csv -o out --n-runs 10 --ml-models rf --ml-models xgboost
```

Use a broad model pool:

```bash
psd train -d out/train_data.csv -o out --n-runs 10 --ml-models all
```

### Outputs

All outputs are written to `--out-dir`.

### 1) Feature engineering outputs (`psd build`)

#### Main outputs used for training

- `train_data.csv`  
  Final training matrix (label is last column; `gene_id` removed).

- `train_data_with_gene.csv`  
  Same as `train_data.csv`, but includes the `gene_id` column for traceability.

#### Raw exports (for transparency and reproducibility)

Depending on your build configuration (species, intervals, TSS/TTS enabled, `--slim`), you may see:

- `raw_kmer_features.csv`, `raw_tf_features.csv`  
  Merged k-mer and TF features, with `gene_id`.

- `raw_all_features.csv`, `raw_all_features_with_gene.csv`  
  Fully merged matrix (k-mer + TF + label), with and without `gene_id`.

- If `--slim` is **not** used, intermediate matrices are exported as well:
  - `raw_kmer_tss.csv`, `raw_kmer_tts.csv`
  - `raw_tf_tss.csv`, `raw_tf_tts.csv`

#### Notes on built-in feature selection during build

During `psd build`, features are filtered to:

- `top_n = max(1, int(n_samples * 0.2))`
- by absolute Pearson correlation with `label`

Supervised feature selection is performed during `psd train`, not during `psd build`. The selector is fitted
within each cross-validation training fold and uses the top `max(1, floor(n_training_samples * 0.2))`
features by absolute Pearson correlation for binary labels or ANOVA F-score for multiclass labels.
The validation fold and holdout set are not used to fit the selector.

### 2) Training outputs (`psd train`)

`psd train` produces two sets of results: **CV** and **holdout test** (PyCaret default split). All are placed under `--out-dir`.

#### Model files

- `best_models/best_model_<i>.pkl`  
  Best model per run (seed), where `i` starts at 1.

- `best_model.pkl`  
  Final selected model across runs, chosen by CV Balanced Accuracy.
- `training_manifest.json`  
  Records the models and corresponding training partitions from the current run.

#### Result tables (CV vs test clearly separated)

Per-seed tables (stored under `model_results/`):

- `model_results/model_comparison_cv_batch_<seed>.csv`  
  Training-set cross-validation comparison for that seed.

- `model_results/model_comparison_test_batch_<seed>.csv`  
  Holdout test comparison for that seed.

Aggregated tables (stored at the root of `--out-dir`):

- `all_model_comparison_results_cv.csv`  
  Merged CV comparisons across all seeds.

- `all_model_comparison_results_test.csv`  
  Merged holdout test comparisons across all seeds.

#### Plots (CV vs test clearly separated)

- `balanced_acc_cv_plot.pdf`, `auc_cv_plot.pdf`  
  Distributions across seeds for CV metrics.

- `balanced_acc_test_plot.pdf`, `auc_test_plot.pdf`  
  Distributions across seeds for holdout test metrics.

### 3) Feature interpretation outputs (`psd top`)

- `top10_features.csv`  
  Top 10 features, aggregated across per-seed best models using permutation importance calculated on each model's training partition.

- `top10_lollipop.pdf`  
  Lollipop plot for the top 10 features.

> **Tip (interpreting TF features):** If PlantSetDelta reports top TF features, 
> we recommend checking the corresponding pretrained DeeperDeepSEA task reliability in 
> `TF_slide_performance.txt` (default `--sliding-window`) or `TF_no_slide_performance.txt` (when sliding-window is disabled), 
> and interpreting TFs jointly by their **AP** and the downstream **feature importance** 
> (e.g., using a simple combined score such as `AP × importance` to prioritize more reliable TF signals).

## Advanced recipes

### 1) Full offline workflow on HPC (recommended)

On login node (with internet):

```bash
export PSD_DATA_DIR=/share/psd_cache
psd download --species ath
psd download --species other
psd download --resource ath_tf_bed
```

On compute node (no internet):

```bash
export PSD_DATA_DIR=/share/psd_cache
psd build --species other --label labels.csv --genome-fa genome.fa --gtf anno.gff3 -o out
psd train -d out/train_data.csv -o out --n-runs 10
psd top -m out/best_model.pkl -d out/train_data.csv -o out
```

### 2) Reproducible comparison: sliding vs strict TF

```bash
# Sliding-window
psd build --species ath --label labels.csv -o out_slide --sliding-window yes

# Strict-bin
psd build --species ath --label labels.csv -o out_strict --sliding-window no
```

Train and compare models in each output directory.

### 3) Reduce runtime

- Reduce `--n-runs` (e.g., 3–5) for quick iterations
- Restrict `--ml-models` to a small subset
- Consider filtering your label set to fewer genes for pilot runs

---

## Troubleshooting

### “Model not found … Please run psd download --species other”
You are trying to compute TF features for `species other` (or strict TF for non-ath species) but the DeeperDeepSEA model file is missing in `PSD_DATA_DIR`.

Fix:

```bash
psd download --species other
```

Ensure `PSD_DATA_DIR` points to the same directory on compute nodes.

### “bedtools not found” / pybedtools errors
Install bedtools:

```bash
conda install -c bioconda bedtools -y
```

### “No gene features found” / empty FASTA
Your annotation may not contain `gene` entries, or gene IDs are not in the expected attributes.

Fix:
- Ensure GFF3/GTF includes `gene` lines
- Ensure gene IDs are present in `ID=` (GFF3) or `gene_id` (GTF)

### Contig name mismatch
If your genome FASTA uses `chr1` but GFF uses `1`, the extractor will skip everything.

Fix:
- Normalize contig names between FASTA and GFF/GTF before running.

### TF feature matrix seems shifted / window definition confusion
This is exactly why `--sliding-window` exists. Use:
- `--sliding-window no` for strict bin semantics
- `--sliding-window yes` for sliding-window semantics

---

## Acknowledgements

- Selene SDK
- PyCaret
- Biopython
- bedtools / pybedtools
