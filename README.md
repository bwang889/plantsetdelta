# PlantSetDelta

A general-purpose toolkit for predicting key regulatory sequence features and transcription factor (TF) binding differences between any gene sets across plant species.

> Developed with k-mer frequency extraction, deep learning–based TF-binding prediction, automated model training (via PyCaret), and interpretable feature analysis.

---

## Installation

We recommend using Python 3.10 and a clean conda environment.

```bash
git clone https://github.com/bwang889/plantsetdelta.git
cd plantsetdelta
pip install .
```

Required dependencies include: `pycaret`, `selene_sdk`, `biopython`, `bedtools`, `scikit-learn`, etc. See `requirements.txt` for full list.

> ⚠️ **External dependency:** [bedtools](https://bedtools.readthedocs.io/en/latest/) is required for sequence extraction. Install via:  
> `conda install -c bioconda bedtools`

---

## Quick Start

### 1. Download precomputed features for Arabidopsis

```bash
psd download --species ath
```

Available species:

- `ath` (*Arabidopsis thaliana*)
- `bna` (*Brassica napus*)
- `osa` (*Oryza sativa*)
- `zma` (*Zea mays*)

> Precomputed features include **TF-binding** and **k-mer (k=7)** matrices.

### 2. Prepare label file (CSV, supports binary or multiclass)

Example format:

```csv
gene_id,label
AT1G01010,0
AT1G01020,1
AT1G01030,1
AT1G01040,0
...
```

- **gene_id**: gene name/identifier
- **label**: integer value, e.g. 0, 1 (binary) or 0/1/2... (multiclass)

### 3. Build feature matrix

**Auto-generate features for a supported species:**

```bash
psd build --species ath --label my_label.csv -o output_dir
```

**For other plant species or custom genomes:**

Provide genome FASTA and annotation GTF/GFF:

```bash
psd build --species other --label my_label.csv --genome-fa genome.fa --gtf annotation.gff3 -o output_dir
```

Or supply pre-extracted 1.5kb TSS/TTS FASTA files:

```bash
psd build --species other --label my_label.csv --tss-fasta tss.fa --tts-fasta tts.fa -o output_dir
```

### 4. Train models

```bash
psd train -d output_dir/train_data.csv -o output_dir
```

You may specify models or use all:

```bash
psd train -d output_dir/train_data.csv -o output_dir --seeds 10 --ml-models xgboost --ml-models rf
```

Supported models include: `catboost`, `lr`, `lightgbm`, `xgboost`, `ada`, `ridge`, `gbc`, `svm`, `lda`, `rf`, `et`, `nb`, `knn`, `dt`, `qda`, `dummy`  
Use `--ml-models all` to include all.

### 5. Get top features

```bash
psd top -m output_dir/best_model.pkl -d output_dir/train_data.csv -o output_dir
```

Outputs:

- `top10_features.csv`
- `top10_lollipop.pdf`

---

## Input File Format

###  `my_label.csv` (REQUIRED)

A CSV file containing at least:

| gene_id     | label |
|-------------|-------|
| AT1G01010   | 0     |
| AT1G01020   | 1     |
| ...         | ...   |

- Labels can be binary (0/1) or multiclass (0/1/2...).

### `genome.fa` (Reference genome, for `--species other`)

- Standard FASTA format.
- Chromosome/contig names must match column 1 in the GFF/GTF file.

###  `annotation.gff3` or `annotation.gtf` (for `--species other`)

- Must include `gene` entries.
- Gene ID must appear as `ID=gene:XXX` (GFF3) or `gene_id "XXX"` (GTF).

###  `tss.fa` / `tts.fa` (Custom 1.5kb TSS/TTS FASTA, for `--species other`)

- FASTA format; each sequence **exactly 1500 bp**.
- Header is **unique gene ID only** (e.g., `>AT1G01010`).

---

## Output Files

- `train_data.csv`: training matrix (`label` is always the last column; gene_id not included)
- `train_data_with_gene.csv`: same as above but keeps gene_id for reference
- `best_model.pkl`: PyCaret-trained model
- `top10_features.csv`: top informative features after model interpretation
- `top10_lollipop.pdf`: feature importance plot

---

## Feature Matrix Structure

Each row: a gene  
Each column: a feature  
- k-mer features: e.g., `ACGTGTA_tss`, `TTTGGCA_tts`
- TF-binding bins: e.g., `WRKY33_bin_1`, `NAC13_bin_7`

**Note:**  
During `build`, all features (auto or custom) are automatically filtered to the **top N features, where N = 20% of sample number** (rounded down, min=1), based on absolute Pearson correlation with the label.

---

## Detailed Usage

### Download: precomputed features or model

```bash
psd download --species ath        # Download features for Arabidopsis
psd download --species other      # Only download pretrained DeeperDeepSEA model (for "other" genomes)
```

#### Parameters

- `--species` : one of `[ath, bna, osa, zma, other]`  
  Supported species codes.

---

### Build: create regulatory feature matrix

```bash
psd build --species ath --label my_label.csv -o output_dir
```

#### Parameters

- `--species` : one of `[ath, bna, osa, zma, other]`  
  Supported species codes.
- `--label` : CSV file containing gene IDs and class labels. **Format:** must have `gene_id` and `label` columns.  
  Supports both binary and multiclass tasks.
- `--features` : (optional) User-supplied feature matrix (CSV). Must contain `gene_id` column; all other columns are treated as features.
- `--genome-fa` : Reference genome in FASTA format (used for "other" species when extracting TSS/TTS).
- `--gtf` : Gene annotation file (GTF or GFF3 format), used with `--genome-fa`.
- `--tss-fasta` : 1.5kb TSS FASTA file (optional; for "other" species, alternative to genome+gtf).
- `--tts-fasta` : 1.5kb TTS FASTA file (optional; for "other" species, alternative to genome+gtf).
- `--k` : k-mer length (`5`, `6`, or `7`). Precomputed features use `k=7`.
- `-o, --out-dir` : Output folder for results.

---

### Train: auto ML pipeline

```bash
psd train -d output_dir/train_data.csv -o output_dir --seeds 10 --ml-models all
```

or specify models:

```bash
psd train -d output_dir/train_data.csv -o output_dir --seeds 10 --ml-models catboost --ml-models lr
```

#### Parameters

- `--data` : Path to training CSV (usually `train_data.csv` output from build step).
- `--out-dir` : Directory to save models and results.
- `--seeds` : Number of repeated training runs for model robustness (default: `10`).
- `--ml-models` : Models to include in training (comma-separated or repeat flag).  
  Examples: `--ml-models lr xgboost rf` or `--ml-models all` to evaluate all models.

**Supported models:**

- `catboost`, `lr`, `lightgbm`, `xgboost`, `ada`, `ridge`, `gbc`, `svm`, `lda`, `rf`, `et`, `nb`, `knn`, `dt`, `qda`, `dummy`

> If `--ml-models` is not specified, defaults to the five models: `lr`, `xgboost`, `nb`, `gbc`, and `rf`.

---

### Top: interpret model

```bash
psd top -m output_dir/best_model.pkl -d output_dir/train_data.csv -o output_dir
```

#### Parameters

- `--model` : Path to the trained model file (`.pkl`).
- `--data` : Path to the training data CSV.
- `--out-dir` : Output directory for results.

---

## Supported Species

| Code  | Species Name           |
| ----- | ---------------------- |
| `ath` | *Arabidopsis thaliana* |
| `bna` | *Brassica napus*       |
| `osa` | *Oryza sativa*         |
| `zma` | *Zea mays*             |

> For custom k-mer (`--k`) or unsupported species, use `--species other` with a reference genome or TSS/TTS FASTA.

---

## Developer Notes

- C++ backend for fast k-mer counting
- DeeperDeepSEA model for TF-binding prediction
- PyCaret AutoML backend (16+ classifiers)
- Use `PSD_DATA_DIR` to customize data path

---

## Acknowledgements

- [Selene SDK](https://github.com/FunctionLab/selene)
- [PyCaret](https://github.com/pycaret/pycaret)

---
