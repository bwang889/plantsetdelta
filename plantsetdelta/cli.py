"""Command-line interface: psd download / build / train / top

Key behaviors implemented:
- --sliding-window yes|no (default: yes)
- Supports TSS-only / TTS-only by allowing one side interval to be 0/0.
  At least one side must be enabled.
- Precomputed behavior (default intervals only):
    sliding-window=yes:
        ath -> use tf_tss_slide/tf_tts_slide
        bna/osa/zma -> use tf_tss/tf_tts (historical sliding)
    sliding-window=no (strict):
        ath -> use tf_tss/tf_tts (strict)
        bna/osa/zma -> NO strict precomputed => force recompute (requires genome+gtf) using center100 model
- Recompute behavior (custom intervals or strict for non-ath or species=other):
    sliding-window=yes:
        ath -> use center200 model to predict (not BED)
        non-ath/other -> use center200 model to predict
    sliding-window=no:
        ath -> use BED strict bins
        non-ath/other -> use center100 model to predict
"""

from __future__ import annotations

import click
import pandas as pd
from pathlib import Path

from .feature_engineering import build_kmer_matrix, build_tf_matrix_variable_bins
from .feature_engineering.ath_tf_bed import build_ath_tf_matrix_from_bed
from .trainer import train_best_model
from .importance import export_top_features

from .utils.config import (
    PRECOMPUTED_SPECIES,
    download_species_data,
    get_species_file_paths,
    FEATURES_PATH,
    download_deeper_model,
    ensure_model_downloaded_offline,
    download_ath_tf_bed,
    get_ath_tf_bed_offline,
    DEFAULT_TSS_UPSTREAM,
    DEFAULT_TSS_DOWNSTREAM,
    DEFAULT_TTS_UPSTREAM,
    DEFAULT_TTS_DOWNSTREAM,
    BIN_SIZE,
    MODEL_PATH,
    MODEL100_PATH,
)
from .utils.extract_sequences import generate_tss_tts_sequences


@click.group()
def cli():
    """Command-line interface for PlantSetDelta"""
    pass


# ============================================================
# DOWNLOAD
# ============================================================
@cli.command()
@click.option("--species", type=str, required=False, help="ath / bna / osa / zma / other")
@click.option(
    "--resource",
    type=click.Choice(["ath_tf_bed"], case_sensitive=False),
    required=False,
    help="Download extra resources for offline build (e.g., ath_tf_bed).",
)
@click.option("--force", is_flag=True, help="Force re-download even if files already exist.")
def download(species, resource, force):
    """
    Download required models or precomputed data/resources (requires internet access).

    Examples:
      psd download --species ath
      psd download --species other          # downloads BOTH center200 and center100 models
      psd download --resource ath_tf_bed
    """
    if (species is None) and (resource is None):
        raise click.BadParameter("You must specify either --species or --resource.")

    if species:
        species = str(species).lower().strip()
        if species in PRECOMPUTED_SPECIES:
            click.echo(f"Downloading precomputed data for {species}...")
            download_species_data(species, force=force)
            click.echo("Precomputed data download complete.")
        elif species == "other":
            # IMPORTANT: download BOTH models to support strict/sliding offline
            click.echo("Downloading DeeperDeepSEA models (center200 + center100)...")
            download_deeper_model(force=force, model_type="both")
            click.echo("Model downloads complete.")
        else:
            raise click.BadParameter("Unknown species. Use one of: ath / bna / osa / zma / other.")

    if resource:
        if resource.lower() == "ath_tf_bed":
            click.echo("Downloading Arabidopsis TF BED resource...")
            p = download_ath_tf_bed(force=force)
            click.echo(f"Arabidopsis TF BED downloaded to: {p}")
        else:
            raise click.BadParameter(f"Unknown resource: {resource}")


# ============================================================
# BUILD
# ============================================================
def _require_file(p: Path, hint: str):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}\n{hint}")


def _empty_gene_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["gene_id"])


def _validate_gene_table(df: pd.DataFrame, name: str, require_label: bool = False) -> pd.DataFrame:
    required = {"gene_id", "label"} if require_label else {"gene_id"}
    if not required.issubset(df.columns):
        raise click.BadParameter(f"{name} must contain columns: {', '.join(sorted(required))}")
    out = df.copy()
    if out["gene_id"].isna().any():
        raise click.BadParameter(f"{name} contains missing gene_id values.")
    out["gene_id"] = out["gene_id"].astype(str).str.strip()
    if out["gene_id"].eq("").any():
        raise click.BadParameter(f"{name} contains empty gene_id values.")
    if out["gene_id"].duplicated().any():
        duplicates = out.loc[out["gene_id"].duplicated(), "gene_id"].head(5).tolist()
        raise click.BadParameter(f"{name} contains duplicate gene_id values, e.g. {duplicates}")
    if require_label and out["label"].isna().any():
        raise click.BadParameter(f"{name} contains missing labels.")
    return out


def _kmer_to_gene_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_gene_df()
    seq_col = "sequence" if "sequence" in df.columns else ("Sequence" if "Sequence" in df.columns else None)
    if seq_col is None:
        raise ValueError("kmer dataframe must contain a sequence/Sequence column (gene_id).")
    out = df.rename(columns={seq_col: "gene_id"})
    out["gene_id"] = out["gene_id"].astype(str)
    return out


def _ensure_gene_id_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_gene_df()
    if "gene_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "gene_id"})
    df["gene_id"] = df["gene_id"].astype(str)
    return df


def _filter_by_gene_ids(df: pd.DataFrame, gene_ids: set[str] | None) -> pd.DataFrame:
    # Keep only label genes to reduce memory footprint across merges.
    # An empty whitelist implies there is nothing to keep, so return empty.
    if df is None or df.empty:
        return _empty_gene_df()
    if not gene_ids:
        return _empty_gene_df()
    return df[df["gene_id"].isin(gene_ids)].reset_index(drop=True)


def _read_parquet_filtered(path: Path, gene_ids: set[str] | None, id_cols: list[str]) -> pd.DataFrame:
    # Load only rows whose gene_id (or equivalent column) is in the whitelist to reduce memory.
    # Falls back to full read if pyarrow isn't available or the ID column isn't in schema.
    if not gene_ids:
        return _empty_gene_df()
    try:
        import pyarrow.dataset as ds
    except Exception:
        return pd.read_parquet(path)

    dataset = ds.dataset(str(path), format="parquet")
    col = next((c for c in id_cols if c in dataset.schema.names), None)
    if col is None:
        return pd.read_parquet(path)

    table = dataset.to_table(filter=ds.field(col).isin(list(gene_ids)))
    return table.to_pandas()


def _suffix_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_gene_df()
    return df.rename(columns=lambda c: f"{c}_{suffix}" if c != "gene_id" else c)


def _merge_feature_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return _empty_gene_df()
    merged = parts[0]
    for p in parts[1:]:
        merged = pd.merge(merged, p, on="gene_id", how="inner")
    return merged


@cli.command()
@click.option("--species", type=str, help="ath / bna / osa / zma / other; if not set, you must specify --features")
@click.option("--label", type=click.Path(exists=True), required=True,
              help="CSV file with columns gene_id and label (supports binary or multiclass)")
@click.option("--features", type=click.Path(exists=True),
              help="User-supplied feature file (CSV, must contain gene_id column)")
@click.option("-o", "--out-dir", type=click.Path(), default="psd_output")

# genome/annotation inputs
@click.option("--genome-fa", type=click.Path(exists=True), help="Reference genome FASTA (required for recompute)")
@click.option("--gtf", type=click.Path(exists=True), help="Gene annotation file (GTF/GFF3; required for recompute)")

# k-mer
@click.option("--k", type=click.IntRange(5, 7), default=7, show_default=True, help="k-mer size (5–7)")

# custom intervals
@click.option("--tss-upstream", type=int, default=DEFAULT_TSS_UPSTREAM, show_default=True)
@click.option("--tss-downstream", type=int, default=DEFAULT_TSS_DOWNSTREAM, show_default=True)
@click.option("--tts-upstream", type=int, default=DEFAULT_TTS_UPSTREAM, show_default=True)
@click.option("--tts-downstream", type=int, default=DEFAULT_TTS_DOWNSTREAM, show_default=True)

# optional: user-provided ath tf bed
@click.option("--ath-tf-bed", type=click.Path(exists=True), help="Arabidopsis TF peaks BED (for ath strict recompute)")

# ✅ Sliding window switch
@click.option(
    "--sliding-window",
    type=click.Choice(["yes", "no"], case_sensitive=False),
    default="yes",
    show_default=True,
    help="yes: sliding-window TF features; no: strict-bin TF features.",
)
# Fast path: skip heavy raw CSV exports when you only need final outputs.
@click.option("--slim", is_flag=True, help="Skip exporting intermediate raw CSV files to save time and disk space.")
def build(
    species,
    label,
    features,
    out_dir,
    genome_fa,
    gtf,
    k,
    tss_upstream,
    tss_downstream,
    tts_upstream,
    tts_downstream,
    ath_tf_bed,
    sliding_window,
    slim,
):
    """
    Build feature matrix.

    Offline principle:
      - build must NOT download anything.
      - required resources must be pre-downloaded via `psd download`
        or explicitly provided via CLI paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sliding_window = str(sliding_window).lower()
    use_sliding = (sliding_window == "yes")

    # validate intervals: allow 0; non-zero must be multiple of BIN_SIZE
    for v, name in [
        (tss_upstream, "tss-upstream"),
        (tss_downstream, "tss-downstream"),
        (tts_upstream, "tts-upstream"),
        (tts_downstream, "tts-downstream"),
    ]:
        if v != 0 and (v % BIN_SIZE != 0):
            raise click.BadOptionUsage(
                f"--{name}",
                f"All interval params must be 0 or multiples of {BIN_SIZE}. Got: {name}={v}"
            )

    use_tss = (int(tss_upstream) + int(tss_downstream)) > 0
    use_tts = (int(tts_upstream) + int(tts_downstream)) > 0
    if not use_tss and not use_tts:
        raise click.BadOptionUsage(
            "--tss-upstream/--tss-downstream/--tts-upstream/--tts-downstream",
            "Both TSS and TTS windows are 0 bp. At least one of TSS or TTS must be > 0."
        )

    custom_intervals = (
        tss_upstream != DEFAULT_TSS_UPSTREAM
        or tss_downstream != DEFAULT_TSS_DOWNSTREAM
        or tts_upstream != DEFAULT_TTS_UPSTREAM
        or tts_downstream != DEFAULT_TTS_DOWNSTREAM
    )

    # Read label file
    label_df = _validate_gene_table(pd.read_csv(label), "--label", require_label=True)
    # Whitelist for early filtering to avoid loading/merging unrelated genes.
    gene_id_set = set(label_df["gene_id"].tolist())

    # If user supplies features: keep your behavior
    if features:
        feature_df = _validate_gene_table(pd.read_csv(features), "--features")
        # Filter user-supplied features to the label gene set before merging.
        feature_df = feature_df[feature_df["gene_id"].isin(gene_id_set)]

        merged = pd.merge(label_df, feature_df, on="gene_id", how="inner")
        if merged.shape[0] == 0:
            raise click.UsageError("After merging label and features, 0 samples remain. Check gene_id consistency.")

        merged.to_csv(out_dir / "raw_all_features_with_gene.csv", index=False)
        merged.drop(columns=["gene_id"]).to_csv(out_dir / "raw_all_features.csv", index=False)

        features_only = merged.drop(columns=["gene_id", "label"])
        if features_only.shape[1] == 0:
            raise click.UsageError("No features found in the provided feature file.")
        # Keep all features. Supervised selection is fitted inside each training/CV fold.
        merged.to_csv(out_dir / "train_data_with_gene.csv", index=False)
        merged.drop(columns=["gene_id"]).to_csv(out_dir / "train_data.csv", index=False)
        click.echo(
            "Labels and features merged. Full feature matrix saved; supervised "
            f"feature selection will occur inside model training in {out_dir}"
        )
        return

    # Auto feature workflow requires species
    if not species:
        raise click.BadOptionUsage("--species", "If --features is not specified, --species is required.")
    species = str(species).lower().strip()

    # Decide whether we can use precomputed parquet
    # - only when default intervals AND species in PRECOMPUTED_SPECIES
    # - BUT strict mode for non-ath must force recompute
    precomputed_possible = (not custom_intervals) and (species in PRECOMPUTED_SPECIES)
    force_recompute_strict_nonath = (not use_sliding) and (species in PRECOMPUTED_SPECIES) and (species != "ath")

    use_precomputed = precomputed_possible and (not force_recompute_strict_nonath)

    # Explicit status log so users can tell precomputed vs recompute immediately.
    if use_precomputed:
        click.echo(f"✅ Using precomputed data for {species} (skipping recompute)...")
    else:
        click.echo(f"⚠️  Recomputing features from scratch (species={species}, custom_intervals={custom_intervals})...")

    # If need recompute, require genome+gtf
    need_recompute = (not use_precomputed) or (species == "other")
    if need_recompute and not (genome_fa and gtf):
        raise click.BadOptionUsage(
            "--genome-fa/--gtf",
            "Recompute requires --genome-fa and --gtf (this includes strict mode for bna/osa/zma, or species=other, or custom intervals)."
        )

    # ---------- Load / build kmer ----------
    if use_precomputed:
        file_paths = get_species_file_paths(species)
        _require_file(file_paths["kmer_tss"], "Run: psd download --species <ath|bna|osa|zma> and set PSD_DATA_DIR.")
        _require_file(file_paths["kmer_tts"], "Run: psd download --species <ath|bna|osa|zma> and set PSD_DATA_DIR.")
        # Filter precomputed k-mer rows by gene IDs as early as possible to save memory.
        kmer_df_tss = (
            _read_parquet_filtered(
                file_paths["kmer_tss"],
                gene_id_set,
                ["sequence", "Sequence", "gene_id", "__index_level_0__"],
            )
            if use_tss else pd.DataFrame(columns=["sequence"])
        )
        kmer_df_tts = (
            _read_parquet_filtered(
                file_paths["kmer_tts"],
                gene_id_set,
                ["sequence", "Sequence", "gene_id", "__index_level_0__"],
            )
            if use_tts else pd.DataFrame(columns=["sequence"])
        )
    else:
        click.echo("Generating TSS/TTS sequences with user-defined intervals...")
        # Only generate FASTA entries for label genes to reduce downstream memory.
        seqs = generate_tss_tts_sequences(
            Path(genome_fa),
            Path(gtf),
            out_dir,
            tss_upstream=tss_upstream,
            tss_downstream=tss_downstream,
            tts_upstream=tts_upstream,
            tts_downstream=tts_downstream,
            gene_id_whitelist=gene_id_set,
        )
        tss_fasta = seqs["tss_fasta"]
        tts_fasta = seqs["tts_fasta"]

        kmer_df_tss = build_kmer_matrix(Path(tss_fasta), k) if use_tss else pd.DataFrame(columns=["sequence"])
        kmer_df_tts = build_kmer_matrix(Path(tts_fasta), k) if use_tts else pd.DataFrame(columns=["sequence"])

    kmer_df_tss = _kmer_to_gene_id(kmer_df_tss) if use_tss else _empty_gene_df()
    kmer_df_tts = _kmer_to_gene_id(kmer_df_tts) if use_tts else _empty_gene_df()
    # Shrink k-mer matrices early to the label gene set.
    kmer_df_tss = _filter_by_gene_ids(kmer_df_tss, gene_id_set) if use_tss else _empty_gene_df()
    kmer_df_tts = _filter_by_gene_ids(kmer_df_tts, gene_id_set) if use_tts else _empty_gene_df()

    # ---------- Load / build TF ----------
    if use_precomputed:
        file_paths = get_species_file_paths(species)

        # choose which precomputed TF to use
        if species == "ath" and use_sliding:
            tf_key_tss = "tf_tss_slide"
            tf_key_tts = "tf_tts_slide"
            if tf_key_tss not in file_paths or tf_key_tts not in file_paths:
                raise FileNotFoundError(
                    "Arabidopsis sliding TF precomputed paths not found in get_species_file_paths(). "
                    "Make sure your config.py returns tf_tss_slide/tf_tts_slide for arabidopsis."
                )
            _require_file(file_paths[tf_key_tss], "Run: psd download --species ath (with updated Zenodo files).")
            _require_file(file_paths[tf_key_tts], "Run: psd download --species ath (with updated Zenodo files).")
            # Filter precomputed TF rows by gene IDs at load time when possible.
            tf_df_tss = (
                _read_parquet_filtered(
                    file_paths[tf_key_tss],
                    gene_id_set,
                    ["gene_id", "__index_level_0__"],
                )
                if use_tss else _empty_gene_df()
            )
            tf_df_tts = (
                _read_parquet_filtered(
                    file_paths[tf_key_tts],
                    gene_id_set,
                    ["gene_id", "__index_level_0__"],
                )
                if use_tts else _empty_gene_df()
            )
        else:
            # bna/osa/zma always use tf_tss/tf_tts (historically sliding), ath strict uses tf_tss/tf_tts
            _require_file(file_paths["tf_tss"], "Run: psd download --species <ath|bna|osa|zma> and set PSD_DATA_DIR.")
            _require_file(file_paths["tf_tts"], "Run: psd download --species <ath|bna|osa|zma> and set PSD_DATA_DIR.")
            # Filter precomputed TF rows by gene IDs at load time when possible.
            tf_df_tss = (
                _read_parquet_filtered(
                    file_paths["tf_tss"],
                    gene_id_set,
                    ["gene_id", "__index_level_0__"],
                )
                if use_tss else _empty_gene_df()
            )
            tf_df_tts = (
                _read_parquet_filtered(
                    file_paths["tf_tts"],
                    gene_id_set,
                    ["gene_id", "__index_level_0__"],
                )
                if use_tts else _empty_gene_df()
            )

        tf_df_tss = _ensure_gene_id_col(tf_df_tss) if use_tss else _empty_gene_df()
        tf_df_tts = _ensure_gene_id_col(tf_df_tts) if use_tts else _empty_gene_df()
        # Filter precomputed TF matrices to label genes to save memory before merge.
        tf_df_tss = _filter_by_gene_ids(tf_df_tss, gene_id_set) if use_tss else _empty_gene_df()
        tf_df_tts = _filter_by_gene_ids(tf_df_tts, gene_id_set) if use_tts else _empty_gene_df()

    else:
        # recompute TF
        # regenerate sequences if not already generated above
        if "seqs" not in locals():
            click.echo("Generating TSS/TTS sequences with user-defined intervals...")
            seqs = generate_tss_tts_sequences(
                Path(genome_fa),
                Path(gtf),
                out_dir,
                tss_upstream=tss_upstream,
                tss_downstream=tss_downstream,
                tts_upstream=tts_upstream,
                tts_downstream=tts_downstream,
                gene_id_whitelist=gene_id_set,
            )
            tss_fasta = seqs["tss_fasta"]
            tts_fasta = seqs["tts_fasta"]

        if species == "ath":
            if not use_sliding:
                # strict for ath: BED-based bin aligned
                bed_path = Path(ath_tf_bed) if ath_tf_bed else get_ath_tf_bed_offline()
                click.echo(f"Computing Arabidopsis TF strict-bin features from BED: {bed_path}")
                tf_tss_mat, tf_tts_mat = build_ath_tf_matrix_from_bed(
                    gff_file=Path(gtf),
                    tf_bed_file=bed_path,
                    tss_upstream=tss_upstream,
                    tss_downstream=tss_downstream,
                    tts_upstream=tts_upstream,
                    tts_downstream=tts_downstream,
                    bin_size=BIN_SIZE,
                )
                tf_df_tss = tf_tss_mat.reset_index().rename(columns={"index": "gene_id"}) if use_tss else _empty_gene_df()
                tf_df_tts = tf_tts_mat.reset_index().rename(columns={"index": "gene_id"}) if use_tts else _empty_gene_df()
            else:
                # sliding for ath: use center200 model prediction
                ensure_model_downloaded_offline("center200")
                click.echo("Computing Arabidopsis TF sliding-window features with DeeperDeepSEA (center200)...")
                tf_df_tss = (
                    build_tf_matrix_variable_bins(Path(tss_fasta), MODEL_PATH, FEATURES_PATH, bin_size=BIN_SIZE)
                    .reset_index().rename(columns={"index": "gene_id"})
                    if use_tss else _empty_gene_df()
                )
                tf_df_tts = (
                    build_tf_matrix_variable_bins(Path(tts_fasta), MODEL_PATH, FEATURES_PATH, bin_size=BIN_SIZE)
                    .reset_index().rename(columns={"index": "gene_id"})
                    if use_tts else _empty_gene_df()
                )
        else:
            # non-ath (including other)
            if use_sliding:
                ensure_model_downloaded_offline("center200")
                model = MODEL_PATH
                click.echo("Computing TF sliding-window features with DeeperDeepSEA (center200)...")
            else:
                ensure_model_downloaded_offline("center100")
                model = MODEL100_PATH
                click.echo("Computing TF strict-bin features with DeeperDeepSEA (center100)...")

            tf_df_tss = (
                build_tf_matrix_variable_bins(Path(tss_fasta), model, FEATURES_PATH, bin_size=BIN_SIZE)
                .reset_index().rename(columns={"index": "gene_id"})
                if use_tss else _empty_gene_df()
            )
            tf_df_tts = (
                build_tf_matrix_variable_bins(Path(tts_fasta), model, FEATURES_PATH, bin_size=BIN_SIZE)
                .reset_index().rename(columns={"index": "gene_id"})
                if use_tts else _empty_gene_df()
            )

        tf_df_tss = _ensure_gene_id_col(tf_df_tss) if use_tss else _empty_gene_df()
        tf_df_tts = _ensure_gene_id_col(tf_df_tts) if use_tts else _empty_gene_df()
        # Filter recomputed TF matrices to label genes as early as possible.
        tf_df_tss = _filter_by_gene_ids(tf_df_tss, gene_id_set) if use_tss else _empty_gene_df()
        tf_df_tts = _filter_by_gene_ids(tf_df_tts, gene_id_set) if use_tts else _empty_gene_df()

    # ---------- Export raw matrices ----------
    # Skip intermediate raw CSVs in slim mode to reduce disk I/O and memory pressure.
    if not slim:
        click.echo("Exporting raw feature matrices (use --slim to skip)...")
        (kmer_df_tss if use_tss else _empty_gene_df()).to_csv(out_dir / "raw_kmer_tss.csv", index=False)
        (kmer_df_tts if use_tts else _empty_gene_df()).to_csv(out_dir / "raw_kmer_tts.csv", index=False)
        (tf_df_tss if use_tss else _empty_gene_df()).to_csv(out_dir / "raw_tf_tss.csv", index=False)
        (tf_df_tts if use_tts else _empty_gene_df()).to_csv(out_dir / "raw_tf_tts.csv", index=False)


    # ---------- Merge & rename ----------
    kmer_parts = []
    if use_tss:
        kmer_parts.append(_suffix_cols(kmer_df_tss, "tss"))
    if use_tts:
        kmer_parts.append(_suffix_cols(kmer_df_tts, "tts"))
    raw_kmer_df = _merge_feature_parts(kmer_parts)
    raw_kmer_df.to_csv(out_dir / "raw_kmer_features.csv", index=False)

    tf_parts = []
    if use_tss:
        tf_parts.append(_suffix_cols(tf_df_tss, "tss"))
    if use_tts:
        tf_parts.append(_suffix_cols(tf_df_tts, "tts"))
    raw_tf_df = _merge_feature_parts(tf_parts)
    raw_tf_df.to_csv(out_dir / "raw_tf_features.csv", index=False)

    raw_all = pd.merge(raw_kmer_df, raw_tf_df, on="gene_id", how="inner")
    raw_all = pd.merge(label_df, raw_all, on="gene_id", how="inner")

    raw_all.to_csv(out_dir / "raw_all_features_with_gene.csv", index=False)
    raw_all.drop(columns=["gene_id"]).to_csv(out_dir / "raw_all_features.csv", index=False)

    if raw_all.shape[0] == 0:
        raise click.UsageError(
            "After merging label + features, there are 0 samples.\n"
            "Most common causes:\n"
            "  - gene_id mismatch between label file and extracted features\n"
            "  - FASTA contig names mismatch with GFF/GTF\n"
            "  - using TSS-only/TTS-only reduces overlap\n"
            f"Current settings: species={species}, sliding_window={sliding_window}, "
            f"use_tss={use_tss}, use_tts={use_tts}"
        )

    # label last
    label_col = raw_all.pop("label")
    raw_all["label"] = label_col

    # Keep the full feature matrix. Supervised selection is fitted inside each training/CV fold.
    raw_all.to_csv(out_dir / "train_data_with_gene.csv", index=False)
    raw_all.drop(columns=["gene_id"]).to_csv(out_dir / "train_data.csv", index=False)

    click.echo(
        f"Raw matrices saved. Full feature matrix saved as train_data.csv/train_data_with_gene.csv in {out_dir}\n"
        f"Mode summary: species={species}, sliding_window={sliding_window}, use_precomputed={use_precomputed}, "
        f"use_tss={use_tss}, use_tts={use_tts}"
    )


# ============================================================
# TRAIN
# ============================================================
@cli.command()
@click.option("-d", "--data", type=click.Path(exists=True), required=True)
@click.option("-o", "--out-dir", type=click.Path(), default="psd_output")
@click.option("--n-runs", type=int, default=10, show_default=True, help="Number of training runs (seeds 0..n-1).")
# Control PyCaret parallelism; set lower values on shared clusters to avoid oversubscription.
@click.option("--n-jobs", type=int, default=4, show_default=True, help="Number of CPU cores for PyCaret training.")
@click.option(
    "--ml-models",
    multiple=True,
    default=["xgboost", "nb", "lr", "gbc", "rf"],
    help="Subset of models to include. Example: --ml-models nb rf gbc; use 'all' to include all models.",
)
def train(data, out_dir, n_runs, n_jobs, ml_models):
    """Train multiple models and select the best one. Also exports per-seed model comparisons and ACC/AUC plots."""
    include_models = "all" if (len(ml_models) == 1 and ml_models[0] == "all") else list(ml_models)

    model_path = train_best_model(
        Path(data),
        target_col="label",
        out_dir=Path(out_dir),
        n_runs=n_runs,
        include_models=include_models,
        n_jobs=n_jobs,
    )
    click.echo(f"Model training complete. Best model saved to: {model_path}")


# ============================================================
# TOP
# ============================================================
@cli.command()
@click.option("-m", "--model", type=click.Path(exists=True), required=True)
@click.option("-d", "--data", type=click.Path(exists=True), required=True)
@click.option("-o", "--out-dir", type=click.Path(), default="psd_output")
def top(model, data, out_dir):
    """Export top-10 important features (CSV + PDF plot)"""
    csv_path, pdf_path = export_top_features(Path(model), Path(data), target_col="label", out_dir=Path(out_dir))
    click.echo(f"Top features exported:\n CSV: {csv_path}\n Plot: {pdf_path}")


if __name__ == "__main__":
    cli()