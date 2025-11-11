"""Command-line interface: psd download / build / train / top

Supports multiclass/binary label (CSV), user-defined feature files, and the original auto-feature workflow.
"""

import click
import pandas as pd
from pathlib import Path

from .feature_engineering import build_kmer_matrix, build_tf_matrix
from .trainer import train_best_model
from .importance import export_top_features
from .utils.config import (
    PRECOMPUTED_SPECIES,
    download_species_data,
    get_species_file_paths,
    MODEL_PATH,
    FEATURES_PATH,
    ensure_model_downloaded,
)
from .utils.extract_sequences import generate_tss_tts_sequences


@click.group()
def cli():
    """Command-line interface for PlantSetDelta"""
    pass

@cli.command()
@click.option("--species", type=str, required=True, help="ath / bna / osa / zma / other")
def download(species):
    """Download required models or precomputed data (requires internet access)"""
    if species in PRECOMPUTED_SPECIES:
        click.echo(f"Downloading precomputed data for {species}...")
        download_species_data(species)
        click.echo("Precomputed data download complete.")
    elif species == "other":
        click.echo("Downloading DeeperDeepSEA model...")
        ensure_model_downloaded()
        click.echo("Model download complete.")
    else:
        raise click.BadParameter("Unknown species type. Use one of: ath / bna / osa / zma / other.")

@cli.command()
@click.option("--species", type=str, help="ath / bna / osa / zma / other; if not set, you must specify --features")
@click.option("--label", type=click.Path(exists=True), required=True, help="CSV file with columns gene_id and label (supports binary or multiclass)")
@click.option("--features", type=click.Path(exists=True), help="User-supplied feature file (CSV, must contain gene_id column; all other columns are features)")
@click.option("-o", "--out-dir", type=click.Path(), default="psd_output")
@click.option("--tss-fasta", type=click.Path(exists=True), help="For 'other' species: TSS 1.5kb sequence file")
@click.option("--tts-fasta", type=click.Path(exists=True), help="For 'other' species: TTS 1.5kb sequence file")
@click.option("--genome-fa", type=click.Path(exists=True), help="For 'other' species: reference genome FASTA")
@click.option("--gtf", type=click.Path(exists=True), help="For 'other' species: gene annotation file (GTF/GFF3)")
@click.option("--k", type=click.IntRange(5, 7), default=7, show_default=True, help="k-mer size (must be between 5 and 7)")
def build(species, label, features, out_dir, tss_fasta, tts_fasta, genome_fa, gtf, k):
    """
    Build the feature matrix and select top features, supporting user-supplied feature files.
    --label: CSV file, must have gene_id and label columns (label supports binary or multiclass).
    --features: If provided, will be merged with label for training; otherwise, features are auto-generated.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read label file, must contain gene_id and label columns
    label_df = pd.read_csv(label)
    if not {'gene_id', 'label'} <= set(label_df.columns):
        raise click.BadOptionUsage("--label", "Label file must contain columns: gene_id and label")
    label_df['gene_id'] = label_df['gene_id'].astype(str)

    if features:
        feature_df = pd.read_csv(features)
        if 'gene_id' not in feature_df.columns:
            raise click.BadOptionUsage("--features", "Feature file must contain a gene_id column")
        feature_df['gene_id'] = feature_df['gene_id'].astype(str)
        merged = pd.merge(label_df, feature_df, on='gene_id', how='inner')

        features_only = merged.drop(columns=['gene_id', 'label'])
        if features_only.shape[1] == 0:
            raise click.UsageError("No features found in the provided feature file.")
        correlations = features_only.corrwith(merged['label']).abs()
        top_n = max(1, int(len(merged) * 0.2))
        top_features = correlations.nlargest(top_n).index.tolist()

        final_df = merged[['gene_id'] + top_features + ['label']]
        final_df.to_csv(out_dir / "train_data_with_gene.csv", index=False)
        final_df.drop(columns=["gene_id"]).to_csv(out_dir / "train_data.csv", index=False)

        click.echo(f"Labels and features merged. Top {top_n} features saved to: train_data.csv and train_data_with_gene.csv in {out_dir}")
        return

    if not species:
        raise click.BadOptionUsage("--species", "If --features is not specified, --species is required.")

    if species in PRECOMPUTED_SPECIES:
        file_paths = get_species_file_paths(species)
        kmer_df_tss = pd.read_parquet(file_paths["kmer_tss"])
        kmer_df_tts = pd.read_parquet(file_paths["kmer_tts"])
        tf_df_tss = pd.read_parquet(file_paths["tf_tss"])
        tf_df_tts = pd.read_parquet(file_paths["tf_tts"])
    else:
        if not (tss_fasta and tts_fasta):
            if not (genome_fa and gtf):
                raise click.BadOptionUsage("--genome-fa", "For 'other' species, you must specify --genome-fa and --gtf")
            click.echo("Generating TSS and TTS sequences...")
            seqs = generate_tss_tts_sequences(Path(genome_fa), Path(gtf), out_dir)
            tss_fasta = seqs["tss_fasta"]
            tts_fasta = seqs["tts_fasta"]

        kmer_df_tss = build_kmer_matrix(Path(tss_fasta), k)
        kmer_df_tts = build_kmer_matrix(Path(tts_fasta), k)
        tf_df_tss = build_tf_matrix(Path(tss_fasta), MODEL_PATH, FEATURES_PATH)
        tf_df_tts = build_tf_matrix(Path(tts_fasta), MODEL_PATH, FEATURES_PATH)

    # Merge all auto-generated features
    seq_col_tss = "sequence" if "sequence" in kmer_df_tss.columns else "Sequence"
    seq_col_tts = "sequence" if "sequence" in kmer_df_tts.columns else "Sequence"
    kmer_df_tss = kmer_df_tss.rename(columns={seq_col_tss: "gene_id"})
    kmer_df_tts = kmer_df_tts.rename(columns={seq_col_tts: "gene_id"})
    kmer_df_tss = kmer_df_tss.rename(columns=lambda c: f"{c}_tss" if c != "gene_id" else c)
    kmer_df_tts = kmer_df_tts.rename(columns=lambda c: f"{c}_tts" if c != "gene_id" else c)
    kmer_df = pd.merge(kmer_df_tss, kmer_df_tts, on="gene_id", how="inner")

    tf_df_tss = tf_df_tss.rename(columns=lambda c: f"{c}_tss" if c != "gene_id" else c).reset_index()
    tf_df_tts = tf_df_tts.rename(columns=lambda c: f"{c}_tts" if c != "gene_id" else c).reset_index()
    tf_df = pd.merge(tf_df_tss, tf_df_tts, on="gene_id", how="inner")

    full_df = pd.merge(kmer_df, tf_df, on="gene_id", how="inner")
    full_df = pd.merge(label_df, full_df, on='gene_id', how='inner')
    label_col = full_df.pop('label')
    full_df['label'] = label_col

    features_only = full_df.drop(columns=['gene_id', 'label'])
    correlations = features_only.corrwith(full_df['label']).abs()
    top_n = max(1, int(len(full_df) * 0.2))
    top_features = correlations.nlargest(top_n).index.tolist()

    final_df = full_df[['gene_id'] + top_features + ['label']]
    final_df.to_csv(out_dir / "train_data_with_gene.csv", index=False)
    final_df.drop(columns=["gene_id"]).to_csv(out_dir / "train_data.csv", index=False)

    click.echo(f"Feature matrix generated. Top {top_n} features saved as: train_data.csv and train_data_with_gene.csv in {out_dir}")


@cli.command()
@click.option("-d", "--data", type=click.Path(exists=True), required=True)
@click.option("-o", "--out-dir", type=click.Path(), default="psd_output")
@click.option("--seeds", type=int, default=10, help="Number of random seeds to train with")
@click.option(
    "--ml-models", multiple=True, default=["xgboost", "nb", "lr", "gbc", "rf"],
    help="Subset of models to include. Example: --ml-models nb rf gbc; use 'all' to include all models."
)
def train(data, out_dir, seeds, ml_models):
    """Train multiple models and select the best one"""
    if len(ml_models) == 1 and ml_models[0] == "all":
        include_models = "all"
    else:
        include_models = list(ml_models)

    model_path = train_best_model(
        Path(data),
        target_col="label",
        out_dir=Path(out_dir),
        seeds=seeds,
        include_models=include_models
    )
    click.echo(f"Model training complete. Best model saved to: {model_path}")


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