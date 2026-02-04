from __future__ import annotations

from pathlib import Path
import tempfile
import pandas as pd
from typing import List, Optional

from Bio import SeqIO
from selene_sdk.predict import AnalyzeSequences
from selene_sdk.utils import DeeperDeepSEA

__all__ = ["build_tf_matrix"]


def _load_features(feature_file: Path) -> List[str]:
    """Load TF feature names from file."""
    with open(feature_file) as f:
        return [line.strip() for line in f if line.strip()]


def _preprocess_sequence(seq: str, expected_length: int = 1500, bin_size: int = 100) -> List[str]:
    """
    Split a fixed-length sequence into bins.

    Default: 1500bp -> 15 bins of 100bp each.
    """
    if len(seq) != expected_length:
        raise ValueError(f"Input sequence must be {expected_length} bp long.")
    if expected_length % bin_size != 0:
        raise ValueError("expected_length must be divisible by bin_size.")
    n_bins = expected_length // bin_size
    return [seq[i * bin_size : (i + 1) * bin_size] for i in range(n_bins)]


def _find_selene_tsv(tmp_dir: Path) -> Path:
    """
    Selene output tsv filename can vary; robustly find it.
    """
    # common pattern: *_predictions.tsv
    tsvs = list(tmp_dir.glob("*predictions.tsv"))
    if tsvs:
        return tsvs[0]
    # fallback: any .tsv
    tsvs = list(tmp_dir.glob("*.tsv"))
    if tsvs:
        return tsvs[0]
    raise FileNotFoundError("Selene prediction output TSV not found.")


def build_tf_matrix(
    fasta: Path,
    model_path: Path,
    feature_file: Optional[Path] = None,
    sequence_length: int = 1000,
    batch_size: int = 64,
    use_cuda: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Predict TF-binding using DeeperDeepSEA and generate a (n_genes × TF × bin) feature matrix
    for fixed 1500bp sequences (15 bins of 100bp).

    Returns:
        pd.DataFrame: index=gene_id, columns like "{TF}_bin_{i}"
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        temp_fasta = tmp_dir / "temp_bins.fa"

        if debug:
            print("Generating binned sequences...")

        # Step 1: Convert to binned FASTA
        with open(temp_fasta, "w") as f:
            for record in SeqIO.parse(str(fasta), "fasta"):
                seq = str(record.seq).upper()
                try:
                    bins = _preprocess_sequence(seq, expected_length=1500, bin_size=100)
                    for i, bin_seq in enumerate(bins, 1):
                        f.write(f">{record.id}_bin_{i}\n{bin_seq}\n")
                except ValueError as e:
                    if debug:
                        print(f"Skipping {record.id}: {str(e)}")
                    continue

        # Step 2: Load model
        features = _load_features(feature_file) if feature_file else None
        model = DeeperDeepSEA(
            sequence_length=sequence_length,
            n_targets=len(features) if features else None,
        )

        analyzer = AnalyzeSequences(
            model=model,
            trained_model_path=str(model_path),
            sequence_length=sequence_length,
            features=features,
            batch_size=batch_size,
            use_cuda=use_cuda,
        )

        if debug:
            print("Running Selene predictions...")

        # Step 3: Predict
        analyzer.get_predictions_for_fasta_file(
            input_path=str(temp_fasta),
            output_dir=str(tmp_dir),
            output_format="tsv",
        )

        # Step 4: Load prediction results
        result_file = _find_selene_tsv(tmp_dir)

        # robust read
        df = pd.read_csv(result_file, sep="\t")

        # Determine IDs column (sequence name)
        id_series = None
        id_col = None

        if "name" in df.columns:
            id_col = "name"
            id_series = df["name"].astype(str)
        else:
            # sometimes first column is the id column (e.g. unnamed)
            first_col = df.columns[0]
            if df[first_col].dtype == object:
                id_col = first_col
                id_series = df[first_col].astype(str)
            else:
                # fallback: try read as index
                df2 = pd.read_csv(result_file, sep="\t", index_col=0)
                id_series = df2.index.astype(str)
                df = df2.reset_index().rename(columns={"index": "name"})
                id_col = "name"
                id_series = df["name"].astype(str)

        # Parse gene_id and bin from ids
        parsed = id_series.str.extract(r"(.*)_bin_(\d+)$")
        df["gene_id"] = parsed[0]
        df["bin"] = parsed[1].astype(float)

        df = df.dropna(subset=["gene_id", "bin"])
        df["bin"] = df["bin"].astype(int)

        # Feature columns
        exclude = {id_col, "gene_id", "bin"}
        feature_cols = [c for c in df.columns if c not in exclude]

        # Step 5: Pivot to TF_bin columns
        long = df.set_index(["gene_id", "bin"])[feature_cols]
        wide = long.unstack("bin")  # MultiIndex columns (feature, bin)
        wide.columns = [f"{tf}_bin_{b}" for tf, b in wide.columns]
        wide.index.name = "gene_id"

        if debug:
            print("TF-binding prediction complete. Matrix shape:", wide.shape)

        return wide