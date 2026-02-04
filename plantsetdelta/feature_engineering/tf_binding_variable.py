from __future__ import annotations

from pathlib import Path
import tempfile
from typing import List, Optional

import pandas as pd
from Bio import SeqIO
from selene_sdk.predict import AnalyzeSequences
from selene_sdk.utils import DeeperDeepSEA


def _load_features(feature_file: Path) -> List[str]:
    with open(feature_file) as f:
        return [line.strip() for line in f if line.strip()]


def _split_bins(seq: str, bin_size: int = 100) -> List[str]:
    if len(seq) % bin_size != 0:
        raise ValueError("Sequence length must be divisible by bin_size.")
    n = len(seq) // bin_size
    return [seq[i * bin_size : (i + 1) * bin_size] for i in range(n)]


def build_tf_matrix_variable_bins(
    fasta: Path,
    model_path: Path,
    feature_file: Path,
    bin_size: int = 100,
    sequence_length: int = 1000,
    batch_size: int = 64,
    use_cuda: bool = False,
    debug: bool = False,
) -> pd.DataFrame:

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        temp_fasta = tmp_dir / "temp_bins.fa"

        # Step 1: binned fasta
        with open(temp_fasta, "w") as f:
            for record in SeqIO.parse(str(fasta), "fasta"):
                seq = str(record.seq).upper()
                try:
                    bins = _split_bins(seq, bin_size=bin_size)
                    for i, bseq in enumerate(bins, 1):
                        f.write(f">{record.id}_bin_{i}\n{bseq}\n")
                except Exception as e:
                    if debug:
                        print(f"Skipping {record.id}: {e}")
                    continue

        features = _load_features(feature_file)
        model = DeeperDeepSEA(sequence_length=sequence_length, n_targets=len(features))

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

        analyzer.get_predictions_for_fasta_file(
            input_path=str(temp_fasta),
            output_dir=str(tmp_dir),
            output_format="tsv",
        )

        # Selene output name can vary; find the tsv
        tsvs = list(tmp_dir.glob("*predictions.tsv"))
        if not tsvs:
            raise FileNotFoundError("Selene prediction output not found in temp dir.")
        result_file = tsvs[0]

        df = pd.read_csv(result_file, sep="\t", index_col=0)

        id_column = "name" if "name" in df.columns else df.columns[0]
        df[id_column] = df[id_column].astype(str)

        tmp = df[id_column].str.extract(r"(.*)_bin_(\d+)")
        df["gene_id"] = tmp[0]
        df["bin"] = tmp[1].astype(int)

        feature_cols = [c for c in df.columns if c not in [id_column, "gene_id", "bin", "index"]]
        pivot_df = df.pivot(index="gene_id", columns="bin", values=feature_cols)
        pivot_df.columns = [f"{tf}_bin_{b}" for tf, b in pivot_df.columns]
        return pivot_df