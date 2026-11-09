from __future__ import annotations

from pathlib import Path
import tempfile
from typing import List

import pandas as pd
from Bio import SeqIO
from selene_sdk.predict import AnalyzeSequences
from selene_sdk.utils import DeeperDeepSEA


def _load_features(feature_file: Path) -> List[str]:
    with open(feature_file, encoding="utf-8") as handle:
        features = [line.strip() for line in handle if line.strip()]
    if not features:
        raise ValueError(f"No TF features found in {feature_file}")
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate TF feature names in {feature_file}")
    return features


def _split_bins(seq: str, bin_size: int = 100) -> List[str]:
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")
    if len(seq) % bin_size != 0:
        raise ValueError("Sequence length must be divisible by bin_size.")
    return [seq[i:i + bin_size] for i in range(0, len(seq), bin_size)]


def _read_prediction_table(result_file: Path) -> pd.DataFrame:
    df = pd.read_csv(result_file, sep="\t")
    if df.empty:
        raise ValueError("Selene prediction table is empty.")
    id_candidates = []
    for column in df.columns:
        values = df[column].astype(str)
        if values.str.match(r".+_bin_\d+$").all():
            id_candidates.append(column)
    if "name" in id_candidates:
        id_column = "name"
    elif id_candidates:
        id_column = id_candidates[0]
    else:
        raise ValueError(
            "Cannot identify sequence-name column in Selene output. "
            f"Columns: {list(df.columns)}"
        )
    df = df.rename(columns={id_column: "sequence_id"})
    parsed = df["sequence_id"].astype(str).str.extract(r"^(.*)_bin_(\d+)$")
    if parsed.isna().any().any():
        raise ValueError("Malformed sequence IDs in Selene prediction output.")
    df["gene_id"] = parsed[0]
    df["bin"] = parsed[1].astype(int)
    return df


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
        input_records = 0
        with open(temp_fasta, "w", encoding="utf-8") as handle:
            for record in SeqIO.parse(str(fasta), "fasta"):
                input_records += 1
                for i, bin_seq in enumerate(_split_bins(str(record.seq).upper(), bin_size), 1):
                    handle.write(f">{record.id}_bin_{i}\n{bin_seq}\n")

        if input_records == 0:
            raise ValueError(f"No FASTA records found in {fasta}")
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

        tsvs = sorted(tmp_dir.glob("*predictions.tsv"))
        if len(tsvs) != 1:
            raise FileNotFoundError(
                f"Expected one Selene predictions TSV, found {len(tsvs)} in {tmp_dir}"
            )
        df = _read_prediction_table(tsvs[0])
        feature_cols = [c for c in df.columns if c not in {"sequence_id", "gene_id", "bin"}]
        if not feature_cols:
            raise ValueError("Selene output contains no TF prediction columns.")
        if df.duplicated(["gene_id", "bin"]).any():
            raise ValueError("Selene output contains duplicate gene_id/bin predictions.")
        wide = df.set_index(["gene_id", "bin"])[feature_cols].unstack("bin")
        wide.columns = [f"{tf}_bin_{bin_number}" for tf, bin_number in wide.columns]
        wide.index.name = "gene_id"
        return wide
