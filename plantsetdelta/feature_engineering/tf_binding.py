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


def _preprocess_sequence(seq: str, expected_length: int = 1500) -> List[str]:
    """
    Split a 1500-bp sequence into 15 bins of 100bp each.
    
    Args:
        seq (str): Input DNA sequence
        expected_length (int): Required input sequence length (default: 1500)

    Returns:
        List[str]: List of 100-bp subsequences
    """
    if len(seq) != expected_length:
        raise ValueError(f"Input sequence must be {expected_length} bp long.")
    return [seq[i * 100: (i + 1) * 100] for i in range(15)]


def build_tf_matrix(
    fasta: Path,
    model_path: Path,
    feature_file: Optional[Path] = None,
    sequence_length: int = 1000,
    batch_size: int = 64,
    use_cuda: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    """
    Predict TF-binding using DeeperDeepSEA and generate a (n_genes × TF × bin) feature matrix.

    Args:
        fasta (Path): Input FASTA file
        model_path (Path): Pretrained model (.pth.tar)
        feature_file (Optional[Path]): File containing list of TF names
        sequence_length (int): Sequence length expected by the model (default: 1000)
        batch_size (int): Batch size for Selene predictions
        use_cuda (bool): Whether to use GPU acceleration
        debug (bool): If True, print intermediate information

    Returns:
        pd.DataFrame: Gene × TF_bin feature matrix
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        temp_fasta = tmp_dir / "temp_bins.fa"

        if debug:
            print("Generating binned sequences...")

        # Step 1: Convert to binned FASTA
        with open(temp_fasta, "w") as f:
            for record in SeqIO.parse(fasta, "fasta"):
                seq = str(record.seq).upper()
                try:
                    bins = _preprocess_sequence(seq)
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
            n_targets=len(features) if features else None
        )

        analyzer = AnalyzeSequences(
            model=model,
            trained_model_path=str(model_path),
            sequence_length=sequence_length,
            features=features,
            batch_size=batch_size,
            use_cuda=use_cuda
        )

        if debug:
            print("Running Selene predictions...")

        # Step 3: Predict
        analyzer.get_predictions_for_fasta_file(
            input_path=str(temp_fasta),
            output_dir=str(tmp_dir),
            output_format="tsv"
        )

        result_file = tmp_dir / "temp_bins_predictions.tsv"
        if not result_file.exists():
            raise FileNotFoundError("Selene prediction output not found.")

        # Step 4: Load prediction results
        df = pd.read_csv(result_file, sep="\t", index_col=0)

        id_column = 'name' if 'name' in df.columns else df.columns[0]
        df[id_column] = df[id_column].astype(str)
        df[['gene_id', 'bin']] = df[id_column].str.extract(r'(.*)_bin_(\d+)')
        df['bin'] = df['bin'].astype(int)

        # Step 5: Pivot to TF_bin columns
        feature_cols = [c for c in df.columns if c not in [id_column, 'gene_id', 'bin', 'index']]
        pivot_df = df.pivot(index='gene_id', columns='bin', values=feature_cols)
        pivot_df.columns = [f"{tf}_bin_{b}" for tf, b in pivot_df.columns]

        if debug:
            print("TF-binding prediction complete. Matrix shape:", pivot_df.shape)

        return pivot_df