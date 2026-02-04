from __future__ import annotations

import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import Counter
from itertools import product
from typing import Dict, List

import pandas as pd
from Bio import SeqIO

__all__ = ["build_kmer_matrix"]

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent
_CPP_SRC = _PKG_ROOT / "cpp" / "kmer_calculator.cpp"
_BIN_DIR = _PKG_ROOT / "bin"
_BIN_DIR.mkdir(exist_ok=True)
_CPP_EXE = _BIN_DIR / "kmer_calculator"

def _ensure_cpp_compiled() -> Path | None:
    """
    Check if the C++ binary exists; if not, compile it.
    On Windows, always return None (use Python fallback).
    """
    if sys.platform.startswith("win"):
        print("[plantsetdelta] Windows detected: Using pure Python k-mer calculation (no C++ acceleration).")
        return None

    if _CPP_EXE.exists():
        return _CPP_EXE

    if not _CPP_SRC.exists():
        return None

    print("[plantsetdelta] Compiling C++ accelerator for the first time...")
    cmd = ["g++", "-O3", "-std=c++17", "-pthread", "-o", str(_CPP_EXE), str(_CPP_SRC)]
    try:
        subprocess.check_call(cmd)
        print("[plantsetdelta] Compilation succeeded:", _CPP_EXE)
        return _CPP_EXE
    except Exception:
        print("[plantsetdelta] Compilation failed. Falling back to Python implementation.")
        return None

# ------------------------------------------------------------
# Python fallback
# ------------------------------------------------------------
def _all_kmers(k: int) -> List[str]:
    return ["".join(p) for p in product("ACGT", repeat=k)]

def _kmer_counts_py(seq: str, k: int) -> Dict[str, int]:
    seq = seq.upper().replace("N", "")
    return Counter(seq[i: i + k] for i in range(len(seq) - k + 1))

# ------------------------------------------------------------
# Main function: build k-mer feature matrix
# ------------------------------------------------------------
def build_kmer_matrix(fasta: Path | str, k: int = 7, threads: int = 4) -> pd.DataFrame:
    """
    Build a k-mer frequency matrix (n_sequences × 4^k).

    Args:
        fasta (Path or str): Input FASTA file path
        k (int): k-mer size (must be between 4 and 7)
        threads (int): Number of threads for C++ acceleration (ignored for Python fallback)

    Returns:
        pd.DataFrame: DataFrame with one row per sequence, columns: [sequence, AAA..., TTT...]
    """
    if not 4 <= k <= 7:
        raise ValueError("k must be between 4 and 7")

    fasta = Path(fasta)

    # Try C++ implementation first (Linux/macOS only)
    cpp_exe = _ensure_cpp_compiled()
    if cpp_exe:
        with tempfile.TemporaryDirectory() as tmpd:
            prefix = Path(tmpd) / "out"
            cmd = [
                str(cpp_exe),
                "-i", str(fasta),
                "-o", str(prefix),
                "-k", str(k),
                "-m", "1",
                "-t", str(threads)
            ]
            subprocess.check_call(cmd)
            result_file = Path(f"{prefix}_kmer_counts.txt")

            df_raw = pd.read_csv(result_file, sep="\t")  # Columns: Sequence, K, KMer, Count
            df = (
                df_raw.pivot_table(index="Sequence", columns="KMer", values="Count", fill_value=0, aggfunc="sum")
                .astype(int)
                .reindex(columns=_all_kmers(k), fill_value=0)
                .reset_index()
                .rename(columns={"Sequence": "sequence"})
            )
            return df

    # Python fallback
    print("[plantsetdelta] Using pure Python k-mer calculation (slower, cross-platform).")
    records = list(SeqIO.parse(str(fasta), "fasta"))
    kmers = _all_kmers(k)
    data = []

    for rec in records:
        counts = _kmer_counts_py(str(rec.seq), k)
        row = [rec.id] + [counts.get(km, 0) for km in kmers]
        data.append(row)

    col_names = ["sequence"] + kmers
    return pd.DataFrame(data, columns=col_names)

# ------------------------------------------------------------
# CLI quick test
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Test k-mer matrix builder")
    p.add_argument("fasta", help="Path to FASTA file")
    p.add_argument("-k", type=int, default=7, help="k-mer size (4–7)")
    args = p.parse_args()

    mat = build_kmer_matrix(args.fasta, k=args.k)
    print(mat.head())