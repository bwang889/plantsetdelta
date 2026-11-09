from __future__ import annotations

import sys
import subprocess
import tempfile
from pathlib import Path
from collections import Counter
from itertools import product
from typing import Dict, List

import pandas as pd
from Bio import SeqIO

__all__ = ["build_kmer_matrix"]

_PKG_ROOT = Path(__file__).resolve().parent.parent
_CPP_SRC = _PKG_ROOT / "cpp" / "kmer_calculator.cpp"
_BIN_DIR = Path.home() / ".cache" / "plantsetdelta" / "bin"
_CPP_EXE = _BIN_DIR / "kmer_calculator"


def _ensure_cpp_compiled() -> Path | None:
    """Compile the optional C++ accelerator into a user-writable cache."""
    if sys.platform.startswith("win"):
        print("[plantsetdelta] Windows detected: using the Python k-mer backend.")
        return None
    if not _CPP_SRC.exists():
        return None
    if _CPP_EXE.exists() and _CPP_EXE.stat().st_mtime >= _CPP_SRC.stat().st_mtime:
        return _CPP_EXE

    compiler = "g++"
    try:
        _BIN_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [compiler, "-O3", "-std=c++17", "-pthread", "-o", str(_CPP_EXE), str(_CPP_SRC)]
        )
        print("[plantsetdelta] C++ k-mer accelerator compiled:", _CPP_EXE)
        return _CPP_EXE
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[plantsetdelta] C++ compilation unavailable ({exc}); using Python backend.")
        return None


def _all_kmers(k: int) -> List[str]:
    return ["".join(p) for p in product("ACGT", repeat=k)]


def _kmer_counts_py(seq: str, k: int) -> Dict[str, int]:
    """Count only windows composed entirely of A/C/G/T; ambiguity breaks a window."""
    seq = seq.upper()
    return Counter(
        window for i in range(len(seq) - k + 1)
        if set(window := seq[i:i + k]) <= {"A", "C", "G", "T"} and len(window) == k
    )


def build_kmer_matrix(fasta: Path | str, k: int = 7, threads: int = 4) -> pd.DataFrame:
    """
    Build a raw-count k-mer matrix (one row per FASTA record, 4**k columns).
    FASTA identifiers follow Biopython's record.id convention (first header token).
    """
    if not 4 <= k <= 7:
        raise ValueError("k must be between 4 and 7")
    fasta = Path(fasta)
    records = list(SeqIO.parse(str(fasta), "fasta"))
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("FASTA contains duplicate record IDs; gene IDs must be unique.")

    kmers = _all_kmers(k)
    cpp_exe = _ensure_cpp_compiled()
    if cpp_exe is not None:
        with tempfile.TemporaryDirectory() as tmpd:
            prefix = Path(tmpd) / "out"
            cmd = [str(cpp_exe), "-i", str(fasta), "-o", str(prefix),
                   "-k", str(k), "-m", "1", "-t", str(max(1, int(threads)))]
            subprocess.check_call(cmd)
            result_file = Path(f"{prefix}_kmer_counts.txt")
            df_raw = pd.read_csv(result_file, sep="\t")
            if df_raw.empty:
                matrix = pd.DataFrame(index=pd.Index([], name="sequence"), columns=kmers)
            else:
                matrix = df_raw.pivot_table(
                    index="Sequence", columns="KMer", values="Count",
                    fill_value=0, aggfunc="sum"
                )
            matrix = matrix.reindex(index=ids, columns=kmers, fill_value=0).fillna(0).astype(int)
            matrix.index.name = "sequence"
            return matrix.reset_index()

    print("[plantsetdelta] Using pure Python k-mer calculation.")
    data = []
    for record in records:
        counts = _kmer_counts_py(str(record.seq), k)
        data.append([record.id] + [counts.get(km, 0) for km in kmers])
    return pd.DataFrame(data, columns=["sequence"] + kmers)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test k-mer matrix builder")
    parser.add_argument("fasta", help="Path to input FASTA")
    parser.add_argument("-k", type=int, default=7, help="k-mer size (4–7)")
    args = parser.parse_args()
    print(build_kmer_matrix(args.fasta, k=args.k).head())
