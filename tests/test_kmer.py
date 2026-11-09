from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from plantsetdelta.feature_engineering import kmer


def test_python_kmer_ambiguity_breaks_windows():
    counts = kmer._kmer_counts_py("AAAAANCCCCC", 5)
    assert counts == {"AAAAA": 1, "CCCCC": 1}


def test_cpp_and_python_backends_match(tmp_path, monkeypatch):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is not available")
    source = Path(kmer.__file__).resolve().parents[1] / "cpp" / "kmer_calculator.cpp"
    executable = tmp_path / "kmer_calculator"
    subprocess.check_call([compiler, "-O2", "-std=c++17", "-pthread", "-o", str(executable), str(source)])

    fasta = tmp_path / "test.fa"
    fasta.write_text(
        ">geneA description\nAAAAANCCCCC\n>geneB\nNNNNN\n>geneC\nACGTACGT\n>geneD empty\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kmer, "_ensure_cpp_compiled", lambda: None)
    python_df = kmer.build_kmer_matrix(fasta, k=5).set_index("sequence").sort_index()
    monkeypatch.setattr(kmer, "_ensure_cpp_compiled", lambda: executable)
    cpp_df = kmer.build_kmer_matrix(fasta, k=5).set_index("sequence").sort_index()
    pd.testing.assert_frame_equal(python_df, cpp_df)


def test_duplicate_fasta_ids_are_rejected(tmp_path):
    fasta = tmp_path / "duplicates.fa"
    fasta.write_text(">gene1 first\nAAAAA\n>gene1 second\nCCCCC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        kmer.build_kmer_matrix(fasta, k=5)
