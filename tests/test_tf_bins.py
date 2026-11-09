import importlib.util
from pathlib import Path
import pytest

pytest.importorskip("pybedtools")
MODULE_PATH = Path(__file__).resolve().parents[1] / "plantsetdelta" / "feature_engineering" / "ath_tf_bed.py"
SPEC = importlib.util.spec_from_file_location("plantsetdelta_ath_tf_bed_test", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
_build_bins_for_gene = module._build_bins_for_gene


def test_minus_strand_bins_are_numbered_in_transcript_orientation():
    gene = {"gene_id": "minus1", "chrom": "chr1"}
    bins = _build_bins_for_gene(
        gene, upstream=100, downstream=100, anchor=1000,
        strand="-", bin_size=100, chrom_length=2000,
    )
    assert [item[3] for item in bins] == ["minus1_bin_2", "minus1_bin_1"]


def test_bed_window_crossing_chromosome_boundary_is_skipped():
    gene = {"gene_id": "edge1", "chrom": "chr1"}
    bins = _build_bins_for_gene(
        gene, upstream=200, downstream=100, anchor=100,
        strand="+", bin_size=100, chrom_length=1000,
    )
    assert bins == []
