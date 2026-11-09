from Bio.Seq import Seq

from plantsetdelta.utils.extract_sequences import (
    _extract_window,
    extract_tss_tts_sequences,
)


def test_window_respects_chromosome_boundaries():
    genome = {"chr1": type("Record", (), {"seq": Seq("ACGTACGTAA")})()}
    region = {"chrom": "chr1"}
    assert _extract_window(region, genome, 2, 3, 1, "+") == ""
    assert _extract_window(region, genome, 8, 1, 4, "+") == ""


def test_negative_strand_windows_are_reverse_complemented():
    genome = {"chr1": type("Record", (), {"seq": Seq("AACCGGTTAACC")})()}
    gene = {"chrom": "chr1", "start": 4, "end": 8, "strand": "-"}
    tss, tts = extract_tss_tts_sequences(
        gene, genome, tss_upstream=2, tss_downstream=1,
        tts_upstream=1, tts_downstream=2,
    )
    assert tss == str(Seq("TTA").reverse_complement())
    assert tts == str(Seq("ACC").reverse_complement())
