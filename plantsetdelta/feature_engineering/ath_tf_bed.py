from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Set
import pandas as pd
import numpy as np
from tqdm import tqdm

# requires pybedtools + external bedtools installed
import pybedtools


def _parse_gff_genes_min(gff_file: Path) -> List[dict]:
    genes = []
    with open(gff_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2].lower() != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attr = fields
            gene_id = None
            for part in attr.split(";"):
                part = part.strip()
                if part.startswith("ID="):
                    gene_id = part.split("=", 1)[-1].split(":")[-1]
                    break
            if not gene_id:
                for part in attr.split(";"):
                    if "gene_id" in part:
                        toks = part.strip().split()
                        if len(toks) >= 2:
                            gene_id = toks[1].replace('"', "").replace(";", "")
                            break
            if gene_id:
                genes.append(
                    {
                        "gene_id": str(gene_id),
                        "chrom": chrom,
                        "start": int(start),
                        "end": int(end),
                        "strand": strand,
                    }
                )
    return genes


def _build_bins_for_gene(
    gene: dict,
    upstream: int,
    downstream: int,
    anchor: int,
    strand: str,
    bin_size: int,
    chrom_length: int | None = None,
) -> List[Tuple[str, int, int, str]]:
    """
    Create bins (BED intervals) for one region around an anchor.
    Returns list of (chrom, start0, end0, name) where start0 is 0-based start for BED.
    """
    length = upstream + downstream
    n_bins = length // bin_size

    if strand == "+":
        region_start_1based = anchor - upstream
        region_end_1based = anchor + downstream - 1
    else:
        # on minus strand, upstream relative to gene direction corresponds to genomic downstream
        # region in genome coords is [anchor - downstream, anchor + upstream]
        region_start_1based = anchor - downstream
        region_end_1based = anchor + upstream - 1

    # Match the FASTA extractor: skip a whole window if it crosses either
    # chromosome boundary rather than creating partial BED bins.
    if region_start_1based < 1 or (chrom_length is not None and region_end_1based > chrom_length):
        return []

    # BED intervals are 0-based, half-open.
    region_start0 = region_start_1based - 1

    bins = []
    for i in range(n_bins):
        b0 = region_start0 + i * bin_size
        b1 = b0 + bin_size
        # Number bins in transcript orientation (the FASTA is reverse-complemented
        # for minus-strand genes).
        bin_number = (n_bins - i) if strand == "-" else (i + 1)
        name = f"{gene['gene_id']}_bin_{bin_number}"
        bins.append((gene["chrom"], b0, b1, name))
    return bins


def build_ath_tf_matrix_from_bed(
    gff_file: Path,
    tf_bed_file: Path,
    tss_upstream: int,
    tss_downstream: int,
    tts_upstream: int,
    tts_downstream: int,
    bin_size: int = 100,
    min_frac_overlap: float = 0.8,
    batch_bins: int = 10000,
    genome_fa: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if (tss_upstream % bin_size) or (tss_downstream % bin_size) or (tts_upstream % bin_size) or (tts_downstream % bin_size):
        raise ValueError("All interval params must be multiples of bin_size (default 100).")

    genes = _parse_gff_genes_min(gff_file)
    chrom_lengths = None
    if genome_fa is not None:
        from Bio import SeqIO
        chrom_lengths = {record.id: len(record.seq) for record in SeqIO.parse(str(genome_fa), "fasta")}
    gene_ids = [g["gene_id"] for g in genes]
    if len(gene_ids) != len(set(gene_ids)):
        raise ValueError("GFF/GTF contains duplicate gene IDs; TF bins require unique gene IDs.")
    invalid_strands = sorted({g["strand"] for g in genes if g["strand"] not in {"+", "-"}})
    if invalid_strands:
        raise ValueError(f"Unsupported gene strand values: {invalid_strands}")
    if batch_bins < 1:
        raise ValueError("batch_bins must be >= 1")
    if not 0 <= min_frac_overlap <= 1:
        raise ValueError("min_frac_overlap must be between 0 and 1")
    tf_bed = pybedtools.BedTool(str(tf_bed_file))

    # Prepare TSS/TTS bins as BED lines
    tss_bins = []
    tts_bins = []
    for g in genes:
        strand = g["strand"]
        if strand == "+":
            tss_anchor = g["start"]
            tts_anchor = g["end"]
            chrom_length = chrom_lengths.get(g["chrom"]) if chrom_lengths is not None else None
            tss_bins.extend(_build_bins_for_gene(g, tss_upstream, tss_downstream, tss_anchor, "+", bin_size, chrom_length))
            tts_bins.extend(_build_bins_for_gene(g, tts_upstream, tts_downstream, tts_anchor, "+", bin_size, chrom_length))
        else:
            tss_anchor = g["end"]
            tts_anchor = g["start"]
            # minus strand uses swapped logic inside _build_bins_for_gene
            chrom_length = chrom_lengths.get(g["chrom"]) if chrom_lengths is not None else None
            tss_bins.extend(_build_bins_for_gene(g, tss_upstream, tss_downstream, tss_anchor, "-", bin_size, chrom_length))
            tts_bins.extend(_build_bins_for_gene(g, tts_upstream, tts_downstream, tts_anchor, "-", bin_size, chrom_length))

    def _compute_matrix(bins: List[Tuple[str, int, int, str]], label: str) -> pd.DataFrame:
        intersect_data: Set[Tuple[str, str, int]] = set()  # (gene_id, tf_name, bin_num)
        n_batches = int(np.ceil(len(bins) / batch_bins))

        for bi in tqdm(range(n_batches), desc=f"Intersect {label} bins"):
            chunk = bins[bi * batch_bins : (bi + 1) * batch_bins]
            bins_bed = pybedtools.BedTool(
                [f"{chrom}\t{start}\t{end}\t{name}" for chrom, start, end, name in chunk]
            )
            # wa wb; f=0.8 means overlap fraction of A? (bins) satisfied
            inter = bins_bed.intersect(tf_bed, wa=True, wb=True, f=min_frac_overlap)

            for line in inter:
                # bins: 4 cols, tf bed: assume >=4 cols, tf name at col 7 (0-based index)
                # if TF bed has extra cols, tf name might not be at 7; we take last col as tf name fallback
                bin_name = line[3]
                tf_name = line[7] if len(line) >= 8 else line[-1]
                gene_id, bin_num = bin_name.split("_bin_")
                intersect_data.add((gene_id, tf_name, int(bin_num)))

        if not intersect_data:
            # return empty but with gene_id index if possible
            return pd.DataFrame(index=sorted({g["gene_id"] for g in genes}))

        df = pd.DataFrame(list(intersect_data), columns=["gene_id", "tf_type", "bin_num"])
        mat = df.pivot_table(index="gene_id", columns=["tf_type", "bin_num"], aggfunc="size", fill_value=0)
        mat.columns = [f"{tf}_bin_{bn}" for tf, bn in mat.columns]
        mat = mat.applymap(lambda x: 1 if x > 0 else 0)
        # ensure all genes exist
        all_gene_ids = sorted({g["gene_id"] for g in genes})
        mat = mat.reindex(all_gene_ids).fillna(0).astype(int)
        return mat

    tss_df = _compute_matrix(tss_bins, "TSS")
    tts_df = _compute_matrix(tts_bins, "TTS")
    return tss_df, tts_df