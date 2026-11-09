from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO


def parse_gff_genes(gff_file: Path) -> List[dict]:
    """
    Parse gene features from a GFF3/GTF annotation file.
    Return list of dicts with chrom, start, end, strand, gene_id.
    """
    genes = []
    with open(gff_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            if fields[2].lower() != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attr = fields
            gene_id = None

            # GFF3: ID=gene:XXX or ID=XXX
            for part in attr.split(";"):
                part = part.strip()
                if part.startswith("ID="):
                    gene_id = part.split("=", 1)[-1].split(":")[-1]
                    break

            # GTF style
            if not gene_id:
                for part in attr.split(";"):
                    if "gene_id" in part:
                        toks = part.strip().split()
                        if len(toks) >= 2:
                            gene_id = toks[1].replace('"', "").replace(";", "")
                            break

            if gene_id:
                if strand not in {"+", "-"}:
                    raise ValueError(f"Gene {gene_id} has unsupported strand: {strand}")
                start_i, end_i = int(start), int(end)
                if start_i < 1 or end_i < start_i:
                    raise ValueError(f"Gene {gene_id} has invalid coordinates: {start}-{end}")
                genes.append(
                    {
                        "chrom": chrom,
                        "start": start_i,
                        "end": end_i,
                        "strand": strand,
                        "gene_id": str(gene_id),
                    }
                )
    gene_ids = [g["gene_id"] for g in genes]
    if len(gene_ids) != len(set(gene_ids)):
        raise ValueError("Annotation contains duplicate gene IDs.")
    return genes


def _extract_window(
    region: dict,
    genome_dict: dict,
    anchor_pos: int,
    upstream: int,
    downstream: int,
    strand: str,
) -> str:
    """
    Extract [anchor-upstream, anchor+downstream) on + strand.
    For - strand, caller should pass correct anchor_pos and upstream/downstream already swapped if needed.
    """
    chrom = region["chrom"]
    seq = genome_dict[chrom].seq

    # convert 1-based anchor_pos to 0-based slice
    start0 = anchor_pos - upstream - 1
    end0 = anchor_pos + downstream - 1  # 0-based exclusive end
    if start0 < 0 or end0 > len(seq) or end0 < start0:
        return ""
    frag = seq[start0:end0]

    if strand == "-":
        frag = frag.reverse_complement()

    return str(frag).upper()


def extract_tss_tts_sequences(
    gene: dict,
    genome_dict: dict,
    tss_upstream: int,
    tss_downstream: int,
    tts_upstream: int,
    tts_downstream: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a gene region, extract TSS and TTS windows with user-defined lengths.
    Handles strand and returns uppercase sequences.

    Returns:
      (tss_seq, tts_seq) where either can be None if its expected length is 0.
    """
    strand = gene["strand"]
    if strand not in {"+", "-"}:
        raise ValueError(f"Unsupported gene strand: {strand}")
    start = gene["start"]
    end = gene["end"]

    expected_tss_len = tss_upstream + tss_downstream
    expected_tts_len = tts_upstream + tts_downstream

    tss_seq: Optional[str] = None
    tts_seq: Optional[str] = None

    if strand == "+":
        tss_anchor = start
        tts_anchor = end
        if expected_tss_len > 0:
            tss_seq = _extract_window(gene, genome_dict, tss_anchor, tss_upstream, tss_downstream, strand="+")
        if expected_tts_len > 0:
            tts_seq = _extract_window(gene, genome_dict, tts_anchor, tts_upstream, tts_downstream, strand="+")
    else:
        # minus strand: TSS at end, TTS at start
        tss_anchor = end
        tts_anchor = start
        # For minus strand, interpret upstream/downstream relative to gene direction.
        # Extract [anchor - downstream, anchor + upstream] in genome coords, then reverse-complement.
        if expected_tss_len > 0:
            tss_seq = _extract_window(gene, genome_dict, tss_anchor, tss_downstream, tss_upstream, strand="-")
        if expected_tts_len > 0:
            tts_seq = _extract_window(gene, genome_dict, tts_anchor, tts_downstream, tts_upstream, strand="-")

    return tss_seq, tts_seq


def generate_tss_tts_sequences(
    genome_fa: Path,
    gff_file: Path,
    output_dir: Path,
    tss_upstream: int = 1000,
    tss_downstream: int = 500,
    tts_upstream: int = 500,
    tts_downstream: int = 1000,
    gene_id_whitelist: Optional[set[str]] = None,
) -> Dict[str, Optional[Path]]:
    """
    Pure Python: Extract TSS/TTS sequences from annotation and genome.

    - Output fasta headers are gene_id.
    - Sequences are uppercase.
    - If sequence hits chromosome boundary and length is shorter than expected, it is skipped
      (保持与原实现一致：只保留长度严格等于期望长度的条目)。
    - If gene_id_whitelist is provided, only those genes are processed to reduce I/O and memory.

    Special:
      - If expected_tss_len == 0, TSS extraction is disabled and tss_fasta=None
      - If expected_tts_len == 0, TTS extraction is disabled and tts_fasta=None

    Returns dict:
      {"tss_fasta": Path|None, "tts_fasta": Path|None}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_tss_len = tss_upstream + tss_downstream
    expected_tts_len = tts_upstream + tts_downstream

    tss_fasta: Optional[Path] = output_dir / "tss_final.fa" if expected_tss_len > 0 else None
    tts_fasta: Optional[Path] = output_dir / "tts_final.fa" if expected_tts_len > 0 else None

    genes = parse_gff_genes(gff_file)
    genome_dict = SeqIO.to_dict(SeqIO.parse(str(genome_fa), "fasta"))

    written_tss, written_tts = set(), set()
    skipped_missing_chrom = 0
    skipped_boundary = 0
    skipped_invalid = 0

    # Open only what we need
    tss_handle = open(tss_fasta, "w") if tss_fasta is not None else None
    tts_handle = open(tts_fasta, "w") if tts_fasta is not None else None

    try:
        for g in genes:
            try:
                # Early filter: skip genes outside label set to avoid generating unused sequences.
                if gene_id_whitelist is not None and g["gene_id"] not in gene_id_whitelist:
                    continue
                if g["chrom"] not in genome_dict:
                    skipped_missing_chrom += 1
                    continue

                tss_seq, tts_seq = extract_tss_tts_sequences(
                    g,
                    genome_dict,
                    tss_upstream=tss_upstream,
                    tss_downstream=tss_downstream,
                    tts_upstream=tts_upstream,
                    tts_downstream=tts_downstream,
                )

                if ((tss_handle is not None and tss_seq is not None and len(tss_seq) != expected_tss_len)
                    or (tts_handle is not None and tts_seq is not None and len(tts_seq) != expected_tts_len)):
                    skipped_boundary += 1

                if tss_handle is not None and tss_seq is not None:
                    if len(tss_seq) == expected_tss_len and g["gene_id"] not in written_tss:
                        tss_handle.write(f">{g['gene_id']}\n{tss_seq}\n")
                        written_tss.add(g["gene_id"])

                if tts_handle is not None and tts_seq is not None:
                    if len(tts_seq) == expected_tts_len and g["gene_id"] not in written_tts:
                        tts_handle.write(f">{g['gene_id']}\n{tts_seq}\n")
                        written_tts.add(g["gene_id"])

            except (ValueError, KeyError, IndexError) as exc:
                skipped_invalid += 1
                print(f"[plantsetdelta] Skipping gene {g.get('gene_id', '<unknown>')}: {exc}")
    finally:
        if tss_handle is not None:
            tss_handle.close()
        if tts_handle is not None:
            tts_handle.close()

    return {"tss_fasta": tss_fasta, "tts_fasta": tts_fasta}