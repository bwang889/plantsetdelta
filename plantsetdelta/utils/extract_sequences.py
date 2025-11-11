from pathlib import Path
from typing import Dict
from Bio import SeqIO

def parse_gff_genes(gff_file: Path) -> list:
    """
    Parse gene features from a GFF3/GTF annotation file.
    Return: list of dicts with chrom, start, end, strand, gene_id
    """
    genes = []
    with open(gff_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2].lower() != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attr = fields
            gene_id = None

            for part in attr.split(";"):
                if part.strip().startswith("ID="):
                    gene_id = part.strip().split("=",1)[-1].split(":")[-1]
                    break
            if not gene_id:

                for part in attr.split(";"):
                    if "gene_id" in part:
                        gene_id = part.strip().split()[1].replace('"','').replace(";","")
                        break
            if gene_id:
                genes.append({
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "gene_id": gene_id
                })
    return genes

def extract_sequence(region, genome_dict, flank5, flank3):
    """
    For a given gene region, extract [start-flank5, start+flank3] for TSS,
    [end-flank5, end+flank3] for TTS. Handles strand.
    """
    chrom = region["chrom"]
    strand = region["strand"]
    seq = genome_dict[chrom].seq

    if strand == "+":
        tss_start = max(region["start"] - flank5 - 1, 0)
        tss_end = region["start"] + flank3 - 1
        tts_start = max(region["end"] - flank3 - 1, 0)
        tts_end = region["end"] + flank5 - 1
    else:
        # For minus strand, swap logic
        tss_start = max(region["end"] - flank3 - 1, 0)
        tss_end = region["end"] + flank5 - 1
        tts_start = max(region["start"] - flank5 - 1, 0)
        tts_end = region["start"] + flank3 - 1

    tss_seq = seq[tss_start:tss_end]
    tts_seq = seq[tts_start:tts_end]

    if strand == "-":
        from Bio.Seq import Seq
        tss_seq = tss_seq.reverse_complement()
        tts_seq = tts_seq.reverse_complement()
    return str(tss_seq).upper(), str(tts_seq).upper()

def generate_tss_tts_sequences(
    genome_fa: Path, gff_file: Path, output_dir: Path
) -> Dict[str, Path]:
    """
    Pure Python: Extract 1.5kb TSS/TTS sequence regions from annotation and reference genome.
    All sequences are converted to uppercase.
    Output: dict of fasta paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tss_fasta = output_dir / "tss_final.fa"
    tts_fasta = output_dir / "tts_final.fa"
    genes = parse_gff_genes(gff_file)
    genome_dict = SeqIO.to_dict(SeqIO.parse(str(genome_fa), "fasta"))
    written_tss, written_tts = set(), set()
    with open(tss_fasta, "w") as tss_out, open(tts_fasta, "w") as tts_out:
        for g in genes:
            try:
                tss_seq, tts_seq = extract_sequence(g, genome_dict, flank5=1000, flank3=500)
                if len(tss_seq) == 1500 and g["gene_id"] not in written_tss:
                    tss_out.write(f">{g['gene_id']}\n{tss_seq}\n")
                    written_tss.add(g["gene_id"])
                if len(tts_seq) == 1500 and g["gene_id"] not in written_tts:
                    tts_out.write(f">{g['gene_id']}\n{tts_seq}\n")
                    written_tts.add(g["gene_id"])
            except Exception as e:
                continue
    return {"tss_fasta": tss_fasta, "tts_fasta": tts_fasta}
