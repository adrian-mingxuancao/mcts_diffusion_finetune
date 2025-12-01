def load_struct_seq_from_fasta(fasta_path: str, name: str) -> str:
    """
    Load structure token sequence from struct.fasta file using SeqIO (same as generate_dplm2_patched_v2.py).
    
    Args:
        fasta_path: Path to struct.fasta file
        name: Structure name (e.g., "7dz2_C")
    
    Returns:
        Comma-separated structure token string
    """
    from Bio import SeqIO
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.name == name:
            return str(record.seq)
    
    raise ValueError(f"{name} not found in {fasta_path}")
