#!/usr/bin/env python3
"""
Generate DPLM-2 150M Baselines with Fixed Seed

This script generates baseline structures using DPLM-2 150M with a fixed seed
for reproducible forward folding experiments.

Usage:
    python generate_dplm2_baselines.py --start 0 --end 5 --seed 42
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from Bio import SeqIO

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from core.dplm2_integration import DPLM2Integration

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Set all random seeds to: {seed}")

def load_cameo_sequences(fasta_path: str) -> dict:
    """Load CAMEO sequences from FASTA file."""
    sequences = {}
    try:
        with open(fasta_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences[record.id] = str(record.seq).replace(" ", "").upper()
        print(f"✅ Loaded {len(sequences)} sequences from {fasta_path}")
    except Exception as e:
        print(f"❌ Failed to load sequences: {e}")
    return sequences

def generate_dplm2_baseline(
    sequence: str,
    structure_name: str,
    dplm2: DPLM2Integration,
    output_dir: Path,
    seed: int
) -> dict:
    """
    Generate baseline structure using DPLM-2 150M with fixed seed.
    
    Returns structure tokens (not coordinates) for consistency with DPLM-2 pipeline.
    """
    print(f"\n{'='*80}")
    print(f"Generating DPLM-2 baseline: {structure_name}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Set seed before generation
    set_seed(seed)
    
    # Generate structure tokens using DPLM-2 150M (expert_id=1)
    print("🔄 Generating structure tokens with DPLM-2 150M...")
    structure_tokens = dplm2.generate_baseline_structure(
        sequence=sequence,
        expert_id=1  # DPLM-2 150M
    )
    
    if not structure_tokens:
        print(f"❌ Failed to generate structure tokens for {structure_name}")
        return None
    
    # Count tokens
    if ',' in structure_tokens:
        token_count = len(structure_tokens.split(','))
    else:
        token_count = len(structure_tokens.split())
    
    print(f"✅ Generated {token_count} structure tokens")
    
    # Save baseline
    output_file = output_dir / f"{structure_name}_dplm2_150m_seed{seed}.txt"
    with open(output_file, 'w') as f:
        f.write(f"# DPLM-2 150M Baseline\n")
        f.write(f"# Structure: {structure_name}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"# Sequence length: {len(sequence)}\n")
        f.write(f"# Token count: {token_count}\n")
        f.write(f"\n[SEQUENCE]\n{sequence}\n")
        f.write(f"\n[STRUCTURE_TOKENS]\n{structure_tokens}\n")
    
    print(f"💾 Saved to: {output_file}")
    
    return {
        'structure_name': structure_name,
        'sequence': sequence,
        'structure_tokens': structure_tokens,
        'token_count': token_count,
        'seed': seed,
        'output_file': str(output_file)
    }

def main():
    parser = argparse.ArgumentParser(description='Generate DPLM-2 150M baselines')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=5, help='End index')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, 
                       default='/net/scratch/caom/dplm2_baselines',
                       help='Output directory')
    parser.add_argument('--fasta_path', type=str,
                       default='/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta',
                       help='Path to CAMEO aatype.fasta')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("DPLM-2 150M BASELINE GENERATION")
    print("="*80)
    print(f"Start index: {args.start}")
    print(f"End index: {args.end}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print(f"FASTA path: {args.fasta_path}")
    print("="*80 + "\n")
    
    # Load sequences
    sequences = load_cameo_sequences(args.fasta_path)
    if not sequences:
        print("❌ No sequences loaded, exiting")
        return
    
    # Initialize DPLM-2
    print("\n🔄 Loading DPLM-2 150M model...")
    dplm2 = DPLM2Integration(device='cuda')
    print("✅ DPLM-2 150M loaded\n")
    
    # Generate baselines for selected sequences
    results = []
    sequence_items = list(sequences.items())[args.start:args.end]
    
    for structure_name, sequence in sequence_items:
        result = generate_dplm2_baseline(
            sequence=sequence,
            structure_name=structure_name,
            dplm2=dplm2,
            output_dir=output_dir,
            seed=args.seed
        )
        
        if result:
            results.append(result)
    
    # Save summary
    summary_file = output_dir / f"summary_seed{args.seed}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"DPLM-2 150M Baseline Generation Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model: DPLM-2 150M (airkingbd/dplm2_150m)\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Structures processed: {len(results)}\n")
        f.write(f"{'='*80}\n\n")
        
        for result in results:
            f.write(f"Structure: {result['structure_name']}\n")
            f.write(f"  Sequence length: {len(result['sequence'])}\n")
            f.write(f"  Structure tokens: {result['token_count']}\n")
            f.write(f"  Output file: {result['output_file']}\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"✅ Generated {len(results)} DPLM-2 baselines")
    print(f"📊 Summary: {summary_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
