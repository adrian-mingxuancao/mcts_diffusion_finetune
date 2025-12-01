#!/usr/bin/env python3
"""
Generate Fixed Baseline Structures using DPLM-2 150M

This script generates baseline structures for forward folding experiments
using DPLM-2 150M with a FIXED SEED for reproducibility.

Usage:
    python generate_fixed_baseline.py --start 0 --end 10 --seed 42
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from core.dplm2_integration import DPLM2Integration
from utils.cameo_data_loader import CAMEODataLoader

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Set all random seeds to: {seed}")

def generate_baseline_structure(
    sequence: str,
    structure_name: str,
    dplm2: DPLM2Integration,
    output_dir: Path,
    seed: int
) -> dict:
    """
    Generate baseline structure using DPLM-2 150M with fixed seed.
    
    Args:
        sequence: Amino acid sequence
        structure_name: Name of the structure (e.g., "7dz2_C")
        dplm2: DPLM2Integration instance
        output_dir: Directory to save results
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with baseline structure information
    """
    print(f"\n{'='*80}")
    print(f"Generating baseline for: {structure_name}")
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
    
    print(f"✅ Generated structure tokens: {len(structure_tokens.split(','))} tokens")
    
    # Save baseline to file
    output_file = output_dir / f"{structure_name}_baseline_dplm2_150m_seed{seed}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Structure: {structure_name}\n")
        f.write(f"Sequence: {sequence}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Model: DPLM-2 150M\n")
        f.write(f"Structure tokens: {structure_tokens}\n")
    
    print(f"💾 Saved baseline to: {output_file}")
    
    return {
        'structure_name': structure_name,
        'sequence': sequence,
        'structure_tokens': structure_tokens,
        'seed': seed,
        'model': 'DPLM-2 150M',
        'output_file': str(output_file)
    }

def main():
    parser = argparse.ArgumentParser(description='Generate fixed baseline structures')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=10, help='End index')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, 
                       default='/net/scratch/caom/folding_baselines',
                       help='Output directory for baselines')
    parser.add_argument('--data_dir', type=str,
                       default='/home/caom/AID3/dplm/data-bin/cameo2022',
                       help='CAMEO data directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("FIXED BASELINE GENERATION - DPLM-2 150M")
    print("="*80)
    print(f"Start index: {args.start}")
    print(f"End index: {args.end}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print(f"Data directory: {args.data_dir}")
    print("="*80 + "\n")
    
    # Initialize DPLM2Integration
    print("🔄 Loading DPLM-2 150M model...")
    dplm2 = DPLM2Integration(device='cuda')
    print("✅ DPLM-2 150M loaded\n")
    
    # Load CAMEO data
    print("📂 Loading CAMEO structures...")
    loader = CAMEODataLoader(args.data_dir)
    print(f"✅ Loaded {len(loader.structures)} structures\n")
    
    # Generate baselines for selected structures
    results = []
    for idx in range(args.start, min(args.end, len(loader.structures))):
        structure = loader.structures[idx]
        
        # Handle both dict and object structures
        if isinstance(structure, dict):
            structure_name = f"{structure.get('pdb_id', 'unknown')}_{structure.get('chain_id', 'A')}"
            aatype = structure.get('aatype', [])
            sequence = structure.get('sequence', '')
        else:
            # Object with attributes
            structure_name = f"{getattr(structure, 'pdb_id', 'unknown')}_{getattr(structure, 'chain_id', 'A')}"
            aatype = getattr(structure, 'aatype', [])
            sequence = getattr(structure, 'sequence', '')
        
        # Get sequence from aatype if needed
        if not sequence and isinstance(aatype, np.ndarray):
            # Convert aatype indices to amino acid sequence
            from byprot.utils.protein.residue_constants import restypes
            sequence = ''.join([restypes[i] for i in aatype if 0 <= i < 20])
        
        if not sequence:
            print(f"⚠️ No sequence found for {structure_name}, skipping")
            continue
        
        result = generate_baseline_structure(
            sequence=sequence,
            structure_name=structure_name,
            dplm2=dplm2,
            output_dir=output_dir,
            seed=args.seed
        )
        
        if result:
            results.append(result)
    
    # Save summary
    summary_file = output_dir / f"baseline_summary_seed{args.seed}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Fixed Baseline Generation Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model: DPLM-2 150M\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Structures processed: {len(results)}\n")
        f.write(f"{'='*80}\n\n")
        
        for result in results:
            f.write(f"Structure: {result['structure_name']}\n")
            f.write(f"  Sequence length: {len(result['sequence'])}\n")
            f.write(f"  Structure tokens: {len(result['structure_tokens'].split(','))}\n")
            f.write(f"  Output file: {result['output_file']}\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"✅ Generated {len(results)} baseline structures")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
