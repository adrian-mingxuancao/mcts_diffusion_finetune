#!/usr/bin/env python3

import sys
sys.path.append('/home/caom/AID3/dplm/src')
import torch
import numpy as np
from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration

def main():
    # Use a simple test sequence
    test_sequence = 'MQGFGVHTSMWTMNWDRPGAERAVAAALKYEVDFIEIPMLNPPAVDTEHT'
    print(f'📊 Test sequence: {test_sequence}')
    print(f'   Length: {len(test_sequence)}')

    # Initialize MCTS with correct parameters
    dplm2_integration = DPLM2Integration()
    
    # Create dummy baseline structure for initialization
    baseline_structure = {
        'coordinates': np.random.rand(len(test_sequence), 3),
        'plddt': np.random.rand(len(test_sequence)) * 100
    }
    
    mcts = GeneralMCTS(
        dplm2_integration=dplm2_integration,
        baseline_structure=baseline_structure,
        reference_sequence=test_sequence,
        task_type='folding',
        max_depth=1,
        num_children_select=2
    )

    print(f'\n🔍 ISSUE 1: What are we masking in folding?')
    print(f'   Current approach: Masking AA sequence (WRONG for folding)')
    print(f'   Correct approach: Should mask STRUCTURE tokens')

    print(f'\n🔄 Step 1: ESMFold baseline generation...')
    baseline_coords = mcts._generate_esmfold_baseline(test_sequence)
    print(f'✅ ESMFold baseline shape: {baseline_coords.shape}')

    print(f'\n🔄 Step 2: Convert ESMFold coords to structure tokens...')
    struct_tokens = mcts._coords_to_structure_tokens(baseline_coords)
    print(f'✅ Structure tokens: {len(struct_tokens)} chars')
    
    is_mask_tokens = '<mask_struct>' in struct_tokens
    print(f'   Type: {"MASK tokens" if is_mask_tokens else "REAL tokens"}')
    print(f'   Sample: {struct_tokens[:100]}...')

    print(f'\n🔍 ISSUE 2: Current pipeline flow analysis')
    if is_mask_tokens:
        print(f'   ❌ PROBLEM: _coords_to_structure_tokens returns MASK tokens')
        print(f'   ❌ This means we cannot do proper structure masking for folding')
        print(f'   ✅ SOLUTION: _coords_to_structure_tokens should return REAL structure tokens')
        print(f'   ✅ Then we can mask THOSE tokens for MCTS optimization')
    else:
        print(f'   ✅ GOOD: _coords_to_structure_tokens returns REAL structure tokens')
        print(f'   ✅ Now we can test detokenization...')
        
        # Test detokenization
        print(f'\n🔄 Step 3: Test structure token → coordinates conversion...')
        recovered_coords = mcts._structure_tokens_to_coords(struct_tokens)
        if recovered_coords is not None:
            print(f'   ✅ Recovered coordinates shape: {recovered_coords.shape}')
            rmsd = np.sqrt(np.mean((baseline_coords - recovered_coords)**2))
            print(f'   📊 RMSD vs original: {rmsd:.3f}Å')
        else:
            print(f'   ❌ Failed to recover coordinates')

    print(f'\n🎯 SUMMARY OF ISSUES:')
    print(f'   1. Masking: Currently masking AA sequence, should mask structure tokens')
    print(f'   2. Tokenization: Need real structure tokens from ESMFold coords')
    print(f'   3. Detokenization: Fixed to use atom37_positions key')

if __name__ == "__main__":
    main()
