"""
Task-specific configuration settings.

This module defines configurations for different protein design tasks:
- Inverse folding (sequence design from structure)
- Forward folding (structure prediction from sequence)
- Motif scaffolding (scaffold design around motifs)
"""

# Inverse Folding Task Configuration
INVERSE_FOLDING_CONFIG = {
    'task_name': 'inverse_folding',
    'description': 'Design sequences that fold into target structures',
    
    # Data settings
    'datasets': ['cameo2022', 'pdb'],
    'default_dataset': 'cameo2022',
    
    # Masking strategy
    'masking': {
        'strategy': 'plddt_quantile',
        'initial_mask_ratio': 0.15,
        'use_structure_confidence': True,
    },
    
    # Reward computation
    'reward': {
        'weights': {
            'aar': 0.4,          # Amino acid recovery
            'sctm': 0.45,        # Structure TM-score
            'biophysical': 0.15, # Biophysical feasibility
        },
        'biophysical_penalties': {
            'charge_threshold': 0.3,    # >30% charged residues
            'hydrophobic_threshold': 0.4, # >40% hydrophobic
        },
    },
    
    # Evaluation metrics
    'metrics': ['aar', 'sctm', 'sequence_identity', 'biophysical_score'],
    
    # Expert models
    'experts': {
        'enabled': [0, 1, 2, 3],  # DPLM-2 models + ProteinMPNN
        'exclude': [],
    },
}

# Forward Folding Task Configuration
FORWARD_FOLDING_CONFIG = {
    'task_name': 'forward_folding',
    'description': 'Optimize protein structures from sequences',
    
    # Data settings
    'datasets': ['cameo2022', 'pdb'],
    'default_dataset': 'cameo2022',
    
    # Baseline generation
    'baseline': {
        'method': 'esmfold',  # ESMFold for initial structure
        'use_structure_tokens': True,
    },
    
    # Masking strategy
    'masking': {
        'strategy': 'plddt_quantile',
        'initial_mask_ratio': 0.15,
        'mask_structure_tokens': True,  # Mask structure, not sequence
        'use_esmfold_plddt': True,      # Use ESMFold confidence
    },
    
    # Reward computation
    'reward': {
        'weights': {
            'rmsd': 0.5,      # RMSD to reference
            'tm_score': 0.5,  # TM-score to reference
        },
        'rmsd_normalization': {
            'method': 'inverse_exponential',
            'scale': 10.0,
        },
    },
    
    # Evaluation metrics
    'metrics': ['rmsd', 'tm_score', 'gdt_ts', 'plddt'],
    
    # Expert models (exclude sequence-only models)
    'experts': {
        'enabled': [0, 1, 2],  # Only DPLM-2 models
        'exclude': [3, 4, 5, 6],  # Exclude ProteinMPNN and external models
    },
}

# Motif Scaffolding Task Configuration
MOTIF_SCAFFOLDING_CONFIG = {
    'task_name': 'motif_scaffolding',
    'description': 'Design scaffolds around functional motifs',
    
    # Data settings
    'datasets': ['scaffolding_pdbs'],
    'default_dataset': 'scaffolding_pdbs',
    
    # Motif handling
    'motif': {
        'preservation_required': True,
        'allow_non_contiguous': True,
        'coverage_threshold': 0.8,  # For non-contiguous motifs
    },
    
    # Scaffold generation
    'scaffold': {
        'min_length': 20,
        'max_length': 200,
        'distribution': 'balanced',  # 'balanced' or 'random'
    },
    
    # Masking strategy
    'masking': {
        'strategy': 'plddt_quantile',
        'initial_mask_ratio': 0.2,
        'preserve_motif': True,
        'progressive': True,
    },
    
    # Reward computation
    'reward': {
        'weights': {
            'motif_preservation': 0.4,
            'structure_quality': 0.3,
            'designability': 0.3,
        },
    },
    
    # Evaluation metrics
    'metrics': ['motif_rmsd', 'motif_coverage', 'scaffold_quality', 'plddt'],
    
    # Expert models
    'experts': {
        'enabled': [0, 1, 2, 4, 5, 6],  # DPLM-2 + external experts
        'exclude': [3],  # Exclude ProteinMPNN
    },
}

# Combined Task Configuration
TASK_CONFIG = {
    'inverse_folding': INVERSE_FOLDING_CONFIG,
    'forward_folding': FORWARD_FOLDING_CONFIG,
    'motif_scaffolding': MOTIF_SCAFFOLDING_CONFIG,
    'default_task': 'inverse_folding',
}

# Task-specific MCTS overrides
TASK_MCTS_OVERRIDES = {
    'inverse_folding': {
        'num_iterations': 5,
        'max_depth': 4,
        'num_rollouts_per_expert': 3,
    },
    'forward_folding': {
        'num_iterations': 5,
        'max_depth': 4,
        'num_rollouts_per_expert': 2,
    },
    'motif_scaffolding': {
        'num_iterations': 10,
        'max_depth': 5,
        'num_rollouts_per_expert': 3,
    },
}
