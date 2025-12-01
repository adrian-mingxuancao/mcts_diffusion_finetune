"""
Path configuration for data, models, and results.

This module centralizes all file paths used in the framework.
Modify these paths based on your system setup.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR

# Data directories
DATA_ROOT = Path('/net/scratch/caom/dplm_datasets')
DATA_PATHS = {
    'cameo2022': {
        'root': DATA_ROOT / 'data-bin' / 'cameo2022',
        'preprocessed': DATA_ROOT / 'data-bin' / 'cameo2022' / 'preprocessed',
        'struct_fasta': DATA_ROOT / 'data-bin' / 'cameo2022' / 'struct.fasta',
        'aa_fasta': DATA_ROOT / 'data-bin' / 'cameo2022' / 'aatype.fasta',
        'coordinates': DATA_ROOT / 'data-bin' / 'cameo2022' / 'preprocessed',
    },
    'pdb': {
        'root': DATA_ROOT / 'pdb_data',
        'coordinates': DATA_ROOT / 'pdb_data' / 'coordinates',
        'sequences': DATA_ROOT / 'pdb_data' / 'sequences',
    },
    'scaffolding': {
        'root': DATA_ROOT / 'data-bin' / 'scaffolding-pdbs',
        'aa_seq': DATA_ROOT / 'data-bin' / 'scaffolding-pdbs' / 'aa_seq.fasta',
        'motif_info': DATA_ROOT / 'data-bin' / 'scaffolding-pdbs' / 'motif_info.json',
    },
}

# Model directories
MODEL_ROOT = Path('/home/caom/AID3/dplm')
MODEL_PATHS = {
    'dplm2': {
        '650m': 'AI4Protein/DPLM2-650M',  # HuggingFace model ID
        '150m': 'AI4Protein/DPLM2-150M',
        '3b': 'AI4Protein/DPLM2-3B',
    },
    'esmfold': {
        'checkpoint': 'facebook/esmfold_v1',  # HuggingFace model ID
    },
    'proteinmpnn': {
        'checkpoint': MODEL_ROOT / 'denovo-protein-server' / 'third_party' / 'proteinmpnn' / 'ca_model_weights' / 'v_48_020.pt',
    },
    'structure_tokenizer': {
        'checkpoint': 'airkingbd/struct_tokenizer',  # HuggingFace model ID
    },
}

# Generation results (pregenerated baselines)
GENERATION_RESULTS = {
    'dplm2_650m': MODEL_ROOT / 'generation-results' / 'dplm2_650m',
    'dplm2_150m': MODEL_ROOT / 'generation-results' / 'dplm2_150m',
    'dplm2_3b': MODEL_ROOT / 'generation-results' / 'dplm2_3b',
    'inverse_folding': MODEL_ROOT / 'generation-results' / 'dplm2_150m' / 'inverse_folding',
}

# Result directories
RESULT_ROOT = PROJECT_ROOT / 'results'
RESULT_PATHS = {
    'inverse_folding': RESULT_ROOT / 'inverse_folding',
    'forward_folding': RESULT_ROOT / 'forward_folding',
    'motif_scaffolding': RESULT_ROOT / 'motif_scaffolding',
    'ablations': RESULT_ROOT / 'ablations',
    'analysis': RESULT_ROOT / 'analysis',
    'visualizations': RESULT_ROOT / 'visualizations',
}

# Log directories
LOG_ROOT = PROJECT_ROOT / 'logs'
LOG_PATHS = {
    'experiments': LOG_ROOT / 'experiments',
    'models': LOG_ROOT / 'models',
    'errors': LOG_ROOT / 'errors',
}

# Cache directories
CACHE_ROOT = PROJECT_ROOT / 'cache'
CACHE_PATHS = {
    'models': CACHE_ROOT / 'models',
    'tokenizers': CACHE_ROOT / 'tokenizers',
    'sequences': CACHE_ROOT / 'sequences',
    'structures': CACHE_ROOT / 'structures',
}

# Checkpoint directories
CHECKPOINT_ROOT = PROJECT_ROOT / 'checkpoints'
CHECKPOINT_PATHS = {
    'mcts': CHECKPOINT_ROOT / 'mcts',
    'models': CHECKPOINT_ROOT / 'models',
}

# Combined paths dictionary
PATHS = {
    'base': BASE_DIR,
    'project_root': PROJECT_ROOT,
    'data': DATA_PATHS,
    'models': MODEL_PATHS,
    'generation_results': GENERATION_RESULTS,
    'results': RESULT_PATHS,
    'logs': LOG_PATHS,
    'cache': CACHE_PATHS,
    'checkpoints': CHECKPOINT_PATHS,
}

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories."""
    dirs_to_create = [
        RESULT_ROOT,
        LOG_ROOT,
        CACHE_ROOT,
        CHECKPOINT_ROOT,
    ]
    
    # Add all subdirectories
    for path_dict in [RESULT_PATHS, LOG_PATHS, CACHE_PATHS, CHECKPOINT_PATHS]:
        dirs_to_create.extend(path_dict.values())
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

# Utility functions
def get_data_path(dataset: str, file_type: str = 'root') -> Path:
    """Get path for a specific dataset and file type."""
    if dataset not in DATA_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if file_type not in DATA_PATHS[dataset]:
        raise ValueError(f"Unknown file type '{file_type}' for dataset '{dataset}'")
    return DATA_PATHS[dataset][file_type]

def get_model_path(model: str, variant: str = None) -> str:
    """Get path for a specific model."""
    if model not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model}")
    if variant:
        if variant not in MODEL_PATHS[model]:
            raise ValueError(f"Unknown variant '{variant}' for model '{model}'")
        return str(MODEL_PATHS[model][variant])
    return str(MODEL_PATHS[model])

def get_result_path(task: str, create: bool = True) -> Path:
    """Get result path for a specific task."""
    if task not in RESULT_PATHS:
        raise ValueError(f"Unknown task: {task}")
    path = RESULT_PATHS[task]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

# Initialize directories on import (optional)
# create_directories()
