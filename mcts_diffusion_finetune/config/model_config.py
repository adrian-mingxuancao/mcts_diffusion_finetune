"""
Model configuration settings.

This module defines configurations for all models used in the framework:
- DPLM-2 models (650M, 150M, 3B)
- ESMFold
- ProteinMPNN
- External expert models (Proteinea, FlowFlow, RFDiffusion)
"""

# DPLM-2 Model Configurations
DPLM2_CONFIG = {
    'models': {
        0: {  # 650M model
            'name': 'dplm2_650m',
            'checkpoint': 'AI4Protein/DPLM2-650M',
            'size': '650M',
            'default': True,
        },
        1: {  # 150M model
            'name': 'dplm2_150m',
            'checkpoint': 'AI4Protein/DPLM2-150M',
            'size': '150M',
        },
        2: {  # 3B model
            'name': 'dplm2_3b',
            'checkpoint': 'AI4Protein/DPLM2-3B',
            'size': '3B',
            'requires_patching': True,  # Needs forward method patching
        },
    },
    'generation': {
        'temperature': 1.0,
        'top_k': None,
        'top_p': None,
        'num_return_sequences': 1,
        'do_sample': True,
        'max_length': 1024,
    },
    'tokenization': {
        'struct_mask_token': '<mask_struct>',
        'aa_mask_token': '<mask_aa>',
        'cls_struct_token': '<cls_struct>',
        'cls_aa_token': '<cls_aa>',
        'eos_struct_token': '<eos_struct>',
        'eos_aa_token': '<eos_aa>',
    },
    'device': 'cuda',
    'dtype': 'float16',
}

# ESMFold Configuration
ESMFOLD_CONFIG = {
    'model_name': 'facebook/esmfold_v1',
    'device': 'cuda',
    'chunk_size': None,  # For long sequences
    'use_esm2_embeddings': True,
}

# ProteinMPNN Configuration
PROTEINMPNN_CONFIG = {
    'model_path': '/home/caom/AID3/dplm/denovo-protein-server/third_party/proteinmpnn/ca_model_weights/v_48_020.pt',
    'num_edges': 48,
    'noise_level': 0.2,
    'temperature': 0.1,
    'num_sequences': 1,
    'device': 'cuda',
}

# External Expert Model Configurations
EXTERNAL_EXPERTS_CONFIG = {
    'proteinea': {
        'enabled': True,
        'api_style': 'protgen_server',
        'temperature': 1.0,
        'num_designs': 1,
    },
    'flowflow': {
        'enabled': True,
        'api_style': 'protgen_server',
        'temperature': 1.0,
        'num_designs': 1,
    },
    'rfdiffusion': {
        'enabled': True,
        'api_style': 'protgen_server',
        'temperature': 1.0,
        'num_designs': 1,
    },
}

# Combined Model Configuration
MODEL_CONFIG = {
    'dplm2': DPLM2_CONFIG,
    'esmfold': ESMFOLD_CONFIG,
    'proteinmpnn': PROTEINMPNN_CONFIG,
    'external_experts': EXTERNAL_EXPERTS_CONFIG,
}

# Expert Model Registry (for MCTS)
EXPERT_CONFIG = {
    'available_experts': {
        0: 'dplm2_650m',
        1: 'dplm2_150m',
        2: 'dplm2_3b',
        3: 'proteinmpnn',
        4: 'proteinea',
        5: 'flowflow',
        6: 'rfdiffusion',
    },
    'default_experts': [0, 1, 2],  # DPLM-2 models
    'exclude_for_folding': [3, 4, 5, 6],  # Exclude sequence-only models for folding
    'memory_management': {
        'load_on_demand': True,
        'unload_after_use': True,
        'max_concurrent_models': 1,
    },
}
