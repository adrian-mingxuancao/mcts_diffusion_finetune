"""
MCTS algorithm configuration settings.

This module defines hyperparameters and strategies for:
- Selection (UCT, PH-UCT)
- Expansion (rollouts, masking)
- Simulation
- Backpropagation
"""

# Core MCTS Configuration
MCTS_CONFIG = {
    'num_iterations': 5,
    'max_depth': 4,
    'backup_rule': 'max',  # 'max' or 'sum'
    'random_seed': None,
    'verbose': True,
}

# Selection Strategy Configuration
SELECTION_CONFIG = {
    'strategy': 'ph_uct',  # 'uct' or 'ph_uct'
    'uct': {
        'exploration_constant': 1.414,  # sqrt(2) for UCB1
    },
    'ph_uct': {
        'exploration_constant': 1.414,
        'entropy_weight': 1.0 / 2.718,  # 1/e as per ERP paper
        'novelty_weight': 0.1,
        'use_entropy': True,
        'use_novelty': True,
    },
}

# Expansion Strategy Configuration
EXPANSION_CONFIG = {
    'num_rollouts_per_expert': 3,
    'top_k_candidates': 2,
    'diversity_filtering': {
        'enabled': True,
        'min_hamming_distance': 5,
        'max_duplicates': 0,
    },
    'masking': {
        'strategy': 'plddt_quantile',  # 'plddt_threshold', 'plddt_quantile', 'random'
        'progressive': True,
        'depth_schedule': {
            0: {'quantile': 0.8, 'min_ratio': 0.05, 'max_ratio': 0.25},
            1: {'quantile': 0.6, 'min_ratio': 0.05, 'max_ratio': 0.20},
            2: {'quantile': 0.4, 'min_ratio': 0.05, 'max_ratio': 0.15},
            3: {'quantile': 0.2, 'min_ratio': 0.05, 'max_ratio': 0.10},
            4: {'quantile': 0.1, 'min_ratio': 0.05, 'max_ratio': 0.05},
        },
        'threshold_schedule': {
            0: {'threshold': 70.0, 'min_ratio': 0.05, 'max_ratio': 0.25},
            1: {'threshold': 60.0, 'min_ratio': 0.05, 'max_ratio': 0.20},
            2: {'threshold': 50.0, 'min_ratio': 0.05, 'max_ratio': 0.15},
            3: {'threshold': 40.0, 'min_ratio': 0.05, 'max_ratio': 0.10},
            4: {'threshold': 30.0, 'min_ratio': 0.05, 'max_ratio': 0.05},
        },
    },
}

# Simulation Configuration
SIMULATION_CONFIG = {
    'use_fast_proxy': True,  # Use AAR-only for initial ranking
    'full_evaluation_top_k': 5,  # Only compute full metrics for top candidates
}

# Backpropagation Configuration
BACKPROPAGATION_CONFIG = {
    'update_rule': 'max',  # 'max' for pure MCTS, 'average' for traditional
    'discount_factor': 1.0,  # No discounting by default
}

# Entropy Computation Configuration
ENTROPY_CONFIG = {
    'method': 'predictive',  # 'predictive' or 'ensemble'
    'predictive': {
        'use_logits': True,
        'temperature': 1.0,
    },
    'ensemble': {
        'num_experts': 3,
        'aggregation': 'mean',  # 'mean' or 'max'
    },
}

# Novelty Computation Configuration
NOVELTY_CONFIG = {
    'metric': 'hamming',  # 'hamming' or 'edit_distance'
    'normalize': True,
    'cache_sequences': True,
}

# Ablation Study Configurations
ABLATION_CONFIGS = {
    'random_no_expert': {
        'selection_strategy': 'random',
        'use_entropy': False,
        'use_novelty': False,
    },
    'single_expert': {
        'num_experts': 1,
        'expert_id': 0,  # Can be overridden
        'use_entropy': True,
        'use_novelty': True,
    },
    'multi_expert': {
        'num_experts': 3,
        'expert_ids': [0, 1, 2],
        'use_entropy': True,
        'use_novelty': True,
    },
    'uct_baseline': {
        'selection_strategy': 'uct',
        'use_entropy': False,
        'use_novelty': False,
    },
}
