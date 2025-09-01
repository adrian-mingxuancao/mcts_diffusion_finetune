# 🎯 Multiple Experts Rollout Implementation

## Overview

This document describes the implementation of **Multiple Experts Rollout** for the MCTS-guided DPLM-2 sequence optimization system. This feature enables the use of multiple DPLM-2 models to generate ensemble predictions for better sequence quality.

## 🚀 Features Implemented

### 1. PH-UCT (Entropy-Reinforced Planning) Selection Algorithm ✅
- **Enhanced MCTSNode**: Added entropy-related properties (`entropy_score`, `diversity_score`, `exploration_potential`)
- **PH-UCT Score**: Combines UCB1 with entropy-based and diversity-based exploration
- **Configurable Weights**: Adjustable parameters for entropy, diversity, and exploration potential
- **Fallback Support**: Can switch between PH-UCT and UCB1 algorithms

### 2. Multiple Experts Rollout ✅
- **Expert Model Loading**: Support for loading multiple DPLM-2 model checkpoints
- **Two Consensus Methods**:
  - **Probability Averaging**: Access model logits, average probabilities, then sample
  - **Majority Voting**: Generate sequences from all experts, vote on each position
- **Automatic Fallback**: Falls back to single model if multiple experts fail
- **Expert Comparison**: Analyze prediction differences and consensus levels

## 🏗️ Architecture

### DPLM2Integration Class Enhancements

```python
class DPLM2Integration:
    def __init__(self, model_name: str, expert_models: List[str] = None):
        # Load main model + expert models
        self.expert_instances = {}  # Store loaded expert models
        self.use_multiple_experts = len(self.expert_models) > 0
    
    def generate_with_multiple_experts(self, structure, target_length, 
                                     masked_sequence, temperature, 
                                     use_probability_averaging=True):
        # Main interface for multiple experts generation
        
    def _generate_with_probability_averaging(self, ...):
        # Method 1: Average logits across experts
        
    def _generate_with_majority_voting(self, ...):
        # Method 2: Vote on amino acid choices
```

### MCTS Integration

```python
class GeneralMCTS:
    def __init__(self, dplm2_integration=None, use_ph_uct=True, ...):
        # PH-UCT configuration
        self.use_ph_uct = use_ph_uct
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        
        # Multiple experts support
        self.use_multiple_experts = False
        if dplm2_integration and hasattr(dplm2_integration, 'use_multiple_experts'):
            self.use_multiple_experts = dplm2_integration.use_multiple_experts
    
    def generate_with_multiple_experts(self, ...):
        # Unified interface for sequence generation
```

## 🔧 Usage Examples

### Basic Multiple Experts Setup

```python
from core.dplm2_integration import DPLM2Integration
from core.sequence_level_mcts import GeneralMCTS

# Initialize with multiple expert models
expert_models = [
    "airkingbd/dplm2_650m",      # Main model
    "other/dplm2_checkpoint",    # Expert 1
    "another/dplm2_model"        # Expert 2
]

dplm2_integration = DPLM2Integration(
    model_name="airkingbd/dplm2_650m",
    expert_models=expert_models
)

# Create MCTS with multiple experts
mcts = GeneralMCTS(
    use_ph_uct=True,  # Enable PH-UCT
    entropy_weight=0.3,
    diversity_weight=0.2,
    dplm2_integration=dplm2_integration
)
```

### Sequence Generation with Multiple Experts

```python
# Method 1: Probability Averaging (Preferred)
sequence = mcts.generate_with_multiple_experts(
    structure=test_structure,
    target_length=50,
    masked_sequence=masked_sequence,
    use_probability_averaging=True  # Use logits averaging
)

# Method 2: Majority Voting (Fallback)
sequence = mcts.generate_with_multiple_experts(
    structure=test_structure,
    target_length=50,
    masked_sequence=masked_sequence,
    use_probability_averaging=False  # Use sequence voting
)
```

### Expert Analysis and Comparison

```python
# Get expert information
expert_info = mcts.get_multiple_experts_info()
print(f"Loaded experts: {expert_info['loaded_experts']}")

# Compare expert predictions
comparison = dplm2_integration.compare_expert_predictions(
    structure, target_length, masked_sequence
)

# Analyze consensus
for pos, analysis in comparison['consensus_analysis'].items():
    print(f"Position {pos}: {analysis['consensus_aa']} "
          f"(consensus: {analysis['consensus_score']:.2f})")
```

## 🎯 PH-UCT Configuration

### Algorithm Parameters

```python
mcts = GeneralMCTS(
    use_ph_uct=True,                    # Enable PH-UCT
    entropy_weight=0.3,                 # Weight for entropy-based exploration
    diversity_weight=0.2,               # Weight for diversity-based exploration
    exploration_potential_weight=0.5,   # Weight for completion-based exploration
    exploration_constant=1.414          # UCB1 exploration constant
)
```

### PH-UCT Score Components

The PH-UCT score combines four components:

1. **UCB1 Base**: `value + c * sqrt(ln(parent_visits) / node_visits)`
2. **Entropy Term**: `entropy_weight * entropy_score` (encourages uncertainty exploration)
3. **Diversity Term**: `diversity_weight * diversity_score` (encourages tree diversity)
4. **Exploration Potential**: `(1 - entropy_weight - diversity_weight) * exploration_potential`

## 🔍 How It Works

### Probability Averaging (Method 1)

1. **Load Expert Models**: Multiple DPLM-2 checkpoints are loaded
2. **Get Logits**: Each expert provides logits for masked positions
3. **Average Probabilities**: Logits are averaged across experts
4. **Sample**: Amino acids are sampled from averaged distribution

```python
# Collect logits from all experts
all_expert_logits = []
for expert_name, expert_model in self.expert_instances.items():
    outputs = expert_model.forward(input_ids=batch["input_tokens"])
    logits = outputs["logits"]
    all_expert_logits.append(logits)

# Average logits across experts
averaged_logits = torch.stack(all_expert_logits).mean(dim=0)

# Convert to probabilities and sample
masked_probs = torch.softmax(averaged_logits[:, masked_positions, :] / temperature, dim=-1)
sampled_tokens = torch.multinomial(masked_probs.view(-1, masked_probs.size(-1)), 1)
```

### Majority Voting (Method 2)

1. **Generate Sequences**: Each expert generates a complete sequence
2. **Count Votes**: For each masked position, count amino acid frequencies
3. **Select Winner**: Choose the most common amino acid

```python
# Generate sequences from all experts
expert_sequences = []
for expert_name, expert_model in self.expert_instances.items():
    expert_seq = self.generate_sequence(structure, target_length, masked_sequence)
    expert_sequences.append(expert_seq)

# Vote on each masked position
for pos in masked_positions:
    position_amino_acids = [seq[pos] for seq in expert_sequences if pos < len(seq)]
    aa_counts = Counter(position_amino_acids)
    most_common_aa = aa_counts.most_common(1)[0][0]
    result_sequence[pos] = most_common_aa
```

## 🧪 Testing and Demonstration

### Run PH-UCT Demo

```bash
cd mcts_diffusion_finetune
python -c "from core.sequence_level_mcts import demonstrate_ph_uct; demonstrate_ph_uct()"
```

### Run Multiple Experts Demo

```bash
cd mcts_diffusion_finetune
python -c "from core.sequence_level_mcts import demonstrate_multiple_experts_mcts; demonstrate_multiple_experts_mcts()"
```

### Run DPLM2 Integration Demo

```bash
cd mcts_diffusion_finetune
python -c "from core.dplm2_integration import demonstrate_multiple_experts; demonstrate_multiple_experts()"
```

## 📊 Expected Benefits

### PH-UCT Improvements
- **Better Exploration**: Entropy-based exploration encourages visiting uncertain nodes
- **Diversity Promotion**: Diversity-based exploration prevents getting stuck in local optima
- **Adaptive Search**: Exploration potential considers completion ratio and depth

### Multiple Experts Improvements
- **Higher Quality**: Ensemble predictions reduce individual model biases
- **Robustness**: Multiple models provide redundancy and error correction
- **Consensus**: Agreement between experts indicates higher confidence predictions

## 🚨 Limitations and Considerations

### Current Limitations
1. **Model Compatibility**: All experts must use compatible tokenizers
2. **Memory Usage**: Loading multiple models increases memory requirements
3. **Inference Time**: Multiple forward passes increase generation time

### Best Practices
1. **Use Probability Averaging** when possible (more principled)
2. **Fallback to Majority Voting** if logits are unavailable
3. **Monitor Expert Agreement** to assess prediction confidence
4. **Balance Model Diversity** vs. computational cost

## 🔮 Future Enhancements

### Potential Improvements
1. **Weighted Averaging**: Weight experts by their performance or confidence
2. **Dynamic Expert Selection**: Choose experts based on task characteristics
3. **Cross-Validation**: Use held-out data to validate expert combinations
4. **Adaptive Consensus**: Adjust consensus thresholds based on task difficulty

### Integration Opportunities
1. **MCTS Rollout**: Use multiple experts during MCTS simulation phase
2. **Confidence Scoring**: Combine expert predictions with confidence estimates
3. **Active Learning**: Use expert disagreement to identify uncertain regions

## 📝 Summary

The Multiple Experts Rollout implementation provides:

1. **PH-UCT Algorithm**: Enhanced tree search with entropy-reinforced planning
2. **Multiple Experts**: Ensemble predictions from multiple DPLM-2 models
3. **Two Consensus Methods**: Probability averaging and majority voting
4. **Seamless Integration**: Works within existing MCTS framework
5. **Automatic Fallback**: Graceful degradation to single model when needed

This implementation significantly enhances the MCTS-guided protein sequence optimization by combining the benefits of improved tree search algorithms with ensemble model predictions.
