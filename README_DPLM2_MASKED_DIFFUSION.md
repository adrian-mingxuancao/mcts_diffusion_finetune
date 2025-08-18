# DPLM-2 Masked Diffusion for MCTS Integration

## Overview

This module integrates DPLM-2's masked diffusion capabilities with Monte Carlo Tree Search (MCTS) to provide biologically plausible protein sequence optimization. Instead of random amino acid generation, DPLM-2 fills masked positions using its understanding of protein structure and sequence relationships.

## Key Innovation

**Traditional MCTS Problem**: Random amino acid generation creates an impossibly large search space (20^L for length L), making optimization infeasible.

**DPLM-2 Solution**: Uses masked diffusion to fill specific positions while preserving unmasked ones, providing a reasonable search space that MCTS can effectively explore.

## How It Works

### 1. Masked Diffusion with Partial Masks

DPLM-2 uses `partial_masks` to control which positions are filled during diffusion:

```python
# partial_masks[i] = True means position i should be preserved (NOT filled)
# partial_masks[i] = False means position i should be filled by diffusion

partial_masks = torch.zeros_like(input_tokens, dtype=torch.bool)

# Mark unmasked positions as preserved
for i, char in enumerate(masked_sequence):
    if char != 'X':
        token_pos = i + 1  # +1 for CLS token
        partial_masks[0, token_pos] = True
```

### 2. Sequence-to-Sequence Diffusion

Unlike structure-to-sequence generation, this approach:
- Takes a sequence with X positions (e.g., "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFXEIPMLDPPXIDTAYF")
- Uses DPLM-2 to fill ONLY the X positions
- Preserves all unmasked positions exactly
- Returns completed sequence (e.g., "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFAEIPMLDPPAIDTAYF")

### 3. MCTS Integration

MCTS can now:
- Start with fully masked sequences
- Iteratively unmask positions based on search decisions
- Use DPLM-2 to fill remaining masked positions
- Explore biologically plausible sequence space

## Implementation Details

### Core Method: `fill_masked_positions`

```python
def fill_masked_positions(self, structure: Dict = None, masked_sequence: str = None, 
                         target_length: int = None, temperature: float = 1.0) -> str:
    """
    Fill masked positions using DPLM-2's partial_masks functionality.
    
    Args:
        structure: Optional structure for conditioning
        masked_sequence: Sequence with 'X' for masked positions
        target_length: Expected sequence length
        temperature: Sampling temperature
        
    Returns:
        Completed sequence with all masked positions filled
    """
```

### Key Features

1. **Position Preservation**: Unmasked positions are preserved exactly
2. **Flexible Masking**: Any number of positions can be masked
3. **Structure Conditioning**: Optional structure input for better generation
4. **Temperature Control**: Adjustable sampling randomness
5. **Quality Verification**: Checks position preservation and sequence validity

## Usage Examples

### Basic Masked Diffusion

```python
from core.dplm2_integration import DPLM2Integration

# Initialize DPLM-2
dplm2 = DPLM2Integration(model_name="airkingbd/dplm2_650m")

# Create masked sequence
masked_seq = "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFXEIPMLDPPXIDTAYF"

# Fill masked positions
completed_seq = dplm2.fill_masked_positions(
    masked_sequence=masked_seq,
    target_length=len(masked_seq),
    temperature=1.0
)

print(f"Original: {masked_seq}")
print(f"Completed: {completed_seq}")
```

### MCTS Integration

```python
from core.sequence_level_mcts import GeneralMCTS

# Initialize MCTS with DPLM-2
mcts = GeneralMCTS(
    initial_sequence="X" * 50,  # Start with fully masked sequence
    dplm2_integration=dplm2
)

# MCTS will automatically use DPLM-2 for sequence expansion
best_sequence = mcts.search()
```

### Structure-Conditional Generation

```python
# With structure information
structure = {
    'coordinates': coords,  # 3D coordinates
    'sequence': 'A' * 50,
    'length': 50
}

completed_seq = dplm2.fill_masked_positions(
    structure=structure,
    masked_sequence=masked_seq,
    target_length=len(masked_seq)
)
```

## Testing

Run the comprehensive test suite:

```bash
cd mcts_diffusion_finetune
python test_dplm2_masked_diffusion.py
```

This tests:
1. Basic masked diffusion functionality
2. Different mask ratios (10%, 20%, 50%)
3. MCTS-style iterative masking
4. Structure-conditional generation
5. MCTS integration

## Benefits Over Random Generation

| Aspect | Random Generation | DPLM-2 Masked Diffusion |
|--------|-------------------|-------------------------|
| **Search Space** | 20^L (impossible) | Biologically plausible |
| **Position Preservation** | None | Perfect |
| **Biological Relevance** | Random | Structure-aware |
| **MCTS Efficiency** | Poor | Excellent |
| **Sequence Quality** | Low | High |

## Technical Details

### Partial Masks Implementation

DPLM-2's `partial_masks` parameter works as follows:

```python
# In DPLM-2 model
def get_non_special_symbol_mask(self, output_tokens, partial_masks=None):
    non_special_symbol_mask = (
        output_tokens.ne(self.pad_id)
        & output_tokens.ne(self.aa_bos_id)
        & output_tokens.ne(self.aa_eos_id)
        & output_tokens.ne(self.struct_bos_id)
        & output_tokens.ne(self.struct_eos_id)
    )
    if partial_masks is not None:
        non_special_symbol_mask &= ~partial_masks  # 🎯 KEY: Exclude preserved positions
    return non_special_symbol_mask
```

### Tokenization Strategy

1. **Input Format**: `[CLS] + sequence + [EOS]`
2. **Masking**: X positions remain as mask tokens
3. **Preservation**: Unmasked positions get their actual amino acid tokens
4. **Generation**: DPLM-2 fills only mask token positions

### Sampling Strategy

Uses the same parameters as motif scaffolding:
- `sampling_strategy="annealing@2.0:1.0"`: Temperature annealing from 2.0 to 1.0
- `max_iter=100`: Reasonable number of diffusion steps
- `temperature=1.0`: Base sampling temperature

## Integration with Existing Pipeline

### 1. MCTS Framework

The existing MCTS framework automatically uses DPLM-2 masked diffusion:

```python
# In sequence_level_mcts.py
def _fill_masked_positions_with_dplm2(self, sequence: str, masked_positions: Set[int], temperature: float = 1.0) -> str:
    # Create masked sequence
    masked_sequence = list(sequence)
    for pos in masked_positions:
        masked_sequence[pos] = 'X'
    
    # Use DPLM-2 masked diffusion
    return self.dplm2_integration.fill_masked_positions(
        structure=None,
        masked_sequence=''.join(masked_sequence),
        target_length=len(sequence),
        temperature=temperature
    )
```

### 2. DPLM-2 Integration

The `DPLM2Integration` class handles:
- Model loading and initialization
- Tokenization and batch creation
- Partial mask generation
- Sequence generation and decoding
- Quality verification

### 3. Error Handling

Robust error handling ensures graceful degradation:
- Model loading failures
- Tokenization errors
- Generation failures
- Position preservation verification

## Future Enhancements

### 1. Advanced Masking Strategies

- **Adaptive Masking**: Mask positions based on confidence scores
- **Structured Masking**: Mask functional domains or regions
- **Progressive Masking**: Gradually unmask positions during optimization

### 2. Multi-Modal Conditioning

- **Sequence + Structure**: Leverage both modalities for better generation
- **Functional Annotations**: Condition on GO terms or enzyme classes
- **Evolutionary Information**: Use multiple sequence alignments

### 3. Optimization Strategies

- **Bayesian Optimization**: Optimize diffusion parameters
- **Reinforcement Learning**: Learn optimal masking strategies
- **Ensemble Methods**: Combine multiple DPLM-2 models

## Troubleshooting

### Common Issues

1. **Low Position Preservation**
   - Check partial mask generation
   - Verify tokenization alignment
   - Adjust temperature parameters

2. **Generation Failures**
   - Ensure DPLM-2 model is loaded
   - Check input sequence format
   - Verify mask token handling

3. **Performance Issues**
   - Reduce max_iter for faster generation
   - Use smaller batch sizes
   - Enable GPU acceleration

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set specific logger
dplm2.logger.setLevel(logging.DEBUG)
```

## References

1. **DPLM-2 Paper**: "DPLM-2: A Multimodal Diffusion Protein Language Model"
2. **Motif Scaffolding**: DPLM-2's approach to conditional generation
3. **Partial Masks**: Technical implementation in DPLM-2 codebase
4. **MCTS Integration**: How search algorithms leverage diffusion models

## Conclusion

DPLM-2 masked diffusion provides a powerful foundation for MCTS-guided protein optimization by:

1. **Reducing Search Space**: From random (20^L) to biologically plausible
2. **Maintaining Quality**: Leveraging DPLM-2's protein understanding
3. **Enabling Exploration**: MCTS can effectively search reasonable sequences
4. **Preserving Context**: Unmasked positions guide generation

This integration transforms MCTS from a theoretical framework to a practical tool for protein sequence optimization.





