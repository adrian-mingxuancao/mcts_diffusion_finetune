# MCTS Framework Extension Plan

## Overview
This document outlines the plan to extend our current MCTS framework from inverse folding to support all DPLM-2 tasks: folding, conditional generation (motif scaffolding), and unconditional generation.

## Current Status
✅ **Inverse Folding**: Fully implemented with:
- pH-UCT selection by dynamic pLDDT masking
- Multiple experts rollout (ranking all n*k rollouts)  
- Compound reward (85% AAR + 10% scTM + 5% biophysical)
- Baseline and final reward tracking

## Extension Tasks

### 1. Folding Task (Sequence → Structure)
**Goal**: Optimize structure prediction quality using MCTS

#### Implementation Plan:
- **Input**: Amino acid sequence
- **Output**: Optimized structure coordinates
- **MCTS Strategy**: 
  - Mask sequence positions with low confidence
  - Use DPLM-2 folding model for structure prediction
  - Evaluate using structural quality metrics (pLDDT, clash scores, Ramachandran)
  - Multiple experts: Use different DPLM-2 folding model sizes

#### Reward Function:
```python
folding_reward = (
    0.60 * plddt_score +      # Structure confidence
    0.20 * clash_penalty +    # Avoid atomic clashes  
    0.10 * ramachandran +     # Backbone geometry
    0.10 * compactness        # Overall fold quality
)
```

#### Key Components:
- `FoldingMCTS` class extending `GeneralMCTS`
- Structure quality evaluation using ESMFold/AlphaFold metrics
- Coordinate-based masking strategies
- Structure refinement through iterative folding

### 2. Conditional Generation (Motif Scaffolding)
**Goal**: Generate sequences that fold around specified structural motifs

#### Implementation Plan:
- **Input**: Target motif structure + scaffold constraints
- **Output**: Complete sequence that incorporates the motif
- **MCTS Strategy**:
  - Mask non-motif regions for sequence generation
  - Preserve motif constraints during search
  - Use DPLM-2 conditional generation capabilities
  - Multiple experts with different conditioning strategies

#### Reward Function:
```python
conditional_reward = (
    0.40 * motif_preservation +    # Maintain motif structure
    0.30 * scaffold_quality +     # Overall fold quality
    0.20 * interface_quality +    # Motif-scaffold interface
    0.10 * designability         # Sequence feasibility
)
```

#### Key Components:
- `ConditionalMCTS` class with motif constraints
- Motif-aware masking strategies
- Interface quality evaluation
- Constraint satisfaction checking

### 3. Unconditional Generation
**Goal**: Generate novel, high-quality protein sequences from scratch

#### Implementation Plan:
- **Input**: Target length + optional structural preferences
- **Output**: Novel protein sequence with good structural properties
- **MCTS Strategy**:
  - Start from random/template sequence
  - Progressive refinement through MCTS search
  - Use structural priors for guidance
  - Multiple experts for diversity

#### Reward Function:
```python
unconditional_reward = (
    0.40 * structural_quality +   # Predicted fold quality
    0.25 * novelty_score +        # Distance from known proteins
    0.20 * designability +        # Sequence feasibility
    0.15 * diversity              # Amino acid composition
)
```

#### Key Components:
- `UnconditionalMCTS` class for de novo design
- Novelty evaluation against protein databases
- Structural quality prediction
- Diversity-driven exploration

## Implementation Roadmap

### Phase 1: Core Framework Extension (2-3 weeks)
1. **Refactor Base Classes**
   - Extract common MCTS logic into `BaseMCTS`
   - Create task-specific reward interfaces
   - Implement task-agnostic multiple experts system

2. **Task-Specific Classes**
   - `FoldingMCTS` for sequence → structure
   - `ConditionalMCTS` for motif scaffolding
   - `UnconditionalMCTS` for de novo generation

3. **Reward System Extension**
   - Modular reward computation system
   - Task-specific metric calculators
   - Configurable reward weighting

### Phase 2: Folding Task Implementation (2-3 weeks)
1. **Structure Prediction Integration**
   - ESMFold/AlphaFold integration for folding
   - Structure quality metrics (pLDDT, clashes, geometry)
   - Coordinate-based evaluation

2. **Folding-Specific MCTS**
   - Sequence masking strategies for folding
   - Structure-guided exploration
   - Multiple folding model experts

3. **Validation and Testing**
   - Test on CASP/CAMEO folding targets
   - Compare against baseline folding methods
   - Performance optimization

### Phase 3: Conditional Generation (3-4 weeks)
1. **Motif Scaffolding Framework**
   - Motif constraint representation
   - Scaffold region identification
   - Interface quality evaluation

2. **Conditional MCTS Implementation**
   - Constraint-aware masking
   - Motif preservation during search
   - Interface optimization

3. **Validation on Design Tasks**
   - Test on enzyme active sites
   - Binding site scaffolding
   - Compare with existing design tools

### Phase 4: Unconditional Generation (2-3 weeks)
1. **De Novo Design Framework**
   - Random sequence initialization
   - Structural quality prediction
   - Novelty evaluation system

2. **Unconditional MCTS**
   - Progressive sequence refinement
   - Diversity-driven exploration
   - Multi-objective optimization

3. **Novel Protein Evaluation**
   - Structural analysis of generated proteins
   - Comparison with natural proteins
   - Experimental validation planning

### Phase 5: Integration and Optimization (2-3 weeks)
1. **Unified Testing Framework**
   - Common evaluation metrics across tasks
   - Benchmarking suite for all tasks
   - Performance comparison tools

2. **Optimization and Scaling**
   - GPU acceleration for multiple experts
   - Parallel MCTS search
   - Memory optimization

3. **Documentation and Examples**
   - Task-specific tutorials
   - API documentation
   - Example workflows

## Technical Considerations

### Multiple Experts Extension
- **Current**: 3 models (650M, 150M, 3B) for inverse folding
- **Extension**: Task-specific expert ensembles
  - Folding: Different structure prediction models
  - Conditional: Various conditioning strategies
  - Unconditional: Diverse generation approaches

### Reward System Architecture
```python
class TaskRewardCalculator:
    def __init__(self, task_type: str, weights: Dict[str, float]):
        self.task_type = task_type
        self.weights = weights
        self.metrics = self._load_task_metrics()
    
    def calculate_reward(self, sequence: str, context: Dict) -> float:
        # Task-specific reward calculation
        pass
```

### MCTS Tree Adaptation
- **Node Representation**: Task-specific state representation
- **Action Space**: Task-appropriate masking/modification strategies  
- **Evaluation**: Task-specific simulation and rollout policies

## Success Metrics

### Folding Task
- **Primary**: Structure accuracy (GDT-TS, LDDT)
- **Secondary**: pLDDT confidence, geometric quality
- **Baseline**: ESMFold, AlphaFold2 predictions

### Conditional Generation
- **Primary**: Motif preservation (RMSD, interface quality)
- **Secondary**: Overall fold quality, designability
- **Baseline**: Existing motif scaffolding tools (RFdiffusion, ProteinMPNN)

### Unconditional Generation  
- **Primary**: Structural quality of generated sequences
- **Secondary**: Novelty, diversity, designability
- **Baseline**: Random sequences, existing generation methods

## Risk Mitigation

### Technical Risks
1. **Model Integration**: Different DPLM-2 task models may have incompatible interfaces
   - *Mitigation*: Create unified model wrapper interface

2. **Computational Cost**: Multiple experts across tasks may be expensive
   - *Mitigation*: Implement efficient batching and caching

3. **Reward Function Design**: Task-specific rewards may be difficult to balance
   - *Mitigation*: Extensive hyperparameter tuning and validation

### Scientific Risks
1. **Limited Improvement**: MCTS may not improve over baseline methods
   - *Mitigation*: Thorough baseline comparison and ablation studies

2. **Overfitting**: Task-specific optimizations may not generalize
   - *Mitigation*: Cross-validation on diverse test sets

## Expected Outcomes

### Short-term (3 months)
- Working implementations for all 4 tasks
- Demonstrated improvements over baselines on standard benchmarks
- Unified framework with consistent API

### Medium-term (6 months)  
- Published results showing MCTS improvements across protein design tasks
- Integration with existing protein design workflows
- Community adoption and feedback

### Long-term (12 months)
- State-of-the-art performance on multiple protein design benchmarks
- Extension to other protein modeling tasks
- Potential for experimental validation of designed proteins

## Conclusion

This framework extension will establish our MCTS approach as a general optimization method for protein design, moving beyond inverse folding to address the full spectrum of computational protein design challenges. The modular architecture ensures maintainability while the multiple experts approach leverages the full power of DPLM-2 across all tasks.
