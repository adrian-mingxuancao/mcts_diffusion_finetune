# MCTS/Imitation Learning Framework for Protein Language Models

## Overview
This project implements a general, modular framework for finetuning protein language models (PLMs)—mainly for diffusion-based models—using expert rollouts (e.g., Monte Carlo Tree Search, MCTS) and imitation learning. The goal is to improve performance on a variety of downstream protein tasks, such as inverse folding, representation learning, structure prediction, and more. Inverse folding is used as an initial testbed, but the framework is designed to be extensible to new tasks, models, and reward functions.

## Motivation & Novelty
- **Why Diffusion Models?** Protein sequences and structures are long and complex. Autoregressive transformer models struggle with long-range dependencies and slow, sequential generation. Diffusion models, by contrast, enable simultaneous generation and masking, making them more scalable and efficient for large proteins and long sequences. This property is leveraged for more effective tree search and expert rollouts.
- **Key Novelty:** This framework is among the first to combine Monte Carlo Tree Search (MCTS) with diffusion-based protein language models for downstream protein tasks. The simultaneous, non-autoregressive nature of diffusion models enables more efficient and flexible tree building and exploration compared to traditional autoregressive approaches.

## Key Features
- **Task-agnostic core:** Modular design to support multiple downstream tasks.
- **Expert/Imitation Learning:** Use MCTS or other expert/planning methods to generate high-quality rollouts for imitation learning or RL.
- **Flexible Reward Functions:** Easily define new reward/value functions for different tasks (e.g., TM-score, plDDT, representation similarity).
- **Model Agnostic:** Supports both diffusion-based and other PLMs.
- **Extensible Evaluation:** Plug in new tasks, reward functions, and evaluation metrics with minimal code changes.

## Framework Structure
1. **Core Framework**: Modular, extensible codebase for expert-guided finetuning of PLMs.
2. **Task Modules**: Pluggable modules for different downstream tasks (e.g., inverse folding, representation learning, structure prediction).
3. **Expert/Planning Module**: MCTS or other tree search/expert rollout logic, supporting flexible action/state spaces.
4. **Reward/Value Module**: Task-specific reward functions, supporting both partial and full rollouts.
5. **Learning/Finetuning Module**: Imitation learning (behavior cloning) and RL (policy/value updates).
6. **Evaluation Module**: Task-specific metrics and benchmarking.

## Example Use Case
- **Inverse folding** (structure → sequence) as the first implemented task.

## Pipeline (with plDDT-based Masking)
1. Input: Target protein structure
2. Predict initial sequence using diffusion model
3. Scan sequence for positions with lowest plDDT (confidence)
4. Mask low-confidence positions (practical, not novel)
5. Use MCTS to sample candidate amino acids for masked positions, leveraging diffusion's simultaneous update capability
6. Predict structure and compute reward (e.g., TM-score, plDDT)
7. Backpropagate rewards and update model via imitation learning or RL

## References
- [ProtInvTree: Reward-guided Tree Search for Protein Inverse Folding](https://arxiv.org/pdf/2506.00925)

## TODO
- [ ] Scaffold modular codebase
- [ ] Implement core expert/planning module (MCTS)
- [ ] Add task modules (start with inverse folding)
- [ ] Integrate reward/value functions
- [ ] Add learning/finetuning logic (imitation learning, RL)
- [ ] Add evaluation/benchmarking tools
- [ ] Document extensibility for new tasks and models 