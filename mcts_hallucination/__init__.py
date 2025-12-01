"""
Hallucination-based MCTS for Protein Design

This package implements MCTS-guided protein hallucination using:
- AlphaFold3 (via ABCFold) for structure hallucination
- ProteinMPNN for inverse folding
- MCTS for guided exploration

Inspired by:
- Protein Hunter: https://www.biorxiv.org/content/10.1101/2025.10.10.681530v1
- Halludesign: https://www.biorxiv.org/content/10.1101/2025.11.08.686881v1
"""

__version__ = "0.1.0"
