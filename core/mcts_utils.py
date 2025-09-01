"""
Utility functions for MCTS masked diffusion search.
"""
import hashlib
from typing import List, Set, Tuple, Dict, Any
import random
import math


def apply_patch(parent_seq: str, proposal_seq: str, mask_idxs: Set[int]) -> str:
    """Apply edits only at the chosen mask positions."""
    out = list(parent_seq)
    for i in mask_idxs:
        if i < len(proposal_seq):
            out[i] = proposal_seq[i]
    return ''.join(out)


def compute_mask_schedule(sequence: str, plddt_scores: List[float], depth: int, max_depth: int = 10) -> Set[int]:
    """
    Smarter progressive masking strategy with higher starting thresholds.
    
    Key principle: Start with meaningful mask sizes, progressively focus on worst regions.
    """
    # Smarter thresholds: start higher for meaningful optimization
    tau_start, tau_end = 0.75, 0.40  # Higher start threshold for more positions
    progress = min(1.0, depth / max_depth) if max_depth > 0 else 0
    tau = tau_start - (tau_start - tau_end) * progress
    
    # Find low-confidence positions
    low_confidence = [i for i, plddt in enumerate(plddt_scores) if plddt < tau]
    
    # Sort by confidence (lowest first) for prioritization
    if low_confidence:
        scored_positions = [(i, plddt_scores[i]) for i in low_confidence]
        scored_positions.sort(key=lambda x: x[1])  # Sort by pLDDT ascending
        mask = [pos for pos, _ in scored_positions]
    else:
        mask = []
    
    # Smarter mask sizing: depth-dependent strategy
    if depth == 0:
        # Root: larger mask for exploration (10-20%)
        min_positions = max(5, int(len(sequence) * 0.10))
        max_positions = int(len(sequence) * 0.20)
    elif depth <= 2:
        # Early depths: moderate masks (5-15%)
        min_positions = max(3, int(len(sequence) * 0.05))
        max_positions = int(len(sequence) * 0.15)
    else:
        # Deep levels: focused masks (1-10%)
        min_positions = 1
        max_positions = max(5, int(len(sequence) * 0.10))
    
    if len(mask) > max_positions:
        # Keep only the lowest confidence positions
        mask = mask[:max_positions]
    elif len(mask) < min_positions and depth < max_depth:
        # Add medium confidence positions if needed
        medium_conf = [i for i, plddt in enumerate(plddt_scores) 
                      if tau <= plddt < (tau + 0.15)]
        if medium_conf:
            needed = min_positions - len(mask)
            additional = random.sample(medium_conf, min(needed, len(medium_conf)))
            mask.extend(additional)
    
    print(f"Depth {depth}: threshold={tau:.2f}, found {len(mask)} positions to mask")
    return set(mask)


def compute_sequence_hash(sequence: str) -> str:
    """Compute hash for sequence deduplication."""
    return hashlib.md5(sequence.encode()).hexdigest()[:16]


def compute_hamming_distance(seq1: str, seq2: str) -> int:
    """Compute Hamming distance between two sequences."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def dedupe_and_filter(proposals: List[Tuple[str, float, str]], min_hamming: int = 8) -> List[Tuple[str, float, str]]:
    """Remove near-duplicates and enforce diversity."""
    if not proposals:
        return []
    
    # Sort by score first
    proposals = sorted(proposals, key=lambda x: x[1], reverse=True)
    
    # Keep diverse proposals
    kept = [proposals[0]]  # Always keep the best
    
    for seq, score, meta in proposals[1:]:
        # Check diversity against all kept sequences
        diverse = all(compute_hamming_distance(seq, kept_seq) >= min_hamming 
                     for kept_seq, _, _ in kept)
        if diverse:
            kept.append((seq, score, meta))
    
    return kept


def compute_fast_aar(sequence: str, reference: str) -> float:
    """Fast AAR calculation for proxy scoring."""
    if len(sequence) != len(reference):
        min_len = min(len(sequence), len(reference))
        sequence = sequence[:min_len]
        reference = reference[:min_len]
    
    matches = sum(s == r for s, r in zip(sequence, reference))
    return matches / len(sequence) if sequence else 0.0


def ph_uct_score(average_value: float, visit_count: int, parent_visits: int, 
                 c: float = 1.414, w_ent: float = 0.1, w_div: float = 0.1, 
                 entropy_proposals: float = 0.0, novelty_vs_parent: float = 0.0) -> float:
    """PH-UCT score using parent visits and cached priors."""
    if visit_count == 0:
        return float('inf')
    
    # Core UCB
    ucb = average_value + c * math.sqrt(math.log(parent_visits + 1) / (visit_count + 1))
    
    # PH terms (cached at expansion time)
    ph_bonus = w_ent * entropy_proposals + w_div * novelty_vs_parent
    
    return ucb + ph_bonus


class SequenceCache:
    """Cache for sequence evaluations to avoid recomputation."""
    
    def __init__(self):
        self.aar_cache = {}
        self.sctm_cache = {}
        self.biophys_cache = {}
        self.rewards = {}  # Add missing rewards cache
    
    def get_aar(self, seq_hash: str) -> float:
        return self.aar_cache.get(seq_hash)
    
    def set_aar(self, seq_hash: str, aar: float):
        self.aar_cache[seq_hash] = aar
    
    def get_sctm(self, seq_hash: str) -> float:
        return self.sctm_cache.get(seq_hash)
    
    def set_sctm(self, seq_hash: str, sctm: float):
        self.sctm_cache[seq_hash] = sctm
    
    def get_biophys(self, seq_hash: str) -> float:
        return self.biophys_cache.get(seq_hash)
    
    def set_biophys(self, seq_hash: str, biophys: float):
        self.biophys_cache[seq_hash] = biophys
    
    def get_reward(self, seq_hash: str) -> float:
        return self.rewards.get(seq_hash)
    
    def set_reward(self, seq_hash: str, reward: float):
        self.rewards[seq_hash] = reward
