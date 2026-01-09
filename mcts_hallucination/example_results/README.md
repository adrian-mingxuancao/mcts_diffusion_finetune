# Hallucination MCTS Example Results

Comparison of different initialization modes and backends for 100aa protein hallucination.

## Results Summary

| Init Mode | Backend | Masking | Avg pLDDT | Min | Max | Notes |
|-----------|---------|---------|-----------|-----|-----|-------|
| all_x | Boltz | inherit_seq | ~84 | 62.8 | 94.0 | **Best** - ProteinHunter style with seq inheritance |
| all_x | Boltz | full_mask | ~83 | 65.2 | 95.2 | Full redesign each iteration |
| random | Boltz | full_mask | ~53 | 41.5 | 73.5 | HalluDesign style |
| all_x | ESMFold | full_mask | ~36 | 26.4 | 42.8 | High initial pLDDT but drops after redesign |
| random | ESMFold | full_mask | ~31 | 26.6 | 37.4 | Poor convergence |

## Convergence Analysis: Does More Compute Help?

### By Iteration (pLDDT avg / max)

| Config | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Trend |
|--------|--------|--------|--------|--------|--------|-------|
| Boltz + inherit_seq | 85.8/92.6 | 81.8/91.0 | 86.2/94.0 | 82.6/89.9 | 82.7/90.1 | **Stable ~83-86** |
| Boltz + full_mask | 90.5/95.2 | 78.1/92.2 | 87.4/94.4 | 84.5/90.2 | 73.4/82.2 | Declining |
| Random + Boltz | 51.9/54.9 | 47.8/52.5 | 59.9/73.5 | 53.8/58.6 | - | Variable |
| ESMFold all_x | 38.0/42.2 | 33.0/36.4 | 35.9/40.7 | 37.9/42.3 | 36.5/42.8 | Flat ~36 |

### By Depth (pLDDT avg / max)

| Config | Depth 1 | Depth 2 | Trend |
|--------|---------|---------|-------|
| Boltz + inherit_seq | 85.8/92.6 | 83.3/94.0 | Slight improvement at depth 2 |
| Boltz + full_mask | 90.5/95.2 | 80.8/94.4 | Drops at depth 2 |
| Random + Boltz | 51.9/54.9 | 53.9/73.5 | Slight improvement |
| ESMFold | 38.0/42.2 | 35.8/42.8 | No improvement |

### Key Observations

1. **Sequence inheritance (ProteinHunter-style) provides more stable convergence**
   - With `inherit_seq`: pLDDT stays stable at ~83-86 across iterations
   - With `full_mask`: pLDDT starts high (90.5) but declines to 73.4 by iter 5

2. **Deeper trees show marginal improvement with Boltz**
   - Best structures (max pLDDT) found at depth 2 for both masking strategies
   - ESMFold shows no improvement with depth

3. **More iterations help find better max pLDDT but don't improve average**
   - The MCTS exploration finds occasional high-quality structures
   - Average pLDDT doesn't consistently improve with more iterations

4. **Boltz >> ESMFold regardless of masking strategy**
   - Boltz maintains 80+ pLDDT; ESMFold stuck at ~36

## Key Findings

1. **Boltz >> ESMFold** for hallucination design
   - Boltz uses diffusion to generate diverse structures from X tokens
   - ESMFold with X tokens gives high initial pLDDT (96.9) but ProteinMPNN redesigns don't fold well

2. **Sequence inheritance improves stability**
   - Inheriting parent sequence (only redesigning X positions) gives more stable convergence
   - Full masking (redesigning all positions) leads to pLDDT decline over iterations

3. **all-X init >> random init** with Boltz
   - all-X allows Boltz to hallucinate optimal backbone topology
   - random init constrains the structure to fit an arbitrary sequence

4. **Convergence metrics** (from Protein Hunter / HalluDesign papers):
   - Primary: **pLDDT** (structure confidence)
   - Secondary: **scTM** (self-consistent TM-score via refolding)
   - NOT parent-child sequence similarity

## Directory Structure

- `all_x_init_boltz/` - Results from all-X initialization with Boltz (full masking)
- `all_x_init_boltz_inherit_seq/` - Results with sequence inheritance (ProteinHunter-style)
- `random_init_boltz/` - Results from random sequence initialization with Boltz
- `all_x_init_esmfold/` - Results from all-X initialization with ESMFold
- `random_init_esmfold/` - Results from random sequence initialization with ESMFold

## How to Run

```bash
# Protein Hunter style (all-X init)
python test_integration.py --real --length 100 --iterations 5 --init-mode all_x --backend boltz --num-candidates 2

# HalluDesign style (random init)
python test_integration.py --real --length 100 --iterations 5 --init-mode random --backend boltz --num-candidates 2
```

## PDB File Naming

Format: `iter{N}_depth{D}_node{ID}_plddt{SCORE}.pdb`

- `iter`: MCTS iteration number
- `depth`: Tree depth (0=root, 1=first cycle, etc.)
- `node`: Unique node ID
- `plddt`: Mean pLDDT score

Files ending with `_BEST.pdb` are the best converged nodes selected by the MCTS algorithm.
