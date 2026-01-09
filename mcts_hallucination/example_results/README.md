# Hallucination MCTS Example Results

Comparison of different initialization modes and backends for 100aa protein hallucination.

## Results Summary

| Init Mode | Backend | Avg pLDDT | Min | Max | Notes |
|-----------|---------|-----------|-----|-----|-------|
| all_x | Boltz | ~84 | 65.2 | 95.2 | **Best** - Protein Hunter style |
| random | Boltz | ~53 | 41.5 | 73.5 | HalluDesign style |
| all_x | ESMFold (no X→A) | ~36 | 26.4 | 42.8 | High initial pLDDT (96.9) but drops after redesign |
| random | ESMFold | ~31 | 26.6 | 37.4 | Poor convergence |

## Key Findings

1. **Boltz >> ESMFold** for hallucination design
   - Boltz uses diffusion to generate diverse structures from X tokens
   - ESMFold with X tokens gives high initial pLDDT (96.9) but ProteinMPNN redesigns don't fold well

2. **all-X init >> random init** with Boltz
   - all-X allows Boltz to hallucinate optimal backbone topology
   - random init constrains the structure to fit an arbitrary sequence

3. **ESMFold all-X behavior** (without X→A conversion):
   - Initial structure from XXX... has pLDDT=96.9 (very high!)
   - But after ProteinMPNN redesign, pLDDT drops to 31-42
   - The X-token structure may not be "designable" by ProteinMPNN

4. **Convergence metrics** (from Protein Hunter / HalluDesign papers):
   - Primary: **pLDDT** (structure confidence)
   - Secondary: **scTM** (self-consistent TM-score via refolding)
   - NOT parent-child sequence similarity

## Directory Structure

- `all_x_init_boltz/` - Results from all-X initialization with Boltz (Protein Hunter style)
- `random_init_boltz/` - Results from random sequence initialization with Boltz (HalluDesign style)
- `all_x_init_esmfold/` - Results from all-X initialization with ESMFold (no X→A conversion)
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
