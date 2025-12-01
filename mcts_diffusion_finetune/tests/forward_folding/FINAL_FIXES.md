# Final Fixes for Random Mode

## 🐛 Bugs Fixed

### 1. Missing `num_children_select` attribute
**Error**: `'GeneralMCTS' object has no attribute 'num_children_select'`

**Fix**: Added line 145 in `sequence_level_mcts.py`:
```python
self.num_children_select = kwargs.get('num_children_select', top_k_candidates)
```

### 2. Random mode now does actual random selection
**Before**: Just copied baseline structure tokens (no randomness)

**After**: Randomly perturbs structure tokens for folding tasks
- Randomly selects ~50% of masked positions
- Replaces them with random structure tokens (range 2816-7999)
- Converts to coordinates and evaluates reward
- Provides true random baseline for comparison

## 🎲 Random Mode Behavior

For **folding tasks**:
1. Takes baseline structure tokens from current node
2. Identifies masked positions (low pLDDT regions)
3. Randomly perturbs ~50% of masked positions with random structure tokens
4. Converts perturbed tokens → coordinates
5. Evaluates RMSD/TM-score reward
6. Creates candidate with random perturbations

For **inverse folding tasks**:
- Uses baseline sequence as random baseline

## ✅ Ready to Submit

All bugs fixed! You can now submit:

```bash
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
sbatch submit_mcts_folding_resume.sh
```

This will rerun 129 tasks (0, 5-132) with:
- ✅ Fixed syntax error
- ✅ Fixed `num_children_select` attribute
- ✅ Fixed pLDDT estimation crash
- ✅ Proper random structure token perturbation

## 📊 Expected Behavior

**Random mode (Config 0)** will now:
- Generate 4 random candidates per expansion
- Each candidate has randomly perturbed structure tokens
- Provides true random baseline for comparison with expert-guided methods
- Should show lower performance than expert methods (as expected for random baseline)
