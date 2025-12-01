# Task Organization for Fair Comparison

## 🎯 Design Goal
Ensure fair comparison across all 7 configurations even if jobs don't complete before deadline.

## 📊 Task Organization Strategy

### ✅ NEW ORGANIZATION (Fair Comparison)
```
Tasks 0-6:   Structures 0-10   → All 7 configs
Tasks 7-13:  Structures 10-20  → All 7 configs
Tasks 14-20: Structures 20-30  → All 7 configs
...
Tasks 126-132: Structures 180-190 → All 7 configs
```

**Benefits:**
- ✅ If 70 tasks complete → 10 structure batches with all 7 configs (fair comparison on 100 structures)
- ✅ If 105 tasks complete → 15 structure batches with all 7 configs (fair comparison on 150 structures)
- ✅ Always have complete data for comparison on completed structure batches

### ❌ OLD ORGANIZATION (Unfair)
```
Tasks 0-18:   Config 0 (Random) → All 190 structures
Tasks 19-37:  Config 1 (150M)   → All 190 structures
Tasks 38-56:  Config 2 (650M)   → All 190 structures
...
```

**Problems:**
- ❌ If 70 tasks complete → Only 3-4 configs have data, can't compare
- ❌ Unbalanced: some configs complete, others don't
- ❌ No fair comparison possible unless ALL tasks finish

## 📋 Example Scenarios

### Scenario 1: Only 49 tasks complete (7 structure batches)
- **Structures covered**: 0-70 (70 structures)
- **Configs available**: All 7 configs for each structure
- **Result**: ✅ Fair comparison on 70 structures

### Scenario 2: Only 84 tasks complete (12 structure batches)
- **Structures covered**: 0-120 (120 structures)
- **Configs available**: All 7 configs for each structure
- **Result**: ✅ Fair comparison on 120 structures

### Scenario 3: All 133 tasks complete (19 structure batches)
- **Structures covered**: 0-190 (190 structures)
- **Configs available**: All 7 configs for each structure
- **Result**: ✅ Fair comparison on all 190 structures

## 🔍 Task ID Mapping

For any task ID, you can determine:
- **Structure Batch**: `BATCH_ID = TASK_ID / 7`
- **Config ID**: `CONFIG_ID = TASK_ID % 7`
- **Structure Range**: `START = BATCH_ID * 10`, `END = START + 10`

**Examples:**
- Task 0: Batch 0 (0-10), Config 0 (Random)
- Task 1: Batch 0 (0-10), Config 1 (150M)
- Task 6: Batch 0 (0-10), Config 6 (MCTS-UCT)
- Task 7: Batch 1 (10-20), Config 0 (Random)
- Task 13: Batch 1 (10-20), Config 6 (MCTS-UCT)
- Task 14: Batch 2 (20-30), Config 0 (Random)

## 🎯 Priority for Deadline

If running out of time, the first N completed batches will give you:
- **First 7 tasks (0-6)**: 10 structures, all configs ✅
- **First 14 tasks (0-13)**: 20 structures, all configs ✅
- **First 21 tasks (0-20)**: 30 structures, all configs ✅
- **First 70 tasks (0-69)**: 100 structures, all configs ✅

This ensures you always have usable data for rebuttal!
