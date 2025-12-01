#!/bin/bash
# Cleanup failed tasks and identify what needs to be rerun

LOG_DIR="/home/caom/AID3/dplm/logs/mcts_folding_ablation"
JOB_ID="569308"

echo "🔍 Analyzing job ${JOB_ID} logs..."
echo ""

# Arrays to track task status
declare -a completed
declare -a failed
declare -a missing

# Check each task
for task_id in {0..132}; do
    out_file="${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.out"
    err_file="${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.err"
    
    if [ ! -f "$out_file" ]; then
        # Task never started or log missing
        missing+=($task_id)
    elif grep -q "SyntaxError" "$out_file" 2>/dev/null || grep -q "SyntaxError" "$err_file" 2>/dev/null; then
        # Syntax error - definitely failed
        failed+=($task_id)
        echo "❌ Task $task_id: SyntaxError"
    elif grep -q "MCTS search failed" "$out_file" 2>/dev/null; then
        # MCTS failed - needs rerun
        failed+=($task_id)
        echo "❌ Task $task_id: MCTS search failed"
    elif grep -q "Traceback (most recent call last)" "$out_file" 2>/dev/null; then
        # Python exception - failed
        failed+=($task_id)
        echo "❌ Task $task_id: Python exception"
    elif grep -q "completed successfully" "$out_file" 2>/dev/null && ! grep -q "MCTS search failed" "$out_file" 2>/dev/null; then
        # Actually completed successfully
        completed+=($task_id)
    else
        # Incomplete or unclear status
        failed+=($task_id)
        echo "⚠️  Task $task_id: Incomplete/unclear"
    fi
done

echo ""
echo "═══════════════════════════════════════════"
echo "📊 Summary:"
echo "═══════════════════════════════════════════"
echo "✅ Completed: ${#completed[@]} tasks"
echo "❌ Failed: ${#failed[@]} tasks"
echo "⚠️  Missing: ${#missing[@]} tasks"
echo ""

# Show completed tasks
if [ ${#completed[@]} -gt 0 ]; then
    echo "✅ Successfully completed tasks:"
    echo "${completed[@]}" | tr ' ' '\n' | sort -n | head -20
    if [ ${#completed[@]} -gt 20 ]; then
        echo "... (${#completed[@]} total)"
    fi
    echo ""
fi

# Combine failed and missing for rerun
all_rerun=("${failed[@]}" "${missing[@]}")
echo "📋 Tasks to rerun: ${#all_rerun[@]}"

# Sort and create comma-separated list
rerun_list=$(printf '%s\n' "${all_rerun[@]}" | sort -n | tr '\n' ',' | sed 's/,$//')
echo "$rerun_list"
echo ""

# Save to file
echo "$rerun_list" > /tmp/rerun_tasks_569308.txt
echo "💾 Saved rerun list to: /tmp/rerun_tasks_569308.txt"
echo ""

# Ask for confirmation before cleanup
echo "🗑️  Ready to delete failed/incomplete log files?"
echo "This will delete ${#all_rerun[@]} .out and .err files"
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Deleting failed task logs..."
    for task_id in "${all_rerun[@]}"; do
        rm -f "${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.out"
        rm -f "${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.err"
    done
    echo "✅ Cleanup complete!"
else
    echo "⏭️  Skipping cleanup"
fi
