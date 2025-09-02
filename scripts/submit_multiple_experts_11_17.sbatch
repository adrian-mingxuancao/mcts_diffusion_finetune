#!/bin/bash
#SBATCH --job-name=cameo_multiexpert_11_17
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00  # 8 hours should be enough for 7 structures
#SBATCH --output=/net/scratch/caom/cameo_multiexpert_11_17_%j.out
#SBATCH --error=/net/scratch/caom/cameo_multiexpert_11_17_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=caom@example.com

# Print job information
echo "======================================================================"
echo "CAMEO 2022 Multiple Experts Rollout - Structures 11-17"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
echo ""

# Set up environment
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH

# Set up CUDA environment
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set up cache directories
export HF_HOME=/net/scratch/caom/huggingface_cache
export TRITON_CACHE_DIR=/net/scratch/caom/triton_cache
mkdir -p $HF_HOME $TRITON_CACHE_DIR

# Change to working directory
cd /home/caom/AID3/dplm/mcts_diffusion_finetune

# Print environment info
echo "Environment setup:"
echo "  Python: $(which python)"
echo "  CUDA version: $(nvcc --version | grep release || echo 'CUDA not found')"
echo "  GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits || echo 'GPU not found')"
echo "  Working directory: $(pwd)"
echo ""

# Create output directory
OUTPUT_DIR="/net/scratch/caom/cameo_evaluation_results"
mkdir -p $OUTPUT_DIR
LOG_FILE="$OUTPUT_DIR/multiple_experts_11_17_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Results will be saved automatically by the Python script"
echo ""

# Run multiple experts evaluation on structures 11-17
echo "======================================================================"
echo "Starting MULTIPLE EXPERTS rollout on CAMEO structures 11-17..."
echo "======================================================================"
echo "This will run multiple experts MCTS on the remaining structures"
echo "Expected duration: 6-8 hours for 7 structures"
echo "Progress will be saved to: $OUTPUT_DIR"
echo ""
echo "🎯 MULTIPLE EXPERTS SCOPE:"
echo "• CAMEO 2022 structures 11-17 (7 structures)"
echo "• DPLM-2 150M baseline sequences for all tests"
echo "• Multiple experts rollout: 3 experts, 3 rollouts each, 2 children selected"
echo "• AAR improvement tracking for each structure"
echo "• Results saved automatically to JSON"
echo ""
echo "⏱️  TIME CONSTRAINTS:"
echo "• General partition limit: 12 hours maximum"
echo "• Job timeout: 8 hours (safe margin)"
echo "• 30 MCTS iterations per test for efficiency"
echo "• Results saved automatically to JSON"

# Set a timeout for the Python script (7 hours to be safe)
TIMEOUT_HOURS=7
TIMEOUT_SECONDS=$((TIMEOUT_HOURS * 3600))

echo "Setting timeout: ${TIMEOUT_HOURS}h (${TIMEOUT_SECONDS}s)"
echo "Starting multiple experts evaluation at: $(date)"
echo ""

# Run the ablation script with multiple experts mode on structures 11-17
timeout $TIMEOUT_SECONDS python tests/mcts_tree_search_ablation.py 11 18 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

# Check if evaluation completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Multiple experts evaluation completed successfully!"
    echo "======================================================================"
    echo "Log saved to: $LOG_FILE"
    
    # Find the most recent results file
    LATEST_JSON=$(ls -t $OUTPUT_DIR/mcts_ablation_results_*.json 2>/dev/null | head -1)
    
    if [ -f "$LATEST_JSON" ]; then
        echo "📊 Results saved to: $LATEST_JSON"
        
        # Display final summary
        echo ""
        echo "======================================================================"
        echo "📊 FINAL SUMMARY - Multiple Experts 11-17"
        echo "======================================================================"
        echo "✅ Multiple experts rollout completed for structures 11-17"
        echo "📊 Results saved to: $LATEST_JSON"
        echo "🎯 This completes your CAMEO evaluation dataset!"
        echo ""
        echo "📈 COMPLETE CAMEO EVALUATION STATUS:"
        echo "• Structures 0-10: Multiple experts ✅ (already completed)"
        echo "• Structures 11-17: Multiple experts ✅ (just completed)"
        echo "• Total: 17 structures with multiple experts rollout"
        
    else
        echo "⚠️ Results file not found - check the log for details"
    fi
    
elif [ $EXIT_CODE -eq 124 ]; then
    echo ""
    echo "======================================================================"
    echo "⏰ Multiple experts evaluation TIMED OUT after ${TIMEOUT_HOURS} hours!"
    echo "======================================================================"
    echo "This suggests the evaluation is taking longer than expected"
    echo "Partial results may be available in the output directory"
    echo "Log saved to: $LOG_FILE"
    
    # Check for partial results
    LATEST_JSON=$(ls -t $OUTPUT_DIR/mcts_ablation_results_*.json 2>/dev/null | head -1)
    if [ -f "$LATEST_JSON" ]; then
        echo "📊 Partial results found: $LATEST_JSON"
        echo "💡 Consider increasing time limit or reducing iterations"
    fi
    
else
    echo ""
    echo "======================================================================"
    echo "❌ Multiple experts evaluation failed with exit code: $EXIT_CODE"
    echo "======================================================================"
    echo "Check log file: $LOG_FILE"
    echo "Error log: /net/scratch/caom/cameo_multiexpert_11_17_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "Total wall time: $(squeue -j $SLURM_JOB_ID --format=%M --noheader 2>/dev/null || echo 'Unknown')"
