# Shared settings for Qwen PCP (predexe) experiments on the extended dataset.
# Source from run/submit scripts; do not execute directly.

# Sampling seeds for the multi-seed runs (temperature 0.7 / top_p 0.8 in the
# model configs).
PCP_SEEDS=(73 94)

PCP_EXPECTED_UK=2946
PCP_EXPECTED_MK=5892

# Split each experiment into this many parallel shards (uk + mk).
PCP_TOTAL_SHARDS=4

# Build up to this many PCP dataset JSONL files concurrently (uk-SOS, uk-K, mk-SOS, mk-K).
PCP_DATASET_JOBS_PARALLEL=4

# vLLM continuous batching: all prompts in a shard are submitted at once and
# vLLM runs up to max_num_seqs (default 64) concurrently, bounded by the KV
# cache. Automatic GPU OOM fallback is enabled in VLLMRunner (max_num_seqs
# 64->32->...->1, then lower gpu_memory_utilization with max_num_seqs 1).
# Set PCP_DISABLE_OOM_FALLBACK=1 to turn off.
# Optional manual overrides: PCP_MAX_NUM_SEQS, PCP_GPU_MEMORY_UTILIZATION
# (PCP_BATCH_SIZE no longer affects vLLM inference concurrency).

# 8 experiments x 2 seeds x PCP_TOTAL_SHARDS = 64 Slurm array tasks per model.
PCP_QWEN_ARRAY_RANGE="0-63"
PCP_QWEN_ARRAY_THROTTLE="%16"

# Per-shard walltime safety margins (worst case ~ mk COT at 3x dataset size).
PCP_QWEN_WALLTIME_3B="18:00:00"
PCP_QWEN_WALLTIME_7B="14:00:00"
PCP_QWEN_WALLTIME_14B="14:00:00"
PCP_QWEN_WALLTIME_32B="16:00:00"       # DA / short runs (default in sbatch script)
PCP_QWEN_WALLTIME_32B_COT="24:00:00"

PCP_QWEN_SBATCH_SCRIPTS=(
  sbatch-pcp-qwen-3b.sh
  sbatch-pcp-qwen-7b.sh
  sbatch-pcp-qwen-14b.sh
  sbatch-pcp-qwen-32b.sh
)
