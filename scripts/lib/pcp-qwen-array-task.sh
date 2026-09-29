# Shared Slurm array task runner for Qwen PCP experiments.
# Source from sbatch-pcp-qwen-{3,7,14,32}b.sh after setting REPO, EXPS, and SEEDS.

pcp_qwen_results_file_for() {
  local exp=$1
  local seed=$2
  local strategy_override=${3:-}
  local setup="" expr="" strategy="" model=""

  if [[ "${exp}" == *_uk_* ]]; then
    setup=uk
  elif [[ "${exp}" == *_mk_* ]]; then
    setup=mk
  else
    echo "ERROR: cannot parse setup from ${exp}" >&2
    return 1
  fi

  if [[ -n "${strategy_override}" ]]; then
    strategy="${strategy_override}"
  elif [[ "${exp}" == *_cot_* ]]; then
    strategy=cot
  else
    strategy=da
  fi

  if [[ "${exp}" == *_imp_sos ]]; then
    expr=IMP-SOS
  elif [[ "${exp}" == *_imp_k ]]; then
    expr=IMP-K
  else
    echo "ERROR: cannot parse semantics from ${exp}" >&2
    return 1
  fi

  if [[ "${exp}" == qwen_coder_32b_* ]]; then
    model="Qwen-Qwen2.5-Coder-32B-Instruct"
  elif [[ "${exp}" == qwen_coder_14b_* ]]; then
    model="Qwen-Qwen2.5-Coder-14B-Instruct"
  elif [[ "${exp}" == qwen_coder_7b_* ]]; then
    model="Qwen-Qwen2.5-Coder-7B-Instruct"
  elif [[ "${exp}" == qwen_coder_3b_* ]]; then
    model="Qwen-Qwen2.5-Coder-3B-Instruct"
  elif [[ "${exp}" == ministral_8b_* ]]; then
    model="mistralai-Ministral-3-8B-Instruct-2512-BF16"
  elif [[ "${exp}" == ministral_14b_* ]]; then
    model="mistralai-Ministral-3-14B-Instruct-2512-BF16"
  elif [[ "${exp}" == deepseek_qwen_14b_* ]]; then
    model="deepseek-ai-DeepSeek-R1-Distill-Qwen-14B"
  elif [[ "${exp}" == deepseek_qwen_32b_* ]]; then
    model="deepseek-ai-DeepSeek-R1-Distill-Qwen-32B"
  else
    echo "ERROR: cannot parse model from ${exp}" >&2
    return 1
  fi

  echo "${REPO}/results/pcp/${setup}/${expr}/${model}-${strategy}/results-${seed}.jsonl"
}

pcp_qwen_shard_results_file_for() {
  local merged_file=$1
  local shard=$2
  local total_shards=$3
  echo "${merged_file%.jsonl}.shard${shard}-of-${total_shards}.jsonl"
}

pcp_qwen_expected_lines_for() {
  local exp=$1
  if [[ "${exp}" == *_uk_* ]]; then
    echo "${PCP_EXPECTED_UK:-2946}"
  elif [[ "${exp}" == *_mk_* ]]; then
    echo "${PCP_EXPECTED_MK:-5892}"
  else
    echo "ERROR: cannot parse setup from ${exp}" >&2
    return 1
  fi
}

pcp_qwen_count_lines() {
  wc -l < "$1" | tr -d ' '
}

pcp_qwen_results_complete() {
  local results_file=$1
  local expected_lines=$2
  [[ -f "${results_file}" ]] \
    && [[ "$(pcp_qwen_count_lines "${results_file}")" -eq "${expected_lines}" ]]
}

pcp_qwen_expected_shard_lines() {
  local exp=$1
  local shard=$2
  local total_shards=$3
  local total
  total="$(pcp_qwen_expected_lines_for "${exp}")"
  local per=$((total / total_shards))
  if [[ "${shard}" -eq $((total_shards - 1)) ]]; then
    echo $((total - per * (total_shards - 1)))
  else
    echo "${per}"
  fi
}

pcp_qwen_shard_complete() {
  local shard_file=$1
  local expected_lines=$2
  [[ -f "${shard_file}" ]] \
    && [[ "$(pcp_qwen_count_lines "${shard_file}")" -eq "${expected_lines}" ]]
}

pcp_qwen_run_array_task() {
  : "${EXPS:?EXPS must be set}"
  : "${SEEDS:?SEEDS must be set}"
  : "${REPO:?REPO must be set}"

  local task_id="${SLURM_ARRAY_TASK_ID:-}"
  if [[ -z "${task_id}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set (submit with sbatch)" >&2
    exit 1
  fi

  local total_shards="${PCP_TOTAL_SHARDS:-1}"
  local num_seeds=${#SEEDS[@]}
  local tasks_per_exp=$((num_seeds * total_shards))
  local exp_idx=$((task_id / tasks_per_exp))
  local remainder=$((task_id % tasks_per_exp))
  local shard_idx=$((remainder / num_seeds))
  local seed_idx=$((remainder % num_seeds))

  if [[ "${exp_idx}" -ge "${#EXPS[@]}" ]]; then
    echo "ERROR: array task ${task_id} out of range for ${#EXPS[@]} experiments x ${num_seeds} seeds x ${total_shards} shards" >&2
    exit 1
  fi

  local exp="${EXPS[${exp_idx}]}"
  local seed="${SEEDS[${seed_idx}]}"
  local expected_lines
  expected_lines="$(pcp_qwen_expected_lines_for "${exp}")"
  local results_file
  results_file="$(pcp_qwen_results_file_for "${exp}" "${seed}")"
  local shard_file
  shard_file="$(pcp_qwen_shard_results_file_for "${results_file}" "${shard_idx}" "${total_shards}")"
  local expected_shard_lines
  expected_shard_lines="$(pcp_qwen_expected_shard_lines "${exp}" "${shard_idx}" "${total_shards}")"

  echo "Array task: ${task_id} (job ${SLURM_ARRAY_JOB_ID:-?}, exp_idx=${exp_idx}, seed_idx=${seed_idx}, shard_idx=${shard_idx})"
  echo "Experiment: ${exp}"
  echo "Seed: ${seed}"
  echo "Shard: ${shard_idx}/${total_shards}"
  echo "Merged results file: ${results_file}"
  echo "Shard results file: ${shard_file}"
  echo "Expected shard lines: ${expected_shard_lines}"

  if pcp_qwen_results_complete "${results_file}" "${expected_lines}"; then
    echo "Skipping: merged results already complete (${expected_lines} lines)"
    return 0
  fi

  if pcp_qwen_shard_complete "${shard_file}" "${expected_shard_lines}"; then
    echo "Skipping inference: shard results already complete (${expected_shard_lines} lines)"
    return 0
  fi

  if [[ -f "${shard_file}" ]]; then
    echo "Removing incomplete shard file (${shard_file}, $(pcp_qwen_count_lines "${shard_file}") lines, expected ${expected_shard_lines})"
    rm -f "${shard_file}"
  fi

  echo "========== Running ${exp} (seed=${seed}, shard=${shard_idx}/${total_shards}) =========="
  echo "OOM fallback: enabled (set PCP_DISABLE_OOM_FALLBACK=1 to disable)"
  export PCP_SHARD="${shard_idx}"
  export PCP_TOTAL_SHARDS="${total_shards}"
  python src/main.py experiment "${exp}" --seed "${seed}"
  exp_status=$?
  if [[ "${exp_status}" -ne 0 ]]; then
    echo "ERROR: experiment failed after OOM fallback attempts (exit ${exp_status})" >&2
    exit "${exp_status}"
  fi
}
