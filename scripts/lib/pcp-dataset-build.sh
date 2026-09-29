# Shared helpers for parallel PCP dataset JSONL generation.
# Source from run-pcp-datasets-and-submit.sh after setting REPO and PCP_EXPECTED_*.

pcp_dataset_log_dir() {
  echo "${REPO}/scratch/run_logs/pcp-dataset-build"
}

pcp_dataset_needs_build() {
  local output_file=$1
  local expected_lines=$2
  [[ ! -f "${output_file}" ]] || [[ "$(wc -l < "${output_file}" | tr -d ' ')" -ne "${expected_lines}" ]]
}

pcp_dataset_run_make() {
  local command_name=$1
  local log_file=$2
  (
    cd "${REPO}/src"
    echo "START ${command_name}: $(date)" >> "${log_file}"
    python main.py data "${command_name}" >> "${log_file}" 2>&1
    status=$?
    echo "END ${command_name}: $(date) exit=${status}" >> "${log_file}"
    exit "${status}"
  )
}

pcp_dataset_build_parallel() {
  local max_jobs="${PCP_DATASET_JOBS_PARALLEL:-4}"
  local log_dir
  log_dir="$(pcp_dataset_log_dir)"
  mkdir -p "${log_dir}"

  local -a jobs=()
  local -a logs=()

  if pcp_dataset_needs_build "${UK_SOS}" "${PCP_EXPECTED_UK}"; then
    jobs+=("make_pcp_unmutated_imp_sos_dataset")
    logs+=("${log_dir}/make_pcp_unmutated_imp_sos_dataset.log")
  else
    echo "Skipping make_pcp_unmutated_imp_sos_dataset (${UK_SOS} already has ${PCP_EXPECTED_UK} lines)."
  fi

  if pcp_dataset_needs_build "${UK_K}" "${PCP_EXPECTED_UK}"; then
    jobs+=("make_pcp_unmutated_imp_k_dataset")
    logs+=("${log_dir}/make_pcp_unmutated_imp_k_dataset.log")
  else
    echo "Skipping make_pcp_unmutated_imp_k_dataset (${UK_K} already has ${PCP_EXPECTED_UK} lines)."
  fi

  if pcp_dataset_needs_build "${MK_SOS}" "${PCP_EXPECTED_MK}"; then
    jobs+=("make_pcp_mutated_imp_sos_dataset")
    logs+=("${log_dir}/make_pcp_mutated_imp_sos_dataset.log")
  else
    echo "Skipping make_pcp_mutated_imp_sos_dataset (${MK_SOS} already has ${PCP_EXPECTED_MK} lines)."
  fi

  if pcp_dataset_needs_build "${MK_K}" "${PCP_EXPECTED_MK}"; then
    jobs+=("make_pcp_mutated_imp_k_dataset")
    logs+=("${log_dir}/make_pcp_mutated_imp_k_dataset.log")
  else
    echo "Skipping make_pcp_mutated_imp_k_dataset (${MK_K} already has ${PCP_EXPECTED_MK} lines)."
  fi

  if [[ "${#jobs[@]}" -eq 0 ]]; then
    echo "All four PCP dataset JSONL files already present with expected line counts."
    return 0
  fi

  echo ""
  echo "========== Building ${#jobs[@]} PCP dataset(s) in parallel (max ${max_jobs} at once) =========="
  for i in "${!jobs[@]}"; do
    echo "  ${jobs[$i]} -> ${logs[$i]}"
  done

  local -a pids=()
  local -a active_cmds=()
  local failed=0

  wait_for_slot() {
    while [[ "${#pids[@]}" -ge "${max_jobs}" ]]; do
      local new_pids=()
      local new_cmds=()
      for idx in "${!pids[@]}"; do
        if kill -0 "${pids[$idx]}" 2>/dev/null; then
          new_pids+=("${pids[$idx]}")
          new_cmds+=("${active_cmds[$idx]}")
          continue
        fi
        if ! wait "${pids[$idx]}"; then
          echo "ERROR: ${active_cmds[$idx]} failed (see ${log_dir}/)" >&2
          failed=1
        fi
      done
      pids=("${new_pids[@]}")
      active_cmds=("${new_cmds[@]}")
      if [[ "${#pids[@]}" -ge "${max_jobs}" ]]; then
        sleep 2
      fi
    done
  }

  for i in "${!jobs[@]}"; do
    wait_for_slot
    echo "Launching ${jobs[$i]}..."
    pcp_dataset_run_make "${jobs[$i]}" "${logs[$i]}" &
    pids+=("$!")
    active_cmds+=("${jobs[$i]}")
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "ERROR: ${active_cmds[$idx]} failed (see ${log_dir}/)" >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "One or more dataset build jobs failed. Logs:" >&2
    for log_file in "${logs[@]}"; do
      echo "  ${log_file}" >&2
    done
    return 1
  fi

  echo "All dataset build jobs completed successfully."
}
