#!/usr/bin/env bash
# Reference vs ProteoForge: zebrac profiling + output parity.
#
# Usage:
#   tools/compare/zebrac.sh
#   tools/compare/zebrac.sh --cases medium large xlarge
#   tools/compare/zebrac.sh --input path.parquet --config config.yaml --case my-run
#
# Requires:
#   - tools/zebrac
#   - ref/ProteoForge_analysis_src/normalize.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZEBRAC="${ROOT}/tools/zebrac"
COMPARE="${ROOT}/tools/compare"
WORK_BASE="${ROOT}/tmp/compare"
SYNTH_ROOT="${ROOT}/benchmarks/fixtures/synthetic"
DEFAULT_INPUT="${ROOT}/benchmarks/fixtures/complete/complete-real.parquet"
DEFAULT_CONFIG="${ROOT}/benchmarks/fixtures/complete/config.yaml"

INPUT="${DEFAULT_INPUT}"
CONFIG="${DEFAULT_CONFIG}"
CASES=()
CUSTOM_CASE=""
DURATION=8000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --case)
      CUSTOM_CASE="$2"
      shift 2
      ;;
    --cases)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CASES+=("$1")
        shift
      done
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${ZEBRAC}" ]]; then
  echo "zebrac not found at ${ZEBRAC}" >&2
  exit 1
fi

if [[ ! -f "${ROOT}/ref/ProteoForge_analysis_src/normalize.py" ]]; then
  echo "Reference normalize.py not found under ref/" >&2
  exit 1
fi

if [[ -n "${CUSTOM_CASE}" ]]; then
  CASES=("${CUSTOM_CASE}")
elif [[ ${#CASES[@]} -eq 0 ]]; then
  CASES=("complete")
fi

needs_synth=false
for case in "${CASES[@]}"; do
  if [[ "${case}" != "complete" ]]; then
    needs_synth=true
    break
  fi
done

if [[ "${needs_synth}" == true ]]; then
  synth_tiers=()
  for case in "${CASES[@]}"; do
    if [[ "${case}" != "complete" ]]; then
      synth_tiers+=("${case}")
    fi
  done
  echo "== Generating synthetic tiers (skip if present) =="
  uv run python "${ROOT}/benchmarks/synthetic/generate.py" --output "${SYNTH_ROOT}" "${synth_tiers[@]}"
fi

duration_for_case() {
  local case="$1"
  if [[ "${case}" == "xlarge" ]]; then
    echo 12000
  elif [[ "${case}" == "large" ]]; then
    echo 8000
  elif [[ "${case}" == "complete" ]]; then
    echo "${DURATION}"
  else
    echo 5000
  fi
}

paths_for_case() {
  local case="$1"
  if [[ "${case}" == "complete" && -z "${CUSTOM_CASE}" ]]; then
    echo "${DEFAULT_INPUT}" "${DEFAULT_CONFIG}"
  elif [[ "${case}" == "complete" ]]; then
    echo "${INPUT}" "${CONFIG}"
  else
    echo "${SYNTH_ROOT}/${case}/peptides.parquet" "${SYNTH_ROOT}/${case}/config.yaml"
  fi
}

run_case() {
  local case="$1"
  local input="$2"
  local config="$3"
  local duration="$4"

  local case_work="${WORK_BASE}/${case}"
  local ref_out="${case_work}/reference/normalized.parquet"
  local pf_out="${case_work}/proteoforge/normalized.parquet"
  local json="${case_work}/zebrac.json"

  if [[ ! -f "${input}" ]]; then
    echo "Missing input: ${input}" >&2
    exit 1
  fi
  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 1
  fi

  mkdir -p "${case_work}/reference" "${case_work}/proteoforge"

  echo
  echo "== ${case}: zebrac =="
  "${ZEBRAC}" -d "${duration}" -w 2 --json "${json}" \
    "uv run python ${COMPARE}/run_reference.py --input ${input} --config ${config} --output ${ref_out}" \
    "uv run python ${COMPARE}/run_proteoforge.py --input ${input} --config ${config} --output ${pf_out}"

  echo
  echo "== ${case}: parity =="
  uv run python "${COMPARE}/diff_outputs.py" --atol 1e-11 "${ref_out}" "${pf_out}"
}

for case in "${CASES[@]}"; do
  read -r case_input case_config <<< "$(paths_for_case "${case}")"
  run_case "${case}" "${case_input}" "${case_config}" "$(duration_for_case "${case}")"
done

echo
echo "== Summary =="
uv run python "${COMPARE}/summarize.py" --work-dir "${WORK_BASE}" --cases "${CASES[@]}"

echo
echo "Results under ${WORK_BASE}/<case>/zebrac.json"
