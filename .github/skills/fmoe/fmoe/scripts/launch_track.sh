#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SKILL_DIR}/../../.." && pwd)"

TRACK="fmoe-main"
DATASETS="movielens1m,retail_rocket"
GPUS="0,1"
SEED_BASE="42"
REPO_ROOT="${REPO_ROOT_DEFAULT}"
DRY_RUN="false"

MAX_EVALS=""
TUNE_EPOCHS=""
TUNE_PATIENCE=""
SEARCH_PROFILE="narrow_ml1"
LOG_WANDB="false"

usage() {
  cat <<USAGE
Usage: $0 [--track fmoe-main|hir-compare|arch-probe] [--datasets <csv>] [--gpus <csv>] [--seed-base <int>] [--dry-run]

Options:
  --track           Track to run. Default: fmoe-main
  --datasets        Comma-separated datasets. Default: movielens1m,retail_rocket
  --gpus            Comma-separated GPU ids. Default: 0,1
  --seed-base       Base seed. Default: 42
  --repo-root       FMoE repo root. Default: auto-detected from skill path
  --dry-run         Print delegated commands without launching training

HiR-only options:
  --max-evals
  --tune-epochs
  --tune-patience
  --search-profile  narrow_ml1|wide
  --log-wandb
  --no-wandb
USAGE
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "[ERROR] Missing required file: $path" >&2
    exit 1
  fi
}

trim_spaces() {
  local s="$1"
  s="${s//[[:space:]]/}"
  printf '%s' "$s"
}

split_csv() {
  local raw
  raw="$(trim_spaces "$1")"
  if [ -z "$raw" ]; then
    return 1
  fi
  IFS=',' read -r -a __CSV_OUT <<< "$raw"
  if [ "${#__CSV_OUT[@]}" -eq 0 ]; then
    return 1
  fi
  return 0
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --track) TRACK="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$SEED_BASE" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --seed-base must be a non-negative integer" >&2
  exit 1
fi

RUN_FMOE_DIR="${REPO_ROOT}/experiments/run/fmoe"
RUN_HIR_DIR="${REPO_ROOT}/experiments/run/fmoe_hir"

run_fmoe_main() {
  local pipeline="${RUN_FMOE_DIR}/pipeline_ml1_rr.sh"
  require_file "$pipeline"

  split_csv "$DATASETS" || { echo "[ERROR] --datasets is empty" >&2; exit 1; }
  local datasets=("${__CSV_OUT[@]}")
  local ds
  for ds in "${datasets[@]}"; do
    case "$ds" in
      movielens1m|retail_rocket) ;;
      *)
        echo "[ERROR] fmoe-main currently supports movielens1m,retail_rocket only (got: $ds)" >&2
        echo "[HINT] Use arch-probe or run experiments/run/fmoe/tune_hparam.sh directly for expansion datasets." >&2
        exit 1
        ;;
    esac
  done

  local cmd=(
    bash "$pipeline"
    --datasets "$DATASETS"
    --gpus "$GPUS"
    --seed-base "$SEED_BASE"
  )
  if [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
  fi

  echo "[TRACK] fmoe-main"
  echo "[REPO]  ${REPO_ROOT}"
  echo "[RUN]   ${cmd[*]}"
  "${cmd[@]}"
}

run_hir_compare() {
  local runner="${RUN_HIR_DIR}/run_4phase_hir.sh"
  require_file "$runner"

  split_csv "$DATASETS" || { echo "[ERROR] --datasets is empty" >&2; exit 1; }
  local datasets=("${__CSV_OUT[@]}")
  split_csv "$GPUS" || { echo "[ERROR] --gpus is empty" >&2; exit 1; }
  local gpus=("${__CSV_OUT[@]}")

  echo "[TRACK] hir-compare"
  echo "[REPO]  ${REPO_ROOT}"

  local i
  for i in "${!datasets[@]}"; do
    local ds="${datasets[$i]}"
    local gpu="${gpus[$((i % ${#gpus[@]}))]}"
    local cmd=(
      bash "$runner"
      --dataset "$ds"
      --gpu "$gpu"
      --seed "$SEED_BASE"
      --search-profile "$SEARCH_PROFILE"
    )
    [ -n "$MAX_EVALS" ] && cmd+=(--max-evals "$MAX_EVALS")
    [ -n "$TUNE_EPOCHS" ] && cmd+=(--tune-epochs "$TUNE_EPOCHS")
    [ -n "$TUNE_PATIENCE" ] && cmd+=(--tune-patience "$TUNE_PATIENCE")
    if [ "$LOG_WANDB" = "true" ]; then
      cmd+=(--log-wandb)
    else
      cmd+=(--no-wandb)
    fi
    if [ "$DRY_RUN" = "true" ]; then
      cmd+=(--dry-run)
    fi

    echo "[RUN:${i}] ${cmd[*]}"
    "${cmd[@]}"
  done
}

run_arch_probe() {
  local train_single="${RUN_FMOE_DIR}/train_single.sh"
  local tune_hparam="${RUN_FMOE_DIR}/tune_hparam.sh"
  local tune_hir="${RUN_HIR_DIR}/tune_hparam_hir.sh"

  require_file "$train_single"
  require_file "$tune_hparam"
  require_file "$tune_hir"

  cat <<EOF
[TRACK] arch-probe (non-mutating templates only)
[REPO]  ${REPO_ROOT}

1) 신규 모델 파일/설정 연결 점검 템플릿:
   - experiments/models/<new_model>/*
   - experiments/configs/model/<new_model>.yaml
   - experiments/recbole_patch.py model registration

2) FMoE 스모크 템플릿:
   bash ${train_single} --dataset movielens1m --gpu 0 --phase P0 --schedule off --layout-id 0

3) FMoE hparam 1-trial 템플릿:
   bash ${tune_hparam} --dataset movielens1m --gpu 0 --max-evals 1 --tune-epochs 1 --tune-patience 1 --phase P1PROBE --schedule-preset off

4) HiR hparam 1-trial 템플릿:
   bash ${tune_hir} --dataset movielens1m --gpu 0 --max-evals 1 --tune-epochs 1 --tune-patience 1 --phase P1HIRPROBE --schedule-preset off --stage-merge-mode serial

5) 결과 요약:
   python3 ${REPO_ROOT}/.codex/skills/fmoe/scripts/collect_results.py --repo-root ${REPO_ROOT} --datasets movielens1m --metric mrr@20
EOF
}

case "${TRACK}" in
  fmoe-main) run_fmoe_main ;;
  hir-compare) run_hir_compare ;;
  arch-probe) run_arch_probe ;;
  *)
    echo "[ERROR] Unsupported --track=${TRACK} (use fmoe-main|hir-compare|arch-probe)" >&2
    exit 1
    ;;
esac
