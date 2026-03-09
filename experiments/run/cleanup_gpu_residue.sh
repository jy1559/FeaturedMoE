#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  cleanup_gpu_residue.sh --gpus 0,1,2 [--yes]

Description:
  Kill all compute processes currently attached to the specified GPU indices.
  No default GPU list is provided; --gpus is mandatory.

Options:
  --gpus   Comma-separated GPU indices (required)
  --yes    Skip interactive confirmation
  -h, --help
USAGE
}

GPU_LIST=""
ASSUME_YES="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus)
      GPU_LIST="${2:-}"
      shift 2
      ;;
    --yes)
      ASSUME_YES="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "$GPU_LIST" ]; then
  echo "[ERROR] --gpus is required. No default is allowed." >&2
  usage
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found." >&2
  exit 1
fi

declare -A TARGET
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
for g in "${GPU_ARR[@]}"; do
  g="$(echo "$g" | xargs)"
  if [[ ! "$g" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid GPU index: '$g'" >&2
    exit 1
  fi
  TARGET["$g"]=1
done

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

declare -A UUID_TO_INDEX
for row in "${GPU_ROWS[@]}"; do
  idx="$(echo "$row" | cut -d',' -f1 | xargs)"
  uuid="$(echo "$row" | cut -d',' -f2 | xargs)"
  UUID_TO_INDEX["$uuid"]="$idx"
done

mapfile -t APP_ROWS < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory --format=csv,noheader,nounits || true)

PIDS=()
echo "=== Target GPUs: ${GPU_LIST} ==="
for row in "${APP_ROWS[@]}"; do
  [ -z "$row" ] && continue
  pid="$(echo "$row" | cut -d',' -f1 | xargs)"
  uuid="$(echo "$row" | cut -d',' -f2 | xargs)"
  pname="$(echo "$row" | cut -d',' -f3 | xargs)"
  mem="$(echo "$row" | cut -d',' -f4 | xargs)"
  idx="${UUID_TO_INDEX[$uuid]:-}"
  [ -z "$idx" ] && continue
  if [ -n "${TARGET[$idx]:-}" ]; then
    echo "  GPU ${idx} | PID ${pid} | ${pname} | ${mem} MiB"
    PIDS+=("$pid")
  fi
done

# Robust fallback: collect real local PIDs holding /dev/nvidia<idx>.
for idx in "${!TARGET[@]}"; do
  dev="/dev/nvidia${idx}"
  [ -e "$dev" ] || continue
  mapfile -t DEV_PIDS < <(fuser "$dev" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sed 's/[^0-9]//g' | sed '/^$/d')
  for pid in "${DEV_PIDS[@]}"; do
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
      PIDS+=("$pid")
    fi
  done
done

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "[INFO] No compute processes found on target GPUs."
  exit 0
fi

if [ "$ASSUME_YES" != "true" ]; then
  echo
  read -r -p "Type KILL to terminate listed PIDs: " CONFIRM
  if [ "$CONFIRM" != "KILL" ]; then
    echo "[ABORT] Nothing was killed."
    exit 1
  fi
fi

UNIQ_PIDS=($(printf "%s\n" "${PIDS[@]}" | sort -u))

echo "[INFO] Candidate PIDs: ${UNIQ_PIDS[*]}"
for pid in "${UNIQ_PIDS[@]}"; do
  ps -p "$pid" -o pid=,ppid=,cmd= 2>/dev/null || true
done

echo "[INFO] Sending SIGTERM..."
kill -TERM "${UNIQ_PIDS[@]}" 2>/dev/null || true
sleep 2

REMAIN=()
for pid in "${UNIQ_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    REMAIN+=("$pid")
  fi
done

if [ "${#REMAIN[@]}" -gt 0 ]; then
  echo "[INFO] Sending SIGKILL to remaining PIDs..."
  kill -KILL "${REMAIN[@]}" 2>/dev/null || true
fi

echo "[DONE] GPU cleanup requested for: ${GPU_LIST}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

echo "[VERIFY] Remaining /dev/nvidia* holders (target GPUs):"
for idx in "${!TARGET[@]}"; do
  dev="/dev/nvidia${idx}"
  [ -e "$dev" ] || continue
  holders="$(fuser "$dev" 2>/dev/null || true)"
  if [ -n "$holders" ]; then
    echo "  GPU ${idx}: ${holders}"
  else
    echo "  GPU ${idx}: none"
  fi
done
