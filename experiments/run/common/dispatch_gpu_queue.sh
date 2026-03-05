#!/usr/bin/env bash
set -euo pipefail

# Source this file from runners.
# Provides GPU slot queue helpers with one PID per GPU.

declare -Ag DISPATCH_GPU_PID=()

dispatch_parse_csv() {
  local csv="${1:-}"
  local -n out_ref=$2
  out_ref=()
  local token
  IFS=',' read -r -a _raw <<< "$csv"
  for token in "${_raw[@]}"; do
    token="${token//[[:space:]]/}"
    [ -n "$token" ] && out_ref+=("$token")
  done
}

dispatch_find_free_gpu() {
  local list_name="$1"
  local -n _gpus_ref="$list_name"
  for gid in "${_gpus_ref[@]}"; do
    local p="${DISPATCH_GPU_PID[$gid]:-}"
    if [[ -z "$p" ]] || ! kill -0 "$p" 2>/dev/null; then
      DISPATCH_GPU_PID["$gid"]=""
      FREE_GPU="$gid"
      return 0
    fi
  done
  return 1
}

dispatch_wait_for_gpu() {
  local list_name="$1"
  while ! dispatch_find_free_gpu "$list_name"; do
    wait -n 2>/dev/null || sleep 2
  done
}

dispatch_set_pid() {
  local gpu_id="${1:?gpu}"
  local pid="${2:?pid}"
  DISPATCH_GPU_PID["$gpu_id"]="$pid"
}

dispatch_wait_all() {
  wait
}

dispatch_collect_tree_pids() {
  local root_pid="${1:?root_pid required}"
  local -a queue=("$root_pid")
  local -a out=()
  local seen=" ${root_pid} "
  local cur child

  while [ "${#queue[@]}" -gt 0 ]; do
    cur="${queue[0]}"
    queue=("${queue[@]:1}")
    out+=("$cur")

    while IFS= read -r child; do
      child="${child//[[:space:]]/}"
      [ -z "$child" ] && continue
      if [[ "$seen" != *" ${child} "* ]]; then
        queue+=("$child")
        seen="${seen}${child} "
      fi
    done < <(ps -o pid= --ppid "$cur" 2>/dev/null || true)
  done

  printf '%s\n' "${out[@]}"
}

dispatch_signal_tree() {
  local sig="${1:?signal}"
  local root_pid="${2:?root_pid}"
  local -a pids=()
  mapfile -t pids < <(dispatch_collect_tree_pids "$root_pid")
  [ "${#pids[@]}" -eq 0 ] && return 0

  local p
  for p in "${pids[@]}"; do
    kill "-${sig}" "$p" 2>/dev/null || true
  done
}

dispatch_signal_group_if_session_leader() {
  local sig="${1:?signal}"
  local pid="${2:?pid}"
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  [ -z "$pgid" ] && return 0

  # Avoid blasting unrelated shell jobs. We only group-kill isolated setsid jobs.
  if [ "$pgid" = "$pid" ]; then
    kill "-${sig}" "-${pgid}" 2>/dev/null || true
  fi
}

dispatch_terminate_all() {
  local list_name="$1"
  local -n _gpus_ref="$list_name"
  local -a pids=()
  local gid
  for gid in "${_gpus_ref[@]}"; do
    local p="${DISPATCH_GPU_PID[$gid]:-}"
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      pids+=("$p")
    fi
  done
  if [ "${#pids[@]}" -eq 0 ]; then
    return 0
  fi

  local p
  for p in "${pids[@]}"; do
    dispatch_signal_group_if_session_leader TERM "$p"
    dispatch_signal_tree TERM "$p"
  done
  sleep 1
  for p in "${pids[@]}"; do
    dispatch_signal_group_if_session_leader KILL "$p"
    dispatch_signal_tree KILL "$p"
  done
  wait || true

  # clear tracked slots
  for gid in "${_gpus_ref[@]}"; do
    DISPATCH_GPU_PID["$gid"]=""
  done
}
