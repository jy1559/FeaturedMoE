#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--force] <tex-file> [latexmk args...]" >&2
  exit 1
}

FORCE_ARGS=()

if [[ $# -lt 1 ]]; then
  usage
fi

if [[ "${1:-}" == "--force" ]]; then
  FORCE_ARGS=(-g)
  shift
fi

if [[ $# -lt 1 ]]; then
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACM_DIR="$SCRIPT_DIR/acmart-primary"
TEX_FILE="$1"
shift || true

cd "$SCRIPT_DIR"

TEXINPUTS="$ACM_DIR//:${TEXINPUTS:-}" \
BSTINPUTS="$ACM_DIR//:${BSTINPUTS:-}" \
BIBINPUTS="$SCRIPT_DIR:${BIBINPUTS:-}" \
latexmk "${FORCE_ARGS[@]}" -pdf -interaction=nonstopmode -halt-on-error "$TEX_FILE" "$@"
