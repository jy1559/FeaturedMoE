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
TEX_FILE="$1"
shift || true

if [[ "$TEX_FILE" = /* ]]; then
  TEX_PATH="$TEX_FILE"
else
  TEX_PATH="$PWD/$TEX_FILE"
fi

TEX_DIR="$(cd "$(dirname "$TEX_PATH")" && pwd)"
TEX_BASENAME="$(basename "$TEX_PATH")"
TEX_STEM="${TEX_BASENAME%.tex}"

if [[ -d "$TEX_DIR/acmart-primary" ]]; then
  ACM_DIR="$TEX_DIR/acmart-primary"
elif [[ -d "$SCRIPT_DIR/acmart-primary" ]]; then
  ACM_DIR="$SCRIPT_DIR/acmart-primary"
else
  echo "Error: could not find acmart-primary next to the TeX file or in $SCRIPT_DIR" >&2
  exit 1
fi

COMMON_TEXINPUTS="TEXINPUTS=$ACM_DIR//:${TEXINPUTS:-}"
COMMON_BSTINPUTS="BSTINPUTS=$ACM_DIR//:${BSTINPUTS:-}"
COMMON_BIBINPUTS="BIBINPUTS=$TEX_DIR:$SCRIPT_DIR:${BIBINPUTS:-}"

cd "$TEX_DIR"

if command -v latexmk >/dev/null 2>&1; then
  env "$COMMON_TEXINPUTS" "$COMMON_BSTINPUTS" "$COMMON_BIBINPUTS" \
    latexmk "${FORCE_ARGS[@]}" -pdf -interaction=nonstopmode -halt-on-error "$TEX_BASENAME" "$@"
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  env "$COMMON_TEXINPUTS" "$COMMON_BSTINPUTS" "$COMMON_BIBINPUTS" \
    pdflatex -interaction=nonstopmode -halt-on-error "$TEX_BASENAME"
  env "$COMMON_TEXINPUTS" "$COMMON_BSTINPUTS" "$COMMON_BIBINPUTS" \
    bibtex "$TEX_STEM"
  env "$COMMON_TEXINPUTS" "$COMMON_BSTINPUTS" "$COMMON_BIBINPUTS" \
    pdflatex -interaction=nonstopmode -halt-on-error "$TEX_BASENAME"
  env "$COMMON_TEXINPUTS" "$COMMON_BSTINPUTS" "$COMMON_BIBINPUTS" \
    pdflatex -interaction=nonstopmode -halt-on-error "$TEX_BASENAME"
else
  echo "Error: latexmk is missing, and pdflatex/bibtex are also unavailable." >&2
  echo "Install one of: latexmk (recommended) or TeX Live base tools (pdflatex + bibtex)." >&2
  exit 1
fi
