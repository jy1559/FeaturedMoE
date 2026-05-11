#!/usr/bin/env python3
"""Run a targeted revised campaign for problematic PAIR60_V4 runs.

This launcher reuses original run commands recorded in PAIR60_V4 logs, then applies
small, explicit overrides for:
- OOM-prone runs: lower train/eval batch sizes.
- Underperforming runs: nudge fixed architecture/search knobs using stronger settings
  from better-performing baselines in the same dataset family.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"

TRACK = "baseline_2"
BASE_AXIS = "PAIR60_V4"
DEFAULT_AXIS = "PAIR60_V4_REVISED"

BASE_LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK / BASE_AXIS
REVISED_LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK

STOP_EVENT = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
ACTIVE_PROCESS_LOCK = threading.Lock()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RevisionSpec:
    run_phase: str
    reason: str
    ref_note: str
    train_batch_size: int | None = None
    eval_batch_size: int | None = None
    lr_range: tuple[float, float] | None = None
    fixed_overrides: Dict[str, Any] | None = None


# Target set derived from current PAIR60_V4 analysis (OOM + strong underperformance outliers).
REVISED_TARGETS: List[RevisionSpec] = [
    RevisionSpec(
        run_phase="PAIR60_V4_DLASTFM0_03_MTISASREC_P053_C2_S1",
        reason="OOM in 3/4 trials",
        ref_note="Keep TiSASRec shape, only reduce memory pressure.",
        train_batch_size=1024,
        eval_batch_size=2048,
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DMOVIELENS1M_MTISASREC_P043_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Use smaller batch and slightly shorter time window for stability.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={"time_span": 256},
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MTISASREC_P013_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Retail Rocket TiSASRec memory cut + softer model width.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={"hidden_size": 96, "embedding_size": 96, "inner_size": 192},
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MDIFSR_P017_C1_S1",
        reason="OOM in 6/6 trials and zero score",
        ref_note="Restore original DIFSR width with moderate batch increase and keep OOM-safe fallback.",
        train_batch_size=768,
        eval_batch_size=1536,
        fixed_overrides={
            "hidden_size": 160,
            "embedding_size": 160,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 320,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "attribute_hidden_size": 160,
            "fusion_type": "gate",
            "use_attribute_predictor": False,
            "lambda_attr": 0.0,
            "selected_features": ["category"],
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MDIFSR_P017_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Restore original C2 DIFSR width with moderate batch increase and keep OOM-safe fallback.",
        train_batch_size=768,
        eval_batch_size=1536,
        fixed_overrides={
            "hidden_size": 112,
            "embedding_size": 112,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 224,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "attribute_hidden_size": 112,
            "fusion_type": "gate",
            "use_attribute_predictor": False,
            "lambda_attr": 0.0,
            "selected_features": ["category"],
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MFDSA_P019_C1_S1",
        reason="OOM in 6/6 trials and zero score",
        ref_note="Restore original FDSA width with moderate batch increase and keep OOM-safe fallback.",
        train_batch_size=768,
        eval_batch_size=1536,
        fixed_overrides={
            "hidden_size": 160,
            "embedding_size": 160,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 320,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "attribute_hidden_size": 160,
            "selected_features": ["category"],
            "pooling_mode": "mean",
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MFDSA_P019_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Restore original C2 FDSA width with moderate batch increase and keep OOM-safe fallback.",
        train_batch_size=768,
        eval_batch_size=1536,
        fixed_overrides={
            "hidden_size": 112,
            "embedding_size": 112,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 224,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "attribute_hidden_size": 112,
            "selected_features": ["category"],
            "pooling_mode": "mean",
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MBSAREC_P005_C2_S1",
        reason="Severe underperformance vs dataset peers",
        ref_note="Reference SASRec-like stable transformer width/dropout.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "n_heads": 2,
            "num_heads": 2,
            "inner_size": 256,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-5,
        },
        lr_range=(1.5e-4, 7.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MDIFSR_P007_C2_S1",
        reason="Near-zero score compared to same-dataset models",
        ref_note="Reference DIFSR gate fusion setting used in stronger runs.",
        train_batch_size=768,
        eval_batch_size=1536,
        fixed_overrides={
            "hidden_size": 160,
            "embedding_size": 160,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 320,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-5,
            "attribute_hidden_size": 160,
            "fusion_type": "gate",
            "use_attribute_predictor": False,
            "lambda_attr": 0.0,
            "selected_features": ["category"],
        },
        lr_range=(1.2e-4, 6.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MFAME_P008_C1_S1",
        reason="Very low score outlier",
        ref_note="Reference stronger FAME setup with more regularized transformer block.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1.5e-4,
            "num_experts": 3,
        },
        lr_range=(1.8e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MGRU4REC_P002_C1_S1",
        reason="Low score vs transformer baselines",
        ref_note="Reference stronger GRU4Rec profile from better datasets.",
        fixed_overrides={
            "hidden_size": 192,
            "embedding_size": 192,
            "num_layers": 1,
            "dropout_prob": 0.25,
            "hidden_dropout_prob": 0.25,
            "weight_decay": 1e-5,
            "MAX_ITEM_LIST_LENGTH": 30,
        },
        lr_range=(2.0e-4, 1.0e-3),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DFOURSQUARE_MBSAREC_P035_C2_S1",
        reason="Underperforming BSARec combo",
        ref_note="Reference BSARec C1-style balanced transformer settings.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "n_heads": 2,
            "num_heads": 2,
            "inner_size": 256,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-5,
        },
        lr_range=(1.2e-4, 6.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DFOURSQUARE_MFAME_P038_C1_S1",
        reason="Underperforming FAME combo",
        ref_note="Reference stronger FAME settings used in lastfm/foursquare families.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.1,
            "hidden_dropout_prob": 0.1,
            "attn_dropout_prob": 0.1,
            "weight_decay": 1e-4,
            "num_experts": 3,
        },
        lr_range=(1.5e-4, 5.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DFOURSQUARE_MGRU4REC_P032_C2_S1",
        reason="Underperforming GRU4Rec combo",
        ref_note="Reference GRU4Rec high-LR profile that performed better in similar runs.",
        fixed_overrides={
            "hidden_size": 192,
            "embedding_size": 192,
            "num_layers": 1,
            "dropout_prob": 0.2,
            "hidden_dropout_prob": 0.2,
            "weight_decay": 1e-5,
            "MAX_ITEM_LIST_LENGTH": 30,
        },
        lr_range=(2.0e-4, 9.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MGRU4REC_P002_C1_S1",
        reason="Low score on KuaiRec strict",
        ref_note="Reference Kuai-friendly GRU4Rec with larger hidden and broader LR.",
        fixed_overrides={
            "hidden_size": 192,
            "embedding_size": 192,
            "num_layers": 1,
            "dropout_prob": 0.25,
            "hidden_dropout_prob": 0.25,
            "weight_decay": 1e-5,
        },
        lr_range=(2.2e-4, 1.1e-3),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MTISASREC_P023_C1_S1",
        reason="Lower-than-expected TiSASRec score",
        ref_note="Reference TiSASRec stable setting from lastfm, with conservative width/time_span.",
        train_batch_size=1536,
        eval_batch_size=3072,
        fixed_overrides={
            "hidden_size": 96,
            "embedding_size": 96,
            "n_layers": 2,
            "num_layers": 2,
            "n_heads": 2,
            "num_heads": 2,
            "inner_size": 192,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-6,
            "time_span": 512,
        },
        lr_range=(1.2e-4, 4.5e-4),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", type=str, default=DEFAULT_AXIS)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--search-algo", type=str, default="random")
    parser.add_argument("--max-evals", type=int, default=6)
    parser.add_argument("--tune-epochs", type=int, default=70)
    parser.add_argument("--tune-patience", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument(
        "--only-run-phases",
        type=str,
        default="",
        help=(
            "Optional comma/space separated run phases to execute. "
            "Accepts original run_phase (PAIR60_V4_...) or revised name "
            "(PAIR60_V4_REVISED_..._REV1)."
        ),
    )
    parser.add_argument(
        "--auto-extend-count",
        type=int,
        default=0,
        help="Append additional low-performance PAIR60_V4 phases from ranking TSV.",
    )
    parser.add_argument(
        "--auto-extend-max-ratio",
        type=float,
        default=0.95,
        help="Include auto candidates with ratio <= this threshold.",
    )
    parser.add_argument(
        "--auto-extend-rank-tsv",
        type=str,
        default=str(REPO_ROOT / "outputs" / "_tmp_pair60_run_ratio_rank.tsv"),
        help="TSV file path with columns including ratio and run_phase.",
    )
    parser.add_argument("--python-bin", type=str, default=os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python"))
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    # Accept both comma-separated and whitespace-separated GPU lists.
    return [x.strip() for x in re.split(r"[,\s]+", str(text or "").strip()) if x.strip()]


def normalize_phase_for_match(value: str, axis: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if token.endswith("_REV1"):
        token = token[: -len("_REV1")]
    if token.startswith(f"{axis}_"):
        token = BASE_AXIS + token[len(axis) :]
    return token


def unique_keep_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def parse_model_code(run_phase: str) -> str:
    m = re.search(r"_M([A-Z0-9_]+)_P\d+_C\d+_S\d+$", str(run_phase))
    return m.group(1) if m else ""


def build_auto_profile_for_phase(run_phase: str, ratio: float) -> RevisionSpec:
    code = parse_model_code(run_phase)
    reason = f"low seen-target ratio={ratio:.3f} vs dataset winner"
    ref_note = "Auto-expanded low-performance rerun profile."

    if code in {"MDIFSR"}:
        return RevisionSpec(
            run_phase=run_phase,
            reason=reason,
            ref_note=ref_note,
            train_batch_size=768,
            eval_batch_size=1536,
            lr_range=(1.0e-4, 4.5e-4),
            fixed_overrides={
                "n_layers": 2,
                "num_layers": 2,
                "dropout_ratio": 0.15,
                "hidden_dropout_prob": 0.15,
                "attn_dropout_prob": 0.15,
                "weight_decay": 2e-4,
                "fusion_type": "gate",
                "use_attribute_predictor": False,
                "lambda_attr": 0.0,
                "selected_features": ["category"],
            },
        )

    if code in {"MFDSA"}:
        return RevisionSpec(
            run_phase=run_phase,
            reason=reason,
            ref_note=ref_note,
            train_batch_size=768,
            eval_batch_size=1536,
            lr_range=(1.0e-4, 4.5e-4),
            fixed_overrides={
                "n_layers": 2,
                "num_layers": 2,
                "dropout_ratio": 0.15,
                "hidden_dropout_prob": 0.15,
                "attn_dropout_prob": 0.15,
                "weight_decay": 2e-4,
                "selected_features": ["category"],
                "pooling_mode": "mean",
            },
        )

    if code in {"MGRU4REC"}:
        return RevisionSpec(
            run_phase=run_phase,
            reason=reason,
            ref_note=ref_note,
            lr_range=(2.0e-4, 1.2e-3),
            fixed_overrides={
                "num_layers": 1,
                "dropout_prob": 0.25,
                "hidden_dropout_prob": 0.25,
                "weight_decay": 1e-5,
                "MAX_ITEM_LIST_LENGTH": 30,
            },
        )

    if code in {"MTISASREC"}:
        return RevisionSpec(
            run_phase=run_phase,
            reason=reason,
            ref_note=ref_note,
            train_batch_size=1024,
            eval_batch_size=2048,
            lr_range=(1.2e-4, 6.0e-4),
            fixed_overrides={
                "n_layers": 2,
                "num_layers": 2,
                "dropout_ratio": 0.12,
                "hidden_dropout_prob": 0.12,
                "attn_dropout_prob": 0.12,
                "weight_decay": 1e-5,
                "time_span": 256,
            },
        )

    if code in {"MSASREC", "MBSAREC", "MDUOREC", "MFEAREC", "MFAME"}:
        return RevisionSpec(
            run_phase=run_phase,
            reason=reason,
            ref_note=ref_note,
            lr_range=(1.2e-4, 7.5e-4),
            fixed_overrides={
                "n_layers": 2,
                "num_layers": 2,
                "dropout_ratio": 0.12,
                "hidden_dropout_prob": 0.12,
                "attn_dropout_prob": 0.12,
                "weight_decay": 1e-5,
            },
        )

    if code in {"MFEATURED_MOE_N3"}:
        return RevisionSpec(
            run_phase=run_phase,
            reason=reason,
            ref_note=ref_note,
            train_batch_size=768,
            eval_batch_size=1536,
            lr_range=(8.0e-5, 3.5e-4),
            fixed_overrides={
                "dropout_ratio": 0.15,
                "hidden_dropout_prob": 0.15,
                "attn_dropout_prob": 0.15,
                "weight_decay": 1e-5,
            },
        )

    return RevisionSpec(
        run_phase=run_phase,
        reason=reason,
        ref_note=ref_note,
        lr_range=(1.2e-4, 7.0e-4),
    )


def load_auto_extend_specs(tsv_path: str, limit: int, max_ratio: float, existing_run_phases: set[str]) -> List[RevisionSpec]:
    if limit <= 0:
        return []
    p = Path(tsv_path)
    if not p.exists():
        return []

    # Keep the lowest ratio per run_phase, and only from original PAIR60_V4 axis.
    phase_best_ratio: Dict[str, float] = {}
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        cols = ln.split("\t")
        if len(cols) < 5:
            continue
        try:
            ratio = float(cols[0])
        except Exception:
            continue
        run_phase = cols[4].strip()
        if not run_phase.startswith(f"{BASE_AXIS}_"):
            continue
        if "_REVISED" in run_phase:
            continue
        prev = phase_best_ratio.get(run_phase)
        if prev is None or ratio < prev:
            phase_best_ratio[run_phase] = ratio

    ranked: List[Tuple[float, str]] = sorted((r, rp) for rp, r in phase_best_ratio.items())
    out: List[RevisionSpec] = []
    for ratio, run_phase in ranked:
        if ratio > float(max_ratio):
            continue
        if run_phase in existing_run_phases:
            continue
        out.append(build_auto_profile_for_phase(run_phase, ratio))
        if len(out) >= int(limit):
            break
    return out


def _active_proc_add(proc: subprocess.Popen[Any]) -> None:
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.add(proc)


def _active_proc_remove(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.discard(proc)


def _terminate_process(proc: subprocess.Popen[Any], sig_num: int) -> None:
    try:
        if proc.poll() is not None:
            return
        if int(sig_num) == int(signal.SIGKILL):
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        return


def terminate_active_children(*, grace_sec: float = 0.4) -> None:
    with ACTIVE_PROCESS_LOCK:
        procs = list(ACTIVE_PROCESSES)
    if not procs:
        return
    for proc in procs:
        _terminate_process(proc, signal.SIGTERM)
    if grace_sec > 0:
        time.sleep(float(grace_sec))
    for proc in procs:
        _terminate_process(proc, signal.SIGKILL)


def install_signal_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        first = not STOP_EVENT.is_set()
        STOP_EVENT.set()
        if first:
            relay(f"[revised] interrupt signal={signum} -> terminating active runs")
            terminate_active_children(grace_sec=0.2)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def find_log_map() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in BASE_LOG_ROOT.rglob("*.log"):
        out[p.stem] = p
    return out


def parse_cmd_from_log(log_path: Path) -> List[str]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for ln in lines[:8]:
        if ln.startswith("# cmd="):
            return shlex.split(ln[len("# cmd=") :].strip())
    raise RuntimeError(f"No '# cmd=' line found in log: {log_path}")


def token_key(token: str) -> str:
    if token.startswith("++") and "=" in token:
        return token[2:].split("=", 1)[0]
    # Plain key=value overrides (e.g. gpu_id=7, model=difsr)
    # are also valid tokens in this launcher pipeline.
    if not token.startswith("--") and "=" in token:
        return token.split("=", 1)[0]
    if token.startswith("--"):
        return token
    return ""


def set_arg_pair(tokens: List[str], flag: str, value: str) -> List[str]:
    out: List[str] = []
    i = 0
    found = False
    while i < len(tokens):
        t = tokens[i]
        if t == flag and i + 1 < len(tokens):
            out.extend([flag, value])
            i += 2
            found = True
            continue
        out.append(t)
        i += 1
    if not found:
        out.extend([flag, value])
    return out


def drop_keys(tokens: List[str], keys: List[str]) -> List[str]:
    keyset = set(keys)
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        k = token_key(t)
        if k in keyset:
            i += 1
            continue
        if t in keyset and i + 1 < len(tokens):
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{hydra_literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Unsupported type: {type(value)}")


def build_revised_command(base_cmd: List[str], spec: RevisionSpec, axis: str, gpu: str, args: argparse.Namespace) -> List[str]:
    run_phase_new = spec.run_phase.replace(BASE_AXIS, axis, 1) + "_REV1"

    # Replace core positional/script and global args.
    cmd = list(base_cmd)
    if cmd:
        cmd[0] = args.python_bin
    cmd = set_arg_pair(cmd, "--run-axis", axis)
    cmd = set_arg_pair(cmd, "--run-phase", run_phase_new)
    cmd = set_arg_pair(cmd, "--search-algo", str(args.search_algo))
    cmd = set_arg_pair(cmd, "--max-evals", str(int(args.max_evals)))
    cmd = set_arg_pair(cmd, "--tune-epochs", str(int(args.tune_epochs)))
    cmd = set_arg_pair(cmd, "--tune-patience", str(int(args.tune_patience)))

    # Remove stale gpu_id/model runtime args that we will append cleanly.
    cmd = drop_keys(
        cmd,
        [
            "gpu_id",
            "search",
            "train_batch_size",
            "eval_batch_size",
            "search.train_batch_size",
            "search.eval_batch_size",
            "search_space_type_overrides.train_batch_size",
            "search_space_type_overrides.eval_batch_size",
        ],
    )

    # Ensure gpu override is last-wins.
    cmd.append(f"gpu_id={gpu}")

    if spec.train_batch_size is not None:
        bs = int(spec.train_batch_size)
        cmd.extend(
            [
                f"++train_batch_size={bs}",
                f"++search.train_batch_size={hydra_literal([bs])}",
                "++search_space_type_overrides.train_batch_size=choice",
            ]
        )
    if spec.eval_batch_size is not None:
        ebs = int(spec.eval_batch_size)
        cmd.extend(
            [
                f"++eval_batch_size={ebs}",
                f"++search.eval_batch_size={hydra_literal([ebs])}",
                "++search_space_type_overrides.eval_batch_size=choice",
            ]
        )

    if spec.fixed_overrides:
        for k, v in spec.fixed_overrides.items():
            cmd.extend(
                [
                    f"++{k}={hydra_literal(v)}",
                    f"++search.{k}={hydra_literal([v])}",
                    f"++search_space_type_overrides.{k}=choice",
                ]
            )

    if spec.lr_range is not None:
        lo, hi = float(spec.lr_range[0]), float(spec.lr_range[1])
        cmd.extend(
            [
                f"++search={hydra_literal({'learning_rate': [lo, hi]})}",
                "++search_space_type_overrides={learning_rate:\"loguniform\"}",
            ]
        )

    # Make seed different but deterministic for revised runs.
    # Keep original ++seed if present, and append last-wins +10000.
    seed_val = None
    for t in cmd:
        if t.startswith("++seed="):
            try:
                seed_val = int(t.split("=", 1)[1].strip().strip('"'))
            except Exception:
                seed_val = None
    if seed_val is not None:
        cmd.append(f"++seed={seed_val + 10000}")

    return cmd


def relay(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def get_last_token_value(cmd: List[str], prefixes: List[str]) -> str | None:
    for tok in reversed(cmd):
        for p in prefixes:
            if tok.startswith(p):
                return tok.split("=", 1)[1]
    return None


def safe_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(str(text).strip().strip('"'))
    except Exception:
        return None


def _scaled_dim(value: int, floor: int) -> int:
    reduced = max(floor, int(round(value * 0.75)))
    return int((reduced // 8) * 8) if reduced >= 8 else reduced


def build_oom_retry_command(cmd: List[str]) -> List[str]:
    retry = list(cmd)

    cur_train = safe_int(get_last_token_value(retry, ["++train_batch_size=", "train_batch_size="]))
    cur_eval = safe_int(get_last_token_value(retry, ["++eval_batch_size=", "eval_batch_size="]))

    next_train = 512 if cur_train is None else max(256, cur_train // 2)
    next_eval = 1024 if cur_eval is None else max(512, cur_eval // 2)

    retry.extend(
        [
            f"++train_batch_size={next_train}",
            f"++search.train_batch_size={hydra_literal([next_train])}",
            "++search_space_type_overrides.train_batch_size=choice",
            f"++eval_batch_size={next_eval}",
            f"++search.eval_batch_size={hydra_literal([next_eval])}",
            "++search_space_type_overrides.eval_batch_size=choice",
        ]
    )

    dim_specs = [
        ("hidden_size", 96),
        ("embedding_size", 96),
        ("inner_size", 192),
        ("attribute_hidden_size", 64),
    ]
    for dim_key, floor in dim_specs:
        cur_dim = safe_int(get_last_token_value(retry, [f"++{dim_key}=", f"{dim_key}="]))
        if cur_dim is None:
            continue
        next_dim = _scaled_dim(cur_dim, floor)
        if next_dim >= cur_dim:
            continue
        retry.extend(
            [
                f"++{dim_key}={next_dim}",
                f"++search.{dim_key}={hydra_literal([next_dim])}",
                f"++search_space_type_overrides.{dim_key}=choice",
            ]
        )

    has_attr_pred = get_last_token_value(retry, ["++use_attribute_predictor="])
    if has_attr_pred is not None:
        retry.extend(
            [
                "++use_attribute_predictor=false",
                "++search.use_attribute_predictor=[false]",
                "++search_space_type_overrides.use_attribute_predictor=choice",
                "++lambda_attr=0.0",
                "++search.lambda_attr=[0.0]",
                "++search_space_type_overrides.lambda_attr=choice",
            ]
        )

    return retry


def is_oom_log(log_path: Path) -> bool:
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    needles = [
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    ]
    return any(k in txt for k in needles)


def parse_trial_ok_counts(log_path: Path) -> tuple[int | None, int | None]:
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return (None, None)
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+trials\s+OK", txt, flags=re.IGNORECASE)
    if not m:
        return (None, None)
    try:
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return (None, None)


def should_retry_oom(log_path: Path, rc: int) -> tuple[bool, str]:
    # Retry on explicit non-zero return with OOM.
    has_oom = is_oom_log(log_path)
    if rc != 0 and has_oom:
        return (True, "rc_nonzero_with_oom")

    # Some hyperopt runs exit rc=0 even when all/most trials failed by OOM.
    ok_cnt, total_cnt = parse_trial_ok_counts(log_path)
    if not has_oom:
        return (False, "no_oom")
    if ok_cnt is None or total_cnt is None:
        return (False, "oom_but_no_trial_summary")
    if ok_cnt == 0 and total_cnt > 0:
        return (True, "oom_all_trials_failed")
    if ok_cnt < total_cnt:
        return (True, "oom_partial_trials_failed")
    return (False, "oom_but_trials_ok")


def run_one(cmd: List[str], log_path: Path, env: Dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# revised_ts={now_utc()}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(EXP_DIR),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
            env=env,
        )
        _active_proc_add(proc)
        try:
            while True:
                if STOP_EVENT.is_set() and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        pass
                    time.sleep(0.3)
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except Exception:
                            pass
                rc = proc.poll()
                if rc is not None:
                    return int(rc)
                time.sleep(0.2)
        finally:
            _active_proc_remove(proc)


def main() -> None:
    args = parse_args()
    STOP_EVENT.clear()
    install_signal_handlers()
    gpus_raw = parse_csv_list(args.gpus) or ["0"]
    gpus = unique_keep_order(gpus_raw)
    if len(gpus) != len(gpus_raw):
        relay(f"[revised][warn] duplicate GPUs removed: input={gpus_raw} unique={gpus}")

    targets = list(REVISED_TARGETS)
    if int(args.auto_extend_count) > 0:
        existing = {t.run_phase for t in targets}
        auto_specs = load_auto_extend_specs(
            tsv_path=str(args.auto_extend_rank_tsv),
            limit=int(args.auto_extend_count),
            max_ratio=float(args.auto_extend_max_ratio),
            existing_run_phases=existing,
        )
        if auto_specs:
            targets.extend(auto_specs)
        relay(
            f"[revised] auto_extend requested={int(args.auto_extend_count)} "
            f"added={len(auto_specs)} max_ratio={float(args.auto_extend_max_ratio):.3f}"
        )

    only_tokens = parse_csv_list(args.only_run_phases)
    if only_tokens:
        wanted = {normalize_phase_for_match(x, args.axis) for x in only_tokens}
        wanted = {x for x in wanted if x}
        targets = [t for t in targets if t.run_phase in wanted]
        relay(
            f"[revised] target_filter only_run_phases={len(only_tokens)} "
            f"matched={len(targets)}"
        )
    if int(args.max_runs) > 0:
        targets = targets[: int(args.max_runs)]

    log_map = find_log_map()
    axis_root = REVISED_LOG_ROOT / str(args.axis)
    axis_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    relay(f"[revised] axis={args.axis} targets={len(targets)} gpus={gpus}")

    runnable_jobs: List[tuple[RevisionSpec, Path]] = []
    for spec in targets:
        base_log = log_map.get(spec.run_phase)
        if base_log is None:
            relay(f"[revised][skip] run_phase={spec.run_phase} reason=base_log_not_found")
            manifest.append(
                {
                    "run_phase": spec.run_phase,
                    "status": "skip",
                    "error": "base_log_not_found",
                    "reason": spec.reason,
                    "ref_note": spec.ref_note,
                }
            )
            continue

        runnable_jobs.append((spec, base_log))

    for idx, (spec, base_log) in enumerate(runnable_jobs):
        gpu = gpus[idx % len(gpus)]
        cmd_base = parse_cmd_from_log(base_log)
        cmd = build_revised_command(cmd_base, spec, args.axis, gpu, args)
        run_phase_new = spec.run_phase.replace(BASE_AXIS, args.axis, 1) + "_REV1"
        log_path = axis_root / "revised_runs" / f"{run_phase_new}.log"

        relay(f"[revised][plan] {spec.run_phase}")
        relay(f"  reason: {spec.reason}")
        relay(f"  ref: {spec.ref_note}")
        if spec.train_batch_size is not None or spec.eval_batch_size is not None:
            relay(
                "  batch: "
                f"train={spec.train_batch_size if spec.train_batch_size is not None else 'unchanged'} "
                f"eval={spec.eval_batch_size if spec.eval_batch_size is not None else 'unchanged'}"
            )
        if spec.lr_range is not None:
            relay(f"  lr_range: [{spec.lr_range[0]}, {spec.lr_range[1]}]")
        if spec.fixed_overrides:
            relay(f"  fixed_overrides: {json.dumps(spec.fixed_overrides, ensure_ascii=False)}")

        if args.dry_run:
            manifest.append(
                {
                    "run_phase": spec.run_phase,
                    "run_phase_revised": run_phase_new,
                    "status": "dry-run",
                    "gpu": gpu,
                    "cuda_visible_devices": gpu,
                    "reason": spec.reason,
                    "ref_note": spec.ref_note,
                    "base_log": str(base_log),
                    "log_path": str(log_path),
                    "cmd": cmd,
                }
            )
    if args.dry_run:
        manifest_path = axis_root / "revised_manifest.json"
        manifest_path.write_text(json.dumps({"axis": args.axis, "generated_at": now_utc(), "runs": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        relay(f"[revised] manifest={manifest_path}")
        return

    job_queue: Queue[tuple[RevisionSpec, Path]] = Queue()
    for spec, base_log in runnable_jobs:
        job_queue.put((spec, base_log))

    manifest_lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            if STOP_EVENT.is_set():
                return
            try:
                spec, base_log = job_queue.get_nowait()
            except Empty:
                return

            cmd_base = parse_cmd_from_log(base_log)
            cmd = build_revised_command(cmd_base, spec, args.axis, gpu_id, args)
            run_phase_new = spec.run_phase.replace(BASE_AXIS, args.axis, 1) + "_REV1"
            log_path = axis_root / "revised_runs" / f"{run_phase_new}.log"
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            relay(f"[revised][start] {run_phase_new} gpu={gpu_id} visible={gpu_id}")
            rc = run_one(cmd, log_path, env=env)
            retried_oom = False
            retry_oom, retry_reason = should_retry_oom(log_path, rc)
            if retry_oom and not STOP_EVENT.is_set():
                retried_oom = True
                retry_cmd = build_oom_retry_command(cmd)
                relay(
                    f"[revised][oom-retry] {run_phase_new} gpu={gpu_id} "
                    f"reason={retry_reason} with tighter batch/dim"
                )
                rc = run_one(retry_cmd, log_path, env=env)
            status = "ok" if rc == 0 else "fail"
            relay(f"[revised][done] {run_phase_new} rc={rc} status={status}")

            with manifest_lock:
                manifest.append(
                    {
                        "run_phase": spec.run_phase,
                        "run_phase_revised": run_phase_new,
                        "status": status,
                        "rc": rc,
                        "gpu": gpu_id,
                        "cuda_visible_devices": gpu_id,
                        "oom_retry": retried_oom,
                        "reason": spec.reason,
                        "ref_note": spec.ref_note,
                        "base_log": str(base_log),
                        "log_path": str(log_path),
                    }
                )

            job_queue.task_done()

    threads: List[threading.Thread] = []
    for gpu in gpus:
        t = threading.Thread(target=worker, args=(gpu,), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    manifest_path = axis_root / "revised_manifest.json"
    manifest_path.write_text(json.dumps({"axis": args.axis, "generated_at": now_utc(), "runs": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    relay(f"[revised] manifest={manifest_path}")


if __name__ == "__main__":
    main()
