#!/usr/bin/env python3
"""Common helpers for FMoE_N4 ablation launchers."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
FMOE_N4_DIR = THIS_DIR.parent
FMOE_N3_DIR = FMOE_N4_DIR.parent / "fmoe_n3"
REPO_ROOT = THIS_DIR.parents[3]
RESULT_ROOT = REPO_ROOT / "experiments" / "run" / "artifacts" / "results" / "fmoe_n4"
LOGS_ROOT = REPO_ROOT / "experiments" / "run" / "artifacts" / "logs" / "fmoe_n4"
ABLATION_LOGS_ROOT = LOGS_ROOT / "ablation"

for extra_path in (FMOE_N4_DIR, FMOE_N3_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import stage1_a12_broad_templates as stage1  # noqa: E402
from run_phase9_auxloss import _parse_csv_ints, _parse_csv_strings, hydra_literal  # noqa: E402
from run_phase_wide_common import build_summary_fieldnames, launch_wide_rows, sanitize_token  # noqa: E402

TRACK = "fmoe_n4"
ARCH_ID = stage1.ARCH_ID
ARCH_KEY = stage1.ARCH_KEY
ARCH_NAME = stage1.ARCH_NAME
DEFAULT_DATASETS = ["beauty", "KuaiRecLargeStrictPosV2_0.2"]
DEFAULT_FEATURE_MODE = "full_v4"
DEFAULT_FEATURE_DATASET_DIR = "feature_added_v4"
DEFAULT_EVAL_MODE = "session_fixed"
DEFAULT_TOPK_PER_DATASET = 4
DEFAULT_GPUS = "cpu"

SETTING_TIER_CHOICES = ["essential", "extended", "optional", "essential_extended", "all"]

CATEGORY_DROP_KEYWORDS = ["cat", "theme"]
TIMESTAMP_DROP_KEYWORDS = [
    "timestamp",
    "gap",
    "pace",
    "int_",
    "_int",
    "sess_age",
    "ctx_valid_r",
    "valid_r",
    "delta_vs_mid",
]
FEATURE_GROUPS = ["Tempo", "Focus", "Memory", "Exposure"]

ALLOWED_BASE_AXES_BY_DATASET: Dict[str, tuple[str, ...]] = {
    "beauty": ("crossdataset_a12_portfolio",),
    "KuaiRecLargeStrictPosV2_0.2": (
        "stage1_a12_broadtemplates",
        "stage2_a12_mixedtemplates",
        "stage3_a12_seenfocus",
    ),
}

PREFERRED_BASE_RUN_PHASE_HINTS: Dict[str, tuple[str, ...]] = {
    "beauty": (
        "B25_LR_H8_SEEN_ANCHOR",
        "B17_HYP_MIDAUX_LOWAUX_H3",
    ),
    "KuaiRecLargeStrictPosV2_0.2": (
        "S02_H14_SEEN_HI",
        "S07_H10_LEN25_F24",
    ),
}

PRESET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "scout": {
        "seeds": "1",
        "max_evals": 10,
        "tune_epochs": 100,
        "tune_patience": 10,
        "lr_mode": "band9",
        "smoke_test": False,
        "smoke_max_runs": 2,
    },
    "confirm": {
        "seeds": "1,2,3,4",
        "max_evals": 10,
        "tune_epochs": 100,
        "tune_patience": 10,
        "lr_mode": "band9",
        "smoke_test": False,
        "smoke_max_runs": 2,
    },
    "smoke": {
        "seeds": "1",
        "max_evals": 1,
        "tune_epochs": 2,
        "tune_patience": 1,
        "lr_mode": "tight3",
        "smoke_test": True,
        "smoke_max_runs": 2,
    },
}

OVERRIDE_KEYS = {
    "layer_layout",
    "stage_compute_mode",
    "stage_router_mode",
    "stage_router_source",
    "stage_router_granularity",
    "stage_feature_injection",
    "stage_router_wrapper",
    "stage_router_primitives",
    "stage_feature_family_mask",
    "stage_feature_family_topk",
    "stage_feature_family_custom",
    "stage_feature_drop_keywords",
    "stage_feature_dropout_prob",
    "stage_family_dropout_prob",
    "stage_feature_dropout_scope",
    "route_consistency_lambda",
    "z_loss_lambda",
    "balance_loss_lambda",
    "topk_scope_mode",
    "moe_top_k",
    "macro_history_window",
    "route_consistency_pairs",
    "route_consistency_min_sim",
    "bias_mode",
    "feature_perturb_mode",
    "feature_perturb_apply",
    "feature_perturb_family",
    "feature_perturb_keywords",
    "feature_perturb_shift",
    "gate_entropy_lambda",
    "route_smoothness_lambda",
    "route_sharpness_lambda",
    "route_monopoly_lambda",
    "route_monopoly_tau",
    "route_prior_lambda",
    "group_prior_align_lambda",
    "factored_group_balance_lambda",
    "rule_agreement_lambda",
    "group_coverage_lambda",
    "feature_group_bias_lambda",
    "rule_bias_scale",
    "primitive_balance_lambda",
    "wrapper_group_feature_align_lambda",
}

IGNORED_COMMAND_KEYS = {
    "special_logging",
    "exclude_unseen_target_from_main_eval",
    "log_unseen_target_metrics",
    "fmoe_debug_logging",
    "fmoe_diag_logging",
    "fmoe_special_logging",
    "fmoe_feature_ablation_logging",
    "fmoe_eval_logging_timing",
    "fmoe_logging_output_root",
    "fmoe_logging_layout",
    "fmoe_architecture_id",
    "fmoe_architecture_key",
    "fmoe_hparam_id",
    "fmoe_phase",
    "phase_run_type",
    "phase_axis_id",
    "phase_setting_key",
    "phase_seed_id",
    "phase_run_id",
}

SUMMARY_EXTRA_COLS = [
    "architecture_id",
    "architecture_name",
    "tuning_stage",
    "family_id",
    "family_group",
    "variant_id",
    "capacity_anchor",
    "search_algo",
    "setting_tier",
    "setting_id",
    "setting_key",
    "setting_group",
    "setting_detail",
    "base_dataset",
    "base_rank",
    "base_key",
    "base_run_phase",
    "base_result_json",
    "base_source",
    "base_test_mrr20",
    "base_valid_mrr20",
    "base_setting_id",
    "base_capacity_anchor",
    "source_feature_mode",
    "runtime_feature_mode",
    "runtime_eval_mode",
    "lr_mode",
]


@dataclass(frozen=True)
class ResultCandidate:
    result_json: str
    dataset: str
    run_phase: str
    run_axis: str
    test_mrr20: float
    valid_mrr20: Optional[float]
    logging_bundle_summary_file: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if out != out or out in {float("inf"), float("-inf")}:
            return None
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            out = float(text)
        except Exception:
            return None
        if out != out or out in {float("inf"), float("-inf")}:
            return None
        return out
    return None


def _nested_metric(block: Any, *keys: str) -> Optional[float]:
    current: Any = block
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _metric_to_float(current)


def _normalize_search_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, dict):
        return copy.deepcopy(spec)
    return {"type": "choice", "values": list(spec if isinstance(spec, list) else [spec])}


def _choice_spec(values: Iterable[Any]) -> Dict[str, Any]:
    return {"type": "choice", "values": list(values)}


def _all_stage_mask(groups: Sequence[str]) -> Dict[str, list[str]]:
    return {
        "macro": list(groups),
        "mid": list(groups),
        "micro": list(groups),
    }


def _json_token(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _dedupe_keep_order(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        token = _json_token(value)
        if token in seen:
            continue
        seen.add(token)
        out.append(value)
    return out


def _parse_hydra_value(raw_value: str) -> Any:
    text = str(raw_value)
    if text == "":
        return ""

    def _skip_ws(cursor: int) -> int:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    def _coerce_scalar(token: str) -> Any:
        lowered = token.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if token.startswith("0") and token not in {"0", "0.0"} and not any(ch in token for ch in ".eE"):
                raise ValueError()
            return int(token)
        except Exception:
            pass
        try:
            return float(token)
        except Exception:
            return token

    def _parse_quoted(cursor: int) -> tuple[str, int]:
        quote = text[cursor]
        cursor += 1
        chars: list[str] = []
        while cursor < len(text):
            ch = text[cursor]
            if ch == "\\" and cursor + 1 < len(text):
                chars.append(text[cursor + 1])
                cursor += 2
                continue
            if ch == quote:
                return "".join(chars), cursor + 1
            chars.append(ch)
            cursor += 1
        raise RuntimeError(f"unterminated quoted hydra literal: {text}")

    def _parse_scalar(cursor: int) -> tuple[Any, int]:
        start = cursor
        while cursor < len(text) and text[cursor] not in {",", "]", "}"}:
            cursor += 1
        token = text[start:cursor].strip()
        if token == "":
            raise RuntimeError(f"empty hydra literal token in: {text}")
        return _coerce_scalar(token), cursor

    def _parse_key(cursor: int) -> tuple[str, int]:
        start = cursor
        while cursor < len(text) and text[cursor] not in {":", ",", "}"}:
            cursor += 1
        token = text[start:cursor].strip()
        if token == "":
            raise RuntimeError(f"empty hydra dict key in: {text}")
        return token, cursor

    def _parse_value(cursor: int) -> tuple[Any, int]:
        cursor = _skip_ws(cursor)
        if cursor >= len(text):
            raise RuntimeError(f"unexpected end of hydra literal: {text}")
        ch = text[cursor]
        if ch == "{":
            return _parse_dict(cursor + 1)
        if ch == "[":
            return _parse_list(cursor + 1)
        if ch in {"'", '"'}:
            return _parse_quoted(cursor)
        return _parse_scalar(cursor)

    def _parse_list(cursor: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        cursor = _skip_ws(cursor)
        if cursor < len(text) and text[cursor] == "]":
            return items, cursor + 1
        while True:
            value, cursor = _parse_value(cursor)
            items.append(value)
            cursor = _skip_ws(cursor)
            if cursor >= len(text):
                raise RuntimeError(f"unterminated hydra list: {text}")
            if text[cursor] == ",":
                cursor += 1
                continue
            if text[cursor] == "]":
                return items, cursor + 1
            raise RuntimeError(f"unexpected hydra list token at {cursor}: {text}")

    def _parse_dict(cursor: int) -> tuple[dict[str, Any], int]:
        mapping: dict[str, Any] = {}
        cursor = _skip_ws(cursor)
        if cursor < len(text) and text[cursor] == "}":
            return mapping, cursor + 1
        while True:
            cursor = _skip_ws(cursor)
            if cursor >= len(text):
                raise RuntimeError(f"unterminated hydra dict: {text}")
            if text[cursor] in {"'", '"'}:
                key_value, cursor = _parse_quoted(cursor)
            else:
                key_value, cursor = _parse_key(cursor)
            cursor = _skip_ws(cursor)
            if cursor >= len(text) or text[cursor] != ":":
                raise RuntimeError(f"missing ':' in hydra dict: {text}")
            cursor += 1
            value, cursor = _parse_value(cursor)
            mapping[str(key_value)] = value
            cursor = _skip_ws(cursor)
            if cursor >= len(text):
                raise RuntimeError(f"unterminated hydra dict: {text}")
            if text[cursor] == ",":
                cursor += 1
                continue
            if text[cursor] == "}":
                return mapping, cursor + 1
            raise RuntimeError(f"unexpected hydra dict token at {cursor}: {text}")

    parsed, cursor = _parse_value(0)
    cursor = _skip_ws(cursor)
    if cursor != len(text):
        raise RuntimeError(f"trailing hydra literal content: {text[cursor:]} in {text}")
    return parsed


def validate_session_fixed_files(dataset: str, feature_dataset_dir: str = DEFAULT_FEATURE_DATASET_DIR) -> None:
    ds_dir = REPO_ROOT / "Datasets" / "processed" / str(feature_dataset_dir) / str(dataset)
    required = [
        ds_dir / f"{dataset}.train.inter",
        ds_dir / f"{dataset}.valid.inter",
        ds_dir / f"{dataset}.test.inter",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"session_fixed files missing for dataset={dataset}: {missing}")


def build_lr_choices(base_lr: float, mode: str) -> list[float]:
    base = float(base_lr)
    if base <= 0:
        raise RuntimeError(f"invalid base learning rate: {base_lr}")
    if mode == "screen5":
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    elif mode == "tight3":
        multipliers = [0.85, 1.0, 1.15]
    elif mode == "band9":
        multipliers = [0.70, 0.82, 0.90, 0.96, 1.0, 1.04, 1.10, 1.18, 1.30]
    else:
        raise RuntimeError(f"unsupported lr_mode: {mode}")
    values = [round(base * float(multiplier), 12) for multiplier in multipliers]
    return _dedupe_keep_order(value for value in values if value > 0)


def clone_base_overrides(base_overrides: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(base_overrides)


def apply_delta_overrides(base_overrides: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base_overrides)
    for key, value in dict(delta or {}).items():
        merged[str(key)] = copy.deepcopy(value)
    return merged


def common_arg_parser(description: str, *, default_datasets: Sequence[str] | None = None, default_scope: str = "core") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datasets", default=",".join(default_datasets or DEFAULT_DATASETS))
    parser.add_argument("--base-result-jsons", default="", help="CSV explicit base result json paths")
    parser.add_argument("--topk-per-dataset", type=int, default=DEFAULT_TOPK_PER_DATASET)
    parser.add_argument("--num-base-per-dataset", type=int, default=0)
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--logs-root", default=str(LOGS_ROOT))
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="CSV GPU ids, or cpu")
    parser.add_argument("--preset", choices=sorted(PRESET_DEFAULTS), default="scout")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed-base", type=int, default=426000)
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument("--tune-epochs", type=int, default=0)
    parser.add_argument("--tune-patience", type=int, default=0)
    parser.add_argument("--lr-mode", choices=["screen5", "tight3", "band9"], default="")
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="random")
    parser.add_argument("--setting-scope", choices=["core", "appendix", "all"], default=default_scope)
    parser.add_argument("--setting-tier", choices=SETTING_TIER_CHOICES, default="essential")
    parser.add_argument("--only-setting", default="")
    parser.add_argument("--only-base", default="")
    parser.add_argument("--feature-mode", default=DEFAULT_FEATURE_MODE)
    parser.add_argument("--feature-dataset-dir", default=DEFAULT_FEATURE_DATASET_DIR)
    parser.add_argument("--eval-mode", default=DEFAULT_EVAL_MODE)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    return parser


def finalize_common_args(args: argparse.Namespace) -> argparse.Namespace:
    preset = dict(PRESET_DEFAULTS[str(args.preset)])
    if not str(args.seeds).strip():
        args.seeds = str(preset["seeds"])
    if int(getattr(args, "max_evals", 0) or 0) <= 0:
        args.max_evals = int(preset["max_evals"])
    if int(getattr(args, "tune_epochs", 0) or 0) <= 0:
        args.tune_epochs = int(preset["tune_epochs"])
    if int(getattr(args, "tune_patience", 0) or 0) <= 0:
        args.tune_patience = int(preset["tune_patience"])
    if not str(getattr(args, "lr_mode", "") or "").strip():
        args.lr_mode = str(preset["lr_mode"])
    if bool(preset.get("smoke_test", False)):
        args.smoke_test = True
        if int(getattr(args, "smoke_max_runs", 0) or 0) <= 0:
            args.smoke_max_runs = int(preset["smoke_max_runs"])
    if int(getattr(args, "num_base_per_dataset", 0) or 0) > 0:
        args.topk_per_dataset = int(args.num_base_per_dataset)
    if int(args.max_evals) < 1:
        raise RuntimeError("--max-evals must be >= 1")
    if int(args.tune_epochs) < 1:
        raise RuntimeError("--tune-epochs must be >= 1")
    if int(args.tune_patience) < 1:
        raise RuntimeError("--tune-patience must be >= 1")
    return args


def maybe_limit_smoke(rows: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    if not bool(getattr(args, "smoke_test", False)):
        return rows
    return list(rows[: max(1, int(getattr(args, "smoke_max_runs", 2) or 2))])


def _extract_test_mrr(payload: Dict[str, Any]) -> Optional[float]:
    test_filter = payload.get("test_main_eval_filter") or {}
    if bool(test_filter.get("enabled", False)):
        direct = _metric_to_float(payload.get("test_mrr@20"))
        if direct is not None:
            return direct
        test_result = payload.get("test_result")
        if isinstance(test_result, dict):
            return _metric_to_float(test_result.get("mrr@20") or test_result.get("MRR@20"))
        return None
    return (
        _nested_metric(payload.get("test_special_metrics"), "overall_seen_target", "mrr@20")
        or _metric_to_float(payload.get("test_mrr@20"))
        or _nested_metric(payload.get("test_result"), "mrr@20")
        or _nested_metric(payload.get("test_result"), "MRR@20")
    )


def _extract_valid_mrr(payload: Dict[str, Any]) -> Optional[float]:
    valid_filter = payload.get("best_valid_main_eval_filter") or {}
    if bool(valid_filter.get("enabled", False)):
        for key in ("best_valid_mrr@20", "best_mrr@20", "best_valid_score"):
            value = _metric_to_float(payload.get(key))
            if value is not None:
                return value
        valid_result = payload.get("best_valid_result")
        if isinstance(valid_result, dict):
            return _metric_to_float(valid_result.get("mrr@20") or valid_result.get("MRR@20"))
        return None
    return (
        _nested_metric(payload.get("best_valid_special_metrics"), "overall_seen_target", "mrr@20")
        or _metric_to_float(payload.get("best_valid_mrr@20"))
        or _metric_to_float(payload.get("best_mrr@20"))
        or _metric_to_float(payload.get("best_valid_score"))
        or _nested_metric(payload.get("best_valid_result"), "mrr@20")
        or _nested_metric(payload.get("best_valid_result"), "MRR@20")
    )


def _extract_best_lr(payload: Dict[str, Any]) -> Optional[float]:
    best_params = payload.get("best_params")
    if isinstance(best_params, dict):
        lr = _metric_to_float(best_params.get("learning_rate"))
        if lr is not None:
            return lr
    return _metric_to_float(payload.get("best_learning_rate"))


def scan_result_candidates(result_root: Path, datasets: Sequence[str]) -> list[ResultCandidate]:
    dataset_filter = set(str(dataset) for dataset in datasets)
    deduped: Dict[tuple[str, str], ResultCandidate] = {}
    for path in sorted(Path(result_root).glob("*.json")):
        if path.name == "meta.json":
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        dataset = str(payload.get("dataset") or "").strip()
        run_phase = str(payload.get("run_phase") or "").strip()
        run_axis = str(payload.get("run_axis") or "").strip()
        if dataset not in dataset_filter or not run_phase:
            continue
        if run_axis.startswith("ablation_") or "transfer" in run_axis:
            continue
        allowed_axes = ALLOWED_BASE_AXES_BY_DATASET.get(dataset)
        if allowed_axes and run_axis not in allowed_axes:
            continue
        test_mrr20 = _extract_test_mrr(payload)
        if test_mrr20 is None:
            continue
        candidate = ResultCandidate(
            result_json=str(path),
            dataset=dataset,
            run_phase=run_phase,
            run_axis=run_axis,
            test_mrr20=float(test_mrr20),
            valid_mrr20=_extract_valid_mrr(payload),
            logging_bundle_summary_file=str(payload.get("logging_bundle_summary_file") or ""),
        )
        key = (candidate.dataset, candidate.run_phase)
        prev = deduped.get(key)
        if prev is None:
            deduped[key] = candidate
            continue
        prev_key = (prev.test_mrr20, prev.valid_mrr20 or -1.0, prev.result_json)
        next_key = (candidate.test_mrr20, candidate.valid_mrr20 or -1.0, candidate.result_json)
        if next_key > prev_key:
            deduped[key] = candidate
    return list(deduped.values())


def _preferred_base_rank(candidate: ResultCandidate) -> int:
    hints = PREFERRED_BASE_RUN_PHASE_HINTS.get(str(candidate.dataset), ())
    run_phase = str(candidate.run_phase).upper()
    for idx, hint in enumerate(hints):
        if str(hint).upper() in run_phase:
            return idx
    return len(hints) + 100


def select_top_result_candidates(result_root: Path, datasets: Sequence[str], topk_per_dataset: int) -> list[ResultCandidate]:
    grouped: Dict[str, list[ResultCandidate]] = {str(dataset): [] for dataset in datasets}
    for candidate in scan_result_candidates(result_root, datasets):
        grouped.setdefault(candidate.dataset, []).append(candidate)
    selected: list[ResultCandidate] = []
    for dataset in datasets:
        dataset_rows = sorted(
            grouped.get(str(dataset), []),
            key=lambda row: (
                -_preferred_base_rank(row),
                row.test_mrr20,
                row.valid_mrr20 or -1.0,
                row.result_json,
            ),
            reverse=True,
        )
        if not dataset_rows:
            raise RuntimeError(f"No candidate results found for dataset={dataset} under {result_root}")
        selected.extend(dataset_rows[: int(topk_per_dataset)])
    return selected


@lru_cache(maxsize=4)
def _manifest_index(logs_root: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(logs_root).rglob("*manifest*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        for row in list(payload.get("rows") or []):
            run_phase = str(row.get("run_phase") or "").strip()
            if not run_phase:
                continue
            if run_phase in index:
                continue
            item = copy.deepcopy(row)
            item["manifest_path"] = str(path)
            item["manifest_axis"] = str(payload.get("axis") or "")
            index[run_phase] = item
    return index


def _read_log_head(path: Path, max_lines: int = 24) -> str:
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(max_lines):
                line = handle.readline()
                if not line:
                    break
                lines.append(line)
    except Exception:
        return ""
    return "".join(lines)


def _find_command_log(payload: Dict[str, Any], result_json: Path, logs_root: Path) -> Path:
    run_phase = str(payload.get("run_phase") or "").strip()
    dataset = str(payload.get("dataset") or "").strip()
    if not run_phase:
        raise RuntimeError(f"run_phase missing in result json: {result_json}")
    for log_path in sorted(logs_root.rglob("*.log")):
        if dataset and dataset not in log_path.parts:
            continue
        head = _read_log_head(log_path)
        if not head:
            continue
        if f"run_phase={run_phase}" in head:
            return log_path
    raise RuntimeError(f"Could not locate command log for run_phase={run_phase} result={result_json}")


def _parse_command_log(log_path: Path) -> Dict[str, Any]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        raise RuntimeError(f"Failed to read log file {log_path}: {exc}") from exc
    command_line = ""
    for idx, line in enumerate(lines):
        if str(line).strip() != "[COMMAND]":
            continue
        for next_line in lines[idx + 1 :]:
            text = str(next_line).strip()
            if not text:
                break
            if text.startswith("- "):
                command_line = text[2:].strip()
                break
        break
    if not command_line:
        raise RuntimeError(f"Could not find [COMMAND] block in {log_path}")

    tokens = shlex.split(command_line)
    parsed: Dict[str, Any] = {
        "axis": "",
        "run_phase": "",
        "dataset": "",
        "search_algo": "random",
        "max_evals": 0,
        "tune_epochs": 0,
        "tune_patience": 0,
        "feature_mode": "",
        "eval_mode": "",
        "fixed_values": {},
        "overrides": {},
        "search_space": {},
        "search_space_type_overrides": {},
        "phase_setting_id": "",
        "phase_hparam_id": "",
        "phase_axis_desc": "",
        "train_batch_size": 0,
        "eval_batch_size": 0,
    }
    idx = 0
    while idx < len(tokens):
        token = str(tokens[idx])
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                key, value_text = key.split("=", 1)
            else:
                if idx + 1 >= len(tokens):
                    break
                value_text = str(tokens[idx + 1])
                idx += 1
            if key == "run-axis":
                parsed["axis"] = value_text
            elif key == "run-phase":
                parsed["run_phase"] = value_text
            elif key == "search-algo":
                parsed["search_algo"] = value_text
            elif key == "max-evals":
                parsed["max_evals"] = int(value_text)
            elif key == "tune-epochs":
                parsed["tune_epochs"] = int(value_text)
            elif key == "tune-patience":
                parsed["tune_patience"] = int(value_text)
            idx += 1
            continue

        if "=" not in token:
            idx += 1
            continue

        key, value_text = token.split("=", 1)
        is_plus = key.startswith("++")
        normalized_key = key.lstrip("+")
        value = _parse_hydra_value(value_text)

        if normalized_key == "dataset":
            parsed["dataset"] = str(value)
        elif normalized_key == "feature_mode":
            parsed["feature_mode"] = str(value)
        elif normalized_key == "eval_mode":
            parsed["eval_mode"] = str(value)
        elif normalized_key == "train_batch_size":
            parsed["train_batch_size"] = int(value)
        elif normalized_key == "eval_batch_size":
            parsed["eval_batch_size"] = int(value)
        elif normalized_key == "phase_setting_id":
            parsed["phase_setting_id"] = str(value)
        elif normalized_key == "phase_hparam_id":
            parsed["phase_hparam_id"] = str(value)
        elif normalized_key == "phase_axis_desc":
            parsed["phase_axis_desc"] = str(value)
        elif normalized_key.startswith("search_space_type_overrides."):
            search_key = normalized_key.split(".", 1)[1]
            parsed["search_space_type_overrides"][search_key] = str(value)
        elif normalized_key.startswith("search."):
            search_key = normalized_key.split(".", 1)[1]
            parsed["search_space"][search_key] = _normalize_search_spec(value)
        elif normalized_key.startswith("eval_sampling.") or normalized_key in IGNORED_COMMAND_KEYS:
            pass
        elif normalized_key in {"model", "gpu_id", "use_gpu", "log_wandb", "enable_tf32"}:
            pass
        elif is_plus or normalized_key in OVERRIDE_KEYS or normalized_key.startswith("stage_"):
            target = parsed["overrides"] if normalized_key in OVERRIDE_KEYS or normalized_key.startswith("stage_") else parsed["fixed_values"]
            target[normalized_key] = value
        else:
            parsed["fixed_values"][normalized_key] = value
        idx += 1

    parsed["log_path"] = str(log_path)
    return parsed


def resolve_base_spec(
    result_json: str | Path,
    *,
    logs_root: str | Path = LOGS_ROOT,
    feature_mode_runtime: str = DEFAULT_FEATURE_MODE,
    eval_mode_runtime: str = DEFAULT_EVAL_MODE,
) -> Dict[str, Any]:
    result_path = Path(result_json)
    payload = _load_json(result_path)
    run_phase = str(payload.get("run_phase") or "").strip()
    if not run_phase:
        raise RuntimeError(f"run_phase missing in result json: {result_path}")

    manifest_row = _manifest_index(str(logs_root)).get(run_phase)
    row_source = "manifest"
    row_payload: Dict[str, Any]
    if manifest_row is not None:
        row_payload = copy.deepcopy(manifest_row)
    else:
        row_source = "command_log"
        log_path = _find_command_log(payload, result_path, Path(logs_root))
        row_payload = _parse_command_log(log_path)

    base_overrides = copy.deepcopy(dict(row_payload.get("overrides") or {}))
    fixed_values = copy.deepcopy(dict(row_payload.get("fixed_values") or {}))
    best_params = copy.deepcopy(dict(payload.get("best_params") or {}))
    best_lr = _extract_best_lr(payload)
    if best_lr is None:
        raise RuntimeError(f"best learning rate missing in result json: {result_path}")
    for key, value in best_params.items():
        if str(key) == "learning_rate":
            continue
        if str(key) in base_overrides:
            base_overrides[str(key)] = copy.deepcopy(value)
        else:
            fixed_values[str(key)] = copy.deepcopy(value)

    setting_id = str(row_payload.get("setting_id") or row_payload.get("phase_setting_id") or run_phase)
    capacity_anchor = str(row_payload.get("capacity_anchor") or row_payload.get("phase_hparam_id") or "")
    dataset = str(payload.get("dataset") or row_payload.get("dataset") or "")
    source_feature_mode = str(row_payload.get("feature_mode") or payload.get("feature_mode") or "")
    source_eval_mode = str(row_payload.get("eval_mode") or payload.get("eval_mode") or "")

    return {
        "result_json": str(result_path),
        "dataset": dataset,
        "run_phase": run_phase,
        "run_axis": str(payload.get("run_axis") or row_payload.get("manifest_axis") or row_payload.get("axis") or ""),
        "setting_id": setting_id,
        "capacity_anchor": capacity_anchor,
        "architecture_id": str(row_payload.get("architecture_id") or ARCH_ID),
        "architecture_key": str(row_payload.get("architecture_key") or ARCH_KEY),
        "architecture_name": str(row_payload.get("architecture_name") or ARCH_NAME),
        "base_key": f"{dataset}:{setting_id}",
        "base_token": sanitize_token(f"{dataset}_{setting_id}", upper=True),
        "source": row_source,
        "source_manifest_file": str(row_payload.get("manifest_path") or ""),
        "source_log_file": str(row_payload.get("log_path") or ""),
        "source_feature_mode": source_feature_mode,
        "runtime_feature_mode": str(feature_mode_runtime),
        "source_eval_mode": source_eval_mode,
        "runtime_eval_mode": str(eval_mode_runtime),
        "search_algo": str(row_payload.get("search_algo") or "random"),
        "best_learning_rate": float(best_lr),
        "test_mrr20": _extract_test_mrr(payload),
        "valid_mrr20": _extract_valid_mrr(payload),
        "fixed_values": fixed_values,
        "overrides": base_overrides,
        "search_space": copy.deepcopy(dict(row_payload.get("search_space") or {})),
        "search_space_type_overrides": copy.deepcopy(dict(row_payload.get("search_space_type_overrides") or {})),
        "train_batch_size": int(row_payload.get("train_batch_size") or payload.get("train_batch_size") or 4096),
        "eval_batch_size": int(row_payload.get("eval_batch_size") or payload.get("eval_batch_size") or 4096),
        "max_evals": int(row_payload.get("max_evals") or payload.get("max_evals") or 1),
        "tune_epochs": int(row_payload.get("tune_epochs") or payload.get("tune_epochs") or 1),
        "tune_patience": int(row_payload.get("tune_patience") or payload.get("tune_patience") or 1),
        "selected_from_stage": str(row_payload.get("selected_from_stage") or row_payload.get("source_family_id") or ""),
        "selection_score": str(row_payload.get("selection_score") or ""),
        "logging_bundle_summary_file": str(payload.get("logging_bundle_summary_file") or ""),
        "logging_bundle_dir": str(payload.get("logging_bundle_dir") or ""),
    }


def resolve_base_specs_from_args(args: argparse.Namespace) -> list[Dict[str, Any]]:
    explicit = _parse_csv_strings(str(getattr(args, "base_result_jsons", "") or ""))
    if explicit:
        result_paths = [Path(path) for path in explicit]
    else:
        datasets = _parse_csv_strings(args.datasets)
        candidates = select_top_result_candidates(Path(args.result_root), datasets, int(args.topk_per_dataset))
        result_paths = [Path(candidate.result_json) for candidate in candidates]

    base_specs: list[Dict[str, Any]] = []
    by_dataset_rank: Dict[str, int] = {}
    for result_path in result_paths:
        base = resolve_base_spec(
            result_path,
            logs_root=args.logs_root,
            feature_mode_runtime=args.feature_mode,
            eval_mode_runtime=args.eval_mode,
        )
        dataset = str(base["dataset"])
        by_dataset_rank[dataset] = by_dataset_rank.get(dataset, 0) + 1
        base["base_rank"] = int(by_dataset_rank[dataset])
        base_specs.append(base)
    return base_specs


def filter_settings(settings: Sequence[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    scope = str(getattr(args, "setting_scope", "core") or "core")
    tier = str(getattr(args, "setting_tier", "essential") or "essential")
    only_setting = {str(value) for value in _parse_csv_strings(str(getattr(args, "only_setting", "") or ""))}
    out: list[Dict[str, Any]] = []
    for setting in settings:
        setting_scope = str(setting.get("scope", "core"))
        setting_tier = str(setting.get("tier", "essential"))
        if scope != "all" and setting_scope != scope:
            continue
        if tier == "essential_extended":
            if setting_tier not in {"essential", "extended"}:
                continue
        elif tier != "all" and setting_tier != tier:
            continue
        if only_setting:
            tokens = {
                str(setting.get("setting_id", "")),
                str(setting.get("setting_key", "")),
            }
            if tokens.isdisjoint(only_setting):
                continue
        out.append(copy.deepcopy(setting))
    return out


def build_study_rows(
    *,
    args: argparse.Namespace,
    base_specs: Sequence[Dict[str, Any]],
    settings: Sequence[Dict[str, Any]],
    phase_id: str,
    axis_id: str,
    axis_desc: str,
    stage_name: str,
    diag_logging: bool,
    special_logging: bool,
    feature_ablation_logging: bool,
) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds selected")
    only_base = {str(value) for value in _parse_csv_strings(str(getattr(args, "only_base", "") or ""))}

    rows: list[Dict[str, Any]] = []
    cursor = 0
    for base in base_specs:
        dataset = str(base["dataset"])
        validate_session_fixed_files(dataset, args.feature_dataset_dir)
        base_tokens = {
            str(base.get("base_key", "")),
            str(base.get("setting_id", "")),
            str(base.get("run_phase", "")),
            str(base.get("base_rank", "")),
        }
        if only_base and base_tokens.isdisjoint(only_base):
            continue

        for setting in settings:
            base_overrides = clone_base_overrides(dict(base.get("overrides") or {}))
            fixed_values = copy.deepcopy(dict(base.get("fixed_values") or {}))
            delta_builder = setting.get("delta_builder")
            if callable(delta_builder):
                delta_overrides = dict(delta_builder(base) or {})
            else:
                delta_overrides = dict(setting.get("delta_overrides") or {})
            overrides = apply_delta_overrides(base_overrides, delta_overrides)
            for key in list(overrides.keys()):
                fixed_values.pop(str(key), None)
            if not bool(setting.get("force_identity", False)) and _json_token(overrides) == _json_token(base_overrides):
                continue

            search_space = {
                "learning_rate": _choice_spec(build_lr_choices(float(base["best_learning_rate"]), str(args.lr_mode)))
            }

            for seed_id in seeds:
                cursor += 1
                run_suffix = (
                    f"{sanitize_token(dataset, upper=True)}_"
                    f"B{int(base['base_rank']):02d}_"
                    f"{sanitize_token(str(base['setting_id']), upper=True)}_"
                    f"{sanitize_token(str(setting['setting_id']), upper=True)}_"
                    f"S{int(seed_id)}"
                )
                run_phase = f"{phase_id}_{run_suffix}"
                family_id = f"B{int(base['base_rank']):02d}_{base['setting_id']}__{setting['setting_id']}"
                rows.append(
                    {
                        "track": TRACK,
                        "axis_id": axis_id,
                        "axis_desc": axis_desc,
                        "phase_id": phase_id,
                        "architecture_id": str(base.get("architecture_id") or ARCH_ID),
                        "architecture_key": str(base.get("architecture_key") or ARCH_KEY),
                        "architecture_name": str(base.get("architecture_name") or ARCH_NAME),
                        "exp_brief": str(base.get("architecture_name") or ARCH_NAME),
                        "dataset": dataset,
                        "run_phase": run_phase,
                        "run_id": run_suffix,
                        "stage": stage_name,
                        "tuning_stage": stage_name,
                        "setting_id": str(setting["setting_id"]),
                        "setting_key": str(setting["setting_key"]),
                        "setting_tier": str(setting.get("tier") or "essential"),
                        "setting_desc": str(setting.get("setting_desc") or setting["setting_key"]),
                        "setting_group": str(setting.get("setting_group") or "ablation"),
                        "setting_detail": str(setting.get("setting_detail") or ""),
                        "family_id": family_id,
                        "family_group": str(setting.get("setting_group") or "ablation"),
                        "variant_id": str(setting.get("variant_id") or setting["setting_id"]),
                        "capacity_anchor": str(base.get("capacity_anchor") or ""),
                        "search_algo": str(args.search_algo),
                        "seed_id": int(seed_id),
                        "runtime_seed": int(args.seed_base) + cursor - 1,
                        "fixed_values": fixed_values,
                        "search_space": search_space,
                        "overrides": overrides,
                        "train_batch_size": int(base.get("train_batch_size") or 4096),
                        "eval_batch_size": int(base.get("eval_batch_size") or 4096),
                        "max_evals": int(args.max_evals),
                        "tune_epochs": int(args.tune_epochs),
                        "tune_patience": int(args.tune_patience),
                        "feature_mode": str(args.feature_mode),
                        "eval_mode": str(args.eval_mode),
                        "diag_logging": bool(diag_logging),
                        "special_logging": bool(special_logging),
                        "feature_ablation_logging": bool(feature_ablation_logging),
                        "base_dataset": dataset,
                        "base_rank": int(base["base_rank"]),
                        "base_key": str(base["base_key"]),
                        "base_run_phase": str(base["run_phase"]),
                        "base_result_json": str(base["result_json"]),
                        "base_source": str(base["source"]),
                        "base_test_mrr20": float(base.get("test_mrr20") or 0.0),
                        "base_valid_mrr20": float(base.get("valid_mrr20") or 0.0),
                        "base_setting_id": str(base.get("setting_id") or ""),
                        "base_capacity_anchor": str(base.get("capacity_anchor") or ""),
                        "source_feature_mode": str(base.get("source_feature_mode") or ""),
                        "runtime_feature_mode": str(args.feature_mode),
                        "runtime_eval_mode": str(args.eval_mode),
                        "lr_mode": str(args.lr_mode),
                    }
                )
    return rows


def build_log_path(*, log_dir: Path, row: Dict[str, Any], phase_id: str) -> Path:
    dataset_dir = log_dir / str(row["dataset"])
    family_dir = dataset_dir / sanitize_token(str(row["family_id"]), upper=False)
    filename = (
        f"{phase_id}_"
        f"{sanitize_token(str(row['base_setting_id']), upper=True)}_"
        f"{sanitize_token(str(row['setting_id']), upper=True)}_"
        f"S{int(row['seed_id'])}.log"
    )
    return family_dir / filename


def _device_assignments(gpu_id: str) -> list[str]:
    token = str(gpu_id).strip().lower()
    if token in {"cpu", "-1", "none"}:
        return ["gpu_id=", "use_gpu=false", "enable_tf32=false"]
    return [f"gpu_id={gpu_id}", "use_gpu=true", "enable_tf32=true"]


def build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = str(Path("/venv/FMoE/bin/python"))
    if not Path(python_bin).exists():
        python_bin = sys.executable

    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--search-algo",
        str(row["search_algo"]),
        "--max-evals",
        str(int(row["max_evals"])),
        "--tune-epochs",
        str(int(row["tune_epochs"])),
        "--tune-patience",
        str(int(row["tune_patience"])),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        str(row.get("track", TRACK)),
        "--run-axis",
        str(args.axis),
        "--run-phase",
        str(row["run_phase"]),
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        f"eval_mode={row['eval_mode']}",
        f"feature_mode={row['feature_mode']}",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++special_logging=true",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
        * _device_assignments(gpu_id),
        "log_wandb=false",
        f"fmoe_debug_logging=false",
        f"fmoe_diag_logging={'true' if row.get('diag_logging', True) else 'false'}",
        f"fmoe_special_logging={'true' if row.get('special_logging', True) else 'false'}",
        f"fmoe_feature_ablation_logging={'true' if row.get('feature_ablation_logging', False) else 'false'}",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_logging_layout={hydra_literal('axis_dataset_arch_hparam')}",
        f"++fmoe_architecture_id={hydra_literal(row['architecture_id'])}",
        f"++fmoe_architecture_key={hydra_literal(row['architecture_key'])}",
        f"++fmoe_hparam_id={hydra_literal(row['base_capacity_anchor'])}",
        f"++fmoe_phase={hydra_literal(row['phase_id'])}",
        f"train_batch_size={int(row['train_batch_size'])}",
        f"eval_batch_size={int(row['eval_batch_size'])}",
        f"++phase_run_type={hydra_literal(row['stage'])}",
        f"++phase_axis_id={hydra_literal(row['axis_id'])}",
        f"++phase_axis_desc={hydra_literal(row['axis_desc'])}",
        f"++phase_setting_id={hydra_literal(row['setting_id'])}",
        f"++phase_setting_key={hydra_literal(row['setting_key'])}",
        f"++phase_hparam_id={hydra_literal(row['base_capacity_anchor'])}",
        f"++phase_seed_id={hydra_literal(int(row['seed_id']))}",
        f"++phase_run_id={hydra_literal(row['run_id'])}",
    ]
    for key, value in dict(row.get("overrides") or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, value in dict(row.get("fixed_values") or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, spec in dict(row.get("search_space") or {}).items():
        normalized = _normalize_search_spec(spec)
        cmd.append(f"++search.{key}={hydra_literal(normalized['values'])}")
        cmd.append(f"++search_space_type_overrides.{key}={normalized['type']}")
    return cmd


def summary_path(log_root: Path, dataset: str) -> Path:
    path = Path(log_root) / str(dataset) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(log_root: Path, args: argparse.Namespace, default_name: str) -> Path:
    if str(args.manifest_out or "").strip():
        raw = Path(str(args.manifest_out))
        if raw.suffix:
            return raw
        return raw / default_name
    return Path(log_root) / default_name


def serialize_base_spec(base: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "result_json",
        "dataset",
        "run_phase",
        "run_axis",
        "setting_id",
        "capacity_anchor",
        "base_rank",
        "base_key",
        "base_token",
        "source",
        "source_manifest_file",
        "source_log_file",
        "source_feature_mode",
        "runtime_feature_mode",
        "source_eval_mode",
        "runtime_eval_mode",
        "search_algo",
        "best_learning_rate",
        "test_mrr20",
        "valid_mrr20",
        "train_batch_size",
        "eval_batch_size",
        "max_evals",
        "tune_epochs",
        "tune_patience",
    }
    return {key: copy.deepcopy(base.get(key)) for key in keep}


def serialize_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "dataset",
        "phase_id",
        "axis_id",
        "axis_desc",
        "architecture_id",
        "architecture_key",
        "architecture_name",
        "exp_brief",
        "run_phase",
        "run_id",
        "stage",
        "tuning_stage",
        "setting_tier",
        "setting_id",
        "setting_key",
        "setting_desc",
        "setting_group",
        "setting_detail",
        "family_id",
        "family_group",
        "variant_id",
        "capacity_anchor",
        "search_algo",
        "seed_id",
        "runtime_seed",
        "fixed_values",
        "search_space",
        "overrides",
        "train_batch_size",
        "eval_batch_size",
        "max_evals",
        "tune_epochs",
        "tune_patience",
        "feature_mode",
        "eval_mode",
        "base_dataset",
        "base_rank",
        "base_key",
        "base_run_phase",
        "base_result_json",
        "base_source",
        "base_test_mrr20",
        "base_valid_mrr20",
        "base_setting_id",
        "base_capacity_anchor",
        "source_feature_mode",
        "runtime_feature_mode",
        "runtime_eval_mode",
        "lr_mode",
    }
    return {key: copy.deepcopy(row.get(key)) for key in keep if key in row}


def write_manifest(
    *,
    args: argparse.Namespace,
    log_root: Path,
    default_name: str,
    axis: str,
    phase_id: str,
    phase_name: str,
    base_specs: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
) -> Path:
    path = manifest_path(log_root, args, default_name)
    payload = {
        "track": TRACK,
        "axis": axis,
        "phase_id": phase_id,
        "phase_name": phase_name,
        "preset": str(args.preset),
        "setting_scope": str(args.setting_scope),
        "setting_tier": str(args.setting_tier),
        "feature_mode": str(args.feature_mode),
        "eval_mode": str(args.eval_mode),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_base_specs": [serialize_base_spec(base) for base in base_specs],
        "run_count": len(rows),
        "rows": [serialize_manifest_row(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_fieldnames(extra_cols: Sequence[str] | None = None) -> list[str]:
    merged = list(SUMMARY_EXTRA_COLS)
    for col in list(extra_cols or []):
        key = str(col).strip()
        if key and key not in merged:
            merged.append(key)
    return build_summary_fieldnames(merged)


def launch_rows(
    *,
    rows: list[Dict[str, Any]],
    args: argparse.Namespace,
    axis: str,
    phase_id: str,
    phase_name: str,
    log_root: Path,
    fieldnames: list[str],
) -> int:
    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPU or cpu token selected")
    return int(
        launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=axis,
            phase_id=phase_id,
            phase_name=phase_name,
            log_dir=Path(log_root),
            summary_path=Path(log_root) / "summary.csv",
            fieldnames=fieldnames,
            extra_cols=[
                col
                for col in fieldnames
                if col
                not in {
                    "global_best_valid_mrr20",
                    "run_best_valid_mrr20",
                    "run_phase",
                    "exp_brief",
                    "stage",
                    "trigger",
                    "dataset",
                    "seed_id",
                    "gpu_id",
                    "status",
                    "test_mrr20",
                    "n_completed",
                    "interrupted",
                    "special_ok",
                    "diag_ok",
                    "result_path",
                    "timestamp_utc",
                }
            ],
            build_command=build_command,
            build_log_path=build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: summary_path(Path(log_root), str(row["dataset"])),
        )
    )
