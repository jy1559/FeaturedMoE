#!/usr/bin/env python3
"""Unified deep verification wrapper for phase10~13 settings.

This launcher accepts user-selected settings from phase10/11/12/13 and runs
`N settings x M hparams x K seeds` on a single unified axis.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import run_phase10_feature_portability as phase10
import run_phase11_stage_semantics as phase11
import run_phase12_layout_composition as phase12
import run_phase13_feature_sanity as phase13
from run_phase9_auxloss import (  # noqa: E402
    ARTIFACT_ROOT,
    EXP_DIR,
    LOG_ROOT,
    TRACK,
    _dataset_tag,
    _load_result_index,
    _metric_to_float,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)

AXIS = "phase10_13_verification_wrapper_v1"
PHASE_ID = "P10_13_2"
PHASE_NAME = "PHASE10_13_VERIFICATION"
RUN_STAGE = "deep"
AXIS_DESC = "verification_wrapper"

PHASE_ORDER = ("P10", "P11", "P12", "P13")
AXIS_ID_BY_PHASE = {
    "P10": "A",
    "P11": "B",
    "P12": "C",
    "P13": "D",
}

TRIAL_METRIC_KV_PATTERN = re.compile(r"([A-Za-z0-9_@./-]+)=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")

HPARAM_BANK: Dict[str, Dict[str, float]] = {
    "H1": {
        "embedding_size": 128,
        "d_ff": 256,
        "d_expert_hidden": 128,
        "d_router_hidden": 64,
        "fixed_weight_decay": 1e-6,
        "fixed_hidden_dropout_prob": 0.15,
    },
    "H2": {
        "embedding_size": 160,
        "d_ff": 320,
        "d_expert_hidden": 160,
        "d_router_hidden": 80,
        "fixed_weight_decay": 5e-7,
        "fixed_hidden_dropout_prob": 0.12,
    },
    "H3": {
        "embedding_size": 160,
        "d_ff": 320,
        "d_expert_hidden": 160,
        "d_router_hidden": 80,
        "fixed_weight_decay": 2e-6,
        "fixed_hidden_dropout_prob": 0.18,
    },
    "H4": {
        "embedding_size": 112,
        "d_ff": 224,
        "d_expert_hidden": 112,
        "d_router_hidden": 56,
        "fixed_weight_decay": 3e-6,
        "fixed_hidden_dropout_prob": 0.20,
    },
    "H5": {
        "embedding_size": 168,
        "d_ff": 336,
        "d_expert_hidden": 168,
        "d_router_hidden": 84,
        "fixed_weight_decay": 1e-6,
        "fixed_hidden_dropout_prob": 0.15,
    },
    "H6": {
        "embedding_size": 144,
        "d_ff": 288,
        "d_expert_hidden": 144,
        "d_router_hidden": 72,
        "fixed_weight_decay": 1.5e-6,
        "fixed_hidden_dropout_prob": 0.17,
    },
    "H7": {
        "embedding_size": 160,
        "d_ff": 320,
        "d_expert_hidden": 160,
        "d_router_hidden": 80,
        "fixed_weight_decay": 1e-6,
        "fixed_hidden_dropout_prob": 0.15,
    },
    "H8": {
        "embedding_size": 128,
        "d_ff": 256,
        "d_expert_hidden": 128,
        "d_router_hidden": 64,
        "fixed_weight_decay": 2.5e-6,
        "fixed_hidden_dropout_prob": 0.19,
    },
}

HPARAM_PRIORITY = ("H1", "H3", "H2", "H4", "H5", "H6", "H7", "H8")


def _sanitize_token(text: str, *, upper: bool = True) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum():
            out.append(ch.upper() if upper else ch.lower())
        else:
            out.append("_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "x"


def _phase_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / _dataset_tag(dataset)


def _summary_path(dataset: str) -> Path:
    return _phase_log_dir(dataset) / "summary.csv"


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_out:
        return Path(args.manifest_out)
    return _phase_log_dir(args.dataset) / "verification_matrix.json"


def _summary_common_fieldnames() -> list[str]:
    return [
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
    ]


def _build_summary_fieldnames(extra_cols: list[str]) -> list[str]:
    base = _summary_common_fieldnames()
    for col in list(extra_cols or []):
        key = str(col).strip()
        if key and key not in base:
            base.append(key)
    return base


def _ensure_summary_csv(path: Path, fieldnames: list[str]) -> None:
    expected = ",".join(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
        except Exception:
            first = ""
        if first == expected:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.legacy_{ts}{path.suffix}")
        try:
            path.rename(backup)
        except Exception:
            pass
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


def _append_summary_event(path: Path, fieldnames: list[str], row: Dict[str, Any]) -> None:
    _ensure_summary_csv(path, fieldnames)
    payload = {key: row.get(key, "") for key in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writerow(payload)


def _load_summary_state(path: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "global_best_valid": None,
        "run_complete_written": set(),
    }
    if not path.exists():
        return state
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                trigger = str(row.get("trigger", "")).strip().lower()
                run_phase = str(row.get("run_phase", "")).strip()
                if trigger == "run_complete" and run_phase:
                    state["run_complete_written"].add(run_phase)
                g = _metric_to_float(row.get("global_best_valid_mrr20"))
                r = _metric_to_float(row.get("run_best_valid_mrr20"))
                for val in (g, r):
                    if val is None:
                        continue
                    if state["global_best_valid"] is None or val > float(state["global_best_valid"]):
                        state["global_best_valid"] = float(val)
    except Exception:
        return state
    return state


def _summary_format_metric(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{int(ndigits)}f}"


def _summary_row_base(
    *,
    row: Dict[str, Any],
    trigger: str,
    status: str,
    global_best_valid: Optional[float],
    run_best_valid: Optional[float],
    test_mrr20: Optional[float],
    n_completed: Optional[int],
    interrupted: Optional[bool],
    special_ok: Optional[bool],
    diag_ok: Optional[bool],
    result_path: str,
    extra_cols: list[str],
) -> Dict[str, Any]:
    payload = {
        "global_best_valid_mrr20": _summary_format_metric(global_best_valid),
        "run_best_valid_mrr20": _summary_format_metric(run_best_valid),
        "run_phase": row.get("run_phase", ""),
        "exp_brief": row.get("exp_brief", ""),
        "stage": row.get("stage", RUN_STAGE),
        "trigger": str(trigger),
        "dataset": row.get("dataset", ""),
        "seed_id": row.get("seed_id", ""),
        "gpu_id": row.get("assigned_gpu", ""),
        "status": str(status),
        "test_mrr20": _summary_format_metric(test_mrr20),
        "n_completed": "" if n_completed is None else int(n_completed),
        "interrupted": "" if interrupted is None else bool(interrupted),
        "special_ok": "" if special_ok is None else bool(special_ok),
        "diag_ok": "" if diag_ok is None else bool(diag_ok),
        "result_path": str(result_path or ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    for col in list(extra_cols or []):
        if col in payload:
            continue
        payload[col] = row.get(col, "")
    return payload


def _parse_trial_metrics_line(line: str) -> Optional[Dict[str, float]]:
    text = str(line or "")
    if "[TRIAL_METRICS]" not in text:
        return None
    metrics: Dict[str, float] = {}
    for key, raw_val in TRIAL_METRIC_KV_PATTERN.findall(text):
        val = _metric_to_float(raw_val)
        if val is None:
            continue
        metrics[str(key)] = float(val)
    return metrics or None


def _scan_trial_metric_updates(log_path: Path, start_offset: int) -> tuple[int, list[Dict[str, float]]]:
    updates: list[Dict[str, float]] = []
    new_offset = int(start_offset)
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(max(int(start_offset), 0))
            while True:
                line = fh.readline()
                if not line:
                    break
                parsed = _parse_trial_metrics_line(line)
                if parsed:
                    updates.append(parsed)
            new_offset = fh.tell()
    except Exception:
        return int(start_offset), []
    return new_offset, updates


def _is_completed_log(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False
    for line in reversed(lines):
        text = str(line).strip()
        if not text:
            continue
        if text == "[RUN_STATUS] END status=normal":
            return True
        return text.startswith("[RUN_STATUS] END status=normal ")
    return False


def _verify_special_diag_from_result(result_json_path: str) -> tuple[bool, bool, str]:
    path = Path(str(result_json_path or ""))
    if not path.exists():
        return False, False, f"result_missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, False, f"result_parse_error:{exc}"

    special_candidates = [
        payload.get("special_log_file", ""),
        payload.get("special_result_file", ""),
    ]
    diag_candidates = [
        payload.get("diag_raw_trial_summary_file", ""),
        payload.get("diag_tier_a_final_file", ""),
        payload.get("diag_meta_file", ""),
    ]

    special_ok = any(str(p).strip() and Path(str(p)).exists() for p in special_candidates)
    diag_ok = any(str(p).strip() and Path(str(p)).exists() for p in diag_candidates)
    detail = (
        f"special={special_ok} diag={diag_ok} "
        f"special_paths={sum(1 for p in special_candidates if str(p).strip())} "
        f"diag_paths={sum(1 for p in diag_candidates if str(p).strip())}"
    )
    return special_ok, diag_ok, detail


def _get_result_row_for_run_phase(dataset: str, run_phase: str, retries: int = 8, sleep_sec: float = 1.0) -> Optional[Dict[str, Any]]:
    for _ in range(max(int(retries), 1)):
        index = _load_result_index(dataset, AXIS)
        row = index.get(str(run_phase))
        if isinstance(row, dict):
            return row
        time.sleep(max(float(sleep_sec), 0.0))
    return None


def _record_trial_new_best_if_any(
    *,
    summary_path: Path,
    fieldnames: list[str],
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
    trial_metrics: Dict[str, float],
    extra_cols: list[str],
) -> None:
    run_best = _metric_to_float(trial_metrics.get("run_best_mrr20"))
    if run_best is None:
        run_best = _metric_to_float(trial_metrics.get("cur_best_mrr20"))
    if run_best is None:
        return

    prev_global = _metric_to_float(summary_state.get("global_best_valid"))
    if prev_global is not None and run_best <= prev_global:
        return

    new_global = run_best if prev_global is None else max(prev_global, run_best)
    payload = _summary_row_base(
        row=row,
        trigger="trial_new_best",
        status="running",
        global_best_valid=new_global,
        run_best_valid=run_best,
        test_mrr20=_metric_to_float(trial_metrics.get("run_test_mrr20")),
        n_completed=None,
        interrupted=None,
        special_ok=None,
        diag_ok=None,
        result_path="",
        extra_cols=extra_cols,
    )
    _append_summary_event(summary_path, fieldnames, payload)
    summary_state["global_best_valid"] = float(new_global)


def _record_run_complete_summary(
    *,
    dataset: str,
    summary_path: Path,
    fieldnames: list[str],
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
    extra_cols: list[str],
    verify_logging: bool,
    return_code: int,
) -> None:
    run_phase = str(row.get("run_phase", "") or "")
    if not run_phase:
        return

    run_complete_written = summary_state.get("run_complete_written")
    if isinstance(run_complete_written, set) and run_phase in run_complete_written:
        return

    result_row = _get_result_row_for_run_phase(dataset, run_phase, retries=10, sleep_sec=1.0)

    status_base = "run_complete" if int(return_code) == 0 else f"run_failed_rc{int(return_code)}"
    status = status_base
    run_best = None
    test_mrr = None
    n_completed = None
    interrupted = None
    result_path = ""
    special_ok = False
    diag_ok = False

    if isinstance(result_row, dict):
        run_best = _metric_to_float(result_row.get("best_mrr"))
        test_mrr = _metric_to_float(result_row.get("test_mrr"))
        n_completed = int(result_row.get("n_completed", 0) or 0)
        interrupted = bool(result_row.get("interrupted", False))
        result_path = str(result_row.get("path", "") or "")
        if result_path:
            special_ok, diag_ok, detail = _verify_special_diag_from_result(result_path)
            print(f"[wrapper][logging-check] run={run_phase} {detail} result={result_path}")
            if verify_logging and (not special_ok or not diag_ok):
                raise RuntimeError(
                    f"Logging verification failed for run_phase={run_phase} "
                    f"(special_ok={special_ok}, diag_ok={diag_ok})"
                )
        else:
            status = f"{status_base}_no_result_path"
    else:
        status = f"{status_base}_no_result"

    prev_global = _metric_to_float(summary_state.get("global_best_valid"))
    new_global = prev_global
    if run_best is not None:
        new_global = run_best if prev_global is None else max(prev_global, run_best)

    payload = _summary_row_base(
        row=row,
        trigger="run_complete",
        status=status,
        global_best_valid=new_global,
        run_best_valid=run_best,
        test_mrr20=test_mrr,
        n_completed=n_completed,
        interrupted=interrupted,
        special_ok=special_ok,
        diag_ok=diag_ok,
        result_path=result_path,
        extra_cols=extra_cols,
    )
    _append_summary_event(summary_path, fieldnames, payload)

    if new_global is not None:
        summary_state["global_best_valid"] = float(new_global)
    if not isinstance(run_complete_written, set):
        summary_state["run_complete_written"] = set()
    summary_state["run_complete_written"].add(run_phase)


def _phase10_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    build_args = argparse.Namespace(
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
        z_loss_lambda=float(args.z_loss_lambda),
        balance_loss_lambda=float(args.balance_loss_lambda),
        macro_history_window=int(args.macro_history_window),
        family_dropout_prob=float(args.family_dropout_prob),
        feature_dropout_prob=float(args.feature_dropout_prob),
        include_extra_24=True,
        only_setting="",
    )
    return phase10._build_settings(build_args)


def _phase11_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    build_args = argparse.Namespace(
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
        z_loss_lambda=float(args.z_loss_lambda),
        balance_loss_lambda=float(args.balance_loss_lambda),
        macro_history_window=int(args.macro_history_window),
        only_setting="",
    )
    return phase11._build_settings(build_args)


def _phase12_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    build_args = argparse.Namespace(
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
        z_loss_lambda=float(args.z_loss_lambda),
        balance_loss_lambda=float(args.balance_loss_lambda),
        macro_history_window=int(args.macro_history_window),
        only_setting="",
    )
    return phase12._build_settings(build_args)


def _phase13_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    build_args = argparse.Namespace(
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
        z_loss_lambda=float(args.z_loss_lambda),
        balance_loss_lambda=float(args.balance_loss_lambda),
        macro_history_window=int(args.macro_history_window),
        only_setting="",
    )
    return phase13._build_settings(build_args)


def _canonical_key_for_phase(phase_id: str, setting: Dict[str, Any]) -> str:
    if phase_id == "P10":
        return str(setting["setting_id"])
    return str(setting["setting_key"])


def _short_key_for_phase(phase_id: str, setting: Dict[str, Any]) -> str:
    if phase_id == "P10":
        return str(setting["setting_key"])
    canonical = str(setting["setting_key"])
    if "_" in canonical:
        return canonical.split("_", 1)[1]
    desc = str(setting.get("setting_desc", "")).strip()
    return desc or canonical


def _catalog_entries_for_phase(
    *,
    phase_id: str,
    source_axis: str,
    settings: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    for setting in settings:
        idx = int(setting["setting_idx"])
        canonical = _canonical_key_for_phase(phase_id, setting)
        short_key = _short_key_for_phase(phase_id, setting)
        setting_desc = str(setting.get("setting_desc", short_key) or short_key)
        setting_group = str(setting.get("setting_group", "") or "")
        setting_detail = str(setting.get("setting_detail", setting_desc) or setting_desc)
        entries.append(
            {
                "phase_id": phase_id,
                "axis_id": AXIS_ID_BY_PHASE[phase_id],
                "axis_desc": AXIS_DESC,
                "source_axis": str(source_axis),
                "setting_idx": idx,
                "setting_uid": f"{phase_id}_{idx:02d}",
                "canonical_key": canonical,
                "short_key": short_key,
                "setting_desc": setting_desc,
                "setting_group": setting_group,
                "setting_detail": setting_detail,
                "source_setting_id": str(setting.get("setting_id", "")),
                "source_setting_key": str(setting.get("setting_key", "")),
                "overrides": copy.deepcopy(dict(setting.get("overrides", {}) or {})),
            }
        )
    entries.sort(key=lambda x: int(x["setting_idx"]))
    return entries


def _build_catalog(args: argparse.Namespace) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    out.extend(
        _catalog_entries_for_phase(
            phase_id="P10",
            source_axis=str(phase10.AXIS),
            settings=_phase10_settings(args),
        )
    )
    out.extend(
        _catalog_entries_for_phase(
            phase_id="P11",
            source_axis=str(phase11.AXIS),
            settings=_phase11_settings(args),
        )
    )
    out.extend(
        _catalog_entries_for_phase(
            phase_id="P12",
            source_axis=str(phase12.AXIS),
            settings=_phase12_settings(args),
        )
    )
    out.extend(
        _catalog_entries_for_phase(
            phase_id="P13",
            source_axis=str(phase13.AXIS),
            settings=_phase13_settings(args),
        )
    )
    return out


def _catalog_by_phase(catalog: list[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {phase: [] for phase in PHASE_ORDER}
    for entry in catalog:
        grouped[str(entry["phase_id"])].append(entry)
    for phase in grouped:
        grouped[phase].sort(key=lambda x: int(x["setting_idx"]))
    return grouped


def _print_catalog(catalog: list[Dict[str, Any]]) -> None:
    grouped = _catalog_by_phase(catalog)
    parts = [f"{phase}={len(grouped.get(phase, []))}" for phase in PHASE_ORDER]
    print(f"[catalog] total={len(catalog)} " + " ".join(parts))
    for phase in PHASE_ORDER:
        rows = grouped.get(phase, [])
        print(f"[{phase}] count={len(rows)}")
        for row in rows:
            print(
                f"- {row['canonical_key']} | uid={row['setting_uid']} "
                f"| short={row['short_key']} | group={row['setting_group']}"
            )


def _load_settings_tokens_from_json(path: str) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("settings", [])
    if not isinstance(payload, list):
        raise ValueError("--settings-json must point to a JSON list (or object with `settings` list)")
    out: list[str] = []
    for raw in payload:
        token = str(raw).strip()
        if token:
            out.append(token)
    return out


def _collect_setting_tokens(args: argparse.Namespace) -> list[str]:
    tokens = list(_parse_csv_strings(args.settings))
    tokens.extend(_load_settings_tokens_from_json(args.settings_json))
    return [tok.strip() for tok in tokens if str(tok).strip()]


def _phase_entry_by_idx(phase_entries: list[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
    for entry in phase_entries:
        if int(entry["setting_idx"]) == int(idx):
            return entry
    return None


def _phase_entry_by_short_key(phase_entries: list[Dict[str, Any]], token_upper: str) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for entry in phase_entries:
        if token_upper == str(entry["short_key"]).upper():
            out.append(entry)
            continue
        if token_upper == str(entry["canonical_key"]).upper():
            out.append(entry)
            continue
        if token_upper == str(entry["source_setting_key"]).upper():
            out.append(entry)
    return out


def _bare_aliases(entry: Dict[str, Any]) -> set[str]:
    idx = int(entry["setting_idx"])
    aliases = {
        str(entry["canonical_key"]).upper(),
        str(entry["short_key"]).upper(),
        str(entry["setting_desc"]).upper(),
        str(entry["setting_uid"]).upper(),
        str(entry["source_setting_id"]).upper(),
        str(entry["source_setting_key"]).upper(),
        str(idx),
        f"{idx:02d}",
    }
    return {tok for tok in aliases if tok}


def _resolve_single_token(token: str, catalog: list[Dict[str, Any]]) -> Dict[str, Any]:
    token = str(token).strip()
    token_up = token.upper()
    if not token_up:
        raise ValueError("Empty setting token is not allowed")

    grouped = _catalog_by_phase(catalog)

    # 1) Canonical key exact
    for entry in catalog:
        if token_up == str(entry["canonical_key"]).upper():
            return entry

    # 2) Phase+idx tolerant format:
    #    - P10-01
    #    - P10_01
    #    - P10-01_ANY_SUFFIX
    #    - P10_01_ANY_SUFFIX
    # Description suffix is ignored and idx is authoritative.
    m = re.match(r"^(P(?:10|11|12|13))[-_](\d{1,2})(?:[_-].*)?$", token_up)
    if m:
        phase_id = m.group(1)
        idx = int(m.group(2))
        phase_entries = grouped.get(phase_id, [])
        found = _phase_entry_by_idx(phase_entries, idx)
        if found is None:
            raise ValueError(f"No setting found for `{token}` (resolved as {phase_id}:{idx:02d})")
        return found

    # 3) Phase-qualified pattern: Pxx:idx or Pxx:SHORT_KEY
    if ":" in token_up:
        lhs, rhs = token_up.split(":", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if lhs not in grouped:
            raise ValueError(f"Unknown phase prefix in token `{token}`. Use one of: {', '.join(PHASE_ORDER)}")
        if not rhs:
            raise ValueError(f"Invalid token `{token}`: missing rhs after `:`")
        phase_entries = grouped[lhs]
        # tolerate rhs like `01_ANY_DESC` by taking leading numeric idx.
        m_rhs = re.match(r"^(\d{1,2})(?:[_-].*)?$", rhs)
        if m_rhs:
            idx = int(m_rhs.group(1))
            found = _phase_entry_by_idx(phase_entries, idx)
            if found is None:
                raise ValueError(f"No setting found for `{lhs}:{rhs}`")
            return found
        if rhs.isdigit():
            found = _phase_entry_by_idx(phase_entries, int(rhs))
            if found is None:
                raise ValueError(f"No setting found for `{lhs}:{rhs}`")
            return found
        matches = _phase_entry_by_short_key(phase_entries, rhs)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"No setting found for `{token}`")
        raise ValueError(
            f"Ambiguous phase-qualified token `{token}`. Candidates: "
            + ", ".join(str(m["canonical_key"]) for m in matches)
        )

    # 4) Bare token: must resolve uniquely across all phases.
    matches = [entry for entry in catalog if token_up in _bare_aliases(entry)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No setting matched bare token `{token}`")
    raise ValueError(
        f"Ambiguous bare token `{token}`. Use canonical key or phase-qualified token. Candidates: "
        + ", ".join(str(m["canonical_key"]) for m in matches)
    )


def _resolve_selected_settings(tokens: list[str], catalog: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    selected: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for token in tokens:
        entry = _resolve_single_token(token, catalog)
        key = str(entry["canonical_key"]).upper()
        if key in seen:
            continue
        seen.add(key)
        selected.append(copy.deepcopy(entry))
    return selected


def _parse_hparam_token(token: str) -> str:
    raw = str(token).strip().upper()
    if not raw:
        raise ValueError("Empty hparam token is not allowed")
    if raw.startswith("H"):
        body = raw[1:]
    else:
        body = raw
    if not body.isdigit():
        raise ValueError(f"Invalid hparam token `{token}`. Use `H1` or `1` style.")
    num = int(body)
    if num < 1 or num > len(HPARAM_BANK):
        raise ValueError(f"Invalid hparam id `{token}`. Supported range is 1~{len(HPARAM_BANK)}")
    return f"H{num}"


def _select_hparams(args: argparse.Namespace) -> list[str]:
    if str(args.hparams or "").strip():
        out: list[str] = []
        seen = set()
        for token in _parse_csv_strings(args.hparams):
            hid = _parse_hparam_token(token)
            if hid in seen:
                continue
            seen.add(hid)
            out.append(hid)
        if not out:
            raise ValueError("--hparams produced empty selection")
        return out

    count = int(args.hparam_count)
    if count < 1 or count > len(HPARAM_PRIORITY):
        raise ValueError(f"--hparam-count must be within 1~{len(HPARAM_PRIORITY)}")
    return list(HPARAM_PRIORITY[:count])


def _hparam_num(hparam_id: str) -> int:
    return int(str(hparam_id).upper().replace("H", "", 1))


def _setting_desc_for_filename(entry: Dict[str, Any]) -> str:
    raw = (
        entry.get("short_key")
        or entry.get("setting_short")
        or entry.get("setting_desc")
        or "setting"
    )
    return _sanitize_token(str(raw), upper=False)


def _build_rows(
    *,
    args: argparse.Namespace,
    selected_settings: list[Dict[str, Any]],
    hparams: list[str],
    seeds: list[int],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    for setting in selected_settings:
        for hparam_id in hparams:
            hnum = _hparam_num(hparam_id)
            if hparam_id not in HPARAM_BANK:
                raise ValueError(f"Unknown hparam profile: {hparam_id}")
            for seed_id in seeds:
                run_cursor += 1
                axis_id = str(setting["axis_id"])
                setting_uid = str(setting["setting_uid"])
                run_id = f"{axis_id}{setting_uid}_H{hnum}_S{int(seed_id)}"
                run_phase = f"{PHASE_ID}_{axis_id}{setting_uid}_H{hnum}_S{int(seed_id)}"
                rows.append(
                    {
                        "dataset": args.dataset,
                        "phase_id": PHASE_ID,
                        "source_phase": str(setting["phase_id"]),
                        "source_axis": str(setting["source_axis"]),
                        "axis_id": axis_id,
                        "axis_desc": AXIS_DESC,
                        "setting_uid": setting_uid,
                        "setting_idx": int(setting["setting_idx"]),
                        "setting_key": str(setting["canonical_key"]),
                        "setting_short": str(setting["short_key"]),
                        "setting_desc": str(setting["setting_desc"]),
                        "setting_group": str(setting["setting_group"]),
                        "setting_detail": str(setting["setting_detail"]),
                        "hparam_id": str(hparam_id),
                        "hparam_num": int(hnum),
                        "seed_id": int(seed_id),
                        "run_phase": run_phase,
                        "run_id": run_id,
                        "exp_brief": (
                            f"{setting['canonical_key']} | {setting['setting_group']} | {setting['setting_detail']}"
                        ),
                        "stage": RUN_STAGE,
                        "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                        "overrides": copy.deepcopy(dict(setting.get("overrides", {}) or {})),
                    }
                )
    return rows


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    h = dict(HPARAM_BANK[str(row["hparam_id"])])
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    sched = _parse_csv_strings(args.search_lr_scheduler)
    if not sched:
        sched = ["warmup_cosine"]

    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(args.tune_epochs)),
        "--tune-patience",
        str(int(args.tune_patience)),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        str(row["run_phase"]),
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        "eval_mode=session",
        "feature_mode=full_v3",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_phase={hydra_literal(PHASE_ID)}",
        f"MAX_ITEM_LIST_LENGTH={int(args.max_item_list_length)}",
        f"train_batch_size={int(args.batch_size)}",
        f"eval_batch_size={int(args.batch_size)}",
        f"embedding_size={int(h['embedding_size'])}",
        f"num_heads={int(args.num_heads)}",
        f"attn_dropout_prob={hydra_literal(float(args.attn_dropout_prob))}",
        f"d_ff={int(h['d_ff'])}",
        f"d_feat_emb={int(args.d_feat_emb)}",
        f"d_expert_hidden={int(h['d_expert_hidden'])}",
        f"d_router_hidden={int(h['d_router_hidden'])}",
        f"expert_scale={int(args.expert_scale)}",
        f"++search.learning_rate={hydra_literal([float(args.search_lr_min), float(args.search_lr_max)])}",
        f"++search.weight_decay={hydra_literal([float(h['fixed_weight_decay'])])}",
        f"++search.hidden_dropout_prob={hydra_literal([float(h['fixed_hidden_dropout_prob'])])}",
        f"++search.lr_scheduler_type={hydra_literal(sched)}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search_space_type_overrides.lr_scheduler_type=choice",
        f"++phase_run_type={hydra_literal(RUN_STAGE)}",
        f"++phase_axis_id={hydra_literal(row['axis_id'])}",
        f"++phase_axis_desc={hydra_literal(row['axis_desc'])}",
        f"++phase_setting_uid={hydra_literal(row['setting_uid'])}",
        f"++phase_setting_key={hydra_literal(row['setting_key'])}",
        f"++phase_setting_desc={hydra_literal(row['setting_desc'])}",
        f"++phase_source_phase={hydra_literal(row['source_phase'])}",
        f"++phase_source_axis={hydra_literal(row['source_axis'])}",
        f"++phase_hparam_id={hydra_literal(row['hparam_id'])}",
        f"++phase_seed_id={hydra_literal(row['seed_id'])}",
        f"++phase_run_id={hydra_literal(row['run_id'])}",
    ]
    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _log_filename_from_row(*, row: Dict[str, Any]) -> str:
    axis_sid = f"{_sanitize_token(row.get('axis_id', 'A'), upper=True)}{_sanitize_token(row.get('setting_uid', 'X'), upper=True)}"
    axis_desc = _sanitize_token(str(row.get("axis_desc", AXIS_DESC)), upper=False)
    setting_desc = _setting_desc_for_filename(row)
    hparam_num = int(row.get("hparam_num", _hparam_num(str(row.get("hparam_id", "H1")))))
    seed_id = int(row.get("seed_id", 1))
    return f"{PHASE_ID}_{axis_sid}_{axis_desc}_{setting_desc}_H{hparam_num}_S{seed_id}.log"


def _log_path_from_row(*, log_dir: Path, row: Dict[str, Any]) -> Path:
    filename = _log_filename_from_row(row=row)
    source_phase = _sanitize_token(str(row.get("source_phase", "PXX")), upper=True)
    setting_folder = _sanitize_token(str(row.get("setting_key", "") or row.get("setting_uid", "setting")), upper=True)
    return log_dir / source_phase / setting_folder / filename


def _legacy_flat_log_path_from_row(*, log_dir: Path, row: Dict[str, Any]) -> Path:
    # Backward compatibility with pre-folder layout:
    # <log_dir>/<filename>.log
    return log_dir / _log_filename_from_row(row=row)
    return log_dir / filename


def _write_log_preamble(
    *,
    log_file: Path,
    row: Dict[str, Any],
    gpu_id: str,
    args: argparse.Namespace,
    cmd: list[str],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{PHASE_NAME}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} run_id={row.get('run_id','')} "
            f"source_phase={row.get('source_phase','')} source_axis={row.get('source_axis','')} "
            f"axis_id={row.get('axis_id','')} setting_uid={row.get('setting_uid','')} "
            f"setting_key={row.get('setting_key','')} hparam_id={row.get('hparam_id','')} "
            f"seed={row.get('seed_id','')}"
        ),
        f"desc={row.get('setting_desc','')}",
        f"dataset={row.get('dataset','')} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        (
            f"max_evals={getattr(args, 'max_evals', '')} tune_epochs={getattr(args, 'tune_epochs', '')} "
            f"tune_patience={getattr(args, 'tune_patience', '')}"
        ),
        f"seed={row.get('runtime_seed','')}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    *,
    args: argparse.Namespace,
    selected_settings: list[Dict[str, Any]],
    rows: list[Dict[str, Any]],
    hparams: list[str],
    seeds: list[int],
) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "dataset": args.dataset,
        "execution_type": RUN_STAGE,
        "setting_count": len(selected_settings),
        "hparam_count": len(hparams),
        "seed_count": len(seeds),
        "run_count": len(rows),
        "run_count_formula": f"{len(selected_settings)} x {len(hparams)} x {len(seeds)}",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_hparams": [
            {
                "hparam_id": hid,
                "embedding_size": int(HPARAM_BANK[hid]["embedding_size"]),
                "d_ff": int(HPARAM_BANK[hid]["d_ff"]),
                "d_expert_hidden": int(HPARAM_BANK[hid]["d_expert_hidden"]),
                "d_router_hidden": int(HPARAM_BANK[hid]["d_router_hidden"]),
                "fixed_weight_decay": float(HPARAM_BANK[hid]["fixed_weight_decay"]),
                "fixed_hidden_dropout_prob": float(HPARAM_BANK[hid]["fixed_hidden_dropout_prob"]),
            }
            for hid in hparams
        ],
        "selected_settings": [
            {
                "source_phase": s["phase_id"],
                "source_axis": s["source_axis"],
                "setting_idx": s["setting_idx"],
                "setting_uid": s["setting_uid"],
                "setting_key": s["canonical_key"],
                "setting_short": s["short_key"],
                "setting_desc": s["setting_desc"],
                "setting_group": s["setting_group"],
                "setting_detail": s["setting_detail"],
            }
            for s in selected_settings
        ],
        "rows": [
            {
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "source_phase": r["source_phase"],
                "axis_id": r["axis_id"],
                "setting_uid": r["setting_uid"],
                "setting_key": r["setting_key"],
                "hparam_id": r["hparam_id"],
                "seed_id": r["seed_id"],
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _launch_rows(
    *,
    rows: list[Dict[str, Any]],
    gpus: list[str],
    args: argparse.Namespace,
    log_dir: Path,
    summary_path: Path,
    fieldnames: list[str],
    extra_cols: list[str],
) -> int:
    if not rows:
        print(f"[{PHASE_ID}] no rows to run")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_summary_csv(summary_path, fieldnames)
    summary_state = _load_summary_state(summary_path)

    for idx, row in enumerate(rows):
        row["assigned_gpu"] = gpus[idx % len(gpus)]
        row["assigned_order"] = idx + 1

    runnable: list[Dict[str, Any]] = []
    skipped = 0
    skipped_legacy = 0
    for row in rows:
        lp = _log_path_from_row(log_dir=log_dir, row=row)
        legacy_lp = _legacy_flat_log_path_from_row(log_dir=log_dir, row=row)
        row["log_path"] = str(lp)
        row["legacy_log_path"] = str(legacy_lp)
        if bool(getattr(args, "resume_from_logs", True)):
            if _is_completed_log(lp):
                skipped += 1
                continue
            if legacy_lp != lp and _is_completed_log(legacy_lp):
                skipped += 1
                skipped_legacy += 1
                continue
        runnable.append(row)

    if skipped > 0:
        msg = f"[{PHASE_ID}] resume_from_logs=on: skipped {skipped} completed runs by strict log-end marker"
        if skipped_legacy > 0:
            msg += f" (legacy flat-path hits={skipped_legacy})"
        print(msg)

    if not runnable:
        print(f"[{PHASE_ID}] all runs are already completed by strict log markers")
        return 0

    if bool(getattr(args, "dry_run", False)):
        for row in runnable:
            lp = Path(str(row["log_path"]))
            cmd = _build_command(row, row["assigned_gpu"], args)
            print(
                f"[dry-run] gpu={row['assigned_gpu']} run_phase={row['run_phase']} "
                f"setting={row['setting_key']} -> {lp}"
            )
            print("          " + " ".join(cmd))
        return 0

    gpu_bins: Dict[str, deque[Dict[str, Any]]] = {gpu: deque() for gpu in gpus}
    for row in runnable:
        gpu_bins[str(row["assigned_gpu"])].append(row)

    active: Dict[str, Dict[str, Any]] = {}
    had_failures = False

    while True:
        for gpu_id in gpus:
            if gpu_id in active:
                continue
            if not gpu_bins[gpu_id]:
                continue

            row = gpu_bins[gpu_id].popleft()
            lp = Path(str(row["log_path"]))
            cmd = _build_command(row, gpu_id, args)
            _write_log_preamble(log_file=lp, row=row, gpu_id=gpu_id, args=args, cmd=cmd)

            env = dict(os.environ)
            env["HYPEROPT_RESULTS_DIR"] = str(ARTIFACT_ROOT / "results")
            with lp.open("a", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, cwd=EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)

            start_offset = 0
            try:
                start_offset = int(lp.stat().st_size)
            except Exception:
                start_offset = 0

            active[gpu_id] = {
                "proc": proc,
                "row": row,
                "log_path": lp,
                "offset": start_offset,
            }
            print(
                f"[launch] phase={PHASE_ID} gpu={gpu_id} run_phase={row.get('run_phase','')} "
                f"setting={row.get('setting_key','')}"
            )

        done_gpu: list[str] = []
        for gpu_id, slot in active.items():
            proc = slot["proc"]
            row = slot["row"]
            lp = slot["log_path"]

            prev_offset = int(slot.get("offset", 0))
            new_offset, updates = _scan_trial_metric_updates(lp, prev_offset)
            slot["offset"] = int(new_offset)

            for trial_metrics in updates:
                _record_trial_new_best_if_any(
                    summary_path=summary_path,
                    fieldnames=fieldnames,
                    summary_state=summary_state,
                    row=row,
                    trial_metrics=trial_metrics,
                    extra_cols=extra_cols,
                )

            rc = proc.poll()
            if rc is None:
                continue

            done_gpu.append(gpu_id)
            if int(rc) != 0:
                had_failures = True
            print(f"[done] phase={PHASE_ID} gpu={gpu_id} run_phase={row.get('run_phase','')} rc={rc} log={lp}")

            _record_run_complete_summary(
                dataset=args.dataset,
                summary_path=summary_path,
                fieldnames=fieldnames,
                summary_state=summary_state,
                row=row,
                extra_cols=extra_cols,
                verify_logging=bool(args.verify_logging),
                return_code=int(rc),
            )

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = any(gpu_bins[g] for g in gpus)
        if not pending and not active:
            break
        time.sleep(3)

    print(f"[{PHASE_ID}] summary updated: {summary_path}")
    return 1 if had_failures else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FMoE_N3 unified deep verification wrapper for phase10~13 settings"
    )

    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")

    parser.add_argument("--settings", default="", help="Comma-separated setting tokens")
    parser.add_argument("--settings-json", default="", help="JSON file path with setting token list")
    parser.add_argument("--list-settings", action="store_true", help="Print full setting catalog and exit")

    parser.add_argument("--hparam-count", type=int, default=2, help="Default count from priority H1,H3,H2,H4,H5,H6,H7,H8")
    parser.add_argument("--hparams", default="", help="Explicit hparams (e.g., 1,3 or H1,H3)")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--seed-base", type=int, default=74000)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)
    parser.add_argument("--z-loss-lambda", type=float, default=1e-4)
    parser.add_argument("--balance-loss-lambda", type=float, default=0.0)
    parser.add_argument("--macro-history-window", type=int, default=5)

    parser.add_argument("--family-dropout-prob", type=float, default=0.10)
    parser.add_argument("--feature-dropout-prob", type=float, default=0.15)

    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)

    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")

    parser.add_argument("--manifest-out", default="", help="Optional matrix JSON output path")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")

    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals = 1
    args.tune_epochs = 1
    args.tune_patience = 1
    args.seeds = "1"
    args.hparam_count = 1
    args.hparams = ""
    gpus = _parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"
    if not str(args.settings or "").strip() and not str(args.settings_json or "").strip():
        args.settings = "P10-00_FULL,P11-00_MACRO_MID_MICRO"


def main() -> int:
    args = _parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    catalog = _build_catalog(args)
    if args.list_settings:
        _print_catalog(catalog)
        return 0

    setting_tokens = _collect_setting_tokens(args)
    if not setting_tokens:
        raise RuntimeError("No settings provided. Use --settings / --settings-json or --list-settings")

    selected_settings = _resolve_selected_settings(setting_tokens, catalog)
    if not selected_settings:
        raise RuntimeError("No settings resolved from provided tokens")

    hparams = _select_hparams(args)
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs provided")

    rows = _build_rows(
        args=args,
        selected_settings=selected_settings,
        hparams=hparams,
        seeds=seeds,
    )
    if args.smoke_test:
        rows = rows[: max(int(args.smoke_max_runs), 1)]
    if not rows:
        raise RuntimeError("No rows to launch")

    manifest_path = _write_manifest(
        args=args,
        selected_settings=selected_settings,
        rows=rows,
        hparams=hparams,
        seeds=seeds,
    )

    print(
        f"[{PHASE_ID}] dataset={args.dataset} settings={len(selected_settings)} hparams={len(hparams)} "
        f"seeds={len(seeds)} rows={len(rows)} axis={AXIS} manifest={manifest_path}"
    )
    print(f"[{PHASE_ID}] selected_hparams={','.join(hparams)}")
    for s in selected_settings:
        print(f"[{PHASE_ID}][setting] {s['canonical_key']} (uid={s['setting_uid']}, source={s['phase_id']})")

    extra_cols = [
        "phase_id",
        "source_phase",
        "source_axis",
        "axis_id",
        "axis_desc",
        "setting_uid",
        "setting_idx",
        "setting_key",
        "setting_short",
        "setting_desc",
        "setting_group",
        "setting_detail",
        "hparam_id",
        "seed_id",
        "run_id",
    ]
    fieldnames = _build_summary_fieldnames(extra_cols)

    return _launch_rows(
        rows=rows,
        gpus=gpus,
        args=args,
        log_dir=_phase_log_dir(args.dataset),
        summary_path=_summary_path(args.dataset),
        fieldnames=fieldnames,
        extra_cols=extra_cols,
    )


if __name__ == "__main__":
    raise SystemExit(main())
