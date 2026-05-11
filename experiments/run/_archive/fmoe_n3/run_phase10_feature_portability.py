#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 Phase10 feature portability runs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from run_phase9_auxloss import (  # noqa: E402
    ARTIFACT_ROOT,
    EXP_DIR,
    LOG_ROOT,
    MODEL_TAG,
    TRACK,
    _apply_base_overrides,
    _base_definitions,
    _base_fixed_overrides,
    _dataset_tag,
    _load_result_index,
    _metric_to_float,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)

AXIS = "phase10_feature_portability_v1"
PHASE = "P10"

FAMILY_ORDER = ("Tempo", "Focus", "Memory", "Exposure")
STAGE_ORDER = ("macro", "mid", "micro")


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _sanitize_token(text: str) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum():
            out.append(ch.upper())
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "X"


def _mask_all(families: list[str]) -> Dict[str, list[str]]:
    selected = [fam for fam in FAMILY_ORDER if fam in set(str(x) for x in families)]
    return _all_stage_map(selected)


def _topk_all(k: int) -> Dict[str, Dict[str, int]]:
    return {
        stage: {family: int(k) for family in FAMILY_ORDER}
        for stage in STAGE_ORDER
    }


def _common_template_custom() -> Dict[str, Dict[str, list[str]]]:
    # Cross-stage reusable compact template (2 representative features per family/stage).
    return {
        "macro": {
            "Tempo": ["mac5_ctx_valid_r", "mac5_gap_last"],
            "Focus": ["mac5_theme_top1_mean", "mac5_theme_repeat_r"],
            "Memory": ["mac5_repeat_mean", "mac5_adj_item_overlap_mean"],
            "Exposure": ["mac5_pop_mean", "mac5_pop_ent_mean"],
        },
        "mid": {
            "Tempo": ["mid_valid_r", "mid_int_mean"],
            "Focus": ["mid_cat_top1", "mid_cat_switch_r"],
            "Memory": ["mid_repeat_r", "mid_item_uniq_r"],
            "Exposure": ["mid_pop_mean", "mid_pop_ent"],
        },
        "micro": {
            "Tempo": ["mic_valid_r", "mic_last_gap"],
            "Focus": ["mic_cat_switch_now", "mic_last_cat_mismatch_r"],
            "Memory": ["mic_is_recons", "mic_suffix_recons_r"],
            "Exposure": ["mic_last_pop", "mic_suffix_pop_ent"],
        },
    }


def _base_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    # Anchor: P9/P9_2 winning family (B4) + mild z stabilizer.
    base_cfg = _base_definitions()["B4"]
    overrides = _base_fixed_overrides()
    _apply_base_overrides(
        overrides=overrides,
        base_cfg=base_cfg,
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
    )
    overrides["z_loss_lambda"] = float(args.z_loss_lambda)
    overrides["balance_loss_lambda"] = float(args.balance_loss_lambda)
    overrides["macro_history_window"] = int(args.macro_history_window)
    overrides["stage_feature_family_mask"] = {}
    overrides["stage_feature_family_topk"] = {}
    overrides["stage_feature_family_custom"] = {}
    overrides["stage_feature_drop_keywords"] = []
    overrides["stage_family_dropout_prob"] = _all_stage_map(0.0)
    overrides["stage_feature_dropout_prob"] = _all_stage_map(0.0)
    overrides["stage_feature_dropout_scope"] = _all_stage_map("token")
    return overrides


def _setting(
    *,
    idx: int,
    key: str,
    group: str,
    desc: str,
    base: Dict[str, Any],
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    setting_code = f"{int(idx):02d}_{_sanitize_token(key)}"
    overrides = copy.deepcopy(base)
    for k, v in dict(extra_overrides or {}).items():
        overrides[str(k)] = copy.deepcopy(v)
    return {
        "setting_idx": int(idx),
        "setting_code": setting_code,
        "setting_id": f"P10-{int(idx):02d}_{key}",
        "setting_key": key,
        "setting_group": group,
        "setting_desc": desc,
        "overrides": overrides,
    }


def _build_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    base = _base_overrides(args)
    settings: list[Dict[str, Any]] = []

    # 4.1 Group subset lattice (15)
    settings.append(_setting(idx=0, key="FULL", group="group_subset", desc="All 4 families", base=base))
    settings.append(
        _setting(
            idx=1,
            key="Tempo",
            group="group_subset",
            desc="Tempo only",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Tempo"])}
        )
    )
    settings.append(
        _setting(
            idx=2,
            key="Focus",
            group="group_subset",
            desc="Focus only",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Focus"])}
        )
    )
    settings.append(
        _setting(
            idx=3,
            key="Memory",
            group="group_subset",
            desc="Memory only",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Memory"])}
        )
    )
    settings.append(
        _setting(
            idx=4,
            key="Exposure",
            group="group_subset",
            desc="Exposure only",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Exposure"])}
        )
    )
    settings.append(
        _setting(
            idx=5,
            key="Tempo_Focus",
            group="group_subset",
            desc="Tempo + Focus",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Tempo", "Focus"])}
        )
    )
    settings.append(
        _setting(
            idx=6,
            key="Tempo_Memory",
            group="group_subset",
            desc="Tempo + Memory",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Tempo", "Memory"])}
        )
    )
    settings.append(
        _setting(
            idx=7,
            key="Tempo_Exposure",
            group="group_subset",
            desc="Tempo + Exposure",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Tempo", "Exposure"])}
        )
    )
    settings.append(
        _setting(
            idx=8,
            key="Focus_Memory",
            group="group_subset",
            desc="Focus + Memory",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Focus", "Memory"])}
        )
    )
    settings.append(
        _setting(
            idx=9,
            key="Focus_Exposure",
            group="group_subset",
            desc="Focus + Exposure",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Focus", "Exposure"])}
        )
    )
    settings.append(
        _setting(
            idx=10,
            key="Memory_Exposure",
            group="group_subset",
            desc="Memory + Exposure",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Memory", "Exposure"])}
        )
    )
    settings.append(
        _setting(
            idx=11,
            key="Tempo_Focus_Memory",
            group="group_subset",
            desc="Tempo + Focus + Memory",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Tempo", "Focus", "Memory"])}
        )
    )
    settings.append(
        _setting(
            idx=12,
            key="Tempo_Focus_Exposure",
            group="group_subset",
            desc="Tempo + Focus + Exposure",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Tempo", "Focus", "Exposure"])}
        )
    )
    settings.append(
        _setting(
            idx=13,
            key="Tempo_Memory_Exposure",
            group="group_subset",
            desc="Tempo + Memory + Exposure",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Tempo", "Memory", "Exposure"])}
        )
    )
    settings.append(
        _setting(
            idx=14,
            key="Focus_Memory_Exposure",
            group="group_subset",
            desc="Focus + Memory + Exposure",
            base=base,
            extra_overrides={"stage_feature_family_mask": _mask_all(["Focus", "Memory", "Exposure"])}
        )
    )

    # 4.2 Intra-group reduction (3)
    settings.append(
        _setting(
            idx=15,
            key="TOP2_PER_GROUP",
            group="compactness",
            desc="Keep top-2 representative features per family/stage",
            base=base,
            extra_overrides={"stage_feature_family_topk": _topk_all(2)},
        )
    )
    settings.append(
        _setting(
            idx=16,
            key="TOP1_PER_GROUP",
            group="compactness",
            desc="Keep top-1 representative feature per family/stage",
            base=base,
            extra_overrides={"stage_feature_family_topk": _topk_all(1)},
        )
    )
    settings.append(
        _setting(
            idx=17,
            key="COMMON_TEMPLATE",
            group="compactness",
            desc="Fixed reusable common template across stages",
            base=base,
            extra_overrides={"stage_feature_family_custom": _common_template_custom()},
        )
    )

    # 4.3 Availability ablation (2)
    settings.append(
        _setting(
            idx=18,
            key="NO_CATEGORY",
            group="availability",
            desc="Drop category/theme-derived columns",
            base=base,
            extra_overrides={"stage_feature_drop_keywords": ["cat", "theme"]},
        )
    )
    settings.append(
        _setting(
            idx=19,
            key="NO_TIMESTAMP",
            group="availability",
            desc="Drop timestamp/pace/interval-derived columns",
            base=base,
            extra_overrides={
                "stage_feature_drop_keywords": [
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
            },
        )
    )

    # 4.4 Stochastic feature usage (2)
    settings.append(
        _setting(
            idx=20,
            key="FAMILY_DROPOUT",
            group="stochastic",
            desc="Train-time family dropout (session scope)",
            base=base,
            extra_overrides={
                "stage_family_dropout_prob": _all_stage_map(float(args.family_dropout_prob)),
                "stage_feature_dropout_scope": _all_stage_map("session"),
            },
        )
    )
    settings.append(
        _setting(
            idx=21,
            key="FEATURE_DROPOUT",
            group="stochastic",
            desc="Train-time element-wise feature dropout",
            base=base,
            extra_overrides={
                "stage_feature_dropout_prob": _all_stage_map(float(args.feature_dropout_prob)),
                "stage_feature_dropout_scope": _all_stage_map("token"),
            },
        )
    )

    if args.include_extra_24:
        settings.append(
            _setting(
                idx=22,
                key="NO_CATEGORY_NO_TIMESTAMP",
                group="availability_plus",
                desc="Drop both category/theme and timestamp-derived columns",
                base=base,
                extra_overrides={
                    "stage_feature_drop_keywords": [
                        "cat",
                        "theme",
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
                },
            )
        )
        settings.append(
            _setting(
                idx=23,
                key="COMMON_TEMPLATE_NO_CATEGORY",
                group="compactness_plus",
                desc="Common template while removing category/theme columns",
                base=base,
                extra_overrides={
                    "stage_feature_family_custom": _common_template_custom(),
                    "stage_feature_drop_keywords": ["cat", "theme"],
                },
            )
        )

    only = {tok.upper() for tok in _parse_csv_strings(args.only_setting)}
    if only:
        settings = [
            s for s in settings
            if str(s["setting_key"]).upper() in only or str(s["setting_id"]).upper() in only
        ]

    if not settings:
        raise RuntimeError("No phase10 settings selected. Check --only-setting / --include-extra-24.")

    # Ensure stable ordering after optional filtering.
    settings.sort(key=lambda x: int(x["setting_idx"]))
    return settings


def _run_phase_name(setting: Dict[str, Any], seed_id: int) -> str:
    return f"P10_{setting['setting_code']}_S{int(seed_id)}"


def _run_id(setting: Dict[str, Any], seed_id: int) -> str:
    return f"{setting['setting_code']}_S{int(seed_id)}"


def _build_rows(args: argparse.Namespace, settings: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")

    rows: list[Dict[str, Any]] = []
    seed_cursor = 0
    for setting in settings:
        for seed_id in seeds:
            run_phase = _run_phase_name(setting, seed_id)
            run_id = _run_id(setting, seed_id)
            rows.append(
                {
                    "dataset": args.dataset,
                    "setting_idx": int(setting["setting_idx"]),
                    "setting_id": str(setting["setting_id"]),
                    "setting_key": str(setting["setting_key"]),
                    "setting_group": str(setting["setting_group"]),
                    "setting_desc": str(setting["setting_desc"]),
                    "seed_id": int(seed_id),
                    "seed_offset": int(seed_cursor),
                    "run_phase": run_phase,
                    "run_id": run_id,
                    "overrides": copy.deepcopy(dict(setting.get("overrides", {}) or {})),
                }
            )
            seed_cursor += 1
    return rows


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
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
        str(int(args.seed_base) + int(row["seed_offset"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        row["run_phase"],
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
        f"++fmoe_phase={hydra_literal(PHASE)}",
        f"MAX_ITEM_LIST_LENGTH={int(args.max_item_list_length)}",
        f"train_batch_size={int(args.batch_size)}",
        f"eval_batch_size={int(args.batch_size)}",
        f"embedding_size={int(args.embedding_size)}",
        f"num_heads={int(args.num_heads)}",
        f"attn_dropout_prob={hydra_literal(float(args.attn_dropout_prob))}",
        f"d_ff={int(args.d_ff)}",
        f"d_feat_emb={int(args.d_feat_emb)}",
        f"d_expert_hidden={int(args.d_expert_hidden)}",
        f"d_router_hidden={int(args.d_router_hidden)}",
        f"expert_scale={int(args.expert_scale)}",
        "++layer_layout=[macro,mid,micro]",
        f"++search.learning_rate={hydra_literal([float(args.search_lr_min), float(args.search_lr_max)])}",
        f"++search.weight_decay={hydra_literal([float(args.fixed_weight_decay)])}",
        f"++search.hidden_dropout_prob={hydra_literal([float(args.fixed_hidden_dropout_prob)])}",
        f"++search.lr_scheduler_type={hydra_literal(_parse_csv_strings(args.search_lr_scheduler))}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search_space_type_overrides.lr_scheduler_type=choice",
        f"++p10_setting_id={hydra_literal(row['setting_id'])}",
        f"++p10_setting_key={hydra_literal(row['setting_key'])}",
        f"++p10_setting_group={hydra_literal(row['setting_group'])}",
        f"++p10_setting_desc={hydra_literal(row['setting_desc'])}",
        f"++p10_run_id={hydra_literal(row['run_id'])}",
    ]
    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _phase10_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset) / MODEL_TAG


def _phase10_axis_dataset_dir(dataset: str) -> Path:
    root = LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _phase10_summary_csv_path(dataset: str) -> Path:
    return _phase10_axis_dataset_dir(dataset) / "summary.csv"


def _summary_fieldnames() -> list[str]:
    return [
        "global_best_valid_mrr20",
        "run_best_valid_mrr20",
        "run_phase",
        "run_id",
        "exp_brief",
        "setting_id",
        "setting_key",
        "setting_group",
        "setting_desc",
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


def _ensure_summary_csv(path: Path) -> None:
    expected = ",".join(_summary_fieldnames())
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
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writeheader()


def _summary_format_metric(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{int(ndigits)}f}"


def _summary_exp_brief(row: Dict[str, Any]) -> str:
    return (
        f"{row.get('setting_key', '')}"
        f" | {row.get('setting_group', '')}"
        f" | {row.get('setting_desc', '')}"
    )


def _load_summary_state(path: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "global_best_valid": None,
        "run_complete_written": set(),
    }
    if not path.exists():
        return state
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            trigger = str(row.get("trigger", "")).strip().lower()
            run_phase = str(row.get("run_phase", "")).strip()
            if trigger == "run_complete" and run_phase:
                state["run_complete_written"].add(run_phase)
            global_col = _metric_to_float(row.get("global_best_valid_mrr20"))
            run_col = _metric_to_float(row.get("run_best_valid_mrr20"))
            for valid in (global_col, run_col):
                if valid is None:
                    continue
                if state["global_best_valid"] is None or valid > float(state["global_best_valid"]):
                    state["global_best_valid"] = float(valid)
    return state


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    _ensure_summary_csv(path)
    payload = {key: row.get(key, "") for key in _summary_fieldnames()}
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writerow(payload)


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
) -> Dict[str, Any]:
    return {
        "global_best_valid_mrr20": _summary_format_metric(global_best_valid),
        "run_best_valid_mrr20": _summary_format_metric(run_best_valid),
        "run_phase": row.get("run_phase", ""),
        "run_id": row.get("run_id", ""),
        "exp_brief": _summary_exp_brief(row),
        "setting_id": row.get("setting_id", ""),
        "setting_key": row.get("setting_key", ""),
        "setting_group": row.get("setting_group", ""),
        "setting_desc": row.get("setting_desc", ""),
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
        "result_path": result_path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _get_result_row_for_run_phase(dataset: str, run_phase: str, retries: int = 8, sleep_sec: float = 1.0) -> Optional[Dict[str, Any]]:
    for _ in range(max(int(retries), 1)):
        index = _load_result_index(dataset, AXIS)
        row = index.get(str(run_phase))
        if isinstance(row, dict):
            return row
        time.sleep(max(float(sleep_sec), 0.0))
    return None


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

    details = (
        f"special={special_ok} "
        f"diag={diag_ok} "
        f"special_paths={sum(1 for p in special_candidates if str(p).strip())} "
        f"diag_paths={sum(1 for p in diag_candidates if str(p).strip())}"
    )
    return special_ok, diag_ok, details


_TRIAL_METRIC_KV_PATTERN = re.compile(r"([A-Za-z0-9_@./-]+)=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")


def _parse_trial_metrics_line(line: str) -> Optional[Dict[str, float]]:
    text = str(line or "")
    if "[TRIAL_METRICS]" not in text:
        return None
    metrics: Dict[str, float] = {}
    for key, raw_val in _TRIAL_METRIC_KV_PATTERN.findall(text):
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


def _record_trial_new_best_if_any(
    *,
    summary_path: Path,
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
    trial_metrics: Dict[str, float],
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
    )
    _append_summary_row(summary_path, payload)
    summary_state["global_best_valid"] = float(new_global)


def _record_run_complete_summary(
    *,
    dataset: str,
    summary_path: Path,
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
    verify_logging: bool,
) -> None:
    run_phase = str(row.get("run_phase", "") or "")
    if not run_phase:
        return
    run_complete_written = summary_state.get("run_complete_written")
    if isinstance(run_complete_written, set) and run_phase in run_complete_written:
        return

    result_row = _get_result_row_for_run_phase(dataset, run_phase, retries=10, sleep_sec=1.0)
    run_best = None
    test_mrr = None
    n_completed = None
    interrupted = None
    result_path = ""
    special_ok = False
    diag_ok = False
    status = "run_complete"

    if isinstance(result_row, dict):
        run_best = _metric_to_float(result_row.get("best_mrr"))
        test_mrr = _metric_to_float(result_row.get("test_mrr"))
        n_completed = int(result_row.get("n_completed", 0) or 0)
        interrupted = bool(result_row.get("interrupted", False))
        result_path = str(result_row.get("path", "") or "")
        if result_path:
            special_ok, diag_ok, detail = _verify_special_diag_from_result(result_path)
            print(f"[phase10][logging-check] run={row['run_id']} {detail} result={result_path}")
            if verify_logging and (not special_ok or not diag_ok):
                raise RuntimeError(
                    f"Logging verification failed for run={row['run_id']} (special_ok={special_ok}, diag_ok={diag_ok})"
                )
        else:
            status = "run_complete_no_result_path"
    else:
        status = "run_complete_no_result"

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
    )
    _append_summary_row(summary_path, payload)

    if new_global is not None:
        summary_state["global_best_valid"] = float(new_global)
    if not isinstance(run_complete_written, set):
        summary_state["run_complete_written"] = set()
    summary_state["run_complete_written"].add(run_phase)


def _log_path(row: Dict[str, Any], dataset: str) -> Path:
    root = _phase10_log_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{row['run_id']}.log"


def _is_completed_log_strict(log_path: Path) -> bool:
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
        # Allow trailing runtime metadata, e.g.
        # "[RUN_STATUS] END status=normal pid=... start=... end=..."
        return text.startswith("[RUN_STATUS] END status=normal ")
    return False


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    lines = [
        "[PHASE10_SETTING_HEADER]",
        (
            f"run_id={row['run_id']} run_phase={row['run_phase']} "
            f"setting_id={row['setting_id']} setting_key={row['setting_key']} "
            f"setting_group={row['setting_group']} seed={row['seed_id']}"
        ),
        f"desc={row['setting_desc']}",
        f"dataset={row['dataset']} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={int(args.seed_base) + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    # Overwrite stale/incomplete log by design.
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _write_matrix_manifest(rows: list[Dict[str, Any]], settings: list[Dict[str, Any]], args: argparse.Namespace) -> Path:
    if args.manifest_out:
        out_path = Path(args.manifest_out)
    else:
        out_path = _phase10_axis_dataset_dir(args.dataset) / "feature_portability_matrix.json"
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE,
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_settings": len(settings),
        "n_rows": len(rows),
        "settings": [
            {
                "setting_idx": s["setting_idx"],
                "setting_id": s["setting_id"],
                "setting_key": s["setting_key"],
                "setting_group": s["setting_group"],
                "setting_desc": s["setting_desc"],
            }
            for s in settings
        ],
        "rows": [
            {
                "run_id": r["run_id"],
                "run_phase": r["run_phase"],
                "setting_id": r["setting_id"],
                "setting_key": r["setting_key"],
                "seed_id": r["seed_id"],
            }
            for r in rows
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _launch_rows(rows: list[Dict[str, Any]], gpus: list[str], args: argparse.Namespace) -> int:
    if not rows:
        print("[phase10] no rows to run.")
        return 0

    for idx, row in enumerate(rows):
        row["assigned_gpu"] = gpus[idx % len(gpus)]
        row["assigned_order"] = idx + 1

    runnable: list[Dict[str, Any]] = []
    skipped = 0
    for row in rows:
        lp = _log_path(row, args.dataset)
        if args.resume_from_logs and _is_completed_log_strict(lp):
            skipped += 1
            continue
        runnable.append(row)

    summary_path = _phase10_summary_csv_path(args.dataset)
    _ensure_summary_csv(summary_path)
    summary_state = _load_summary_state(summary_path)

    if skipped > 0:
        print(f"[phase10] resume_from_logs=on: skipped {skipped} completed runs by strict log-end marker.")

    if not runnable:
        print("[phase10] all runs are already completed by strict log markers.")
        return 0

    if args.dry_run:
        for row in runnable:
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, row["assigned_gpu"], args)
            print(
                f"[dry-run] gpu={row['assigned_gpu']} run={row['run_id']} "
                f"setting={row['setting_key']} seed={row['seed_id']} -> {lp}"
            )
            print("          " + " ".join(cmd))
        return 0

    gpu_bins: Dict[str, deque[Dict[str, Any]]] = {gpu: deque() for gpu in gpus}
    for row in runnable:
        gpu_bins[str(row["assigned_gpu"])].append(row)

    active: Dict[str, Dict[str, Any]] = {}
    while True:
        for gpu_id in gpus:
            if gpu_id in active:
                continue
            if not gpu_bins[gpu_id]:
                continue
            row = gpu_bins[gpu_id].popleft()
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, gpu_id, args)
            _write_log_preamble(lp, row, gpu_id, args, cmd)
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
                f"[launch] gpu={gpu_id} run={row['run_id']} "
                f"setting={row['setting_key']}"
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
                    summary_state=summary_state,
                    row=row,
                    trial_metrics=trial_metrics,
                )
            rc = proc.poll()
            if rc is None:
                continue
            done_gpu.append(gpu_id)
            print(f"[done] gpu={gpu_id} run={row['run_id']} rc={rc} log={lp}")
            if int(rc) == 0:
                _record_run_complete_summary(
                    dataset=args.dataset,
                    summary_path=summary_path,
                    summary_state=summary_state,
                    row=row,
                    verify_logging=bool(args.verify_logging),
                )

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = any(gpu_bins[g] for g in gpus)
        if not pending and not active:
            break
        time.sleep(3)

    print(f"[phase10] summary updated: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Phase10 feature portability launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=68000)

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

    # Fixed from phase9_2 strong region (H3 family).
    parser.add_argument("--embedding-size", type=int, default=160)
    parser.add_argument("--d-ff", type=int, default=320)
    parser.add_argument("--d-expert-hidden", type=int, default=160)
    parser.add_argument("--d-router-hidden", type=int, default=80)
    parser.add_argument("--fixed-weight-decay", type=float, default=2e-6)
    parser.add_argument("--fixed-hidden-dropout-prob", type=float, default=0.18)

    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")

    parser.add_argument("--only-setting", default="", help="Comma-separated subset of setting keys or IDs")
    parser.add_argument("--include-extra-24", dest="include_extra_24", action="store_true")
    parser.add_argument("--no-extra-24", dest="include_extra_24", action="store_false")
    parser.set_defaults(include_extra_24=True)

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
    args.gpus = _parse_csv_strings(args.gpus)[0] if _parse_csv_strings(args.gpus) else "0"
    if not args.only_setting:
        # Keep one baseline + one non-trivial variant when possible.
        args.only_setting = "FULL,FEATURE_DROPOUT"


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided")

    settings = _build_settings(args)
    rows = _build_rows(args, settings)
    if args.smoke_test:
        rows = rows[: max(int(args.smoke_max_runs), 1)]
    if not rows:
        raise SystemExit("No rows matched filters")

    manifest_path = _write_matrix_manifest(rows, settings, args)
    print(
        f"[phase10] dataset={args.dataset} settings={len(settings)} rows={len(rows)} "
        f"axis={AXIS} phase={PHASE} manifest={manifest_path}"
    )
    print(
        f"[phase10] resume_from_logs={args.resume_from_logs} verify_logging={args.verify_logging} "
        f"include_extra_24={args.include_extra_24}"
    )
    return _launch_rows(rows=rows, gpus=gpus, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
