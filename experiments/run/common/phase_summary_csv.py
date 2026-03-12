#!/usr/bin/env python3
"""Build compact phase-local CSV summaries from live hyperopt logs."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


FLOAT_PATTERN = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_KEYS = ("best_mrr@20", "best_hr@10", "test_mrr@20", "test_hr@10")
METRIC_COLUMNS = {
    "best_mrr@20": "best_mrr20",
    "best_hr@10": "best_hr10",
    "test_mrr@20": "test_mrr20",
    "test_hr@10": "test_hr10",
}
FIELDNAMES = [
    "model",
    "best_mrr20_group",
    "best_mrr20_run",
    "best_mrr20_trial",
    "best_hr10_group",
    "best_hr10_run",
    "best_hr10_trial",
    "test_mrr20_group",
    "test_mrr20_run",
    "test_mrr20_trial",
    "test_hr10_group",
    "test_hr10_run",
    "test_hr10_trial",
    "hit_best_mrr20",
    "hit_best_hr10",
    "hit_test_mrr20",
    "hit_test_hr10",
    "trial_idx",
    "trials_progress",
    "status",
    "run_phase",
    "group_key",
    "experiment_note",
    "trial_knobs",
    "log_rel_path",
]
RUN_PHASE_RE = re.compile(r"phase=([A-Za-z0-9._-]+)")
RUN_STATUS_RE = re.compile(r"^\[RUN_STATUS\]\s+END\s+status=([^\s]+)", re.MULTILINE)
TRIAL_HEADER_RE = re.compile(r"^\[(?P<trial>\d+)/(?P<total>\d+)\]\s+(?P<params>.+)$")
TRIAL_METRICS_LINE_RE = re.compile(
    r"^\[TRIAL_METRICS\]\s+"
    r"cur_best_mrr20=(?P<cur_best_mrr20>{f})\s+"
    r"cur_best_hr10=(?P<cur_best_hr10>{f})\s+"
    r"cur_test_mrr20=(?P<cur_test_mrr20>{f})\s+"
    r"cur_test_hr10=(?P<cur_test_hr10>{f})".format(f=FLOAT_PATTERN)
)
LOG_NAME_RE = re.compile(r"^(?P<model>[^_]+)_(?P<idx>\d{3})_(?P<desc>.+)\.log$")
LEGACY_LOG_NAME_RE = re.compile(r"^(?P<idx>\d{3})_(?P<model>[^_]+)_(?P<desc>.+)\.log$")
DATASET_RE = re.compile(r"Hyperopt TPE\s+\|\s+.+?\s+x\s+(?P<dataset>[^\n]+)")
FIXED_ENTRY_RE = re.compile(r"^\s{4}(?P<key>\S.*?)\s{2,}(?P<value>.+?)\s*$")
COMBO_RE = re.compile(r"(?:^|_)(?P<combo>[A-Z]\d{2})(?:$|_)")
BASELINE_PHASE_RE = re.compile(r"_(?P<combo>C\d+)$")
DATASET_TAG_MAP = {
    "KU01": "KuaiRecSmall0.1",
    "LF03": "lastfm0.03",
}
BASELINE_KNOB_ALIASES = {
    "learning_rate": "lr",
    "weight_decay": "wd",
    "dropout_ratio": "drop",
}
FMOE_N_KNOB_ALIASES = {
    "learning_rate": "lr",
    "weight_decay": "wd",
    "hidden_dropout_prob": "drop",
    "balance_loss_lambda": "bal",
    "moe_top_k": "topk",
}


@dataclass
class TrialSnapshot:
    trial_idx: int
    total_trials: int
    metrics: dict[str, float | None]
    knob_assignments: list[tuple[str, str]]


@dataclass
class RunSummary:
    dataset: str
    model: str
    group_key: str
    experiment_note: str
    log_rel_path: str
    run_phase: str
    status: str
    trials: list[TrialSnapshot]
    run_best: dict[str, float | None]
    sort_group: tuple[object, ...]
    sort_run: tuple[object, ...]


@dataclass
class SummaryRow:
    run: RunSummary
    trial: TrialSnapshot
    hit_flags: dict[str, bool] = field(default_factory=dict)
    run_best: dict[str, float | None] = field(default_factory=dict)
    group_best: dict[str, float | None] = field(default_factory=dict)


def _experiments_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _logs_root() -> Path:
    raw = os.environ.get("RUN_LOGS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _experiments_root() / "run" / "artifacts" / "logs"


def _to_float(raw: str | float | int | None) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _fmt_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_float_token(value: float) -> str:
    return f"{value:.3g}"


def _format_knob_value(key: str, raw: str) -> str:
    value = _to_float(raw)
    if value is None:
        return str(raw).strip()
    if key in {"hidden_dropout_prob", "dropout_ratio"}:
        return f"{value:.2f}"
    if float(value).is_integer():
        return str(int(value))
    return _format_float_token(value)


def _parse_status(text: str) -> str:
    matches = RUN_STATUS_RE.findall(text)
    if matches:
        status = str(matches[-1]).strip()
        return "success" if status == "normal" else status
    if "[RUN_STATUS] TERMINATED" in text:
        return "terminated"
    if "Traceback (most recent call last):" in text or "ConfigCompositionException" in text:
        return "fail"
    return "running"


def _parse_run_phase(text: str, fallback: str) -> str:
    matches = RUN_PHASE_RE.findall(text)
    if matches:
        return str(matches[-1]).strip()
    return fallback


def _parse_assignments(raw: str) -> list[tuple[str, str]]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in str(raw):
        if ch in "{[(":
            depth += 1
        elif ch in "}])" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        items.append("".join(current).strip())

    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed.append((key, value))
    return parsed


def _parse_trials(text: str) -> list[TrialSnapshot]:
    trials: list[TrialSnapshot] = []
    pending_header: tuple[int, int, list[tuple[str, str]]] | None = None
    for line in text.splitlines():
        header_match = TRIAL_HEADER_RE.match(line)
        if header_match:
            pending_header = (
                int(header_match.group("trial")),
                int(header_match.group("total")),
                _parse_assignments(header_match.group("params")),
            )
            continue

        metric_match = TRIAL_METRICS_LINE_RE.match(line)
        if not metric_match:
            continue

        if pending_header is None:
            fallback_idx = len(trials) + 1
            pending_header = (fallback_idx, fallback_idx, [])

        trial_idx, total_trials, knob_assignments = pending_header
        trials.append(
            TrialSnapshot(
                trial_idx=trial_idx,
                total_trials=total_trials,
                knob_assignments=knob_assignments,
                metrics={
                    "best_mrr@20": _to_float(metric_match.group("cur_best_mrr20")),
                    "best_hr@10": _to_float(metric_match.group("cur_best_hr10")),
                    "test_mrr@20": _to_float(metric_match.group("cur_test_mrr20")),
                    "test_hr@10": _to_float(metric_match.group("cur_test_hr10")),
                },
            )
        )
        pending_header = None

    return trials


def _compute_run_best(trials: Iterable[TrialSnapshot]) -> dict[str, float | None]:
    best = {key: None for key in METRIC_KEYS}
    for trial in trials:
        for key in METRIC_KEYS:
            value = trial.metrics.get(key)
            if value is None:
                continue
            if best[key] is None or value > best[key]:
                best[key] = value
    return best


def _build_milestone_rows(runs: list[RunSummary]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []

    for run in runs:
        seen_best = {key: None for key in METRIC_KEYS}
        for trial in run.trials:
            hit_flags: dict[str, bool] = {}
            for key in METRIC_KEYS:
                current = trial.metrics.get(key)
                previous = seen_best[key]
                hit_flags[key] = current is not None and (previous is None or current > previous)
            for key in METRIC_KEYS:
                if hit_flags[key]:
                    seen_best[key] = trial.metrics.get(key)
            if any(hit_flags.values()):
                rows.append(
                    SummaryRow(
                        run=run,
                        trial=trial,
                        hit_flags=hit_flags,
                        run_best=dict(seen_best),
                        group_best={key: None for key in METRIC_KEYS},
                    )
                )

    rows.sort(
        key=lambda row: (
            row.run.sort_group,
            row.run.sort_run,
            row.trial.trial_idx,
        ),
    )

    group_best: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in rows:
        scope_key = (row.run.dataset, row.run.model)
        best_for_group = group_best.setdefault(scope_key, {key: None for key in METRIC_KEYS})
        for key in METRIC_KEYS:
            run_value = row.run_best.get(key)
            if run_value is None:
                continue
            if best_for_group[key] is None or run_value > best_for_group[key]:
                best_for_group[key] = run_value
        row.group_best = dict(best_for_group)
    return rows


def _format_trial_knobs(assignments: list[tuple[str, str]], aliases: dict[str, str]) -> str:
    parts: list[str] = []
    for key, value in assignments:
        label = aliases.get(key, key)
        parts.append(f"{label}={_format_knob_value(key, value)}")
    return " ".join(parts)


def _write_csv(out_path: Path, rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")

    def write_inner() -> None:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    if fcntl is None:
        write_inner()
        return

    with lock_path.open("w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        write_inner()
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _serialize_rows(rows: list[SummaryRow], aliases: dict[str, str]) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for row in rows:
        item = {
            "model": row.run.model,
            "hit_best_mrr20": "Y" if row.hit_flags.get("best_mrr@20") else "",
            "hit_best_hr10": "Y" if row.hit_flags.get("best_hr@10") else "",
            "hit_test_mrr20": "Y" if row.hit_flags.get("test_mrr@20") else "",
            "hit_test_hr10": "Y" if row.hit_flags.get("test_hr@10") else "",
            "trial_idx": str(row.trial.trial_idx),
            "trials_progress": f"{row.trial.trial_idx}/{row.trial.total_trials}",
            "status": row.run.status,
            "run_phase": row.run.run_phase,
            "group_key": row.run.model,
            "experiment_note": row.run.experiment_note,
            "trial_knobs": _format_trial_knobs(row.trial.knob_assignments, aliases),
            "log_rel_path": row.run.log_rel_path,
        }
        for metric_key, column_prefix in METRIC_COLUMNS.items():
            item[f"{column_prefix}_group"] = _fmt_metric(row.group_best.get(metric_key))
            item[f"{column_prefix}_run"] = _fmt_metric(row.run_best.get(metric_key))
            item[f"{column_prefix}_trial"] = _fmt_metric(row.trial.metrics.get(metric_key))
        serialized.append(item)
    return serialized


def _baseline_note(log_name: str, run_phase: str) -> str:
    match = LOG_NAME_RE.match(log_name) or LEGACY_LOG_NAME_RE.match(log_name)
    desc = str(match.group("desc") if match else Path(log_name).stem)
    parts = [part for part in desc.split("_") if part]
    combo_id = ""
    suffix_parts: list[str] = []
    for idx, part in enumerate(parts):
        if re.fullmatch(r"c\d+", part, re.IGNORECASE):
            combo_id = part.upper()
            suffix_parts = parts[idx + 1 :]
            break
    if not combo_id:
        phase_match = BASELINE_PHASE_RE.search(run_phase)
        if phase_match:
            combo_id = phase_match.group("combo").upper()
    if combo_id and suffix_parts:
        return f"{combo_id} {'_'.join(suffix_parts)}"
    if combo_id:
        return combo_id
    return desc


def _parse_fixed_params(text: str) -> dict[str, str]:
    fixed: dict[str, str] = {}
    capture = False
    for line in text.splitlines():
        if line.startswith("Fixed params(singleton/non-list):"):
            capture = True
            continue
        if not capture:
            continue
        if (
            TRIAL_HEADER_RE.match(line)
            or line.startswith("[Wandb]")
            or line.startswith("[RUN_STATUS]")
            or line.startswith("Changed knobs:")
            or line.startswith("Layout details:")
        ):
            break
        if not line.strip():
            continue
        if line.lstrip().startswith("[") and line.rstrip().endswith("]"):
            continue
        match = FIXED_ENTRY_RE.match(line)
        if match:
            fixed[match.group("key").strip()] = match.group("value").strip()
    return fixed


def _parse_fmoe_n_dataset(text: str, path: Path) -> str:
    match = DATASET_RE.search(text)
    if match:
        return str(match.group("dataset")).strip()
    for part in path.parts:
        if part in DATASET_TAG_MAP:
            return DATASET_TAG_MAP[part]
    return "unknown"


def _parse_combo_id(run_phase: str, log_name: str) -> str:
    for source in (run_phase, Path(log_name).stem):
        match = COMBO_RE.search(str(source or ""))
        if match:
            return str(match.group("combo"))
    return "-"


def _family_from_fixed(fixed: dict[str, str]) -> str:
    router_impl = str(fixed.get("router_impl_by_stage", "")).strip()
    rule_bias = _to_float(fixed.get("rule_bias_scale"))
    if "rule_soft" in router_impl:
        return "hybrid"
    if rule_bias is not None and rule_bias > 0:
        return "bias"
    return "plain"


def _layout_from_fixed(fixed: dict[str, str]) -> str:
    raw = str(fixed.get("fmoe_v2_layout_id") or fixed.get("arch_layout_id") or "").strip()
    if not raw:
        return ""
    return raw if raw.upper().startswith("L") else f"L{raw}"


def _baseline_sort_idx(log_name: str, default_idx: int) -> int:
    match = LOG_NAME_RE.match(log_name) or LEGACY_LOG_NAME_RE.match(log_name)
    if match:
        return int(match.group("idx"))
    return default_idx


def _combo_sort_key(combo_id: str, fallback: int) -> tuple[int, str]:
    match = re.search(r"(\d+)", combo_id)
    if match:
        return (int(match.group(1)), combo_id)
    return (fallback, combo_id)


def _fmoe_n_note(fixed: dict[str, str], combo_id: str) -> str:
    state_tag = str(fixed.get("arch_state_tag", "")).strip()
    family = _family_from_fixed(fixed)
    layout = _layout_from_fixed(fixed) or combo_id
    tokens = []
    if state_tag:
        tokens.append(state_tag)
    tokens.extend([family, layout])

    topk = _to_float(fixed.get("moe_top_k"))
    if topk is not None and int(topk) != 0:
        tokens.append(f"topk{int(topk)}")

    expert_scale = _to_float(fixed.get("expert_scale"))
    if expert_scale is not None and int(expert_scale) != 3:
        tokens.append(f"scale{int(expert_scale)}")

    d_feat = _to_float(fixed.get("d_feat_emb"))
    if d_feat is not None and int(d_feat) != 16:
        tokens.append(f"dfeat{int(d_feat)}")

    feature_mode = str(fixed.get("feature_encoder_mode", "")).strip()
    if feature_mode and feature_mode != "linear":
        tokens.append(f"feat:{feature_mode}")

    inter_style = str(fixed.get("stage_inter_layer_style", "")).strip()
    if inter_style and inter_style != "attn":
        tokens.append(f"inter:{inter_style}")

    balance = _to_float(fixed.get("balance_loss_lambda"))
    if balance is not None and abs(balance - 0.002) > 1e-12:
        if abs(balance) < 1e-12:
            tokens.append("bal0")
        else:
            tokens.append(f"bal{_format_float_token(balance)}")

    return " ".join(tokens)


def build_baseline_summary(dataset: str, phase: str) -> Path:
    dataset = str(dataset).strip()
    phase = str(phase).strip()
    logs_dir = _logs_root() / "baseline" / dataset / phase
    out_path = _logs_root() / "baseline" / dataset / f"{phase}_summary.csv"

    runs: list[RunSummary] = []
    if logs_dir.exists():
        for default_idx, log_path in enumerate(sorted(logs_dir.glob("*.log"), key=lambda path: path.name), start=100000):
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            trials = _parse_trials(text)
            run_phase = _parse_run_phase(text, phase)
            log_name = log_path.name
            match = LOG_NAME_RE.match(log_name) or LEGACY_LOG_NAME_RE.match(log_name)
            model = str(match.group("model") if match else "unknown")
            runs.append(
                RunSummary(
                    dataset=dataset,
                    model=model,
                    group_key=model,
                    experiment_note=_baseline_note(log_name, run_phase),
                    log_rel_path=str(log_path.relative_to(_logs_root() / "baseline" / dataset)),
                    run_phase=run_phase,
                    status=_parse_status(text),
                    trials=trials,
                    run_best=_compute_run_best(trials),
                    sort_group=(model,),
                    sort_run=(_baseline_sort_idx(log_name, default_idx), log_name),
                )
            )

    rows = _build_milestone_rows(runs)
    _write_csv(out_path, _serialize_rows(rows, BASELINE_KNOB_ALIASES))
    return out_path


def build_fmoe_n_summaries(axis: str, phase: str) -> list[Path]:
    axis = str(axis).strip() or "hparam"
    phase = str(phase).strip()
    phase_dir = _logs_root() / "fmoe_n" / axis / phase

    runs_by_dataset: dict[str, list[RunSummary]] = {}
    if phase_dir.exists():
        log_paths = sorted(phase_dir.rglob("*.log"), key=lambda path: str(path.relative_to(phase_dir)))
        for default_idx, log_path in enumerate(log_paths, start=100000):
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""

            dataset = _parse_fmoe_n_dataset(text, log_path)
            run_phase = _parse_run_phase(text, phase)
            combo_id = _parse_combo_id(run_phase, log_path.name)
            fixed = _parse_fixed_params(text)
            trials = _parse_trials(text)
            runs_by_dataset.setdefault(dataset, []).append(
                RunSummary(
                    dataset=dataset,
                    model="FeaturedMoE_N",
                    group_key="FeaturedMoE_N",
                    experiment_note=_fmoe_n_note(fixed, combo_id),
                    log_rel_path=str(log_path.relative_to(phase_dir)),
                    run_phase=run_phase,
                    status=_parse_status(text),
                    trials=trials,
                    run_best=_compute_run_best(trials),
                    sort_group=("FeaturedMoE_N",),
                    sort_run=(_combo_sort_key(combo_id, default_idx), str(log_path.relative_to(phase_dir))),
                )
            )

    written: list[Path] = []
    for dataset, runs in runs_by_dataset.items():
        out_path = phase_dir / dataset / f"{phase}_summary.csv"
        rows = _build_milestone_rows(runs)
        _write_csv(out_path, _serialize_rows(rows, FMOE_N_KNOB_ALIASES))
        written.append(out_path)
    return sorted(written)
