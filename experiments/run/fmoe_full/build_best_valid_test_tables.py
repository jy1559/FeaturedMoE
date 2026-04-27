from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/workspace/FeaturedMoE")
RESULTS_ROOT = REPO_ROOT / "experiments/run/artifacts/results/fmoe_full"
OUTPUT_PATH = REPO_ROOT / "experiments/run/fmoe_full/docs/fmoe_full_best_valid_test_tables_v1.md"

METRICS = [
    "hit@5",
    "hit@10",
    "hit@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "mrr@5",
    "mrr@10",
    "mrr@20",
]

SECTIONS = [
    ("Seen", "overall_seen_target"),
    ("Overall", "overall"),
    ("Unseen", "overall_unseen_target"),
]

STRATEGIES = [
    ("valid_best_mrr20", "valid best mrr@20"),
    ("test_best_mrr20", "test best mrr@20"),
    ("valid_metric_avg", "valid metric avg"),
    ("test_metric_avg", "test metric avg"),
    ("valid_test_metric_avg", "valid+test avg"),
]

COMPARISON_COLUMN = ("metric_wise_best", "metric-wise best")

PREFERRED_DATASET_ORDER = [
    "beauty",
    "retail_rocket",
    "foursquare",
    "movielens1m",
    "lastfm0.03",
    "KuaiRecLargeStrictPosV2_0.2",
]


@dataclass
class Candidate:
    dataset: str
    run_phase: str
    result_path: Path
    timestamp: str
    complete: bool
    payload: dict[str, Any]
    special_payload: dict[str, Any] | None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_top_level_result_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.json") if path.is_file())


def dedupe_candidates(paths: list[Path]) -> list[Candidate]:
    by_phase: dict[str, Candidate] = {}

    for path in paths:
        payload = load_json(path)
        run_phase = payload.get("run_phase") or path.stem
        timestamp = str(payload.get("timestamp") or "")
        max_evals = int(payload.get("max_evals") or 0)
        n_completed = int(payload.get("n_completed") or 0)
        complete = max_evals > 0 and n_completed >= max_evals
        special_payload = load_special_payload(payload)
        candidate = Candidate(
            dataset=str(payload.get("dataset") or payload.get("dataset_raw") or "unknown"),
            run_phase=run_phase,
            result_path=path,
            timestamp=timestamp,
            complete=complete,
            payload=payload,
            special_payload=special_payload,
        )

        current = by_phase.get(run_phase)
        if current is None or candidate_priority(candidate) > candidate_priority(current):
            by_phase[run_phase] = candidate

    return list(by_phase.values())


def candidate_priority(candidate: Candidate) -> tuple[int, int, str, str]:
    payload = candidate.payload
    return (
        int(candidate.complete),
        int(payload.get("n_completed") or 0),
        candidate.timestamp,
        candidate.result_path.name,
    )


def load_special_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    root_has_specials = any(payload.get(key) for key in (
        "best_valid_special_metrics",
        "test_special_metrics",
        "early_valid_special_metrics",
    ))
    if root_has_specials:
        return payload

    special_result_file = payload.get("special_result_file")
    if not special_result_file:
        return payload

    special_path = Path(str(special_result_file))
    if not special_path.is_file():
        return payload

    return load_json(special_path)


def normalize_metric_dict(values: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(values, dict):
        return None
    normalized: dict[str, float] = {}
    for metric in METRICS:
        if metric not in values:
            return None
        normalized[metric] = float(values[metric])
    return normalized


def get_valid_metrics(candidate: Candidate, section_key: str) -> dict[str, float] | None:
    if section_key == "overall":
        return normalize_metric_dict(candidate.payload.get("best_valid_result"))

    special_payload = candidate.special_payload or {}
    best_valid_specials = special_payload.get("best_valid_special_metrics")
    if not isinstance(best_valid_specials, dict):
        return None
    return normalize_metric_dict(best_valid_specials.get(section_key))


def get_test_metrics(candidate: Candidate, section_key: str) -> dict[str, float] | None:
    if section_key == "overall":
        return normalize_metric_dict(candidate.payload.get("test_result"))

    special_payload = candidate.special_payload or {}
    test_specials = special_payload.get("test_special_metrics")
    if not isinstance(test_specials, dict):
        return None
    return normalize_metric_dict(test_specials.get(section_key))


def mean_metric(values: dict[str, float]) -> float:
    return sum(values[metric] for metric in METRICS) / len(METRICS)


def select_candidate(candidates: list[Candidate], section_key: str, strategy_key: str) -> Candidate | None:
    best_candidate: Candidate | None = None
    best_score: tuple[float, float, str] | None = None

    for candidate in candidates:
        if not candidate.complete:
            continue

        valid_metrics = get_valid_metrics(candidate, section_key)
        test_metrics = get_test_metrics(candidate, section_key)
        if valid_metrics is None or test_metrics is None:
            continue

        if strategy_key == "valid_best_mrr20":
            score = valid_metrics["mrr@20"]
        elif strategy_key == "test_best_mrr20":
            score = test_metrics["mrr@20"]
        elif strategy_key == "valid_metric_avg":
            score = mean_metric(valid_metrics)
        elif strategy_key == "test_metric_avg":
            score = mean_metric(test_metrics)
        elif strategy_key == "valid_test_metric_avg":
            score = (mean_metric(valid_metrics) + mean_metric(test_metrics)) / 2.0
        else:
            raise ValueError(f"Unknown strategy: {strategy_key}")

        tie_break = test_metrics["mrr@20"]
        candidate_score = (score, tie_break, candidate.timestamp)
        if best_score is None or candidate_score > best_score:
            best_candidate = candidate
            best_score = candidate_score

    return best_candidate


def decorate_metric(value: float | None, peer_values: list[float]) -> str:
    if value is None:
        return "--"

    text = f"{value:.4f}"
    if not peer_values:
        return text

    best = max(peer_values)
    if best <= 0:
        return text

    distinct_desc = sorted(set(peer_values), reverse=True)
    second = distinct_desc[1] if len(distinct_desc) > 1 else None
    minimum = min(peer_values)

    if value != best and (value == minimum or value <= best * 0.75):
        text = f"<u>{text}</u>"
    if second is not None and value == second and value != best:
        text = f"*{text}"
    if value == best:
        text = f"**{text}**"
    return text


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    if dataset in PREFERRED_DATASET_ORDER:
        return (PREFERRED_DATASET_ORDER.index(dataset), dataset)
    return (len(PREFERRED_DATASET_ORDER), dataset)


def build_table_rows(candidates: list[Candidate], section_key: str) -> list[str]:
    selected_by_strategy: dict[str, dict[str, float] | None] = {}
    for strategy_key, _ in STRATEGIES:
        selected = select_candidate(candidates, section_key, strategy_key)
        selected_by_strategy[strategy_key] = None if selected is None else get_test_metrics(selected, section_key)

    metric_wise_best: dict[str, float | None] = {}
    for metric in METRICS:
        values = []
        for candidate in candidates:
            if not candidate.complete:
                continue
            test_metrics = get_test_metrics(candidate, section_key)
            if test_metrics is None:
                continue
            values.append(test_metrics[metric])
        metric_wise_best[metric] = max(values) if values else None

    lines = [
        "| metric | " + " | ".join(label for _, label in STRATEGIES) + f" | {COMPARISON_COLUMN[1]} |",
        "|---|" + "|".join("---:" for _ in STRATEGIES) + "|---:|",
    ]

    for metric in METRICS:
        numeric_values = [
            metrics[metric]
            for metrics in selected_by_strategy.values()
            if metrics is not None
        ]
        rendered = []
        for strategy_key, _ in STRATEGIES:
            metrics = selected_by_strategy[strategy_key]
            value = None if metrics is None else metrics[metric]
            rendered.append(decorate_metric(value, numeric_values))
        metric_best = metric_wise_best[metric]
        metric_best_text = "--" if metric_best is None else f"{metric_best:.4f}"
        lines.append(f"| {metric} | " + " | ".join(rendered) + f" | {metric_best_text} |")

    return lines


def build_markdown(candidates: list[Candidate]) -> str:
    by_dataset: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_dataset[candidate.dataset].append(candidate)

    datasets = sorted(by_dataset, key=dataset_sort_key)

    lines = [
        "# fmoe_full best-valid/test tables",
        "",
        "Selection rule:",
        "- scan `artifacts/results/fmoe_full/*.json` only, excluding mirrored subdirectories under `normal/`, `special/`, and `diag/`.",
        "- deduplicate reruns by `run_phase`, keeping the most complete artifact and then the latest timestamp.",
        "- use completed runs only (`n_completed >= max_evals`). Incomplete runs are treated as unavailable and rendered as `--`.",
        "- for each section (`Seen`, `Overall`, `Unseen`), pick the run by the requested criterion and report that run's test metrics for the same section.",
        "- if a run has no usable seen/unseen split metrics in its result JSON or linked `special_result_file`, the corresponding cells stay `--`.",
        "",
        "Formatting rule: 1st place is bold, 2nd place has `*`, and the last place or any value at or below 75% of the best value is underlined.",
        "",
    ]

    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        dataset_candidates = by_dataset[dataset]
        for section_label, section_key in SECTIONS:
            table_lines = build_table_rows(dataset_candidates, section_key)
            if section_label == "Seen":
                lines.append(f"### {section_label}")
                lines.append("")
                lines.extend(table_lines)
                lines.append("")
                continue

            lines.append("<details>")
            lines.append(f"<summary>{section_label}</summary>")
            lines.append("")
            lines.extend(table_lines)
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    result_files = iter_top_level_result_files(RESULTS_ROOT)
    candidates = dedupe_candidates(result_files)
    markdown = build_markdown(candidates)
    OUTPUT_PATH.write_text(markdown)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()