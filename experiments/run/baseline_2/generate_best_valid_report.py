from __future__ import annotations

import json
from pathlib import Path


BASELINE_RESULTS_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/logs/baseline_2")
ROUTEREC_RESULTS_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/logs/fmoe_n4")
OUTPUT_PATH = Path(
    "/workspace/FeaturedMoE/experiments/run/baseline_2/docs/baseline_2_best_valid_test_tables_v4.md"
)

DATASET_ORDER = [
    "beauty",
    "retail_rocket",
    "foursquare",
    "movielens1m",
    "lastfm0.03",
    "KuaiRecLargeStrictPosV2_0.2",
]

BASELINE_MODEL_SPECS = [
    ("SASRec", "SASRec"),
    ("GRU4Rec", "GRU4Rec"),
    ("TiSASRec", "TiSASRec"),
    ("FEARec", "FEARec"),
    ("DuoRec", "DuoRec"),
    ("BSARec", "BSARec"),
    ("FAME", "FAME"),
    ("DIFSR", "DIFSR"),
    ("FDSA", "FDSA"),
]

ROUTEREC_MODEL = "FeaturedMoE_N3"
ROUTEREC_DISPLAY_VALID = "RouteRec(valid)"
ROUTEREC_DISPLAY_TEST = "RouteRec(test)"
ROUTEREC_DISPLAY_MAX = "RouteRec(max)"

METRIC_ORDER = [
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

SECTION_SPECS = [
    ("Seen", "overall_seen_target"),
    ("Overall", "overall"),
    ("Unseen", "overall_unseen_target"),
]


def _load_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metric_block_average(block: dict | None) -> float | None:
    if not isinstance(block, dict):
        return None
    values: list[float] = []
    for metric in METRIC_ORDER:
        value = block.get(metric)
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return _mean(values)


def _test_block_average(payload: dict) -> float | None:
    test_special = payload.get("test_special_metrics")
    if not isinstance(test_special, dict):
        return None

    values: list[float] = []
    for _, section_key in SECTION_SPECS:
        section = test_special.get(section_key)
        if not isinstance(section, dict):
            return None
        for metric in METRIC_ORDER:
            value = section.get(metric)
            if not isinstance(value, (int, float)):
                return None
            values.append(float(value))
    return _mean(values)


def _has_any_displayed_test_metric(payload: dict) -> bool:
    return _test_block_average(payload) is not None


def _metric_value(payload: dict, section_key: str, metric: str) -> float:
    test_special = payload.get("test_special_metrics")
    if not isinstance(test_special, dict):
        return 0.0
    section = test_special.get(section_key)
    if not isinstance(section, dict):
        return 0.0
    value = section.get(metric)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _collect_results(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("**/*_special_metrics.json")):
        payload = _load_json(path)
        if payload is None:
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def _select_baseline_runs(results: list[dict]) -> dict[tuple[str, str], dict]:
    selected: dict[tuple[str, str], dict] = {}
    valid_models = {model for model, _ in BASELINE_MODEL_SPECS}

    for payload in results:
        dataset = payload.get("dataset")
        model = payload.get("model")
        if dataset not in DATASET_ORDER or model not in valid_models:
            continue

        seen_valid_block = (payload.get("best_valid_special_metrics") or {}).get("overall_seen_target")
        valid_avg = _metric_block_average(seen_valid_block)
        if valid_avg is None:
            continue

        key = (dataset, model)
        tie_breaker = (
            valid_avg,
            float((seen_valid_block or {}).get("mrr@20", -1.0)),
            str(payload.get("timestamp", "")),
            str(payload.get("_path", "")),
        )
        current = selected.get(key)
        if current is None or tie_breaker > current["tie_breaker"]:
            selected[key] = {
                "payload": payload,
                "tie_breaker": tie_breaker,
            }

    missing = [
        (dataset, model)
        for dataset in DATASET_ORDER
        for model, _ in BASELINE_MODEL_SPECS
        if (dataset, model) not in selected
    ]
    if missing:
        missing_text = ", ".join(f"{dataset}/{model}" for dataset, model in missing)
        raise RuntimeError(f"Missing baseline selections for: {missing_text}")

    return selected


def _collect_routerec_runs(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {dataset: [] for dataset in DATASET_ORDER}
    for payload in results:
        dataset = payload.get("dataset")
        model = payload.get("model")
        if dataset not in DATASET_ORDER or model != ROUTEREC_MODEL:
            continue
        grouped[dataset].append(payload)
    return grouped


def _select_routerec_valid_runs(grouped: dict[str, list[dict]]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for dataset, payloads in grouped.items():
        best_payload: dict | None = None
        best_tie_breaker: tuple | None = None
        for payload in payloads:
            if not _has_any_displayed_test_metric(payload):
                continue
            seen_valid_block = (payload.get("best_valid_special_metrics") or {}).get("overall_seen_target")
            valid_avg = _metric_block_average(seen_valid_block)
            if valid_avg is None:
                continue
            tie_breaker = (
                valid_avg,
                float((seen_valid_block or {}).get("mrr@20", -1.0)),
                str(payload.get("timestamp", "")),
                str(payload.get("_path", "")),
            )
            if best_tie_breaker is None or tie_breaker > best_tie_breaker:
                best_tie_breaker = tie_breaker
                best_payload = payload
        if best_payload is None:
            raise RuntimeError(f"Missing RouteRec valid-avg candidate for dataset: {dataset}")
        selected[dataset] = best_payload
    return selected


def _select_routerec_test_runs(grouped: dict[str, list[dict]]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for dataset, payloads in grouped.items():
        best_payload: dict | None = None
        best_tie_breaker: tuple | None = None
        for payload in payloads:
            test_avg = _test_block_average(payload)
            if test_avg is None:
                continue
            tie_breaker = (
                test_avg,
                _metric_value(payload, "overall_seen_target", "mrr@20"),
                _metric_value(payload, "overall", "mrr@20"),
                str(payload.get("timestamp", "")),
                str(payload.get("_path", "")),
            )
            if best_tie_breaker is None or tie_breaker > best_tie_breaker:
                best_tie_breaker = tie_breaker
                best_payload = payload
        if best_payload is None:
            raise RuntimeError(f"Missing RouteRec test-avg candidate for dataset: {dataset}")
        selected[dataset] = best_payload
    return selected


def _collect_routerec_metric_max(grouped: dict[str, list[dict]]) -> dict[str, dict[tuple[str, str], float]]:
    maxima: dict[str, dict[tuple[str, str], float]] = {}
    for dataset, payloads in grouped.items():
        dataset_max: dict[tuple[str, str], float] = {}
        for _, section_key in SECTION_SPECS:
            for metric in METRIC_ORDER:
                dataset_max[(section_key, metric)] = 0.0

        for payload in payloads:
            for _, section_key in SECTION_SPECS:
                for metric in METRIC_ORDER:
                    value = _metric_value(payload, section_key, metric)
                    key = (section_key, metric)
                    if value > dataset_max[key]:
                        dataset_max[key] = value
        maxima[dataset] = dataset_max
    return maxima


def _format_ranked_value(value: float, best: float, second: float | None, worst: float) -> str:
    text = f"{value:.4f}"
    if best <= 0:
        return text

    is_best = abs(value - best) < 1e-12
    is_second = second is not None and abs(value - second) < 1e-12 and not is_best
    is_underlined = abs(value - worst) < 1e-12 or value <= 0.75 * best + 1e-12

    if is_underlined:
        text = f"<u>{text}</u>"
    if is_best:
        text = f"**{text}**"
    elif is_second:
        text = f"*{text}"
    return text


def _render_table(
    dataset: str,
    section_key: str,
    baseline_selected: dict[tuple[str, str], dict],
    routerec_valid_selected: dict[str, dict],
    routerec_test_selected: dict[str, dict],
    routerec_maxima: dict[str, dict[tuple[str, str], float]],
) -> str:
    headers = (
        ["metric"]
        + [display for _, display in BASELINE_MODEL_SPECS]
        + [ROUTEREC_DISPLAY_VALID, ROUTEREC_DISPLAY_TEST, ROUTEREC_DISPLAY_MAX]
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|" + "|".join("---:" for _ in headers[1:]) + "|",
    ]

    ranked_models = [model for model, _ in BASELINE_MODEL_SPECS]

    for metric in METRIC_ORDER:
        ranked_values = [
            _metric_value(baseline_selected[(dataset, model)]["payload"], section_key, metric)
            for model in ranked_models
        ]
        route_valid_value = _metric_value(routerec_valid_selected[dataset], section_key, metric)
        ranked_values.append(route_valid_value)

        unique_desc = sorted(set(ranked_values), reverse=True)
        best = unique_desc[0] if unique_desc else 0.0
        second = unique_desc[1] if len(unique_desc) > 1 else None
        worst = min(ranked_values) if ranked_values else 0.0

        row = [metric]
        for value in ranked_values:
            row.append(_format_ranked_value(value, best, second, worst))

        route_test_value = _metric_value(routerec_test_selected[dataset], section_key, metric)
        route_max_value = routerec_maxima[dataset][(section_key, metric)]
        row.append(f"{route_test_value:.4f}")
        row.append(f"{route_max_value:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def build_report() -> str:
    baseline_results = _collect_results(BASELINE_RESULTS_ROOT)
    routerec_results = _collect_results(ROUTEREC_RESULTS_ROOT)

    baseline_selected = _select_baseline_runs(baseline_results)
    routerec_grouped = _collect_routerec_runs(routerec_results)
    routerec_valid_selected = _select_routerec_valid_runs(routerec_grouped)
    routerec_test_selected = _select_routerec_test_runs(routerec_grouped)
    routerec_maxima = _collect_routerec_metric_max(routerec_grouped)

    lines: list[str] = []
    lines.append("# baseline_2 best-valid test tables (RouteRec 3-way view)")
    lines.append("")
    lines.append(
        "Selection rule: baseline columns use the best run by arithmetic mean of the 9 metrics in `best_valid_special_metrics.overall_seen_target` from `experiments/run/artifacts/logs/baseline_2`."
    )
    lines.append(
        "RouteRec columns use only `experiments/run/artifacts/logs/fmoe_n4`: `RouteRec(valid)` = highest arithmetic mean of the 9 metrics in `best_valid_special_metrics.overall_seen_target`, `RouteRec(test)` = highest arithmetic mean across the 27 displayed test metrics (`overall`, `overall_seen_target`, `overall_unseen_target`), `RouteRec(max)` = per-metric maximum across all RouteRec runs regardless of run identity."
    )
    lines.append(
        "Formatting rule: ranking, bold, `*`, and underline are computed only over baseline columns plus `RouteRec(valid)`. The right two RouteRec columns are reference-only and are shown without ranking markup."
    )
    lines.append("")

    for dataset in DATASET_ORDER:
        lines.append(f"## {dataset}")
        lines.append("")
        for section_title, section_key in SECTION_SPECS:
            if section_title == "Seen":
                lines.append("### Seen")
                lines.append("")
                lines.append(
                    _render_table(
                        dataset,
                        section_key,
                        baseline_selected,
                        routerec_valid_selected,
                        routerec_test_selected,
                        routerec_maxima,
                    )
                )
                lines.append("")
                continue

            lines.append("<details>")
            lines.append(f"<summary>{section_title}</summary>")
            lines.append("")
            lines.append(
                _render_table(
                    dataset,
                    section_key,
                    baseline_selected,
                    routerec_valid_selected,
                    routerec_test_selected,
                    routerec_maxima,
                )
            )
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
