from __future__ import annotations

import json
from pathlib import Path


BASELINE_RESULTS_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/results/baseline_2")
FMOE_N4_RESULTS_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4")
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

MODEL_SPECS = [
    ("SASRec", "SASRec"),
    ("GRU4Rec", "GRU4Rec"),
    ("TiSASRec", "TiSASRec"),
    ("FEARec", "FEARec"),
    ("DuoRec", "DuoRec"),
    ("BSARec", "BSARec"),
    ("FAME", "FAME"),
    ("DIFSR", "DIFSR"),
    ("FDSA", "FDSA"),
    ("FeaturedMoE_N3", "RouteRec"),
]

MODEL_ORDER = [model for model, _ in MODEL_SPECS]
MODEL_DISPLAY = {model: display for model, display in MODEL_SPECS}
ROUTEREC_MODEL = "FeaturedMoE_N3"

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

SECTION_MAP = [
    ("Seen", "overall_seen_target"),
    ("Overall", "overall"),
    ("Unseen", "overall_unseen_target"),
]


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_special_metrics_payload(payload: dict) -> dict:
    test_special_metrics = payload.get("test_special_metrics") or {}
    if test_special_metrics:
        return test_special_metrics

    special_result_file = payload.get("special_result_file")
    if not special_result_file:
        return {}

    special_payload = load_json(Path(special_result_file))
    if not special_payload:
        return {}

    return special_payload.get("test_special_metrics") or {}


def format_value(value: float, values: list[float]) -> str:
    eps = 1e-12
    max_value = max(values)
    min_value = min(values)
    all_equal = all(abs(candidate - values[0]) < eps for candidate in values)

    second_value = None
    lower_values = sorted(
        {round(candidate, 12) for candidate in values if candidate < max_value - eps},
        reverse=True,
    )
    if lower_values:
        second_value = lower_values[0]

    text = f"{value:.4f}"
    is_top = abs(value - max_value) < eps and not all_equal
    is_second = second_value is not None and abs(value - second_value) < 1e-9

    is_low = False
    if not all_equal:
        if abs(value - min_value) < eps:
            is_low = True
        elif max_value > eps and value <= 0.75 * max_value + eps:
            is_low = True

    if is_low:
        text = f"<u>{text}</u>"
    if is_top:
        text = f"**{text}**"
    elif is_second:
        text = f"*{text}"

    return text


def build_candidate_meta(payload: dict, source: str, axis: str | None) -> dict | None:
    dataset = payload.get("dataset")
    model = payload.get("model")
    if dataset not in DATASET_ORDER or model not in MODEL_ORDER:
        return None

    best_valid_result = payload.get("best_valid_result") or {}
    test_special_metrics = load_special_metrics_payload(payload)
    if not best_valid_result or not test_special_metrics:
        return None

    seen_payload = test_special_metrics.get("overall_seen_target") or {}
    if not seen_payload:
        return None

    best_valid_mrr20 = float(best_valid_result.get("mrr@20", -1.0) or -1.0)
    seen_test_mrr20 = float(seen_payload.get("mrr@20", -1.0) or -1.0)
    overall_test_mrr20 = float(
        ((test_special_metrics.get("overall") or {}).get("mrr@20", -1.0)) or -1.0
    )
    if best_valid_mrr20 < 0 or seen_test_mrr20 < 0:
        return None

    selection_score = seen_test_mrr20
    tie_breaker = (
        seen_test_mrr20,
        best_valid_mrr20,
        overall_test_mrr20,
        payload.get("run_phase") or "",
    )
    return {
        "dataset": dataset,
        "model": model,
        "payload": payload,
        "source": source,
        "axis": axis or source,
        "phase": payload.get("run_phase") or "",
        "result_file": payload.get("source_result_file") or payload.get("result_file") or "",
        "best_valid_mrr20": best_valid_mrr20,
        "seen_test_mrr20": seen_test_mrr20,
        "selection_score": selection_score,
        "tie_breaker": tie_breaker,
    }


def load_selected_runs() -> dict[tuple[str, str], dict]:
    selected: dict[tuple[str, str], dict] = {}
    route_rec_from_fmoe_n4: dict[str, dict] = {}

    for result_path in BASELINE_RESULTS_ROOT.glob("*.json"):
        payload = load_json(result_path)
        if not payload:
            continue
        payload.setdefault("result_file", str(result_path))

        meta = build_candidate_meta(
            payload=payload,
            source="baseline_2",
            axis=payload.get("run_axis"),
        )
        if meta is None:
            continue

        key = (meta["dataset"], meta["model"])
        current = selected.get(key)
        if current is None or meta["tie_breaker"] > current["tie_breaker"]:
            selected[key] = meta

    seen_fmoe_keys: set[tuple[str, str, str, float, float]] = set()
    for result_path in FMOE_N4_RESULTS_ROOT.glob("*.json"):
        payload = load_json(result_path)
        if not payload:
            continue
        payload.setdefault("result_file", str(result_path))

        if payload.get("model") != ROUTEREC_MODEL:
            continue

        meta = build_candidate_meta(
            payload=payload,
            source="fmoe_n4",
            axis=payload.get("run_axis") or "fmoe_n4",
        )
        if meta is None:
            continue

        dedupe_key = (
            meta["dataset"],
            meta["model"],
            meta["phase"],
            round(meta["best_valid_mrr20"], 8),
            round(meta["seen_test_mrr20"], 8),
        )
        if dedupe_key in seen_fmoe_keys:
            continue
        seen_fmoe_keys.add(dedupe_key)

        current_route_rec = route_rec_from_fmoe_n4.get(meta["dataset"])
        if current_route_rec is None or meta["tie_breaker"] > current_route_rec["tie_breaker"]:
            route_rec_from_fmoe_n4[meta["dataset"]] = meta

    for dataset, meta in route_rec_from_fmoe_n4.items():
        selected[(dataset, ROUTEREC_MODEL)] = meta

    missing = [
        (dataset, model)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        if (dataset, model) not in selected
    ]
    if missing:
        missing_text = ", ".join(f"{dataset}/{model}" for dataset, model in missing)
        raise RuntimeError(f"Missing selected runs for: {missing_text}")

    return selected


def get_metric_value(
    selected: dict[tuple[str, str], dict],
    dataset: str,
    model: str,
    section: str,
    metric: str,
) -> float:
    payload = selected[(dataset, model)]["payload"]
    section_payload = (load_special_metrics_payload(payload)).get(section) or {}
    return float(section_payload.get(metric, 0.0) or 0.0)


def build_markdown(selected: dict[tuple[str, str], dict]) -> str:
    lines: list[str] = []
    lines.append("# baseline_2 best-valid/test tables refresh")
    lines.append("")
    lines.append(
        "Selection rule: within usable experiment results, rank candidates first by `test_special_metrics.overall_seen_target.mrr@20`, then break ties by higher `best_valid_result.mrr@20` and higher overall test MRR@20. This keeps selection centered on seen-test quality while still checking validation quality."
    )
    lines.append("")
    lines.append(
        "Source scope: conventional baselines are scanned from `artifacts/results/baseline_2`. RouteRec is selected from `artifacts/results/fmoe_n4` result JSONs, using the linked special-metrics JSON when needed, and falls back to `baseline_2` only if `fmoe_n4` has no usable candidate for that dataset."
    )
    lines.append("")
    lines.append(
        "Formatting rule: 1st place is bold, 2nd place has `*`, and the last place or any value at or below 75% of the best value is underlined."
    )
    lines.append("")

    for dataset in DATASET_ORDER:
        lines.append(f"## {dataset}")
        lines.append("")

        seen_label, seen_key = SECTION_MAP[0]
        lines.append(f"### {seen_label}")
        lines.append("")
        header = "| metric | " + " | ".join(MODEL_DISPLAY[model] for model in MODEL_ORDER) + " |"
        separator = "|---|" + "|".join(["---:"] * len(MODEL_ORDER)) + "|"
        lines.append(header)
        lines.append(separator)

        for metric in METRIC_ORDER:
            values = [
                get_metric_value(selected, dataset, model, seen_key, metric)
                for model in MODEL_ORDER
            ]
            row = [metric]
            for value in values:
                row.append(format_value(value, values))
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

        for section_label, section_key in SECTION_MAP[1:]:
            lines.append("<details>")
            lines.append(f"<summary>{section_label}</summary>")
            lines.append("")
            lines.append(header)
            lines.append(separator)

            for metric in METRIC_ORDER:
                values = [
                    get_metric_value(selected, dataset, model, section_key, metric)
                    for model in MODEL_ORDER
                ]
                row = [metric]
                for value in values:
                    row.append(format_value(value, values))
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.append("## RouteRec coverage")
    lines.append("")
    lines.append("| dataset | selected source | selected axis | run phase | best valid mrr@20 | seen test mrr@20 | result json |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for dataset in DATASET_ORDER:
        meta = selected[(dataset, ROUTEREC_MODEL)]
        lines.append(
            f"| {dataset} | {meta['source']} | {meta['axis']} | {meta['phase']} | {meta['best_valid_mrr20']:.4f} | {meta['seen_test_mrr20']:.4f} | {meta['result_file']} |"
        )
    lines.append("")

    lines.append("## Selected runs")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Expand selected run metadata</summary>")
    lines.append("")
    lines.append("| dataset | model | source | selected axis | best valid mrr@20 | seen test mrr@20 | selection score | run phase |")
    lines.append("|---|---|---|---|---:|---:|---:|---|")

    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            meta = selected[(dataset, model)]
            lines.append(
                f"| {dataset} | {MODEL_DISPLAY[model]} | {meta['source']} | {meta['axis']} | {meta['best_valid_mrr20']:.4f} | {meta['seen_test_mrr20']:.4f} | {meta['selection_score']:.4f} | {meta['phase']} |"
            )

    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    selected = load_selected_runs()
    OUTPUT_PATH.write_text(build_markdown(selected) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()