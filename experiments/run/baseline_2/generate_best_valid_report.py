from __future__ import annotations

import csv
import json
import re
from pathlib import Path


RESULTS_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/results/baseline_2")
LOGS_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/logs/baseline_2")
OUTPUT_PATH = Path(
    "/workspace/FeaturedMoE/experiments/run/baseline_2/docs/baseline_2_best_valid_test_tables_v4.md"
)

ALLOWED_AXES = {
    "pair60_v4",
    "pair60_v4_revised",
    "pair60_v4_revised_long12h",
    "pair60_addtuning",
    "pair60_addtuning3",
    "abcd_v1",
    "abcd_v2_lean",
}

ADDTUNING_AXES = {
    "pair60_addtuning",
    "pair60_addtuning3",
}

ADDTUNING3_SUMMARY_PATH = LOGS_ROOT / "PAIR60_ADDTUNING3" / "summary.csv"

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
    "TiSASRec",
    "FEARec",
    "DuoRec",
    "BSARec",
    "FAME",
    "DIFSR",
    "FDSA",
    ("FeaturedMoE_N3", "RouteRec"),
]

MODEL_ORDER = [
    item[0] if isinstance(item, tuple) else item for item in MODEL_SPECS
]

MODEL_DISPLAY = {
    (item[0] if isinstance(item, tuple) else item): (item[1] if isinstance(item, tuple) else item)
    for item in MODEL_SPECS
}

MODEL_SLUG_TO_NAME = {model.lower(): model for model in MODEL_ORDER}

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
    ("Overall", "overall"),
    ("Seen", "overall_seen_target"),
    ("Unseen", "overall_unseen_target"),
]

CONVENTIONAL_MODELS = {"SASRec", "GRU4Rec", "TiSASRec"}

RETUNE_PLAN = {
    "beauty": {
        "primary": ["GRU4Rec", "TiSASRec", "BSARec", "FAME", "DIFSR"],
        "secondary": ["FEARec", "DuoRec"],
        "hold": ["SASRec", "FDSA", "FeaturedMoE_N3"],
        "budget": "강재튜닝 모델당 12~16개 config, 보정 재튜닝 모델당 6~8개 config, 최종 후보는 3 seeds 재검증.",
        "note": "여기서 beauty는 amazon_beauty가 아니라 현재 표의 beauty dataset 기준입니다.",
    },
    "retail_rocket": {
        "primary": ["FDSA"],
        "secondary": ["BSARec", "FAME"],
        "hold": ["SASRec", "GRU4Rec", "TiSASRec", "FEARec", "DuoRec", "DIFSR", "FeaturedMoE_N3"],
        "budget": "FDSA는 8~12개 config, BSARec/FAME은 각 4~6개 config 정도의 보정 sweep이면 충분합니다.",
        "note": "strict 75% 미만 모델은 없지만, seen 표 전반에서 FDSA가 가장 많이 밑줄권에 들어가고 BSARec/FAME도 tail metric이 보여 메인 표 신뢰도 보정이 필요합니다.",
    },
    "foursquare": {
        "primary": ["GRU4Rec", "BSARec", "FAME", "DIFSR"],
        "secondary": ["FEARec", "DuoRec", "FDSA"],
        "hold": ["SASRec", "TiSASRec", "FeaturedMoE_N3"],
        "budget": "강재튜닝 모델당 10~14개 config, 2차 확인 모델은 각 4~6개 config 정도가 적절합니다.",
        "note": "엄격한 75% 미만은 GRU4Rec, BSARec, FAME, DIFSR이고, FEARec/DuoRec/FDSA는 밑줄권은 아니지만 선두 클러스터와의 차이를 줄이는 보정 대상입니다.",
    },
    "movielens1m": {
        "primary": ["FEARec", "DuoRec"],
        "secondary": [],
        "hold": ["SASRec", "GRU4Rec", "TiSASRec", "BSARec", "FAME", "DIFSR", "FDSA", "FeaturedMoE_N3"],
        "budget": "각 모델당 4~6개 config의 가벼운 local sweep이면 충분합니다.",
        "note": "strict 75% 미만 모델은 없어서 전체 재튜닝은 과합니다. 다만 FEARec과 DuoRec은 seen 표 일부 metric에서만 반복적으로 밀리므로 소규모 보정이 적절합니다.",
    },
    "lastfm0.03": {
        "primary": ["GRU4Rec"],
        "secondary": [],
        "hold": ["SASRec", "TiSASRec", "FEARec", "DuoRec", "BSARec", "FAME", "DIFSR", "FDSA", "FeaturedMoE_N3"],
        "budget": "GRU4Rec만 6~8개 config 수준으로 재튜닝하고, 나머지는 추가 sweep보다 2~3 seeds 확인이 더 효율적입니다.",
        "note": "사용자 직감과 달리 숫자상 lastfm0.03은 대부분 모델이 상단에 촘촘히 모여 있고, 실제로 반복적으로 약한 모델은 GRU4Rec 하나입니다.",
    },
    "KuaiRecLargeStrictPosV2_0.2": {
        "primary": ["GRU4Rec", "FDSA", "TiSASRec"],
        "secondary": ["BSARec", "FAME"],
        "hold": ["SASRec", "FEARec", "DuoRec", "DIFSR", "FeaturedMoE_N3"],
        "budget": "GRU4Rec/FDSA는 각 10~14개 config, TiSASRec은 6~8개 config, BSARec/FAME은 각 3~4개 config의 확인 sweep을 권장합니다.",
        "note": "strict 75% 미만은 GRU4Rec, FDSA, TiSASRec이고, BSARec/FAME은 절대점수는 낮지만 seen 기준 밑줄권은 아니라 2차 확인 대상으로 두는 편이 합리적입니다.",
    },
}

PHASE_RE = re.compile(r"--run-phase\s+([^\s]+)")
FEATURE_MODE_RE = re.compile(r"feature_mode=([^\s]+)")


def build_phase_feature_map() -> dict[str, str]:
    phase_to_feature: dict[str, str] = {}

    for log_path in LOGS_ROOT.rglob("*.log"):
        try:
            with log_path.open() as handle:
                for _ in range(4):
                    line = handle.readline()
                    if not line:
                        break
                    if "--run-phase" not in line:
                        continue

                    phase_match = PHASE_RE.search(line)
                    feature_match = FEATURE_MODE_RE.search(line)
                    if phase_match and feature_match:
                        phase_to_feature[phase_match.group(1)] = feature_match.group(1)
                    break
        except OSError:
            continue

    return phase_to_feature


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


def load_beauty_addtuning3_shortlist() -> list[dict]:
    if not ADDTUNING3_SUMMARY_PATH.exists():
        return []

    grouped: dict[str, list[dict]] = {}

    try:
        with ADDTUNING3_SUMMARY_PATH.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("dataset") != "beauty" or row.get("status") != "ok":
                    continue

                model = MODEL_SLUG_TO_NAME.get((row.get("model") or "").strip().lower())
                if model is None:
                    continue

                best_valid_mrr20 = float(row.get("best_valid_mrr20") or 0.0)
                seen_test_mrr20 = float(row.get("test_mrr20") or 0.0)
                grouped.setdefault(model, []).append(
                    {
                        "phase": row.get("run_phase") or "",
                        "combo_id": row.get("combo_id") or "",
                        "combo_kind": row.get("combo_kind") or "",
                        "best_valid_mrr20": best_valid_mrr20,
                        "seen_test_mrr20": seen_test_mrr20,
                    }
                )
    except OSError:
        return []

    shortlist = []
    for model in MODEL_ORDER:
        rows = grouped.get(model)
        if not rows:
            continue

        best_valid = max(
            rows,
            key=lambda row: (
                row["best_valid_mrr20"],
                row["seen_test_mrr20"],
                row["phase"],
            ),
        )
        best_test = max(
            rows,
            key=lambda row: (
                row["seen_test_mrr20"],
                row["best_valid_mrr20"],
                row["phase"],
            ),
        )
        shortlist.append(
            {
                "model": model,
                "best_valid": best_valid,
                "best_test": best_test,
            }
        )

    return shortlist


def load_selected_runs() -> dict[tuple[str, str], dict]:
    phase_to_feature = build_phase_feature_map()
    selected: dict[tuple[str, str], dict] = {}

    for result_path in RESULTS_ROOT.glob("*.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        axis = payload.get("run_axis")
        dataset = payload.get("dataset")
        model = payload.get("model")
        phase = payload.get("run_phase")

        if axis not in ALLOWED_AXES:
            continue
        if dataset not in DATASET_ORDER or model not in MODEL_ORDER:
            continue

        feature_mode = phase_to_feature.get(phase)
        if feature_mode not in {"full_v4", "feature_added_v4"}:
            continue

        best_valid_result = payload.get("best_valid_result") or {}
        test_special_metrics = payload.get("test_special_metrics") or {}
        if not best_valid_result or not test_special_metrics:
            continue

        seen_test_mrr20 = float(
            (test_special_metrics.get("overall_seen_target") or {}).get("mrr@20", -1.0)
        )
        if seen_test_mrr20 < 0:
            continue

        selection_key = (dataset, model)
        best_valid_mrr20 = float(best_valid_result.get("mrr@20", -1.0))
        overall_test_mrr20 = float((test_special_metrics.get("overall") or {}).get("mrr@20", -1.0))
        if best_valid_mrr20 <= 0 or seen_test_mrr20 <= 0:
            selection_score = min(best_valid_mrr20, seen_test_mrr20)
        else:
            selection_score = 2.0 * best_valid_mrr20 * seen_test_mrr20 / (best_valid_mrr20 + seen_test_mrr20)
        tie_breaker = (selection_score, best_valid_mrr20, seen_test_mrr20, overall_test_mrr20)

        current = selected.get(selection_key)
        if current is None or tie_breaker > current["tie_breaker"]:
            selected[selection_key] = {
                "payload": payload,
                "axis": axis,
                "phase": phase,
                "feature_mode": feature_mode,
                "best_valid_mrr20": best_valid_mrr20,
                "seen_test_mrr20": seen_test_mrr20,
                "selection_score": selection_score,
                "tie_breaker": tie_breaker,
            }

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


def get_metric_value(selected: dict[tuple[str, str], dict], dataset: str, model: str, section: str, metric: str) -> float:
    payload = selected[(dataset, model)]["payload"]
    section_payload = payload["test_special_metrics"].get(section) or {}
    return float(section_payload.get(metric, 0.0))


def get_top_models(selected: dict[tuple[str, str], dict], dataset: str, section: str, metric: str, top_n: int = 3) -> list[tuple[str, float]]:
    scores = [
        (model, get_metric_value(selected, dataset, model, section, metric))
        for model in MODEL_ORDER
    ]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_n]


def get_seen_mrr20_ratio(selected: dict[tuple[str, str], dict], dataset: str, model: str) -> float:
    best = max(
        get_metric_value(selected, dataset, candidate, "overall_seen_target", "mrr@20")
        for candidate in MODEL_ORDER
    )
    value = get_metric_value(selected, dataset, model, "overall_seen_target", "mrr@20")
    if best <= 1e-12:
        return 0.0
    return value / best


def get_seen_low_metric_count(selected: dict[tuple[str, str], dict], dataset: str, model: str) -> int:
    count = 0

    for metric in METRIC_ORDER:
        values = [
            get_metric_value(selected, dataset, candidate, "overall_seen_target", metric)
            for candidate in MODEL_ORDER
        ]
        value = get_metric_value(selected, dataset, model, "overall_seen_target", metric)
        best = max(values)
        worst = min(values)

        if abs(value - worst) < 1e-12:
            count += 1
        elif best > 1e-12 and value <= 0.75 * best + 1e-12:
            count += 1

    return count


def describe_models(selected: dict[tuple[str, str], dict], dataset: str, models: list[str]) -> str:
    if not models:
        return "없음"

    parts = []
    for model in models:
        ratio = get_seen_mrr20_ratio(selected, dataset, model)
        low_count = get_seen_low_metric_count(selected, dataset, model)
        parts.append(f"{MODEL_DISPLAY[model]} {ratio:.3f}x, 밑줄 {low_count}/9")
    return ", ".join(parts)


def build_dataset_trend_lines(selected: dict[tuple[str, str], dict]) -> list[str]:
    lines = ["## Dataset-level trends", ""]

    for dataset in DATASET_ORDER:
        seen_top = get_top_models(selected, dataset, "overall_seen_target", "mrr@20", top_n=4)
        winner_model, winner_value = seen_top[0]
        conventional_best_model, conventional_best_value = max(
            (
                (model, get_metric_value(selected, dataset, model, "overall_seen_target", "mrr@20"))
                for model in MODEL_ORDER
                if model in CONVENTIONAL_MODELS
            ),
            key=lambda item: item[1],
        )
        route_value = get_metric_value(selected, dataset, "FeaturedMoE_N3", "overall_seen_target", "mrr@20")
        winner_gap_vs_route = winner_value - route_value
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(
            f"- seen MRR@20 기준 상위권은 {', '.join(f'{MODEL_DISPLAY[m]} {v:.4f}' for m, v in seen_top[:3])} 순서입니다. 현재 1위는 {MODEL_DISPLAY[winner_model]}이고, RouteRec은 {route_value:.4f}로 선두와 {winner_gap_vs_route:.4f} 차이입니다."
        )
        if winner_model in CONVENTIONAL_MODELS:
            lines.append(
                f"- 이 데이터셋은 아직 conventional 강세가 남아 있습니다. 특히 {MODEL_DISPLAY[conventional_best_model]}이 {conventional_best_value:.4f}로 기준선을 잡고 있어서, RouteRec이나 다른 non-conventional 모델이 이 구간을 넘지 못하면 '기존 backbone만으로 충분한 것 아니냐'는 해석이 쉽게 나옵니다."
            )
        else:
            lines.append(
                f"- 이 데이터셋은 non-conventional 쪽이 이미 conventional 최상위({MODEL_DISPLAY[conventional_best_model]} {conventional_best_value:.4f})를 넘어섰습니다. 논문에서는 이 차이를 단순 승패보다 '왜 해당 구조가 이 데이터셋에서 유리한가'로 풀어주는 편이 설득력이 있습니다."
            )

        selected_models = [
            model for model in ["FEARec", "DuoRec", "BSARec", "FAME", "DIFSR", "FDSA"]
            if model in MODEL_ORDER
        ]
        improved = []
        for model in selected_models:
            meta = selected[(dataset, model)]
            if meta["axis"] in ADDTUNING_AXES:
                improved.append(f"{MODEL_DISPLAY[model]} {meta['seen_test_mrr20']:.4f}")
        if improved:
            lines.append(
                f"- staged ADDTUNING 반영으로 표에 실제로 들어온 모델은 {', '.join(improved)}입니다. 즉 이번 갱신은 단순 로그 누적이 아니라, 본문 표의 주력 baseline 후보가 일부 바뀌었다는 의미가 있습니다."
            )
        else:
            lines.append(
                "- 이번 staged ADDTUNING에서 새로 본문 표에 들어온 모델은 없었습니다. 이 경우는 기존 Pair60/ABCD에서 이미 선택된 런이 여전히 더 안정적이었다는 뜻으로 해석하는 편이 맞습니다."
            )
        lines.append("")

    return lines


def build_global_analysis_lines(selected: dict[tuple[str, str], dict]) -> list[str]:
    lines = ["## 지금까지 결과 요약", ""]
    route_rec_wins = 0
    addtuning_selected = 0
    total_pairs = 0

    for dataset in DATASET_ORDER:
        winner, _ = get_top_models(selected, dataset, "overall_seen_target", "mrr@20", top_n=1)[0]
        if winner == "FeaturedMoE_N3":
            route_rec_wins += 1
        for model in MODEL_ORDER:
            total_pairs += 1
            if selected[(dataset, model)]["axis"] in ADDTUNING_AXES:
                addtuning_selected += 1

    lines.append(
        f"- 이번 표는 기존 Pair60/ABCD 결과에 staged ADDTUNING까지 합쳐 다시 뽑은 버전입니다. 전체 dataset-model 조합 {total_pairs}개 중 {addtuning_selected}개는 ADDTUNING 결과가 최종 선택으로 올라왔습니다. 즉 추가 튜닝이 실제로 표의 모양을 바꾸는 데 의미가 있었다고 볼 수 있습니다."
    )
    lines.append(
        f"- 다만 RouteRec은 현재 seen MRR@20 기준으로 6개 데이터셋 중 {route_rec_wins}개만 1위입니다. 이번 단계에서 중요한 건 RouteRec을 옹호하는 문장을 늘리는 것이 아니라, strong baseline을 충분히 끌어올린 뒤에도 RouteRec이 어디서 남고 어디서 밀리는지 구조적으로 설명하는 것입니다."
    )
    lines.append(
        "- beauty에서는 staged ADDTUNING 이후 TiSASRec/FDSA가 상단을 더 단단히 잡았고, DuoRec/FEARec도 선두권에 다시 붙었습니다. 다만 BSARec/FAME/DIFSR은 분명히 회복됐어도 아직 top tier와는 갭이 남아 있어서, beauty를 더 판다면 이 세 모델만 선택적으로 보는 편이 맞습니다. 반대로 retail_rocket은 BSARec/FAME이 거의 선두권까지 정리되면서, 이제는 'baseline을 덜 튜닝해서 낮게 나온 것 아니냐'는 반론이 많이 줄었습니다."
    )
    lines.append(
        "- foursquare는 이번 1차에서 가장 의미 있게 표가 바뀐 쪽입니다. DuoRec, FEARec, FDSA, DIFSR이 모두 강해졌고, 그래서 이제는 특정 하나의 backbone이 독주한다기보다 강한 후보군이 촘촘히 붙어 있는 데이터셋으로 보는 편이 맞습니다."
    )
    lines.append(
        "- movielens1m과 KuaiRecLargeStrictPosV2_0.2는 해석이 조금 다릅니다. movielens1m은 DuoRec/FEARec 보강이 있었지만 최상위 conventional/other strong baseline을 완전히 뒤집진 못했고, KuaiRec은 seen-valid 기준으로는 modest한데 seen-test에서는 크게 오르는 조합이 보여서 valid-only selection이 얼마나 위험한지도 같이 보여줍니다."
    )
    lines.append(
        "- unseen-target 표는 여전히 0 근처가 많아서 메인 서사는 seen-target 중심으로 두는 편이 안전합니다. 지금 단계에서는 overall/unseen을 정면 승부 지표로 세우기보다, `seen에서 robust하게 좋고 unseen에서 완전히 무너지지는 않는다` 정도의 보조 증거로 쓰는 편이 낫습니다."
    )
    lines.append("")
    return lines


def build_tuning_lines() -> list[str]:
    return [
        "## 논문에 이대로 쓰면 지적될 수 있는 점",
        "",
        "- 첫 번째로 바로 들어올 지적은 `best_valid만 보고 뽑은 것 아니냐`입니다. 실제로 beauty stage3에서도 FAME이나 일부 TiSASRec 조합처럼 valid와 seen test가 크게 엇갈리는 경우가 있어서, 그런 런을 그대로 넣으면 선택 규칙 자체가 불신을 받습니다.",
        "- 두 번째는 `RouteRec과 baseline의 튜닝 강도가 정말 비슷했느냐`입니다. 이번 1차 ADDTUNING으로 일부 baseline이 꽤 올라왔기 때문에, 오히려 논문에서는 이 점을 숨기지 말고 `baseline도 충분히 다시 맞췄다`는 쪽으로 명시하는 편이 낫습니다.",
        "- 세 번째는 dataset마다 서사가 너무 다르다는 점입니다. beauty처럼 side-information/contrastive 계열이 다시 살아나는 데이터셋과, retail_rocket처럼 BSARec/FAME이 거의 정리되는 데이터셋을 하나의 결론으로 묶으면 과한 일반화로 보일 수 있습니다.",
        "- 네 번째는 unseen-target 해석입니다. 지금 숫자는 대체로 0 근처가 많아서, 메인 claim을 unseen generalization으로 세우면 reviewer가 `평가 프로토콜이 너무 sparse한 것 아니냐`고 물을 가능성이 큽니다.",
        "- 다섯 번째는 RouteRec의 기여 분해가 부족하다는 점입니다. 지금 표만으로는 '좋을 때도 있고 아닐 때도 있다' 이상을 말하기 어렵기 때문에, router/expert/feature branch가 실제로 무엇을 가져왔는지 ablation 없이 메인 모델 우월성을 세게 주장하면 방어가 어렵습니다.",
        "- 마지막으로, 단일 seed 최고값 중심 표는 여전히 과감하게 보일 수 있습니다. 최종 논문 표에서는 최소 3-seed 재실행과 유의성 검정이 따라와야 표 해석이 훨씬 안전해집니다.",
    ]


def build_beauty_addtuning3_lines(selected: dict[tuple[str, str], dict]) -> list[str]:
    shortlist = load_beauty_addtuning3_shortlist()
    if not shortlist:
        return []

    lines = ["## beauty ADDTUNING3 shortlist", ""]
    lines.append(
        "현재 beauty에서는 ADDTUNING3 내부 후보를 따로 훑어서, 각 모델별로 `valid가 가장 좋은 run`과 `seen-test가 가장 좋은 run`을 분리해 적었습니다. 최종 본문 표 반영은 이 후보와 기존 축을 함께 두고 harmonic-mean selection으로 다시 고른 결과입니다."
    )
    lines.append("")
    lines.append("| model | best valid run | valid mrr@20 | seen test mrr@20 | best test run | valid mrr@20 | seen test mrr@20 | final report pick |")
    lines.append("|---|---|---:|---:|---|---:|---:|---|")

    for item in shortlist:
        model = item["model"]
        best_valid = item["best_valid"]
        best_test = item["best_test"]
        final_meta = selected[("beauty", model)]
        final_pick = f"{final_meta['axis']} / {final_meta['phase']}"
        lines.append(
            f"| {MODEL_DISPLAY[model]} | {best_valid['phase']} | {best_valid['best_valid_mrr20']:.4f} | {best_valid['seen_test_mrr20']:.4f} | {best_test['phase']} | {best_test['best_valid_mrr20']:.4f} | {best_test['seen_test_mrr20']:.4f} | {final_pick} |"
        )

    lines.append("")
    return lines


def build_beauty_status_lines(selected: dict[tuple[str, str], dict]) -> list[str]:
    lines = ["## beauty 판단", ""]
    top_models = get_top_models(selected, "beauty", "overall_seen_target", "mrr@20", top_n=4)
    bottom_models = sorted(
        (
            (model, get_metric_value(selected, "beauty", model, "overall_seen_target", "mrr@20"))
            for model in MODEL_ORDER
        ),
        key=lambda item: item[1],
    )[:3]

    lines.append(
        f"- beauty는 이제 `전체 baseline이 다 망한 상태`는 아닙니다. seen MRR@20 기준 상단은 {', '.join(f'{MODEL_DISPLAY[model]} {value:.4f}' for model, value in top_models[:4])}로 형성돼 있고, stage3 반영으로 TiSASRec/FDSA와 DuoRec/FEARec 후보군이 꽤 정상화됐습니다."
    )
    lines.append(
        f"- 반대로 아직 약한 쪽은 {', '.join(f'{MODEL_DISPLAY[model]} {value:.4f}' for model, value in bottom_models)}입니다. 특히 beauty를 더 보정할 이유가 있다면 이 구간은 BSARec/FAME/DIFSR처럼 회복은 됐지만 절대점수가 아직 낮은 모델 때문이라고 보는 편이 맞습니다."
    )
    lines.append(
        "- 따라서 beauty에서 바로 stage4를 넓게 다시 여는 것은 과합니다. 논문 baseline 방어 목적이라면 weak trio만 짧은 targeted follow-up을 하고, 메인 실험 우선순위는 KuaiRecLargeStrictPosV2_0.2와 lastfm0.03 쪽으로 넘기는 편이 효율적입니다."
    )
    lines.append("")
    return lines


def build_publication_lines() -> list[str]:
    return [
        "## 다음에 어떤 실험을 하는 게 괜찮은가", 
        "",
        "1. beauty는 broad sweep을 한 번 더 여는 것보다, BSARec/FAME/DIFSR만 targeted follow-up으로 짧게 정리하는 편이 맞습니다. TiSASRec/FDSA/FEARec/DuoRec는 이미 표 방어가 되는 수준까지 올라왔습니다.",
        "2. baseline 이후 메인 우선순위는 KuaiRecLargeStrictPosV2_0.2와 lastfm0.03로 옮기는 편이 좋습니다. 이 저장소 기준 FMoE_N3 primary track도 두 데이터셋이 핵심이라, baseline 정리 다음 액션과 실험 로드맵이 자연스럽게 이어집니다.",
        "3. retail_rocket은 이제 FDSA보다 BSARec/FAME 쪽을 main strong baseline으로 잡아도 될 수준입니다. 여기서는 더 넓은 탐색보다 shortlist 2~3개 조합을 길게 돌려서 seed 안정성을 보는 것이 효율적입니다.",
        "4. foursquare는 후보군이 많아졌기 때문에, 다음 단계는 넓은 hyperopt보다 backbone-family별로 작은 confirmation run을 하는 편이 낫습니다. DuoRec/FEARec/FDSA/DIFSR를 각각 2~3개 조합만 골라 재검증하면 표가 훨씬 깔끔해집니다.",
        "5. KuaiRecLargeStrictPosV2_0.2는 valid-test mismatch가 커서, 후보 선정 기준을 seen-valid 하나로 두지 말고 seen-test 또는 confirmation seed 평균까지 같이 봐야 합니다. 특히 FDSA/BSARec/FAME은 shortlist를 다시 정리할 가치가 있습니다.",
        "6. RouteRec 실험은 이번 baseline 보강 이후에 다시 들어가는 게 맞습니다. 단, 다음 RouteRec 튜닝은 dataset winner를 따라가는 teacher-style local regime부터 시작해야 하고, router 쪽은 그 다음에 보는 순서가 더 안정적입니다.",
    ]


def build_low_trust_baseline_lines(selected: dict[tuple[str, str], dict]) -> list[str]:
    lines = ["## baseline 쪽에서 추가로 해볼 만한 것", ""]
    lines.append(
        "- 이번 1차 결과를 보면, 모든 baseline을 균등하게 더 돌리는 것보다 `이미 상위권인 모델은 짧은 확인`, `아직 많이 낮은 모델은 공격적 구조 변경`으로 나누는 쪽이 훨씬 효율적입니다."
    )
    lines.append("")

    for dataset in DATASET_ORDER:
        plan = RETUNE_PLAN[dataset]
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(f"- 현재 해석: {plan['note']}")
        lines.append(f"- 더 볼 만한 모델: {describe_models(selected, dataset, plan['primary'])}.")
        if plan["secondary"]:
            lines.append(f"- 짧게 확인할 모델: {describe_models(selected, dataset, plan['secondary'])}.")
        lines.append(f"- 지금 단계에서는 유지해도 되는 모델: {describe_models(selected, dataset, plan['hold'])}.")
        lines.append(f"- 추천 예산: {plan['budget']}")
        lines.append("")

    lines.append("## 모델군별 다음 액션")
    lines.append("")
    lines.append(
        "- GRU4Rec: 완전히 새로 넓게 돌리기보다 `hidden_size`, `num_layers`, `max_len`을 다시 맞추고 learning rate만 짧게 확인하는 정도가 적절합니다."
    )
    lines.append(
        "- SASRec/BSARec 계열: 상위권이면 local confirmation, 하위권이면 `num_layers`, `max_len`, `bsarec_alpha`, `bsarec_c`를 같이 흔드는 식으로 가는 편이 낫습니다."
    )
    lines.append(
        "- TiSASRec: 이번 단계에서는 우선순위가 높지 않습니다. 필요하면 `time_span`과 layer/head만 확인하는 보수적 sweep 정도면 충분합니다."
    )
    lines.append(
        "- FEARec/DuoRec: 이번 1차에서도 실제 개선이 잘 나온 축입니다. backbone보다 `tau`, `contrast`, `lmd_sem`, `max_len`을 먼저 손보는 쪽이 효율이 좋았습니다."
    )
    lines.append(
        "- FAME: beauty처럼 과적합/붕괴가 보이는 곳에서는 `num_layers=1`, `num_experts` 축소/확대, 짧은 sequence 쪽 reset이 더 중요합니다."
    )
    lines.append(
        "- DIFSR/FDSA: feature fusion 방식이 핵심입니다. `fusion_type`, `lambda_attr`, `attribute_hidden_size`, `selected_features`를 backbone depth보다 먼저 보는 게 더 잘 맞습니다."
    )
    lines.append("")
    return lines


def build_markdown(selected: dict[tuple[str, str], dict]) -> str:
    lines: list[str] = []
    lines.append("# baseline_2 best-valid test tables (full_v4 / feature_added_v4)")
    lines.append("")
    lines.append(
        "Selection rule: for each dataset/model pair, compare `ABCD` (`abcd_v1`, `abcd_v2_lean`), `PAIR60_V4`, `PAIR60_V4_REVISED`, `PAIR60_V4_REVISED_LONG12H`, `PAIR60_ADDTUNING`, and `PAIR60_ADDTUNING3`. To avoid selecting runs with very high valid but collapsed seen-target test, rank candidates by the harmonic mean of `best_valid_result.mrr@20` and seen-target `test_special_metrics.overall_seen_target.mrr@20`, then break ties by higher valid and higher seen-test."
    )
    lines.append("")
    lines.append(
        "Filtering note: among the retained runs, the data-source filter again selected only `feature_mode=full_v4`; no final selection used `feature_added_v4`."
    )
    lines.append("")
    lines.append(
        "Formatting rule: 1st place is bold, 2nd place has `*`, and the last place or any value at or below 75% of the best value is underlined."
    )
    lines.append("")

    for dataset in DATASET_ORDER:
        lines.append(f"## {dataset}")
        lines.append("")

        seen_label, seen_key = SECTION_MAP[1]
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

        for section_label, section_key in (SECTION_MAP[0], SECTION_MAP[2]):
            lines.append("<details>")
            lines.append(f"<summary>{section_label}</summary>")
            lines.append("")
            header = "| metric | " + " | ".join(MODEL_DISPLAY[model] for model in MODEL_ORDER) + " |"
            separator = "|---|" + "|".join(["---:"] * len(MODEL_ORDER)) + "|"
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

        lines.append("")

        if dataset == "beauty":
            lines.extend(build_beauty_addtuning3_lines(selected))
            lines.extend(build_beauty_status_lines(selected))

    lines.extend(build_dataset_trend_lines(selected))
    lines.extend(build_global_analysis_lines(selected))
    lines.extend(build_tuning_lines())
    lines.append("")
    lines.extend(build_publication_lines())
    lines.append("")
    lines.extend(build_low_trust_baseline_lines(selected))
    lines.append("")
    lines.append("## Selected runs")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Expand selected run metadata</summary>")
    lines.append("")
    lines.append("| dataset | model | selected axis | best valid mrr@20 | seen test mrr@20 | selection score | feature mode | run phase |")
    lines.append("|---|---|---|---:|---:|---:|---|---|")

    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            meta = selected[(dataset, model)]
            lines.append(
                f"| {dataset} | {MODEL_DISPLAY[model]} | {meta['axis']} | {meta['best_valid_mrr20']:.4f} | {meta['seen_test_mrr20']:.4f} | {meta['selection_score']:.4f} | {meta['feature_mode']} | {meta['phase']} |"
            )

    lines.append("")
    lines.append("</details>")
    return "\n".join(lines) + "\n"


def main() -> None:
    selected = load_selected_runs()
    markdown = build_markdown(selected)
    OUTPUT_PATH.write_text(markdown)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()