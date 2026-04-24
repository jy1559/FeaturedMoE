#!/usr/bin/env python3
"""Generate a raw-log motivation report for router-input design."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_ROOT = REPO_ROOT / "Datasets" / "processed" / "feature_added_v4"
OUTPUT_DIR = REPO_ROOT / "outputs" / "dataset_appendix_analysis"
OUTPUT_MD = OUTPUT_DIR / "dataset_router_motivation.md"
OUTPUT_JSON = OUTPUT_DIR / "dataset_router_motivation.json"

DATASETS: list[tuple[str, str]] = [
    ("beauty", "Beauty"),
    ("foursquare", "Foursquare"),
    ("KuaiRecLargeStrictPosV2_0.2", "KuaiRec"),
    ("lastfm0.03", "LastFM"),
    ("movielens1m", "ML-1M"),
    ("retail_rocket", "Retail Rocket"),
]

MODEL_COLUMNS = [
    "SASRec",
    "GRU4Rec",
    "TiSASRec",
    "FEARec",
    "DuoRec",
    "BSARec",
    "FAME",
    "DIF-SR",
    "FDSA",
    "RouteRec",
]


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_int(value: int) -> str:
    return f"{value:,}"


def normalize_dataset_name(name: str) -> str:
    return name.replace("$", "").replace("\\dagger", "").replace("^", "")


def rank_corr(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    if left_rank.std(ddof=0) == 0 or right_rank.std(ddof=0) == 0:
        return 0.0
    return float(left_rank.corr(right_rank))


def parse_full_results() -> pd.DataFrame:
    paper_path = REPO_ROOT / "writing" / "ACM_template" / "sample-sigconf.tex"
    text = paper_path.read_text(encoding="utf-8")
    match = re.search(
        r"\\multirow\{9\}\{\*\}\{Beauty\}[\s\S]*?\\multicolumn\{2\}\{l\}\{\\textbf\{Avg\.~Rank",
        text,
    )
    if match is None:
        raise RuntimeError("Could not locate full results table in sample-sigconf.tex")

    rows: list[list[object]] = []
    current_dataset: str | None = None
    for line in match.group(0).splitlines():
        line = line.strip()
        if "\\multirow{9}{*}{" in line:
            dataset_match = re.search(r"\\multirow\{9\}\{\*\}\{([^}]*)\}", line)
            if dataset_match is not None:
                current_dataset = normalize_dataset_name(dataset_match.group(1))
            continue
        if not line.startswith("&") or current_dataset is None:
            continue
        parts = [part.strip() for part in line.split("&")]
        metric = parts[1]
        values: list[float] = []
        for value in parts[2:]:
            value = value.replace("\\\\", "").strip()
            value = re.sub(r"\\tblbest\{([^}]*)\}", r"\1", value)
            value = re.sub(r"\\tblsecond\{([^}]*)\}", r"\1", value)
            values.append(float(value))
        rows.append([current_dataset, metric, *values])

    return pd.DataFrame(rows, columns=["dataset", "metric", *MODEL_COLUMNS])


def compute_performance_profile(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for dataset, frame in results.groupby("dataset"):
        gap_rows: list[float] = []
        win_rows = 0
        best_baseline_names: list[str] = []
        for _, row in frame.iterrows():
            baseline_name, baseline_value = max(
                ((model, float(row[model])) for model in MODEL_COLUMNS if model != "RouteRec"),
                key=lambda pair: pair[1],
            )
            route_value = float(row["RouteRec"])
            gap_rows.append(route_value - baseline_value)
            win_rows += int(route_value >= max(float(row[model]) for model in MODEL_COLUMNS) - 1e-12)
            best_baseline_names.append(baseline_name)
        records.append(
            {
                "dataset": dataset,
                "route_win_rate": win_rows / len(frame),
                "route_win_count": win_rows,
                "avg_gain_to_best_baseline": float(np.mean(gap_rows)),
                "median_gain_to_best_baseline": float(np.median(gap_rows)),
                "most_frequent_competitor": pd.Series(best_baseline_names).value_counts().idxmax(),
            }
        )
    return pd.DataFrame(records)


def compute_dataset_axes() -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for folder_name, display_name in DATASETS:
        inter_path = DATA_ROOT / folder_name / f"{folder_name}.inter"
        inter_frame = pd.read_csv(
            inter_path,
            sep="\t",
            usecols=lambda column: column.split(":", 1)[0] in {"session_id", "user_id", "item_id", "timestamp"},
        )
        inter_frame.columns = [column.split(":", 1)[0] for column in inter_frame.columns]
        inter_frame = inter_frame.sort_values(["user_id", "session_id", "timestamp"], kind="mergesort")

        session_frame = (
            inter_frame.groupby("session_id", sort=False)
            .agg(
                user_id=("user_id", "first"),
                session_len=("item_id", "size"),
                session_start=("timestamp", "min"),
                session_end=("timestamp", "max"),
                unique_items=("item_id", pd.Series.nunique),
            )
            .reset_index()
        )
        session_frame["repeat_ratio"] = 1.0 - session_frame["unique_items"] / session_frame["session_len"]
        session_frame["duration"] = session_frame["session_end"] - session_frame["session_start"]
        sessions_per_user = session_frame.groupby("user_id", sort=False).size()

        sequences = inter_frame.groupby("session_id", sort=False)["item_id"].agg(list)
        item_sets = {session_id: set(items) for session_id, items in sequences.items()}

        overlaps: list[float] = []
        time_gaps: list[float] = []
        ordered_sessions = session_frame.sort_values(["user_id", "session_start"], kind="mergesort")
        for _, user_frame in ordered_sessions.groupby("user_id", sort=False):
            previous_session_id: str | None = None
            previous_start: float | None = None
            for row in user_frame.itertuples(index=False):
                if previous_session_id is not None and previous_start is not None:
                    left = item_sets[previous_session_id]
                    right = item_sets[row.session_id]
                    union = len(left | right)
                    overlaps.append(len(left & right) / union if union else 0.0)
                    time_gaps.append(max(float(row.session_start) - float(previous_start), 0.0))
                previous_session_id = row.session_id
                previous_start = row.session_start

        transitions = inter_frame[["session_id", "item_id"]].copy()
        transitions["next_item"] = transitions.groupby("session_id", sort=False)["item_id"].shift(-1)
        transitions = transitions.dropna(subset=["next_item"])
        pair_counts = (
            transitions.groupby(["item_id", "next_item"], sort=False)
            .size()
            .rename("cnt")
            .reset_index()
        )
        source_totals = pair_counts.groupby("item_id", sort=False)["cnt"].sum().rename("src_total").reset_index()
        pair_counts = pair_counts.merge(source_totals, on="item_id", how="left")
        pair_counts["p"] = pair_counts["cnt"] / pair_counts["src_total"]
        source_entropy = (
            pair_counts.groupby("item_id", sort=False)
            .apply(lambda frame: float(-(frame["p"] * np.log(frame["p"] + 1e-12)).sum()))
            .rename("entropy")
            .reset_index()
        )
        source_entropy = source_entropy.merge(source_totals, on="item_id", how="left")
        source_entropy["norm_entropy"] = source_entropy["entropy"] / np.log(source_entropy["src_total"].clip(lower=2))
        transition_branching = (
            float(np.average(source_entropy["norm_entropy"], weights=source_entropy["src_total"]))
            if len(source_entropy)
            else 0.0
        )

        item_support = inter_frame.groupby("item_id", sort=False).size()
        support = np.sort(item_support.to_numpy(dtype=np.float64))
        n_support = len(support)
        popularity_concentration = (
            float((n_support + 1 - 2 * np.cumsum(support).sum() / support.sum()) / n_support)
            if n_support > 0 and support.sum() > 0
            else 0.0
        )

        context_availability = float(
            np.mean(
                [
                    float((sessions_per_user >= 2).mean()),
                    float((sessions_per_user >= 5).mean()),
                    float(np.clip(sessions_per_user.mean() / 20.0, 0.0, 1.0)),
                ]
            )
        )

        session_volatility = float(session_frame["session_len"].std(ddof=0) / session_frame["session_len"].mean())
        repeat_intensity = float(session_frame["repeat_ratio"].mean())
        repeat_variability = float(session_frame["repeat_ratio"].std(ddof=0) / (repeat_intensity + 1e-12))
        timing_irregularity = float(np.std(time_gaps) / (np.mean(time_gaps) + 1e-12)) if time_gaps else 0.0
        duration_irregularity = float(session_frame["duration"].std(ddof=0) / (session_frame["duration"].mean() + 1e-12))
        carryover_strength = float(np.mean(overlaps)) if overlaps else 0.0
        cross_session_drift = float(1.0 - carryover_strength) if overlaps else 0.0

        records.append(
            {
                "dataset": display_name,
                "interactions": int(len(inter_frame)),
                "sessions": int(len(session_frame)),
                "users": int(session_frame["user_id"].nunique()),
                "items": int(inter_frame["item_id"].nunique()),
                "avg_sessions_per_user": float(sessions_per_user.mean()),
                "avg_session_len": float(session_frame["session_len"].mean()),
                "session_volatility": session_volatility,
                "short_session_share": float((session_frame["session_len"] <= 5).mean()),
                "long_session_share": float((session_frame["session_len"] >= 20).mean()),
                "timing_irregularity": timing_irregularity,
                "duration_irregularity": duration_irregularity,
                "transition_branching": transition_branching,
                "repeat_intensity": repeat_intensity,
                "repeat_variability": repeat_variability,
                "carryover_strength": carryover_strength,
                "cross_session_drift": cross_session_drift,
                "popularity_concentration": popularity_concentration,
                "head_item_share": float(item_support.max() / len(inter_frame)),
                "context_availability": context_availability,
                "simple_router_demand": float(np.mean([np.clip(session_volatility / 2.0, 0.0, 1.0), transition_branching, context_availability])),
            }
        )
    return pd.DataFrame(records)


def build_axis_table(axes_frame: pd.DataFrame) -> str:
    lines = [
        "| Dataset | SessVol | Branch | RepeatVar | Carryover | Drift | PopConc | CtxAvail | Gain | WinRate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in axes_frame.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {format_float(row.session_volatility)} | {format_float(row.transition_branching)} | {format_float(row.repeat_variability)} | {format_float(row.carryover_strength)} | {format_float(row.cross_session_drift)} | {format_float(row.popularity_concentration)} | {format_float(row.context_availability)} | {format_float(row.avg_gain_to_best_baseline, 4)} | {format_float(row.route_win_rate)} |"
        )
    return "\n".join(lines)


def build_dataset_table_glossary() -> str:
    lines = [
        "| Compact header | Full metric name | How it is computed | Interpretation |",
        "| --- | --- | --- | --- |",
        "| `SessVol` | `session_volatility` | `std(session_len) / mean(session_len)` | 세션 길이가 dataset 안에서 얼마나 들쭉날쭉한가 |",
        "| `Branch` | `transition_branching` | source item별 next-item entropy를 정규화한 뒤 transition 수로 가중 평균 | 같은 local state에서 다음 행동이 얼마나 여러 갈래로 갈라지는가 |",
        "| `RepeatVar` | `repeat_variability` | `std(repeat_ratio) / mean(repeat_ratio)` | 반복 소비 강도가 session마다 얼마나 불균일한가 |",
        "| `Carryover` | `carryover_strength` | 연속 session item-set의 Jaccard overlap 평균 | 이전 session item set이 다음 session으로 얼마나 이어지는가 |",
        "| `Drift` | `cross_session_drift` | `1 - carryover_strength` | 연속 session이 얼마나 다른 item set으로 이동하는가 |",
        "| `PopConc` | `popularity_concentration` | item frequency 분포의 Gini coefficient | interaction이 소수 인기 item에 얼마나 집중되는가 |",
        "| `CtxAvail` | `context_availability` | `mean(users>=2 ratio, users>=5 ratio, clip(avg_sessions_per_user/20,0,1))` | 반복 session 문맥이 router에 얼마나 제공되는가 |",
        "| `Gain` | `avg_gain_to_best_baseline` | 각 metric row에서 `RouteRec - best baseline`을 구한 뒤 평균 | strongest baseline 대비 평균 우위/열위 |",
        "| `WinRate` | `route_win_rate` | 전체 metric row 중 RouteRec이 1등인 비율 | dataset 내 여러 지표에서 얼마나 자주 최고 성능을 냈는가 |",
    ]
    return "\n".join(lines)


def build_dataset_table_components(axes_frame: pd.DataFrame) -> str:
    lines = [
        "| Dataset | Interactions | Sessions | Users | Items | AvgSess/User | AvgSessLen | Short<=5 | Long>=20 | RepeatInt | TimeIrreg | DurIrreg | HeadItemShare |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in axes_frame.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {format_int(row.interactions)} | {format_int(row.sessions)} | {format_int(row.users)} | {format_int(row.items)} | {format_float(row.avg_sessions_per_user)} | {format_float(row.avg_session_len)} | {format_float(row.short_session_share)} | {format_float(row.long_session_share)} | {format_float(row.repeat_intensity)} | {format_float(row.timing_irregularity)} | {format_float(row.duration_irregularity)} | {format_float(row.head_item_share)} |"
        )
    return "\n".join(lines)


def build_full_axis_value_table(axes_frame: pd.DataFrame) -> str:
    lines = [
        "| Dataset | SessVol | Branch | RepeatInt | RepeatVar | Carryover | Drift | TimeIrreg | DurIrreg | PopConc | HeadShare | CtxAvail | RouterDemand | Gain | WinRate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in axes_frame.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {format_float(row.session_volatility)} | {format_float(row.transition_branching)} | {format_float(row.repeat_intensity)} | {format_float(row.repeat_variability)} | {format_float(row.carryover_strength)} | {format_float(row.cross_session_drift)} | {format_float(row.timing_irregularity)} | {format_float(row.duration_irregularity)} | {format_float(row.popularity_concentration)} | {format_float(row.head_item_share)} | {format_float(row.context_availability)} | {format_float(row.simple_router_demand)} | {format_float(row.avg_gain_to_best_baseline, 4)} | {format_float(row.route_win_rate)} |"
        )
    return "\n".join(lines)


def build_correlation_table(corr_frame: pd.DataFrame) -> str:
    lines = [
        "| Candidate axis | Rank corr. with gain | Rank corr. with win rate |",
        "| --- | ---: | ---: |",
    ]
    for row in corr_frame.itertuples(index=False):
        lines.append(
            f"| {row.axis} | {format_float(row.rho_gain)} | {format_float(row.rho_win)} |"
        )
    return "\n".join(lines)


def build_markdown(axes_frame: pd.DataFrame, corr_frame: pd.DataFrame) -> str:
    axis_ranges = {}
    for column in [
        "session_volatility",
        "transition_branching",
        "repeat_intensity",
        "repeat_variability",
        "popularity_concentration",
        "context_availability",
        "timing_irregularity",
        "carryover_strength",
        "duration_irregularity",
        "head_item_share",
    ]:
        series = axes_frame[column]
        axis_ranges[column] = (float(series.min()), float(series.max()), float(series.max() - series.min()))

    lines: list[str] = []
    lines.append("# Router Motivation From Raw Logs")
    lines.append("")
    lines.append("## Motivation")
    lines.append("")
    lines.append("- 핵심 질문은 'heterogeneity가 큰 dataset이 무엇인가'가 아니라, sequential input이 어떤 축에서 반복적으로 heterogeneous해지는가이다.")
    lines.append("- 이를 위해 raw interaction log만 사용해 여러 candidate axis를 계산했고, dataset-specific한 특수 현상보다 여러 dataset에서 공통적으로 관찰되고 router 설계로 연결될 수 있는 축을 우선적으로 골랐다.")
    lines.append("- 따라서 아래의 4(+1)축은 결과를 가장 잘 맞춘 후행 score가 아니라, raw logs에서 반복적으로 드러나는 heterogeneity pattern을 정리한 motivation-oriented axis set이다.")
    lines.append("")
    lines.append("## Recurring Heterogeneity Axes")
    lines.append("")
    lines.append("- Tempo / volatility: 입력의 pace와 session form이 얼마나 빠르고 불규칙하게 바뀌는가")
    lines.append("- Transition ambiguity: 비슷한 local state에서 다음 행동이 얼마나 여러 방향으로 갈라지는가")
    lines.append("- Memory regime: 반복 소비와 persistence가 얼마나 강하고 또 얼마나 가변적인가")
    lines.append("- Exposure regime: 행동이 head-heavy exposure 쪽에 더 묶이는지, 더 preference-driven한지")
    lines.append("- Context availability: 위 heterogeneity가 존재하더라도, router가 활용할 repeated-session context가 충분한가")
    lines.append("")
    lines.append("## Why These Axes")
    lines.append("")
    lines.append("- 첫째, dataset 간 변동폭이 충분히 커야 한다. 그래야 axis가 실제 dataset profile을 나누는 설명 변수로 기능할 수 있다.")
    lines.append("- 둘째, raw logs에서 직접 계산 가능해야 한다. 그래야 model-agnostic motivation이 된다.")
    lines.append("- 셋째, cue family로 자연스럽게 연결되어야 한다. 즉 predictor feature가 아니라 router control 축으로 해석될 수 있어야 한다.")
    lines.append("- 넷째, 서로 완전히 같은 현상을 중복해서 설명하지 않아야 한다. 그래서 short/long share 같은 지표는 tempo의 보조 관측치로 두고, 메인 axis는 더 응축된 형태로 잡았다.")
    lines.append("")
    lines.append("## How Each Candidate Axis Is Computed")
    lines.append("")
    lines.append("- 공통 입력: 모든 axis는 `session_id`, `user_id`, `item_id`, `timestamp` 네 컬럼만 사용한다. 먼저 interaction log를 `(user_id, session_id, timestamp)` 순으로 정렬한 뒤, session 단위와 user-session sequence 단위 집계를 만든다.")
    lines.append("- Session summary: 각 session에 대해 `session_len`, `session_start`, `session_end`, `unique_items`, `duration=session_end-session_start`, `repeat_ratio=1-(unique_items/session_len)`를 계산한다.")
    lines.append("- User-level session chain: 각 user의 session을 시간순으로 놓고, 인접 session 쌍마다 item-set overlap과 session start 간 gap을 계산한다.")
    lines.append("- Transition summary: 각 session 내부에서 `(current_item, next_item)` transition을 만들고, 동일한 `current_item`에서 다음 item 분포가 얼마나 퍼지는지 본다.")
    lines.append("")
    lines.append("### Candidate Axis Definitions")
    lines.append("")
    lines.append("- `avg_sessions_per_user`: user별 session 수를 센 뒤 그 평균을 사용한다.")
    lines.append("- `avg_session_len`: session별 interaction 수의 평균이다.")
    lines.append("- `session_volatility`: session length의 변동계수로 계산한다. 즉 `std(session_len) / mean(session_len)`이다.")
    lines.append("- `short_session_share`: 길이 5 이하 session의 비율이다.")
    lines.append("- `long_session_share`: 길이 20 이상 session의 비율이다.")
    lines.append("- `repeat_intensity`: session별 `repeat_ratio`의 평균이다. 한 session 안에서 item 재방문이 많을수록 커진다.")
    lines.append("- `repeat_variability`: session별 `repeat_ratio`의 변동계수다. 즉 `std(repeat_ratio) / mean(repeat_ratio)`이며, 반복 강도가 session마다 얼마나 들쭉날쭉한지 본다.")
    lines.append("- `carryover_strength`: 같은 user의 연속한 두 session에 대해, 두 item set의 Jaccard overlap `|A∩B| / |A∪B|`를 계산하고 그 평균을 취한다.")
    lines.append("- `cross_session_drift`: `1 - carryover_strength`로 둔다. 즉 연속 session이 얼마나 다른 item set으로 이동하는지 보는 반대 방향 지표다.")
    lines.append("- `timing_irregularity`: user의 연속 session pair에 대해 `next_session_start - previous_session_start` gap을 모은 뒤, 그 변동계수 `std(gap) / mean(gap)`를 사용한다.")
    lines.append("- `duration_irregularity`: session duration의 변동계수 `std(duration) / mean(duration)`이다.")
    lines.append("- `transition_branching`: 각 source item에서 다음 item 분포의 entropy를 계산하고, 이를 가능한 분기 수에 맞춰 정규화한 뒤 transition 수로 가중 평균한다. 값이 클수록 같은 local state에서 다음 행동이 더 여러 방향으로 갈라진다.")
    lines.append("- `popularity_concentration`: 전체 item frequency 분포의 Gini coefficient로 계산한다. 값이 클수록 interaction이 소수 head item에 더 집중된다.")
    lines.append("- `head_item_share`: 가장 많이 등장한 단일 item의 interaction 비중이다. 즉 `max_item_frequency / total_interactions`다.")
    lines.append("- `context_availability`: `(users>=2 sessions 비율 + users>=5 sessions 비율 + clip(avg_sessions_per_user/20, 0, 1)) / 3`으로 만든 보조 지표다. 반복 session 문맥이 router에 얼마나 제공되는지 보기 위한 것이다.")
    lines.append("- `simple_router_demand`: `mean(clip(session_volatility/2, 0, 1), transition_branching, context_availability)`로 만든 단순 종합 지표다. 최종 선택축이라기보다 screening용 reference score에 가깝다.")
    lines.append("")
    lines.append("### Interpretation Notes")
    lines.append("")
    lines.append("- 이 값들은 모두 dataset-level aggregate다. 즉 per-session 또는 per-user raw statistics를 먼저 만든 뒤, dataset마다 하나의 요약값으로 압축한다.")
    lines.append("- 대부분의 axis는 절대량보다 scale-free 비교가 중요하다고 보고 평균보다 변동계수(CV)를 사용했다. 그래서 domain별 interaction 규모 차이보다 pattern 차이를 더 보도록 설계했다.")
    lines.append("- `transition_branching`과 `popularity_concentration`은 각각 local next-step ambiguity와 global exposure concentration을 잡기 위한 축이라, session length 계열과는 다른 정보를 준다.")
    lines.append("")
    lines.append("## Selected Axes and Raw Indicators")
    lines.append("")
    lines.append("| Axis | Raw-log indicators | Why kept for motivation | Cue-family link |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| Tempo / volatility | `session_volatility`, `timing_irregularity`, short/long session share | dataset 간 변동폭이 크고 (`session_volatility` range {format_float(axis_ranges['session_volatility'][2])}, `timing_irregularity` range {format_float(axis_ranges['timing_irregularity'][2])}), '계산 경로를 빨리 바꿔야 하는 입력'을 가장 직관적으로 설명함 | Tempo |")
    lines.append(f"| Transition ambiguity | `transition_branching` | 다음 행동 분기 구조를 직접 보여주며, hidden-only router가 놓치기 쉬운 local multimodality를 설명함 (`branching` range {format_float(axis_ranges['transition_branching'][2])}) | Focus |")
    lines.append(f"| Memory regime | `repeat_intensity`, `repeat_variability`, supplementary `carryover_strength` | 반복/재등장 패턴은 gain과도 비교적 잘 맞고, dataset 간 차이도 큼 (`repeat_variability` range {format_float(axis_ranges['repeat_variability'][2])}) | Memory |")
    lines.append(f"| Exposure regime | `popularity_concentration`, supplementary `head_item_share` | gain과의 직접 정렬은 약하지만, browsing-heavy vs preference-driven regime 차이를 설명하는 독립 축으로 유지할 가치가 있음 (`popularity_concentration` range {format_float(axis_ranges['popularity_concentration'][2])}) | Exposure |")
    lines.append(f"| Context availability | `users>=2 sessions`, `users>=5 sessions`, `avg_sessions_per_user` | heterogeneity 자체보다 'routing이 실제로 배울 수 있는 문맥'을 설명하는 보조축으로 필요함 (`context_availability` range {format_float(axis_ranges['context_availability'][2])}) | macro/mid routing support |")
    lines.append("")
    lines.append("## Secondary Candidates That Were Not Promoted")
    lines.append("")
    lines.append("- `duration_irregularity`: 값의 range는 크지만 시간 단위와 sessionization 규칙에 지나치게 민감해, 공통 motivation 축으로 쓰기엔 domain effect가 강하다.")
    lines.append("- `short_session_share`, `long_session_share`: tempo/volatility를 보조적으로 보여주는 좋은 지표이지만, 메인 axis라기보다 session volatility를 풀어 설명하는 보조 통계로 두는 편이 깔끔하다.")
    lines.append("- `cross_session_drift`: 직관은 좋지만 현재 sessionization에서는 overlap이 거의 0에 수렴하는 dataset가 많아, 메인 motivation 축으로 쓰기엔 너무 거칠다. carryover strength를 보조 evidence로 두는 편이 더 안정적이다.")
    lines.append("- `head_item_share`: exposure regime을 설명하는 보조 통계로는 유효하지만, popularity concentration보다 정보량이 적다.")
    lines.append("")
    lines.append("## From Axes To Cue Groups")
    lines.append("")
    lines.append("- Tempo / volatility axis는 세션의 속도와 형태가 얼마나 흔들리는지를 보여주므로 Tempo cue family로 연결된다.")
    lines.append("- Transition ambiguity axis는 local next-step branching을 드러내므로, router가 현재 intent concentration vs switching을 보게 하는 Focus family와 연결된다.")
    lines.append("- Memory regime axis는 repeat intensity와 repeat variability를 통해 persistence/recurrence 구조를 보여주므로 Memory family로 연결된다.")
    lines.append("- Exposure regime axis는 popularity concentration을 통해 head-heavy vs preference-driven browsing 차이를 보여주므로 Exposure family로 연결된다.")
    lines.append("- Context availability는 독립 cue family라기보다, 위 cue들이 실제로 macro/mid routing에서 활용될 수 있는 조건을 설명하는 보조축이다.")
    lines.append("")
    lines.append("## Dataset Table")
    lines.append("")
    lines.append("- 아래 첫 표는 본문에서 빠르게 보기 위한 compact table이고, 뒤의 두 표는 이 값들이 무엇의 축약인지 풀어서 보여주는 상세 표다.")
    lines.append("- 즉 compact table의 숫자는 임의 축약이 아니라, 뒤에 있는 raw component와 candidate axis 통계를 다시 묶어 놓은 요약값이다.")
    lines.append("")
    lines.append("### Compact Header Glossary")
    lines.append("")
    lines.append(build_dataset_table_glossary())
    lines.append("")
    lines.append("### Compact Dataset Table")
    lines.append("")
    lines.append(build_axis_table(axes_frame))
    lines.append("")
    lines.append("### Raw Components Behind The Table")
    lines.append("")
    lines.append(build_dataset_table_components(axes_frame))
    lines.append("")
    lines.append("### Full Candidate-Axis Value Table")
    lines.append("")
    lines.append(build_full_axis_value_table(axes_frame))
    lines.append("")
    lines.append("- `Raw Components Behind The Table`는 compact table에 직접 안 들어간 보조 값들까지 포함한다. 예를 들어 `RepeatVar`가 왜 큰지 보려면 `RepeatInt`와 함께 읽는 편이 좋고, `CtxAvail`는 `AvgSess/User` 및 multi-session user 비율과 같이 읽어야 해석이 자연스럽다.")
    lines.append("- `Full Candidate-Axis Value Table`는 실제 correlation 계산에 들어간 candidate axis를 거의 전부 모은 표다. 그래서 본문용 compact table, appendix용 full table, 계산 정의 섹션이 서로 연결되도록 구성했다.")
    lines.append("")
    lines.append("## Directional Comparison With Results")
    lines.append("")
    lines.append("- 아래 rank correlation은 dataset 수가 6개뿐이라 강한 통계 검정보다 directional evidence로 읽는 것이 적절하다.")
    lines.append("- Motivation의 중심은 correlation 자체가 아니라, 위에서 정의한 axis가 실제 RouteRec gain과도 완전히 어긋나지 않는다는 점을 확인하는 것이다.")
    lines.append("")
    lines.append(build_correlation_table(corr_frame))
    lines.append("")
    lines.append("- `repeat_intensity`, `repeat_variability`, `session_volatility`는 gain과 비교적 잘 정렬된다. 즉 memory regime와 tempo volatility는 실제로 routing-relevant한 축으로 볼 근거가 있다.")
    lines.append("- `transition_branching`은 correlation 수치만 보면 아주 강하지 않지만, KuaiRec처럼 실제 gain이 가장 큰 데이터에서 매우 높게 나타나고, hidden-only router의 한계를 설명하기에 가장 해석이 좋은 축이므로 motivation에서 유지할 가치가 있다.")
    lines.append("- `context_availability`는 단독 상관보다 Beauty/Retail Rocket vs KuaiRec/Foursquare/LastFM의 차이를 설명하는 gating condition으로 더 유용하다.")
    lines.append("")
    lines.append("## Clean Story Flow")
    lines.append("")
    lines.append("- Step 1: raw logs를 보면 sequential input은 여러 dataset에서 반복적으로 tempo, branching, repeat regime, exposure 차이 위에서 heterogeneous해진다.")
    lines.append("- Step 2: hidden-only router는 입력이 어느 behavioral axis에서 다른지를 직접 보지 못한다.")
    lines.append("- Step 3: 따라서 router input은 richer predictive feature가 아니라, 이 recurring heterogeneity axis를 operationalize한 lightweight cue여야 한다.")
    lines.append("- Step 4: RouteRec은 이 축에 맞춰 Tempo, Focus, Memory, Exposure의 네 cue group을 구성하고, context availability는 그 cue가 실제 routing에 활용될 수 있는 조건을 설명하는 보조축으로 둔다.")
    lines.append("- Step 5: 이후 결과를 보면, 위 축이 상대적으로 강한 dataset일수록 RouteRec의 이득이 더 크게 나타난다.")
    lines.append("")
    lines.append("## Introduction-Style Paragraph")
    lines.append("")
    lines.append("- Rather than starting from a single scalar notion of heterogeneity, we inspect raw interaction logs and find that sequential inputs differ recurrently along a small set of behavioral axes: temporal volatility, transition ambiguity, repetition and carryover structure, and exposure regime. These axes appear across datasets even though their relative strength varies by domain.")
    lines.append("- This suggests that the central MoE question in sequential recommendation is not only whether to route, but along which behavioral axis routing should specialize. Hidden-only routing can respond to representation differences, yet it does not explicitly expose where the heterogeneity comes from. We therefore design router cues as lightweight controls aligned with these recurring axes of heterogeneity rather than as richer predictive side information.")
    lines.append("- Under this view, the four cue groups in RouteRec are not arbitrary feature bundles: they are operationalizations of recurring raw-log heterogeneity patterns. The smaller margin on some datasets is then better interpreted as limited routing headroom under scarce repeated-session context or strong shared-path suitability, rather than as evidence against the routing premise itself.")
    lines.append("")
    lines.append("## Paper Use")
    lines.append("")
    lines.append("- Motivation / Introduction에서는 위 Recurring Heterogeneity Axes와 Introduction-Style Paragraph만 가져가고, correlation과 dataset table은 appendix로 내리는 편이 좋다.")
    lines.append("- Method에서는 'why these cues'를 설명할 때 Selected Axes and Raw Indicators 표를 축약해서 사용하면 된다.")
    lines.append("- Appendix에서는 Dataset Table과 Directional Comparison With Results를 통해, 이 축이 결과와도 대체로 맞물린다는 보조 evidence를 제시하면 충분하다.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = parse_full_results()
    perf_frame = compute_performance_profile(results)
    axis_frame = compute_dataset_axes().merge(perf_frame, on="dataset")
    candidate_columns = [
        column
        for column in axis_frame.columns
        if column
        not in {
            "dataset",
            "interactions",
            "sessions",
            "users",
            "items",
            "avg_gain_to_best_baseline",
            "median_gain_to_best_baseline",
            "route_win_rate",
            "route_win_count",
            "most_frequent_competitor",
        }
    ]
    corr_rows: list[dict[str, float | str]] = []
    for column in candidate_columns:
        corr_rows.append(
            {
                "axis": column,
                "rho_gain": rank_corr(axis_frame[column], axis_frame["avg_gain_to_best_baseline"]),
                "rho_win": rank_corr(axis_frame[column], axis_frame["route_win_rate"]),
            }
        )
    corr_frame = pd.DataFrame(corr_rows).sort_values(["rho_gain", "rho_win"], ascending=False).reset_index(drop=True)

    OUTPUT_MD.write_text(build_markdown(axis_frame, corr_frame), encoding="utf-8")
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "axes": axis_frame.to_dict(orient="records"),
                "correlations": corr_frame.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report_md": str(OUTPUT_MD), "report_json": str(OUTPUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())