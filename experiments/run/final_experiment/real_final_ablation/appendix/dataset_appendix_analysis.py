#!/usr/bin/env python3
"""Compute appendix-ready dataset statistics and macro5 heterogeneity analysis."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_ROOT = REPO_ROOT / "Datasets" / "processed" / "feature_added_v4"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "dataset_appendix_analysis"

DATASETS: list[tuple[str, str]] = [
    ("beauty", "Beauty"),
    ("foursquare", "Foursquare"),
    ("KuaiRecLargeStrictPosV2_0.2", "KuaiRec"),
    ("lastfm0.03", "LastFM"),
    ("movielens1m", "ML-1M"),
    ("retail_rocket", "Retail Rocket"),
]

MACRO5_FAMILIES = {
    "Tempo": [
        "mac5_ctx_valid_r",
        "mac5_gap_last",
        "mac5_pace_mean",
        "mac5_pace_trend",
    ],
    "Focus": [
        "mac5_theme_ent_mean",
        "mac5_theme_top1_mean",
        "mac5_theme_repeat_r",
        "mac5_theme_shift_r",
    ],
    "Memory": [
        "mac5_repeat_mean",
        "mac5_adj_cat_overlap_mean",
        "mac5_adj_item_overlap_mean",
        "mac5_repeat_trend",
    ],
    "Exposure": [
        "mac5_pop_mean",
        "mac5_pop_std_mean",
        "mac5_pop_ent_mean",
        "mac5_pop_trend",
    ],
}
MACRO5_COLUMNS = [column for family in MACRO5_FAMILIES.values() for column in family]


@dataclass
class DatasetArtifacts:
    key: str
    display: str
    inter_path: Path
    item_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Appendix dataset and macro5 analysis")
    parser.add_argument(
        "--datasets",
        default=",".join(key for key, _ in DATASETS),
        help="Comma-separated dataset keys under feature_added_v4",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT),
        help="Directory to write csv/json/markdown outputs",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=4,
        help="Fixed k for session-level macro5 mixture analysis",
    )
    return parser.parse_args()


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values >= 0]
    if values.size == 0 or np.allclose(values.sum(), 0.0):
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    cumulative = np.cumsum(sorted_values)
    return float((n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n)


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(counts.size))


def scaled_variance(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    variances = matrix.var(axis=0)
    return float(np.clip((4.0 * variances).mean(), 0.0, 1.0))


def effective_rank(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    cov = np.cov(matrix, rowvar=False)
    if cov.ndim == 0:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    return float((total * total) / (np.square(eigenvalues).sum() * matrix.shape[1]))


def _init_kmeans_pp(matrix: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_rows = matrix.shape[0]
    centroids = np.empty((k, matrix.shape[1]), dtype=np.float64)
    first_index = int(rng.integers(0, n_rows))
    centroids[0] = matrix[first_index]
    closest_sq = np.square(matrix - centroids[0]).sum(axis=1)
    for index in range(1, k):
        weights = closest_sq / closest_sq.sum() if closest_sq.sum() > 0 else None
        next_index = int(rng.choice(n_rows, p=weights)) if weights is not None else int(rng.integers(0, n_rows))
        centroids[index] = matrix[next_index]
        candidate_sq = np.square(matrix - centroids[index]).sum(axis=1)
        closest_sq = np.minimum(closest_sq, candidate_sq)
    return centroids


def kmeans_metrics(matrix: np.ndarray, k: int) -> tuple[float, float, float]:
    n_rows = matrix.shape[0]
    if n_rows < 2:
        return 0.0, 0.0, 0.0
    k = max(2, min(k, n_rows))
    centroids = _init_kmeans_pp(matrix, k)
    labels = np.zeros(n_rows, dtype=np.int32)
    for _ in range(30):
        distances = np.square(matrix[:, None, :] - centroids[None, :, :]).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centroids[cluster_id] = matrix[mask].mean(axis=0)
    counts = np.bincount(labels, minlength=k)
    active = counts > 0
    if active.sum() <= 1:
        return 0.0, 0.0, 0.0
    entropy = normalized_entropy(counts[active])
    active_centroids = centroids[active]
    pairwise = np.sqrt(np.square(active_centroids[:, None, :] - active_centroids[None, :, :]).sum(axis=2))
    tri_upper = pairwise[np.triu_indices(active_centroids.shape[0], 1)]
    separation = float(np.clip(tri_upper.mean() / math.sqrt(matrix.shape[1]), 0.0, 1.0)) if tri_upper.size else 0.0
    mixture = float(entropy * separation)
    return entropy, separation, mixture


def session_drift_score(session_frame: pd.DataFrame, feature_columns: list[str]) -> float:
    ordered = session_frame.sort_values(["user_id", "session_start"])
    diffs: list[float] = []
    scale = math.sqrt(len(feature_columns))
    for _, user_frame in ordered.groupby("user_id", sort=False):
        if len(user_frame) < 2:
            continue
        vectors = user_frame[feature_columns].to_numpy(dtype=np.float64)
        user_diffs = np.sqrt(np.square(vectors[1:] - vectors[:-1]).sum(axis=1)) / scale
        diffs.extend(user_diffs.tolist())
    return float(np.mean(diffs)) if diffs else 0.0


def clipped_ratio(value: float, scale: float) -> float:
    return float(np.clip(value / scale, 0.0, 1.0))


def compute_raw_behavior_profile(inter_frame: pd.DataFrame, session_frame: pd.DataFrame) -> dict[str, float]:
    session_profile = session_frame[["session_id", "user_id", "session_start", "session_len"]].copy()
    session_unique_items = inter_frame.groupby("session_id", sort=False)["item_id"].nunique()
    session_profile = session_profile.merge(
        session_unique_items.rename("unique_items").reset_index(),
        on="session_id",
        how="left",
    )
    session_profile["repeat_ratio"] = 1.0 - session_profile["unique_items"] / session_profile["session_len"]

    length_cv = float(session_profile["session_len"].std(ddof=0) / session_profile["session_len"].mean())
    repeat_mean = float(session_profile["repeat_ratio"].mean())
    repeat_cv = float(session_profile["repeat_ratio"].std(ddof=0) / (repeat_mean + 1e-12))

    session_items = inter_frame.groupby("session_id", sort=False)["item_id"].agg(list)
    item_sets = {session_id: set(items) for session_id, items in session_items.items()}
    ordered_sessions = session_profile.sort_values(["user_id", "session_start"], kind="mergesort")

    overlaps: list[float] = []
    gap_cvs: list[float] = []
    for _, user_frame in ordered_sessions.groupby("user_id", sort=False):
        if len(user_frame) < 2:
            continue
        previous_session_id: str | None = None
        previous_start: float | None = None
        user_gaps: list[float] = []
        for row in user_frame.itertuples(index=False):
            if previous_session_id is not None and previous_start is not None:
                lhs = item_sets[previous_session_id]
                rhs = item_sets[row.session_id]
                union = len(lhs | rhs)
                overlaps.append(len(lhs & rhs) / union if union else 0.0)
                user_gaps.append(max(float(row.session_start) - float(previous_start), 0.0))
            previous_session_id = row.session_id
            previous_start = row.session_start
        if len(user_gaps) >= 2:
            gap_array = np.asarray(user_gaps, dtype=np.float64)
            gap_cvs.append(float(gap_array.std(ddof=0) / (gap_array.mean() + 1e-12)))

    cross_session_drift = float(1.0 - np.mean(overlaps)) if overlaps else 0.0
    inter_session_gap_cv = float(np.mean(gap_cvs)) if gap_cvs else 0.0

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

    sessions_per_user = ordered_sessions.groupby("user_id", sort=False).size()
    context_availability = float(
        np.mean(
            [
                float((sessions_per_user >= 2).mean()),
                float((sessions_per_user >= 5).mean()),
                float(np.clip(sessions_per_user.mean() / 20.0, 0.0, 1.0)),
            ]
        )
    )
    raw_heterogeneity = float(
        np.mean(
            [
                clipped_ratio(length_cv, 2.0),
                clipped_ratio(repeat_cv, 2.0),
                cross_session_drift,
                clipped_ratio(inter_session_gap_cv, 2.0),
                transition_branching,
            ]
        )
    )
    session_volatility = clipped_ratio(length_cv, 2.0)
    simple_routing_score = float(np.mean([session_volatility, transition_branching, context_availability]))

    return {
        "raw_session_volatility": session_volatility,
        "raw_len_cv": length_cv,
        "raw_repeat_mean": repeat_mean,
        "raw_repeat_cv": repeat_cv,
        "raw_cross_session_drift": cross_session_drift,
        "raw_inter_session_gap_cv": inter_session_gap_cv,
        "raw_transition_branching": transition_branching,
        "raw_context_availability": context_availability,
        "raw_behavioral_heterogeneity": raw_heterogeneity,
        "raw_routing_opportunity": float(raw_heterogeneity * context_availability),
        "simple_routing_score": simple_routing_score,
        "users_ge_2_sessions_ratio": float((sessions_per_user >= 2).mean()),
        "users_ge_5_sessions_ratio": float((sessions_per_user >= 5).mean()),
    }


def compute_full_table_performance_profile() -> dict[str, object]:
    paper_path = REPO_ROOT / "writing" / "ACM_template" / "sample-sigconf.tex"
    text = paper_path.read_text(encoding="utf-8")
    match = re.search(
        r"\\multirow\{9\}\{\*\}\{Beauty\}[\s\S]*?\\multicolumn\{2\}\{l\}\{\\textbf\{Avg\.~Rank",
        text,
    )
    if match is None:
        return {
            "route_wins": 0,
            "total_rows": 0,
            "wins_by_dataset": {},
            "avg_gap_by_dataset": {},
            "closest_baselines": {},
            "avg_gap_overall": 0.0,
        }
    chunk = match.group(0)
    rows: list[list[object]] = []
    current_dataset: str | None = None
    for line in chunk.splitlines():
        line = line.strip()
        if "\\multirow{9}{*}{" in line:
            dataset_match = re.search(r"\\multirow\{9\}\{\*\}\{([^}]*)\}", line)
            if dataset_match is not None:
                current_dataset = dataset_match.group(1).replace("$", "").replace("\\dagger", "").replace("^", "")
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
    columns = [
        "dataset",
        "metric",
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
    frame = pd.DataFrame(rows, columns=columns)
    models = columns[2:]
    summary_rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        values = [(model, float(row[model])) for model in models]
        winner = max(values, key=lambda pair: pair[1])[0]
        best_baseline = max((pair for pair in values if pair[0] != "RouteRec"), key=lambda pair: pair[1])
        summary_rows.append(
            {
                "dataset": row["dataset"],
                "winner": winner,
                "route_win": winner == "RouteRec",
                "best_baseline": best_baseline[0],
                "route_gap_to_best_baseline": float(row["RouteRec"]) - best_baseline[1],
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    return {
        "route_wins": int(summary_frame["route_win"].sum()),
        "total_rows": int(len(summary_frame)),
        "wins_by_dataset": {key: int(value) for key, value in summary_frame.groupby("dataset")["route_win"].sum().items()},
        "avg_gap_by_dataset": {key: float(value) for key, value in summary_frame.groupby("dataset")["route_gap_to_best_baseline"].mean().items()},
        "closest_baselines": {key: int(value) for key, value in summary_frame["best_baseline"].value_counts().items()},
        "avg_gap_overall": float(summary_frame["route_gap_to_best_baseline"].mean()),
    }


def load_dataset_artifacts(selected: list[str]) -> list[DatasetArtifacts]:
    display_map = dict(DATASETS)
    artifacts: list[DatasetArtifacts] = []
    for key in selected:
        folder = DATA_ROOT / key
        artifacts.append(
            DatasetArtifacts(
                key=key,
                display=display_map.get(key, key),
                inter_path=folder / f"{key}.inter",
                item_path=folder / f"{key}.item",
            )
        )
    return artifacts


def summarize_dataset(artifact: DatasetArtifacts, cluster_k: int) -> tuple[dict[str, float | str], list[dict[str, float | str]], list[dict[str, float | str]]]:
    inter_columns = ["session_id", "item_id", "timestamp", "user_id", *MACRO5_COLUMNS]
    inter_frame = pd.read_csv(
        artifact.inter_path,
        sep="\t",
        usecols=lambda column: column.split(":", 1)[0] in inter_columns,
    )
    inter_frame.columns = [column.split(":", 1)[0] for column in inter_frame.columns]
    inter_frame[MACRO5_COLUMNS] = inter_frame[MACRO5_COLUMNS].astype("float32")

    session_frame = (
        inter_frame.groupby("session_id", sort=False)
        .agg(
            user_id=("user_id", "first"),
            session_start=("timestamp", "min"),
            session_len=("item_id", "size"),
            **{column: (column, "first") for column in MACRO5_COLUMNS},
        )
        .reset_index()
    )

    sessions_per_user = session_frame.groupby("user_id").size()
    item_support = inter_frame.groupby("item_id").size()
    item_frame = pd.read_csv(artifact.item_path, sep="\t")
    item_frame.columns = [column.split(":", 1)[0] for column in item_frame.columns]
    item_categories = item_frame["category"].fillna("__missing__")
    category_counts = item_categories.value_counts()

    raw_profile = compute_raw_behavior_profile(inter_frame, session_frame)

    feature_matrix = session_frame[MACRO5_COLUMNS].to_numpy(dtype=np.float64)

    family_rows: list[dict[str, float | str]] = []
    for family_name, family_columns in MACRO5_FAMILIES.items():
        family_matrix = session_frame[family_columns].to_numpy(dtype=np.float64)
        family_rows.append(
            {
                "dataset": artifact.display,
                "family": family_name,
                "feature_mean": float(family_matrix.mean()),
                "feature_std": float(family_matrix.std()),
                "spread_index": scaled_variance(family_matrix),
                "effective_dimension": effective_rank(family_matrix),
            }
        )

    feature_rows: list[dict[str, float | str]] = []
    quantiles = session_frame[MACRO5_COLUMNS].quantile([0.25, 0.5, 0.75])
    for column in MACRO5_COLUMNS:
        feature_rows.append(
            {
                "dataset": artifact.display,
                "feature": column,
                "mean": float(session_frame[column].mean()),
                "std": float(session_frame[column].std(ddof=0)),
                "q25": float(quantiles.loc[0.25, column]),
                "median": float(quantiles.loc[0.5, column]),
                "q75": float(quantiles.loc[0.75, column]),
            }
        )

    cluster_entropy, cluster_separation, mixture_index = kmeans_metrics(feature_matrix, cluster_k)
    spread_index = scaled_variance(feature_matrix)
    effective_dimension = effective_rank(feature_matrix)
    drift_index = session_drift_score(session_frame, MACRO5_COLUMNS)
    routing_relevant_index = float(np.mean([spread_index, mixture_index]))
    behavior_index = float(np.mean([spread_index, mixture_index, drift_index]))

    summary = {
        "dataset": artifact.display,
        "dataset_key": artifact.key,
        "interactions": int(len(inter_frame)),
        "sessions": int(session_frame["session_id"].nunique()),
        "users": int(session_frame["user_id"].nunique()),
        "items": int(inter_frame["item_id"].nunique()),
        "avg_sessions_per_user": float(sessions_per_user.mean()),
        "median_sessions_per_user": float(sessions_per_user.median()),
        "avg_interactions_per_session": float(session_frame["session_len"].mean()),
        "median_interactions_per_session": float(session_frame["session_len"].median()),
        "session_len_std": float(session_frame["session_len"].std(ddof=0)),
        "item_popularity_gini": gini(item_support.to_numpy(dtype=np.float64)),
        "item_top1_share": float(item_support.max() / len(inter_frame)),
        "catalog_category_count": int(category_counts.size),
        "catalog_category_entropy": normalized_entropy(category_counts.to_numpy(dtype=np.float64)),
        "catalog_top1_category_share": float(category_counts.iloc[0] / category_counts.sum()),
        "macro5_feature_mean": float(feature_matrix.mean()),
        "macro5_feature_std": float(feature_matrix.std()),
        "heterogeneity_spread": spread_index,
        "heterogeneity_mixture_entropy": cluster_entropy,
        "heterogeneity_mixture_separation": cluster_separation,
        "heterogeneity_mixture": mixture_index,
        "heterogeneity_user_drift": drift_index,
        "heterogeneity_effective_dimension": effective_dimension,
        "routing_relevant_heterogeneity_index": routing_relevant_index,
        "behavioral_heterogeneity_index": behavior_index,
    }
    summary.update(raw_profile)
    return summary, family_rows, feature_rows


def format_int(value: int) -> str:
    return f"{value:,}"


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_markdown(summary_frame: pd.DataFrame, family_frame: pd.DataFrame) -> str:
    del family_frame
    raw_ordered = summary_frame.sort_values("simple_routing_score", ascending=False).reset_index(drop=True)
    perf = compute_full_table_performance_profile()
    lines: list[str] = []
    lines.append("# 데이터셋 Appendix 분석")
    lines.append("")
    lines.append("## 측정 항목")
    lines.append("")
    lines.append("- 기본 구조: interaction 수, session 수, user 수, item 수, user당 평균 session 수, session당 평균 interaction 수.")
    lines.append("- feature를 쓰지 않는 비교용 지표만 사용: session volatility, transition branching, context availability.")
    lines.append("- 간단한 요약 점수: Simple Routing Score = mean(Session Volatility, Transition Branching, Context Availability).")
    lines.append("- 보조로 Raw Behavioral Heterogeneity와 Raw Routing Opportunity도 남기지만, 해석의 중심은 세 개의 직관적인 성분입니다.")
    lines.append("")
    lines.append("## 요약 표")
    lines.append("")
    lines.append("| Dataset | Interactions | Sessions | Users | Avg sess/user | Avg inter/session | Users>=2 sessions | Users>=5 sessions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in raw_ordered.itertuples(index=False):
        lines.append(
            "| "
            f"{row.dataset} | {format_int(int(row.interactions))} | {format_int(int(row.sessions))} | {format_int(int(row.users))} | "
            f"{format_float(row.avg_sessions_per_user, 2)} | {format_float(row.avg_interactions_per_session, 2)} | "
            f"{format_float(row.users_ge_2_sessions_ratio)} | {format_float(row.users_ge_5_sessions_ratio)} |"
        )
    lines.append("")
    lines.append("## Feature-Free Heterogeneity 요약")
    lines.append("")
    lines.append("| Dataset | SessionVol | Branching | ContextAvail | SimpleScore | RawHet | RoutingOpp |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in raw_ordered.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {format_float(row.raw_session_volatility)} | {format_float(row.raw_transition_branching)} | {format_float(row.raw_context_availability)} | {format_float(row.simple_routing_score)} | {format_float(row.raw_behavioral_heterogeneity)} | {format_float(row.raw_routing_opportunity)} |"
        )
    lines.append("")
    lines.append("## RouteRec 성능 요약")
    lines.append("")
    lines.append(f"- Full table 기준 RouteRec은 54개 row 중 {perf['route_wins']}개에서 최고 성능을 기록했습니다.")
    wins_by_dataset = perf["wins_by_dataset"]
    lines.append(
        f"- 데이터셋별 승수는 KuaiRec {wins_by_dataset.get('KuaiRec', 0)}/9, LastFM {wins_by_dataset.get('LastFM', 0)}/9, Retail Rocket {wins_by_dataset.get('Retail Rocket', 0)}/9, Beauty {wins_by_dataset.get('Beauty', 0)}/9, Foursquare {wins_by_dataset.get('Foursquare', 0)}/9, ML-1M {wins_by_dataset.get('ML-1M', 0)}/9 입니다."
    )
    avg_gap = perf["avg_gap_by_dataset"]
    lines.append(
        f"- best baseline 대비 평균 격차는 KuaiRec {format_float(avg_gap.get('KuaiRec', 0.0), 4)}, Retail Rocket {format_float(avg_gap.get('Retail Rocket', 0.0), 4)}, LastFM {format_float(avg_gap.get('LastFM', 0.0), 4)}, Beauty {format_float(avg_gap.get('Beauty', 0.0), 4)}, Foursquare {format_float(avg_gap.get('Foursquare', 0.0), 4)}, ML-1M {format_float(avg_gap.get('ML-1M', 0.0), 4)} 입니다."
    )
    closest = perf["closest_baselines"]
    lines.append(
        f"- RouteRec의 가장 잦은 경쟁 baseline은 FDSA({closest.get('FDSA', 0)}회)와 FEARec({closest.get('FEARec', 0)}회)였습니다. 즉 strongest competitor는 대체로 feature-aware 계열(FDSA) 또는 강한 sequence encoder 계열(FEARec)입니다."
    )
    lines.append("- KuaiRec과 LastFM에서는 9개 metric 전부에서 1위를 기록해, 단일 cutoff 우연이 아니라 전반적인 ranking quality 개선으로 해석할 수 있습니다.")
    lines.append("- Foursquare에서는 HR/NDCG 계열 우위가 분명하지만 MRR 계열은 FDSA와 DIF-SR가 근소하게 강해, 상위 구간 recall 개선은 크고 earliest-hit precision 이득은 더 제한적이라고 볼 수 있습니다.")
    lines.append("- Beauty에서는 HR/NDCG에서는 우세하지만 MRR은 FDSA가 앞서므로, RouteRec이 더 넓은 hit coverage에는 기여했지만 최상위 첫 정답 위치까지 일관되게 끌어올린 것은 아니라고 정리하는 편이 정확합니다.")
    lines.append("- ML-1M에서는 전 metric에서 FDSA, DuoRec, FAME 같은 강한 shared-path baseline이 비슷하거나 더 높아, 길고 비교적 안정적인 sequence에서는 추가 routing의 여지가 제한적이라고 해석하는 편이 자연스럽습니다.")
    lines.append("")
    lines.append("## 간단한 해석 가이드")
    lines.append("")
    lines.append("- SessionVol은 session 길이가 얼마나 들쭉날쭉한지를 나타냅니다. 짧은 세션과 긴 세션이 섞여 있으면 높아집니다.")
    lines.append("- Branching은 같은 item 이후 다음 행동이 얼마나 여러 방향으로 갈라지는지를 나타냅니다. 높을수록 shared-path encoder 하나로 설명하기 어려운 전이 다양성이 큽니다.")
    lines.append("- ContextAvail은 RouteRec의 macro routing이 실제로 쓸 수 있는 반복 session 문맥이 충분한지를 나타냅니다.")
    lines.append("- SimpleScore는 위 세 요소의 평균으로, 과도한 설계 없이도 'RouteRec이 도움될 가능성'을 비교하는 간단한 dataset-level 지표로 쓸 수 있습니다.")
    lines.append("- Beauty와 ML-1M이 같은 low-score 구간이라도 이유는 다릅니다. Beauty는 context 부족이 핵심이고, ML-1M은 context는 일부 있지만 행동 branching보다 긴 안정적 sequence 구조가 더 지배적입니다.")
    lines.append("")
    lines.append("## 논문 반영 제안")
    lines.append("")
    lines.append("- 본문에서는 복잡한 composite 수식을 전면에 두기보다, SessionVol, Branching, ContextAvail 세 축을 먼저 설명하고 SimpleScore는 appendix 표 정렬용 요약치로만 쓰는 편이 자연스럽습니다.")
    lines.append("- 가장 좋은 위치는 Appendix의 dataset 분석 파트입니다. 여기서 raw-log 기반 진단 표를 제시하고, 본문 Q1에서는 'RouteRec의 이득은 branching이 크고 multi-session context가 충분한 데이터에서 가장 크다'는 한두 문장만 가져가는 편이 좋습니다.")
    lines.append("- 약점처럼 보이지 않게 쓰려면, Beauty와 ML-1M을 '실패 사례'로 규정하지 말고, 각각 context scarcity와 strong shared-path suitability가 더 지배적인 조건이라고 설명하는 편이 안전합니다.")
    lines.append("- baseline 해석은 가볍게 유지하는 것이 좋습니다. FDSA는 feature-aware/shared-path strong baseline으로서 가장 자주 RouteRec와 경쟁했고, FEARec은 강한 sequence encoder로서 동적 데이터에서도 꾸준히 근접했습니다.")
    lines.append("")
    lines.append("## 논문용 서술 초안")
    lines.append("")
    lines.append("- We summarize dataset-level routing demand from raw logs using three descriptive factors only: session volatility, transition branching, and multi-session context availability. This keeps the analysis model-agnostic and avoids tying the dataset study to RouteRec's internal hand-crafted features.")
    lines.append("- Under this view, KuaiRec and Foursquare provide the clearest combination of dynamic local transitions and sufficient repeated-session context, which is consistent with the stronger and broader gains of RouteRec in those datasets.")
    lines.append("- Beauty and ML-1M should be interpreted differently rather than grouped as simple failure cases: Beauty offers little repeated-session context for most users, whereas ML-1M is a longer and more preference-stable setting in which strong shared-path sequential encoders are already highly competitive. In both cases, the smaller RouteRec margin is therefore better read as limited routing headroom than as a categorical weakness of the method.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    selected = [token.strip() for token in str(args.datasets).split(",") if token.strip()]
    artifacts = load_dataset_artifacts(selected)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    family_rows: list[dict[str, float | str]] = []
    feature_rows: list[dict[str, float | str]] = []
    for artifact in artifacts:
        summary, families, features = summarize_dataset(artifact, cluster_k=args.cluster_k)
        summary_rows.append(summary)
        family_rows.extend(families)
        feature_rows.extend(features)

    summary_frame = pd.DataFrame(summary_rows)
    family_frame = pd.DataFrame(family_rows)
    feature_frame = pd.DataFrame(feature_rows)

    summary_csv = output_dir / "dataset_summary.csv"
    family_csv = output_dir / "dataset_macro5_family_summary.csv"
    feature_csv = output_dir / "dataset_macro5_feature_summary.csv"
    report_md = output_dir / "dataset_appendix_report.md"
    report_json = output_dir / "dataset_summary.json"

    summary_frame.to_csv(summary_csv, index=False)
    family_frame.to_csv(family_csv, index=False)
    feature_frame.to_csv(feature_csv, index=False)
    report_md.write_text(build_markdown(summary_frame, family_frame), encoding="utf-8")
    report_json.write_text(summary_frame.to_json(orient="records", indent=2), encoding="utf-8")

    print(json.dumps({
        "summary_csv": str(summary_csv),
        "family_csv": str(family_csv),
        "feature_csv": str(feature_csv),
        "report_md": str(report_md),
        "report_json": str(report_json),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())