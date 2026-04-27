#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from real_final_viz_helpers import DATASET_LABELS, PALETTE, apply_style, clean_axes


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

PHASE1_SUMMARY_CSV = REPO_ROOT / "experiments/run/artifacts/logs/real_final_ablation/sep_main/summary.csv"
DIAG_SUMMARY_CSV = REPO_ROOT / "experiments/run/artifacts/logs/real_final_ablation/sep_main_diag/summary.csv"
DIAG_INDEX_CSV = REPO_ROOT / "experiments/run/artifacts/logs/real_final_ablation/sep_main_diag/sep_main_diag_case_eval_index.csv"

DATASET_ORDER = ["KuaiRecLargeStrictPosV2_0.2", "foursquare"]
STATE_ORDER = ["original", "pure_union", "permissive_union"]
TIER_ORDER = ["pure", "permissive"]
FAMILY_ORDER = ["memory", "focus", "tempo", "exposure"]
SELECTION_ORDER = ["best", "high_sep"]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    payload_path = Path(path)
    if not payload_path.exists():
        return {}
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def _selection_kind(setting_key: object) -> str:
    raw = str(setting_key or "").strip().lower()
    if raw.startswith("high_sep"):
        return "high_sep"
    return "best"


def _nested_value(payload: dict, path: tuple[str, ...]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_metric(payload: dict, *paths: tuple[str, ...]) -> float:
    for path in paths:
        value = _nested_value(payload, path)
        if value is None:
            continue
        metric = _safe_float(value)
        if not np.isnan(metric):
            return metric
    return float("nan")


def _load_diag_summary() -> pd.DataFrame:
    diag = pd.read_csv(DIAG_SUMMARY_CSV)
    diag = diag[diag["status"].str.lower() == "ok"].copy()
    diag["selection_kind"] = diag["setting_key"].map(_selection_kind)
    diag["test_score"] = pd.to_numeric(diag["test_score"], errors="coerce")
    diag["test_mrr20"] = pd.to_numeric(diag["test_mrr20"], errors="coerce")
    diag["dataset_label"] = diag["dataset"].map(_dataset_label)

    params_rows: list[dict[str, object]] = []
    for row in diag.to_dict("records"):
        payload = _read_json(row.get("result_path"))
        params_rows.append(
            {
                "job_id": row["job_id"],
                "diag_learning_rate": _first_metric(payload, ("best_params", "learning_rate"), ("config", "learning_rate")),
                "diag_weight_decay": _first_metric(payload, ("best_params", "weight_decay"), ("config", "weight_decay")),
                "diag_route_consistency_lambda": _first_metric(payload, ("best_params", "route_consistency_lambda"), ("config", "route_consistency_lambda")),
                "diag_route_separation_lambda": _first_metric(payload, ("best_params", "route_separation_lambda"), ("config", "route_separation_lambda")),
            }
        )
    params_df = pd.DataFrame(params_rows)
    diag = diag.merge(params_df, on="job_id", how="left")

    phase1 = pd.read_csv(PHASE1_SUMMARY_CSV)
    phase1 = phase1[["job_id", "dataset", "base_rank", "base_tag", "test_score", "test_mrr20"]].rename(
        columns={
            "job_id": "parent_job_id",
            "test_score": "phase1_test_score",
            "test_mrr20": "phase1_test_mrr20",
        }
    )
    diag = diag.merge(phase1, on=["parent_job_id", "dataset", "base_rank", "base_tag"], how="left")

    index_df = pd.read_csv(DIAG_INDEX_CSV)
    diag = diag.merge(index_df[["result_path", "case_eval_export_dir"]], on="result_path", how="left")
    dataset_order = {dataset: idx for idx, dataset in enumerate(DATASET_ORDER)}
    selection_order = {kind: idx for idx, kind in enumerate(SELECTION_ORDER)}
    return diag.sort_values(
        ["dataset", "selection_kind"],
        key=lambda s: s.map(dataset_order) if s.name == "dataset" else s.map(selection_order),
    ).reset_index(drop=True)


def _build_state_metrics(diag_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in diag_df.to_dict("records"):
        table_dir = Path(str(item.get("case_eval_export_dir", "")))
        perf_path = table_dir / "case_eval_performance.csv"
        if not perf_path.exists():
            continue
        perf = pd.read_csv(perf_path)
        perf = perf[(perf["status"].str.lower() == "ok") & perf["test_seen_mrr20"].notna()].copy()
        for row in perf.to_dict("records"):
            scope = str(row.get("scope", ""))
            tier = str(row.get("tier", ""))
            if scope == "original":
                state = "original"
            elif scope == "tier_union" and tier == "pure":
                state = "pure_union"
            elif scope == "tier_union" and tier == "permissive":
                state = "permissive_union"
            else:
                continue
            rows.append(
                {
                    "dataset": item["dataset"],
                    "dataset_label": item["dataset_label"],
                    "selection_kind": item["selection_kind"],
                    "state": state,
                    "test_seen_mrr20": _safe_float(row.get("test_seen_mrr20")),
                    "test_seen_ndcg20": _safe_float(row.get("test_seen_ndcg20")),
                    "test_seen_hit10": _safe_float(row.get("test_seen_hit10")),
                }
            )
    return pd.DataFrame(rows)


def _build_group_shift(diag_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in diag_df.to_dict("records"):
        table_dir = Path(str(item.get("case_eval_export_dir", "")))
        profile_path = table_dir / "case_eval_routing_profile.csv"
        if not profile_path.exists():
            continue
        routing = pd.read_csv(profile_path)
        routing = routing[(routing["status"].str.lower() == "ok") & (routing["eval_split"].str.lower() == "test")].copy()
        routing = routing[(routing["scope"] == "tier_group") & (routing["stage_name"].str.lower() == "macro")].copy()
        routing = routing[routing["tier"].isin(TIER_ORDER) & routing["selected_polarity"].isin(["plus", "minus"])].copy()
        routing["selected_family"] = routing["selected_family"].astype(str).str.lower()
        routing["routed_family"] = routing["routed_family"].astype(str).str.lower()
        grouped = routing.groupby(["tier", "selected_family", "routed_family", "selected_polarity"], as_index=False)["usage_share"].mean()
        pivot = grouped.pivot_table(
            index=["tier", "selected_family", "routed_family"],
            columns="selected_polarity",
            values="usage_share",
            fill_value=0.0,
        ).reset_index()
        pivot["usage_plus"] = pd.to_numeric(pivot.get("plus", 0.0), errors="coerce").fillna(0.0)
        pivot["usage_minus"] = pd.to_numeric(pivot.get("minus", 0.0), errors="coerce").fillna(0.0)
        pivot["usage_delta"] = pivot["usage_plus"] - pivot["usage_minus"]
        pivot["dataset"] = item["dataset"]
        pivot["selection_kind"] = item["selection_kind"]
        rows.extend(pivot.to_dict("records"))
    return pd.DataFrame(rows)


def _build_expert_shift(diag_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in diag_df.to_dict("records"):
        table_dir = Path(str(item.get("case_eval_export_dir", "")))
        profile_path = table_dir / "case_eval_expert_profile.csv"
        if not profile_path.exists():
            continue
        experts = pd.read_csv(profile_path)
        experts = experts[(experts["status"].str.lower() == "ok") & (experts["eval_split"].str.lower() == "test")].copy()
        experts = experts[(experts["scope"] == "tier_group") & (experts["stage_name"].str.lower() == "macro")].copy()
        experts = experts[experts["tier"].isin(TIER_ORDER) & experts["selected_polarity"].isin(["plus", "minus"])].copy()
        experts["selected_family"] = experts["selected_family"].astype(str).str.lower()
        experts["expert_name"] = experts["expert_name"].astype(str)
        experts["expert_index"] = pd.to_numeric(experts["expert_index"], errors="coerce")
        grouped = experts.groupby(["tier", "selected_family", "expert_name", "expert_index", "selected_polarity"], as_index=False)["usage_share"].mean()
        pivot = grouped.pivot_table(
            index=["tier", "selected_family", "expert_name", "expert_index"],
            columns="selected_polarity",
            values="usage_share",
            fill_value=0.0,
        ).reset_index()
        pivot["usage_plus"] = pd.to_numeric(pivot.get("plus", 0.0), errors="coerce").fillna(0.0)
        pivot["usage_minus"] = pd.to_numeric(pivot.get("minus", 0.0), errors="coerce").fillna(0.0)
        pivot["usage_delta"] = pivot["usage_plus"] - pivot["usage_minus"]
        pivot["dataset"] = item["dataset"]
        pivot["selection_kind"] = item["selection_kind"]
        rows.extend(pivot.to_dict("records"))
    return pd.DataFrame(rows)


def _plot_overview(diag_df: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)
    score_pivot = diag_df.pivot(index="dataset_label", columns="selection_kind", values="test_score").reindex([_dataset_label(d) for d in DATASET_ORDER])
    sep_pivot = diag_df.pivot(index="dataset_label", columns="selection_kind", values="diag_route_separation_lambda").reindex([_dataset_label(d) for d in DATASET_ORDER])
    x = np.arange(len(score_pivot.index))
    width = 0.33
    for ax, pivot, title, ylabel in [
        (axes[0], score_pivot, "Diag Seen-target Score", "Seen-target score"),
        (axes[1], sep_pivot, "Selected Separation Lambda", "Route separation lambda"),
    ]:
        ax.bar(x - width / 2, pivot.get("best"), width=width, label="Best", color=PALETTE["route"], alpha=0.9)
        ax.bar(x + width / 2, pivot.get("high_sep"), width=width, label="High-sep", color=PALETTE["orange"], alpha=0.85)
        ax.set_xticks(x, score_pivot.index, rotation=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        clean_axes(ax)
    axes[0].legend(loc="upper left", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_state_metrics(state_df: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(len(DATASET_ORDER), len(SELECTION_ORDER), figsize=(11.4, 7.8), constrained_layout=True)
    state_palette = {"original": PALETTE["muted"], "pure_union": PALETTE["blue"], "permissive_union": PALETTE["orange"]}
    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, selection_kind in enumerate(SELECTION_ORDER):
            ax = axes[row_idx, col_idx]
            frame = state_df[(state_df["dataset"] == dataset) & (state_df["selection_kind"] == selection_kind)].copy()
            frame = frame.set_index("state").reindex(STATE_ORDER)
            values = frame["test_seen_mrr20"].to_numpy(dtype=float) if not frame.empty else np.zeros(len(STATE_ORDER))
            x = np.arange(len(STATE_ORDER))
            ax.bar(x, values, color=[state_palette[state] for state in STATE_ORDER], alpha=0.86)
            ax.set_xticks(x, [state.replace("_", "\n") for state in STATE_ORDER])
            ax.set_title(f"{_dataset_label(dataset)} | {selection_kind}")
            if col_idx == 0:
                ax.set_ylabel("Seen MRR@20")
            clean_axes(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_group_shift(group_df: pd.DataFrame, dataset: str, selection_kind: str, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    for ax, tier in zip(axes, TIER_ORDER):
        frame = group_df[(group_df["dataset"] == dataset) & (group_df["selection_kind"] == selection_kind) & (group_df["tier"] == tier)].copy()
        heat = frame.pivot(index="selected_family", columns="routed_family", values="usage_delta").reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER).fillna(0.0)
        sns.heatmap(heat, ax=ax, cmap="RdBu_r", center=0.0, vmin=-0.12, vmax=0.12, annot=True, fmt=".02f", cbar=(tier == "permissive"))
        ax.set_title(f"{_dataset_label(dataset)} {selection_kind} {tier}")
        ax.set_xlabel("Routed family")
        ax.set_ylabel("Selected family")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_expert_shift(expert_df: pd.DataFrame, dataset: str, selection_kind: str, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 7.0), constrained_layout=True)
    for ax, tier in zip(axes, TIER_ORDER):
        frame = expert_df[(expert_df["dataset"] == dataset) & (expert_df["selection_kind"] == selection_kind) & (expert_df["tier"] == tier)].copy()
        expert_order = frame.sort_values("expert_index")["expert_name"].drop_duplicates().tolist()
        heat = frame.pivot(index="selected_family", columns="expert_name", values="usage_delta").reindex(index=FAMILY_ORDER, columns=expert_order).fillna(0.0)
        sns.heatmap(heat, ax=ax, cmap="RdBu_r", center=0.0, vmin=-0.06, vmax=0.06, annot=False, cbar=(tier == "permissive"))
        ax.set_title(f"{_dataset_label(dataset)} {selection_kind} {tier}")
        ax.set_xlabel("Expert slot")
        ax.set_ylabel("Selected family")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    diag = _load_diag_summary()
    if diag.empty:
        raise RuntimeError("No completed sep_main_diag rows found")
    state_df = _build_state_metrics(diag)
    group_df = _build_group_shift(diag)
    expert_df = _build_expert_shift(diag)

    diag.to_csv(DATA_DIR / "sep_main_diag_summary.csv", index=False)
    state_df.to_csv(DATA_DIR / "sep_main_diag_state_metrics.csv", index=False)
    group_df.to_csv(DATA_DIR / "sep_main_diag_group_shift.csv", index=False)
    expert_df.to_csv(DATA_DIR / "sep_main_diag_expert_shift.csv", index=False)

    _plot_overview(diag, FIG_DIR / "sep_main_diag_overview.png")
    if not state_df.empty:
        _plot_state_metrics(state_df, FIG_DIR / "sep_main_diag_state_metrics.png")

    for dataset in DATASET_ORDER:
        for selection_kind in SELECTION_ORDER:
            if not group_df[(group_df["dataset"] == dataset) & (group_df["selection_kind"] == selection_kind)].empty:
                _plot_group_shift(group_df, dataset, selection_kind, FIG_DIR / f"sep_main_{dataset}_{selection_kind}_group_shift.png")
            if not expert_df[(expert_df["dataset"] == dataset) & (expert_df["selection_kind"] == selection_kind)].empty:
                _plot_expert_shift(expert_df, dataset, selection_kind, FIG_DIR / f"sep_main_{dataset}_{selection_kind}_expert_shift.png")

    print("sep_main diag summary")
    print(
        diag[[
            "dataset_label",
            "selection_kind",
            "phase1_test_score",
            "test_score",
            "diag_route_consistency_lambda",
            "diag_route_separation_lambda",
        ]].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())