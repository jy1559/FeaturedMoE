#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from real_final_viz_helpers import DATASET_LABELS, PALETTE, apply_style, clean_axes


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

Q6_SUMMARY_CSV = REPO_ROOT / "experiments/run/artifacts/logs/real_final_ablation/q6/summary.csv"
Q6_DIAG_SUMMARY_CSV = REPO_ROOT / "experiments/run/artifacts/logs/real_final_ablation/q6_diag/summary.csv"
Q6_DIAG_INDEX_CSV = REPO_ROOT / "experiments/run/artifacts/logs/real_final_ablation/q6_diag/q6_diag_case_eval_index.csv"

DATASET_ORDER = [
    "KuaiRecLargeStrictPosV2_0.2",
    "beauty",
    "foursquare",
    "retail_rocket",
    "movielens1m",
    "lastfm0.03",
]
STATE_ORDER = ["original", "pure_union", "permissive_union"]
TIER_ORDER = ["pure", "permissive"]
FAMILY_ORDER = ["memory", "focus", "tempo", "exposure"]


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


def _dataset_sort_key(dataset: str) -> int:
    try:
        return DATASET_ORDER.index(dataset)
    except ValueError:
        return len(DATASET_ORDER)


def _dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def _sep_lambda_from_setting_key(setting_key: object) -> float:
    raw = str(setting_key or "").strip().lower()
    if raw.startswith("sep_0"):
        return 0.0
    return float("nan")


def _load_top_phase1_rows() -> pd.DataFrame:
    summary = pd.read_csv(Q6_SUMMARY_CSV)
    summary = summary[summary["status"].str.lower() == "ok"].copy()
    summary["test_score"] = pd.to_numeric(summary["test_score"], errors="coerce")
    summary = summary.sort_values(["dataset", "test_score"], ascending=[True, False])
    top = summary.groupby("dataset", as_index=False).head(1).copy()

    records: list[dict[str, object]] = []
    for row in top.to_dict("records"):
        phase1_payload = _read_json(row.get("result_path"))
        base_payload = _read_json(row.get("base_result_json"))
        best_params = phase1_payload.get("best_params") or {}
        records.append(
            {
                "dataset": row["dataset"],
                "dataset_label": _dataset_label(str(row["dataset"])),
                "base_rank": int(row["base_rank"]),
                "base_tag": row["base_tag"],
                "phase1_valid_score": _safe_float(row.get("valid_score")),
                "phase1_test_score": _safe_float(row.get("test_score")),
                "phase1_best_valid_mrr20": _safe_float(row.get("best_valid_mrr20")),
                "phase1_test_mrr20": _safe_float(row.get("test_mrr20")),
                "base_test_mrr20": _first_metric(
                    base_payload,
                    ("test_result", "mrr@20"),
                    ("test_special_metrics", "overall_seen_target", "mrr@20"),
                    ("test_special_metrics", "overall", "mrr@20"),
                ),
                "best_learning_rate": _safe_float(best_params.get("learning_rate")),
                "best_route_consistency_lambda": _safe_float(best_params.get("route_consistency_lambda")),
                "best_route_separation_lambda": _safe_float(best_params.get("route_separation_lambda")),
                "phase1_result_path": row.get("result_path"),
                "base_result_json": row.get("base_result_json"),
            }
        )

    phase1 = pd.DataFrame(records)
    phase1["phase1_minus_base_test_mrr20"] = phase1["phase1_test_mrr20"] - phase1["base_test_mrr20"]
    phase1 = phase1.sort_values("dataset", key=lambda s: s.map(_dataset_sort_key)).reset_index(drop=True)
    return phase1


def _load_top_archived_diag_rows() -> pd.DataFrame:
    summary = pd.read_csv(Q6_DIAG_SUMMARY_CSV)
    index_df = pd.read_csv(Q6_DIAG_INDEX_CSV)
    summary = summary[summary["status"].str.lower() == "ok"].copy()
    summary["test_score"] = pd.to_numeric(summary["test_score"], errors="coerce")
    summary = summary.sort_values(["dataset", "test_score"], ascending=[True, False])
    top = summary.groupby("dataset", as_index=False).head(1).copy()
    top = top.merge(
        index_df[["dataset", "base_rank", "case_eval_export_dir"]],
        on=["dataset", "base_rank"],
        how="left",
    )

    records: list[dict[str, object]] = []
    for row in top.to_dict("records"):
        payload = _read_json(row.get("result_path"))
        diag_sep = _first_metric(
            payload,
            ("best_params", "route_separation_lambda"),
            ("route_separation_lambda",),
            ("config", "route_separation_lambda"),
        )
        if np.isnan(diag_sep):
            diag_sep = _sep_lambda_from_setting_key(row.get("setting_key"))
        records.append(
            {
                "dataset": row["dataset"],
                "diag_base_rank": int(row["base_rank"]),
                "diag_test_score": _safe_float(row.get("test_score")),
                "diag_test_mrr20": _safe_float(row.get("test_mrr20")),
                "diag_learning_rate": _first_metric(
                    payload,
                    ("best_params", "learning_rate"),
                    ("learning_rate",),
                    ("config", "learning_rate"),
                ),
                "diag_route_consistency_lambda": _first_metric(
                    payload,
                    ("best_params", "route_consistency_lambda"),
                    ("route_consistency_lambda",),
                    ("config", "route_consistency_lambda"),
                ),
                "diag_route_separation_lambda": diag_sep,
                "diag_case_eval_export_dir": row.get("case_eval_export_dir"),
                "diag_result_path": row.get("result_path"),
            }
        )

    diag = pd.DataFrame(records)
    diag = diag.sort_values("dataset", key=lambda s: s.map(_dataset_sort_key)).reset_index(drop=True)
    return diag


def _build_case_state_metrics(diag_top: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in diag_top.to_dict("records"):
        table_dir = Path(str(item.get("diag_case_eval_export_dir", "")))
        perf_path = table_dir / "case_eval_performance.csv"
        if not perf_path.exists():
            continue
        perf = pd.read_csv(perf_path)
        perf = perf[perf["status"].str.lower() == "ok"].copy()
        perf = perf[perf["source_result_json"].notna()].copy()
        perf = perf[perf["source_result_json"] != ""].copy()
        perf = perf[perf["test_seen_mrr20"].notna()].copy()
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
                    "dataset_label": _dataset_label(str(item["dataset"])),
                    "state": state,
                    "test_seen_mrr20": _safe_float(row.get("test_seen_mrr20")),
                    "test_seen_ndcg20": _safe_float(row.get("test_seen_ndcg20")),
                    "test_seen_hit10": _safe_float(row.get("test_seen_hit10")),
                }
            )
    out = pd.DataFrame(rows)
    out["state"] = pd.Categorical(out["state"], categories=STATE_ORDER, ordered=True)
    out = out.sort_values(["dataset", "state"], key=lambda s: s.map(_dataset_sort_key) if s.name == "dataset" else s)
    return out.reset_index(drop=True)


def _select_beauty_case_dir(diag_top: pd.DataFrame) -> Path:
    beauty = diag_top[diag_top["dataset"] == "beauty"]
    if beauty.empty:
        raise FileNotFoundError("No archived q6_diag beauty case-eval export found")
    return Path(str(beauty.iloc[0]["diag_case_eval_export_dir"]))


def _build_group_shift(diag_top: pd.DataFrame) -> pd.DataFrame:
    table_dir = _select_beauty_case_dir(diag_top)
    routing = pd.read_csv(table_dir / "case_eval_routing_profile.csv")
    routing = routing[(routing["status"].str.lower() == "ok") & (routing["eval_split"].str.lower() == "test")].copy()
    routing = routing[(routing["scope"] == "tier_group") & (routing["stage_name"].str.lower() == "macro")].copy()
    routing = routing[routing["tier"].isin(TIER_ORDER)].copy()
    routing = routing[routing["selected_polarity"].isin(["plus", "minus"])].copy()
    routing["selected_family"] = routing["selected_family"].astype(str).str.lower()
    routing["routed_family"] = routing["routed_family"].astype(str).str.lower()

    grouped = (
        routing.groupby(["tier", "selected_family", "routed_family", "selected_polarity"], as_index=False)["usage_share"]
        .mean()
    )
    pivot = grouped.pivot_table(
        index=["tier", "selected_family", "routed_family"],
        columns="selected_polarity",
        values="usage_share",
        fill_value=0.0,
    ).reset_index()
    pivot["usage_plus"] = pd.to_numeric(pivot.get("plus", 0.0), errors="coerce").fillna(0.0)
    pivot["usage_minus"] = pd.to_numeric(pivot.get("minus", 0.0), errors="coerce").fillna(0.0)
    pivot["usage_delta"] = pivot["usage_plus"] - pivot["usage_minus"]
    return pivot.sort_values(["tier", "selected_family", "routed_family"]).reset_index(drop=True)


def _build_expert_shift(diag_top: pd.DataFrame) -> pd.DataFrame:
    table_dir = _select_beauty_case_dir(diag_top)
    experts = pd.read_csv(table_dir / "case_eval_expert_profile.csv")
    experts = experts[(experts["status"].str.lower() == "ok") & (experts["eval_split"].str.lower() == "test")].copy()
    experts = experts[(experts["scope"] == "tier_group") & (experts["stage_name"].str.lower() == "macro")].copy()
    experts = experts[experts["tier"].isin(TIER_ORDER)].copy()
    experts = experts[experts["selected_polarity"].isin(["plus", "minus"])].copy()
    experts["selected_family"] = experts["selected_family"].astype(str).str.lower()
    experts["expert_name"] = experts["expert_name"].astype(str)
    experts["expert_index"] = pd.to_numeric(experts["expert_index"], errors="coerce")

    grouped = (
        experts.groupby(["tier", "selected_family", "expert_name", "expert_index", "selected_polarity"], as_index=False)["usage_share"]
        .mean()
    )
    pivot = grouped.pivot_table(
        index=["tier", "selected_family", "expert_name", "expert_index"],
        columns="selected_polarity",
        values="usage_share",
        fill_value=0.0,
    ).reset_index()
    pivot["usage_plus"] = pd.to_numeric(pivot.get("plus", 0.0), errors="coerce").fillna(0.0)
    pivot["usage_minus"] = pd.to_numeric(pivot.get("minus", 0.0), errors="coerce").fillna(0.0)
    pivot["usage_delta"] = pivot["usage_plus"] - pivot["usage_minus"]
    return pivot.sort_values(["tier", "selected_family", "expert_index"]).reset_index(drop=True)


def _plot_phase1_overview(summary_df: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6), constrained_layout=True)

    plot_df = summary_df.copy()
    plot_df["dataset_label"] = plot_df["dataset"].map(_dataset_label)
    x = np.arange(len(plot_df))
    width = 0.34

    axes[0].bar(x - width / 2, plot_df["base_test_mrr20"], width=width, color=PALETTE["muted"], alpha=0.7, label="Base")
    axes[0].bar(x + width / 2, plot_df["phase1_test_mrr20"], width=width, color=PALETTE["route"], alpha=0.9, label="Q6 Phase-1 best")
    axes[0].set_xticks(x, plot_df["dataset_label"], rotation=0)
    axes[0].set_ylabel("Test MRR@20")
    axes[0].set_title("Phase-1 Best vs Source Base")
    clean_axes(axes[0])
    axes[0].legend(loc="upper left", frameon=True)
    for idx, delta in enumerate(plot_df["phase1_minus_base_test_mrr20"]):
        axes[0].text(x[idx], max(plot_df.iloc[idx]["base_test_mrr20"], plot_df.iloc[idx]["phase1_test_mrr20"]) + 0.003, f"{delta:+.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x - width / 2, plot_df["phase1_test_score"], width=width, color=PALETTE["route"], alpha=0.9, label="Phase-1 best")
    axes[1].bar(x + width / 2, plot_df["diag_test_score"], width=width, color=PALETTE["orange"], alpha=0.8, label="Archived Phase-2 diag")
    axes[1].set_xticks(x, plot_df["dataset_label"], rotation=0)
    axes[1].set_ylabel("Seen-target score")
    axes[1].set_title("Phase-1 Best vs Archived Phase-2")
    clean_axes(axes[1])
    axes[1].legend(loc="upper left", frameon=True)

    fig.suptitle("Q6 Route Separation Overview", fontsize=14, fontweight="semibold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_archived_state_metrics(state_df: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)
    metrics = [("test_seen_mrr20", "Seen MRR@20"), ("test_seen_ndcg20", "Seen NDCG@20")]
    state_palette = {
        "original": PALETTE["muted"],
        "pure_union": PALETTE["blue"],
        "permissive_union": PALETTE["orange"],
    }

    for ax, (metric, title) in zip(axes, metrics):
        pivot = state_df.pivot(index="dataset_label", columns="state", values=metric).reindex([_dataset_label(d) for d in DATASET_ORDER])
        pivot = pivot.reindex(columns=STATE_ORDER)
        x = np.arange(len(pivot.index))
        width = 0.24
        for offset, state in zip([-width, 0.0, width], STATE_ORDER):
            ax.bar(x + offset, pivot[state], width=width, label=state.replace("_", " ").title(), color=state_palette[state], alpha=0.86)
        ax.set_xticks(x, pivot.index, rotation=0)
        ax.set_title(title)
        ax.set_ylabel(title)
        clean_axes(ax)
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("Archived Q6 Diag State Metrics", fontsize=14, fontweight="semibold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _heatmap_frame(df: pd.DataFrame, tier: str, value_col: str, row_col: str, col_col: str) -> pd.DataFrame:
    frame = df[df["tier"] == tier].copy()
    heat = frame.pivot(index=row_col, columns=col_col, values=value_col).fillna(0.0)
    return heat


def _plot_group_shift(group_df: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    for ax, tier in zip(axes, TIER_ORDER):
        heat = _heatmap_frame(group_df, tier, "usage_delta", "selected_family", "routed_family")
        heat = heat.reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER).fillna(0.0)
        sns.heatmap(heat, ax=ax, cmap="RdBu_r", center=0.0, vmin=-0.12, vmax=0.12, annot=True, fmt=".02f", cbar=(tier == "permissive"))
        ax.set_title(f"Beauty {tier.title()} plus-minus group shift")
        ax.set_xlabel("Routed family")
        ax.set_ylabel("Selected family")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_expert_shift(expert_df: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 7.2), constrained_layout=True)
    for ax, tier in zip(axes, TIER_ORDER):
        heat = _heatmap_frame(expert_df, tier, "usage_delta", "selected_family", "expert_name")
        expert_order = (
            expert_df[expert_df["tier"] == tier]
            .sort_values("expert_index")["expert_name"]
            .drop_duplicates()
            .tolist()
        )
        heat = heat.reindex(index=FAMILY_ORDER, columns=expert_order).fillna(0.0)
        sns.heatmap(heat, ax=ax, cmap="RdBu_r", center=0.0, vmin=-0.06, vmax=0.06, annot=False, cbar=(tier == "permissive"))
        ax.set_title(f"Beauty {tier.title()} plus-minus expert-slot shift")
        ax.set_xlabel("Expert slot")
        ax.set_ylabel("Selected family")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    phase1 = _load_top_phase1_rows()
    diag = _load_top_archived_diag_rows()
    summary = phase1.merge(diag, on="dataset", how="left")
    summary["diag_matches_phase1"] = (
        np.isclose(summary["best_route_consistency_lambda"], summary["diag_route_consistency_lambda"], equal_nan=False)
        & np.isclose(summary["best_route_separation_lambda"], summary["diag_route_separation_lambda"], equal_nan=False)
    )
    summary["phase2_caveat"] = np.where(
        summary["diag_matches_phase1"],
        "archived diag matches phase1 tuned lambdas",
        "archived diag rerun does not preserve phase1 tuned lambdas",
    )

    state_metrics = _build_case_state_metrics(diag)
    group_shift = _build_group_shift(diag)
    expert_shift = _build_expert_shift(diag)

    summary.to_csv(DATA_DIR / "q6_phase1_archived_summary.csv", index=False)
    state_metrics.to_csv(DATA_DIR / "q6_archived_state_metrics.csv", index=False)
    group_shift.to_csv(DATA_DIR / "q6_beauty_group_shift.csv", index=False)
    expert_shift.to_csv(DATA_DIR / "q6_beauty_expert_shift.csv", index=False)

    _plot_phase1_overview(summary, FIG_DIR / "q6_phase1_overview.png")
    _plot_archived_state_metrics(state_metrics, FIG_DIR / "q6_archived_state_metrics.png")
    _plot_group_shift(group_shift, FIG_DIR / "q6_beauty_group_shift.png")
    _plot_expert_shift(expert_shift, FIG_DIR / "q6_beauty_expert_shift.png")

    print("Q6 Phase-1 best summary")
    print(summary[[
        "dataset_label",
        "base_rank",
        "phase1_test_score",
        "phase1_test_mrr20",
        "phase1_minus_base_test_mrr20",
        "best_route_consistency_lambda",
        "best_route_separation_lambda",
        "diag_route_consistency_lambda",
        "diag_route_separation_lambda",
        "phase2_caveat",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())