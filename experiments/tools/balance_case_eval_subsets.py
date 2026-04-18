from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "beauty",
    "foursquare",
    "movielens1m",
    "retail_rocket",
]

GROUP_ORDER = [
    "memory_plus",
    "memory_minus",
    "focus_plus",
    "focus_minus",
    "tempo_plus",
    "tempo_minus",
    "exposure_plus",
    "exposure_minus",
]


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance case-eval groups by session count and materialize final subset datasets.")
    parser.add_argument("--manifest", default="/workspace/FeaturedMoE/outputs/case_mining_v2/case_manifest.csv")
    parser.add_argument("--src-root", default="/workspace/FeaturedMoE/Datasets/processed/feature_added_v4")
    parser.add_argument("--out-root", default="/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_balanced_v1")
    parser.add_argument("--stats-root", default="/workspace/FeaturedMoE/outputs/case_mining_v2_balanced")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    stats_root = Path(args.stats_root)
    stats_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, low_memory=False)

    session_manifest = (
        manifest.groupby(["dataset", "split", "tier", "group", "session_id"], as_index=False)
        .agg(
            item_id=("item_id", "first"),
            timestamp=("timestamp", "max"),
            family=("family", "first"),
            polarity=("polarity", "first"),
            selection_score=("selection_score", "max"),
            core_score=("core_score", "max"),
            balance_score=("balance_score", "max"),
            contamination_score=("contamination_score", "min"),
            memory_plus=("memory_plus", "mean"),
            focus_plus=("focus_plus", "mean"),
            tempo_plus=("tempo_plus", "mean"),
            exposure_plus=("exposure_plus", "mean"),
        )
    )

    selected_parts = []
    group_stats = []

    for (dataset, split, tier), sdf in session_manifest.groupby(["dataset", "split", "tier"], sort=True):
        counts = sdf.groupby("group")["session_id"].nunique().reindex(GROUP_ORDER, fill_value=0)
        positive_counts = counts[counts > 0]
        quota = int(positive_counts.min()) if not positive_counts.empty else 0

        for group in GROUP_ORDER:
            gdf = sdf[sdf["group"] == group].sort_values(["selection_score", "core_score", "balance_score", "timestamp"], ascending=[False, False, False, False]).copy()
            available = int(gdf["session_id"].nunique())
            selected = gdf.head(quota).copy() if quota > 0 else gdf.head(0).copy()
            selected["balanced_quota"] = quota
            selected["balanced_rank"] = range(1, len(selected) + 1)
            selected_parts.append(selected)

            group_stats.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "tier": tier,
                    "group": group,
                    "available_sessions": available,
                    "selected_sessions": int(len(selected)),
                    "quota": quota,
                    "selection_score_mean": float(selected["selection_score"].mean()) if len(selected) else float("nan"),
                    "selection_score_median": float(selected["selection_score"].median()) if len(selected) else float("nan"),
                    "selection_score_min": float(selected["selection_score"].min()) if len(selected) else float("nan"),
                    "selection_score_max": float(selected["selection_score"].max()) if len(selected) else float("nan"),
                    "core_score_mean": float(selected["core_score"].mean()) if len(selected) else float("nan"),
                    "balance_score_mean": float(selected["balance_score"].mean()) if len(selected) else float("nan"),
                }
            )

    balanced_sessions = pd.concat(selected_parts, ignore_index=True)
    group_stats_df = pd.DataFrame(group_stats)

    baseline_rows = []
    for dataset in DATASETS:
        for split in ["valid", "test"]:
            split_path = src_root / dataset / f"{dataset}.{split}.inter"
            base = pd.read_csv(split_path, sep="\t", usecols=["session_id:token"])
            baseline_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "full_rows": int(len(base)),
                    "full_sessions": int(base["session_id:token"].astype(str).nunique()),
                }
            )
    baseline_df = pd.DataFrame(baseline_rows)

    summary_rows = []
    for (dataset, split, tier), sdf in balanced_sessions.groupby(["dataset", "split", "tier"], sort=True):
        base_row = baseline_df[(baseline_df["dataset"] == dataset) & (baseline_df["split"] == split)].iloc[0]
        unique_sessions = sdf["session_id"].nunique()
        summary_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "tier": tier,
                "quota": int(sdf["balanced_quota"].max()) if len(sdf) else 0,
                "groups_with_support": int((group_stats_df[(group_stats_df["dataset"] == dataset) & (group_stats_df["split"] == split) & (group_stats_df["tier"] == tier)]["available_sessions"] > 0).sum()),
                "selected_session_rows": int(len(sdf)),
                "selected_unique_sessions": int(unique_sessions),
                "full_sessions": int(base_row["full_sessions"]),
                "session_pct_of_split": float(unique_sessions / base_row["full_sessions"]) if base_row["full_sessions"] else 0.0,
                "selection_score_mean": float(sdf["selection_score"].mean()) if len(sdf) else float("nan"),
                "selection_score_median": float(sdf["selection_score"].median()) if len(sdf) else float("nan"),
                "selection_score_min": float(sdf["selection_score"].min()) if len(sdf) else float("nan"),
            }
        )

    balanced_sessions.to_csv(stats_root / "balanced_session_manifest.csv", index=False)
    group_stats_df.to_csv(stats_root / "balanced_group_stats.csv", index=False)
    summary_df.to_csv(stats_root / "balanced_tier_summary.csv", index=False)

    materialized_rows = []
    for dataset in DATASETS:
        src_dir = src_root / dataset
        dst_dir = out_root / dataset
        dst_dir.mkdir(parents=True, exist_ok=True)
        ensure_symlink(src_dir / f"{dataset}.train.inter", dst_dir / f"{dataset}.train.inter")
        ensure_symlink(src_dir / f"{dataset}.item", dst_dir / f"{dataset}.item")
        if (src_dir / f"{dataset}.inter").exists():
            ensure_symlink(src_dir / f"{dataset}.inter", dst_dir / f"{dataset}.inter")
        if (src_dir / "feature_meta_v3.json").exists():
            ensure_symlink(src_dir / "feature_meta_v3.json", dst_dir / "feature_meta_v3.json")

        for split in ["valid", "test"]:
            split_src = pd.read_csv(src_dir / f"{dataset}.{split}.inter", sep="\t")
            split_src["session_id:token"] = split_src["session_id:token"].astype(str)
            keep_sessions = set(balanced_sessions[(balanced_sessions["dataset"] == dataset) & (balanced_sessions["split"] == split)]["session_id"].astype(str).tolist())
            filtered = split_src[split_src["session_id:token"].isin(keep_sessions)].copy()
            filtered.to_csv(dst_dir / f"{dataset}.{split}.inter", sep="\t", index=False)

            base_row = baseline_df[(baseline_df["dataset"] == dataset) & (baseline_df["split"] == split)].iloc[0]
            materialized_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "subset_rows": int(len(filtered)),
                    "subset_sessions": int(filtered["session_id:token"].nunique()),
                    "full_rows": int(base_row["full_rows"]),
                    "full_sessions": int(base_row["full_sessions"]),
                    "row_pct_of_split": float(len(filtered) / base_row["full_rows"]) if base_row["full_rows"] else 0.0,
                    "session_pct_of_split": float(filtered["session_id:token"].nunique() / base_row["full_sessions"]) if base_row["full_sessions"] else 0.0,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    materialized_df = pd.DataFrame(materialized_rows)

    materialized_df.to_csv(stats_root / "balanced_materialized_subset_sizes.csv", index=False)

    md = ["# Balanced Case Eval Summary", ""]
    for dataset in DATASETS:
        md.append(f"## {dataset}")
        md.append("")
        tier_view = summary_df[summary_df["dataset"] == dataset].copy()
        if not tier_view.empty:
            md.append(tier_view.to_markdown(index=False, floatfmt=".4f"))
            md.append("")
        size_view = materialized_df[materialized_df["dataset"] == dataset].copy()
        if not size_view.empty:
            md.append("Materialized subset size:")
            md.append("")
            md.append(size_view.to_markdown(index=False, floatfmt=".4f"))
            md.append("")
    (stats_root / "balanced_summary.md").write_text("\n".join(md))

    print(f"wrote balanced manifest: {stats_root / 'balanced_session_manifest.csv'}")
    print(f"wrote group stats: {stats_root / 'balanced_group_stats.csv'}")
    print(f"wrote tier summary: {stats_root / 'balanced_tier_summary.csv'}")
    print(f"wrote subset sizes: {stats_root / 'balanced_materialized_subset_sizes.csv'}")
    print(f"wrote balanced datasets under: {out_root}")


if __name__ == "__main__":
    main()