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

TIERS = ["pure", "permissive"]
GROUPS = [
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


def materialize_tier_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    dataset: str,
    split_session_map: dict[str, set[str]],
) -> list[dict[str, object]]:
    src_dir = src_root / dataset
    dst_dir = dst_root / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)

    ensure_symlink(src_dir / f"{dataset}.train.inter", dst_dir / f"{dataset}.train.inter")
    ensure_symlink(src_dir / f"{dataset}.item", dst_dir / f"{dataset}.item")
    if (src_dir / f"{dataset}.inter").exists():
        ensure_symlink(src_dir / f"{dataset}.inter", dst_dir / f"{dataset}.inter")
    if (src_dir / "feature_meta_v3.json").exists():
        ensure_symlink(src_dir / "feature_meta_v3.json", dst_dir / "feature_meta_v3.json")

    out_rows: list[dict[str, object]] = []
    for split in ["valid", "test"]:
        split_df = pd.read_csv(src_dir / f"{dataset}.{split}.inter", sep="\t")
        split_df["session_id:token"] = split_df["session_id:token"].astype(str)
        keep_sessions = split_session_map.get(split, set())
        filtered = split_df[split_df["session_id:token"].isin(keep_sessions)].copy()
        filtered.to_csv(dst_dir / f"{dataset}.{split}.inter", sep="\t", index=False)
        out_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "subset_rows": int(len(filtered)),
                "subset_sessions": int(filtered["session_id:token"].nunique()),
            }
        )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize case-eval subsets with capped permissive tier.")
    parser.add_argument("--balanced-manifest", default="/workspace/FeaturedMoE/outputs/case_mining_v2_balanced/balanced_session_manifest.csv")
    parser.add_argument("--src-root", default="/workspace/FeaturedMoE/Datasets/processed/feature_added_v4")
    parser.add_argument("--out-root", default="/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_final_v1")
    parser.add_argument("--stats-root", default="/workspace/FeaturedMoE/outputs/case_mining_v2_final")
    parser.add_argument("--permissive-cap", type=int, default=128)
    args = parser.parse_args()

    balanced_manifest = Path(args.balanced_manifest)
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    stats_root = Path(args.stats_root)
    stats_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(balanced_manifest, low_memory=False)
    manifest["session_id"] = manifest["session_id"].astype(str)

    selected_parts = []
    for tier, tier_df in manifest.groupby("tier", sort=True):
        if tier == "permissive":
            tier_df = tier_df[tier_df["balanced_rank"] <= int(args.permissive_cap)].copy()
        else:
            tier_df = tier_df.copy()
        tier_df["final_cap"] = int(args.permissive_cap) if tier == "permissive" else pd.NA
        selected_parts.append(tier_df)
    final_manifest = pd.concat(selected_parts, ignore_index=True)

    baseline_rows = []
    for dataset in DATASETS:
        for split in ["valid", "test"]:
            base_df = pd.read_csv(src_root / dataset / f"{dataset}.{split}.inter", sep="\t", usecols=["session_id:token"])
            baseline_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "full_rows": int(len(base_df)),
                    "full_sessions": int(base_df["session_id:token"].astype(str).nunique()),
                }
            )
    baseline_df = pd.DataFrame(baseline_rows)

    group_stats = []
    for (dataset, split, tier, group), sdf in final_manifest.groupby(["dataset", "split", "tier", "group"], sort=True):
        base_row = baseline_df[(baseline_df["dataset"] == dataset) & (baseline_df["split"] == split)].iloc[0]
        group_stats.append(
            {
                "dataset": dataset,
                "split": split,
                "tier": tier,
                "group": group,
                "selected_session_rows": int(len(sdf)),
                "selected_unique_sessions": int(sdf["session_id"].nunique()),
                "full_sessions": int(base_row["full_sessions"]),
                "session_pct_of_split": float(sdf["session_id"].nunique() / base_row["full_sessions"]),
                "selection_score_mean": float(sdf["selection_score"].mean()),
                "selection_score_median": float(sdf["selection_score"].median()),
                "selection_score_min": float(sdf["selection_score"].min()),
                "selection_score_max": float(sdf["selection_score"].max()),
                "core_score_mean": float(sdf["core_score"].mean()),
                "balance_score_mean": float(sdf["balance_score"].mean()),
                "balanced_quota_before_cap": int(sdf["balanced_quota"].max()),
                "effective_quota_after_cap": int(sdf["balanced_rank"].max()),
            }
        )
    group_stats_df = pd.DataFrame(group_stats)

    materialized_rows: list[dict[str, object]] = []
    group_materialized_rows: list[dict[str, object]] = []
    union_materialized_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        for split in ["valid", "test"]:
            keep_sessions = set(
                final_manifest[
                    (final_manifest["dataset"] == dataset) & (final_manifest["split"] == split)
                ]["session_id"].astype(str)
            )
            split_df = pd.read_csv(src_root / dataset / f"{dataset}.{split}.inter", sep="\t", usecols=["session_id:token"])
            split_df["session_id:token"] = split_df["session_id:token"].astype(str)
            subset_rows = int(split_df["session_id:token"].isin(keep_sessions).sum())
            base_row = baseline_df[(baseline_df["dataset"] == dataset) & (baseline_df["split"] == split)].iloc[0]
            union_materialized_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "subset_rows": subset_rows,
                    "subset_sessions": int(len(keep_sessions)),
                    "full_rows": int(base_row["full_rows"]),
                    "full_sessions": int(base_row["full_sessions"]),
                    "row_pct_of_split": float(subset_rows / base_row["full_rows"]),
                    "session_pct_of_split": float(len(keep_sessions) / base_row["full_sessions"]),
                }
            )

    tier_summary_rows = []
    for (dataset, split, tier), sdf in final_manifest.groupby(["dataset", "split", "tier"], sort=True):
        base_row = baseline_df[(baseline_df["dataset"] == dataset) & (baseline_df["split"] == split)].iloc[0]
        tier_summary_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "tier": tier,
                "selected_entries": int(len(sdf)),
                "selected_unique_sessions": int(sdf["session_id"].nunique()),
                "full_sessions": int(base_row["full_sessions"]),
                "session_pct_of_split": float(sdf["session_id"].nunique() / base_row["full_sessions"]),
                "selection_score_mean": float(sdf["selection_score"].mean()),
                "selection_score_median": float(sdf["selection_score"].median()),
                "selection_score_min": float(sdf["selection_score"].min()),
                "selection_score_max": float(sdf["selection_score"].max()),
                "core_score_mean": float(sdf["core_score"].mean()),
                "balance_score_mean": float(sdf["balance_score"].mean()),
                "effective_quota_after_cap": int(sdf["balanced_rank"].max()),
            }
        )
    tier_summary_df = pd.DataFrame(tier_summary_rows)
    union_df = pd.DataFrame(union_materialized_rows)

    final_manifest.to_csv(stats_root / "final_case_session_manifest.csv", index=False)
    group_stats_df.to_csv(stats_root / "final_case_group_stats.csv", index=False)
    tier_summary_df.to_csv(stats_root / "final_case_tier_summary.csv", index=False)
    union_df.to_csv(stats_root / "final_case_all_tiers_subset_sizes.csv", index=False)

    print(f"wrote final manifest: {stats_root / 'final_case_session_manifest.csv'}", flush=True)
    print(f"wrote final group stats: {stats_root / 'final_case_group_stats.csv'}", flush=True)
    print(f"wrote final tier summary: {stats_root / 'final_case_tier_summary.csv'}", flush=True)
    print(f"wrote final union subset sizes: {stats_root / 'final_case_all_tiers_subset_sizes.csv'}", flush=True)

    for tier in TIERS:
        print(f"materializing tier={tier}", flush=True)
        tier_root = out_root / tier
        tier_manifest = final_manifest[final_manifest["tier"] == tier].copy()
        for dataset in DATASETS:
            split_session_map = {}
            for split in ["valid", "test"]:
                split_session_map[split] = set(
                    tier_manifest[
                        (tier_manifest["dataset"] == dataset) & (tier_manifest["split"] == split)
                    ]["session_id"].astype(str)
                )
            rows = materialize_tier_dataset(src_root=src_root, dst_root=tier_root, dataset=dataset, split_session_map=split_session_map)
            for row in rows:
                row["tier"] = tier
                base_row = baseline_df[(baseline_df["dataset"] == row["dataset"]) & (baseline_df["split"] == row["split"] )].iloc[0]
                row["full_rows"] = int(base_row["full_rows"])
                row["full_sessions"] = int(base_row["full_sessions"])
                row["row_pct_of_split"] = float(row["subset_rows"] / base_row["full_rows"])
                row["session_pct_of_split"] = float(row["subset_sessions"] / base_row["full_sessions"])
                materialized_rows.append(row)
            print(f"materialized tier={tier} dataset={dataset}", flush=True)

    for tier in TIERS:
        print(f"materializing group datasets tier={tier}", flush=True)
        tier_manifest = final_manifest[final_manifest["tier"] == tier].copy()
        for group in GROUPS:
            group_root = out_root / "by_tier_group" / tier / group
            group_manifest = tier_manifest[tier_manifest["group"] == group].copy()
            if group_manifest.empty:
                continue
            for dataset in DATASETS:
                split_session_map = {}
                for split in ["valid", "test"]:
                    split_session_map[split] = set(
                        group_manifest[
                            (group_manifest["dataset"] == dataset) & (group_manifest["split"] == split)
                        ]["session_id"].astype(str)
                    )
                rows = materialize_tier_dataset(src_root=src_root, dst_root=group_root, dataset=dataset, split_session_map=split_session_map)
                for row in rows:
                    row["tier"] = tier
                    row["group"] = group
                    base_row = baseline_df[(baseline_df["dataset"] == row["dataset"]) & (baseline_df["split"] == row["split"] )].iloc[0]
                    row["full_rows"] = int(base_row["full_rows"])
                    row["full_sessions"] = int(base_row["full_sessions"])
                    row["row_pct_of_split"] = float(row["subset_rows"] / base_row["full_rows"])
                    row["session_pct_of_split"] = float(row["subset_sessions"] / base_row["full_sessions"])
                    group_materialized_rows.append(row)
                print(f"materialized tier={tier} group={group} dataset={dataset}", flush=True)

    materialized_df = pd.DataFrame(materialized_rows)
    group_materialized_df = pd.DataFrame(group_materialized_rows)
    materialized_df.to_csv(stats_root / "final_case_tier_subset_sizes.csv", index=False)
    group_materialized_df.to_csv(stats_root / "final_case_group_subset_sizes.csv", index=False)

    md_lines = [
        "# Final Case Eval Summary",
        "",
        f"Permissive cap: {int(args.permissive_cap)} sessions per group, per dataset, per split.",
        "",
    ]
    for dataset in DATASETS:
        md_lines.append(f"## {dataset}")
        md_lines.append("")
        tier_view = tier_summary_df[tier_summary_df["dataset"] == dataset]
        if not tier_view.empty:
            md_lines.append(tier_view.to_markdown(index=False, floatfmt=".4f"))
            md_lines.append("")
        size_view = materialized_df[materialized_df["dataset"] == dataset]
        if not size_view.empty:
            md_lines.append("Tier dataset sizes:")
            md_lines.append("")
            md_lines.append(size_view.to_markdown(index=False, floatfmt=".4f"))
            md_lines.append("")
        union_view = union_df[union_df["dataset"] == dataset]
        if not union_view.empty:
            md_lines.append("All-tier union dataset size:")
            md_lines.append("")
            md_lines.append(union_view.to_markdown(index=False, floatfmt=".4f"))
            md_lines.append("")
    (stats_root / "final_case_summary.md").write_text("\n".join(md_lines))

    print(f"wrote final tier subset sizes: {stats_root / 'final_case_tier_subset_sizes.csv'}")
    print(f"wrote final group subset sizes: {stats_root / 'final_case_group_subset_sizes.csv'}")
    print(f"wrote final union subset sizes: {stats_root / 'final_case_all_tiers_subset_sizes.csv'}")
    print(f"wrote final datasets under: {out_root}")


if __name__ == "__main__":
    main()