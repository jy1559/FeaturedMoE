from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "beauty",
    "foursquare",
    "movielens1m",
    "retail_rocket",
]

FAMILIES = ["memory", "focus", "tempo", "exposure"]
GROUPS = [
    ("memory_plus", "memory", "plus"),
    ("memory_minus", "memory", "minus"),
    ("focus_plus", "focus", "plus"),
    ("focus_minus", "focus", "minus"),
    ("tempo_plus", "tempo", "plus"),
    ("tempo_minus", "tempo", "minus"),
    ("exposure_plus", "exposure", "plus"),
    ("exposure_minus", "exposure", "minus"),
]

PURE_CONFIG = {
    "memory_plus": {"core_q": 0.90, "contam_max_q": 0.85},
    "memory_minus": {"core_q": 0.90, "contam_max_q": 0.90},
    "focus_plus": {"core_q": 0.90, "contam_max_q": 0.90},
    "focus_minus": {"core_q": 0.90, "contam_max_q": 0.90},
    "tempo_plus": {"core_q": 0.90, "contam_max_q": 0.90},
    "tempo_minus": {"core_q": 0.90, "contam_max_q": 0.90},
    "exposure_plus": {"core_q": 0.88, "contam_max_q": 0.92},
    "exposure_minus": {"core_q": 0.88, "contam_max_q": 0.92},
}

PERMISSIVE_CONFIG = {
    "memory_plus": {"core_q": 0.80, "contam_max_q": 0.95},
    "memory_minus": {"core_q": 0.82, "contam_max_q": 0.97},
    "focus_plus": {"core_q": 0.82, "contam_max_q": 0.97},
    "focus_minus": {"core_q": 0.82, "contam_max_q": 0.97},
    "tempo_plus": {"core_q": 0.82, "contam_max_q": 0.97},
    "tempo_minus": {"core_q": 0.82, "contam_max_q": 0.97},
    "exposure_plus": {"core_q": 0.78, "contam_max_q": 0.98},
    "exposure_minus": {"core_q": 0.78, "contam_max_q": 0.98},
}


def clip01(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def safe_mean(*parts) -> np.ndarray:
    return np.vstack([clip01(p) for p in parts]).mean(axis=0)


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    memory_plus = safe_mean(
        df["mid_repeat_r:float"],
        1.0 - df["mid_item_uniq_r:float"],
        df["mid_max_run_i:float"],
        df["mic_suffix_recons_r:float"],
        df["mic_is_recons:float"],
    )
    focus_plus = safe_mean(
        df["mid_cat_top1:float"],
        1.0 - df["mid_cat_ent:float"],
        1.0 - df["mid_cat_switch_r:float"],
        1.0 - df["mic_suffix_cat_ent:float"],
        1.0 - df["mic_last_cat_mismatch_r:float"],
    )
    tempo_plus = safe_mean(
        1.0 - df["mid_int_mean:float"],
        1.0 - df["mic_gap_mean:float"],
        1.0 - df["mic_last_gap:float"],
    )
    exposure_plus = safe_mean(
        df["mid_pop_mean:float"],
        df["mic_last_pop:float"],
        1.0 - df["mid_pop_std:float"],
    )
    return pd.DataFrame(
        {
            "session_id": df["session_id:token"].astype(str),
            "item_id": df["item_id:token"].astype(str),
            "timestamp": df["timestamp:float"].astype(float),
            "memory_plus": memory_plus,
            "focus_plus": focus_plus,
            "tempo_plus": tempo_plus,
            "exposure_plus": exposure_plus,
        }
    )


def compute_masks(frame: pd.DataFrame, group_name: str, family: str, polarity: str):
    family_value = frame[f"{family}_plus"].to_numpy()
    core = family_value if polarity == "plus" else (1.0 - family_value)

    contamination_parts = []
    for other in FAMILIES:
        if other == family:
            continue
        other_plus = frame[f"{other}_plus"].to_numpy()
        contamination_parts.append(2.0 * np.abs(other_plus - 0.5))
    contamination = np.vstack(contamination_parts).max(axis=0)
    balance = 1.0 - contamination

    def _mask_from(cfg):
        core_cut = np.quantile(core, float(cfg["core_q"]))
        contam_cut = np.quantile(contamination, float(cfg["contam_max_q"]))
        mask = (core >= core_cut) & (contamination <= contam_cut)
        return mask, core_cut, contam_cut

    pure_mask, pure_core_cut, pure_contam_cut = _mask_from(PURE_CONFIG[group_name])
    permissive_mask, permissive_core_cut, permissive_contam_cut = _mask_from(PERMISSIVE_CONFIG[group_name])

    return {
        "core": core,
        "balance": balance,
        "contamination": contamination,
        "pure_mask": pure_mask,
        "permissive_mask": permissive_mask,
        "pure_core_cut": pure_core_cut,
        "pure_contam_cut": pure_contam_cut,
        "perm_core_cut": permissive_core_cut,
        "perm_contam_cut": permissive_contam_cut,
    }


def summarize_tier(dataset: str, split_name: str, group_name: str, family: str, polarity: str, tier_name: str, rows: pd.DataFrame, cut_info: dict) -> dict:
    out = {
        "dataset": dataset,
        "split": split_name,
        "group": group_name,
        "family": family,
        "polarity": polarity,
        "tier": tier_name,
        "support_rows": int(len(rows)),
        "support_sessions": int(rows["session_id"].nunique()) if len(rows) else 0,
        "core_score_mean": float(rows["core_score"].mean()) if len(rows) else np.nan,
        "core_score_median": float(rows["core_score"].median()) if len(rows) else np.nan,
        "balance_mean": float(rows["balance_score"].mean()) if len(rows) else np.nan,
        "contam_mean": float(rows["contamination_score"].mean()) if len(rows) else np.nan,
        "memory_plus_mean": float(rows["memory_plus"].mean()) if len(rows) else np.nan,
        "focus_plus_mean": float(rows["focus_plus"].mean()) if len(rows) else np.nan,
        "tempo_plus_mean": float(rows["tempo_plus"].mean()) if len(rows) else np.nan,
        "exposure_plus_mean": float(rows["exposure_plus"].mean()) if len(rows) else np.nan,
    }
    if tier_name == "pure":
        out["core_cut"] = float(cut_info["pure_core_cut"])
        out["contam_cut"] = float(cut_info["pure_contam_cut"])
    else:
        out["core_cut"] = float(cut_info["perm_core_cut"])
        out["contam_cut"] = float(cut_info["perm_contam_cut"])
    return out


def materialize_dataset(base_dir: Path, out_root: Path, dataset: str, manifest: pd.DataFrame) -> None:
    src_dir = base_dir / dataset
    dst_dir = out_root / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)

    ensure_symlink(src_dir / f"{dataset}.train.inter", dst_dir / f"{dataset}.train.inter")
    ensure_symlink(src_dir / f"{dataset}.item", dst_dir / f"{dataset}.item")
    if (src_dir / f"{dataset}.inter").exists():
        ensure_symlink(src_dir / f"{dataset}.inter", dst_dir / f"{dataset}.inter")
    if (src_dir / "feature_meta_v3.json").exists():
        ensure_symlink(src_dir / "feature_meta_v3.json", dst_dir / "feature_meta_v3.json")

    for split_name in ["valid", "test"]:
        split_src = pd.read_csv(src_dir / f"{dataset}.{split_name}.inter", sep="\t")
        split_src["session_id:token"] = split_src["session_id:token"].astype(str)
        split_src["item_id:token"] = split_src["item_id:token"].astype(str)
        split_src["timestamp:float"] = split_src["timestamp:float"].astype(float)
        keep = manifest[manifest["split"] == split_name][["session_id", "item_id", "timestamp"]].drop_duplicates()
        keep["session_id"] = keep["session_id"].astype(str)
        keep["item_id"] = keep["item_id"].astype(str)
        keep["timestamp"] = keep["timestamp"].astype(float)
        filtered = split_src.merge(
            keep,
            left_on=["session_id:token", "item_id:token", "timestamp:float"],
            right_on=["session_id", "item_id", "timestamp"],
            how="inner",
        ).drop(columns=["session_id", "item_id", "timestamp"])
        filtered.to_csv(dst_dir / f"{dataset}.{split_name}.inter", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build case-eval subset datasets and manifests.")
    parser.add_argument("--src-root", default="/workspace/FeaturedMoE/Datasets/processed/feature_added_v4")
    parser.add_argument("--out-root", default="/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_v1")
    parser.add_argument("--stats-root", default="/workspace/FeaturedMoE/outputs/case_mining_v2")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    stats_root = Path(args.stats_root)
    stats_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    tier_rows = []

    for dataset in DATASETS:
        for split_name in ["valid", "test"]:
            split_path = src_root / dataset / f"{dataset}.{split_name}.inter"
            raw = pd.read_csv(split_path, sep="\t")
            frame = build_feature_frame(raw)

            for group_name, family, polarity in GROUPS:
                mask_info = compute_masks(frame, group_name, family, polarity)
                for tier_name, tier_mask in [("pure", mask_info["pure_mask"]), ("permissive", mask_info["permissive_mask"] & ~mask_info["pure_mask"] )]:
                    rows = frame.loc[tier_mask].copy()
                    rows["dataset"] = dataset
                    rows["split"] = split_name
                    rows["group"] = group_name
                    rows["family"] = family
                    rows["polarity"] = polarity
                    rows["tier"] = tier_name
                    rows["core_score"] = mask_info["core"][tier_mask]
                    rows["balance_score"] = mask_info["balance"][tier_mask]
                    rows["contamination_score"] = mask_info["contamination"][tier_mask]
                    rows["selection_score"] = 0.85 * rows["core_score"] + 0.15 * rows["balance_score"]
                    session_rank = rows.sort_values("selection_score", ascending=False).drop_duplicates("session_id")
                    session_rank["session_rank"] = np.arange(1, len(session_rank) + 1)
                    rows = rows.merge(session_rank[["session_id", "session_rank"]], on="session_id", how="left")
                    manifest_rows.append(rows)
                    tier_rows.append(summarize_tier(dataset, split_name, group_name, family, polarity, tier_name, rows, mask_info))

        dataset_manifest = pd.concat([m for m in manifest_rows if isinstance(m, pd.DataFrame) and not m.empty and m.iloc[0]["dataset"] == dataset], ignore_index=True)
        materialize_dataset(src_root, out_root, dataset, dataset_manifest)

    manifest = pd.concat(manifest_rows, ignore_index=True)
    stats = pd.DataFrame(tier_rows)
    manifest.to_csv(stats_root / "case_manifest.csv", index=False)
    stats.to_csv(stats_root / "case_group_stats.csv", index=False)

    dataset_stats = []
    for (dataset, split_name, group_name), sdf in manifest.groupby(["dataset", "split", "group"]):
        pure_sessions = int(sdf.loc[sdf["tier"] == "pure", "session_id"].nunique())
        permissive_sessions = int(sdf.loc[sdf["tier"] == "permissive", "session_id"].nunique())
        dataset_stats.append(
            {
                "dataset": dataset,
                "split": split_name,
                "group": group_name,
                "pure_sessions": pure_sessions,
                "permissive_sessions": permissive_sessions,
                "total_sessions": pure_sessions + permissive_sessions,
            }
        )
    dataset_stats_df = pd.DataFrame(dataset_stats)
    dataset_stats_df.to_csv(stats_root / "case_dataset_split_counts.csv", index=False)

    md_lines = ["# Case Eval V2 Summary", ""]
    for dataset in DATASETS:
        md_lines.append(f"## {dataset}")
        md_lines.append("")
        view = dataset_stats_df[dataset_stats_df["dataset"] == dataset].copy()
        md_lines.append(view.to_markdown(index=False))
        md_lines.append("")
    (stats_root / "case_dataset_split_counts.md").write_text("\n".join(md_lines))

    print(f"wrote manifest: {stats_root / 'case_manifest.csv'}")
    print(f"wrote stats: {stats_root / 'case_group_stats.csv'}")
    print(f"wrote counts: {stats_root / 'case_dataset_split_counts.csv'}")
    print(f"wrote case-eval datasets under: {out_root}")


if __name__ == "__main__":
    main()