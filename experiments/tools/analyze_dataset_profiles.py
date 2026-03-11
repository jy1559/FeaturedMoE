#!/usr/bin/env python3
"""Summarize dataset structure and feature probes for FMoE experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE = ["session_id", "item_id", "timestamp", "user_id"]
PROBE_FEATURES = [
    "mac_is_new",
    "mac_is_weekend",
    "mac_hist_ent",
    "mac_hist_uniq_r",
    "mac_hist_switch_r",
    "mid_valid_r",
    "mid_novel_r",
    "mid_uniq_item_r",
    "mid_uniq_cat_r",
    "mid_max_run",
    "mic_valid_r",
    "mic_switch",
    "mic_is_recons",
    "mic_recons_r",
    "mic_max_run_i",
    "mic_pop_ent",
]
RAW_Z_KEYS = [
    "mac_sess_gap",
    "mac_hist_len",
    "mac_hist_pop_avg",
    "mid_sess_time",
    "mid_int_avg",
    "mid_pop_drift",
    "mic_short_avg",
    "mic_pop_avg",
    "mic_pop_drift",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="KuaiRecSmall0.1,lastfm0.03,movielens1m,retail_rocket,foursquare,amazon_beauty",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default="/workspace/jy1559/FMoE/Datasets/processed/feature_added_v2",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/jy1559/FMoE/experiments/run/artifacts/analysis/dataset_profiles",
    )
    return parser.parse_args()


def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def resolve_inter_path(processed_root: Path, dataset: str) -> Path:
    path = processed_root / dataset / f"{dataset}.inter"
    if not path.exists():
        raise FileNotFoundError(f"Missing inter file: {path}")
    return path


def resolve_meta_path(processed_root: Path, dataset: str) -> Path:
    path = processed_root / dataset / "feature_meta_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature meta: {path}")
    return path


def load_probe_df(inter_path: Path) -> pd.DataFrame:
    header = pd.read_csv(inter_path, sep="\t", nrows=0)
    col_map = {strip_type(col): col for col in header.columns}
    wanted = [c for c in CORE + PROBE_FEATURES if c in col_map]
    usecols = [col_map[c] for c in wanted]
    dtype = {col_map[c]: "string" for c in CORE if c != "timestamp" and c in col_map}
    dtype.update({col_map[c]: "float32" for c in PROBE_FEATURES if c in col_map})
    if "timestamp" in col_map:
        dtype[col_map["timestamp"]] = "float64"
    df = pd.read_csv(inter_path, sep="\t", usecols=usecols, dtype=dtype)
    return df.rename(columns={v: k for k, v in col_map.items() if v in usecols})


def summarize_structure(df: pd.DataFrame) -> dict[str, float]:
    sess_size = df.groupby("session_id", sort=False).size()
    user_sess = df.groupby("user_id", sort=False)["session_id"].nunique()
    user_rows = df.groupby("user_id", sort=False).size()
    item_counts = df.groupby("item_id", sort=False).size().sort_values(ascending=False)
    top1 = float(item_counts.iloc[0] / len(df)) if len(item_counts) else 0.0
    top10 = float(item_counts.head(10).sum() / len(df)) if len(item_counts) else 0.0
    top100 = float(item_counts.head(100).sum() / len(df)) if len(item_counts) else 0.0
    return {
        "rows": int(len(df)),
        "sessions": int(sess_size.shape[0]),
        "users": int(user_sess.shape[0]),
        "items": int(df["item_id"].nunique()),
        "avg_session_len": float(sess_size.mean()),
        "median_session_len": float(sess_size.median()),
        "p90_session_len": float(sess_size.quantile(0.9)),
        "avg_sessions_per_user": float(user_sess.mean()),
        "avg_rows_per_user": float(user_rows.mean()),
        "effective_tokens@10": int(np.minimum(sess_size.to_numpy(dtype=np.int64), 10).sum()),
        "effective_tokens@30": int(np.minimum(sess_size.to_numpy(dtype=np.int64), 30).sum()),
        "long_session_ratio_>10": float((sess_size > 10).mean()),
        "long_session_ratio_>30": float((sess_size > 30).mean()),
        "top1_item_share": top1,
        "top10_item_share": top10,
        "top100_item_share": top100,
        "items_per_user": float(df["item_id"].nunique() / user_sess.shape[0]),
        "rows_per_item": float(len(df) / max(1, df["item_id"].nunique())),
    }


def summarize_probe_features(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in PROBE_FEATURES:
        if key not in df.columns:
            continue
        vals = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[key] = {"mean": 0.0, "p50": 0.0, "p90": 0.0}
            continue
        out[key] = {
            "mean": float(np.mean(vals)),
            "p50": float(np.quantile(vals, 0.5)),
            "p90": float(np.quantile(vals, 0.9)),
        }
    return out


def load_raw_z_summary(meta_path: Path) -> dict[str, dict[str, float]]:
    meta = json.loads(meta_path.read_text())
    z_params = meta.get("z_params", {})
    out = {}
    for key in RAW_Z_KEYS:
        if key in z_params:
            out[key] = {
                "mean": float(z_params[key]["mean"]),
                "std": float(z_params[key]["std"]),
            }
    return {
        "dataset": meta.get("dataset"),
        "timestamp_unit": meta.get("timestamp_unit"),
        "fit_rows": int(meta.get("zscore_fit", {}).get("fit_rows", 0)),
        "total_rows": int(meta.get("zscore_fit", {}).get("total_rows", 0)),
        "raw_z": out,
    }


def render_markdown(results: dict[str, dict]) -> str:
    lines = []
    lines.append("# Dataset Profiles")
    lines.append("")
    lines.append("## Structure")
    lines.append("")
    lines.append("| dataset | rows | sessions | users | items | avg_sess | p90_sess | eff@10 | eff@30 | avg_sess/user | top10 share | top100 share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ds, res in results.items():
        s = res["structure"]
        lines.append(
            f"| {ds} | {s['rows']:,} | {s['sessions']:,} | {s['users']:,} | {s['items']:,} | "
            f"{s['avg_session_len']:.2f} | {s['p90_session_len']:.1f} | {s['effective_tokens@10']:,} | "
            f"{s['effective_tokens@30']:,} | {s['avg_sessions_per_user']:.2f} | "
            f"{s['top10_item_share']:.3f} | {s['top100_item_share']:.3f} |"
        )
    lines.append("")
    lines.append("## Raw Feature Probes")
    lines.append("")
    lines.append("| dataset | weekend mean | mid_valid mean | mid_novel mean | mic_valid mean | mic_switch mean | mic_recons mean | hist_switch mean | mic_pop_ent mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ds, res in results.items():
        p = res["probe"]
        lines.append(
            f"| {ds} | {p['mac_is_weekend']['mean']:.3f} | {p['mid_valid_r']['mean']:.3f} | "
            f"{p['mid_novel_r']['mean']:.3f} | {p['mic_valid_r']['mean']:.3f} | "
            f"{p['mic_switch']['mean']:.3f} | {p['mic_is_recons']['mean']:.3f} | "
            f"{p['mac_hist_switch_r']['mean']:.3f} | {p['mic_pop_ent']['mean']:.3f} |"
        )
    lines.append("")
    lines.append("## Raw-Unit Z Features")
    lines.append("")
    lines.append("| dataset | ts unit | mac_sess_gap mean | mac_hist_len mean | mac_hist_pop_avg mean | mid_sess_time mean | mid_int_avg mean | mid_pop_drift mean | mic_short_avg mean | mic_pop_drift mean |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ds, res in results.items():
        z = res["raw_z"]["raw_z"]
        lines.append(
            f"| {ds} | {res['raw_z']['timestamp_unit']} | {z['mac_sess_gap']['mean']:.3f} | "
            f"{z['mac_hist_len']['mean']:.3f} | {z['mac_hist_pop_avg']['mean']:.3f} | "
            f"{z['mid_sess_time']['mean']:.3f} | {z['mid_int_avg']['mean']:.3f} | "
            f"{z['mid_pop_drift']['mean']:.3f} | {z['mic_short_avg']['mean']:.3f} | "
            f"{z['mic_pop_drift']['mean']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for dataset in datasets:
        inter_path = resolve_inter_path(processed_root, dataset)
        meta_path = resolve_meta_path(processed_root, dataset)
        df = load_probe_df(inter_path)
        results[dataset] = {
            "inter_path": str(inter_path),
            "meta_path": str(meta_path),
            "structure": summarize_structure(df),
            "probe": summarize_probe_features(df),
            "raw_z": load_raw_z_summary(meta_path),
        }

    md = render_markdown(results)
    md_path = output_dir / "dataset_profiles.md"
    json_path = output_dir / "dataset_profiles.json"
    md_path.write_text(md + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
