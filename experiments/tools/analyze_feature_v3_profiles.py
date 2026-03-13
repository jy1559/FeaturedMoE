#!/usr/bin/env python3
"""Summarize feature_added_v3 datasets and compare lightly against v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE = ["session_id", "item_id", "timestamp", "user_id"]


def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="KuaiRecLargeStrictPosV2_0.2,lastfm0.03",
    )
    parser.add_argument(
        "--processed_root_v3",
        type=str,
        default="/workspace/jy1559/FMoE/Datasets/processed/feature_added_v3",
    )
    parser.add_argument(
        "--processed_root_v2",
        type=str,
        default="/workspace/jy1559/FMoE/Datasets/processed/feature_added_v2",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/jy1559/FMoE/experiments/run/artifacts/analysis/feature_v3_profiles",
    )
    return parser.parse_args()


def resolve_inter_path(processed_root: Path, dataset: str) -> Path:
    path = processed_root / dataset / f"{dataset}.inter"
    if not path.exists():
        raise FileNotFoundError(f"Missing inter file: {path}")
    return path


def resolve_meta_path(processed_root: Path, dataset: str, version: str) -> Path:
    path = processed_root / dataset / f"feature_meta_{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature meta: {path}")
    return path


def load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def load_feature_df(inter_path: Path, features: list[str]) -> pd.DataFrame:
    header = pd.read_csv(inter_path, sep="\t", nrows=0)
    col_map = {strip_type(col): col for col in header.columns}
    wanted = [c for c in CORE + features if c in col_map]
    usecols = [col_map[c] for c in wanted]
    dtype = {}
    for c in CORE:
        if c in col_map:
            dtype[col_map[c]] = "float64" if c == "timestamp" else "string"
    for c in features:
        if c in col_map:
            dtype[col_map[c]] = "float32"
    df = pd.read_csv(inter_path, sep="\t", usecols=usecols, dtype=dtype)
    return df.rename(columns={v: k for k, v in col_map.items() if v in usecols})


def summarize_structure(df: pd.DataFrame) -> dict[str, float]:
    sess_size = df.groupby("session_id", sort=False).size()
    user_sess = df.groupby("user_id", sort=False)["session_id"].nunique()
    item_counts = df.groupby("item_id", sort=False).size().sort_values(ascending=False)
    return {
        "rows": int(len(df)),
        "sessions": int(sess_size.shape[0]),
        "users": int(user_sess.shape[0]),
        "items": int(df["item_id"].nunique()),
        "avg_session_len": float(sess_size.mean()),
        "median_session_len": float(sess_size.median()),
        "p90_session_len": float(sess_size.quantile(0.9)),
        "effective_tokens@10": int(np.minimum(sess_size.to_numpy(dtype=np.int64), 10).sum()),
        "effective_tokens@30": int(np.minimum(sess_size.to_numpy(dtype=np.int64), 30).sum()),
        "avg_sessions_per_user": float(user_sess.mean()),
        "top10_item_share": float(item_counts.head(10).sum() / len(df)) if len(item_counts) else 0.0,
        "top100_item_share": float(item_counts.head(100).sum() / len(df)) if len(item_counts) else 0.0,
    }


def summarize_macro_context(df: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    bins = np.linspace(0.0, 1.0, 6)
    labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    for key in ("mac5_ctx_valid_r", "mac10_ctx_valid_r"):
        vals = pd.to_numeric(df[key], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        hist, _ = np.histogram(vals, bins=bins)
        out[key] = {
            "mean": float(np.mean(vals)),
            "p50": float(np.quantile(vals, 0.5)),
            "p90": float(np.quantile(vals, 0.9)),
            "histogram": {labels[i]: int(hist[i]) for i in range(len(hist))},
        }
    return out


def summarize_feature_stats(df: pd.DataFrame, features: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for feature in features:
        vals = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[feature] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "p01": 0.0,
                "p10": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "max": 0.0,
            }
            continue
        out[feature] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "min": float(np.min(vals)),
            "p01": float(np.quantile(vals, 0.01)),
            "p10": float(np.quantile(vals, 0.10)),
            "p50": float(np.quantile(vals, 0.50)),
            "p90": float(np.quantile(vals, 0.90)),
            "p99": float(np.quantile(vals, 0.99)),
            "max": float(np.max(vals)),
        }
    return out


def summarize_family_means(df: pd.DataFrame, families: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for stage, family_map in families.items():
        out[stage] = {}
        for family, features in family_map.items():
            out[stage][family] = float(df[features].mean(axis=1).mean())
    return out


def summarize_family_correlations(df: pd.DataFrame, families: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, float]]:
    family_cols = {}
    for stage, family_map in families.items():
        for family, features in family_map.items():
            key = f"{stage}.{family}"
            family_cols[key] = df[features].mean(axis=1)
    fam_df = pd.DataFrame(family_cols)
    corr = fam_df.corr().fillna(0.0)
    return {
        row: {col: float(corr.loc[row, col]) for col in corr.columns}
        for row in corr.index
    }


def summarize_v2_overlap(processed_root_v2: Path, dataset: str, v3_df: pd.DataFrame, v3_features: list[str]) -> dict:
    try:
        inter_path = resolve_inter_path(processed_root_v2, dataset)
        header = pd.read_csv(inter_path, sep="\t", nrows=0)
    except FileNotFoundError:
        return {}

    col_map = {strip_type(col): col for col in header.columns}
    overlap = [feat for feat in v3_features if feat in col_map]
    if not overlap:
        overlap = [feat for feat in ("mid_valid_r", "mid_novel_r", "mic_valid_r", "mic_is_recons") if feat in col_map and feat in v3_df.columns]
    if not overlap:
        return {}

    usecols = [col_map[feat] for feat in overlap]
    v2_df = pd.read_csv(inter_path, sep="\t", usecols=usecols, dtype={col: "float32" for col in usecols})
    v2_df = v2_df.rename(columns={col_map[k]: k for k in overlap})
    out = {}
    for feat in overlap:
        v2_vals = pd.to_numeric(v2_df[feat], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        v3_vals = pd.to_numeric(v3_df[feat], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out[feat] = {
            "v2_mean": float(np.mean(v2_vals)),
            "v3_mean": float(np.mean(v3_vals)),
            "v2_p90": float(np.quantile(v2_vals, 0.9)),
            "v3_p90": float(np.quantile(v3_vals, 0.9)),
        }
    return out


def render_markdown(results: dict[str, dict]) -> str:
    lines = []
    lines.append("# Feature V3 Profiles")
    lines.append("")
    lines.append("## Structure")
    lines.append("")
    lines.append("| dataset | rows | sessions | users | items | avg_sess | p90_sess | eff@10 | top100 share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset, res in results.items():
        s = res["structure"]
        lines.append(
            f"| {dataset} | {s['rows']:,} | {s['sessions']:,} | {s['users']:,} | {s['items']:,} | "
            f"{s['avg_session_len']:.2f} | {s['p90_session_len']:.1f} | {s['effective_tokens@10']:,} | "
            f"{s['top100_item_share']:.3f} |"
        )
    lines.append("")
    lines.append("## Macro Context Availability")
    lines.append("")
    lines.append("| dataset | mac5 mean | mac5 p90 | mac10 mean | mac10 p90 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for dataset, res in results.items():
        mac = res["macro_context"]
        lines.append(
            f"| {dataset} | {mac['mac5_ctx_valid_r']['mean']:.3f} | {mac['mac5_ctx_valid_r']['p90']:.3f} | "
            f"{mac['mac10_ctx_valid_r']['mean']:.3f} | {mac['mac10_ctx_valid_r']['p90']:.3f} |"
        )
    lines.append("")
    lines.append("## Light V2 Overlap")
    lines.append("")
    for dataset, res in results.items():
        if not res["v2_overlap"]:
            continue
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| feature | v2 mean | v3 mean | v2 p90 | v3 p90 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for feature, row in res["v2_overlap"].items():
            lines.append(
                f"| {feature} | {row['v2_mean']:.3f} | {row['v3_mean']:.3f} | {row['v2_p90']:.3f} | {row['v3_p90']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def analyze_dataset(processed_root_v3: Path, processed_root_v2: Path, dataset: str) -> dict:
    meta_path = resolve_meta_path(processed_root_v3, dataset, version="v3")
    meta = load_meta(meta_path)
    features = list(meta["all_features"])
    inter_path = resolve_inter_path(processed_root_v3, dataset)
    df = load_feature_df(inter_path, features)
    return {
        "inter_path": str(inter_path),
        "meta_path": str(meta_path),
        "structure": summarize_structure(df),
        "macro_context": summarize_macro_context(df),
        "feature_stats": summarize_feature_stats(df, features),
        "family_means": summarize_family_means(df, meta["families"]),
        "family_correlations": summarize_family_correlations(df, meta["families"]),
        "v2_overlap": summarize_v2_overlap(processed_root_v2, dataset, df, features),
        "families": meta["families"],
        "all_features": features,
    }


def main() -> None:
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    processed_root_v3 = Path(args.processed_root_v3)
    processed_root_v2 = Path(args.processed_root_v2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        dataset: analyze_dataset(processed_root_v3, processed_root_v2, dataset)
        for dataset in datasets
    }

    md_path = output_dir / "feature_v3_profiles.md"
    json_path = output_dir / "feature_v3_profiles.json"
    md_path.write_text(render_markdown(results) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
