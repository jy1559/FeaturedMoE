#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


RESULTS_ROOT = Path("/workspace/FeaturedMoE/writing/results")


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in dedent(code).strip().splitlines()],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(relative_path: str, cells: list[dict]) -> None:
    target_path = RESULTS_ROOT / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


COMMON_IMPORTS = """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

RESULTS_ROOT = Path("/workspace/FeaturedMoE/writing/results")
if str(RESULTS_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULTS_ROOT))

from _shared.io_helpers import export_figure, load_csv_or_demo
from _shared.paper_theme import set_paper_theme
from _shared.plot_builders import (
    grouped_barplot,
    heatmap_from_long,
    lineplot_with_markers,
    scatterplot_with_annotations,
)

set_paper_theme(context="notebook")
"""


def build_main_overall() -> list[dict]:
    return [
        md_cell(
            """
# 01 Main Overall

This notebook reads the compact main-paper overall results CSV and renders a quick three-panel comparison for sanity-checking the manuscript table.

Inputs:

- `01_main_overall.csv`
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
CSV_PATH = RESULTS_ROOT / "01_main_overall/01_main_overall.csv"
REQUIRED_COLUMNS = [
    "paper_section",
    "panel",
    "dataset",
    "variant_or_model",
    "metric",
    "cutoff",
    "value",
    "split",
    "selection_rule",
    "run_id",
    "source_path",
    "notes",
]


def demo_overall() -> pd.DataFrame:
    rows = []
    demo_values = {
        "Beauty": {"RouteRec": (0.1256, 0.0826, 0.0763), "SASRec": (0.1082, 0.0714, 0.0667), "BSARec": (0.1168, 0.0750, 0.0698), "DuoRec": (0.1187, 0.0778, 0.0727)},
        "Foursquare": {"RouteRec": (0.2447, 0.1589, 0.1430), "SASRec": (0.2361, 0.1527, 0.1378), "BSARec": (0.2050, 0.1301, 0.1168), "DuoRec": (0.2335, 0.1494, 0.1336)},
        "KuaiRec": {"RouteRec": (0.1832, 0.1264, 0.1185), "SASRec": (0.1798, 0.1236, 0.1160), "BSARec": (0.1764, 0.1208, 0.1133), "DuoRec": (0.1809, 0.1241, 0.1164)},
        "LastFM": {"RouteRec": (0.4305, 0.3391, 0.3083), "SASRec": (0.4456, 0.3428, 0.3114), "BSARec": (0.4280, 0.3345, 0.3045), "DuoRec": (0.2694, 0.2236, 0.2126)},
        "ML-1M": {"RouteRec": (0.1826, 0.0978, 0.0741), "SASRec": (0.1887, 0.0952, 0.0783), "BSARec": (0.1841, 0.0983, 0.0747), "DuoRec": (0.1652, 0.0838, 0.0659)},
        "Retail Rocket": {"RouteRec": (0.4723, 0.3094, 0.2495), "SASRec": (0.4391, 0.2907, 0.2348), "BSARec": (0.4327, 0.2870, 0.2278), "DuoRec": (0.4427, 0.2921, 0.2362)},
    }
    metrics = [("HR", 10), ("NDCG", 10), ("MRR", 20)]
    for dataset, models in demo_values.items():
        for model, values in models.items():
            for (metric, cutoff), value in zip(metrics, values):
                rows.append(
                    {
                        "paper_section": "01_main_overall",
                        "panel": "main",
                        "dataset": dataset,
                        "variant_or_model": model,
                        "metric": metric,
                        "cutoff": cutoff,
                        "value": value,
                        "split": "test",
                        "selection_rule": "demo",
                        "run_id": "demo",
                        "source_path": "demo",
                        "notes": "demo data",
                    }
                )
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
df, mode = load_csv_or_demo(CSV_PATH, REQUIRED_COLUMNS, demo_builder=demo_overall)
display(Markdown(f"**Load mode:** {mode}"))
display(df.head())
"""
        ),
        code_cell(
            """
metric_specs = [("HR", 10, "HR@10"), ("NDCG", 10, "NDCG@10"), ("MRR", 20, "MRR@20")]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
for ax, (metric, cutoff, title) in zip(axes, metric_specs):
    subset = df[(df["metric"] == metric) & (df["cutoff"] == cutoff)].copy()
    grouped_barplot(
        subset,
        x="dataset",
        hue="variant_or_model",
        y="value",
        ax=ax,
        title=title,
        ylabel="Score",
        xlabel="Dataset",
        rotate=30,
    )
    if ax is not axes[0] and ax.get_legend() is not None:
        ax.get_legend().remove()
axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.02))
saved_paths = export_figure(fig, "01_main_overall_overview", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_routing_control() -> list[dict]:
    return [
        md_cell(
            """
# 02 Routing Control

This notebook combines the routing-control quality and consistency CSVs into the main routing-control figure.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
QUALITY_PATH = RESULTS_ROOT / "02_routing_control/02a_routing_control_quality.csv"
CONSISTENCY_PATH = RESULTS_ROOT / "02_routing_control/02b_routing_control_consistency.csv"
QUALITY_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "metric", "cutoff", "value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]
CONSISTENCY_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "similarity_bucket", "consistency_value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]


def demo_quality() -> pd.DataFrame:
    rows = []
    datasets = ["Beauty", "Foursquare", "KuaiRec", "Retail Rocket"]
    variants = {
        "shared_ffn": [0.063, 0.109, 0.102, 0.201],
        "hidden_only": [0.069, 0.123, 0.110, 0.218],
        "mixed": [0.071, 0.129, 0.114, 0.224],
        "behavior_only": [0.076, 0.143, 0.119, 0.250],
    }
    for variant, values in variants.items():
        for dataset, value in zip(datasets, values):
            rows.append(
                {
                    "paper_section": "02_routing_control",
                    "panel": "quality",
                    "dataset": dataset,
                    "variant_or_model": variant,
                    "metric": "MRR",
                    "cutoff": 20,
                    "value": value,
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                }
            )
    return pd.DataFrame(rows)


def demo_consistency() -> pd.DataFrame:
    rows = []
    buckets = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    variants = {
        "hidden_only": [0.31, 0.38, 0.44, 0.50, 0.56],
        "behavior_only": [0.35, 0.44, 0.53, 0.61, 0.69],
    }
    for variant, values in variants.items():
        for bucket, value in zip(buckets, values):
            rows.append(
                {
                    "paper_section": "02_routing_control",
                    "panel": "consistency",
                    "dataset": "aggregate",
                    "variant_or_model": variant,
                    "similarity_bucket": bucket,
                    "consistency_value": value,
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                }
            )
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
quality_df, quality_mode = load_csv_or_demo(QUALITY_PATH, QUALITY_COLUMNS, demo_builder=demo_quality)
consistency_df, consistency_mode = load_csv_or_demo(CONSISTENCY_PATH, CONSISTENCY_COLUMNS, demo_builder=demo_consistency)
display(Markdown(f"**Load mode:** quality={quality_mode}, consistency={consistency_mode}"))
display(quality_df.head())
display(consistency_df.head())
"""
        ),
        code_cell(
            """
bucket_order = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
consistency_df = consistency_df.copy()
consistency_df["similarity_bucket"] = pd.Categorical(consistency_df["similarity_bucket"], categories=bucket_order, ordered=True)
consistency_df = consistency_df.sort_values("similarity_bucket")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), constrained_layout=True)
grouped_barplot(
    quality_df,
    x="dataset",
    hue="variant_or_model",
    y="value",
    ax=axes[0],
    title="(a) Ranking quality",
    ylabel="MRR@20",
    xlabel="Dataset",
    rotate=25,
)
lineplot_with_markers(
    consistency_df,
    x="similarity_bucket",
    y="consistency_value",
    hue="variant_or_model",
    ax=axes[1],
    title="(b) Route consistency",
    ylabel="Consistency",
    xlabel="Feature-similarity bucket",
    annotate_points=True,
)
saved_paths = export_figure(fig, "02_routing_control", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_stage_structure() -> list[dict]:
    return [
        md_cell(
            """
# 03 Stage Structure

This notebook renders the stage-removal, dense-vs-staged, and wrapper/order comparisons from three section-local CSVs.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
STAGE_PATH = RESULTS_ROOT / "03_stage_structure/03a_stage_ablation.csv"
DENSE_PATH = RESULTS_ROOT / "03_stage_structure/03b_dense_vs_staged.csv"
WRAPPER_PATH = RESULTS_ROOT / "03_stage_structure/03c_wrapper_order.csv"

STAGE_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "ablation_group", "metric", "cutoff", "value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]
DENSE_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "layout_variant", "stage_count", "metric", "cutoff", "value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]
WRAPPER_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "wrapper_variant", "stage_order", "metric", "cutoff", "value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]


def demo_stage() -> pd.DataFrame:
    rows = []
    values = {"full": 0.1185, "remove_macro": 0.1152, "remove_mid": 0.1144, "remove_micro": 0.1129}
    for group, value in values.items():
        rows.append(
            {
                "paper_section": "03_stage_structure",
                "panel": "stage_ablation",
                "dataset": "KuaiRec",
                "variant_or_model": "RouteRec",
                "ablation_group": group,
                "metric": "MRR",
                "cutoff": 20,
                "value": value,
                "split": "test",
                "selection_rule": "demo",
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            }
        )
    return pd.DataFrame(rows)


def demo_dense() -> pd.DataFrame:
    rows = []
    values = [("dense_ffn", 0, 0.1082), ("single_stage", 1, 0.1126), ("two_stage", 2, 0.1161), ("three_stage", 3, 0.1185)]
    for layout_variant, stage_count, value in values:
        rows.append(
            {
                "paper_section": "03_stage_structure",
                "panel": "dense_vs_staged",
                "dataset": "KuaiRec",
                "variant_or_model": "RouteRec",
                "layout_variant": layout_variant,
                "stage_count": stage_count,
                "metric": "MRR",
                "cutoff": 20,
                "value": value,
                "split": "test",
                "selection_rule": "demo",
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            }
        )
    return pd.DataFrame(rows)


def demo_wrapper() -> pd.DataFrame:
    rows = []
    values = [("A8", "macro-mid-micro", 0.1141), ("A10", "macro-mid-micro", 0.1185), ("A11", "mid-macro-micro", 0.1163), ("A12", "macro-micro-mid", 0.1170)]
    for wrapper_variant, stage_order, value in values:
        rows.append(
            {
                "paper_section": "03_stage_structure",
                "panel": "wrapper_order",
                "dataset": "KuaiRec",
                "variant_or_model": "RouteRec",
                "wrapper_variant": wrapper_variant,
                "stage_order": stage_order,
                "metric": "MRR",
                "cutoff": 20,
                "value": value,
                "split": "test",
                "selection_rule": "demo",
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            }
        )
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
stage_df, stage_mode = load_csv_or_demo(STAGE_PATH, STAGE_COLUMNS, demo_builder=demo_stage)
dense_df, dense_mode = load_csv_or_demo(DENSE_PATH, DENSE_COLUMNS, demo_builder=demo_dense)
wrapper_df, wrapper_mode = load_csv_or_demo(WRAPPER_PATH, WRAPPER_COLUMNS, demo_builder=demo_wrapper)
display(Markdown(f"**Load mode:** stage={stage_mode}, dense={dense_mode}, wrapper={wrapper_mode}"))
display(stage_df)
display(dense_df)
display(wrapper_df)
"""
        ),
        code_cell(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
grouped_barplot(stage_df, x="ablation_group", hue="variant_or_model", y="value", ax=axes[0], title="(a) Stage removal", ylabel="MRR@20", xlabel="Variant", rotate=25)
grouped_barplot(dense_df, x="layout_variant", hue="variant_or_model", y="value", ax=axes[1], title="(b) Dense vs staged", ylabel="MRR@20", xlabel="Layout", rotate=25)
grouped_barplot(wrapper_df, x="wrapper_variant", hue="variant_or_model", y="value", ax=axes[2], title="(c) Wrapper and order", ylabel="MRR@20", xlabel="Variant", rotate=0)
for ax in axes[1:]:
    if ax.get_legend() is not None:
        ax.get_legend().remove()
saved_paths = export_figure(fig, "03_stage_structure", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_cue_ablation() -> list[dict]:
    return [
        md_cell(
            """
# 04 Cue Ablation

This notebook renders the lightweight-cue ablation and retention views from the cue-ablation CSV templates.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
ABLATION_PATH = RESULTS_ROOT / "04_cue_ablation/04a_cue_ablation.csv"
RETENTION_PATH = RESULTS_ROOT / "04_cue_ablation/04b_cue_retention.csv"
ABLATION_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "cue_setting", "cue_family_removed", "metric", "cutoff", "value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]
RETENTION_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "retention_target", "metric", "cutoff", "value", "reference_value",
    "relative_gain", "split", "selection_rule", "run_id", "source_path", "notes",
]


def demo_ablation() -> pd.DataFrame:
    rows = []
    values = {
        "full": [0.0763, 0.1430, 0.1185, 0.2495],
        "remove_category": [0.0731, 0.1394, 0.1169, 0.2432],
        "remove_time": [0.0725, 0.1370, 0.1158, 0.2405],
        "sequence_only": [0.0709, 0.1322, 0.1134, 0.2354],
    }
    datasets = ["Beauty", "Foursquare", "KuaiRec", "Retail Rocket"]
    for cue_setting, setting_values in values.items():
        for dataset, value in zip(datasets, setting_values):
            rows.append(
                {
                    "paper_section": "04_cue_ablation",
                    "panel": "ablation",
                    "dataset": dataset,
                    "variant_or_model": "RouteRec",
                    "cue_setting": cue_setting,
                    "cue_family_removed": cue_setting,
                    "metric": "MRR",
                    "cutoff": 20,
                    "value": value,
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                }
            )
    return pd.DataFrame(rows)


def demo_retention() -> pd.DataFrame:
    rows = []
    values = [
        ("Beauty", "remove_category", 0.0731, 0.0763, 0.96),
        ("Beauty", "remove_time", 0.0725, 0.0763, 0.95),
        ("Beauty", "sequence_only", 0.0709, 0.0763, 0.93),
        ("Foursquare", "remove_category", 0.1394, 0.1430, 0.97),
        ("Foursquare", "remove_time", 0.1370, 0.1430, 0.96),
        ("Foursquare", "sequence_only", 0.1322, 0.1430, 0.92),
    ]
    for dataset, target, value, reference_value, relative_gain in values:
        rows.append(
            {
                "paper_section": "04_cue_ablation",
                "panel": "retention",
                "dataset": dataset,
                "variant_or_model": "RouteRec",
                "retention_target": target,
                "metric": "MRR",
                "cutoff": 20,
                "value": value,
                "reference_value": reference_value,
                "relative_gain": relative_gain,
                "split": "test",
                "selection_rule": "demo",
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            }
        )
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
ablation_df, ablation_mode = load_csv_or_demo(ABLATION_PATH, ABLATION_COLUMNS, demo_builder=demo_ablation)
retention_df, retention_mode = load_csv_or_demo(RETENTION_PATH, RETENTION_COLUMNS, demo_builder=demo_retention)
display(Markdown(f"**Load mode:** ablation={ablation_mode}, retention={retention_mode}"))
display(ablation_df.head())
display(retention_df.head())
"""
        ),
        code_cell(
            """
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), constrained_layout=True)
grouped_barplot(
    ablation_df,
    x="dataset",
    hue="cue_setting",
    y="value",
    ax=axes[0],
    title="(a) Cue-family ablation",
    ylabel="MRR@20",
    xlabel="Dataset",
    rotate=25,
)
scatterplot_with_annotations(
    retention_df,
    x="reference_value",
    y="relative_gain",
    hue="retention_target",
    annotate_column="dataset",
    ax=axes[1],
    title="(b) Gain retention",
    ylabel="Relative gain",
    xlabel="Full-cue reference score",
)
saved_paths = export_figure(fig, "04_cue_ablation", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_full_results() -> list[dict]:
    return [
        md_cell(
            """
# A01 Full Results

This notebook rebuilds the appendix full-cutoff results view and provides a compact heatmap of RouteRec deltas against the strongest baseline.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
CSV_PATH = RESULTS_ROOT / "A01_full_results/A01_full_results.csv"
REQUIRED_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "metric", "cutoff", "value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]


def demo_full_results() -> pd.DataFrame:
    rows = []
    demo_values = {
        ("Beauty", "RouteRec"): [0.0988, 0.1256, 0.1589, 0.0745, 0.0826, 0.0896, 0.0687, 0.0739, 0.0763],
        ("Beauty", "FEARec"): [0.0952, 0.1219, 0.1548, 0.0717, 0.0798, 0.0871, 0.0661, 0.0713, 0.0741],
        ("Foursquare", "RouteRec"): [0.2049, 0.2447, 0.2939, 0.1457, 0.1589, 0.1718, 0.1339, 0.1400, 0.1430],
        ("Foursquare", "SASRec"): [0.1964, 0.2361, 0.2850, 0.1396, 0.1527, 0.1655, 0.1282, 0.1344, 0.1378],
        ("KuaiRec", "RouteRec"): [0.1451, 0.1832, 0.2287, 0.1138, 0.1264, 0.1376, 0.1079, 0.1147, 0.1185],
        ("KuaiRec", "DuoRec"): [0.1434, 0.1809, 0.2261, 0.1116, 0.1241, 0.1352, 0.1056, 0.1123, 0.1164],
    }
    metric_specs = [("HR", 5), ("HR", 10), ("HR", 20), ("NDCG", 5), ("NDCG", 10), ("NDCG", 20), ("MRR", 5), ("MRR", 10), ("MRR", 20)]
    for (dataset, model), values in demo_values.items():
        for (metric, cutoff), value in zip(metric_specs, values):
            rows.append(
                {
                    "paper_section": "A01_full_results",
                    "panel": "full_results",
                    "dataset": dataset,
                    "variant_or_model": model,
                    "metric": metric,
                    "cutoff": cutoff,
                    "value": value,
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                }
            )
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
df, mode = load_csv_or_demo(CSV_PATH, REQUIRED_COLUMNS, demo_builder=demo_full_results)
display(Markdown(f"**Load mode:** {mode}"))
display(df.head())
"""
        ),
        code_cell(
            """
preview = df.copy()
preview["metric_at_k"] = preview["metric"] + "@" + preview["cutoff"].astype(str)
display(preview.pivot_table(index=["dataset", "variant_or_model"], columns="metric_at_k", values="value"))

metric_heatmap = preview.copy()
route_df = metric_heatmap[metric_heatmap["variant_or_model"] == "RouteRec"].copy()
baseline_df = metric_heatmap[metric_heatmap["variant_or_model"] != "RouteRec"].groupby(["dataset", "metric_at_k"], as_index=False)["value"].max().rename(columns={"value": "best_baseline"})
delta_df = route_df.merge(baseline_df, on=["dataset", "metric_at_k"], how="inner")
delta_df["delta"] = delta_df["value"] - delta_df["best_baseline"]

fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
heatmap_from_long(delta_df, index="dataset", columns="metric_at_k", values="delta", ax=ax, title="RouteRec delta vs best baseline", cmap="viridis", fmt=".4f")
saved_paths = export_figure(fig, "A01_full_results_delta_heatmap", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_objective_variants() -> list[dict]:
    return [
        md_cell(
            """
# A02 Objective Variants

This notebook compares quality, consistency, and stability across objective variants.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
CSV_PATH = RESULTS_ROOT / "A02_objective_variants/A02_objective_variants.csv"
REQUIRED_COLUMNS = [
    "paper_section", "panel", "dataset", "variant_or_model", "objective_variant", "quality_metric", "quality_value",
    "consistency_metric", "consistency_value", "stability_metric", "stability_value", "split", "selection_rule",
    "run_id", "source_path", "notes",
]


def demo_objectives() -> pd.DataFrame:
    rows = [
        {"objective_variant": "no_aux", "quality_value": 0.1112, "consistency_value": 0.41, "stability_value": 0.67},
        {"objective_variant": "knn_only", "quality_value": 0.1146, "consistency_value": 0.53, "stability_value": 0.69},
        {"objective_variant": "z_only", "quality_value": 0.1138, "consistency_value": 0.46, "stability_value": 0.77},
        {"objective_variant": "balance_only", "quality_value": 0.1127, "consistency_value": 0.44, "stability_value": 0.73},
        {"objective_variant": "consistency_plus_z", "quality_value": 0.1170, "consistency_value": 0.58, "stability_value": 0.80},
        {"objective_variant": "full", "quality_value": 0.1185, "consistency_value": 0.61, "stability_value": 0.83},
    ]
    enriched = []
    for row in rows:
        enriched.append(
            {
                "paper_section": "A02_objective_variants",
                "panel": "objective",
                "dataset": "KuaiRec",
                "variant_or_model": "RouteRec",
                "objective_variant": row["objective_variant"],
                "quality_metric": "MRR@20",
                "quality_value": row["quality_value"],
                "consistency_metric": "consistency",
                "consistency_value": row["consistency_value"],
                "stability_metric": "stability",
                "stability_value": row["stability_value"],
                "split": "test",
                "selection_rule": "demo",
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            }
        )
    return pd.DataFrame(enriched)
"""
        ),
        code_cell(
            """
df, mode = load_csv_or_demo(CSV_PATH, REQUIRED_COLUMNS, demo_builder=demo_objectives)
display(Markdown(f"**Load mode:** {mode}"))
display(df)
"""
        ),
        code_cell(
            """
melted = pd.concat(
    [
        df[["objective_variant", "quality_value"]].rename(columns={"quality_value": "value"}).assign(measure="quality"),
        df[["objective_variant", "consistency_value"]].rename(columns={"consistency_value": "value"}).assign(measure="consistency"),
        df[["objective_variant", "stability_value"]].rename(columns={"stability_value": "value"}).assign(measure="stability"),
    ],
    ignore_index=True,
)

fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
grouped_barplot(melted, x="objective_variant", hue="measure", y="value", ax=ax, title="Objective variants", ylabel="Value", xlabel="Objective variant", rotate=25)
saved_paths = export_figure(fig, "A02_objective_variants", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_routing_diagnostics() -> list[dict]:
    return [
        md_cell(
            """
# A03 Routing Diagnostics

This notebook combines expert usage, entropy, stage consistency, and feature-bucket routing patterns into one diagnostic figure.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
USAGE_PATH = RESULTS_ROOT / "A03_routing_diagnostics/A03a_expert_usage.csv"
ENTROPY_PATH = RESULTS_ROOT / "A03_routing_diagnostics/A03b_entropy_effective_experts.csv"
CONSISTENCY_PATH = RESULTS_ROOT / "A03_routing_diagnostics/A03c_stage_consistency.csv"
FEATURE_PATH = RESULTS_ROOT / "A03_routing_diagnostics/A03d_feature_bucket_patterns.csv"

USAGE_COLUMNS = ["paper_section", "panel", "dataset", "stage", "variant_or_model", "expert_id", "group_id", "usage_weight", "usage_rank", "run_id", "source_path", "notes"]
ENTROPY_COLUMNS = ["paper_section", "panel", "dataset", "stage", "variant_or_model", "measure", "value", "run_id", "source_path", "notes"]
CONSISTENCY_COLUMNS = ["paper_section", "panel", "dataset", "stage", "variant_or_model", "similarity_bucket", "consistency_value", "run_id", "source_path", "notes"]
FEATURE_COLUMNS = ["paper_section", "panel", "dataset", "stage", "variant_or_model", "feature_family", "bucket_label", "expert_id", "route_weight", "run_id", "source_path", "notes"]


def demo_usage() -> pd.DataFrame:
    rows = []
    for stage, weights in {"macro": [0.12, 0.14, 0.11, 0.13], "mid": [0.18, 0.24, 0.20, 0.19], "micro": [0.26, 0.31, 0.24, 0.19]}.items():
        for expert_id, weight in enumerate(weights, start=1):
            rows.append({
                "paper_section": "A03_routing_diagnostics",
                "panel": "usage",
                "dataset": "KuaiRec",
                "stage": stage,
                "variant_or_model": "RouteRec",
                "expert_id": expert_id,
                "group_id": 1 if expert_id <= 2 else 2,
                "usage_weight": weight,
                "usage_rank": expert_id,
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            })
    return pd.DataFrame(rows)


def demo_entropy() -> pd.DataFrame:
    rows = []
    values = [("macro", "entropy", 1.72), ("macro", "effective_experts", 3.4), ("mid", "entropy", 1.41), ("mid", "effective_experts", 2.8), ("micro", "entropy", 1.12), ("micro", "effective_experts", 2.2)]
    for stage, measure, value in values:
        rows.append({
            "paper_section": "A03_routing_diagnostics",
            "panel": "entropy",
            "dataset": "KuaiRec",
            "stage": stage,
            "variant_or_model": "RouteRec",
            "measure": measure,
            "value": value,
            "run_id": "demo",
            "source_path": "demo",
            "notes": "demo data",
        })
    return pd.DataFrame(rows)


def demo_consistency() -> pd.DataFrame:
    rows = []
    buckets = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    curves = {
        "macro": [0.45, 0.54, 0.63, 0.71, 0.77],
        "mid": [0.39, 0.47, 0.56, 0.64, 0.71],
        "micro": [0.28, 0.35, 0.44, 0.53, 0.61],
    }
    for stage, values in curves.items():
        for bucket, value in zip(buckets, values):
            rows.append({
                "paper_section": "A03_routing_diagnostics",
                "panel": "consistency",
                "dataset": "KuaiRec",
                "stage": stage,
                "variant_or_model": "RouteRec",
                "similarity_bucket": bucket,
                "consistency_value": value,
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            })
    return pd.DataFrame(rows)


def demo_feature_patterns() -> pd.DataFrame:
    rows = []
    feature_families = ["Tempo", "Focus", "Memory", "Exposure"]
    bucket_labels = ["low", "mid", "high"]
    for family_index, feature_family in enumerate(feature_families, start=1):
        for bucket_index, bucket_label in enumerate(bucket_labels, start=1):
            rows.append({
                "paper_section": "A03_routing_diagnostics",
                "panel": "feature_patterns",
                "dataset": "KuaiRec",
                "stage": "micro",
                "variant_or_model": "RouteRec",
                "feature_family": feature_family,
                "bucket_label": bucket_label,
                "expert_id": family_index,
                "route_weight": 0.12 + 0.05 * bucket_index + 0.03 * family_index,
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            })
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
usage_df, usage_mode = load_csv_or_demo(USAGE_PATH, USAGE_COLUMNS, demo_builder=demo_usage)
entropy_df, entropy_mode = load_csv_or_demo(ENTROPY_PATH, ENTROPY_COLUMNS, demo_builder=demo_entropy)
consistency_df, consistency_mode = load_csv_or_demo(CONSISTENCY_PATH, CONSISTENCY_COLUMNS, demo_builder=demo_consistency)
feature_df, feature_mode = load_csv_or_demo(FEATURE_PATH, FEATURE_COLUMNS, demo_builder=demo_feature_patterns)
display(Markdown(f"**Load mode:** usage={usage_mode}, entropy={entropy_mode}, consistency={consistency_mode}, feature={feature_mode}"))
"""
        ),
        code_cell(
            """
usage_heatmap = usage_df.copy()
usage_heatmap["expert_label"] = "E" + usage_heatmap["expert_id"].astype(str)

entropy_plot = entropy_df.copy()
entropy_plot["series"] = entropy_plot["stage"] + "-" + entropy_plot["measure"]

bucket_order = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
consistency_df = consistency_df.copy()
consistency_df["similarity_bucket"] = pd.Categorical(consistency_df["similarity_bucket"], categories=bucket_order, ordered=True)
consistency_df = consistency_df.sort_values("similarity_bucket")

feature_heatmap = feature_df.copy()
feature_heatmap["family_bucket"] = feature_heatmap["feature_family"] + "-" + feature_heatmap["bucket_label"]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
heatmap_from_long(usage_heatmap, index="stage", columns="expert_label", values="usage_weight", ax=axes[0, 0], title="(a) Expert usage by stage", cmap="mako", fmt=".2f")
grouped_barplot(entropy_plot, x="stage", hue="measure", y="value", ax=axes[0, 1], title="(b) Entropy and effective experts", ylabel="Value", xlabel="Stage")
lineplot_with_markers(consistency_df, x="similarity_bucket", y="consistency_value", hue="stage", ax=axes[1, 0], title="(c) Stage-wise consistency", ylabel="Consistency", xlabel="Similarity bucket", annotate_points=True)
heatmap_from_long(feature_heatmap, index="feature_family", columns="bucket_label", values="route_weight", ax=axes[1, 1], title="(d) Feature-bucket patterns", cmap="rocket", fmt=".2f")
saved_paths = export_figure(fig, "A03_routing_diagnostics", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_behavior_slices() -> list[dict]:
    return [
        md_cell(
            """
# A04 Behavior Slices

This notebook localizes RouteRec gains to behavioral regimes and compares gain with routing concentration.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
METRICS_PATH = RESULTS_ROOT / "A04_behavior_slices/A04a_slice_metrics.csv"
GAIN_PATH = RESULTS_ROOT / "A04_behavior_slices/A04b_slice_gain_concentration.csv"
METRIC_COLUMNS = [
    "paper_section", "panel", "dataset", "slice_name", "variant_or_model", "metric", "cutoff", "value", "sample_count",
    "split", "selection_rule", "run_id", "source_path", "notes",
]
GAIN_COLUMNS = [
    "paper_section", "panel", "dataset", "slice_name", "variant_or_model", "measure", "value", "split", "selection_rule",
    "run_id", "source_path", "notes",
]


def demo_slice_metrics() -> pd.DataFrame:
    rows = []
    slices = ["repeat-heavy", "fast-tempo", "narrow-focus", "exploration-heavy"]
    values = {
        "RouteRec": [0.079, 0.085, 0.082, 0.076],
        "SASRec": [0.073, 0.079, 0.078, 0.071],
        "BSARec": [0.075, 0.081, 0.079, 0.072],
    }
    for model, scores in values.items():
        for slice_name, value in zip(slices, scores):
            rows.append({
                "paper_section": "A04_behavior_slices",
                "panel": "slice_metrics",
                "dataset": "Beauty",
                "slice_name": slice_name,
                "variant_or_model": model,
                "metric": "MRR",
                "cutoff": 20,
                "value": value,
                "sample_count": 1000,
                "split": "test",
                "selection_rule": "demo",
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            })
    return pd.DataFrame(rows)


def demo_slice_gain() -> pd.DataFrame:
    rows = []
    slice_points = {
        "repeat-heavy": (0.71, 0.020),
        "fast-tempo": (0.77, 0.028),
        "narrow-focus": (0.69, 0.017),
        "exploration-heavy": (0.81, 0.031),
    }
    for slice_name, (concentration, relative_gain) in slice_points.items():
        rows.extend(
            [
                {
                    "paper_section": "A04_behavior_slices",
                    "panel": "slice_gain",
                    "dataset": "Beauty",
                    "slice_name": slice_name,
                    "variant_or_model": "RouteRec",
                    "measure": "route_concentration",
                    "value": concentration,
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                },
                {
                    "paper_section": "A04_behavior_slices",
                    "panel": "slice_gain",
                    "dataset": "Beauty",
                    "slice_name": slice_name,
                    "variant_or_model": "RouteRec",
                    "measure": "relative_gain",
                    "value": relative_gain,
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                },
            ]
        )
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
metrics_df, metrics_mode = load_csv_or_demo(METRICS_PATH, METRIC_COLUMNS, demo_builder=demo_slice_metrics)
gain_df, gain_mode = load_csv_or_demo(GAIN_PATH, GAIN_COLUMNS, demo_builder=demo_slice_gain)
display(Markdown(f"**Load mode:** metrics={metrics_mode}, gain={gain_mode}"))
"""
        ),
        code_cell(
            """
gain_pivot = gain_df.pivot_table(index=["dataset", "slice_name", "variant_or_model"], columns="measure", values="value").reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
grouped_barplot(metrics_df, x="slice_name", hue="variant_or_model", y="value", ax=axes[0], title="(a) Slice-wise ranking quality", ylabel="MRR@20", xlabel="Behavior slice", rotate=25)
scatterplot_with_annotations(gain_pivot, x="route_concentration", y="relative_gain", hue="slice_name", annotate_column="slice_name", ax=axes[1], title="(b) Gain vs concentration", ylabel="Relative gain", xlabel="Route concentration")
saved_paths = export_figure(fig, "A04_behavior_slices", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def build_transfer() -> list[dict]:
    return [
        md_cell(
            """
# A05 Transfer

This notebook combines the transfer matrix, low-resource curves, and transfer-variant comparisons.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
MATRIX_PATH = RESULTS_ROOT / "A05_transfer/A05a_transfer_matrix.csv"
LOW_RESOURCE_PATH = RESULTS_ROOT / "A05_transfer/A05b_low_resource_transfer.csv"
VARIANTS_PATH = RESULTS_ROOT / "A05_transfer/A05c_transfer_variants.csv"
MATRIX_COLUMNS = [
    "paper_section", "panel", "source_dataset", "target_dataset", "variant_or_model", "metric", "cutoff", "value",
    "split", "selection_rule", "run_id", "source_path", "notes",
]
LOW_RESOURCE_COLUMNS = [
    "paper_section", "panel", "source_dataset", "target_dataset", "target_fraction", "variant_or_model", "metric",
    "cutoff", "value", "split", "selection_rule", "run_id", "source_path", "notes",
]
VARIANT_COLUMNS = [
    "paper_section", "panel", "source_dataset", "target_dataset", "variant_group", "variant_or_model", "metric",
    "cutoff", "value", "split", "selection_rule", "run_id", "source_path", "notes",
]


def demo_matrix() -> pd.DataFrame:
    rows = []
    datasets = ["Beauty", "Foursquare", "KuaiRec"]
    values = {
        ("Beauty", "Beauty"): 0.076,
        ("Beauty", "Foursquare"): 0.119,
        ("Beauty", "KuaiRec"): 0.112,
        ("Foursquare", "Beauty"): 0.072,
        ("Foursquare", "Foursquare"): 0.143,
        ("Foursquare", "KuaiRec"): 0.117,
        ("KuaiRec", "Beauty"): 0.071,
        ("KuaiRec", "Foursquare"): 0.136,
        ("KuaiRec", "KuaiRec"): 0.119,
    }
    for source_dataset in datasets:
        for target_dataset in datasets:
            rows.append(
                {
                    "paper_section": "A05_transfer",
                    "panel": "matrix",
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "variant_or_model": "router_transfer",
                    "metric": "MRR",
                    "cutoff": 20,
                    "value": values[(source_dataset, target_dataset)],
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                }
            )
    return pd.DataFrame(rows)


def demo_low_resource() -> pd.DataFrame:
    rows = []
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    curves = {
        "scratch": [0.061, 0.067, 0.071, 0.074, 0.076],
        "router_transfer": [0.066, 0.071, 0.074, 0.075, 0.076],
        "full_transfer": [0.069, 0.073, 0.075, 0.076, 0.077],
    }
    for variant, values in curves.items():
        for target_fraction, value in zip(fractions, values):
            rows.append(
                {
                    "paper_section": "A05_transfer",
                    "panel": "low_resource",
                    "source_dataset": "KuaiRec",
                    "target_dataset": "Beauty",
                    "target_fraction": target_fraction,
                    "variant_or_model": variant,
                    "metric": "MRR",
                    "cutoff": 20,
                    "value": value,
                    "split": "test",
                    "selection_rule": "demo",
                    "run_id": "demo",
                    "source_path": "demo",
                    "notes": "demo data",
                }
            )
    return pd.DataFrame(rows)


def demo_variants() -> pd.DataFrame:
    rows = []
    variant_rows = [
        ("router_granularity", "group_router", 0.073),
        ("router_granularity", "full_router", 0.076),
        ("training_mode", "frozen", 0.070),
        ("training_mode", "finetuned", 0.077),
        ("training_mode", "anchor", 0.074),
    ]
    for variant_group, variant_or_model, value in variant_rows:
        rows.append(
            {
                "paper_section": "A05_transfer",
                "panel": "variants",
                "source_dataset": "KuaiRec",
                "target_dataset": "Beauty",
                "variant_group": variant_group,
                "variant_or_model": variant_or_model,
                "metric": "MRR",
                "cutoff": 20,
                "value": value,
                "split": "test",
                "selection_rule": "demo",
                "run_id": "demo",
                "source_path": "demo",
                "notes": "demo data",
            }
        )
    return pd.DataFrame(rows)
"""
        ),
        code_cell(
            """
matrix_df, matrix_mode = load_csv_or_demo(MATRIX_PATH, MATRIX_COLUMNS, demo_builder=demo_matrix)
low_df, low_mode = load_csv_or_demo(LOW_RESOURCE_PATH, LOW_RESOURCE_COLUMNS, demo_builder=demo_low_resource)
variant_df, variant_mode = load_csv_or_demo(VARIANTS_PATH, VARIANT_COLUMNS, demo_builder=demo_variants)
display(Markdown(f"**Load mode:** matrix={matrix_mode}, low_resource={low_mode}, variants={variant_mode}"))
"""
        ),
        code_cell(
            """
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
heatmap_from_long(matrix_df, index="source_dataset", columns="target_dataset", values="value", ax=axes[0], title="(a) Transfer matrix", cmap="viridis", fmt=".3f")
lineplot_with_markers(low_df, x="target_fraction", y="value", hue="variant_or_model", ax=axes[1], title="(b) Low-resource transfer", ylabel="MRR@20", xlabel="Target fraction", annotate_points=True)
grouped_barplot(variant_df, x="variant_group", hue="variant_or_model", y="value", ax=axes[2], title="(c) Transfer variants", ylabel="MRR@20", xlabel="Variant group")
saved_paths = export_figure(fig, "A05_transfer", RESULTS_ROOT)
display(Markdown("Saved figures: " + ", ".join(str(path) for path in saved_paths)))
plt.show()
"""
        ),
    ]


def main() -> None:
    specs = {
        "01_main_overall/01_main_overall.ipynb": build_main_overall(),
        "02_routing_control/02_routing_control.ipynb": build_routing_control(),
        "03_stage_structure/03_stage_structure.ipynb": build_stage_structure(),
        "04_cue_ablation/04_cue_ablation.ipynb": build_cue_ablation(),
        "A01_full_results/A01_full_results.ipynb": build_full_results(),
        "A02_objective_variants/A02_objective_variants.ipynb": build_objective_variants(),
        "A03_routing_diagnostics/A03_routing_diagnostics.ipynb": build_routing_diagnostics(),
        "A04_behavior_slices/A04_behavior_slices.ipynb": build_behavior_slices(),
        "A05_transfer/A05_transfer.ipynb": build_transfer(),
    }

    for relative_path, cells in specs.items():
        write_notebook(relative_path, cells)
        print(f"Wrote {relative_path}")


if __name__ == "__main__":
    main()