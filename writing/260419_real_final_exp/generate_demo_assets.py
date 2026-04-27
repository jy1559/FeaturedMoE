#!/usr/bin/env python3
"""Generate preview-friendly demo CSVs and notebooks for 260419_real_final_exp."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/workspace/FeaturedMoE/writing/260419_real_final_exp")
DATA_DIR = ROOT / "data"
APP_ROOT = ROOT / "appendix"
APP_DATA_DIR = APP_ROOT / "data"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_notebook(path: Path, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def build_main_csvs() -> None:
    status = "demo_dummy"
    note = "Preview-only synthetic values for notebook design checks."
    q2_rows = []
    q2_vals = {
        "beauty": {"Shared FFN": (0.074, 0.148), "Hidden only": (0.078, 0.154), "Fusion bias": (0.081, 0.160), "Mixed": (0.085, 0.166), "Behavior-guided": (0.091, 0.173)},
        "foursquare": {"Shared FFN": (0.197, 0.312), "Hidden only": (0.201, 0.317), "Fusion bias": (0.203, 0.321), "Mixed": (0.206, 0.325), "Behavior-guided": (0.213, 0.336)},
        "KuaiRecLargeStrictPosV2_0.2": {"Shared FFN": (0.091, 0.105), "Hidden only": (0.092, 0.107), "Fusion bias": (0.093, 0.108), "Mixed": (0.095, 0.110), "Behavior-guided": (0.100, 0.116)},
        "movielens1m": {"Shared FFN": (0.083, 0.155), "Hidden only": (0.084, 0.157), "Fusion bias": (0.085, 0.158), "Mixed": (0.086, 0.160), "Behavior-guided": (0.088, 0.164)},
    }
    for dataset, mapping in q2_vals.items():
        for i, (variant, (ndcg, hit)) in enumerate(mapping.items(), start=1):
            q2_rows.append({"dataset": dataset, "variant_label": variant, "test_ndcg20": ndcg, "test_hit10": hit, "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note, "plot_group": "main_q2", "variant_order": i})
    write_csv(DATA_DIR / "q2_quality.csv", q2_rows, ["dataset", "variant_label", "test_ndcg20", "test_hit10", "base_rank", "seed_id", "data_status", "data_note", "plot_group", "variant_order"])

    q2_prof = []
    for dataset in ["beauty", "foursquare", "KuaiRecLargeStrictPosV2_0.2"]:
        for stage, masses in {
            "macro": {"memory": 0.34, "focus": 0.22, "tempo": 0.18, "exposure": 0.26},
            "mid": {"memory": 0.22, "focus": 0.37, "tempo": 0.17, "exposure": 0.24},
            "micro": {"memory": 0.16, "focus": 0.25, "tempo": 0.42, "exposure": 0.17},
        }.items():
            for family, share in masses.items():
                q2_prof.append({"dataset": dataset, "setting_key": "behavior_guided", "stage_name": stage, "routed_family": family, "usage_share": share, "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note})
    write_csv(DATA_DIR / "q2_routing_profile.csv", q2_prof, ["dataset", "setting_key", "stage_name", "routed_family", "usage_share", "base_rank", "seed_id", "data_status", "data_note"])

    q3_temp = []
    q3_org = []
    temporal = ["Single-view", "Best 2-view", "Final 3-stage"]
    temporal_vals = {
        "beauty": [0.080, 0.086, 0.091],
        "foursquare": [0.198, 0.206, 0.213],
        "KuaiRecLargeStrictPosV2_0.2": [0.094, 0.097, 0.100],
    }
    for dataset, vals in temporal_vals.items():
        for idx, (label, ndcg) in enumerate(zip(temporal, vals), start=1):
            q3_temp.append({"dataset": dataset, "variant_label": label, "variant_group": "temporal_decomp", "variant_order": idx, "test_ndcg20": ndcg, "test_hit10": ndcg * 1.65, "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note})
    org_vals = {
        "beauty": {"Semantic grouped": 0.091, "Shuffled grouping": 0.086, "Flat scalar bag": 0.083, "Random group assignment": 0.081},
        "foursquare": {"Semantic grouped": 0.213, "Shuffled grouping": 0.208, "Flat scalar bag": 0.204, "Random group assignment": 0.201},
        "KuaiRecLargeStrictPosV2_0.2": {"Semantic grouped": 0.100, "Shuffled grouping": 0.097, "Flat scalar bag": 0.095, "Random group assignment": 0.093},
    }
    for dataset, mapping in org_vals.items():
        for idx, (label, ndcg) in enumerate(mapping.items(), start=1):
            q3_org.append({"dataset": dataset, "variant_label": label, "variant_group": "routing_org", "variant_order": idx, "test_ndcg20": ndcg, "test_hit10": ndcg * 1.72, "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note})
    write_csv(DATA_DIR / "q3_temporal_decomp.csv", q3_temp, ["dataset", "variant_label", "variant_group", "variant_order", "test_ndcg20", "test_hit10", "base_rank", "seed_id", "data_status", "data_note"])
    write_csv(DATA_DIR / "q3_routing_org.csv", q3_org, ["dataset", "variant_label", "variant_group", "variant_order", "test_ndcg20", "test_hit10", "base_rank", "seed_id", "data_status", "data_note"])

    q4_rows = [
        {"question": "q4", "dataset_scope": "dynamic", "dataset": "beauty", "model_name": "SASRec", "model": "sasrec", "status": "demo_dummy", "error": "", "benchmark_epochs": 1, "build_sec": 4.2, "train_sec": 11.8, "infer_sec": 1.90, "total_params": 1254016, "active_params": 1254016, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.00, "infer_time_ratio": 1.00, "data_status": status, "data_note": note},
        {"question": "q4", "dataset_scope": "dynamic", "dataset": "beauty", "model_name": "RouteRec-dense", "model": "featured_moe_n3", "status": "demo_dummy", "error": "", "benchmark_epochs": 1, "build_sec": 4.9, "train_sec": 14.1, "infer_sec": 2.31, "total_params": 1782400, "active_params": 1782400, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.19, "infer_time_ratio": 1.22, "data_status": status, "data_note": note},
        {"question": "q4", "dataset_scope": "dynamic", "dataset": "beauty", "model_name": "RouteRec-sparse-final", "model": "featured_moe_n3", "status": "demo_dummy", "error": "", "benchmark_epochs": 1, "build_sec": 5.0, "train_sec": 13.0, "infer_sec": 2.02, "total_params": 1782400, "active_params": 1493200, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.10, "infer_time_ratio": 1.06, "data_status": status, "data_note": note},
        {"question": "q4", "dataset_scope": "stable", "dataset": "movielens1m", "model_name": "SASRec", "model": "sasrec", "status": "demo_dummy", "error": "", "benchmark_epochs": 1, "build_sec": 7.9, "train_sec": 19.4, "infer_sec": 3.4, "total_params": 1254016, "active_params": 1254016, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.00, "infer_time_ratio": 1.00, "data_status": status, "data_note": note},
        {"question": "q4", "dataset_scope": "stable", "dataset": "movielens1m", "model_name": "RouteRec-dense", "model": "featured_moe_n3", "status": "demo_dummy", "error": "", "benchmark_epochs": 1, "build_sec": 8.4, "train_sec": 23.1, "infer_sec": 4.0, "total_params": 1782400, "active_params": 1782400, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.19, "infer_time_ratio": 1.18, "data_status": status, "data_note": note},
        {"question": "q4", "dataset_scope": "stable", "dataset": "movielens1m", "model_name": "RouteRec-sparse-final", "model": "featured_moe_n3", "status": "demo_dummy", "error": "", "benchmark_epochs": 1, "build_sec": 8.5, "train_sec": 21.0, "infer_sec": 3.6, "total_params": 1782400, "active_params": 1493200, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.08, "infer_time_ratio": 1.06, "data_status": status, "data_note": note},
    ]
    write_csv(DATA_DIR / "q4_efficiency_table.csv", q4_rows, list(q4_rows[0].keys()))

    q5_case = []
    cases = {
        "Repeat-heavy": {"macro": {"memory": 0.44, "focus": 0.19, "tempo": 0.14, "exposure": 0.23}, "mid": {"memory": 0.35, "focus": 0.24, "tempo": 0.17, "exposure": 0.24}, "micro": {"memory": 0.29, "focus": 0.18, "tempo": 0.34, "exposure": 0.19}},
        "Fast exploratory": {"macro": {"memory": 0.18, "focus": 0.22, "tempo": 0.39, "exposure": 0.21}, "mid": {"memory": 0.15, "focus": 0.26, "tempo": 0.41, "exposure": 0.18}, "micro": {"memory": 0.11, "focus": 0.21, "tempo": 0.51, "exposure": 0.17}},
        "Narrow-focus": {"macro": {"memory": 0.20, "focus": 0.42, "tempo": 0.16, "exposure": 0.22}, "mid": {"memory": 0.18, "focus": 0.46, "tempo": 0.15, "exposure": 0.21}, "micro": {"memory": 0.16, "focus": 0.43, "tempo": 0.19, "exposure": 0.22}},
    }
    for dataset in ["beauty", "foursquare", "KuaiRecLargeStrictPosV2_0.2"]:
        for case_name, stage_map in cases.items():
            case_type = case_name.lower().replace("-", "_").replace(" ", "_")
            for stage, fam_map in stage_map.items():
                for family, mass in fam_map.items():
                    q5_case.append({"dataset": dataset, "case_name": case_name, "case_type": case_type, "stage": stage, "group_name": family, "expert_rank_or_slot": "group_total", "selected_mass": mass, "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note})
    write_csv(DATA_DIR / "q5_case_heatmap.csv", q5_case, ["dataset", "case_name", "case_type", "stage", "group_name", "expert_rank_or_slot", "selected_mass", "base_rank", "seed_id", "data_status", "data_note"])

    q5_intervention = []
    for dataset, base in {"beauty": 0.091, "foursquare": 0.213, "KuaiRecLargeStrictPosV2_0.2": 0.100}.items():
        for intervention, delta, fam in [("Full", 0.0, "all"), ("Feature Zero All", -0.013, "all"), ("Feature Shuffle All", -0.009, "all"), ("Repeat Flatten", -0.006, "memory"), ("Switch Boost", -0.004, "focus"), ("Tempo Compress", -0.008, "tempo")]:
            q5_intervention.append({"dataset": dataset, "intervention": intervention.lower().replace(" ", "_"), "intervention_label": intervention, "target_family": fam, "test_mrr20": round(base + delta, 3), "test_seen_mrr20": round(base + delta + 0.002, 3), "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note})
    write_csv(DATA_DIR / "q5_intervention_summary.csv", q5_intervention, ["dataset", "intervention", "intervention_label", "target_family", "test_mrr20", "test_seen_mrr20", "base_rank", "seed_id", "data_status", "data_note"])

    manifest = {"generated_at": "2026-04-19T00:00:00Z", "data_status": status, "data_note": note, "mode": "demo_preview", "files": sorted([p.name for p in DATA_DIR.glob('*') if p.is_file()])}
    (DATA_DIR / "q_suite_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    run_index = []
    for row in q2_rows[:4]:
        run_index.append({"question": "q2", "dataset": row["dataset"], "setting_key": row["variant_label"].lower().replace(" ", "_"), "model_name": "featured_moe_n3", "status": status, "data_note": note})
    for row in q4_rows:
        run_index.append({"question": "q4", "dataset": row["dataset"], "setting_key": row["model_name"].lower().replace("-", "_"), "model_name": row["model_name"], "status": status, "data_note": note})
    write_csv(DATA_DIR / "q_suite_run_index.csv", run_index, ["question", "dataset", "setting_key", "model_name", "status", "data_note"])


def build_appendix_csvs() -> None:
    status = "demo_dummy"
    note = "Preview-only synthetic values for appendix figure and table design checks."
    ds_rows = [
        {"dataset": "beauty", "interactions": 33488, "sessions": 4243, "items": 3625, "avg_session_len": 7.9, "data_status": status, "data_note": note},
        {"dataset": "foursquare", "interactions": 145238, "sessions": 25369, "items": 30588, "avg_session_len": 5.7, "data_status": status, "data_note": note},
        {"dataset": "KuaiRecLargeStrictPosV2_0.2", "interactions": 287411, "sessions": 24458, "items": 6477, "avg_session_len": 11.8, "data_status": status, "data_note": note},
        {"dataset": "movielens1m", "interactions": 575281, "sessions": 14539, "items": 3533, "avg_session_len": 39.6, "data_status": status, "data_note": note},
    ]
    write_csv(APP_DATA_DIR / "appendix_dataset_stats.csv", ds_rows, list(ds_rows[0].keys()))

    full = []
    for dataset, route_mrr in {"beauty": 0.091, "foursquare": 0.213, "KuaiRecLargeStrictPosV2_0.2": 0.100, "movielens1m": 0.088}.items():
        for split, shift in [("valid", -0.002), ("test", 0.0)]:
            vals = {"hit@5": route_mrr * 1.85 + shift, "hit@10": route_mrr * 2.75 + shift, "hit@20": route_mrr * 3.8 + shift, "ndcg@10": route_mrr * 1.07 + shift, "ndcg@20": route_mrr * 1.18 + shift, "mrr@20": route_mrr + shift}
            for metric, value in vals.items():
                full.append({"dataset": dataset, "model": "featured_moe_n3", "base_rank": 1, "base_tag": f"{dataset}_demo", "split": split, "metric": metric, "value": round(value, 4), "result_json": f"/demo/{dataset}.json", "data_status": status, "data_note": note})
    write_csv(APP_DATA_DIR / "appendix_full_results_long.csv", full, list(full[0].keys()))

    structural = []
    for dataset, vals in {
        "beauty": [("Family Prior Intact", 0.091), ("Fewer Semantic Family Groups", 0.087), ("Family Groups Shuffled", 0.085), ("Flattened Scalar Bag", 0.083), ("Correct Temporal Roles", 0.091), ("All Stages Same Scope", 0.086)],
        "KuaiRecLargeStrictPosV2_0.2": [("Family Prior Intact", 0.100), ("Fewer Semantic Family Groups", 0.097), ("Family Groups Shuffled", 0.095), ("Flattened Scalar Bag", 0.094), ("Correct Temporal Roles", 0.100), ("All Stages Same Scope", 0.096)],
    }.items():
        for i, (label, ndcg) in enumerate(vals, start=1):
            structural.append({"dataset": dataset, "variant_label": label, "variant_group": "structural", "variant_order": i, "test_ndcg20": ndcg, "test_hit10": round(ndcg * 1.7, 4), "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note})
    write_csv(APP_DATA_DIR / "appendix_structural_variants.csv", structural, list(structural[0].keys()))

    sparse = []
    diag = []
    sparse_map = [("Dense full mixture", 0.098, 12), ("Flat sparse top-4", 0.096, 4), ("Top-2 groups, top-2 experts per group", 0.097, 4), ("Top-3 groups, top-2 experts per group", 0.100, 6), ("Top-4 groups, top-2 experts per group", 0.099, 8), ("Top-3 groups, top-3 experts per group", 0.099, 9)]
    for dataset in ["beauty", "KuaiRecLargeStrictPosV2_0.2"]:
        for i, (label, mrr, active) in enumerate(sparse_map, start=1):
            setting_key = label.lower().replace(" ", "_").replace(",", "").replace("-", "_")
            sparse.append({"dataset": dataset, "setting_key": setting_key, "setting_label": label, "variant_label": label, "test_seen_mrr20": mrr + (0.0 if dataset == "beauty" else 0.004), "test_ndcg20": mrr + 0.009, "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note, "active_experts": active})
            for stage, entropy, neff in [("macro", 0.78 - i * 0.03, min(active / 2.5, 4.0)), ("mid", 0.72 - i * 0.025, min(active / 2.2, 4.0)), ("micro", 0.66 - i * 0.02, min(active / 2.0, 4.0))]:
                diag.append({"dataset": dataset, "question": "sparse", "setting_key": setting_key, "setting_label": label, "variant_label": label, "stage_key": stage, "group_entropy_mean": round(max(entropy, 0.28), 3), "group_n_eff": round(neff, 3), "group_top1_max_frac": round(0.26 + i * 0.03, 3), "data_status": status, "data_note": note})
    write_csv(APP_DATA_DIR / "appendix_sparse_tradeoff.csv", sparse, list(sparse[0].keys()))
    write_csv(APP_DATA_DIR / "appendix_sparse_diagnostics.csv", diag, list(diag[0].keys()))
    write_csv(APP_DATA_DIR / "appendix_routing_diagnostics.csv", diag, list(diag[0].keys()))

    objective = []
    objective_special = []
    obj_rows = [("Full objective", 0.100), ("No auxiliary loss", 0.094), ("Consistency only", 0.097), ("z-loss only", 0.095), ("Balance only", 0.093), ("Consistency + z-loss", 0.099)]
    for dataset in ["beauty", "KuaiRecLargeStrictPosV2_0.2"]:
        for i, (label, mrr) in enumerate(obj_rows, start=1):
            setting_key = label.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
            objective.append({"dataset": dataset, "setting_key": setting_key, "setting_label": label, "variant_label": label, "test_seen_mrr20": mrr + (0.0 if dataset == "beauty" else 0.004), "test_ndcg20": mrr + 0.01, "base_rank": 1, "seed_id": 1, "data_status": status, "data_note": note})
            for block, metric, value in [("consistency", "knn_score", 0.94 + i * 0.008), ("stability", "entropy", 0.72 - i * 0.03)]:
                objective_special.append({"dataset": dataset, "question": "objective", "setting_key": setting_key, "split": "test", "special_block": block, "metric": metric, "value": round(value, 4), "data_status": status, "data_note": note})
    write_csv(APP_DATA_DIR / "appendix_objective_variants.csv", objective, list(objective[0].keys()))
    write_csv(APP_DATA_DIR / "appendix_objective_special_metrics.csv", objective_special, list(objective_special[0].keys()))

    cost = [
        {"question": "cost", "dataset_scope": "dynamic", "dataset": "beauty", "model_name": "SASRec", "model": "sasrec", "status": status, "error": "", "benchmark_epochs": 1, "build_sec": 4.2, "train_sec": 11.8, "infer_sec": 1.90, "total_params": 1254016, "active_params": 1254016, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.00, "infer_time_ratio": 1.00, "data_status": status, "data_note": note},
        {"question": "cost", "dataset_scope": "dynamic", "dataset": "beauty", "model_name": "RouteRec-sparse-final", "model": "featured_moe_n3", "status": status, "error": "", "benchmark_epochs": 1, "build_sec": 5.0, "train_sec": 13.0, "infer_sec": 2.02, "total_params": 1782400, "active_params": 1493200, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.10, "infer_time_ratio": 1.06, "data_status": status, "data_note": note},
        {"question": "cost", "dataset_scope": "stable", "dataset": "movielens1m", "model_name": "SASRec", "model": "sasrec", "status": status, "error": "", "benchmark_epochs": 1, "build_sec": 7.9, "train_sec": 19.4, "infer_sec": 3.4, "total_params": 1254016, "active_params": 1254016, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.00, "infer_time_ratio": 1.00, "data_status": status, "data_note": note},
        {"question": "cost", "dataset_scope": "stable", "dataset": "movielens1m", "model_name": "RouteRec-sparse-final", "model": "featured_moe_n3", "status": status, "error": "", "benchmark_epochs": 1, "build_sec": 8.5, "train_sec": 21.0, "infer_sec": 3.6, "total_params": 1782400, "active_params": 1493200, "timestamp_utc": "2026-04-19T00:00:00Z", "train_time_ratio": 1.08, "infer_time_ratio": 1.06, "data_status": status, "data_note": note},
    ]
    write_csv(APP_DATA_DIR / "appendix_cost_summary.csv", cost, list(cost[0].keys()))

    special_bins = []
    for dataset, base in {"beauty": 0.091, "foursquare": 0.213, "KuaiRecLargeStrictPosV2_0.2": 0.100}.items():
        for group, val in [("short_session", base - 0.008), ("medium_session", base), ("long_session", base + 0.006), ("rare_target", base - 0.004), ("mid_target", base + 0.001), ("frequent_target", base + 0.004)]:
            special_bins.append({"dataset": dataset, "group": group, "metric": "mrr@20", "split": "test", "best_valid_seen_mrr20": round(val - 0.002, 3), "test_seen_mrr20": round(val, 3), "data_status": status, "data_note": note})
    write_csv(APP_DATA_DIR / "appendix_special_bins.csv", special_bins, list(special_bins[0].keys()))

    behavior_q = []
    behavior_p = []
    case_routing = []
    diag_case = []
    for dataset, base in {"beauty": 0.091, "foursquare": 0.213, "KuaiRecLargeStrictPosV2_0.2": 0.100}.items():
        for group, mrr, conc in [("memory_plus", base + 0.007, 0.61), ("focus_plus", base + 0.004, 0.58), ("tempo_plus", base + 0.010, 0.67), ("exposure_minus", base - 0.003, 0.49)]:
            behavior_q.append({"dataset": dataset, "group": group, "eval_split": "test", "test_seen_mrr20": round(mrr, 3), "route_concentration": conc, "data_status": status, "data_note": note})
        for feat, val in [("repeat_ratio", 0.44), ("switch_rate", 0.29), ("mean_gap", 0.18), ("pop_entropy", 0.37)]:
            behavior_p.append({"dataset": dataset, "group": "tempo_plus", "feature_name": feat, "feature_value": val, "data_status": status, "data_note": note})
        for stage, fams in {"macro": {"memory": 0.41, "focus": 0.19, "tempo": 0.18, "exposure": 0.22}, "mid": {"memory": 0.23, "focus": 0.38, "tempo": 0.17, "exposure": 0.22}, "micro": {"memory": 0.15, "focus": 0.22, "tempo": 0.45, "exposure": 0.18}}.items():
            for fam, share in fams.items():
                row = {"dataset": dataset, "group": "tempo_plus", "stage_name": stage, "routed_family": fam, "usage_share": share, "data_status": status, "data_note": note}
                case_routing.append(row)
                diag_case.append(row)
    write_csv(APP_DATA_DIR / "appendix_behavior_slice_quality.csv", behavior_q, list(behavior_q[0].keys()))
    write_csv(APP_DATA_DIR / "appendix_behavior_slice_profiles.csv", behavior_p, list(behavior_p[0].keys()))
    write_csv(APP_DATA_DIR / "appendix_case_routing_profile.csv", case_routing, list(case_routing[0].keys()))
    write_csv(APP_DATA_DIR / "appendix_diagnostic_case_profile.csv", diag_case, list(diag_case[0].keys()))

    interventions = []
    for dataset, base in {"beauty": 0.091, "foursquare": 0.213, "KuaiRecLargeStrictPosV2_0.2": 0.100}.items():
        for label, fam, delta in [("Full", "all", 0.0), ("Feature Zero All", "all", -0.013), ("Feature Shuffle All", "all", -0.009), ("Repeat Flatten", "memory", -0.006), ("Tempo Compress", "tempo", -0.008)]:
            interventions.append({"dataset": dataset, "intervention": label.lower().replace(" ", "_"), "intervention_label": label, "target_family": fam, "test_mrr20": round(base + delta, 3), "test_seen_mrr20": round(base + delta + 0.002, 3), "data_status": status, "data_note": note})
    write_csv(APP_DATA_DIR / "appendix_intervention_summary.csv", interventions, list(interventions[0].keys()))

    transfer = [
        {"dataset": "beauty_to_kuairec", "setting_key": "data_frac_100", "data_fraction": 1.00, "route_mrr20": 0.104, "baseline_mrr20": 0.097, "status": status, "data_status": status, "data_note": note},
        {"dataset": "beauty_to_kuairec", "setting_key": "data_frac_050", "data_fraction": 0.50, "route_mrr20": 0.100, "baseline_mrr20": 0.093, "status": status, "data_status": status, "data_note": note},
        {"dataset": "beauty_to_kuairec", "setting_key": "data_frac_025", "data_fraction": 0.25, "route_mrr20": 0.095, "baseline_mrr20": 0.089, "status": status, "data_status": status, "data_note": note},
        {"dataset": "beauty_to_kuairec", "setting_key": "data_frac_010", "data_fraction": 0.10, "route_mrr20": 0.091, "baseline_mrr20": 0.084, "status": status, "data_status": status, "data_note": note},
    ]
    write_csv(APP_DATA_DIR / "appendix_transfer_summary.csv", transfer, list(transfer[0].keys()))

    selected = [{"dataset": "beauty", "model": "featured_moe_n3", "base_rank": 1, "base_tag": "beauty_demo", "result_json": "/demo/beauty.json", "checkpoint_file": "/demo/beauty.pth", "data_status": status, "data_note": note}]
    write_csv(APP_DATA_DIR / "appendix_selected_runs.csv", selected, list(selected[0].keys()))

    run_index = []
    for row in structural[:4]:
        run_index.append({"question": "structural", "dataset": row["dataset"], "setting_key": row["variant_label"].lower().replace(" ", "_"), "model_name": "featured_moe_n3", "status": status, "data_note": note})
    for row in sparse[:4]:
        run_index.append({"question": "sparse", "dataset": row["dataset"], "setting_key": row["setting_key"], "model_name": "featured_moe_n3", "status": status, "data_note": note})
    write_csv(APP_DATA_DIR / "appendix_run_index.csv", run_index, ["question", "dataset", "setting_key", "model_name", "status", "data_note"])

    manifest = {"generated_at": "2026-04-19T00:00:00Z", "data_status": status, "data_note": note, "mode": "demo_preview", "files": sorted([p.name for p in APP_DATA_DIR.glob('*') if p.is_file()])}
    (APP_DATA_DIR / "appendix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_main_notebooks() -> None:
    write_notebook(
        ROOT / "02_q2_routing_control.ipynb",
        [
            md("# 02 Q2 Routing Control\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "import seaborn as sns\n"
                "from real_final_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, metric_limits, palette_for, panel_label\n"
                "apply_style()\n"
                "quality = load_csv('q2_quality.csv')\n"
                "profile = load_csv('q2_routing_profile.csv')\n"
                "quality['dataset_label'] = quality['dataset'].map(dataset_label)\n"
                "profile['dataset_label'] = profile['dataset'].map(dataset_label)\n"
                "quality['variant_label'] = pd.Categorical(quality['variant_label'], categories=['Shared FFN', 'Hidden only', 'Fusion bias', 'Mixed', 'Behavior-guided'], ordered=True)\n"
                "line_df = quality.sort_values(['variant_label', 'dataset_label']).copy()\n"
                "heat_df = profile.groupby(['stage_name', 'routed_family'], as_index=False)['usage_share'].mean()\n"
                "delta_df = quality.pivot_table(index='dataset_label', columns='variant_label', values='test_ndcg20', aggfunc='mean', observed=False).reset_index()\n"
                "delta_df['gain'] = delta_df['Behavior-guided'] - delta_df['Shared FFN']\n"
                "fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True, gridspec_kw={'width_ratios': [1.35, 1.0, 0.9]})\n"
                "ax = axes[0]\n"
                "variant_order = [str(v) for v in line_df['variant_label'].cat.categories]\n"
                "sns.lineplot(data=line_df, x='dataset_label', y='test_ndcg20', hue='variant_label', hue_order=variant_order, palette=palette_for(variant_order), marker='o', linewidth=2.2, ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('NDCG@20')\n"
                "ax.set_ylim(*metric_limits(line_df['test_ndcg20'], padding=0.22, floor=0.06))\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='lower right', ncol=1)\n"
                "panel_label(ax, 'a')\n"
                "ax = axes[1]\n"
                "heat = heat_df.pivot(index='stage_name', columns='routed_family', values='usage_share').loc[['macro', 'mid', 'micro']]\n"
                "sns.heatmap(heat, cmap=sns.light_palette('#0F766E', as_cmap=True), vmin=0.12, vmax=0.44, annot=True, fmt='.2f', cbar=False, linewidths=0.6, linecolor='white', ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('')\n"
                "panel_label(ax, 'b')\n"
                "ax = axes[2]\n"
                "delta_df = delta_df.sort_values('gain', ascending=True)\n"
                "ax.barh(delta_df['dataset_label'], delta_df['gain'], color='#C96567', edgecolor='white', linewidth=0.8)\n"
                "ax.axvline(0.0, color='#66757D', linewidth=0.9)\n"
                "ax.set_xlabel('NDCG@20 gain')\n"
                "ax.set_ylabel('')\n"
                "ax.set_xlim(*metric_limits(delta_df['gain'], padding=0.24))\n"
                "clean_axes(ax, grid_axis='x')\n"
                "panel_label(ax, 'c')\n"
                "annotate_demo(fig, quality)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        ROOT / "03_q3_design_justification.ipynb",
        [
            md("# 03 Q3 Design Justification\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "import seaborn as sns\n"
                "from real_final_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, metric_limits, palette_for, panel_label\n"
                "apply_style()\n"
                "temporal = load_csv('q3_temporal_decomp.csv')\n"
                "organization = load_csv('q3_routing_org.csv')\n"
                "for df in [temporal, organization]:\n"
                "    df['dataset_label'] = df['dataset'].map(dataset_label)\n"
                "fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), constrained_layout=True)\n"
                "for ax, df, order, label in [\n"
                "    (axes[0], temporal, ['Single-view', 'Best 2-view', 'Final 3-stage'], 'a'),\n"
                "    (axes[1], organization, ['Semantic grouped', 'Shuffled grouping', 'Flat scalar bag', 'Random group assignment'], 'b'),\n"
                "]:\n"
                "    df = df.copy()\n"
                "    df['variant_label'] = pd.Categorical(df['variant_label'], categories=order, ordered=True)\n"
                "    sns.lineplot(data=df.sort_values(['variant_label', 'dataset_label']), x='variant_label', y='test_ndcg20', hue='dataset_label', palette=palette_for(sorted(df['dataset_label'].unique())), marker='o', linewidth=2.2, ax=ax)\n"
                "    ax.set_xlabel('')\n"
                "    ax.set_ylabel('NDCG@20')\n"
                "    ax.tick_params(axis='x', rotation=20)\n"
                "    ax.set_ylim(*metric_limits(df['test_ndcg20'], padding=0.22, floor=0.07))\n"
                "    clean_axes(ax)\n"
                "    ax.legend(loc='best', title='')\n"
                "    panel_label(ax, label)\n"
                "annotate_demo(fig, temporal)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        ROOT / "04_q4_efficiency.ipynb",
        [
            md("# 04 Q4 Efficiency\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "import seaborn as sns\n"
                "from real_final_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, metric_limits, panel_label\n"
                "apply_style()\n"
                "eff = load_csv('q4_efficiency_table.csv')\n"
                "eff['dataset_label'] = eff['dataset'].map(dataset_label)\n"
                "ratio_df = eff.melt(id_vars=['dataset_label', 'model_name', 'data_status'], value_vars=['train_time_ratio', 'infer_time_ratio'], var_name='metric', value_name='ratio')\n"
                "ratio_df['metric'] = ratio_df['metric'].map({'train_time_ratio': 'Train ratio', 'infer_time_ratio': 'Infer ratio'})\n"
                "param_df = eff[['dataset_label', 'model_name', 'total_params', 'active_params']].copy()\n"
                "param_df = param_df[param_df['model_name'] != 'SASRec'].sort_values('dataset_label')\n"
                "fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), constrained_layout=True)\n"
                "ax = axes[0]\n"
                "sns.barplot(data=ratio_df, x='dataset_label', y='ratio', hue='metric', palette=['#5B7C99', '#0F766E'], ax=ax)\n"
                "ax.axhline(1.0, color='#66757D', linewidth=0.9, linestyle='--')\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('Relative runtime')\n"
                "ax.set_ylim(*metric_limits(ratio_df['ratio'], padding=0.22, floor=0.94))\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='upper left', title='')\n"
                "panel_label(ax, 'a')\n"
                "ax = axes[1]\n"
                "ypos = range(len(param_df))\n"
                "ax.hlines(y=list(ypos), xmin=param_df['active_params'] / 1e6, xmax=param_df['total_params'] / 1e6, color='#D9D8D2', linewidth=2.0)\n"
                "ax.scatter(param_df['total_params'] / 1e6, list(ypos), color='#C96567', s=52, label='Total params', zorder=3)\n"
                "ax.scatter(param_df['active_params'] / 1e6, list(ypos), color='#0F766E', s=52, label='Active params', zorder=3)\n"
                "ax.set_yticks(list(ypos), [f\"{d} | {m}\" for d, m in zip(param_df['dataset_label'], param_df['model_name'])])\n"
                "ax.set_xlabel('Parameters (M)')\n"
                "ax.set_ylabel('')\n"
                "clean_axes(ax, grid_axis='x')\n"
                "ax.legend(loc='lower right', title='')\n"
                "panel_label(ax, 'b')\n"
                "annotate_demo(fig, eff)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        ROOT / "05_q5_behavior_semantics.ipynb",
        [
            md("# 05 Q5 Behavior Semantics\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "import seaborn as sns\n"
                "from real_final_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, metric_limits, panel_label\n"
                "apply_style()\n"
                "case_df = load_csv('q5_case_heatmap.csv')\n"
                "interventions = load_csv('q5_intervention_summary.csv')\n"
                "interventions['dataset_label'] = interventions['dataset'].map(dataset_label)\n"
                "base = interventions[interventions['intervention_label'] == 'Full'][['dataset', 'test_seen_mrr20']].rename(columns={'test_seen_mrr20': 'base'})\n"
                "delta_df = interventions.merge(base, on='dataset')\n"
                "delta_df['delta'] = delta_df['test_seen_mrr20'] - delta_df['base']\n"
                "delta_df = delta_df[delta_df['intervention_label'] != 'Full'].copy()\n"
                "delta_df['dataset_label'] = delta_df['dataset'].map(dataset_label)\n"
                "heat_fast = case_df[(case_df['dataset'] == 'KuaiRecLargeStrictPosV2_0.2') & (case_df['case_name'] == 'Fast exploratory')].pivot(index='stage', columns='group_name', values='selected_mass').loc[['macro', 'mid', 'micro']]\n"
                "heat_repeat = case_df[(case_df['dataset'] == 'KuaiRecLargeStrictPosV2_0.2') & (case_df['case_name'] == 'Repeat-heavy')].pivot(index='stage', columns='group_name', values='selected_mass').loc[['macro', 'mid', 'micro']]\n"
                "fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.2), constrained_layout=True, gridspec_kw={'width_ratios': [1.45, 1.0, 1.0]})\n"
                "ax = axes[0]\n"
                "sns.barplot(data=delta_df, x='intervention_label', y='delta', hue='dataset_label', palette=['#5B7C99', '#D28B36', '#0F766E'], ax=ax)\n"
                "ax.axhline(0.0, color='#66757D', linewidth=0.9)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('MRR@20 delta')\n"
                "ax.tick_params(axis='x', rotation=24)\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='lower left', title='')\n"
                "panel_label(ax, 'a')\n"
                "for idx, (ax, heat, label) in enumerate([(axes[1], heat_fast, 'b'), (axes[2], heat_repeat, 'c')]):\n"
                "    sns.heatmap(heat, cmap=sns.light_palette('#0F766E', as_cmap=True), vmin=0.10, vmax=0.52, annot=True, fmt='.2f', cbar=False, linewidths=0.6, linecolor='white', ax=ax)\n"
                "    ax.set_xlabel('')\n"
                "    ax.set_ylabel('' if idx else '')\n"
                "    panel_label(ax, label)\n"
                "annotate_demo(fig, interventions)\n"
                "plt.show()\n"
            ),
        ],
    )


def build_appendix_notebooks() -> None:
    write_notebook(
        APP_ROOT / "A01_appendix_full_results.ipynb",
        [
            md("# A01 Appendix Full Results\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from appendix_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, panel_label\n"
                "apply_style()\n"
                "stats = load_csv('appendix_dataset_stats.csv')\n"
                "results = load_csv('appendix_full_results_long.csv')\n"
                "stats['dataset_label'] = stats['dataset'].map(dataset_label)\n"
                "results['dataset_label'] = results['dataset'].map(dataset_label)\n"
                "pivot = results[results['metric'].isin(['mrr@20', 'ndcg@20', 'hit@20'])].pivot_table(index='dataset_label', columns=['split', 'metric'], values='value', aggfunc='mean')\n"
                "pivot.columns = [f\"{split}\\n{metric}\" for split, metric in pivot.columns]\n"
                "fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True, gridspec_kw={'width_ratios': [1.05, 1.2]})\n"
                "ax = axes[0]\n"
                "stats = stats.sort_values('interactions', ascending=True)\n"
                "ax.barh(stats['dataset_label'], stats['interactions'] / 1000.0, color='#5B7C99', edgecolor='white', linewidth=0.8)\n"
                "ax.set_xlabel('Interactions (K)')\n"
                "ax.set_ylabel('')\n"
                "clean_axes(ax, grid_axis='x')\n"
                "panel_label(ax, 'a')\n"
                "ax = axes[1]\n"
                "sns.heatmap(pivot, cmap=sns.light_palette('#0F766E', as_cmap=True), annot=True, fmt='.3f', linewidths=0.6, linecolor='white', cbar=False, ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('')\n"
                "panel_label(ax, 'b')\n"
                "annotate_demo(fig, results)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        APP_ROOT / "A02_appendix_structural_ablation.ipynb",
        [
            md("# A02 Appendix Structural Ablation\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "import seaborn as sns\n"
                "from appendix_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, metric_limits, palette_for, panel_label\n"
                "apply_style()\n"
                "structural = load_csv('appendix_structural_variants.csv')\n"
                "structural['dataset_label'] = structural['dataset'].map(dataset_label)\n"
                "order = ['Final 3-stage', 'No macro stage', 'No mid stage', 'No micro stage']\n"
                "structural['variant_label'] = pd.Categorical(structural['variant_label'], categories=order, ordered=True)\n"
                "fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.2), constrained_layout=True)\n"
                "for ax, dataset_name, label in [(axes[0], 'Beauty', 'a'), (axes[1], 'KuaiRec', 'b')]:\n"
                "    sub = structural[structural['dataset_label'] == dataset_name].copy()\n"
                "    sub['variant_label'] = sub['variant_label'].astype(str)\n"
                "    sub_labels = [name for name in order if name in set(sub['variant_label'])]\n"
                "    sub = sub.set_index('variant_label').loc[sub_labels].reset_index()\n"
                "    colors = [palette_for(order)[name] for name in sub_labels]\n"
                "    ax.bar(sub['variant_label'], sub['test_ndcg20'], color=colors, edgecolor='white', linewidth=0.8)\n"
                "    ax.set_xlabel('')\n"
                "    ax.set_ylabel('NDCG@20')\n"
                "    ax.tick_params(axis='x', rotation=20)\n"
                "    ax.set_ylim(*metric_limits(sub['test_ndcg20'], padding=0.22, floor=0.07))\n"
                "    clean_axes(ax)\n"
                "    panel_label(ax, label)\n"
                "annotate_demo(fig, structural)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        APP_ROOT / "A03_appendix_sparse_and_diagnostics.ipynb",
        [
            md("# A03 Appendix Sparse And Diagnostics\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from appendix_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, metric_limits, panel_label\n"
                "apply_style()\n"
                "sparse = load_csv('appendix_sparse_tradeoff.csv')\n"
                "diag = load_csv('appendix_sparse_diagnostics.csv')\n"
                "sparse['dataset_label'] = sparse['dataset'].map(dataset_label)\n"
                "diag['dataset_label'] = diag['dataset'].map(dataset_label)\n"
                "beauty_diag = diag[diag['dataset_label'] == 'Beauty'].copy()\n"
                "fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), constrained_layout=True, gridspec_kw={'width_ratios': [1.1, 1.0, 1.0]})\n"
                "ax = axes[0]\n"
                "sns.lineplot(data=sparse.sort_values('active_experts'), x='active_experts', y='test_seen_mrr20', hue='dataset_label', marker='o', linewidth=2.2, palette=['#5B7C99', '#0F766E'], ax=ax)\n"
                "ax.set_xlabel('Active experts')\n"
                "ax.set_ylabel('MRR@20')\n"
                "ax.set_ylim(*metric_limits(sparse['test_seen_mrr20'], padding=0.22, floor=0.085))\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='lower right', title='')\n"
                "panel_label(ax, 'a')\n"
                "ax = axes[1]\n"
                "sns.lineplot(data=beauty_diag, x='setting_label', y='group_n_eff', hue='stage_key', marker='o', linewidth=2.0, palette=['#5B7C99', '#D28B36', '#0F766E'], ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('Effective experts')\n"
                "ax.tick_params(axis='x', rotation=30)\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='upper right', title='')\n"
                "panel_label(ax, 'b')\n"
                "ax = axes[2]\n"
                "entropy = beauty_diag.pivot(index='stage_key', columns='setting_label', values='group_entropy_mean').loc[['macro', 'mid', 'micro']]\n"
                "sns.heatmap(entropy, cmap=sns.light_palette('#C96567', as_cmap=True), annot=True, fmt='.2f', linewidths=0.6, linecolor='white', cbar=False, ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('')\n"
                "panel_label(ax, 'c')\n"
                "annotate_demo(fig, sparse)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        APP_ROOT / "A04_appendix_behavior_and_bins.ipynb",
        [
            md("# A04 Appendix Behavior And Bins\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from appendix_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, metric_limits, panel_label\n"
                "apply_style()\n"
                "bins = load_csv('appendix_special_bins.csv')\n"
                "quality = load_csv('appendix_behavior_slice_quality.csv')\n"
                "profiles = load_csv('appendix_behavior_slice_profiles.csv')\n"
                "bins['dataset_label'] = bins['dataset'].map(dataset_label)\n"
                "quality['dataset_label'] = quality['dataset'].map(dataset_label)\n"
                "heat = profiles.pivot(index='feature_name', columns='dataset', values='feature_value')\n"
                "fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True, gridspec_kw={'width_ratios': [1.2, 1.0, 0.95]})\n"
                "ax = axes[0]\n"
                "sub = bins[bins['group'].isin(['short_session', 'medium_session', 'long_session'])].copy()\n"
                "sns.barplot(data=sub, x='group', y='test_seen_mrr20', hue='dataset_label', palette=['#5B7C99', '#D28B36', '#0F766E'], ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('MRR@20')\n"
                "ax.tick_params(axis='x', rotation=20)\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='lower right', title='')\n"
                "panel_label(ax, 'a')\n"
                "ax = axes[1]\n"
                "sns.scatterplot(data=quality, x='route_concentration', y='test_seen_mrr20', hue='dataset_label', style='group', s=95, palette=['#5B7C99', '#D28B36', '#0F766E'], ax=ax)\n"
                "ax.set_xlabel('Route concentration')\n"
                "ax.set_ylabel('MRR@20')\n"
                "ax.set_ylim(*metric_limits(quality['test_seen_mrr20'], padding=0.24, floor=0.08))\n"
                "clean_axes(ax, grid_axis='both')\n"
                "ax.legend(loc='lower right', title='')\n"
                "panel_label(ax, 'b')\n"
                "ax = axes[2]\n"
                "sns.heatmap(heat, cmap=sns.light_palette('#0F766E', as_cmap=True), annot=True, fmt='.2f', linewidths=0.6, linecolor='white', cbar=False, ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('')\n"
                "panel_label(ax, 'c')\n"
                "annotate_demo(fig, bins)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        APP_ROOT / "A05_appendix_interventions_and_cases.ipynb",
        [
            md("# A05 Appendix Interventions And Cases\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from appendix_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, panel_label\n"
                "apply_style()\n"
                "cases = load_csv('appendix_case_routing_profile.csv')\n"
                "interventions = load_csv('appendix_intervention_summary.csv')\n"
                "interventions['dataset_label'] = interventions['dataset'].map(dataset_label)\n"
                "base = interventions[interventions['intervention_label'] == 'Full'][['dataset', 'test_seen_mrr20']].rename(columns={'test_seen_mrr20': 'base'})\n"
                "interventions = interventions.merge(base, on='dataset')\n"
                "interventions['delta'] = interventions['test_seen_mrr20'] - interventions['base']\n"
                "interventions = interventions[interventions['intervention_label'] != 'Full']\n"
                "heat = cases[cases['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].pivot(index='stage_name', columns='routed_family', values='usage_share').loc[['macro', 'mid', 'micro']]\n"
                "fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.2), constrained_layout=True)\n"
                "ax = axes[0]\n"
                "sns.barplot(data=interventions, x='intervention_label', y='delta', hue='dataset_label', palette=['#5B7C99', '#D28B36', '#0F766E'], ax=ax)\n"
                "ax.axhline(0.0, color='#66757D', linewidth=0.9)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('MRR@20 delta')\n"
                "ax.tick_params(axis='x', rotation=22)\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='lower left', title='')\n"
                "panel_label(ax, 'a')\n"
                "ax = axes[1]\n"
                "sns.heatmap(heat, cmap=sns.light_palette('#0F766E', as_cmap=True), annot=True, fmt='.2f', linewidths=0.6, linecolor='white', cbar=False, ax=ax)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('')\n"
                "panel_label(ax, 'b')\n"
                "annotate_demo(fig, cases)\n"
                "plt.show()\n"
            ),
        ],
    )
    write_notebook(
        APP_ROOT / "A06_appendix_optional_transfer.ipynb",
        [
            md("# A06 Appendix Optional Transfer\n"),
            code(
                "import matplotlib.pyplot as plt\n"
                "from appendix_viz_helpers import apply_style, load_csv, annotate_demo, clean_axes, dataset_label, panel_label\n"
                "apply_style()\n"
                "transfer = load_csv('appendix_transfer_summary.csv').sort_values('data_fraction')\n"
                "transfer['dataset_label'] = transfer['dataset'].map(dataset_label)\n"
                "transfer['fraction_label'] = transfer['data_fraction'].map(lambda x: f'{int(x * 100)}%')\n"
                "fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), constrained_layout=True)\n"
                "ax = axes[0]\n"
                "ax.plot(transfer['data_fraction'], transfer['baseline_mrr20'], marker='o', linewidth=2.0, color='#5B7C99', label='Baseline')\n"
                "ax.plot(transfer['data_fraction'], transfer['route_mrr20'], marker='o', linewidth=2.2, color='#0F766E', label='RouteRec')\n"
                "ax.set_xlabel('Transfer data fraction')\n"
                "ax.set_ylabel('MRR@20')\n"
                "ax.set_xticks(transfer['data_fraction'], transfer['fraction_label'])\n"
                "clean_axes(ax)\n"
                "ax.legend(loc='lower right', title='')\n"
                "panel_label(ax, 'a')\n"
                "ax = axes[1]\n"
                "gain = transfer.copy()\n"
                "gain['gain'] = gain['route_mrr20'] - gain['baseline_mrr20']\n"
                "ax.bar(gain['fraction_label'], gain['gain'], color='#C96567', edgecolor='white', linewidth=0.8)\n"
                "ax.axhline(0.0, color='#66757D', linewidth=0.9)\n"
                "ax.set_xlabel('')\n"
                "ax.set_ylabel('MRR@20 gain')\n"
                "ax.tick_params(axis='x', rotation=20)\n"
                "clean_axes(ax)\n"
                "panel_label(ax, 'b')\n"
                "annotate_demo(fig, transfer)\n"
                "plt.show()\n"
            ),
        ],
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_main_csvs()
    build_appendix_csvs()
    build_main_notebooks()
    build_appendix_notebooks()
    print("Demo assets generated.")


if __name__ == "__main__":
    main()
