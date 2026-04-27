"""Regenerate all appendix demo-dummy CSV files with proper schemas.

Run from appendix/ directory:
    python make_appendix_dummy.py

This overwrites ./data/*.csv with consistent dummy data that has correct
column names matching the appendix notebooks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

DATASETS = ["beauty", "foursquare", "KuaiRecLargeStrictPosV2_0.2", "movielens1m"]
DS_LABEL = {
    "beauty": "Beauty",
    "foursquare": "Foursquare",
    "KuaiRecLargeStrictPosV2_0.2": "KuaiRec",
    "movielens1m": "ML-1M",
}
# Per-dataset MRR@20 base level so results look realistic
DS_BASE = {"beauty": 0.091, "foursquare": 0.213, "KuaiRecLargeStrictPosV2_0.2": 0.101, "movielens1m": 0.182}
STATUS = "demo_dummy"
NOTE = "Preview-only synthetic values — replace with real experiment outputs."


def jitter(base, scale=0.005, n=1):
    return float(base + RNG.normal(0, scale, n)[0])


def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT / name, index=False)
    print(f"  wrote {name} ({len(df)} rows)")


# ─── appendix_dataset_stats.csv ──────────────────────────────────────────────
rows = [
    dict(dataset=d, interactions=n, sessions=s, items=it, avg_session_len=al,
         data_status=STATUS, data_note=NOTE)
    for d, n, s, it, al in [
        ("beauty",                         33_488,  4_243,  3_625, 7.9),
        ("foursquare",                    145_238, 25_369, 30_588, 5.7),
        ("KuaiRecLargeStrictPosV2_0.2",   312_040, 58_210, 11_880, 5.4),
        ("movielens1m",                   575_281, 37_420,  3_706, 15.4),
    ]
]
save(pd.DataFrame(rows), "appendix_dataset_stats.csv")


# ─── appendix_full_results_long.csv ──────────────────────────────────────────
MODELS = ["RouteRec", "SASRec", "BERT4Rec", "GRU4Rec", "BPR"]
METRICS = ["hit@5", "hit@10", "hit@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@5", "mrr@10", "mrr@20"]
METRIC_BASE = {"hit@5": 0.14, "hit@10": 0.22, "hit@20": 0.31, "ndcg@5": 0.092, "ndcg@10": 0.108, "ndcg@20": 0.122,
               "mrr@5": 0.083, "mrr@10": 0.088, "mrr@20": 0.091}
MODEL_DELTA = {"RouteRec": 0.010, "SASRec": 0.000, "BERT4Rec": 0.005, "GRU4Rec": -0.004, "BPR": -0.015}
rows = []
for ds in DATASETS:
    b = DS_BASE[ds]
    for mdl in MODELS:
        for split in ["valid", "test"]:
            for metric in METRICS:
                base_v = METRIC_BASE[metric] * (b / 0.091) + MODEL_DELTA[mdl]
                rows.append(dict(
                    dataset=ds, model=mdl, split=split, metric=metric,
                    value=round(max(0.01, jitter(base_v, 0.003)), 5),
                    data_status=STATUS, data_note=NOTE,
                ))
save(pd.DataFrame(rows), "appendix_full_results_long.csv")


# ─── appendix_structural_variants.csv ────────────────────────────────────────
STRUCTURAL_VARIANTS = [
    # (label, group, order, delta_from_full)
    ("Final 3-stage",              "temporal", 1, 0.000),
    ("Single-view (macro only)",   "temporal", 2, -0.007),
    ("Two-view (macro+mid)",       "temporal", 3, -0.003),
    ("Local-first ordering",       "temporal", 4, -0.005),
    ("Global-late ordering",       "temporal", 5, -0.004),
    ("Duplicated mid stage",       "temporal", 6, -0.006),
    ("Family Prior Intact",        "cue_org",  1, 0.000),
    ("Fewer Semantic Groups",      "cue_org",  2, -0.004),
    ("Groups Shuffled",            "cue_org",  3, -0.006),
    ("Flat Scalar Bag",            "cue_org",  4, -0.008),
    ("Random Group Assignment",    "cue_org",  5, -0.009),
]
rows = []
for ds in DATASETS:
    b = DS_BASE[ds]
    for label, grp, order, delta in STRUCTURAL_VARIANTS:
        ndcg = round(max(0.01, jitter(b * 1.07 + delta, 0.002)), 5)
        rows.append(dict(
            dataset=ds, variant_label=label, variant_group=grp, variant_order=order,
            test_ndcg20=ndcg, test_hit10=round(ndcg * 1.7, 5),
            base_rank=1, seed_id=1, data_status=STATUS, data_note=NOTE,
        ))
save(pd.DataFrame(rows), "appendix_structural_variants.csv")


# ─── appendix_sparse_tradeoff.csv ────────────────────────────────────────────
SPARSE_VARIANTS = [
    ("dense_full",     "Dense full mixture",       "Dense full mixture",      12),
    ("flat_top6",      "Flat sparse top-6",        "Flat sparse top-6",        6),
    ("g4_e2",          "Top-4gr / Top-2ex per gr", "Top-4gr Top-2ex (8 act.)",  8),
    ("g2_e4",          "Top-2gr / Top-4ex per gr", "Top-2gr Top-4ex (8 act.)",  8),
    ("g3_e2",          "Top-3gr / Top-2ex per gr", "Top-3gr Top-2ex — main",    6),
    ("g2_e1",          "Top-2gr / Top-1ex per gr", "Top-2gr Top-1ex (2 act.)",  2),
    ("g3_e3",          "Top-3gr / Top-3ex per gr", "Top-3gr Top-3ex (9 act.)",  9),
]
SPARSE_DELTA = {k: d for k, (_, _, _, _), d in zip(
    [x[0] for x in SPARSE_VARIANTS],
    SPARSE_VARIANTS,
    [0.004, 0.002, 0.001, 0.000, 0.006, -0.002, 0.005],
)}
rows = []
for ds in DATASETS:
    b = DS_BASE[ds]
    for sk, sl, vl, nact in SPARSE_VARIANTS:
        delta = SPARSE_DELTA[sk]
        mrr = round(max(0.01, jitter(b + delta, 0.002)), 5)
        rows.append(dict(
            dataset=ds, setting_key=sk, setting_label=sl, variant_label=vl,
            test_seen_mrr20=mrr, test_ndcg20=round(mrr * 1.2, 5),
            active_experts=nact,
            base_rank=1, seed_id=1, data_status=STATUS, data_note=NOTE,
        ))
save(pd.DataFrame(rows), "appendix_sparse_tradeoff.csv")


# ─── appendix_sparse_diagnostics.csv ─────────────────────────────────────────
rows = []
for ds in DATASETS:
    for sk, sl, vl, nact in SPARSE_VARIANTS:
        for stage in ["macro", "mid", "micro"]:
            ent_base = {"dense_full": 1.39, "flat_top6": 1.1, "g4_e2": 0.95, "g2_e4": 0.68,
                        "g3_e2": 0.82, "g2_e1": 0.55, "g3_e3": 0.98}[sk]
            eff_base = {"dense_full": 4.0, "flat_top6": 3.5, "g4_e2": 3.0, "g2_e4": 2.5,
                        "g3_e2": 2.8, "g2_e1": 1.8, "g3_e3": 3.2}[sk]
            stage_mult = {"macro": 1.0, "mid": 0.93, "micro": 0.87}[stage]
            rows.append(dict(
                dataset=ds, question="sparse", setting_key=sk, setting_label=sl, variant_label=vl,
                stage_key=stage,
                group_entropy_mean=round(jitter(ent_base * stage_mult, 0.03), 4),
                group_n_eff=round(jitter(eff_base * stage_mult, 0.1), 3),
                group_top1_max_frac=round(jitter(0.42 / (eff_base * stage_mult), 0.02), 4),
                data_status=STATUS, data_note=NOTE,
            ))
save(pd.DataFrame(rows), "appendix_sparse_diagnostics.csv")


# ─── appendix_objective_variants.csv ─────────────────────────────────────────
OBJ_VARIANTS = [
    ("full_objective",        "Full objective",           "Full objective",            0.000),
    ("no_auxiliary_loss",     "No auxiliary loss",        "No auxiliary loss",         -0.006),
    ("knn_consistency_only",  "KNN consistency only",     "KNN consistency only",      -0.002),
    ("z_loss_only",           "Z-loss only",              "Z-loss only",               -0.003),
    ("balance_loss_only",     "Balance loss only",        "Balance loss only",         -0.004),
    ("consistency_plus_z",    "Consistency + Z-loss",     "Consistency + Z-loss",      -0.001),
]
rows = []
for ds in DATASETS:
    b = DS_BASE[ds]
    for sk, sl, vl, delta in OBJ_VARIANTS:
        mrr = round(max(0.01, jitter(b + delta, 0.002)), 5)
        rows.append(dict(
            dataset=ds, setting_key=sk, setting_label=sl, variant_label=vl,
            test_seen_mrr20=mrr, test_ndcg20=round(mrr * 1.2, 5),
            base_rank=1, seed_id=1, data_status=STATUS, data_note=NOTE,
        ))
save(pd.DataFrame(rows), "appendix_objective_variants.csv")


# ─── appendix_routing_diagnostics.csv (routing question variant) ─────────────
# Same schema as sparse_diagnostics but for the main routing setting
rows = []
BEHAVIOR_GROUPS = ["memory_plus", "focus_plus", "tempo_plus", "exploration_plus"]
for ds in DATASETS:
    for grp in BEHAVIOR_GROUPS:
        for stage in ["macro", "mid", "micro"]:
            ent_by_grp = {"memory_plus": 0.70, "focus_plus": 0.75, "tempo_plus": 0.80, "exploration_plus": 0.90}[grp]
            stage_mult = {"macro": 1.0, "mid": 0.94, "micro": 0.88}[stage]
            rows.append(dict(
                dataset=ds, question="routing", setting_key="main",
                setting_label="RouteRec main", variant_label=grp,
                stage_key=stage,
                group_entropy_mean=round(jitter(ent_by_grp * stage_mult, 0.03), 4),
                group_n_eff=round(jitter(2.8 * stage_mult, 0.1), 3),
                group_top1_max_frac=round(jitter(0.38, 0.03), 4),
                data_status=STATUS, data_note=NOTE,
            ))
save(pd.DataFrame(rows), "appendix_routing_diagnostics.csv")


# ─── appendix_special_bins.csv  (multi-model: RouteRec, SASRec, BestBaseline) ─
SESSION_BINS = ["short (1-3)", "medium (4-8)", "long (9+)"]
FREQ_BINS    = ["tail (1-5)", "mid (6-20)", "head (21+)"]
COMPARE_MODELS = ["RouteRec", "SASRec", "BestBaseline"]
MODEL_ADJ = {"RouteRec": 0.006, "SASRec": 0.000, "BestBaseline": 0.003}
SESSION_ADJ = {"short (1-3)": -0.008, "medium (4-8)": 0.000, "long (9+)": 0.005}
FREQ_ADJ    = {"tail (1-5)": -0.006, "mid (6-20)": 0.002, "head (21+)": 0.004}

rows = []
for ds in DATASETS:
    b = DS_BASE[ds]
    for mdl in COMPARE_MODELS:
        for grp in SESSION_BINS:
            mrr = round(max(0.01, jitter(b + MODEL_ADJ[mdl] + SESSION_ADJ[grp], 0.002)), 5)
            rows.append(dict(dataset=ds, model=mdl, bin_type="session",
                             group=grp, test_seen_mrr20=mrr, data_status=STATUS, data_note=NOTE))
        for grp in FREQ_BINS:
            mrr = round(max(0.01, jitter(b + MODEL_ADJ[mdl] + FREQ_ADJ[grp], 0.002)), 5)
            rows.append(dict(dataset=ds, model=mdl, bin_type="freq",
                             group=grp, test_seen_mrr20=mrr, data_status=STATUS, data_note=NOTE))
save(pd.DataFrame(rows), "appendix_special_bins.csv")


# ─── appendix_behavior_slice_quality.csv  (multi-model) ──────────────────────
SLICES = ["memory_plus", "focus_plus", "tempo_plus", "exploration_plus"]
SLICE_ADJ = {"memory_plus": 0.005, "focus_plus": 0.002, "tempo_plus": -0.001, "exploration_plus": 0.003}

rows = []
for ds in DATASETS:
    b = DS_BASE[ds]
    for mdl in COMPARE_MODELS:
        for slc in SLICES:
            mrr = round(max(0.01, jitter(b + MODEL_ADJ[mdl] + SLICE_ADJ[slc], 0.002)), 5)
            conc = round(jitter(0.58 + (MODEL_ADJ[mdl] * 3), 0.04), 4) if mdl == "RouteRec" else float("nan")
            rows.append(dict(dataset=ds, model=mdl, group=slc,
                             test_seen_mrr20=mrr, route_concentration=conc,
                             data_status=STATUS, data_note=NOTE))
save(pd.DataFrame(rows), "appendix_behavior_slice_quality.csv")


# ─── appendix_behavior_slice_profiles.csv ────────────────────────────────────
PROFILE_FEATURES = {
    "memory_plus":     {"repeat_ratio": 0.55, "switch_rate": 0.18, "mean_gap_s": 4200, "dom_group_frac": 0.62, "pop_exposure": 0.44},
    "focus_plus":      {"repeat_ratio": 0.28, "switch_rate": 0.12, "mean_gap_s": 3800, "dom_group_frac": 0.75, "pop_exposure": 0.50},
    "tempo_plus":      {"repeat_ratio": 0.21, "switch_rate": 0.48, "mean_gap_s": 820,  "dom_group_frac": 0.40, "pop_exposure": 0.55},
    "exploration_plus":{"repeat_ratio": 0.14, "switch_rate": 0.58, "mean_gap_s": 2100, "dom_group_frac": 0.35, "pop_exposure": 0.61},
}
rows = []
for ds in DATASETS:
    for grp, feats in PROFILE_FEATURES.items():
        for feat, val in feats.items():
            rows.append(dict(dataset=ds, group=grp, feature_name=feat,
                             feature_value=round(jitter(val, val * 0.08), 4),
                             data_status=STATUS, data_note=NOTE))
save(pd.DataFrame(rows), "appendix_behavior_slice_profiles.csv")


# ─── appendix_cost_summary.csv ───────────────────────────────────────────────
COST_MODELS = [
    ("SASRec",              "sasrec",           1_254_016, 1_254_016, 1.00, 1.00),
    ("BERT4Rec",            "bert4rec",         1_548_288, 1_548_288, 1.22, 1.18),
    ("RouteRec-dense",      "featured_moe_n3",  1_782_400, 1_782_400, 1.38, 1.34),
    ("RouteRec-sparse",     "featured_moe_n3",  1_782_400, 1_493_200, 1.10, 1.06),
]
rows = []
for ds in DATASETS[:2]:  # show 2 representative datasets
    scope = "dynamic" if ds in ("beauty", "KuaiRecLargeStrictPosV2_0.2") else "stable"
    for mname, mkey, total_p, active_p, tr_ratio, inf_ratio in COST_MODELS:
        rows.append(dict(
            question="cost", dataset_scope=scope, dataset=ds,
            model_name=mname, model=mkey, status=STATUS,
            total_params=total_p, active_params=active_p,
            train_time_ratio=tr_ratio, infer_time_ratio=inf_ratio,
            data_status=STATUS, data_note=NOTE,
        ))
save(pd.DataFrame(rows), "appendix_cost_summary.csv")


# ─── appendix_intervention_summary.csv ───────────────────────────────────────
INTERVENTIONS = [
    ("full",               "Full (no perturbation)", "all",    0.000),
    ("feature_zero_all",   "Zero all cues",          "all",   -0.013),
    ("zero_tempo",         "Zero Tempo cues",        "tempo", -0.005),
    ("zero_focus",         "Zero Focus cues",        "focus", -0.004),
    ("zero_memory",        "Zero Memory cues",       "memory",-0.003),
    ("zero_exposure",      "Zero Exposure cues",     "exposure",-0.002),
    ("shuffle_tempo",      "Shuffle Tempo cues",     "tempo", -0.004),
    ("shuffle_focus",      "Shuffle Focus cues",     "focus", -0.003),
]
rows = []
for ds in DATASETS:
    b = DS_BASE[ds]
    for inv, label, fam, delta in INTERVENTIONS:
        rows.append(dict(
            dataset=ds, intervention=inv, intervention_label=label, target_family=fam,
            test_mrr20=round(max(0.005, jitter(b + delta, 0.002)), 5),
            test_seen_mrr20=round(max(0.005, jitter(b + delta + 0.002, 0.002)), 5),
            data_status=STATUS, data_note=NOTE,
        ))
save(pd.DataFrame(rows), "appendix_intervention_summary.csv")


# ─── appendix_case_routing_profile.csv ───────────────────────────────────────
CASE_GROUPS = ["memory_plus", "focus_plus", "tempo_plus", "exploration_plus"]
FAMILIES = ["tempo", "focus", "memory", "exposure"]
CASE_ROUTING = {
    # (group) -> per-stage family distribution
    "memory_plus":      {"macro": [0.14, 0.18, 0.55, 0.13], "mid": [0.11, 0.16, 0.59, 0.14], "micro": [0.12, 0.15, 0.61, 0.12]},
    "focus_plus":       {"macro": [0.13, 0.62, 0.13, 0.12], "mid": [0.10, 0.65, 0.12, 0.13], "micro": [0.11, 0.66, 0.12, 0.11]},
    "tempo_plus":       {"macro": [0.58, 0.16, 0.14, 0.12], "mid": [0.60, 0.14, 0.13, 0.13], "micro": [0.61, 0.15, 0.12, 0.12]},
    "exploration_plus": {"macro": [0.28, 0.26, 0.22, 0.24], "mid": [0.27, 0.25, 0.23, 0.25], "micro": [0.30, 0.23, 0.23, 0.24]},
}
rows = []
for ds in DATASETS[:2]:  # 2 representative datasets
    for grp in CASE_GROUPS:
        for stage in ["macro", "mid", "micro"]:
            for fam, share in zip(FAMILIES, CASE_ROUTING[grp][stage]):
                rows.append(dict(
                    dataset=ds, group=grp, stage_name=stage, routed_family=fam,
                    usage_share=round(jitter(share, 0.02), 4),
                    data_status=STATUS, data_note=NOTE,
                ))
save(pd.DataFrame(rows), "appendix_case_routing_profile.csv")


# ─── appendix_diagnostic_case_profile.csv (same schema) ──────────────────────
rows = []
for ds in DATASETS[:2]:
    for grp in CASE_GROUPS:
        for stage in ["macro", "mid", "micro"]:
            for fam, share in zip(FAMILIES, CASE_ROUTING[grp][stage]):
                rows.append(dict(
                    dataset=ds, group=grp, stage_name=stage, routed_family=fam,
                    usage_share=round(jitter(share, 0.015), 4),
                    data_status=STATUS, data_note=NOTE,
                ))
save(pd.DataFrame(rows), "appendix_diagnostic_case_profile.csv")


# ─── appendix_transfer_summary.csv ───────────────────────────────────────────
DATA_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]
TRANSFER_SETTINGS = [
    ("frozen_router",       "Frozen router",       0.000, -0.012),
    ("finetuned_router",    "Finetuned router",    0.000, -0.005),
    ("anchor_transfer",     "Anchor transfer",     0.000, -0.007),
    ("full_finetune",       "Full fine-tune",       0.000, -0.003),
    ("no_transfer",         "No transfer (scratch)",0.000, -0.020),
]
rows = []
for frac in DATA_FRACS:
    scarcity_penalty = (1.0 - frac) * 0.018
    for sk, sl, _, base_pen in TRANSFER_SETTINGS:
        route_mrr = round(max(0.01, jitter(DS_BASE["beauty"] - scarcity_penalty + base_pen + 0.007, 0.003)), 5)
        base_mrr  = round(max(0.01, jitter(DS_BASE["beauty"] - scarcity_penalty + base_pen, 0.003)), 5)
        rows.append(dict(
            dataset="beauty_to_kuairec", setting_key=sk, setting_label=sl,
            data_fraction=frac, route_mrr20=route_mrr, baseline_mrr20=base_mrr,
            status=STATUS, data_status=STATUS, data_note=NOTE,
        ))
save(pd.DataFrame(rows), "appendix_transfer_summary.csv")


# ─── appendix_run_index.csv  (manifest) ──────────────────────────────────────
rows = [
    dict(csv_name=f, description="see data note", data_status=STATUS)
    for f in sorted(p.name for p in OUT.glob("*.csv") if p.name != "appendix_run_index.csv")
]
save(pd.DataFrame(rows), "appendix_run_index.csv")

print("\nAll done.")
