#!/usr/bin/env python3
"""
Generate all appendix figures and save to ACM_template/figures/appendix/.
This script replaces running notebooks manually.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import appendix_viz_helpers as viz

apply_style    = viz.apply_style
load_csv       = viz.load_csv
clean_axes     = viz.clean_axes
dataset_label  = viz.dataset_label
metric_limits  = viz.metric_limits
palette_for    = viz.palette_for
bar_line_panel = viz.bar_line_panel
panel_label    = viz.panel_label
PALETTE        = viz.PALETTE
DATASET_LABELS = viz.DATASET_LABELS

apply_style()

OUT_DIR = ROOT.parents[2] / "writing" / "ACM_template" / "figures" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_OUT_DIR = ROOT / "data"
LEGACY_OUT_DIR.mkdir(parents=True, exist_ok=True)

DSET_ORDER = ["beauty", "KuaiRecLargeStrictPosV2_0.2"]


def savefig(name: str, fig: plt.Figure | None = None, extra_paths: list[Path] | None = None) -> None:
    figure = fig or plt.gcf()
    path = OUT_DIR / name
    figure.savefig(path, bbox_inches="tight", dpi=220)
    for extra_path in extra_paths or []:
        figure.savefig(extra_path, bbox_inches="tight", dpi=220)
    plt.close("all")
    print(f"  [saved] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# A01: Full results & dataset statistics
# ─────────────────────────────────────────────────────────────────────────────
def fig_a01_full_results() -> None:
    results = load_csv("appendix_full_results_long.csv")
    results["dataset_label"] = results["dataset"].map(dataset_label)
    METRICS_KEEP = ["hit@10", "hit@20", "ndcg@10", "ndcg@20", "mrr@10", "mrr@20"]
    MODEL_ORDER  = ["RouteRec", "SASRec", "BERT4Rec", "GRU4Rec", "BPR"]
    sub = results[(results["split"] == "test") & (results["metric"].isin(METRICS_KEEP))]
    if sub.empty:
        print("  [SKIP] A01: no test data")
        return
    pivot = (
        sub.pivot_table(index=["dataset_label", "metric"], columns="model", values="value", aggfunc="mean")
        .reindex(columns=[m for m in MODEL_ORDER if m in sub["model"].unique()])
    )
    if pivot.empty or len(pivot.columns) == 0:
        print("  [SKIP] A01: no model columns after filtering")
        return
    fig, ax = plt.subplots(figsize=(8, len(pivot) * 0.45 + 1.2), constrained_layout=True)
    ax.axis("off")
    tbl = ax.table(cellText=pivot.values.round(4), rowLabels=pivot.index.tolist(),
                   colLabels=list(pivot.columns), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title("Full Seen-Target Test Results", fontsize=12, fontweight="semibold", pad=6)
    savefig("a01_full_results_table.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# A02: Structural ablations
# ─────────────────────────────────────────────────────────────────────────────
def fig_a02_structural() -> None:
    structural = load_csv("appendix_structural_variants.csv")
    structural["dataset_label"] = structural["dataset"].map(dataset_label)

    for group_key, group_name, fig_name, legacy_name in [
        ("temporal",  "Temporal / stage-layout variants", "a02_structural_temporal.pdf", "fig_D_temporal_variants.pdf"),
        ("cue_org",   "Cue / routing-organisation variants", "a02_structural_cue_org.pdf", "fig_D_cue_org_variants.pdf"),
    ]:
        sub = structural[structural["variant_group"] == group_key].copy()
        if sub.empty:
            print(f"  [SKIP] A02 {group_key}: no data")
            continue
        order = (sub[["variant_label", "variant_order"]].drop_duplicates()
                 .sort_values("variant_order")["variant_label"].tolist())
        avg = (
            sub.groupby(["variant_label", "variant_order"], as_index=False)[["test_ndcg20", "test_hit10"]]
            .mean()
            .sort_values("variant_order")
        )
        fig, ax = plt.subplots(figsize=(6.9, 3.8), constrained_layout=True)
        bar_line_panel(
            avg,
            category_col="variant_label",
            bar_col="test_ndcg20",
            line_col="test_hit10",
            ax=ax,
            order=order,
            xrotation=20,
        )
        ax.set_title("Average across available datasets", fontsize=11)
        fig.suptitle(group_name, fontsize=11, y=1.03)
        savefig(fig_name, fig, extra_paths=[LEGACY_OUT_DIR / legacy_name])


# ─────────────────────────────────────────────────────────────────────────────
# A03: Sparse routing, objective variants, diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def fig_a03_sparse_and_objective() -> None:
    sparse  = load_csv("appendix_sparse_tradeoff.csv")
    obj_df  = load_csv("appendix_objective_variants.csv")
    diag_df = load_csv("appendix_sparse_diagnostics.csv")
    route_diag = load_csv("appendix_routing_diagnostics.csv")
    for df in (sparse, obj_df, diag_df, route_diag):
        df["dataset_label"] = df["dataset"].map(dataset_label)

    # ── Fig E1: Sparse variant quality ──
    VARIANT_ORDER = ["Dense full mixture", "Flat sparse top-6",
                     "Top-4gr Top-2ex (8 act.)", "Top-3gr Top-2ex — main"]
    VARIANT_SHORT = {
        "Dense full mixture":         "Dense\nfull",
        "Flat sparse top-6":          "Flat\ntop-6",
        "Top-4gr Top-2ex (8 act.)":   "4gr\n2ex",
        "Top-3gr Top-2ex — main":     "3gr2ex\n(main)",
    }
    sparse["variant_short"] = sparse["variant_label"].map(VARIANT_SHORT).fillna(sparse["variant_label"])
    short_order = [VARIANT_SHORT.get(v, v) for v in VARIANT_ORDER if VARIANT_SHORT.get(v, v) in sparse["variant_short"].values]
    datasets_e = [d for d in DSET_ORDER if d in sparse["dataset"].unique()]
    if datasets_e:
        n = len(datasets_e)
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), constrained_layout=True)
        if n == 1: axes = [axes]
        for ax, ds in zip(axes, datasets_e):
            s = sparse[sparse["dataset"] == ds].copy()
            bar_line_panel(s, category_col="variant_short",
                           bar_col="test_seen_mrr20", line_col="test_ndcg20",
                           ax=ax, order=short_order,
                           bar_label="MRR@20 (seen)", line_label="NDCG@20",
                           xrotation=0)
            ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
        savefig("a03_sparse_variants.pdf", fig, extra_paths=[LEGACY_OUT_DIR / "fig_E1_sparse_variants.pdf"])

    # ── Fig F1: Objective variant quality ──
    OBJ_ORDER = ["Full objective", "No auxiliary loss", "KNN consistency only", "Z-loss only", "Consistency + Z-loss"]
    OBJ_SHORT = {
        "Full objective": "Full\nobjective",
        "No auxiliary loss": "No aux\nloss",
        "KNN consistency only": "KNN\nconsist.",
        "Z-loss only": "Z-loss\nonly",
        "Consistency + Z-loss": "Consist.\n+Z-loss",
    }
    obj_df["variant_short"] = obj_df["variant_label"].map(OBJ_SHORT).fillna(obj_df["variant_label"])
    short_obj_order = [OBJ_SHORT.get(v, v) for v in OBJ_ORDER if OBJ_SHORT.get(v, v) in obj_df["variant_short"].values]
    datasets_f = [d for d in DSET_ORDER if d in obj_df["dataset"].unique()]
    if datasets_f:
        n = len(datasets_f)
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), constrained_layout=True)
        if n == 1: axes = [axes]
        for ax, ds in zip(axes, datasets_f):
            s = obj_df[obj_df["dataset"] == ds].copy()
            bar_line_panel(s, category_col="variant_short",
                           bar_col="test_seen_mrr20", line_col="test_ndcg20",
                           ax=ax, order=short_obj_order,
                           bar_label="MRR@20 (seen)", line_label="NDCG@20",
                           xrotation=0)
            ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
        savefig("a03_objective_variants.pdf", fig, extra_paths=[LEGACY_OUT_DIR / "fig_F1_obj_variants.pdf"])

    # ── Fig E2: Route entropy + n_eff by stage ──
    STAGE_ORDER   = ["macro", "mid", "micro"]
    SPARSE_SUBSET = ["Dense full mixture", "Top-3gr Top-2ex — main"]
    SUBSET_PAL    = [PALETTE["blue"], PALETTE["route"]]

    if not diag_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        for vlabel, col in zip(SPARSE_SUBSET, SUBSET_PAL):
            sub = (diag_df[diag_df["variant_label"] == vlabel]
                   .groupby("stage_key", as_index=False)["group_entropy_mean"].mean())
            sub["stage_key"] = pd.Categorical(sub["stage_key"], categories=STAGE_ORDER, ordered=True)
            sub = sub.sort_values("stage_key")
            ax1.plot(sub["stage_key"], sub["group_entropy_mean"], marker="o",
                     label=vlabel, color=col, linewidth=2.1, markersize=5.4)
        ax1.set_xlabel("Stage")
        ax1.set_ylabel("Group routing entropy")
        ax1.legend(fontsize=8.5)
        clean_axes(ax1)

        for vlabel, col in zip(SPARSE_SUBSET, SUBSET_PAL):
            sub = (diag_df[diag_df["variant_label"] == vlabel]
                   .groupby("stage_key", as_index=False)["group_n_eff"].mean())
            sub["stage_key"] = pd.Categorical(sub["stage_key"], categories=STAGE_ORDER, ordered=True)
            sub = sub.sort_values("stage_key")
            ax2.plot(sub["stage_key"], sub["group_n_eff"], marker="s",
                     label=vlabel, color=col, linewidth=2.1, markersize=5.4)
        ax2.set_xlabel("Stage")
        ax2.set_ylabel("Effective expert count")
        ax2.legend(fontsize=8.5)
        clean_axes(ax2)
        savefig("a03_routing_diagnostics_lines.pdf", fig)

    # ── Fig H1: Routing entropy heatmaps ──
    if not route_diag.empty:
        GROUPS = route_diag["variant_label"].unique().tolist()[:4]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        heat_ent = (route_diag.groupby(["variant_label", "stage_key"])["group_entropy_mean"]
                    .mean().unstack("stage_key"))
        for col in STAGE_ORDER:
            if col not in heat_ent.columns:
                heat_ent[col] = np.nan
        heat_ent = heat_ent.reindex(columns=STAGE_ORDER)
        if not heat_ent.empty:
            sns.heatmap(heat_ent, ax=ax1, cmap="YlOrRd", annot=True, fmt=".2f",
                        annot_kws={"size": 9}, linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax1.set_title("Route entropy by group × stage")
            ax1.set_xlabel("Stage")
            ax1.set_ylabel("Routing variant")

        heat_top1 = (route_diag.groupby(["variant_label", "stage_key"])["group_top1_max_frac"]
                     .mean().unstack("stage_key"))
        for col in STAGE_ORDER:
            if col not in heat_top1.columns:
                heat_top1[col] = np.nan
        heat_top1 = heat_top1.reindex(columns=STAGE_ORDER)
        if not heat_top1.empty:
            sns.heatmap(heat_top1, ax=ax2, cmap="Blues", annot=True, fmt=".2f",
                        annot_kws={"size": 9}, linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax2.set_title("Top-1 group mass fraction by group × stage")
            ax2.set_xlabel("Stage")
            ax2.set_ylabel("")
        savefig("a03_routing_heatmaps.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# A04: Special bins & behavioral slices
# ─────────────────────────────────────────────────────────────────────────────
def fig_a04_bins_and_slices() -> None:
    bins_df   = load_csv("appendix_special_bins.csv")
    slices_df = load_csv("appendix_behavior_slice_quality.csv")
    bins_df["dataset_label"]   = bins_df["dataset"].map(dataset_label)
    slices_df["dataset_label"] = slices_df["dataset"].map(dataset_label)

    COMPARE_MODELS = ["RouteRec"]
    MODEL_COLORS = {"RouteRec": PALETTE["route"], "SASRec": PALETTE["blue"]}

    # ── Fig C1: Session-length and frequency bins ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)

    sub_sess = bins_df[bins_df["bin_type"] == "session"]
    SESSION_BIN_ORDER = [g for g in ["short (1-3)", "medium (4-8)", "long (9+)", "<=7", "8-12", "13+", "1-2", "3-5", "6-10"] if g in sub_sess["group"].unique()]
    for mdl in [m for m in ["RouteRec", "SASRec"] if m in sub_sess["model"].unique()]:
        vals = (sub_sess[sub_sess["model"] == mdl]
                .groupby("group", as_index=False)["test_seen_mrr20"].mean()
                .set_index("group").reindex(SESSION_BIN_ORDER)["test_seen_mrr20"])
        ax1.plot(SESSION_BIN_ORDER, vals.values, marker="o", label=mdl,
                 color=MODEL_COLORS.get(mdl, PALETTE["route"]), linewidth=2.1, markersize=5.4)
    ax1.set_xlabel("Session-length bin")
    ax1.set_ylabel("MRR@20 (seen target)")
    vals_all = sub_sess["test_seen_mrr20"].dropna()
    if not vals_all.empty:
        ax1.set_ylim(*metric_limits(vals_all, padding=0.22, floor=0.0))
    ax1.legend(fontsize=9)
    clean_axes(ax1)
    panel_label(ax1, "a")

    FREQ_BIN_ORDER = ["tail (1-5)", "mid (6-20)", "head (21+)"]
    sub_freq = bins_df[bins_df["bin_type"] == "freq"]
    for mdl in [m for m in ["RouteRec", "SASRec"] if m in sub_freq["model"].unique()]:
        vals = (sub_freq[sub_freq["model"] == mdl]
                .groupby("group", as_index=False)["test_seen_mrr20"].mean()
                .set_index("group").reindex(FREQ_BIN_ORDER)["test_seen_mrr20"])
        ax2.plot(FREQ_BIN_ORDER, vals.values, marker="o", label=mdl,
                 color=MODEL_COLORS.get(mdl, PALETTE["route"]), linewidth=2.1, markersize=5.4)
    ax2.set_xlabel("Target-frequency bin")
    ax2.set_ylabel("MRR@20 (seen target)")
    vals_all = sub_freq["test_seen_mrr20"].dropna()
    if not vals_all.empty:
        ax2.set_ylim(*metric_limits(vals_all, padding=0.22, floor=0.0))
    ax2.legend(fontsize=9)
    clean_axes(ax2)
    panel_label(ax2, "b")
    savefig("a04_special_bins.pdf", fig)

    # ── Fig I1: Behavioral slice analysis ──
    SLICE_ORDER  = ["memory_plus", "focus_plus", "tempo_plus", "exploration_plus"]
    SLICE_LABELS = {"memory_plus": "Memory", "focus_plus": "Focus",
                    "tempo_plus": "Tempo", "exploration_plus": "Explor."}
    slices_df["slice_label"] = slices_df["group"].map(SLICE_LABELS).fillna(slices_df["group"])
    slice_short_order = [SLICE_LABELS.get(s, s) for s in SLICE_ORDER]

    datasets_i = [d for d in DSET_ORDER if d in slices_df["dataset"].unique()]
    if not datasets_i:
        print("  [SKIP] A04 slices: no data")
        return

    fig, axes = plt.subplots(1, len(datasets_i), figsize=(4.0 * len(datasets_i), 3.8),
                              constrained_layout=True)
    if len(datasets_i) == 1: axes = [axes]
    for ax, ds in zip(axes, datasets_i):
        sub_rr = slices_df[(slices_df["dataset"] == ds) & (slices_df["model"] == "RouteRec")].copy()
        if sub_rr.empty:
            continue
        bar_line_panel(sub_rr, category_col="slice_label",
                       bar_col="test_seen_mrr20", line_col="route_concentration",
                       ax=ax, order=slice_short_order,
                       bar_label="MRR@20 (seen)", line_label="Route conc.",
                       xrotation=0)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
    savefig("a04_behavior_slices.pdf", fig)

    # ── Fig I1(b): Relative gain ──
    sasrec_s = (slices_df[slices_df["model"] == "SASRec"]
                .groupby("group")["test_seen_mrr20"].mean().reindex(SLICE_ORDER))
    route_s  = (slices_df[slices_df["model"] == "RouteRec"]
                .groupby("group")["test_seen_mrr20"].mean().reindex(SLICE_ORDER))
    if not sasrec_s.isna().all() and not route_s.isna().all():
        gain_s = ((route_s - sasrec_s) / sasrec_s.clip(lower=1e-8)) * 100
        cols_s = [PALETTE["route"] if g >= 0 else PALETTE["rose"] for g in gain_s.values]
        fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
        ax.bar([SLICE_LABELS.get(s, s) for s in SLICE_ORDER], gain_s.values, color=cols_s,
               edgecolor="white", linewidth=0.8)
        ax.axhline(0, color=PALETTE["muted"], linewidth=0.9, linestyle="--")
        ax.set_ylabel("Relative MRR@20 gain over SASRec (%)")
        clean_axes(ax)
        savefig("a04_slice_relative_gain.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# A05: Interventions & routing cases
# ─────────────────────────────────────────────────────────────────────────────
def fig_a05_interventions() -> None:
    interv_df = load_csv("appendix_intervention_summary.csv")
    cases_df  = load_csv("appendix_case_routing_profile.csv")
    interv_df["dataset_label"] = interv_df["dataset"].map(dataset_label)
    cases_df["dataset_label"]  = cases_df["dataset"].map(dataset_label)

    full_scores = (interv_df[interv_df["intervention"] == "full"]
                   .groupby("dataset")["test_seen_mrr20"].mean())
    interv_df["score_drop"] = interv_df.apply(
        lambda r: full_scores.get(r["dataset"], np.nan) - r["test_seen_mrr20"], axis=1)

    # ── Fig J1(a): Score drop ──
    INV_ORDER = ["feature_zero_all", "zero_tempo", "zero_focus", "zero_memory",
                 "zero_exposure", "shuffle_tempo", "shuffle_focus"]
    INV_LABELS = {
        "feature_zero_all": "Zero all",   "zero_tempo": "Zero Tempo",
        "zero_focus": "Zero Focus",       "zero_memory": "Zero Memory",
        "zero_exposure": "Zero Exposure", "shuffle_tempo": "Shuffle Tempo",
        "shuffle_focus": "Shuffle Focus",
    }
    PAL_LIST = [PALETTE["rose"], PALETTE["route"], PALETTE["blue"],
                PALETTE["orange"], PALETTE["plum"], PALETTE["gold"], PALETTE["muted"]]

    sub = interv_df[interv_df["intervention"] != "full"].copy()
    present_inv = [i for i in INV_ORDER if i in sub["intervention"].unique()]
    drops  = [sub[sub["intervention"] == inv]["score_drop"].mean() for inv in present_inv]
    labels = [INV_LABELS.get(i, i) for i in present_inv]
    colors = PAL_LIST[:len(present_inv)]

    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    ax.bar(labels, drops, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.9, linestyle="--")
    ax.set_ylabel("Score drop vs. full cues (MRR@20)")
    ax.tick_params(axis="x", rotation=22)
    clean_axes(ax)
    panel_label(ax, "a")
    savefig("a05_intervention_score_drop.pdf", fig)

    # ── Fig J1(b): Stage routing profiles ──
    CASE_GROUPS = ["memory_plus", "focus_plus", "tempo_plus", "exploration_plus", "exposure_plus"]
    STAGE_ORDER = ["macro", "mid", "micro"]
    ROUTED_FAMS = ["memory", "focus", "tempo", "exposure"]
    CASE_LABELS = {"memory_plus": "Memory", "focus_plus": "Focus",
                   "tempo_plus": "Tempo", "exploration_plus": "Explor.", "exposure_plus": "Explor."}

    case_groups_present = [g for g in CASE_GROUPS if g in cases_df["group"].unique()]
    if not case_groups_present:
        print("  [SKIP] A05 routing profiles: no case groups")
        return

    fig, axes = plt.subplots(1, len(case_groups_present),
                              figsize=(3.6 * len(case_groups_present), 3.8),
                              constrained_layout=True, sharey=True)
    if len(case_groups_present) == 1: axes = [axes]
    for ax, grp in zip(axes, case_groups_present):
        sub = cases_df[cases_df["group"] == grp]
        hm = (sub.groupby(["routed_family", "stage_name"])["usage_share"].mean()
              .unstack("stage_name")
              .reindex(index=ROUTED_FAMS, columns=STAGE_ORDER)
              .fillna(0))
        sns.heatmap(hm, ax=ax, cmap="rocket_r", vmin=0, vmax=0.7, linewidths=0.5,
                    annot=True, fmt=".2f", annot_kws={"size": 8},
                    cbar=(ax is axes[-1]))
        ax.set_title(CASE_LABELS.get(grp, grp), fontsize=10)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Routed family" if ax is axes[0] else "")
    panel_label(axes[0], "b")
    savefig("a05_routing_profiles.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# A06: Transfer learning
# ─────────────────────────────────────────────────────────────────────────────
def fig_a06_transfer() -> None:
    transfer_df = load_csv("appendix_transfer_summary.csv")
    if transfer_df.empty or "setting_label" not in transfer_df.columns:
        print("  [SKIP] A06: transfer data unavailable")
        return
    transfer_df["dataset_label"] = transfer_df["dataset"].map(dataset_label)

    SETTINGS_ORDER = ["Full fine-tune", "Finetuned router", "Frozen router", "No transfer (scratch)"]
    settings_present = [s for s in SETTINGS_ORDER if s in transfer_df["setting_label"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)

    # Low-resource curves
    for sl in settings_present:
        sub = (transfer_df[transfer_df["setting_label"] == sl]
               .groupby("data_fraction", as_index=False)["route_mrr20"].mean()
               .sort_values("data_fraction"))
        ax1.plot(sub["data_fraction"], sub["route_mrr20"], marker="o", label=sl, linewidth=2.1, markersize=5.4)
    ax1.set_xlabel("Training data fraction")
    ax1.set_ylabel("MRR@20 (seen)")
    ax1.legend(fontsize=8)
    clean_axes(ax1)
    panel_label(ax1, "a")

    # Relative gain
    no_transfer = (
        transfer_df[transfer_df["setting_label"] == "No transfer (scratch)"]
        .groupby("data_fraction", as_index=False)["route_mrr20"].mean()
        .sort_values("data_fraction")
        .set_index("data_fraction")["route_mrr20"]
    )
    for sl in settings_present:
        sub = (transfer_df[transfer_df["setting_label"] == sl]
               .groupby("data_fraction", as_index=False)["route_mrr20"].mean()
               .sort_values("data_fraction"))
        gain = (sub.set_index("data_fraction")["route_mrr20"] - no_transfer) / no_transfer.clip(lower=1e-8) * 100
        ax2.plot(gain.index, gain.values, marker="s", label=sl, linewidth=2.1, markersize=5.4)
    ax2.axhline(0, color=PALETTE["muted"], linewidth=0.9, linestyle="--")
    ax2.set_xlabel("Training data fraction")
    ax2.set_ylabel("Relative gain over no-transfer (%)")
    ax2.legend(fontsize=8)
    clean_axes(ax2)
    panel_label(ax2, "b")
    savefig("a06_transfer.pdf", fig)


if __name__ == "__main__":
    print("=== Generating appendix figures ===")
    fig_a01_full_results()
    fig_a02_structural()
    fig_a03_sparse_and_objective()
    fig_a04_bins_and_slices()
    fig_a05_interventions()
    fig_a06_transfer()
    print(f"=== Done. Figures saved to {OUT_DIR} ===")
