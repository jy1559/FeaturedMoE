#!/usr/bin/env python3
"""Generate per-phase (P10~P13) wide/verification notebooks focused on setting-level analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

PHASES = ["P10", "P11", "P12", "P13"]

PHASE_HOOK = {
    "P10": "compact feature subset로도 성능을 유지할 수 있는지",
    "P11": "stage order/granularity가 실제 의미를 가지는지",
    "P12": "동일 stage set에서도 layout composition이 차이를 만드는지",
    "P13": "성능 향상이 feature alignment 활용에 의해 발생했는지",
}


def mcell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def ccell(text: str):
    # Some templates keep the first line at col-0 and the rest uniformly indented.
    # Normalize that shape so generated notebook cells are valid Python.
    s = dedent(text).strip("\n")
    lines = s.splitlines()
    if len(lines) > 1:
        first_indent = len(lines[0]) - len(lines[0].lstrip())
        rest_indents = [len(ln) - len(ln.lstrip()) for ln in lines[1:] if ln.strip()]
        if rest_indents:
            min_rest = min(rest_indents)
            if first_indent == 0 and min_rest > 0:
                fixed = [lines[0]]
                for ln in lines[1:]:
                    if not ln.strip():
                        fixed.append(ln)
                    else:
                        fixed.append(ln[min_rest:])
                lines = fixed
    s = "\n".join(lines)
    return nbf.v4.new_code_cell(s + "\n")


def fill_template(text: str, phase: str, hook: str) -> str:
    return dedent(text).replace("__PHASE__", phase).replace("__HOOK__", hook).strip() + "\n"


def wide_cells(phase: str):
    hook = PHASE_HOOK[phase]

    setup = fill_template(
        """
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from IPython.display import display

        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams.update({
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        })

        PHASE = "__PHASE__"
        DATA_DIR = Path("../data/phase10_13")

        wide = pd.read_csv(DATA_DIR / "wide_all_dedup.csv")
        waxis = pd.read_csv(DATA_DIR / "wide_axis_summary.csv")
        intent = pd.read_csv(DATA_DIR / "intent_vs_observed_summary.csv")
        router_family = pd.read_csv(DATA_DIR / "router_family_expert_long.csv")
        router_pos = pd.read_csv(DATA_DIR / "router_position_expert_long.csv")

        to_num = [
            "n_completed", "best_valid_mrr20", "test_mrr20", "cold_item_mrr20", "long_session_mrr20",
            "diag_top1_max_frac", "diag_cv_usage", "diag_n_eff", "diag_available",
            "mean_valid_main", "mean_test_main", "best_valid_minus_anchor", "best_test_minus_anchor",
            "family_expert_share_norm", "position_expert_share_norm",
        ]
        for df in [wide, waxis, intent, router_family, router_pos]:
            for col in to_num:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        phase_all = wide[wide["source_phase"] == PHASE].copy()
        phase_main = phase_all[phase_all["n_completed"] >= 20].copy()
        axis_phase = waxis[waxis["source_phase"] == PHASE].copy()
        intent_phase = intent[intent["source_phase"] == PHASE].copy()
        router_family_phase = router_family[router_family["source_phase"] == PHASE].copy()
        router_pos_phase = router_pos[router_pos["source_phase"] == PHASE].copy()

        def tight_ylim(ax, values, pad_frac=0.15, min_pad=0.0005):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return
            lo = float(arr.min())
            hi = float(arr.max())
            if np.isclose(lo, hi):
                pad = max(min_pad, abs(lo) * 0.05 + 1e-6)
            else:
                pad = max((hi - lo) * pad_frac, min_pad)
            ax.set_ylim(lo - pad, hi + pad)

        def add_bar_labels(ax, fmt="{:.4f}", fontsize=8):
            for p in ax.patches:
                h = p.get_height()
                if not np.isfinite(h):
                    continue
                ax.annotate(
                    fmt.format(h),
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    xytext=(0, 2),
                    textcoords="offset points",
                )

        DIAG_CANDIDATES = [
            "diag_n_eff",
            "diag_cv_usage",
            "diag_top1_max_frac",
            "diag_entropy_mean",
            "diag_route_jitter_adjacent",
            "diag_route_consistency_knn_score",
            "diag_route_consistency_group_knn_score",
            "diag_route_consistency_intra_group_knn_mean_score",
            "diag_family_top_expert_mean_share",
        ]
        DIAG_ALIAS = {
            "diag_n_eff": "n_eff",
            "diag_cv_usage": "cv_usage",
            "diag_top1_max_frac": "top1_max",
            "diag_entropy_mean": "entropy",
            "diag_route_jitter_adjacent": "jitter_adj",
            "diag_route_consistency_knn_score": "knn_cons",
            "diag_route_consistency_group_knn_score": "group_knn_cons",
            "diag_route_consistency_intra_group_knn_mean_score": "intra_knn_cons",
            "diag_family_top_expert_mean_share": "family_top_share",
        }

        def diag_alias(col):
            return DIAG_ALIAS.get(col, col.replace("diag_", ""))

        def available_diag_metrics(df, min_unique=3, min_rows=8):
            metrics = []
            for c in DIAG_CANDIDATES:
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() >= min_rows and s.nunique(dropna=True) >= min_unique:
                    metrics.append(c)
            return metrics

        def diag_facet_scatter(df, target_col, hue_col="setting_group", style_col="hparam_id", title="Diag vs Target", max_metrics=9):
            metrics = available_diag_metrics(df)
            if not metrics:
                print("no available diag metrics for scatter")
                return None
            metrics = metrics[:max_metrics]
            n = len(metrics)
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.9 * nrows), squeeze=False)
            axes_list = axes.flatten()

            for idx, m in enumerate(metrics):
                ax = axes_list[idx]
                cols = [m, target_col, hue_col]
                if style_col in df.columns:
                    cols.append(style_col)
                sub = df[cols].dropna(subset=[m, target_col]).copy()
                if sub.empty or sub[m].nunique() < 2:
                    ax.set_visible(False)
                    continue

                use_style = style_col in sub.columns and 1 < sub[style_col].nunique() <= 8
                plot_args = dict(data=sub, x=m, y=target_col, hue=hue_col, s=30, alpha=0.85, ax=ax)
                if use_style:
                    plot_args["style"] = style_col
                sns.scatterplot(**plot_args)

                ax.set_title(diag_alias(m))
                ax.set_xlabel("diag_value")
                ax.set_ylabel(target_col)
                tight_ylim(ax, sub[target_col], pad_frac=0.12, min_pad=0.0003)
                leg = ax.get_legend()
                if leg is not None:
                    if idx == 0:
                        leg.set_bbox_to_anchor((1.02, 1.0))
                        leg._loc = 2
                    else:
                        leg.remove()

            for j in range(len(metrics), len(axes_list)):
                axes_list[j].set_visible(False)

            fig.suptitle(title, fontsize=13)
            plt.tight_layout()
            return fig

        def diag_group_corr_heatmap(df, group_col, target_col, title):
            metrics = available_diag_metrics(df, min_unique=2, min_rows=4)
            groups = sorted(df[group_col].dropna().unique().tolist()) if group_col in df.columns else []
            if not groups or not metrics:
                print("insufficient data for group-level correlation heatmap")
                return None

            mat = pd.DataFrame(index=groups, columns=[diag_alias(m) for m in metrics], dtype=float)
            for g in groups:
                subg = df[df[group_col] == g]
                for m in metrics:
                    pair = subg[[m, target_col]].dropna()
                    if len(pair) < 4 or pair[m].nunique() < 2 or pair[target_col].nunique() < 2:
                        continue
                    mat.loc[g, diag_alias(m)] = pair[m].corr(pair[target_col], method="spearman")

            if mat.dropna(how="all").empty:
                print("no valid correlations after filtering")
                return None

            plt.figure(figsize=(max(8.0, 0.9 * len(metrics) + 2.0), max(3.5, 0.55 * len(groups) + 1.6)))
            sns.heatmap(
                mat.astype(float),
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0.0,
                vmin=-1.0,
                vmax=1.0,
                annot_kws={"size": 8},
            )
            plt.title(title)
            plt.xlabel("diag_metric")
            plt.ylabel(group_col)
            plt.tight_layout()
            return mat

        def diag_quantile_profile(df, target_col, title, q=5, max_metrics=9):
            metrics = available_diag_metrics(df, min_unique=3, min_rows=max(8, q + 2))
            if not metrics:
                print("no available diag metrics for quantile profile")
                return None
            metrics = metrics[:max_metrics]
            n = len(metrics)
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
            axes_list = axes.flatten()

            for idx, m in enumerate(metrics):
                ax = axes_list[idx]
                sub = df[[m, target_col]].dropna().copy()
                if len(sub) < q + 2 or sub[m].nunique() < 3:
                    ax.set_visible(False)
                    continue
                try:
                    bins = pd.qcut(sub[m], q=q, labels=False, duplicates="drop")
                except ValueError:
                    ax.set_visible(False)
                    continue
                tmp = sub.assign(diag_quantile=bins + 1).groupby("diag_quantile", as_index=False)[target_col].mean()
                if tmp.empty:
                    ax.set_visible(False)
                    continue
                ax.plot(tmp["diag_quantile"], tmp[target_col], marker="o", linewidth=1.5, markersize=4, color="#4C78A8")
                for _, r in tmp.iterrows():
                    ax.text(r["diag_quantile"], r[target_col], f"{r[target_col]:.4f}", fontsize=7, ha="center", va="bottom")
                ax.set_title(diag_alias(m))
                ax.set_xlabel("diag quantile")
                ax.set_ylabel(target_col)
                tight_ylim(ax, tmp[target_col], pad_frac=0.18, min_pad=0.0003)

            for j in range(len(metrics), len(axes_list)):
                axes_list[j].set_visible(False)
            fig.suptitle(title, fontsize=13)
            plt.tight_layout()
            return fig

        def family_expert_pca_scatter(router_family_df, title):
            required = {"run_phase", "setting_group", "family", "expert", "family_expert_share_norm"}
            if not required.issubset(set(router_family_df.columns)):
                print("missing columns for PCA scatter")
                return None

            rf = router_family_df.copy()
            if "stage_name" in rf.columns:
                stage_candidates = rf["stage_name"].dropna()
                if not stage_candidates.empty:
                    stage_mode = stage_candidates.mode().iloc[0]
                    rf = rf[rf["stage_name"] == stage_mode].copy()
                else:
                    stage_mode = None
            else:
                stage_mode = None

            rf["family_expert_share_norm"] = pd.to_numeric(rf["family_expert_share_norm"], errors="coerce")
            rf = rf.dropna(subset=["family_expert_share_norm"])
            if rf.empty:
                print("empty router family data after filtering")
                return None

            idx_cols = ["run_phase", "setting_group", "family"]
            if "hparam_id" in rf.columns:
                idx_cols.append("hparam_id")
            mat = rf.pivot_table(index=idx_cols, columns="expert", values="family_expert_share_norm", aggfunc="mean", fill_value=0.0)
            if mat.shape[0] < 3 or mat.shape[1] < 2:
                print("insufficient shape for PCA scatter", mat.shape)
                return None

            X = mat.to_numpy(dtype=float)
            X = X - X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0
            X = X / std
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            if S.shape[0] < 2:
                print("insufficient singular values for 2D PCA")
                return None
            pc1 = U[:, 0] * S[0]
            pc2 = U[:, 1] * S[1]

            p = mat.reset_index()
            p["pc1"] = pc1
            p["pc2"] = pc2

            plt.figure(figsize=(9.4, 6.5))
            ax = plt.gca()
            sns.scatterplot(data=p, x="pc1", y="pc2", hue="setting_group", style="family", s=45, alpha=0.85, ax=ax)
            ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.axvline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.set_title(title)
            ax.set_xlabel("pc1")
            ax.set_ylabel("pc2")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
            plt.tight_layout()
            if stage_mode is not None:
                print("PCA stage:", stage_mode)
            return p

        DIAG_CANDIDATES = [
            "diag_n_eff",
            "diag_cv_usage",
            "diag_top1_max_frac",
            "diag_entropy_mean",
            "diag_route_jitter_adjacent",
            "diag_route_consistency_knn_score",
            "diag_route_consistency_group_knn_score",
            "diag_route_consistency_intra_group_knn_mean_score",
            "diag_family_top_expert_mean_share",
        ]
        DIAG_ALIAS = {
            "diag_n_eff": "n_eff",
            "diag_cv_usage": "cv_usage",
            "diag_top1_max_frac": "top1_max",
            "diag_entropy_mean": "entropy",
            "diag_route_jitter_adjacent": "jitter_adj",
            "diag_route_consistency_knn_score": "knn_cons",
            "diag_route_consistency_group_knn_score": "group_knn_cons",
            "diag_route_consistency_intra_group_knn_mean_score": "intra_knn_cons",
            "diag_family_top_expert_mean_share": "family_top_share",
        }

        def diag_alias(col):
            return DIAG_ALIAS.get(col, col.replace("diag_", ""))

        def available_diag_metrics(df, min_unique=3, min_rows=8):
            metrics = []
            for c in DIAG_CANDIDATES:
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() >= min_rows and s.nunique(dropna=True) >= min_unique:
                    metrics.append(c)
            return metrics

        def diag_facet_scatter(df, target_col, hue_col="setting_group", style_col="hparam_id", title="Diag vs Target", max_metrics=9):
            metrics = available_diag_metrics(df)
            if not metrics:
                print("no available diag metrics for scatter")
                return None
            metrics = metrics[:max_metrics]
            n = len(metrics)
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.9 * nrows), squeeze=False)
            axes_list = axes.flatten()

            for idx, m in enumerate(metrics):
                ax = axes_list[idx]
                cols = [m, target_col, hue_col]
                if style_col in df.columns:
                    cols.append(style_col)
                sub = df[cols].dropna(subset=[m, target_col]).copy()
                if sub.empty or sub[m].nunique() < 2:
                    ax.set_visible(False)
                    continue

                use_style = style_col in sub.columns and 1 < sub[style_col].nunique() <= 8
                plot_args = dict(data=sub, x=m, y=target_col, hue=hue_col, s=30, alpha=0.85, ax=ax)
                if use_style:
                    plot_args["style"] = style_col
                sns.scatterplot(**plot_args)

                ax.set_title(diag_alias(m))
                ax.set_xlabel("diag_value")
                ax.set_ylabel(target_col)
                tight_ylim(ax, sub[target_col], pad_frac=0.12, min_pad=0.0003)
                leg = ax.get_legend()
                if leg is not None:
                    if idx == 0:
                        leg.set_bbox_to_anchor((1.02, 1.0))
                        leg._loc = 2
                    else:
                        leg.remove()

            for j in range(len(metrics), len(axes_list)):
                axes_list[j].set_visible(False)

            fig.suptitle(title, fontsize=13)
            plt.tight_layout()
            return fig

        def diag_group_corr_heatmap(df, group_col, target_col, title):
            metrics = available_diag_metrics(df, min_unique=2, min_rows=4)
            groups = sorted(df[group_col].dropna().unique().tolist()) if group_col in df.columns else []
            if not groups or not metrics:
                print("insufficient data for group-level correlation heatmap")
                return None

            mat = pd.DataFrame(index=groups, columns=[diag_alias(m) for m in metrics], dtype=float)
            for g in groups:
                subg = df[df[group_col] == g]
                for m in metrics:
                    pair = subg[[m, target_col]].dropna()
                    if len(pair) < 4 or pair[m].nunique() < 2 or pair[target_col].nunique() < 2:
                        continue
                    mat.loc[g, diag_alias(m)] = pair[m].corr(pair[target_col], method="spearman")

            if mat.dropna(how="all").empty:
                print("no valid correlations after filtering")
                return None

            plt.figure(figsize=(max(8.0, 0.9 * len(metrics) + 2.0), max(3.5, 0.55 * len(groups) + 1.6)))
            sns.heatmap(
                mat.astype(float),
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0.0,
                vmin=-1.0,
                vmax=1.0,
                annot_kws={"size": 8},
            )
            plt.title(title)
            plt.xlabel("diag_metric")
            plt.ylabel(group_col)
            plt.tight_layout()
            return mat

        def diag_quantile_profile(df, target_col, title, q=5, max_metrics=9):
            metrics = available_diag_metrics(df, min_unique=3, min_rows=max(8, q + 2))
            if not metrics:
                print("no available diag metrics for quantile profile")
                return None
            metrics = metrics[:max_metrics]
            n = len(metrics)
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
            axes_list = axes.flatten()

            for idx, m in enumerate(metrics):
                ax = axes_list[idx]
                sub = df[[m, target_col]].dropna().copy()
                if len(sub) < q + 2 or sub[m].nunique() < 3:
                    ax.set_visible(False)
                    continue
                try:
                    bins = pd.qcut(sub[m], q=q, labels=False, duplicates="drop")
                except ValueError:
                    ax.set_visible(False)
                    continue
                tmp = sub.assign(diag_quantile=bins + 1).groupby("diag_quantile", as_index=False)[target_col].mean()
                if tmp.empty:
                    ax.set_visible(False)
                    continue
                ax.plot(tmp["diag_quantile"], tmp[target_col], marker="o", linewidth=1.5, markersize=4, color="#4C78A8")
                for _, r in tmp.iterrows():
                    ax.text(r["diag_quantile"], r[target_col], f"{r[target_col]:.4f}", fontsize=7, ha="center", va="bottom")
                ax.set_title(diag_alias(m))
                ax.set_xlabel("diag quantile")
                ax.set_ylabel(target_col)
                tight_ylim(ax, tmp[target_col], pad_frac=0.18, min_pad=0.0003)

            for j in range(len(metrics), len(axes_list)):
                axes_list[j].set_visible(False)
            fig.suptitle(title, fontsize=13)
            plt.tight_layout()
            return fig

        def family_expert_pca_scatter(router_family_df, title):
            required = {"run_phase", "setting_group", "family", "expert", "family_expert_share_norm"}
            if not required.issubset(set(router_family_df.columns)):
                print("missing columns for PCA scatter")
                return None

            rf = router_family_df.copy()
            if "stage_name" in rf.columns:
                stage_candidates = rf["stage_name"].dropna()
                if not stage_candidates.empty:
                    stage_mode = stage_candidates.mode().iloc[0]
                    rf = rf[rf["stage_name"] == stage_mode].copy()
                else:
                    stage_mode = None
            else:
                stage_mode = None

            rf["family_expert_share_norm"] = pd.to_numeric(rf["family_expert_share_norm"], errors="coerce")
            rf = rf.dropna(subset=["family_expert_share_norm"])
            if rf.empty:
                print("empty router family data after filtering")
                return None

            idx_cols = ["run_phase", "setting_group", "family"]
            if "hparam_id" in rf.columns:
                idx_cols.append("hparam_id")
            mat = rf.pivot_table(index=idx_cols, columns="expert", values="family_expert_share_norm", aggfunc="mean", fill_value=0.0)
            if mat.shape[0] < 3 or mat.shape[1] < 2:
                print("insufficient shape for PCA scatter", mat.shape)
                return None

            X = mat.to_numpy(dtype=float)
            X = X - X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0
            X = X / std
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            if S.shape[0] < 2:
                print("insufficient singular values for 2D PCA")
                return None
            pc1 = U[:, 0] * S[0]
            pc2 = U[:, 1] * S[1]

            p = mat.reset_index()
            p["pc1"] = pc1
            p["pc2"] = pc2

            plt.figure(figsize=(9.4, 6.5))
            ax = plt.gca()
            sns.scatterplot(data=p, x="pc1", y="pc2", hue="setting_group", style="family", s=45, alpha=0.85, ax=ax)
            ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.axvline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.set_title(title)
            ax.set_xlabel("pc1")
            ax.set_ylabel("pc2")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
            plt.tight_layout()
            if stage_mode is not None:
                print("PCA stage:", stage_mode)
            return p

        print("phase:", PHASE)
        print("wide rows (all/main):", len(phase_all), "/", len(phase_main))
        print("가설 초점:", "__HOOK__")
        """,
        phase,
        hook,
    )

    summary = """
    print("[요약] phase 스냅샷과 상위 setting을 먼저 확인합니다.")

    if phase_main.empty:
        print("main rows are empty.")
    else:
        summary = {
            "rows_all": len(phase_all),
            "rows_main_n20": len(phase_main),
            "best_valid": phase_main["best_valid_mrr20"].max(),
            "best_test": phase_main["test_mrr20"].max(),
            "mean_valid": phase_main["best_valid_mrr20"].mean(),
            "mean_test": phase_main["test_mrr20"].mean(),
            "mean_cold": phase_main["cold_item_mrr20"].mean(),
            "mean_long": phase_main["long_session_mrr20"].mean(),
        }
        display(pd.DataFrame([summary]).round(6))

        cols = [
            "setting_key", "setting_group", "setting_desc", "setting_detail",
            "best_valid_mrr20", "test_mrr20", "cold_item_mrr20", "long_session_mrr20",
            "diag_top1_max_frac", "diag_cv_usage", "diag_n_eff",
        ]
        top = phase_main.sort_values(["best_valid_mrr20", "test_mrr20"], ascending=False).head(10)
        display(top[cols].round(6))
    """

    axis_optional = """
    print("[참고] axis-level 요약은 보조 지표로만 확인합니다.")

    if axis_phase.empty:
        print("axis summary is empty.")
    else:
        show_cols = [
            "setting_group", "best_setting_key", "best_valid_mrr20", "best_test_mrr20",
            "mean_valid_main", "mean_test_main", "best_valid_minus_anchor", "best_test_minus_anchor"
        ]
        display(axis_phase[show_cols].sort_values("best_valid_mrr20", ascending=False).round(6))
    """

    setting_compare = """
    print("[핵심] 축 내부 세팅 비교: 같은 축 안에서 어떤 세팅이 유리/불리한지 확인합니다.")

    groups = sorted(phase_main["setting_group"].dropna().unique().tolist())
    if not groups:
        print("no setting groups in phase_main")

    for group in groups:
        print("\\n" + "=" * 90)
        print(f"Group: {group}")
        g = phase_main[phase_main["setting_group"] == group].copy()

        if g.empty:
            print("empty group")
            continue

        g["setting_desc"] = g["setting_desc"].fillna("")
        g["setting_detail"] = g["setting_detail"].fillna("")

        agg = (
            g.groupby(["setting_key"], as_index=False)
             .agg(
                 setting_desc=("setting_desc", lambda s: next((x for x in s if str(x).strip()), "")),
                 setting_detail=("setting_detail", lambda s: next((x for x in s if str(x).strip()), "")),
                 best_valid_mrr20=("best_valid_mrr20", "mean"),
                 test_mrr20=("test_mrr20", "mean"),
                 cold_item_mrr20=("cold_item_mrr20", "mean"),
                 long_session_mrr20=("long_session_mrr20", "mean"),
                 diag_top1_max_frac=("diag_top1_max_frac", "mean"),
                 diag_cv_usage=("diag_cv_usage", "mean"),
                 diag_n_eff=("diag_n_eff", "mean"),
             )
        )

        agg = agg.sort_values(["best_valid_mrr20", "test_mrr20"], ascending=False).reset_index(drop=True)
        if agg.empty:
            print("no rows after aggregation; skip this group")
            continue
        best_valid = float(agg["best_valid_mrr20"].max())
        best_test = float(agg["test_mrr20"].max())
        agg["delta_valid_vs_group_best"] = agg["best_valid_mrr20"] - best_valid
        agg["delta_test_vs_group_best"] = agg["test_mrr20"] - best_test

        show_cols = [
            "setting_key", "setting_desc", "setting_detail",
            "best_valid_mrr20", "delta_valid_vs_group_best",
            "test_mrr20", "delta_test_vs_group_best",
            "cold_item_mrr20", "long_session_mrr20",
            "diag_top1_max_frac", "diag_cv_usage", "diag_n_eff",
        ]
        display(agg[show_cols].round(6))

        winner_v = agg.sort_values("best_valid_mrr20", ascending=False).iloc[0]
        winner_t = agg.sort_values("test_mrr20", ascending=False).iloc[0]
        worst_v = agg.sort_values("best_valid_mrr20", ascending=True).iloc[0]

        risk = []
        if worst_v["cold_item_mrr20"] < agg["cold_item_mrr20"].median() - 0.001:
            risk.append("cold 성능 약화")
        if worst_v["long_session_mrr20"] < agg["long_session_mrr20"].median() - 0.001:
            risk.append("long-session 성능 약화")
        if worst_v["diag_top1_max_frac"] > agg["diag_top1_max_frac"].median() + 0.10:
            risk.append("router concentration 증가")
        if worst_v["diag_cv_usage"] > agg["diag_cv_usage"].median() + 0.10:
            risk.append("router usage 변동성 증가")

        print(f"- 추천(valid): {winner_v['setting_key']} ({winner_v['best_valid_mrr20']:.4f})")
        print(f"- 추천(test) : {winner_t['setting_key']} ({winner_t['test_mrr20']:.4f})")
        if risk:
            print(f"- 주의 setting: {worst_v['setting_key']} -> {', '.join(risk)}")

        # Setting-level plots within this group
        plot_df = agg.copy()
        w = max(8.5, 0.55 * len(plot_df) + 4.0)
        fig, axes = plt.subplots(1, 2, figsize=(w * 1.6, 4.8))

        axes[0].bar(plot_df["setting_key"], plot_df["best_valid_mrr20"], color="#4C78A8", alpha=0.9)
        axes[0].set_title(f"{group}: Setting Best Valid")
        axes[0].set_xlabel("Setting")
        axes[0].set_ylabel("Best Valid MRR@20")
        axes[0].tick_params(axis="x", rotation=35)
        add_bar_labels(axes[0])
        tight_ylim(axes[0], plot_df["best_valid_mrr20"], pad_frac=0.25, min_pad=0.0004)

        axes[1].bar(plot_df["setting_key"], plot_df["test_mrr20"], color="#F58518", alpha=0.9)
        axes[1].set_title(f"{group}: Setting Test")
        axes[1].set_xlabel("Setting")
        axes[1].set_ylabel("Test MRR@20")
        axes[1].tick_params(axis="x", rotation=35)
        add_bar_labels(axes[1])
        tight_ylim(axes[1], plot_df["test_mrr20"], pad_frac=0.25, min_pad=0.0004)

        plt.tight_layout()
        plt.show()
    """

    separated_special = """
    print("[중요] test/cold/long metric은 스케일 차이 때문에 분리 플롯으로 확인합니다.")

    metric_specs = [
        ("test_mrr20", "Setting Test MRR@20", "#F58518"),
        ("cold_item_mrr20", "Setting Cold Item MRR@20", "#54A24B"),
        ("long_session_mrr20", "Setting Long Session MRR@20", "#B279A2"),
    ]

    for metric, title, color in metric_specs:
        print(f"\\n- plotting {metric}")
        plot_df = (
            phase_main[["setting_key", "setting_group", metric]]
            .groupby(["setting_key", "setting_group"], as_index=False)
            .mean()
            .sort_values(metric, ascending=False)
        )
        if plot_df.empty:
            print("empty metric table")
            continue

        w = max(10.0, 0.45 * len(plot_df) + 4.0)
        plt.figure(figsize=(w, 4.8))
        ax = plt.gca()
        ax.bar(plot_df["setting_key"], plot_df[metric], color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Setting")
        ax.set_ylabel("MRR@20")
        ax.tick_params(axis="x", rotation=35)
        add_bar_labels(ax)
        tight_ylim(ax, plot_df[metric], pad_frac=0.25, min_pad=0.0004)
        plt.tight_layout()
        plt.show()
    """

    diag_setting = """
    print("[Diag] 축 내부 setting별 router 지표를 확인합니다.")

    diag = phase_main[phase_main["diag_available"] == 1].copy()
    missing = phase_main[phase_main["diag_available"] != 1]["run_phase"].tolist()
    if missing:
        print("diag missing runs:", missing)

    groups = sorted(diag["setting_group"].dropna().unique().tolist())
    for group in groups:
        print("\\n" + "-" * 90)
        print(f"Diag Group: {group}")
        dg = diag[diag["setting_group"] == group].copy()

        if dg.empty:
            continue

        dset = (
            dg.groupby("setting_key", as_index=False)
              .agg(
                  diag_top1_max_frac=("diag_top1_max_frac", "mean"),
                  diag_cv_usage=("diag_cv_usage", "mean"),
                  diag_n_eff=("diag_n_eff", "mean"),
                  best_valid_mrr20=("best_valid_mrr20", "mean"),
                  test_mrr20=("test_mrr20", "mean"),
              )
              .sort_values("best_valid_mrr20", ascending=False)
        )

        display(dset.round(6))

        metrics = [
            ("diag_top1_max_frac", "Top1 Concentration"),
            ("diag_cv_usage", "Usage CV"),
            ("diag_n_eff", "Effective Experts"),
        ]
        for col, title in metrics:
            w = max(8.0, 0.5 * len(dset) + 3.5)
            plt.figure(figsize=(w, 4.2))
            ax = plt.gca()
            ax.bar(dset["setting_key"], dset[col], color="#72B7B2", alpha=0.9)
            ax.set_title(f"{group}: {title}")
            ax.set_xlabel("Setting")
            ax.set_ylabel(col)
            ax.tick_params(axis="x", rotation=35)
            add_bar_labels(ax)
            tight_ylim(ax, dset[col], pad_frac=0.20, min_pad=0.001)
            plt.tight_layout()
            plt.show()
    """

    diag_scatter_advanced = fill_template(
        """
        print("[Diag Deep] scatter: diag metric vs valid/test를 multi-panel로 확인합니다.")
        diag_df = phase_main[phase_main["diag_available"] == 1].copy()
        if diag_df.empty:
            print("diag rows are empty for advanced scatter")
        else:
            _ = diag_facet_scatter(
                diag_df,
                target_col="best_valid_mrr20",
                hue_col="setting_group",
                style_col="seed_id",
                title="Diag vs Valid (__PHASE__ Wide)"
            )
            plt.show()
            _ = diag_facet_scatter(
                diag_df,
                target_col="test_mrr20",
                hue_col="setting_group",
                style_col="seed_id",
                title="Diag vs Test (__PHASE__ Wide)"
            )
            plt.show()
        """,
        phase,
        hook,
    )

    diag_corr_advanced = fill_template(
        """
        print("[Diag Deep] group-level Spearman correlation heatmap을 확인합니다.")
        diag_df = phase_main[phase_main["diag_available"] == 1].copy()
        if diag_df.empty:
            print("diag rows are empty for correlation heatmap")
        else:
            _ = diag_group_corr_heatmap(
                diag_df,
                group_col="setting_group",
                target_col="best_valid_mrr20",
                title="Group-level Spearman Corr (Diag vs Valid, __PHASE__ Wide)"
            )
            plt.show()
            _ = diag_group_corr_heatmap(
                diag_df,
                group_col="setting_group",
                target_col="test_mrr20",
                title="Group-level Spearman Corr (Diag vs Test, __PHASE__ Wide)"
            )
            plt.show()
        """,
        phase,
        hook,
    )

    diag_quantile_advanced = fill_template(
        """
        print("[Diag Deep] diag quantile trend를 확인합니다.")
        diag_df = phase_main[phase_main["diag_available"] == 1].copy()
        if diag_df.empty:
            print("diag rows are empty for quantile trend")
        else:
            _ = diag_quantile_profile(
                diag_df,
                target_col="best_valid_mrr20",
                title="Diag Quantile Trend (Valid, __PHASE__ Wide)"
            )
            plt.show()
            _ = diag_quantile_profile(
                diag_df,
                target_col="test_mrr20",
                title="Diag Quantile Trend (Test, __PHASE__ Wide)"
            )
            plt.show()
        """,
        phase,
        hook,
    )

    family_pca_advanced = fill_template(
        """
        print("[Router Deep] family-expert PCA scatter를 확인합니다.")
        if phase_main.empty or router_family_phase.empty:
            print("router family rows are empty for PCA")
        else:
            top_keys = (
                phase_main[["setting_key", "best_valid_mrr20"]]
                .drop_duplicates()
                .sort_values("best_valid_mrr20", ascending=False)["setting_key"]
                .head(10)
                .tolist()
            )
            pca_df = router_family_phase[router_family_phase["setting_key"].isin(top_keys)].copy()
            if pca_df.empty:
                print("router family rows empty after top-key filter for PCA")
            else:
                _ = family_expert_pca_scatter(pca_df, title="Feature-Family PCA (__PHASE__ Wide)")
                plt.show()
        """,
        phase,
        hook,
    )

    router_family_cell = """
    print("[Router-Family] 상위 setting 기준 family usage를 봅니다.")

    if phase_main.empty or router_family_phase.empty:
        print("router family data is empty")
    else:
        top_settings = (
            phase_main[["setting_key", "best_valid_mrr20"]]
            .drop_duplicates()
            .sort_values("best_valid_mrr20", ascending=False)["setting_key"]
            .head(8)
            .tolist()
        )

        rf = router_family_phase[router_family_phase["setting_key"].isin(top_settings)].copy()
        if rf.empty:
            print("router family rows empty after setting filter")
        else:
            stage_candidates = rf["stage_name"].dropna()
            if stage_candidates.empty:
                print("stage_name is empty after setting filter")
            else:
                stage_mode = stage_candidates.mode().iloc[0]
                rf = rf[rf["stage_name"] == stage_mode].copy()

                fam = (
                    rf.groupby(["setting_key", "family"], as_index=False)["family_expert_share_norm"]
                      .mean()
                )
                heat = fam.pivot(index="setting_key", columns="family", values="family_expert_share_norm").fillna(0.0)
                order = [k for k in top_settings if k in heat.index]
                heat = heat.reindex(order)

                h = max(3.2, 0.45 * len(heat.index) + 1.5)
                plt.figure(figsize=(8.6, h))
                sns.heatmap(
                    heat,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    annot_kws={"size": 8},
                    cbar_kws={"label": "Mean Share"},
                )
                plt.title("Family Usage Heatmap")
                plt.xlabel("Family")
                plt.ylabel("Setting")
                plt.tight_layout()
                plt.show()
                print("selected stage:", stage_mode)
    """

    router_position_cell = """
    print("[Router-Position] 상위 setting 기준 position concentration을 봅니다.")

    if phase_main.empty or router_pos_phase.empty:
        print("router position data is empty")
    else:
        top_settings = (
            phase_main[["setting_key", "best_valid_mrr20"]]
            .drop_duplicates()
            .sort_values("best_valid_mrr20", ascending=False)["setting_key"]
            .head(8)
            .tolist()
        )

        rp = router_pos_phase[router_pos_phase["setting_key"].isin(top_settings)].copy()
        if rp.empty:
            print("router position rows empty after setting filter")
        else:
            stage_candidates = rp["stage_name"].dropna()
            if stage_candidates.empty:
                print("stage_name is empty after setting filter")
            else:
                stage_mode = stage_candidates.mode().iloc[0]
                rp = rp[rp["stage_name"] == stage_mode].copy()

                pos_top = (
                    rp.groupby(["setting_key", "position_index"], as_index=False)["position_expert_share_norm"]
                      .max()
                )
                heat = pos_top.pivot(index="setting_key", columns="position_index", values="position_expert_share_norm").fillna(0.0)

                cols = pd.to_numeric(pd.Index(heat.columns), errors="coerce")
                order_idx = np.argsort(np.where(np.isnan(cols), 1e9, cols))
                heat = heat.iloc[:, order_idx]
                if heat.shape[1] > 20:
                    heat = heat.iloc[:, :20]

                order = [k for k in top_settings if k in heat.index]
                heat = heat.reindex(order)

                w = min(16.0, 0.55 * heat.shape[1] + 4.5)
                h = max(3.2, 0.45 * len(heat.index) + 1.4)
                plt.figure(figsize=(w, h))
                sns.heatmap(
                    heat,
                    annot=False,
                    cmap="magma",
                    cbar_kws={"label": "Max Expert Share"},
                )
                plt.title("Position-wise Concentration Heatmap")
                plt.xlabel("Position Index")
                plt.ylabel("Setting")
                plt.tight_layout()
                plt.show()
                print("selected stage:", stage_mode)
    """

    hypothesis = fill_template(
        """
        print("[가설 대비] 계획 의도와 관찰 결과를 정리합니다.")

        if intent_phase.empty:
            print("intent table is empty")
        else:
            cols = [
                "setting_group", "plan_intent", "expected_pattern", "observed_tag",
                "best_setting_key", "delta_best_valid_vs_anchor", "delta_best_test_vs_anchor", "match_flag"
            ]
            display(intent_phase[cols].sort_values("delta_best_valid_vs_anchor", ascending=False))

        if not phase_main.empty:
            best_valid = phase_main.sort_values("best_valid_mrr20", ascending=False).iloc[0]
            best_test = phase_main.sort_values("test_mrr20", ascending=False).iloc[0]
            print("권장 서술 초안:")
            print(
                "- " + PHASE
                + "에서는 " + "__HOOK__"
                + " 가설을 setting 단위로 점검했다. "
                + "valid 대표는 " + str(best_valid["setting_key"]) + f" ({best_valid['best_valid_mrr20']:.4f}), "
                + "test 대표는 " + str(best_test["setting_key"]) + f" ({best_test['test_mrr20']:.4f})였다."
            )
        """,
        phase,
        hook,
    )

    return [
        mcell(
            f"""
            # {phase} Wide Visualization
            Figure text is English only. Korean explanations are provided via print().
            """
        ),
        ccell(setup),
        ccell(summary),
        ccell(axis_optional),
        ccell(setting_compare),
        ccell(separated_special),
        ccell(diag_setting),
        ccell(diag_scatter_advanced),
        ccell(diag_corr_advanced),
        ccell(diag_quantile_advanced),
        ccell(family_pca_advanced),
        ccell(router_family_cell),
        ccell(router_position_cell),
        ccell(hypothesis),
    ]


def verification_cells(phase: str):
    hook = PHASE_HOOK[phase]

    setup = fill_template(
        """
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from IPython.display import display

        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams.update({
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        })

        PHASE = "__PHASE__"
        DATA_DIR = Path("../data/phase10_13")

        ver_main = pd.read_csv(DATA_DIR / "verification_main_h3_n20.csv")
        ver_support = pd.read_csv(DATA_DIR / "verification_support_p10_n10.csv")
        seed_stats = pd.read_csv(DATA_DIR / "verification_setting_seed_stats.csv")
        intent = pd.read_csv(DATA_DIR / "intent_vs_observed_summary.csv")
        router_family = pd.read_csv(DATA_DIR / "router_family_expert_long.csv")
        router_pos = pd.read_csv(DATA_DIR / "router_position_expert_long.csv")

        to_num = [
            "n_completed", "best_valid_mrr20", "test_mrr20", "cold_item_mrr20", "long_session_mrr20",
            "diag_top1_max_frac", "diag_cv_usage", "diag_n_eff", "diag_available",
            "best_valid_mean", "best_valid_std", "test_mean", "test_std",
            "cold_mean", "cold_std", "long_session_mean", "long_session_std",
            "family_expert_share_norm", "position_expert_share_norm",
            "delta_best_valid_vs_anchor", "delta_best_test_vs_anchor",
        ]
        for df in [ver_main, ver_support, seed_stats, intent, router_family, router_pos]:
            for col in to_num:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        if PHASE == "P10":
            run_df = ver_support[ver_support["source_phase"] == PHASE].copy()
            seed_df = seed_stats[seed_stats["source_phase"] == PHASE].copy()
            scope_label = "support (H1/H3, n=10)"
        else:
            run_df = ver_main[ver_main["source_phase"] == PHASE].copy()
            seed_df = seed_stats[(seed_stats["source_phase"] == PHASE) & (seed_stats["hparam_id"] == "H3")].copy()
            scope_label = "main (H3, n>=20)"

        intent_phase = intent[intent["source_phase"] == PHASE].copy()
        router_family_phase = router_family[router_family["source_phase"] == PHASE].copy()
        router_pos_phase = router_pos[router_pos["source_phase"] == PHASE].copy()

        # attach metadata robustly (avoid setting_group suffix collisions)
        merge_keys = ["setting_key"]
        if "hparam_id" in run_df.columns and "hparam_id" in seed_df.columns:
            merge_keys.append("hparam_id")

        group_meta = run_df[merge_keys + ["setting_group"]].drop_duplicates().copy()
        seed_df = seed_df.merge(group_meta, on=merge_keys, how="left", suffixes=("", "_from_run"))
        if "setting_group_from_run" in seed_df.columns:
            if "setting_group" in seed_df.columns:
                seed_df["setting_group"] = seed_df["setting_group"].fillna(seed_df["setting_group_from_run"])
            else:
                seed_df["setting_group"] = seed_df["setting_group_from_run"]
            seed_df = seed_df.drop(columns=["setting_group_from_run"])

        desc_meta = run_df[merge_keys + ["setting_desc", "setting_detail"]].drop_duplicates().copy()
        seed_df = seed_df.merge(desc_meta, on=merge_keys, how="left", suffixes=("", "_from_run"))
        for col in ["setting_desc", "setting_detail"]:
            from_col = f"{col}_from_run"
            if from_col in seed_df.columns:
                if col in seed_df.columns:
                    seed_df[col] = seed_df[col].fillna(seed_df[from_col])
                else:
                    seed_df[col] = seed_df[from_col]
                seed_df = seed_df.drop(columns=[from_col])

        if PHASE == "P10":
            run_df["setting_plot_key"] = run_df["hparam_id"].astype(str) + "|" + run_df["setting_key"].astype(str)
            seed_df["setting_plot_key"] = seed_df["hparam_id"].astype(str) + "|" + seed_df["setting_key"].astype(str)
        else:
            run_df["setting_plot_key"] = run_df["setting_key"].astype(str)
            seed_df["setting_plot_key"] = seed_df["setting_key"].astype(str)

        def tight_ylim(ax, values, pad_frac=0.15, min_pad=0.0005):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return
            lo = float(arr.min())
            hi = float(arr.max())
            if np.isclose(lo, hi):
                pad = max(min_pad, abs(lo) * 0.05 + 1e-6)
            else:
                pad = max((hi - lo) * pad_frac, min_pad)
            ax.set_ylim(lo - pad, hi + pad)

        def add_bar_labels(ax, fmt="{:.4f}", fontsize=8):
            for p in ax.patches:
                h = p.get_height()
                if not np.isfinite(h):
                    continue
                ax.annotate(
                    fmt.format(h),
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    xytext=(0, 2),
                    textcoords="offset points",
                )

        print("phase:", PHASE)
        print("verification scope:", scope_label)
        print("rows (run/seed):", len(run_df), "/", len(seed_df))
        print("가설 초점:", "__HOOK__")
        """,
        phase,
        hook,
    )

    rationale = """
    print("[핵심] setting별 실험 의도(왜 넣었는지 / 무엇이 바뀌었는지)부터 확인합니다.")

    if seed_df.empty:
        print("seed table is empty")
    else:
        meta_cols = ["setting_plot_key", "setting_group", "setting_desc", "setting_detail"]
        meta = seed_df[meta_cols].drop_duplicates().sort_values(["setting_group", "setting_plot_key"])
        display(meta)

        groups = sorted(meta["setting_group"].dropna().unique().tolist())
        for group in groups:
            m = meta[meta["setting_group"] == group]
            print("\\n-", group)
            for _, r in m.iterrows():
                print(f"  * {r['setting_plot_key']}: {r['setting_detail']}")
    """

    setting_seed_compare = """
    print("[핵심] 축 내부 setting별 mean/std 비교입니다 (axis 평균 비교보다 setting 차이에 집중).")

    groups = sorted(seed_df["setting_group"].dropna().unique().tolist())
    if not groups:
        print("no groups in seed_df")

    for group in groups:
        print("\\n" + "=" * 90)
        print(f"Group: {group}")
        gs = seed_df[seed_df["setting_group"] == group].copy()
        if gs.empty:
            continue

        show_cols = [
            "setting_plot_key", "setting_desc", "setting_detail",
            "best_valid_mean", "best_valid_std", "test_mean", "test_std",
            "cold_mean", "cold_std", "long_session_mean", "long_session_std",
        ]
        gs = gs.sort_values(["best_valid_mean", "test_mean"], ascending=False)
        display(gs[show_cols].round(6))

        best_valid = gs.sort_values("best_valid_mean", ascending=False).iloc[0]
        best_test = gs.sort_values("test_mean", ascending=False).iloc[0]
        stable = gs.sort_values("best_valid_std", ascending=True).iloc[0]

        print(f"- 추천(valid): {best_valid['setting_plot_key']} ({best_valid['best_valid_mean']:.4f} +/- {best_valid['best_valid_std']:.4f})")
        print(f"- 추천(test) : {best_test['setting_plot_key']} ({best_test['test_mean']:.4f} +/- {best_test['test_std']:.4f})")
        print(f"- 안정성 후보 : {stable['setting_plot_key']} (valid std {stable['best_valid_std']:.4f})")

        # Separate valid/test plots for this group
        x = np.arange(len(gs))
        labels = gs["setting_plot_key"].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(max(12.0, 0.75 * len(gs) + 6.0), 5.0))

        axes[0].bar(
            x,
            gs["best_valid_mean"],
            yerr=gs["best_valid_std"].fillna(0.0),
            capsize=3,
            color="#4C78A8",
            alpha=0.9,
        )
        axes[0].set_title(f"{group}: Mean Best Valid +/- Std")
        axes[0].set_xlabel("Setting")
        axes[0].set_ylabel("Best Valid MRR@20")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=35, ha="right")
        add_bar_labels(axes[0])
        tight_ylim(axes[0], gs["best_valid_mean"] + gs["best_valid_std"].fillna(0.0), pad_frac=0.25, min_pad=0.0004)

        axes[1].bar(
            x,
            gs["test_mean"],
            yerr=gs["test_std"].fillna(0.0),
            capsize=3,
            color="#F58518",
            alpha=0.9,
        )
        axes[1].set_title(f"{group}: Mean Test +/- Std")
        axes[1].set_xlabel("Setting")
        axes[1].set_ylabel("Test MRR@20")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=35, ha="right")
        add_bar_labels(axes[1])
        tight_ylim(axes[1], gs["test_mean"] + gs["test_std"].fillna(0.0), pad_frac=0.25, min_pad=0.0004)

        plt.tight_layout()
        plt.show()
    """

    seed_distribution = """
    print("[Seed 분포] setting별 raw seed 분포를 그룹 단위로 확인합니다.")

    groups = sorted(run_df["setting_group"].dropna().unique().tolist())
    for group in groups:
        print("\\n" + "-" * 90)
        print(f"Seed Distribution Group: {group}")

        gr = run_df[run_df["setting_group"] == group].copy()
        if gr.empty:
            continue

        order = (
            seed_df[seed_df["setting_group"] == group]
            .sort_values("best_valid_mean", ascending=False)["setting_plot_key"]
            .tolist()
        )

        if not order:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(max(12.0, 0.70 * len(order) + 6.0), 5.0))

        sns.boxplot(
            data=gr,
            x="setting_plot_key", y="best_valid_mrr20",
            order=order, color="#A0CBE8", showfliers=False, ax=axes[0]
        )
        sns.stripplot(
            data=gr,
            x="setting_plot_key", y="best_valid_mrr20",
            order=order, color="black", alpha=0.65, size=4, jitter=0.12, ax=axes[0]
        )
        axes[0].set_title(f"{group}: Seed Best Valid")
        axes[0].set_xlabel("Setting")
        axes[0].set_ylabel("Best Valid MRR@20")
        axes[0].tick_params(axis="x", rotation=35)
        tight_ylim(axes[0], gr["best_valid_mrr20"], pad_frac=0.25, min_pad=0.0004)

        sns.boxplot(
            data=gr,
            x="setting_plot_key", y="test_mrr20",
            order=order, color="#FFBE7D", showfliers=False, ax=axes[1]
        )
        sns.stripplot(
            data=gr,
            x="setting_plot_key", y="test_mrr20",
            order=order, color="black", alpha=0.65, size=4, jitter=0.12, ax=axes[1]
        )
        axes[1].set_title(f"{group}: Seed Test")
        axes[1].set_xlabel("Setting")
        axes[1].set_ylabel("Test MRR@20")
        axes[1].tick_params(axis="x", rotation=35)
        tight_ylim(axes[1], gr["test_mrr20"], pad_frac=0.25, min_pad=0.0004)

        plt.tight_layout()
        plt.show()
    """

    separated_special = """
    print("[중요] test/cold/long metric은 분리 플롯으로 봅니다.")

    if seed_df.empty:
        print("seed_df is empty")
    else:
        specs = [
            ("test_mean", "test_std", "Setting Mean Test", "#F58518"),
            ("cold_mean", "cold_std", "Setting Mean Cold", "#54A24B"),
            ("long_session_mean", "long_session_std", "Setting Mean Long Session", "#B279A2"),
        ]

        for metric, std_col, title, color in specs:
            plot_df = seed_df[["setting_plot_key", metric, std_col]].copy().sort_values(metric, ascending=False)
            if plot_df.empty:
                continue

            x = np.arange(len(plot_df))
            labels = plot_df["setting_plot_key"].tolist()

            plt.figure(figsize=(max(10.0, 0.55 * len(plot_df) + 4.0), 4.8))
            ax = plt.gca()
            ax.bar(
                x,
                plot_df[metric],
                yerr=plot_df[std_col].fillna(0.0),
                capsize=3,
                color=color,
                alpha=0.9,
            )
            ax.set_title(title)
            ax.set_xlabel("Setting")
            ax.set_ylabel("MRR@20")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            for xi, v in zip(x, plot_df[metric]):
                ax.annotate(f"{v:.4f}", (xi, v), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
            tight_ylim(ax, plot_df[metric] + plot_df[std_col].fillna(0.0), pad_frac=0.25, min_pad=0.0004)
            plt.tight_layout()
            plt.show()
    """

    diag_setting = """
    print("[Diag] setting별 router 지표를 봅니다.")

    diag = run_df[run_df["diag_available"] == 1].copy()
    missing = run_df[run_df["diag_available"] != 1]["run_phase"].tolist()
    if missing:
        print("diag missing runs:", missing)

    if diag.empty:
        print("diag table empty")
    else:
        dsum = (
            diag.groupby(["setting_group", "setting_plot_key"], as_index=False)
                .agg(
                    best_valid_mrr20=("best_valid_mrr20", "mean"),
                    test_mrr20=("test_mrr20", "mean"),
                    diag_top1_max_frac=("diag_top1_max_frac", "mean"),
                    diag_cv_usage=("diag_cv_usage", "mean"),
                    diag_n_eff=("diag_n_eff", "mean"),
                )
        )

        groups = sorted(dsum["setting_group"].dropna().unique().tolist())
        for group in groups:
            print("\\n" + "-" * 90)
            print(f"Diag Group: {group}")
            dg = dsum[dsum["setting_group"] == group].sort_values("best_valid_mrr20", ascending=False)
            display(dg.round(6))

            metrics = [
                ("diag_top1_max_frac", "Top1 Concentration", "#72B7B2"),
                ("diag_cv_usage", "Usage CV", "#E45756"),
                ("diag_n_eff", "Effective Experts", "#4C78A8"),
            ]
            for col, title, color in metrics:
                plt.figure(figsize=(max(8.5, 0.55 * len(dg) + 4.0), 4.2))
                ax = plt.gca()
                ax.bar(dg["setting_plot_key"], dg[col], color=color, alpha=0.9)
                ax.set_title(f"{group}: {title}")
                ax.set_xlabel("Setting")
                ax.set_ylabel(col)
                ax.tick_params(axis="x", rotation=35)
                add_bar_labels(ax)
                tight_ylim(ax, dg[col], pad_frac=0.20, min_pad=0.001)
                plt.tight_layout()
                plt.show()
    """

    diag_scatter_advanced = fill_template(
        """
        print("[Diag Deep] scatter: diag metric vs valid/test를 multi-panel로 확인합니다.")
        diag_df = run_df[run_df["diag_available"] == 1].copy()
        if diag_df.empty:
            print("diag rows are empty for advanced scatter")
        else:
            _ = diag_facet_scatter(
                diag_df,
                target_col="best_valid_mrr20",
                hue_col="setting_group",
                style_col="hparam_id",
                title="Diag vs Valid (__PHASE__ Verification)"
            )
            plt.show()
            _ = diag_facet_scatter(
                diag_df,
                target_col="test_mrr20",
                hue_col="setting_group",
                style_col="hparam_id",
                title="Diag vs Test (__PHASE__ Verification)"
            )
            plt.show()
        """,
        phase,
        hook,
    )

    diag_corr_advanced = fill_template(
        """
        print("[Diag Deep] group-level Spearman correlation heatmap을 확인합니다.")
        diag_df = run_df[run_df["diag_available"] == 1].copy()
        if diag_df.empty:
            print("diag rows are empty for correlation heatmap")
        else:
            _ = diag_group_corr_heatmap(
                diag_df,
                group_col="setting_group",
                target_col="best_valid_mrr20",
                title="Group-level Spearman Corr (Diag vs Valid, __PHASE__ Verification)"
            )
            plt.show()
            _ = diag_group_corr_heatmap(
                diag_df,
                group_col="setting_group",
                target_col="test_mrr20",
                title="Group-level Spearman Corr (Diag vs Test, __PHASE__ Verification)"
            )
            plt.show()
        """,
        phase,
        hook,
    )

    diag_quantile_advanced = fill_template(
        """
        print("[Diag Deep] diag quantile trend를 확인합니다.")
        diag_df = run_df[run_df["diag_available"] == 1].copy()
        if diag_df.empty:
            print("diag rows are empty for quantile trend")
        else:
            _ = diag_quantile_profile(
                diag_df,
                target_col="best_valid_mrr20",
                title="Diag Quantile Trend (Valid, __PHASE__ Verification)"
            )
            plt.show()
            _ = diag_quantile_profile(
                diag_df,
                target_col="test_mrr20",
                title="Diag Quantile Trend (Test, __PHASE__ Verification)"
            )
            plt.show()
        """,
        phase,
        hook,
    )

    family_pca_advanced = fill_template(
        """
        print("[Router Deep] family-expert PCA scatter를 확인합니다.")
        if seed_df.empty or router_family_phase.empty:
            print("router family rows are empty for PCA")
        else:
            top_plot_keys = seed_df.sort_values("best_valid_mean", ascending=False)["setting_plot_key"].head(10).tolist()
            key_map = run_df[["setting_key", "hparam_id", "setting_plot_key"]].drop_duplicates()
            pca_df = router_family_phase.merge(key_map, on=["setting_key", "hparam_id"], how="inner")
            pca_df = pca_df[pca_df["setting_plot_key"].isin(top_plot_keys)].copy()
            if pca_df.empty:
                print("router family rows empty after top-key filter for PCA")
            else:
                _ = family_expert_pca_scatter(pca_df, title="Feature-Family PCA (__PHASE__ Verification)")
                plt.show()
        """,
        phase,
        hook,
    )

    router_family_cell = """
    print("[Router-Family] 상위 setting 중심 family usage를 확인합니다.")

    if seed_df.empty or router_family_phase.empty:
        print("router family empty")
    else:
        top_plot_keys = seed_df.sort_values("best_valid_mean", ascending=False)["setting_plot_key"].head(8).tolist()

        key_map = run_df[["setting_key", "hparam_id", "setting_plot_key"]].drop_duplicates()
        rf = router_family_phase.merge(key_map, on=["setting_key", "hparam_id"], how="inner")
        rf = rf[rf["setting_plot_key"].isin(top_plot_keys)].copy()

        if rf.empty:
            print("router family rows empty after merge/filter")
        else:
            stage_candidates = rf["stage_name"].dropna()
            if stage_candidates.empty:
                print("stage_name is empty after merge/filter")
            else:
                stage_mode = stage_candidates.mode().iloc[0]
                rf = rf[rf["stage_name"] == stage_mode].copy()

                fam = (
                    rf.groupby(["setting_plot_key", "family"], as_index=False)["family_expert_share_norm"]
                      .mean()
                )
                heat = fam.pivot(index="setting_plot_key", columns="family", values="family_expert_share_norm").fillna(0.0)
                order = [k for k in top_plot_keys if k in heat.index]
                heat = heat.reindex(order)

                h = max(3.4, 0.45 * len(heat.index) + 1.5)
                plt.figure(figsize=(8.8, h))
                sns.heatmap(
                    heat,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    annot_kws={"size": 8},
                    cbar_kws={"label": "Mean Share"},
                )
                plt.title("Family Usage Heatmap")
                plt.xlabel("Family")
                plt.ylabel("Setting")
                plt.tight_layout()
                plt.show()
                print("selected stage:", stage_mode)
    """

    router_position_cell = """
    print("[Router-Position] 상위 setting 중심 position concentration을 확인합니다.")

    if seed_df.empty or router_pos_phase.empty:
        print("router position empty")
    else:
        top_plot_keys = seed_df.sort_values("best_valid_mean", ascending=False)["setting_plot_key"].head(8).tolist()

        key_map = run_df[["setting_key", "hparam_id", "setting_plot_key"]].drop_duplicates()
        rp = router_pos_phase.merge(key_map, on=["setting_key", "hparam_id"], how="inner")
        rp = rp[rp["setting_plot_key"].isin(top_plot_keys)].copy()

        if rp.empty:
            print("router position rows empty after merge/filter")
        else:
            stage_candidates = rp["stage_name"].dropna()
            if stage_candidates.empty:
                print("stage_name is empty after merge/filter")
            else:
                stage_mode = stage_candidates.mode().iloc[0]
                rp = rp[rp["stage_name"] == stage_mode].copy()

                pos_top = (
                    rp.groupby(["setting_plot_key", "position_index"], as_index=False)["position_expert_share_norm"]
                      .max()
                )
                heat = pos_top.pivot(index="setting_plot_key", columns="position_index", values="position_expert_share_norm").fillna(0.0)

                cols = pd.to_numeric(pd.Index(heat.columns), errors="coerce")
                order_idx = np.argsort(np.where(np.isnan(cols), 1e9, cols))
                heat = heat.iloc[:, order_idx]
                if heat.shape[1] > 20:
                    heat = heat.iloc[:, :20]

                order = [k for k in top_plot_keys if k in heat.index]
                heat = heat.reindex(order)

                w = min(16.0, 0.55 * heat.shape[1] + 4.5)
                h = max(3.4, 0.45 * len(heat.index) + 1.5)
                plt.figure(figsize=(w, h))
                sns.heatmap(
                    heat,
                    annot=False,
                    cmap="magma",
                    cbar_kws={"label": "Max Expert Share"},
                )
                plt.title("Position-wise Concentration Heatmap")
                plt.xlabel("Position Index")
                plt.ylabel("Setting")
                plt.tight_layout()
                plt.show()
                print("selected stage:", stage_mode)
    """

    hypothesis = fill_template(
        """
        print("[가설 대비] verification 관찰과 주장 문장을 정리합니다.")

        if intent_phase.empty:
            print("intent table is empty")
        else:
            cols = [
                "setting_group", "plan_intent", "expected_pattern", "observed_tag",
                "best_setting_key", "delta_best_valid_vs_anchor", "delta_best_test_vs_anchor", "match_flag"
            ]
            display(intent_phase[cols].sort_values("delta_best_valid_vs_anchor", ascending=False))

        if not seed_df.empty:
            best_valid = seed_df.sort_values("best_valid_mean", ascending=False).iloc[0]
            best_test = seed_df.sort_values("test_mean", ascending=False).iloc[0]
            stable = seed_df.sort_values("best_valid_std", ascending=True).iloc[0]

            print("권장 서술 초안:")
            print(
                "- " + PHASE
                + " verification에서는 " + "__HOOK__"
                + " 가설을 setting별 seed 통계로 점검했다. "
                + "valid 대표는 " + str(best_valid["setting_plot_key"]) + f" ({best_valid['best_valid_mean']:.4f} +/- {best_valid['best_valid_std']:.4f}), "
                + "test 대표는 " + str(best_test["setting_plot_key"]) + f" ({best_test['test_mean']:.4f} +/- {best_test['test_std']:.4f}), "
                + "안정성 후보는 " + str(stable["setting_plot_key"]) + f" (valid std {stable['best_valid_std']:.4f})였다."
            )
        """,
        phase,
        hook,
    )

    return [
        mcell(
            f"""
            # {phase} Verification Visualization
            Figure text is English only. Korean explanations are provided via print().
            """
        ),
        ccell(setup),
        ccell(rationale),
        ccell(setting_seed_compare),
        ccell(seed_distribution),
        ccell(separated_special),
        ccell(diag_setting),
        ccell(diag_scatter_advanced),
        ccell(diag_corr_advanced),
        ccell(diag_quantile_advanced),
        ccell(family_pca_advanced),
        ccell(router_family_cell),
        ccell(router_position_cell),
        ccell(hypothesis),
    ]


def write_notebook(path: Path, cells: list):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def build_all(out_dir: Path):
    for phase in PHASES:
        pnum = phase.lower().replace("p", "phase")
        wide_path = out_dir / f"{pnum}_wide.ipynb"
        ver_path = out_dir / f"{pnum}_verification.ipynb"
        write_notebook(wide_path, wide_cells(phase))
        write_notebook(ver_path, verification_cells(phase))
        print(f"[ok] {wide_path}")
        print(f"[ok] {ver_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-phase phase10~13 notebooks (setting-level focus)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/jy1559/FMoE/experiments/run/fmoe_n3/docs/visualization"),
        help="Output directory for generated notebooks",
    )
    args = parser.parse_args()
    build_all(args.out_dir)
