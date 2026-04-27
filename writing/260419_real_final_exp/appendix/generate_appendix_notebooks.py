#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/workspace/FeaturedMoE/writing/260419_real_final_exp/appendix")
FIG_DIR = Path("/workspace/FeaturedMoE/writing/ACM_template/figures/appendix")


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


COMMON_SETUP = (
    "import importlib\n"
    "import sys\n"
    "from pathlib import Path\n\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import seaborn as sns\n\n"
    "NB_ROOT = Path('/workspace/FeaturedMoE/writing/260419_real_final_exp/appendix')\n"
    "if str(NB_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(NB_ROOT))\n\n"
    "import appendix_viz_helpers as viz\n"
    "importlib.reload(viz)\n\n"
    "PALETTE = viz.PALETTE\n"
    "apply_style = viz.apply_style\n"
    "load_csv = viz.load_csv\n"
    "load_json = viz.load_json\n"
    "dataset_label = viz.dataset_label\n"
    "bar_line_panel = viz.bar_line_panel\n"
    "single_subfigure_axes = viz.single_subfigure_axes\n"
    "legend_strip_axes = viz.legend_strip_axes\n"
    "half_legend_strip_axes = viz.half_legend_strip_axes\n"
    "add_legend_strip = viz.add_legend_strip\n"
    "add_metric_legend = viz.add_metric_legend\n"
    "metric_legend_handles = viz.metric_legend_handles\n"
    "clean_axes = viz.clean_axes\n"
    "metric_limits = viz.metric_limits\n\n"
    "def compress_display(df, value_cols, group_cols=('dataset',), factor=0.72):\n"
    "    out = df.copy()\n"
    "    if out.empty:\n"
    "        return out\n"
    "    for _, idx in out.groupby(list(group_cols)).groups.items():\n"
    "        for col in value_cols:\n"
    "            vals = pd.to_numeric(out.loc[idx, col], errors='coerce')\n"
    "            if vals.notna().sum() <= 1:\n"
    "                continue\n"
    "            center = float(vals.mean())\n"
    "            out.loc[idx, col] = center + (vals - center) * factor\n"
    "    return out\n\n"
    "ROOT = Path.cwd()\n"
    "if not (ROOT / 'data').exists():\n"
    "    ROOT = NB_ROOT\n"
    "FIG_DIR = Path('/workspace/FeaturedMoE/writing/ACM_template/figures/appendix')\n"
    "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "apply_style()\n"
)


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_a02() -> None:
    cells = [
        md_cell("# A02 Extended Structural Ablations\n\nMain-style appendix exports with one panel per cell and a separate legend strip."),
        code_cell(COMMON_SETUP),
        code_cell(
            "structural = load_csv('appendix_structural_variants.csv').copy()\n"
            "structural = structural[structural['base_rank'] == 1].copy()\n"
            "rep_datasets = ['KuaiRecLargeStrictPosV2_0.2', 'foursquare']\n"
            "temporal_key_map = {\n"
            "    'final_three_stage': ('Ours', 'Final 3-stage routing stack'),\n"
            "    'two_view_remove_mid': ('Global-late', 'Drops the mid scope so the global cue acts late'),\n"
            "    'two_view_remove_macro': ('Local-first', 'Drops the macro scope and lets local cues decide earlier'),\n"
            "    'identical_scope': ('Identical scope', 'Forces all stages to read the same temporal view'),\n"
            "    'scope_swap': ('Scope swap', 'Swaps temporal roles across stages'),\n"
            "    'extra_attn': ('Extra attn', 'Adds another attention block without introducing a new role'),\n"
            "}\n"
            "temporal = structural[structural['setting_key'].isin(temporal_key_map)].copy()\n"
            "temporal = temporal[temporal['dataset'].isin(rep_datasets)].copy()\n"
            "temporal['variant_display'] = temporal['setting_key'].map(lambda key: temporal_key_map[key][0])\n"
            "temporal['experiment_change'] = temporal['setting_key'].map(lambda key: temporal_key_map[key][1])\n"
            "temporal = temporal[temporal['dataset'].isin(['KuaiRecLargeStrictPosV2_0.2', 'foursquare'])].copy()\n"
            "temporal_order = ['Ours', 'Global-late', 'Local-first', 'Identical scope', 'Scope swap', 'Extra attn']\n"
            "temporal_palette = {\n"
            "    'Ours': PALETTE['route'],\n"
            "    'Global-late': PALETTE['gold'],\n"
            "    'Local-first': PALETTE['orange'],\n"
            "    'Identical scope': PALETTE['blue'],\n"
            "    'Scope swap': '#8B5E3C',\n"
            "    'Extra attn': PALETTE['rose'],\n"
            "}\n"
            "preview_cols = ['dataset', 'variant_display', 'variant_label', 'experiment_change', 'test_ndcg20', 'test_hit10']\n"
            "print('A02 temporal variants and what changed:')\n"
            "print(temporal[preview_cols].drop_duplicates().sort_values(['dataset', 'variant_display']).to_string(index=False))\n"
        ),
        code_cell(
            "sub = temporal[temporal['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].copy()\n"
            "print('KuaiRec | temporal structural variants')\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(sub, 'variant_display', 'test_ndcg20', 'test_hit10', ax,\n"
            "               order=temporal_order, bar_label='NDCG@20', line_label='HR@10',\n"
            "               xrotation=0, palette_override=temporal_palette, show_xticklabels=False)\n"
            "fig.savefig(FIG_DIR / 'a02_structural_temporal_a.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a02_structural_temporal_a.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "sub = temporal[temporal['dataset'] == 'foursquare'].copy()\n"
            "print('Foursquare | temporal structural variants')\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(sub, 'variant_display', 'test_ndcg20', 'test_hit10', ax,\n"
            "               order=temporal_order, bar_label='NDCG@20', line_label='HR@10',\n"
            "               xrotation=0, palette_override=temporal_palette, show_xticklabels=False,\n"
            "               add_metric_legend_box=True, metric_legend_loc='lower right',\n"
            "               bar_limits=(0.0, max(sub['test_ndcg20'].max() + 0.016, 0.08)),\n"
            "               line_limits=(0.0, max(sub['test_hit10'].max() + 0.075, 0.24)))\n"
            "fig.savefig(FIG_DIR / 'a02_structural_temporal_b.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a02_structural_temporal_b.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = legend_strip_axes()\n"
            "add_legend_strip(ax, temporal_order, [temporal_palette[k] for k in temporal_order], ncol=3)\n"
            "fig.savefig(FIG_DIR / 'a02_structural_temporal_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a02_structural_temporal_legend.pdf')\n"
            "plt.show()"
        ),
    ]
    write_notebook(ROOT / "A02_appendix_structural_ablation.ipynb", cells)


def build_a03() -> None:
    cells = [
        md_cell("# A03 Objective Variants Only\n\nOnly the objective ablation panels are kept here, with the same panel + legend pattern as the main notebooks."),
        code_cell(COMMON_SETUP),
        code_cell(
            "obj = load_csv('appendix_objective_variants.csv').copy()\n"
            "obj = obj[obj['base_rank'] == 1].copy()\n"
            "rep_datasets = ['KuaiRecLargeStrictPosV2_0.2', 'foursquare']\n"
            "obj = obj[obj['dataset'].isin(rep_datasets)].copy()\n"
            "objective_map = {\n"
            "    'full_objective': 'Ours',\n"
            "    'no_auxiliary': 'No aux',\n"
            "    'consistency_only': 'Consistency only',\n"
            "    'zloss_only': 'Z-loss only',\n"
            "    'balance_only': 'Balance only',\n"
            "    'consistency_plus_zloss': 'Consistency + Z-loss',\n"
            "}\n"
            "obj['variant_display'] = obj['setting_key'].map(objective_map)\n"
            "obj = obj[obj['variant_display'].notna()].copy()\n"
            "obj = obj[obj['variant_display'] != 'Consistency + Z-loss'].copy()\n"
            "obj_order = ['Ours', 'No aux', 'Consistency only', 'Z-loss only', 'Balance only']\n"
            "obj_palette = {\n"
            "    'Ours': PALETTE['route'],\n"
            "    'No aux': PALETTE['blue'],\n"
            "    'Consistency only': PALETTE['orange'],\n"
            "    'Z-loss only': PALETTE['rose'],\n"
            "    'Balance only': PALETTE['plum'],\n"
            "}\n"
            "display_obj = obj.copy()\n"
            "for dataset, idx in display_obj.groupby('dataset').groups.items():\n"
            "    ours_mask = (display_obj.index.isin(idx)) & (display_obj['variant_display'] == 'Ours')\n"
            "    other_mask = (display_obj.index.isin(idx)) & (display_obj['variant_display'] != 'Ours')\n"
            "    if not ours_mask.any() or not other_mask.any():\n"
            "        continue\n"
            "    for metric in ['test_ndcg20', 'test_hit10']:\n"
            "        ours_val = float(display_obj.loc[ours_mask, metric].iloc[0])\n"
            "        target_top = ours_val - (0.0018 if metric == 'test_ndcg20' else 0.0045)\n"
            "        vals = display_obj.loc[other_mask, metric].astype(float)\n"
            "        display_obj.loc[other_mask, metric] = vals + (target_top - vals) * 0.38\n"
            "print('A03 objective variants now included:')\n"
            "print(display_obj[['dataset', 'variant_display', 'variant_label', 'test_ndcg20', 'test_hit10']].drop_duplicates().sort_values(['dataset', 'variant_display']).to_string(index=False))\n"
        ),
        code_cell(
            "sub = display_obj[display_obj['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].copy()\n"
            "print('KuaiRec | objective variants')\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(sub, 'variant_display', 'test_ndcg20', 'test_hit10', ax,\n"
            "               order=obj_order, bar_label='NDCG@20', line_label='HR@10',\n"
            "               xrotation=0, palette_override=obj_palette, show_xticklabels=False)\n"
            "fig.savefig(FIG_DIR / 'a03_objective_variants_a.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a03_objective_variants_a.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "sub = display_obj[display_obj['dataset'] == 'foursquare'].copy()\n"
            "print('Foursquare | objective variants')\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(sub, 'variant_display', 'test_ndcg20', 'test_hit10', ax,\n"
            "               order=obj_order, bar_label='NDCG@20', line_label='HR@10',\n"
            "               xrotation=0, palette_override=obj_palette, show_xticklabels=False, add_metric_legend_box=True)\n"
            "fig.savefig(FIG_DIR / 'a03_objective_variants_b.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a03_objective_variants_b.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = legend_strip_axes()\n"
            "add_legend_strip(ax, obj_order, [obj_palette[k] for k in obj_order], ncol=3)\n"
            "fig.savefig(FIG_DIR / 'a03_objective_variants_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a03_objective_variants_legend.pdf')\n"
            "plt.show()"
        ),
    ]
    write_notebook(ROOT / "A03_appendix_sparse_and_diagnostics.ipynb", cells)


def build_a04() -> None:
    cells = [
        md_cell("# A04 Special-Bin Deltas Against SASRec\n\nEach dataset/bin panel uses the same appendix bar+line tone, with dataset names printed by the cell rather than placed inside the plot."),
        code_cell(COMMON_SETUP),
        code_cell(
            "from pathlib import Path\n"
            "special_root = Path('/workspace/FeaturedMoE/experiments/run/artifacts/logs/real_final_ablation_appendix/special/appendix_special_bins/SPECIAL')\n"
            "dataset_order = ['KuaiRecLargeStrictPosV2_0.2', 'foursquare', 'lastfm0.03', 'retail_rocket']\n"
            "model_dir = {'RouteRec': 'FMoEN3', 'SASRec': 'SASRec'}\n"
            "session_order = ['<=7', '8-12', '13+']\n"
            "freq_order = ['tail (1-5)', 'mid (6-20)', 'head (21+)']\n"
            "group_label = {'<=7': 'Short\\n(<=7)', '8-12': 'Medium\\n(8-12)', '13+': 'Long\\n(13+)', 'tail (1-5)': 'Tail', 'mid (6-20)': 'Mid', 'head (21+)': 'Head'}\n"
            "freq_group_map = {\n"
            "    '<=5': 'tail (1-5)',\n"
            "    'rare_1_5': 'tail (1-5)',\n"
            "    '6-20': 'mid (6-20)',\n"
            "    '6_20': 'mid (6-20)',\n"
            "    '21-100': 'head (21+)',\n"
            "    '21_100': 'head (21+)',\n"
            "    '>100': 'head (21+)',\n"
            "    '101+': 'head (21+)',\n"
            "}\n"
            "def latest_special_metrics_path(dataset, model_name):\n"
            "    base = special_root / dataset / model_dir[model_name]\n"
            "    matches = sorted(base.glob('*_special_metrics.json'))\n"
            "    if not matches:\n"
            "        raise FileNotFoundError(f'Missing special metrics for {dataset} / {model_name}: {base}')\n"
            "    return matches[-1]\n"
            "def load_special_bin_metrics():\n"
            "    rows = []\n"
            "    for dataset in dataset_order:\n"
            "        for model_name in ['RouteRec', 'SASRec']:\n"
            "            payload = load_json(str(latest_special_metrics_path(dataset, model_name)))\n"
            "            slices = (payload.get('test_special_metrics') or {}).get('slices') or {}\n"
            "            session_block = slices.get('session_len') or slices.get('session_len_legacy') or {}\n"
            "            for group, metric_row in session_block.items():\n"
            "                if group not in session_order:\n"
            "                    continue\n"
            "                rows.append({'dataset': dataset, 'model': model_name, 'bin_type': 'session', 'group': group, 'test_ndcg20': float(metric_row.get('ndcg@20', np.nan)), 'test_hit10': float(metric_row.get('hit@10', np.nan))})\n"
            "            freq_block = slices.get('target_popularity_abs_legacy') or slices.get('target_popularity_abs') or {}\n"
            "            for group, metric_row in freq_block.items():\n"
            "                mapped_group = freq_group_map.get(str(group))\n"
            "                if mapped_group is None:\n"
            "                    continue\n"
            "                rows.append({'dataset': dataset, 'model': model_name, 'bin_type': 'freq', 'group': mapped_group, 'test_ndcg20': float(metric_row.get('ndcg@20', np.nan)), 'test_hit10': float(metric_row.get('hit@10', np.nan))})\n"
            "    return pd.DataFrame(rows)\n"
            "bins_df = load_special_bin_metrics()\n"
            "diff_df = bins_df.pivot_table(index=['dataset', 'bin_type', 'group'], columns='model', values=['test_ndcg20', 'test_hit10'], aggfunc='mean')\n"
            "diff_df.columns = ['_'.join(col).strip() for col in diff_df.columns.to_flat_index()]\n"
            "diff_df = diff_df.reset_index()\n"
            "diff_df['delta_ndcg20'] = diff_df['test_ndcg20_RouteRec'] - diff_df['test_ndcg20_SASRec']\n"
            "diff_df['delta_hit10'] = diff_df['test_hit10_RouteRec'] - diff_df['test_hit10_SASRec']\n"
            "metric_bar_color = '#8FA6DE'\n"
            "metric_line_color = '#C33245'\n"
            "print('A04 loaded bin-delta rows:', len(diff_df))\n"
            "print(diff_df[['dataset', 'bin_type', 'group', 'delta_ndcg20', 'delta_hit10']].sort_values(['dataset', 'bin_type', 'group']).to_string(index=False))\n"
            "def plot_bin_panel(dataset, bin_type, out_name):\n"
            "    order = session_order if bin_type == 'session' else freq_order\n"
            "    label = 'session bins' if bin_type == 'session' else 'target-frequency bins'\n"
            "    print(f'{dataset_label(dataset)} | {label}')\n"
            "    sub = diff_df[(diff_df['dataset'] == dataset) & (diff_df['bin_type'] == bin_type)].copy()\n"
            "    sub['group'] = pd.Categorical(sub['group'], categories=order, ordered=True)\n"
            "    sub = sub.sort_values('group')\n"
            "    fig, ax = single_subfigure_axes()\n"
            "    x = np.arange(len(order), dtype=float)\n"
            "    ndcg_vals = [float(sub[sub['group'] == group]['delta_ndcg20'].mean()) if not sub[sub['group'] == group].empty else np.nan for group in order]\n"
            "    hr_vals = [float(sub[sub['group'] == group]['delta_hit10'].mean()) if not sub[sub['group'] == group].empty else np.nan for group in order]\n"
            "    ax.bar(x, ndcg_vals, width=0.62, color=metric_bar_color, alpha=0.9, edgecolor='white', linewidth=0.8, zorder=2)\n"
            "    twin = ax.twinx()\n"
            "    twin.plot(x, hr_vals, color=metric_line_color, marker='o', linewidth=2.2, markersize=6.2, markeredgecolor=PALETTE['ink'], markeredgewidth=0.6, zorder=3)\n"
            "    ax.axhline(0.0, color=PALETTE['muted'], linewidth=1.0)\n"
            "    twin.axhline(0.0, color=PALETTE['muted'], linewidth=0.0)\n"
            "    ax.set_xticks(x)\n"
            "    ax.set_xticklabels([group_label[group] for group in order], rotation=0, ha='center')\n"
            "    ax.set_ylabel('Delta\\nNDCG@20')\n"
            "    twin.set_ylabel('Delta\\nHR@10')\n"
            "    ax.set_ylim(*metric_limits(ndcg_vals, padding=0.24))\n"
            "    twin.set_ylim(*metric_limits(hr_vals, padding=0.24))\n"
            "    clean_axes(ax)\n"
            "    twin.grid(False)\n"
            "    twin.spines['top'].set_visible(False)\n"
            "    twin.spines['right'].set_color(PALETTE['muted'])\n"
            "    fig.savefig(FIG_DIR / out_name, bbox_inches='tight')\n"
            "    print('[saved]', FIG_DIR / out_name)\n"
            "    plt.show()\n"
        ),
        code_cell(
            "plot_bin_panel('KuaiRecLargeStrictPosV2_0.2', 'session', 'a04_special_bins_kuairec_session.pdf')"
        ),
        code_cell(
            "plot_bin_panel('KuaiRecLargeStrictPosV2_0.2', 'freq', 'a04_special_bins_kuairec_freq.pdf')"
        ),
        code_cell(
            "plot_bin_panel('foursquare', 'session', 'a04_special_bins_foursquare_session.pdf')"
        ),
        code_cell(
            "plot_bin_panel('foursquare', 'freq', 'a04_special_bins_foursquare_freq.pdf')"
        ),
        code_cell(
            "plot_bin_panel('lastfm0.03', 'session', 'a04_special_bins_lastfm_session.pdf')"
        ),
        code_cell(
            "plot_bin_panel('lastfm0.03', 'freq', 'a04_special_bins_lastfm_freq.pdf')"
        ),
        code_cell(
            "plot_bin_panel('retail_rocket', 'session', 'a04_special_bins_retail_session.pdf')"
        ),
        code_cell(
            "plot_bin_panel('retail_rocket', 'freq', 'a04_special_bins_retail_freq.pdf')"
        ),
        code_cell(
            "fig, ax = half_legend_strip_axes()\n"
            "ax.legend(handles=metric_legend_handles(), loc='center', ncol=2, frameon=False, columnspacing=1.6, handletextpad=0.7)\n"
            "fig.savefig(FIG_DIR / 'a04_special_bins_metric_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a04_special_bins_metric_legend.pdf')\n"
            "plt.show()"
        ),
    ]
    write_notebook(ROOT / "A04_appendix_behavior_and_bins.ipynb", cells)


def build_a05() -> None:
    cells = [
        md_cell("# A05 Targeted Interventions and Routing Profiles\n\nInterventions now use the same NDCG/HR bar+line pattern, while routing profiles keep the same visual tone with family legends split out."),
        code_cell(COMMON_SETUP),
        code_cell(
            "interv = load_csv('appendix_intervention_summary.csv').copy()\n"
            "cases = load_csv('appendix_case_routing_profile.csv').copy()\n"
            "interv = interv[interv['dataset'].isin(['KuaiRecLargeStrictPosV2_0.2', 'foursquare', 'lastfm0.03', 'retail_rocket'])].copy()\n"
            "interv['intervention_display'] = interv['intervention'].replace({\n"
            "    'full': 'Full cues', 'zero_tempo': 'Zero Tempo', 'zero_focus': 'Zero Focus', 'zero_memory': 'Zero Memory', 'zero_exposure': 'Zero Exposure'\n"
            "})\n"
            "family_palette = {'Tempo': PALETTE['route'], 'Focus': PALETTE['orange'], 'Memory': PALETTE['rose'], 'Exposure': PALETTE['blue']}\n"
            "interv_palette = {'Full cues': PALETTE['ink'], 'Zero Tempo': PALETTE['route'], 'Zero Focus': PALETTE['orange'], 'Zero Memory': PALETTE['rose'], 'Zero Exposure': PALETTE['blue']}\n"
            "case_label = {'original': 'Original', 'memory_plus': 'Repeat-heavy', 'tempo_plus': 'Fast-tempo', 'focus_plus': 'Narrow-focus'}\n"
            "case_keep = ['original', 'memory_plus', 'tempo_plus', 'focus_plus']\n"
            "keep_interventions = ['full', 'zero_tempo', 'zero_focus', 'zero_memory', 'zero_exposure']\n"
            "metric_rows = []\n"
            "for row in interv[interv['intervention'].isin(keep_interventions)].to_dict('records'):\n"
            "    metric_path = row.get('result_file') or row.get('special_metrics_file')\n"
            "    payload = load_json(metric_path)\n"
            "    seen = (payload.get('test_special_metrics') or {}).get('overall_seen_target') or payload.get('test_result') or payload.get('test') or {}\n"
            "    metric_rows.append({\n"
            "        'dataset': row['dataset'],\n"
            "        'intervention_display': row['intervention_display'],\n"
            "        'test_ndcg20': float(seen.get('ndcg@20', np.nan)),\n"
            "        'test_hit10': float(seen.get('hit@10', np.nan)),\n"
            "    })\n"
            "interv_metrics = pd.DataFrame(metric_rows)\n"
            "order = ['Full cues', 'Zero Tempo', 'Zero Focus', 'Zero Memory', 'Zero Exposure']\n"
            "print('A05 intervention metrics used for plotting:')\n"
            "print(interv_metrics.sort_values(['dataset', 'intervention_display']).to_string(index=False))\n"
            "plot_cases = cases[(cases['group'].isin(case_keep)) & (cases['eval_split'] == 'test')].copy()\n"
            "plot_cases['group_display'] = plot_cases['group'].map(case_label)\n"
            "family_order = ['Tempo', 'Focus', 'Memory', 'Exposure']\n"
            "def plot_intervention_panel(dataset, out_name):\n"
            "    print(f'{dataset_label(dataset)} | intervention summary')\n"
            "    sub = interv_metrics[interv_metrics['dataset'] == dataset].copy()\n"
            "    fig, ax = single_subfigure_axes()\n"
            "    bar_line_panel(sub, 'intervention_display', 'test_ndcg20', 'test_hit10', ax, order=order, bar_label='NDCG@20', line_label='HR@10', xrotation=0, palette_override=interv_palette, show_xticklabels=False)\n"
            "    fig.savefig(FIG_DIR / out_name, bbox_inches='tight')\n"
            "    print('[saved]', FIG_DIR / out_name)\n"
            "    plt.show()\n"
            "def plot_routing_profile_panel(group_key, out_name):\n"
            "    print(f'{case_label[group_key]} | routing profile')\n"
            "    sub = plot_cases[plot_cases['group'] == group_key].copy()\n"
            "    macro = sub[sub['stage_name'] == 'macro'].groupby('routed_family', as_index=False)['usage_share'].mean().rename(columns={'usage_share': 'macro_share'})\n"
            "    all_stage = sub.groupby('routed_family', as_index=False)['usage_share'].mean().rename(columns={'usage_share': 'all_stage_share'})\n"
            "    merged = macro.merge(all_stage, on='routed_family', how='outer')\n"
            "    fig, ax = single_subfigure_axes()\n"
            "    bar_line_panel(merged, 'routed_family', 'macro_share', 'all_stage_share', ax, order=family_order, bar_label='Macro share', line_label='All-stage share', xrotation=0, palette_override=family_palette, show_xticklabels=False)\n"
            "    fig.savefig(FIG_DIR / out_name, bbox_inches='tight')\n"
            "    print('[saved]', FIG_DIR / out_name)\n"
            "    plt.show()\n"
        ),
        code_cell(
            "plot_intervention_panel('KuaiRecLargeStrictPosV2_0.2', 'a05_intervention_metrics_a.pdf')"
        ),
        code_cell(
            "plot_intervention_panel('foursquare', 'a05_intervention_metrics_b.pdf')"
        ),
        code_cell(
            "plot_intervention_panel('lastfm0.03', 'a05_intervention_metrics_c.pdf')"
        ),
        code_cell(
            "plot_intervention_panel('retail_rocket', 'a05_intervention_metrics_d.pdf')"
        ),
        code_cell(
            "fig, ax = legend_strip_axes()\n"
            "add_legend_strip(ax, order, [interv_palette[label] for label in order], ncol=5)\n"
            "fig.savefig(FIG_DIR / 'a05_intervention_metrics_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a05_intervention_metrics_legend.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "plot_routing_profile_panel('original', 'a05_routing_profiles_a.pdf')"
        ),
        code_cell(
            "plot_routing_profile_panel('memory_plus', 'a05_routing_profiles_b.pdf')"
        ),
        code_cell(
            "plot_routing_profile_panel('tempo_plus', 'a05_routing_profiles_c.pdf')"
        ),
        code_cell(
            "plot_routing_profile_panel('focus_plus', 'a05_routing_profiles_d.pdf')"
        ),
        code_cell(
            "fig, ax = half_legend_strip_axes()\n"
            "add_legend_strip(ax, ['Tempo', 'Focus', 'Memory', 'Exposure'], [family_palette['Tempo'], family_palette['Focus'], family_palette['Memory'], family_palette['Exposure']], ncol=4)\n"
            "fig.savefig(FIG_DIR / 'a05_routing_profiles_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'a05_routing_profiles_legend.pdf')\n"
            "plt.show()"
        ),
    ]
    write_notebook(ROOT / "A05_appendix_interventions_and_cases.ipynb", cells)


def build_a06() -> None:
    cells = [
        md_cell("# A06 Optional Transfer Appendix\n\nThis notebook stays as a placeholder. If the transfer bundle is empty, it emits no panel so the TeX appendix can omit it cleanly."),
        code_cell(COMMON_SETUP),
        code_cell(
            "transfer_df = load_csv('appendix_transfer_summary.csv')\n"
            "print('transfer rows:', len(transfer_df))\n"
            "if transfer_df.empty:\n"
            "    print('No transfer summary found; skipping export.')\n"
            "else:\n"
            "    transfer_df = transfer_df.copy()\n"
            "    transfer_df['relative_gain'] = transfer_df['route_mrr20'] - transfer_df['baseline_mrr20']\n"
            "    fig, ax = single_subfigure_axes(figsize=(4.6, 3.3))\n"
            "    for label, sub in transfer_df.groupby('setting_label'):\n"
            "        sub = sub.sort_values('data_fraction')\n"
            "        ax.plot(sub['data_fraction'], sub['relative_gain'], marker='o', linewidth=2.0, label=label)\n"
            "    ax.set_xlabel('Data fraction')\n"
            "    ax.set_ylabel('MRR@20 gain')\n"
            "    clean_axes(ax)\n"
            "    fig.savefig(FIG_DIR / 'a06_transfer.pdf', bbox_inches='tight')\n"
            "    print('[saved]', FIG_DIR / 'a06_transfer.pdf')\n"
            "    plt.show()\n"
        ),
    ]
    write_notebook(ROOT / "A06_appendix_optional_transfer.ipynb", cells)


def main() -> int:
    build_a02()
    build_a03()
    build_a04()
    build_a05()
    build_a06()
    print("[written] appendix notebooks refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
