#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/workspace/FeaturedMoE/writing/260419_real_final_exp")
OUT = ROOT / "04_q4_portability.ipynb"


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


nb = {
    "cells": [
        md_cell("# 04 Q4 Cue Portability and Feature Efficacy\n\nPaper-ready single-subfigure exports for Q4."),
        code_cell(
            "import importlib\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "import real_final_viz_helpers as viz\n"
            "importlib.reload(viz)\n\n"
            "PALETTE = viz.PALETTE\n"
            "apply_style = viz.apply_style\n"
            "bar_line_panel = viz.bar_line_panel\n"
            "load_csv = viz.load_csv\n"
            "mark_ours_first = viz.mark_ours_first\n"
            "legend_strip_axes = viz.legend_strip_axes\n"
            "single_subfigure_axes = viz.single_subfigure_axes\n"
            "add_legend_strip = viz.add_legend_strip\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'data').exists():\n"
            "    ROOT = Path('/workspace/FeaturedMoE/writing/260419_real_final_exp')\n"
            "FIG_DIR = Path('/workspace/FeaturedMoE/writing/ACM_template/figures')\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
            "apply_style()"
        ),
        code_cell(
            "portability = load_csv('q4_portability_table.csv').copy()\n"
            "efficacy = load_csv('q4_feature_efficacy.csv').copy()\n\n"
            "def load_json(path):\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n\n"
            "def select_best_hparam_then_average(df, group_cols):\n"
            "    scored = (\n"
            "        df.groupby(group_cols + ['base_rank'], as_index=False)[['best_valid_seen_mrr20', 'test_ndcg20', 'test_hit10']]\n"
            "        .mean()\n"
            "    )\n"
            "    best = (\n"
            "        scored.sort_values(\n"
            "            group_cols + ['best_valid_seen_mrr20', 'test_ndcg20', 'test_hit10', 'base_rank'],\n"
            "            ascending=[True] * len(group_cols) + [False, False, False, True],\n"
            "            kind='stable',\n"
            "        )\n"
            "        .drop_duplicates(group_cols, keep='first')\n"
            "        .sort_values(group_cols, kind='stable')\n"
            "        .reset_index(drop=True)\n"
            "    )\n"
            "    return best\n\n"
            "portability['best_valid_seen_mrr20'] = pd.to_numeric(portability['best_valid_seen_mrr20'], errors='coerce')\n"
            "portability['test_seen_mrr20'] = pd.to_numeric(portability['test_seen_mrr20'], errors='coerce')\n"
            "portability = portability[portability['status'].astype(str).str.lower() == 'ok'].copy()\n"
            "portability['test_ndcg20'] = portability['result_path'].apply(lambda p: float((load_json(p).get('test_result') or {}).get('ndcg@20', 0.0)))\n"
            "portability['test_hit10'] = portability['result_path'].apply(lambda p: float((load_json(p).get('test_result') or {}).get('hit@10', 0.0)))\n"
            "portability_plot = select_best_hparam_then_average(portability, ['dataset', 'setting_key', 'setting_label', 'variant_order'])\n"
            "portability_order = ['Full cues', 'Portable core', 'No time cues', 'No group cues']\n"
            "portability_key_to_display = {\n"
            "    'full': 'Full cues',\n"
            "    'portable_core': 'Portable core',\n"
            "    'remove_time': 'No time cues',\n"
            "    'remove_category': 'No group cues',\n"
            "}\n"
            "portability_plot['variant_display'] = portability_plot['setting_key'].map(portability_key_to_display)\n"
            "portability_legend = mark_ours_first(portability_order)\n"
            "portability_palette = {\n"
            "    'Full cues': PALETTE['route'],\n"
            "    'Portable core': PALETTE['rose'],\n"
            "    'No time cues': PALETTE['blue'],\n"
            "    'No group cues': PALETTE['orange'],\n"
            "}\n"
            "portability_colors = [portability_palette[label] for label in portability_order]\n\n"
            "efficacy['best_valid_seen_mrr20'] = pd.to_numeric(efficacy['best_valid_seen_mrr20'], errors='coerce')\n"
            "efficacy['test_seen_mrr20'] = pd.to_numeric(efficacy['test_seen_mrr20'], errors='coerce')\n"
            "efficacy = efficacy[efficacy['status'].astype(str).str.lower() == 'ok'].copy()\n"
            "efficacy['test_ndcg20'] = efficacy['result_file'].apply(lambda p: float((load_json(p).get('test_result') or {}).get('ndcg@20', 0.0)))\n"
            "efficacy['test_hit10'] = efficacy['result_file'].apply(lambda p: float((load_json(p).get('test_result') or {}).get('hit@10', 0.0)))\n"
            "efficacy_order = ['Intact', 'Cross-sample permute', 'Zero all', 'Intra-seq permute']\n"
            "efficacy_label_map = {\n"
            "    'intact': 'Intact',\n"
            "    'cross_sample_permute': 'Cross-sample permute',\n"
            "    'zero_all': 'Zero all',\n"
            "    'position_permute': 'Intra-seq permute',\n"
            "}\n"
            "efficacy['variant_display'] = efficacy['intervention'].map(efficacy_label_map).fillna(efficacy['intervention_label'])\n"
            "efficacy_legend = mark_ours_first(efficacy_order)\n"
            "efficacy_palette = {\n"
            "    'Intact': PALETTE['route'],\n"
            "    'Cross-sample permute': PALETTE['blue'],\n"
            "    'Zero all': PALETTE['rose'],\n"
            "    'Intra-seq permute': PALETTE['orange'],\n"
            "}\n"
            "efficacy_colors = [efficacy_palette[label] for label in efficacy_order]\n"
            "print('prepared Q4 rows', len(portability_plot), len(efficacy))"
        ),
        code_cell(
            "sub = portability_plot[portability_plot['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].copy()\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    sub,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=portability_order,\n"
            "    xrotation=0,\n"
            "    palette_override=portability_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=False,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q4_portability_a.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q4_portability_a.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "sub = portability_plot[portability_plot['dataset'] == 'foursquare'].copy()\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    sub,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=portability_order,\n"
            "    xrotation=0,\n"
            "    palette_override=portability_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=True,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q4_portability_b.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q4_portability_b.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = legend_strip_axes()\n"
            "add_legend_strip(ax, portability_legend, portability_colors)\n"
            "fig.savefig(FIG_DIR / 'fig_q4_portability_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q4_portability_legend.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "sub = efficacy[efficacy['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].copy()\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    sub,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=efficacy_order,\n"
            "    xrotation=0,\n"
            "    palette_override=efficacy_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=False,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q4_portability_c.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q4_portability_c.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "sub = efficacy[efficacy['dataset'] == 'foursquare'].copy()\n"
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    sub,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=efficacy_order,\n"
            "    xrotation=0,\n"
            "    palette_override=efficacy_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=True,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q4_portability_d.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q4_portability_d.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = legend_strip_axes()\n"
            "add_legend_strip(ax, efficacy_legend, efficacy_colors)\n"
            "fig.savefig(FIG_DIR / 'fig_q4_efficacy_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q4_efficacy_legend.pdf')\n"
            "plt.show()"
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[written] {OUT}")
