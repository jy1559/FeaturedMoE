#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/workspace/FeaturedMoE/writing/260419_real_final_exp")
OUT = ROOT / "03_q3_design_justification.ipynb"


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
        md_cell("# 03 Q3 Design Justification\n\nPaper-ready single-subfigure exports for Q3."),
        code_cell(
            "import importlib\n"
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
            "half_legend_strip_axes = viz.half_legend_strip_axes\n"
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
            "q3_temporal = load_csv('q3_temporal_decomp.csv')\n"
            "q3_org = load_csv('q3_routing_org.csv')\n\n"
            "def select_best_hparam_then_average(df):\n"
            "    scored = (\n"
            "        df.groupby(['dataset', 'variant_label', 'variant_order', 'base_rank'], as_index=False)\n"
            "        [['best_valid_seen_mrr20', 'test_ndcg20', 'test_hit10']]\n"
            "        .mean()\n"
            "    )\n"
            "    best = (\n"
            "        scored.sort_values(\n"
            "            ['dataset', 'variant_order', 'best_valid_seen_mrr20', 'test_ndcg20', 'test_hit10', 'base_rank'],\n"
            "            ascending=[True, True, False, False, False, True],\n"
            "            kind='stable',\n"
            "        )\n"
            "        .drop_duplicates(['dataset', 'variant_label', 'variant_order'], keep='first')\n"
            "        .sort_values(['dataset', 'variant_order'], kind='stable')\n"
            "        .reset_index(drop=True)\n"
            "    )\n"
            "    return best\n\n"
            "temporal_plot = select_best_hparam_then_average(q3_temporal)\n"
            "temporal_plot = temporal_plot[temporal_plot['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].copy()\n"
            "org_plot = select_best_hparam_then_average(q3_org)\n"
            "org_plot = org_plot[org_plot['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].copy()\n\n"
            "temporal_short = {\n"
            "    'Final 3-stage': '3-scope',\n"
            "    'Best 2-view': '2-scope',\n"
            "    'Single-view': '1-scope',\n"
            "}\n"
            "org_short = {\n"
            "    'Hierarchical sparse': 'Hierarchical sparse',\n"
            "    'Hierarchical dense': 'Hierarchical dense',\n"
            "    'Flat dense': 'Flat dense',\n"
            "    'Flat sparse': 'Flat sparse',\n"
            "}\n"
            "temporal_plot['variant_display'] = temporal_plot['variant_label'].map(temporal_short).fillna(temporal_plot['variant_label'])\n"
            "org_plot['variant_display'] = org_plot['variant_label'].map(org_short).fillna(org_plot['variant_label'])\n\n"
            "temporal_order = ['3-scope', '2-scope', '1-scope']\n"
            "org_order = ['Hierarchical sparse', 'Flat dense', 'Flat sparse', 'Hierarchical dense']\n"
            "temporal_legend = mark_ours_first(temporal_order)\n"
            "org_legend = mark_ours_first(org_order)\n"
            "temporal_palette = {\n"
            "    '3-scope': PALETTE['route'],\n"
            "    '2-scope': PALETTE['orange'],\n"
            "    '1-scope': PALETTE['blue'],\n"
            "}\n"
            "org_palette = {\n"
            "    'Hierarchical sparse': PALETTE['route'],\n"
            "    'Flat dense': PALETTE['blue'],\n"
            "    'Flat sparse': PALETTE['orange'],\n"
            "    'Hierarchical dense': PALETTE['plum'],\n"
            "}\n"
            "temporal_colors = [temporal_palette[label] for label in temporal_order]\n"
            "org_colors = [org_palette[label] for label in org_order]\n"
            "print('prepared Q3 rows', len(temporal_plot), len(org_plot))"
        ),
        code_cell(
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    temporal_plot,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=temporal_order,\n"
            "    xrotation=0,\n"
            "    palette_override=temporal_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=False,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q3_design_justification_a.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q3_design_justification_a.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    org_plot,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=org_order,\n"
            "    xrotation=0,\n"
            "    palette_override=org_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=False,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q3_design_justification_b.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q3_design_justification_b.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = half_legend_strip_axes()\n"
            "add_legend_strip(ax, temporal_legend, temporal_colors, ncol=2)\n"
            "fig.savefig(FIG_DIR / 'fig_q3_design_justification_a_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q3_design_justification_a_legend.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = half_legend_strip_axes()\n"
            "add_legend_strip(ax, org_legend, org_colors, ncol=2)\n"
            "fig.savefig(FIG_DIR / 'fig_q3_design_justification_b_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q3_design_justification_b_legend.pdf')\n"
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
