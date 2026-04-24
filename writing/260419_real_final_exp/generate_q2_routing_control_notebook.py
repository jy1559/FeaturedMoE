#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/workspace/FeaturedMoE/writing/260419_real_final_exp")
OUT = ROOT / "02_q2_routing_control.ipynb"


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
        md_cell("# 02 Q2 Routing Control\n\nPaper-ready single-subfigure exports for Q2."),
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
            "dataset_label = viz.dataset_label\n"
            "load_csv = viz.load_csv\n"
            "mark_ours_first = viz.mark_ours_first\n"
            "legend_strip_axes = viz.legend_strip_axes\n"
            "single_subfigure_axes = viz.single_subfigure_axes\n"
            "add_legend_strip = viz.add_legend_strip\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'data').exists():\n"
            "    ROOT = Path('/workspace/FeaturedMoE/writing/260419_real_final_exp')\n"
            "PLOT_DATA = ROOT / 'data' / 'paper_plot_values'\n"
            "FIG_DIR = Path('/workspace/FeaturedMoE/writing/ACM_template/figures')\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
            "apply_style()"
        ),
        code_cell(
            "quality_plot = pd.read_csv(PLOT_DATA / 'q2_plot_values.csv')\n"
            "variant_label_map = {\n"
            "    'Behavior-guided': 'Behavior-guided',\n"
            "    'Shared FFN': 'Shared FFN',\n"
            "    'Hidden only': 'Hidden router',\n"
            "    'Fusion bias': 'Fusion bias',\n"
            "    'Mixed': 'Mixed H+B',\n"
            "}\n"
            "quality_plot['variant_display'] = quality_plot['variant_label'].map(variant_label_map).fillna(quality_plot['variant_label'])\n"
            "display_order = ['Behavior-guided', 'Hidden router', 'Mixed H+B', 'Fusion bias', 'Shared FFN']\n"
            "legend_order = mark_ours_first(display_order)\n"
            "quality_plot['variant_display'] = pd.Categorical(quality_plot['variant_display'], categories=display_order, ordered=True)\n"
            "quality_plot = quality_plot.sort_values(['dataset', 'variant_display'], kind='stable').reset_index(drop=True)\n"
            "bar_palette = {\n"
            "    'Behavior-guided': PALETTE['route'],\n"
            "    'Hidden router': PALETTE['orange'],\n"
            "    'Mixed H+B': PALETTE['rose'],\n"
            "    'Fusion bias': PALETTE['plum'],\n"
            "    'Shared FFN': PALETTE['blue'],\n"
            "}\n"
            "legend_colors = [bar_palette[label] for label in display_order]\n"
            "q2_dynamic = quality_plot[quality_plot['dataset'] == 'KuaiRecLargeStrictPosV2_0.2'].copy()\n"
            "q2_stable = quality_plot[quality_plot['dataset'] == 'foursquare'].copy()\n"
            "print('prepared Q2 rows', len(quality_plot))"
        ),
        code_cell(
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    q2_dynamic,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=display_order,\n"
            "    xrotation=0,\n"
            "    palette_override=bar_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=False,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q2_routing_control_a.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q2_routing_control_a.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = single_subfigure_axes()\n"
            "bar_line_panel(\n"
            "    q2_stable,\n"
            "    category_col='variant_display',\n"
            "    ndcg_col='test_ndcg20',\n"
            "    hr_col='test_hit10',\n"
            "    ax=ax,\n"
            "    order=display_order,\n"
            "    xrotation=0,\n"
            "    palette_override=bar_palette,\n"
            "    show_xticklabels=False,\n"
            "    add_metric_legend_box=True,\n"
            ")\n"
            "fig.savefig(FIG_DIR / 'fig_q2_routing_control_b.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q2_routing_control_b.pdf')\n"
            "plt.show()"
        ),
        code_cell(
            "fig, ax = legend_strip_axes()\n"
            "add_legend_strip(ax, legend_order, legend_colors)\n"
            "fig.savefig(FIG_DIR / 'fig_q2_routing_control_legend.pdf', bbox_inches='tight')\n"
            "print('[saved]', FIG_DIR / 'fig_q2_routing_control_legend.pdf')\n"
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
