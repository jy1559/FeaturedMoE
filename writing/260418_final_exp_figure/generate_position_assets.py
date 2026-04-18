from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from textwrap import dedent

import pandas as pd


ROOT = Path("/workspace/FeaturedMoE/writing/260418_final_exp_figure")
DATA_DIR = ROOT / "data"

CELL_COUNTER = count(1)


def next_cell_id() -> str:
    return f"cell-{next(CELL_COUNTER):04d}"


def _to_lines(text: str) -> list[str]:
    return [line if line.endswith("\n") else line + "\n" for line in text.strip().splitlines()]


def md_cell(text: str) -> dict:
    cell_id = next_cell_id()
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"id": cell_id, "language": "markdown"},
        "source": _to_lines(text),
    }


def code_cell(code: str) -> dict:
    cell_id = next_cell_id()
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"id": cell_id, "language": "python"},
        "outputs": [],
        "source": _to_lines(dedent(code)),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    target = ROOT / name
    target.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA_DIR / name, index=False)


COMMON_IMPORTS = """
from pathlib import Path
import sys
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

ROOT = Path('/workspace/FeaturedMoE/writing/260418_final_exp_figure')
DATA_DIR = ROOT / 'data'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import slot_viz_helpers as viz
importlib.reload(viz)

from slot_viz_helpers import (
    category_bar_line_plot,
    dual_metric_grouped_plot,
    heatmap_panel,
    line_panel,
    scatter_panel,
    setup_style,
    single_metric_bar,
    style_plain_table,
    style_ranked_table,
)

setup_style()


def pivot_metric_frame(df, id_cols, metric_map):
    wide = df.pivot_table(index=id_cols, columns=['metric', 'cutoff'], values='value', aggfunc='first').reset_index()
    flattened = []
    for col in wide.columns:
        if isinstance(col, tuple):
            left, right = col
            if right == '':
                flattened.append(left)
            elif left == '':
                flattened.append(str(right))
            else:
                flattened.append(f"{left}_{right}")
        else:
            flattened.append(col)
    wide.columns = flattened
    rename_map = {}
    for new_name, (metric, cutoff) in metric_map.items():
        rename_map[f"{metric}_{cutoff}"] = new_name
    wide = wide.rename(columns=rename_map)
    for new_name in metric_map:
        if new_name not in wide.columns:
            wide[new_name] = np.nan
    return wide


def show_status_notes(df, placeholder_note=None, ready_note=None):
    if 'status' not in df.columns:
        return
    status_series = df['status'].dropna().astype(str)
    if status_series.empty:
        return
    if status_series.str.contains('placeholder', case=False).any() and placeholder_note:
        display(Markdown(placeholder_note))
    elif ready_note:
        display(Markdown(ready_note))
"""


def build_main_results_cells() -> list[dict]:
    return [
        md_cell(
            """
# 01 Main Results Table

LaTeX position:

- `tab:main-overall`

이 notebook은 main table과 간단한 summary design을 같이 본다.
현재 값은 paper draft에 맞춘 preview다.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
table_df = pd.read_csv(DATA_DIR / '01_main_results_table.csv')
metric_plot_df = pd.read_csv(DATA_DIR / '01_main_results_plot.csv')

display(Markdown('### Main table preview'))
display(style_ranked_table(
    table_df,
    numeric_columns=['SASRec','GRU4Rec','TiSASRec','FEARec','DuoRec','BSARec','FAME','DIF-SR','FDSA','RouteRec'],
    lower_is_better=False,
    caption='Main experimental results preview',
))
"""
        ),
        code_cell(
            """
display(Markdown('### Summary metric design'))
fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
dual_metric_grouped_plot(
    metric_plot_df,
    category_col='dataset',
    variant_col='variant',
    bar_col='ndcg20',
    line_col='hr10',
    ax=ax,
    title='Main comparison summary',
    bar_label='NDCG@20',
    line_label='HR@10',
    category_order=['Beauty', 'Foursquare', 'KuaiRec', 'Retail Rocket'],
    variant_order=['SASRec', 'BSARec', 'DuoRec', 'RouteRec'],
    rotate=25,
)
plt.show()
"""
        ),
    ]


def build_routing_cells() -> list[dict]:
    return [
        md_cell(
            """
# 02 Q2 Routing Control

LaTeX position:

- `fig:routing-control-panels`

This version keeps one direct quality panel, one compact stage-averaged group profile, and one compact case panel.

What this figure should prove:

- Panel (a): the routing source matters for ranking quality within each dataset.
- Panel (b): behavior groups still separate cleanly even after averaging over the 3 routing stages.
- Panel (c): several representative cases also split differently inside one semantic family even after stage averaging.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
quality_df = pd.read_csv(DATA_DIR / '02_routing_quality.csv')
profile_df = pd.read_csv(DATA_DIR / '02_routing_group_profile.csv')
intragroup_df = pd.read_csv(DATA_DIR / '02_routing_intragroup_profile.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Ranking quality by dataset'))
show_status_notes(
    quality_df,
    placeholder_note='**Template status**: current rows are draft-fill values. Replace them with the final export from the routing-control summary CSV once the confirm runs finish.',
    ready_note='**Export path**: this panel is already aligned with the final summary schema used for routing-control exports.',
)
print('Required export columns: dataset, variant_or_model, metric, cutoff, value, split, selection_rule, run_id')
print('Display rule: each dataset gets its own axis, and variants are compared inside that dataset.')

quality_plot_df = pivot_metric_frame(
    quality_df,
    id_cols=['dataset', 'variant_or_model'],
    metric_map={'ndcg20': ('NDCG', 20), 'hr10': ('HR', 10)},
).rename(columns={'variant_or_model': 'variant'})

variant_order = ['shared_ffn', 'hidden_only', 'mixed_hidden_behavior', 'behavior_guided']
variant_label_map = {
    'shared_ffn': 'Shared\nFFN',
    'hidden_only': 'Hidden\nonly',
    'mixed_hidden_behavior': 'Mixed',
    'behavior_guided': 'Behavior\nguided',
}
datasets = ['Beauty', 'Foursquare', 'KuaiRec', 'Retail Rocket']

fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), constrained_layout=True)
for axis, dataset in zip(axes.flat, datasets):
    dataset_plot_df = quality_plot_df[quality_plot_df['dataset'] == dataset].copy()
    category_bar_line_plot(
        dataset_plot_df,
        category_col='variant',
        bar_col='ndcg20',
        line_col='hr10',
        ax=axis,
        title=dataset,
        ylabel='NDCG@20',
        xlabel='',
        line_label='HR@10',
        order=variant_order,
        category_labels=variant_label_map,
        rotate=0,
    )
    axis.tick_params(axis='x', pad=8)

fig.suptitle('(a) Routing source comparison by dataset', y=1.02, fontsize=15)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Stage-averaged group routing profile'))
show_status_notes(
    profile_df,
    placeholder_note='**Template status**: this panel uses placeholder group-level routing profiles. To fill it with real values, keep `fmoe_diag_logging=true` and export per-session stage-level expert usage before aggregation.',
)
print('Needed logging for a real panel: dataset, session_id, split, stage, expert_family, expert_weight, repeat_ratio, switch_rate, focus_entropy, mean_gap')
print('Aggregation target for the main text: mean expert usage by (behavior_group, expert_family) after averaging over the 3 stages.')

profile_plot_df = (
    profile_df
    .groupby(['behavior_group', 'expert_family'], as_index=False)['usage']
    .mean()
)

fig, ax = plt.subplots(figsize=(8.7, 4.6), constrained_layout=True)
heatmap_panel(
    profile_plot_df,
    index='behavior_group',
    columns='expert_family',
    values='usage',
    ax=ax,
    title='(b) Stage-averaged routing profile',
    cmap='mako',
    fmt='.2f',
)
ax.set_xlabel('Expert family')
ax.set_ylabel('Behavior group')
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (c) Stage-averaged representative cases'))
show_status_notes(
    intragroup_df,
    placeholder_note='**Template status**: this panel stays placeholder until representative sessions are selected and within-family expert weights are exported for each stage.',
)
print('This panel should stay compact but more diverse: average over macro, mid, and micro first, then compare several representative cases.')
print('Recommended mix: repeat-heavy, focused, fast-tempo, and exploratory sessions so the semantic families feel distinct.')
print('Needed logging: case_name, stage, expert_group, expert_member, expert_weight, plus a short textual case description outside the plot.')

intragroup_plot_df = (
    intragroup_df
    .groupby(['case_name', 'expert_group', 'expert_member'], as_index=False)['usage']
    .mean()
)
intragroup_plot_df['case_label'] = intragroup_plot_df['case_name'] + ' (' + intragroup_plot_df['expert_group'] + ')'

cases = intragroup_plot_df['case_name'].drop_duplicates().tolist()
ncols = 2
nrows = int(np.ceil(len(cases) / ncols))
fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(10.2, 2.8 * nrows + 0.8),
    constrained_layout=True,
)
axes = np.atleast_1d(axes).ravel()

for axis, case_name in zip(axes, cases):
    case_df = intragroup_plot_df[intragroup_plot_df['case_name'] == case_name].copy()
    heatmap_panel(
        case_df,
        index='case_label',
        columns='expert_member',
        values='usage',
        ax=axis,
        title=case_name,
        cmap='crest',
        fmt='.2f',
        cbar=(case_name == cases[-1]),
    )
    axis.set_xlabel('Experts inside the selected semantic family')
    axis.set_ylabel('')

for axis in axes[len(cases):]:
    axis.axis('off')

plt.show()
"""
        ),
    ]


def build_stage_cells() -> list[dict]:
    return [
        md_cell(
            """
# 03 Q3 Stage Structure

LaTeX position:

- `fig:stage-structure-panels`

This figure should keep the final RouteRec layout visually fixed and test only the structural alternatives that sharpen the main claim.

What this figure should prove:

- Panel (a): the final 3-stage design outperforms reduced-stage or dense alternatives.
- Panel (b): the gain is not just stage count; the slow-to-fast order itself matters.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
dense_stage_df = pd.read_csv(DATA_DIR / '03_dense_vs_staged.csv')
wrapper_order_df = pd.read_csv(DATA_DIR / '03_wrapper_order.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Final layout vs reduced-stage alternatives'))
print('Main-text policy: keep the final design on the left, then step down toward simpler alternatives.')

dense_stage_plot_df = pivot_metric_frame(
    dense_stage_df,
    id_cols=['layout_variant', 'variant_or_model'],
    metric_map={'ndcg20': ('NDCG', 20), 'hr10': ('HR', 10)},
)
layout_label_map = {
    'three_stage': 'Final\\n3-stage',
    'best_two_stage': 'Best\\n2-stage',
    'best_single_stage': 'Best\\n1-stage',
    'dense_ffn': 'Dense\\nFFN',
}

fig, ax = plt.subplots(figsize=(9.1, 5.1), constrained_layout=True)
dual_metric_grouped_plot(
    dense_stage_plot_df,
    category_col='layout_variant',
    variant_col='variant_or_model',
    bar_col='ndcg20',
    line_col='hr10',
    ax=ax,
    title='(a) Final layout vs reduced-stage alternatives',
    bar_label='NDCG@20',
    line_label='HR@10',
    category_order=['three_stage', 'best_two_stage', 'best_single_stage', 'dense_ffn'],
    category_labels=layout_label_map,
    variant_order=['RouteRec'],
    rotate=0,
)
ax.tick_params(axis='x', pad=8)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Slow-to-fast ordering matters'))
print('Main-text panel (b) keeps only one intuitive order neighborhood around the final design.')
print('Recommended labels: final = slow-to-fast, micro_early = local-first, macro_late = global-late, mid_repeat = duplicated-mid.')

wrapper_order_plot_df = pivot_metric_frame(
    wrapper_order_df,
    id_cols=['layout_variant', 'variant_or_model'],
    metric_map={'ndcg20': ('NDCG', 20), 'hr10': ('HR', 10)},
)
order_label_map = {
    'final': 'Slow-to-fast\\n(final)',
    'micro_early': 'Local-first',
    'macro_late': 'Global-late',
    'mid_repeat': 'Duplicated\\nmid',
}

fig, ax = plt.subplots(figsize=(9.1, 5.1), constrained_layout=True)
dual_metric_grouped_plot(
    wrapper_order_plot_df,
    category_col='layout_variant',
    variant_col='variant_or_model',
    bar_col='ndcg20',
    line_col='hr10',
    ax=ax,
    title='(b) Slow-to-fast ordering matters',
    bar_label='NDCG@20',
    line_label='HR@10',
    category_order=['final', 'micro_early', 'macro_late', 'mid_repeat'],
    category_labels=order_label_map,
    variant_order=['RouteRec'],
    rotate=0,
)
ax.tick_params(axis='x', pad=8)
plt.show()
"""
        ),
    ]


def build_cue_cells() -> list[dict]:
    return [
        md_cell(
            """
# 04 Q4 Lightweight Cues

LaTeX position:

- `fig:cue-ablation-panels`

This figure should separate two claims clearly.

- Panel (a): lightweight cues still preserve ranking quality across datasets.
- Panel (b): removing metadata moves the router only mildly in operating space, instead of collapsing it into a trivial path.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
cue_df = pd.read_csv(DATA_DIR / '04_cue_scores.csv')
routing_stats_df = pd.read_csv(DATA_DIR / '04_cue_routing_stats.csv')
retention_df = pd.read_csv(DATA_DIR / '04_cue_retention.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Cue reduction by dataset'))
show_status_notes(
    cue_df,
    placeholder_note='**Template status**: current rows are draft-fill values. Replace them with the cue-family confirm export.',
    ready_note='**Export path**: panel (a) already matches the cue-family ablation summary schema.',
)
print('Required export columns: dataset, cue_setting, metric, cutoff, value, split, selection_rule, run_id')
print('Each subplot keeps one dataset on its own axis so the metric ranges do not fight each other.')

if 'cue_setting' not in cue_df.columns and 'variant_or_model' in cue_df.columns:
    cue_df = cue_df.rename(columns={'variant_or_model': 'cue_setting'})
cue_df['cue_setting'] = cue_df['cue_setting'].replace({'sequence_only': 'sequence_only_portable'})

cue_plot_df = pivot_metric_frame(
    cue_df,
    id_cols=['dataset', 'cue_setting'],
    metric_map={'ndcg20': ('NDCG', 20), 'hr10': ('HR', 10)},
)
cue_label_map = {
    'full': 'Full',
    'remove_category': 'No\\ncategory',
    'remove_time': 'No\\ntime',
    'sequence_only_portable': 'Sequence\\nonly',
}
datasets = ['Beauty', 'Foursquare', 'KuaiRec', 'Retail Rocket']

fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), constrained_layout=True)
for axis, dataset in zip(axes.flat, datasets):
    dataset_plot_df = cue_plot_df[cue_plot_df['dataset'] == dataset].copy()
    dual_metric_grouped_plot(
        dataset_plot_df,
        category_col='cue_setting',
        variant_col='dataset',
        bar_col='ndcg20',
        line_col='hr10',
        ax=axis,
        title=dataset,
        bar_label='NDCG@20',
        line_label='HR@10',
        category_order=['full', 'remove_category', 'remove_time', 'sequence_only_portable'],
        category_labels=cue_label_map,
        variant_order=[dataset],
        rotate=0,
        show_legend=False,
    )
    axis.tick_params(axis='x', pad=8)

fig.suptitle('(a) Cue reduction by dataset', y=1.02, fontsize=15)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Routing operating map under cue removal'))
show_status_notes(
    routing_stats_df,
    placeholder_note='**Template status**: this panel is placeholder until per-setting routing-richness diagnostics are exported.',
)
print('This panel uses internal operating-state evidence instead of another standalone ranking plot.')
print('Axes show how broad the router stays; bubble size shows how much of the full-model gain is retained after removing metadata cues.')
print('Recommended export columns: dataset, cue_setting, route_entropy, effective_experts, relative_gain')

stats_df = routing_stats_df.copy()
stats_df['cue_setting'] = stats_df['cue_setting'].replace({'sequence_only': 'sequence_only_portable'})
retention_plot_df = retention_df[['dataset', 'retention_target', 'relative_gain']].rename(
    columns={'retention_target': 'cue_setting'}
)
full_ref = stats_df[['dataset']].drop_duplicates().assign(cue_setting='full', relative_gain=1.0)
operating_df = stats_df.merge(
    pd.concat([full_ref, retention_plot_df], ignore_index=True),
    on=['dataset', 'cue_setting'],
    how='left',
)
cue_order = ['full', 'remove_category', 'remove_time', 'sequence_only_portable']
operating_df['cue_setting'] = pd.Categorical(
    operating_df['cue_setting'],
    categories=cue_order,
    ordered=True,
)
operating_df = operating_df.sort_values(['dataset', 'cue_setting'])

dataset_palette = {
    'Beauty': '#5B7C99',
    'Foursquare': '#0F766E',
    'KuaiRec': '#C96567',
    'Retail Rocket': '#D97706',
}
setting_short = {
    'full': 'F',
    'remove_category': 'C-',
    'remove_time': 'T-',
    'sequence_only_portable': 'Seq',
}

fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
for dataset, dataset_df in operating_df.groupby('dataset', sort=False):
    ax.plot(
        dataset_df['effective_experts'],
        dataset_df['route_entropy'],
        color=dataset_palette.get(dataset, '#666666'),
        linewidth=1.8,
        alpha=0.4,
        zorder=1,
    )
    for _, row in dataset_df.iterrows():
        ax.scatter(
            row['effective_experts'],
            row['route_entropy'],
            s=280 * row['relative_gain'],
            color=dataset_palette.get(dataset, '#666666'),
            edgecolor='white',
            linewidth=0.9,
            alpha=0.82,
            zorder=2,
            label=dataset if row['cue_setting'] == 'full' else None,
        )
        ax.annotate(
            setting_short[row['cue_setting']],
            (row['effective_experts'], row['route_entropy']),
            textcoords='offset points',
            xytext=(0, 7),
            ha='center',
            fontsize=9,
            color='#243447',
        )

ax.set_title('(b) Routing operating map under cue removal')
ax.set_xlabel('Effective experts')
ax.set_ylabel('Route entropy')
ax.grid(alpha=0.35)

dataset_legend = ax.legend(loc='lower left', title='dataset')
size_values = [1.00, 0.95, 0.90]
size_handles = [
    ax.scatter([], [], s=280 * value, color='#B8BFC8', edgecolor='white', linewidth=0.9)
    for value in size_values
]
ax.legend(
    size_handles,
    [f'{int(value * 100)}% gain retained' for value in size_values],
    loc='upper right',
    title='bubble size',
    scatterpoints=1,
    frameon=False,
)
ax.add_artist(dataset_legend)
plt.show()
"""
        ),
    ]


def build_behavior_cells() -> list[dict]:
    return [
        md_cell(
            """
# 05 Q5 Feature Intervention

LaTeX position:

- `fig:behavior-regime-panels`

This slot is now used for counterfactual cue interventions and regime-sensitive evidence that routing is doing useful work.

What this figure should prove:

- Panel (a): semantically targeted interventions hurt quality in different ways.
- Panel (b): RouteRec gains are larger in behavioral regimes where routing becomes more selective, and that pattern is visible both slice by slice and in summary.

This keeps the focus on why MoE-style routing helps rather than on another family-level diagnostic heatmap.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
intervention_quality_df = pd.read_csv(DATA_DIR / '05_feature_intervention_scores.csv')
slice_quality_df = pd.read_csv(DATA_DIR / '05_behavior_slice_quality.csv')
slice_gain_df = pd.read_csv(DATA_DIR / '05_behavior_slice_gain.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Semantic intervention quality'))
show_status_notes(
    intervention_quality_df,
    placeholder_note='**Template status**: this panel is placeholder until eval-time semantic intervention hooks are added to the selected RouteRec run.',
)
print('Recommended semantic interventions: shuffle_all, repeat_flatten, switch_boost, tempo_compress, popularity_spike')
print('Needed export columns: intervention, metric, cutoff, value, split, selection_rule, run_id')

intervention_quality_plot_df = pivot_metric_frame(
    intervention_quality_df,
    id_cols=['intervention', 'variant_or_model'],
    metric_map={'ndcg20': ('NDCG', 20), 'hr10': ('HR', 10)},
).rename(columns={'variant_or_model': 'variant'})
intervention_label_map = {
    'full': 'Full',
    'shuffle_all': 'Shuffle\\nall',
    'repeat_flatten': 'Flatten\\nrepeat',
    'switch_boost': 'Boost\\nswitch',
    'tempo_compress': 'Compress\\ntempo',
    'popularity_spike': 'Spike\\npopularity',
}

fig, ax = plt.subplots(figsize=(11.1, 5.1), constrained_layout=True)
dual_metric_grouped_plot(
    intervention_quality_plot_df,
    category_col='intervention',
    variant_col='variant',
    bar_col='ndcg20',
    line_col='hr10',
    ax=ax,
    title='(a) Quality under semantic interventions',
    bar_label='NDCG@20',
    line_label='HR@10',
    category_order=['full', 'shuffle_all', 'repeat_flatten', 'switch_boost', 'tempo_compress', 'popularity_spike'],
    category_labels=intervention_label_map,
    variant_order=['RouteRec'],
    rotate=0,
)
ax.tick_params(axis='x', pad=8)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Where routing helps most'))
show_status_notes(
    slice_gain_df,
    placeholder_note='**Template status**: this panel is placeholder until behavior-slice gain summaries are exported from held-out diagnostics.',
)
print('This panel combines a slice-wise outcome view with a gain-vs-selectivity summary so the evidence does not feel too thin.')
print('Needed logging: slice_name, variant_or_model, metric, cutoff, value, route_concentration, relative_gain after held-out slice aggregation')

slice_quality_plot_df = pivot_metric_frame(
    slice_quality_df,
    id_cols=['slice_name', 'variant_or_model'],
    metric_map={'ndcg20': ('NDCG', 20), 'hr10': ('HR', 10)},
).rename(columns={'variant_or_model': 'variant'})
slice_gain_plot_df = slice_gain_df.copy()
slice_gain_plot_df['relative_gain_pct'] = slice_gain_plot_df['relative_gain'] * 100.0
slice_gain_plot_df['slice_display'] = slice_gain_plot_df['slice_name'].map({
    'repeat-heavy': 'Repeat-heavy',
    'fast-tempo': 'Fast-tempo',
    'focused': 'Focused',
    'exploration-heavy': 'Exploration-heavy',
})
slice_order = ['repeat-heavy', 'fast-tempo', 'focused', 'exploration-heavy']
slice_label_map = {
    'repeat-heavy': 'Repeat\\nheavy',
    'fast-tempo': 'Fast\\ntempo',
    'focused': 'Focused',
    'exploration-heavy': 'Exploration\\nheavy',
}
variant_order = ['best_baseline', 'shared_ffn', 'RouteRec']
variant_palette = {
    'best_baseline': '#5B7C99',
    'shared_ffn': '#C7CDD4',
    'RouteRec': '#0F766E',
}

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

ax = axes[0]
x = np.arange(len(slice_order), dtype=float)
width = 0.22
for idx, variant in enumerate(variant_order):
    subset = slice_quality_plot_df[slice_quality_plot_df['variant'] == variant].set_index('slice_name')
    values = [subset.loc[slice_name, 'ndcg20'] for slice_name in slice_order]
    ax.bar(
        x + (idx - 1) * width,
        values,
        width=width,
        color=variant_palette[variant],
        alpha=0.82,
        edgecolor='white',
        linewidth=0.8,
        label=variant,
        zorder=2,
    )
ax.set_xticks(x)
ax.set_xticklabels([slice_label_map[key] for key in slice_order])
ax.set_ylabel('NDCG@20')
ax.set_title('Slice-wise quality')
ax.grid(axis='y', alpha=0.35)

twin = ax.twinx()
route_line = slice_gain_plot_df.set_index('slice_name').loc[slice_order, 'route_concentration']
twin.plot(
    x,
    route_line.values,
    color='#D97706',
    marker='o',
    linewidth=2.0,
    markersize=5.5,
    label='Routing concentration',
    zorder=3,
)
twin.set_ylabel('Routing concentration')
handles, labels = ax.get_legend_handles_labels()
line_handles, line_labels = twin.get_legend_handles_labels()
ax.legend(handles + line_handles, labels + line_labels, loc='upper left', frameon=False)

scatter_panel(
    slice_gain_plot_df,
    x='route_concentration',
    y='relative_gain_pct',
    hue='slice_display',
    label_col='slice_display',
    ax=axes[1],
    title='Gain vs routing selectivity',
    ylabel='RouteRec gain over comparator (%)',
    xlabel='Routing concentration',
)
coef = np.polyfit(slice_gain_plot_df['route_concentration'], slice_gain_plot_df['relative_gain_pct'], deg=1)
x_grid = np.linspace(slice_gain_plot_df['route_concentration'].min(), slice_gain_plot_df['route_concentration'].max(), 100)
axes[1].plot(x_grid, coef[0] * x_grid + coef[1], linestyle='--', linewidth=1.8, color='#6B7280', alpha=0.9)
axes[1].axhline(0.0, linestyle=':', linewidth=1.1, color='#888888')

fig.suptitle('(b) Where routing helps most', y=1.02, fontsize=15)
plt.show()
"""
        ),
    ]


def build_plain_table_cells(title: str, csv_name: str, caption: str, precision: int = 3) -> list[dict]:
    return [
        md_cell(f"# {title}\n\nLaTeX table position preview notebook."),
        code_cell(
            COMMON_IMPORTS
            + f"""
table_df = pd.read_csv(DATA_DIR / '{csv_name}')
display(style_plain_table(table_df, caption='{caption}', precision={precision}))
"""
        ),
    ]


def build_full_results_cells() -> list[dict]:
    return [
        md_cell(
            """
# A05 Appendix Full Results Table

LaTeX position:

- `tab:appendix-full-seen`

이 notebook은 full cutoff grid preview 전용이다.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
table_df = pd.read_csv(DATA_DIR / 'A05_full_results_table.csv')
display(style_ranked_table(
    table_df,
    numeric_columns=['SASRec','GRU4Rec','TiSASRec','FEARec','DuoRec','BSARec','FAME','DIF-SR','FDSA','RouteRec'],
    lower_is_better=False,
    caption='Appendix full results preview',
))
"""
        ),
    ]


def build_extended_structure_cells() -> list[dict]:
    return [
        md_cell(
            """
# A06 Appendix Extended Structural Ablations

LaTeX figure position `fig:appendix-stage-layout`.

이 notebook은 본문 Q3에서 뺀 세부 variant를 appendix용으로 모아둔다.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
semantic_df = pd.read_csv(DATA_DIR / 'A06_semantic_variants.csv')
scope_df = pd.read_csv(DATA_DIR / 'A06_scope_layout_variants.csv')
order_df = pd.read_csv(DATA_DIR / 'A06_order_router_variants.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Semantic cue and family variants'))
fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
dual_metric_grouped_plot(
    semantic_df,
    'variant',
    'series',
    'ndcg20',
    'hr10',
    ax=ax,
    title='(a) Semantic cue and family variants',
    bar_label='NDCG@20',
    line_label='HR@10',
    variant_order=['RouteRec'],
    rotate=18,
)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Stage semantics and layout'))
fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
dual_metric_grouped_plot(
    scope_df,
    'variant',
    'series',
    'ndcg20',
    'hr10',
    ax=ax,
    title='(b) Stage semantics and layout',
    bar_label='NDCG@20',
    line_label='HR@10',
    variant_order=['RouteRec'],
    rotate=18,
)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (c) Placement, order, and router variants'))
fig, ax = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
dual_metric_grouped_plot(
    order_df,
    'variant',
    'series',
    'ndcg20',
    'hr10',
    ax=ax,
    title='(c) Placement, order, and router variants',
    bar_label='NDCG@20',
    line_label='HR@10',
    variant_order=['RouteRec'],
    rotate=18,
)
plt.show()
"""
        ),
    ]


def build_diagnostics_cells() -> list[dict]:
    return [
        md_cell(
            """
# A07 Appendix Routing Diagnostics

LaTeX figure position `fig:appendix-diagnostics`.

이 notebook은 main-text에서 뺀 indirect diagnostics를 appendix에서 소화한다.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
usage_df = pd.read_csv(DATA_DIR / 'A07_expert_usage.csv')
entropy_df = pd.read_csv(DATA_DIR / 'A07_entropy_effective.csv')
consistency_df = pd.read_csv(DATA_DIR / 'A07_stage_consistency.csv')
feature_df = pd.read_csv(DATA_DIR / 'A07_feature_bucket_patterns.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Expert usage by stage'))
fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
heatmap_panel(usage_df, 'stage', 'expert_label', 'usage', ax=ax, title='(a) Expert usage by stage', cmap='mako', fmt='.2f')
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Entropy and effective experts'))
entropy_long = entropy_df.melt(id_vars=['stage'], value_vars=['entropy', 'effective_experts'], var_name='measure', value_name='value')
fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
single_metric_bar(entropy_long, x='stage', y='value', hue='measure', ax=ax, title='(b) Entropy and effective experts', ylabel='Value', xlabel='Stage', rotate=0)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (c) Routing consistency diagnostic'))
fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
line_panel(consistency_df, x='similarity_bucket', y='consistency', hue='stage', ax=ax, title='(c) Routing consistency diagnostic', ylabel='Consistency', xlabel='Similarity bucket', hue_order=['macro', 'mid', 'micro'])
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (d) Feature-bucket patterns'))
fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
heatmap_panel(feature_df, 'feature_family', 'bucket_label', 'usage', ax=ax, title='(d) Feature-bucket patterns', cmap='rocket', fmt='.2f')
plt.show()
"""
        ),
    ]


def build_appendix_behavior_cells() -> list[dict]:
    return [
        md_cell(
            """
# A08 Appendix Behavior Slices

LaTeX figure position `fig:appendix-behavior-slices`.

이 notebook은 main Q5를 보강하는 확장 slice diagnostics 전용이다.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
slice_quality_df = pd.read_csv(DATA_DIR / 'A08_behavior_slice_quality.csv')
slice_gain_df = pd.read_csv(DATA_DIR / 'A08_behavior_slice_gain.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Slice-wise ranking quality'))
slice_quality_plot_df = pivot_metric_frame(
    slice_quality_df,
    id_cols=['slice_name', 'variant_or_model'],
    metric_map={'ndcg20': ('NDCG', 20), 'hr10': ('HR', 10)},
).rename(columns={'variant_or_model': 'variant'})

fig, ax = plt.subplots(figsize=(10.6, 4.8), constrained_layout=True)
dual_metric_grouped_plot(
    slice_quality_plot_df,
    'slice_name',
    'variant',
    'ndcg20',
    'hr10',
    ax=ax,
    title='(a) Slice-wise ranking quality',
    bar_label='NDCG@20',
    line_label='HR@10',
    variant_order=['best_baseline', 'shared_ffn', 'RouteRec'],
    rotate=20,
)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Relative gain and concentration'))
fig, ax = plt.subplots(figsize=(9.0, 4.7), constrained_layout=True)
category_bar_line_plot(
    slice_gain_df,
    category_col='slice_name',
    bar_col='relative_gain',
    line_col='route_concentration',
    ax=ax,
    title='(b) Relative gain and concentration',
    ylabel='Relative gain',
    xlabel='Behavior slice',
    line_label='Routing concentration',
    order=['repeat-heavy', 'fast-tempo', 'focused', 'exploration-heavy'],
    rotate=20,
)
plt.show()
"""
        ),
    ]


def build_transfer_cells() -> list[dict]:
    return [
        md_cell(
            """
# A09 Appendix Transfer or Portability Variants

LaTeX figure position `fig:appendix-transfer-panels`.

이 notebook은 low-resource portability와 transfer variants를 appendix에서 정리한다.
"""
        ),
        code_cell(
            COMMON_IMPORTS
            + """
low_resource_df = pd.read_csv(DATA_DIR / 'A09_low_resource_transfer.csv')
variant_df = pd.read_csv(DATA_DIR / 'A09_transfer_variants.csv')
order_df = pd.read_csv(DATA_DIR / 'A09_multi_source_order.csv')
"""
        ),
        code_cell(
            """
display(Markdown('### (a) Low-resource transfer curves'))
fig, ax = plt.subplots(figsize=(8.8, 4.7), constrained_layout=True)
line_panel(low_resource_df, x='shot_ratio', y='ndcg20', hue='variant', ax=ax, title='(a) Low-resource transfer curves', ylabel='NDCG@20', xlabel='Training ratio', hue_order=['full', 'sequence_only', 'shared_ffn'])
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (b) Transfer strategy variants'))
fig, ax = plt.subplots(figsize=(8.8, 4.7), constrained_layout=True)
dual_metric_grouped_plot(variant_df, 'variant', 'series', 'ndcg20', 'hr10', ax=ax, title='(b) Transfer strategy variants', bar_label='NDCG@20', line_label='HR@10', variant_order=['RouteRec'], rotate=20)
plt.show()
"""
        ),
        code_cell(
            """
display(Markdown('### (c) Multi-source order effects'))
fig, ax = plt.subplots(figsize=(8.8, 4.7), constrained_layout=True)
dual_metric_grouped_plot(order_df, 'variant', 'series', 'ndcg20', 'hr10', ax=ax, title='(c) Multi-source order effects', bar_label='NDCG@20', line_label='HR@10', variant_order=['RouteRec'], rotate=20)
plt.show()
"""
        ),
    ]


def metric_rows(entries: list[tuple[str, str, float, float]], *, panel: str, source_path: str, notes: str, status: str) -> list[dict]:
    rows: list[dict] = []
    for dataset, variant, ndcg20, hr10 in entries:
        rows.append(
            {
                'paper_section': panel.split('_')[0],
                'panel': panel,
                'dataset': dataset,
                'variant_or_model': variant,
                'metric': 'NDCG',
                'cutoff': 20,
                'value': ndcg20,
                'split': 'test',
                'selection_rule': 'best_valid_seen_target',
                'run_id': 'template_seed',
                'source_path': source_path,
                'notes': notes,
                'status': status,
            }
        )
        rows.append(
            {
                'paper_section': panel.split('_')[0],
                'panel': panel,
                'dataset': dataset,
                'variant_or_model': variant,
                'metric': 'HR',
                'cutoff': 10,
                'value': hr10,
                'split': 'test',
                'selection_rule': 'best_valid_seen_target',
                'run_id': 'template_seed',
                'source_path': source_path,
                'notes': notes,
                'status': status,
            }
        )
    return rows


def variant_metric_rows(entries: list[tuple[str, str, float, float]], *, panel: str, category_key: str, source_path: str, notes: str, status: str) -> list[dict]:
    rows: list[dict] = []
    for category_value, variant, ndcg20, hr10 in entries:
        for metric, cutoff, value in [('NDCG', 20, ndcg20), ('HR', 10, hr10)]:
            rows.append(
                {
                    'paper_section': panel.split('_')[0],
                    'panel': panel,
                    category_key: category_value,
                    'variant_or_model': variant,
                    'metric': metric,
                    'cutoff': cutoff,
                    'value': value,
                    'split': 'test',
                    'selection_rule': 'best_valid_seen_target',
                    'run_id': 'template_seed',
                    'source_path': source_path,
                    'notes': notes,
                    'status': status,
                }
            )
    return rows


def write_all_csvs() -> None:
    write_csv(
        '01_main_results_table.csv',
        [
            {'dataset': 'Beauty', 'metric': 'HR@10', 'SASRec': 0.135, 'GRU4Rec': 0.072, 'TiSASRec': 0.132, 'FEARec': 0.126, 'DuoRec': 0.115, 'BSARec': 0.052, 'FAME': 0.032, 'DIF-SR': 0.106, 'FDSA': 0.135, 'RouteRec': 0.162},
            {'dataset': 'Beauty', 'metric': 'NDCG@10', 'SASRec': 0.075, 'GRU4Rec': 0.034, 'TiSASRec': 0.088, 'FEARec': 0.070, 'DuoRec': 0.068, 'BSARec': 0.032, 'FAME': 0.021, 'DIF-SR': 0.069, 'FDSA': 0.091, 'RouteRec': 0.098},
            {'dataset': 'Beauty', 'metric': 'MRR@20', 'SASRec': 0.059, 'GRU4Rec': 0.025, 'TiSASRec': 0.077, 'FEARec': 0.054, 'DuoRec': 0.057, 'BSARec': 0.026, 'FAME': 0.019, 'DIF-SR': 0.059, 'FDSA': 0.081, 'RouteRec': 0.082},
            {'dataset': 'Retail Rocket', 'metric': 'HR@10', 'SASRec': 0.561, 'GRU4Rec': 0.521, 'TiSASRec': 0.544, 'FEARec': 0.544, 'DuoRec': 0.538, 'BSARec': 0.470, 'FAME': 0.470, 'DIF-SR': 0.437, 'FDSA': 0.513, 'RouteRec': 0.561},
            {'dataset': 'Retail Rocket', 'metric': 'NDCG@10', 'SASRec': 0.410, 'GRU4Rec': 0.386, 'TiSASRec': 0.396, 'FEARec': 0.394, 'DuoRec': 0.386, 'BSARec': 0.396, 'FAME': 0.393, 'DIF-SR': 0.390, 'FDSA': 0.365, 'RouteRec': 0.415},
            {'dataset': 'Retail Rocket', 'metric': 'MRR@20', 'SASRec': 0.367, 'GRU4Rec': 0.348, 'TiSASRec': 0.355, 'FEARec': 0.352, 'DuoRec': 0.344, 'BSARec': 0.374, 'FAME': 0.371, 'DIF-SR': 0.374, 'FDSA': 0.324, 'RouteRec': 0.374},
            {'dataset': 'Foursquare', 'metric': 'HR@10', 'SASRec': 0.323, 'GRU4Rec': 0.213, 'TiSASRec': 0.316, 'FEARec': 0.304, 'DuoRec': 0.320, 'BSARec': 0.246, 'FAME': 0.255, 'DIF-SR': 0.303, 'FDSA': 0.300, 'RouteRec': 0.323},
            {'dataset': 'Foursquare', 'metric': 'NDCG@10', 'SASRec': 0.201, 'GRU4Rec': 0.146, 'TiSASRec': 0.202, 'FEARec': 0.193, 'DuoRec': 0.196, 'BSARec': 0.164, 'FAME': 0.165, 'DIF-SR': 0.198, 'FDSA': 0.200, 'RouteRec': 0.204},
            {'dataset': 'Foursquare', 'metric': 'MRR@20', 'SASRec': 0.168, 'GRU4Rec': 0.128, 'TiSASRec': 0.170, 'FEARec': 0.162, 'DuoRec': 0.161, 'BSARec': 0.141, 'FAME': 0.140, 'DIF-SR': 0.169, 'FDSA': 0.172, 'RouteRec': 0.171},
            {'dataset': 'ML-1M', 'metric': 'HR@10', 'SASRec': 0.148, 'GRU4Rec': 0.141, 'TiSASRec': 0.154, 'FEARec': 0.149, 'DuoRec': 0.152, 'BSARec': 0.159, 'FAME': 0.156, 'DIF-SR': 0.150, 'FDSA': 0.159, 'RouteRec': 0.156},
            {'dataset': 'ML-1M', 'metric': 'NDCG@10', 'SASRec': 0.077, 'GRU4Rec': 0.074, 'TiSASRec': 0.084, 'FEARec': 0.072, 'DuoRec': 0.081, 'BSARec': 0.085, 'FAME': 0.085, 'DIF-SR': 0.076, 'FDSA': 0.084, 'RouteRec': 0.077},
            {'dataset': 'ML-1M', 'metric': 'MRR@20', 'SASRec': 0.061, 'GRU4Rec': 0.059, 'TiSASRec': 0.069, 'FEARec': 0.055, 'DuoRec': 0.065, 'BSARec': 0.068, 'FAME': 0.069, 'DIF-SR': 0.060, 'FDSA': 0.066, 'RouteRec': 0.059},
            {'dataset': 'LastFM', 'metric': 'HR@10', 'SASRec': 0.399, 'GRU4Rec': 0.309, 'TiSASRec': 0.378, 'FEARec': 0.377, 'DuoRec': 0.383, 'BSARec': 0.345, 'FAME': 0.331, 'DIF-SR': 0.362, 'FDSA': 0.366, 'RouteRec': 0.379},
            {'dataset': 'LastFM', 'metric': 'NDCG@10', 'SASRec': 0.327, 'GRU4Rec': 0.267, 'TiSASRec': 0.320, 'FEARec': 0.317, 'DuoRec': 0.308, 'BSARec': 0.302, 'FAME': 0.290, 'DIF-SR': 0.315, 'FDSA': 0.317, 'RouteRec': 0.325},
            {'dataset': 'LastFM', 'metric': 'MRR@20', 'SASRec': 0.305, 'GRU4Rec': 0.256, 'TiSASRec': 0.303, 'FEARec': 0.299, 'DuoRec': 0.286, 'BSARec': 0.290, 'FAME': 0.278, 'DIF-SR': 0.302, 'FDSA': 0.303, 'RouteRec': 0.310},
            {'dataset': 'KuaiRec', 'metric': 'HR@10', 'SASRec': 0.106, 'GRU4Rec': 0.095, 'TiSASRec': 0.101, 'FEARec': 0.101, 'DuoRec': 0.100, 'BSARec': 0.092, 'FAME': 0.101, 'DIF-SR': 0.092, 'FDSA': 0.097, 'RouteRec': 0.106},
            {'dataset': 'KuaiRec', 'metric': 'NDCG@10', 'SASRec': 0.094, 'GRU4Rec': 0.079, 'TiSASRec': 0.092, 'FEARec': 0.092, 'DuoRec': 0.092, 'BSARec': 0.089, 'FAME': 0.092, 'DIF-SR': 0.087, 'FDSA': 0.087, 'RouteRec': 0.097},
            {'dataset': 'KuaiRec', 'metric': 'MRR@20', 'SASRec': 0.092, 'GRU4Rec': 0.075, 'TiSASRec': 0.089, 'FEARec': 0.090, 'DuoRec': 0.091, 'BSARec': 0.088, 'FAME': 0.090, 'DIF-SR': 0.086, 'FDSA': 0.084, 'RouteRec': 0.095},
            {'dataset': 'All', 'metric': 'AvgRank↓', 'SASRec': 3.420, 'GRU4Rec': 9.060, 'TiSASRec': 3.720, 'FEARec': 5.920, 'DuoRec': 5.690, 'BSARec': 6.560, 'FAME': 6.860, 'DIF-SR': 6.420, 'FDSA': 5.140, 'RouteRec': 2.220},
        ],
    )
    write_csv(
        '01_main_results_plot.csv',
        [
            {'dataset': 'Beauty', 'variant': 'SASRec', 'ndcg20': 0.084, 'hr10': 0.135},
            {'dataset': 'Beauty', 'variant': 'BSARec', 'ndcg20': 0.034, 'hr10': 0.052},
            {'dataset': 'Beauty', 'variant': 'DuoRec', 'ndcg20': 0.083, 'hr10': 0.115},
            {'dataset': 'Beauty', 'variant': 'RouteRec', 'ndcg20': 0.108, 'hr10': 0.162},
            {'dataset': 'Foursquare', 'variant': 'SASRec', 'ndcg20': 0.217, 'hr10': 0.323},
            {'dataset': 'Foursquare', 'variant': 'BSARec', 'ndcg20': 0.174, 'hr10': 0.246},
            {'dataset': 'Foursquare', 'variant': 'DuoRec', 'ndcg20': 0.209, 'hr10': 0.320},
            {'dataset': 'Foursquare', 'variant': 'RouteRec', 'ndcg20': 0.218, 'hr10': 0.323},
            {'dataset': 'KuaiRec', 'variant': 'SASRec', 'ndcg20': 0.100, 'hr10': 0.106},
            {'dataset': 'KuaiRec', 'variant': 'BSARec', 'ndcg20': 0.089, 'hr10': 0.092},
            {'dataset': 'KuaiRec', 'variant': 'DuoRec', 'ndcg20': 0.097, 'hr10': 0.100},
            {'dataset': 'KuaiRec', 'variant': 'RouteRec', 'ndcg20': 0.111, 'hr10': 0.106},
            {'dataset': 'Retail Rocket', 'variant': 'SASRec', 'ndcg20': 0.428, 'hr10': 0.561},
            {'dataset': 'Retail Rocket', 'variant': 'BSARec', 'ndcg20': 0.403, 'hr10': 0.470},
            {'dataset': 'Retail Rocket', 'variant': 'DuoRec', 'ndcg20': 0.406, 'hr10': 0.538},
            {'dataset': 'Retail Rocket', 'variant': 'RouteRec', 'ndcg20': 0.433, 'hr10': 0.561},
        ],
    )

    write_csv(
        '02_routing_quality.csv',
        metric_rows(
            [
                ('Beauty', 'shared_ffn', 0.068, 0.118),
                ('Beauty', 'hidden_only', 0.072, 0.125),
                ('Beauty', 'mixed_hidden_behavior', 0.075, 0.130),
                ('Beauty', 'behavior_guided', 0.081, 0.139),
                ('Foursquare', 'shared_ffn', 0.134, 0.241),
                ('Foursquare', 'hidden_only', 0.138, 0.251),
                ('Foursquare', 'mixed_hidden_behavior', 0.142, 0.257),
                ('Foursquare', 'behavior_guided', 0.149, 0.269),
                ('KuaiRec', 'shared_ffn', 0.108, 0.165),
                ('KuaiRec', 'hidden_only', 0.112, 0.172),
                ('KuaiRec', 'mixed_hidden_behavior', 0.115, 0.175),
                ('KuaiRec', 'behavior_guided', 0.120, 0.181),
                ('Retail Rocket', 'shared_ffn', 0.284, 0.520),
                ('Retail Rocket', 'hidden_only', 0.298, 0.537),
                ('Retail Rocket', 'mixed_hidden_behavior', 0.304, 0.545),
                ('Retail Rocket', 'behavior_guided', 0.319, 0.562),
            ],
            panel='Q2_quality',
            source_path='writing/results/02_routing_control/02a_routing_control_quality.csv',
            notes='Draft-fill values for layout preview. Replace with confirm export.',
            status='draft_fill',
        ),
    )
    write_csv(
        '02_routing_group_profile.csv',
        [
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'macro', 'expert_family': 'memory', 'usage': 0.42, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'macro', 'expert_family': 'focus', 'usage': 0.18, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'macro', 'expert_family': 'tempo', 'usage': 0.15, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'macro', 'expert_family': 'exposure', 'usage': 0.25, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'mid', 'expert_family': 'memory', 'usage': 0.39, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'mid', 'expert_family': 'focus', 'usage': 0.23, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'mid', 'expert_family': 'tempo', 'usage': 0.12, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'mid', 'expert_family': 'exposure', 'usage': 0.26, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'micro', 'expert_family': 'memory', 'usage': 0.33, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'micro', 'expert_family': 'focus', 'usage': 0.31, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'micro', 'expert_family': 'tempo', 'usage': 0.09, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'repeat-heavy', 'stage': 'micro', 'expert_family': 'exposure', 'usage': 0.27, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'macro', 'expert_family': 'memory', 'usage': 0.16, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'macro', 'expert_family': 'focus', 'usage': 0.14, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'macro', 'expert_family': 'tempo', 'usage': 0.44, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'macro', 'expert_family': 'exposure', 'usage': 0.26, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'mid', 'expert_family': 'memory', 'usage': 0.18, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'mid', 'expert_family': 'focus', 'usage': 0.19, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'mid', 'expert_family': 'tempo', 'usage': 0.37, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'fast-tempo', 'stage': 'mid', 'expert_family': 'exposure', 'usage': 0.26, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'focused', 'stage': 'macro', 'expert_family': 'memory', 'usage': 0.21, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'focused', 'stage': 'macro', 'expert_family': 'focus', 'usage': 0.38, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'focused', 'stage': 'macro', 'expert_family': 'tempo', 'usage': 0.10, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'focused', 'stage': 'macro', 'expert_family': 'exposure', 'usage': 0.31, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'exploration-heavy', 'stage': 'macro', 'expert_family': 'memory', 'usage': 0.14, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'exploration-heavy', 'stage': 'macro', 'expert_family': 'focus', 'usage': 0.17, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'exploration-heavy', 'stage': 'macro', 'expert_family': 'tempo', 'usage': 0.24, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_profile', 'behavior_group': 'exploration-heavy', 'stage': 'macro', 'expert_family': 'exposure', 'usage': 0.45, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/group_profile.csv', 'notes': 'Placeholder profile until diag logging is exported.', 'status': 'placeholder_requires_logging'},
        ],
    )
    write_csv(
        '02_routing_intragroup_profile.csv',
        [
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'macro', 'expert_group': 'memory', 'expert_member': 'e1', 'usage': 0.30, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'macro', 'expert_group': 'memory', 'expert_member': 'e2', 'usage': 0.11, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'macro', 'expert_group': 'memory', 'expert_member': 'e3', 'usage': 0.05, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'macro', 'expert_group': 'memory', 'expert_member': 'e4', 'usage': 0.02, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'mid', 'expert_group': 'memory', 'expert_member': 'e1', 'usage': 0.24, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'mid', 'expert_group': 'memory', 'expert_member': 'e2', 'usage': 0.10, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'mid', 'expert_group': 'memory', 'expert_member': 'e3', 'usage': 0.04, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'mid', 'expert_group': 'memory', 'expert_member': 'e4', 'usage': 0.01, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'micro', 'expert_group': 'memory', 'expert_member': 'e1', 'usage': 0.18, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'micro', 'expert_group': 'memory', 'expert_member': 'e2', 'usage': 0.09, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'micro', 'expert_group': 'memory', 'expert_member': 'e3', 'usage': 0.05, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'repeat-heavy user', 'stage': 'micro', 'expert_group': 'memory', 'expert_member': 'e4', 'usage': 0.03, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'macro', 'expert_group': 'exposure', 'expert_member': 'e1', 'usage': 0.07, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'macro', 'expert_group': 'exposure', 'expert_member': 'e2', 'usage': 0.13, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'macro', 'expert_group': 'exposure', 'expert_member': 'e3', 'usage': 0.17, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'macro', 'expert_group': 'exposure', 'expert_member': 'e4', 'usage': 0.10, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'mid', 'expert_group': 'exposure', 'expert_member': 'e1', 'usage': 0.06, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'mid', 'expert_group': 'exposure', 'expert_member': 'e2', 'usage': 0.11, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'mid', 'expert_group': 'exposure', 'expert_member': 'e3', 'usage': 0.15, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'mid', 'expert_group': 'exposure', 'expert_member': 'e4', 'usage': 0.09, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'micro', 'expert_group': 'exposure', 'expert_member': 'e1', 'usage': 0.04, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'micro', 'expert_group': 'exposure', 'expert_member': 'e2', 'usage': 0.09, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'micro', 'expert_group': 'exposure', 'expert_member': 'e3', 'usage': 0.14, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'exploratory user', 'stage': 'micro', 'expert_group': 'exposure', 'expert_member': 'e4', 'usage': 0.10, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'macro', 'expert_group': 'focus', 'expert_member': 'e1', 'usage': 0.09, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'macro', 'expert_group': 'focus', 'expert_member': 'e2', 'usage': 0.18, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'macro', 'expert_group': 'focus', 'expert_member': 'e3', 'usage': 0.07, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'macro', 'expert_group': 'focus', 'expert_member': 'e4', 'usage': 0.04, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'mid', 'expert_group': 'focus', 'expert_member': 'e1', 'usage': 0.07, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'mid', 'expert_group': 'focus', 'expert_member': 'e2', 'usage': 0.20, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'mid', 'expert_group': 'focus', 'expert_member': 'e3', 'usage': 0.09, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'mid', 'expert_group': 'focus', 'expert_member': 'e4', 'usage': 0.05, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'micro', 'expert_group': 'focus', 'expert_member': 'e1', 'usage': 0.05, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'micro', 'expert_group': 'focus', 'expert_member': 'e2', 'usage': 0.22, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'micro', 'expert_group': 'focus', 'expert_member': 'e3', 'usage': 0.12, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'focused user', 'stage': 'micro', 'expert_group': 'focus', 'expert_member': 'e4', 'usage': 0.07, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'macro', 'expert_group': 'tempo', 'expert_member': 'e1', 'usage': 0.06, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'macro', 'expert_group': 'tempo', 'expert_member': 'e2', 'usage': 0.12, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'macro', 'expert_group': 'tempo', 'expert_member': 'e3', 'usage': 0.18, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'macro', 'expert_group': 'tempo', 'expert_member': 'e4', 'usage': 0.08, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'mid', 'expert_group': 'tempo', 'expert_member': 'e1', 'usage': 0.05, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'mid', 'expert_group': 'tempo', 'expert_member': 'e2', 'usage': 0.11, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'mid', 'expert_group': 'tempo', 'expert_member': 'e3', 'usage': 0.16, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'mid', 'expert_group': 'tempo', 'expert_member': 'e4', 'usage': 0.05, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'micro', 'expert_group': 'tempo', 'expert_member': 'e1', 'usage': 0.04, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'micro', 'expert_group': 'tempo', 'expert_member': 'e2', 'usage': 0.09, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'micro', 'expert_group': 'tempo', 'expert_member': 'e3', 'usage': 0.14, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q2', 'panel': 'Q2_intragroup', 'case_name': 'fast-tempo user', 'stage': 'micro', 'expert_group': 'tempo', 'expert_member': 'e4', 'usage': 0.04, 'split': 'test', 'selection_rule': 'external_grouping', 'run_id': 'template_profile', 'source_path': 'experiments/run/fmoe_n4/diag_exports/intragroup_profile.csv', 'notes': 'Placeholder case profile until within-family expert weights are exported for representative sessions.', 'status': 'placeholder_requires_logging'},
        ],
    )

    write_csv(
        '03_stage_removal.csv',
        variant_metric_rows(
            [
                ('full', 'RouteRec', 0.118, 0.187),
                ('remove_macro', 'RouteRec', 0.115, 0.182),
                ('remove_mid', 'RouteRec', 0.114, 0.179),
                ('remove_micro', 'RouteRec', 0.113, 0.176),
            ],
            panel='Q3_stage_removal',
            category_key='ablation_group',
            source_path='writing/results/03_stage_structure/03a_stage_ablation.csv',
            notes='Draft-fill structural summary. Replace with confirm export.',
            status='draft_fill',
        ),
    )
    write_csv(
        '03_dense_vs_staged.csv',
        variant_metric_rows(
            [
                ('dense_ffn', 'RouteRec', 0.108, 0.167),
                ('best_single_stage', 'RouteRec', 0.113, 0.176),
                ('best_two_stage', 'RouteRec', 0.116, 0.182),
                ('three_stage', 'RouteRec', 0.118, 0.187),
            ],
            panel='Q3_dense_vs_staged',
            category_key='layout_variant',
            source_path='writing/results/03_stage_structure/03b_dense_vs_staged.csv',
            notes='Draft-fill stage-count summary. Replace with best-per-group export.',
            status='draft_fill',
        ),
    )
    write_csv(
        '03_wrapper_order.csv',
        variant_metric_rows(
            [
                ('final', 'RouteRec', 0.118, 0.187),
                ('macro_late', 'RouteRec', 0.115, 0.181),
                ('micro_early', 'RouteRec', 0.116, 0.183),
                ('mid_repeat', 'RouteRec', 0.114, 0.179),
            ],
            panel='Q3_wrapper_order',
            category_key='layout_variant',
            source_path='writing/results/03_stage_structure/03c_wrapper_order.csv',
            notes='Order-neighborhood comparison for main text. Wrapper sweep moves to appendix.',
            status='draft_fill',
        ),
    )

    write_csv(
        '04_cue_scores.csv',
        variant_metric_rows(
            [
                ('Beauty', 'full', 0.076, 0.128),
                ('Beauty', 'remove_category', 0.073, 0.123),
                ('Beauty', 'remove_time', 0.072, 0.122),
                ('Beauty', 'sequence_only_portable', 0.071, 0.119),
                ('Foursquare', 'full', 0.143, 0.246),
                ('Foursquare', 'remove_category', 0.139, 0.240),
                ('Foursquare', 'remove_time', 0.137, 0.236),
                ('Foursquare', 'sequence_only_portable', 0.132, 0.227),
                ('KuaiRec', 'full', 0.118, 0.184),
                ('KuaiRec', 'remove_category', 0.117, 0.182),
                ('KuaiRec', 'remove_time', 0.116, 0.180),
                ('KuaiRec', 'sequence_only_portable', 0.113, 0.176),
                ('Retail Rocket', 'full', 0.249, 0.558),
                ('Retail Rocket', 'remove_category', 0.243, 0.551),
                ('Retail Rocket', 'remove_time', 0.240, 0.545),
                ('Retail Rocket', 'sequence_only_portable', 0.235, 0.537),
            ],
            panel='Q4_cue_scores',
            category_key='dataset',
            source_path='writing/results/04_cue_ablation/04a_cue_ablation.csv',
            notes='Draft-fill cue ablation summary. Replace with confirm export.',
            status='draft_fill',
        ),
    )
    write_csv(
        '04_cue_family_profile.csv',
        [
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'full', 'expert_family': 'memory', 'usage': 0.34, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'full', 'expert_family': 'focus', 'usage': 0.27, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'full', 'expert_family': 'tempo', 'usage': 0.16, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'full', 'expert_family': 'exposure', 'usage': 0.23, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'sequence_only_portable', 'expert_family': 'memory', 'usage': 0.31, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'sequence_only_portable', 'expert_family': 'focus', 'usage': 0.24, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'sequence_only_portable', 'expert_family': 'tempo', 'usage': 0.18, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Beauty', 'cue_setting': 'sequence_only_portable', 'expert_family': 'exposure', 'usage': 0.27, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'full', 'expert_family': 'memory', 'usage': 0.22, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'full', 'expert_family': 'focus', 'usage': 0.18, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'full', 'expert_family': 'tempo', 'usage': 0.14, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'full', 'expert_family': 'exposure', 'usage': 0.46, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'sequence_only_portable', 'expert_family': 'memory', 'usage': 0.20, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'sequence_only_portable', 'expert_family': 'focus', 'usage': 0.17, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'sequence_only_portable', 'expert_family': 'tempo', 'usage': 0.16, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q4', 'panel': 'Q4_family_profile', 'dataset': 'Retail Rocket', 'cue_setting': 'sequence_only_portable', 'expert_family': 'exposure', 'usage': 0.47, 'source_path': 'experiments/run/fmoe_n4/diag_exports/cue_family_profile.csv', 'notes': 'Placeholder family profile until per-setting family mass is exported.', 'status': 'placeholder_requires_logging'},
        ],
    )

    write_csv(
        '05_feature_intervention_scores.csv',
        variant_metric_rows(
            [
                ('full', 'RouteRec', 0.120, 0.189),
                ('shuffle_all', 'RouteRec', 0.109, 0.175),
                ('repeat_flatten', 'RouteRec', 0.112, 0.179),
                ('switch_boost', 'RouteRec', 0.114, 0.181),
                ('tempo_compress', 'RouteRec', 0.116, 0.183),
                ('popularity_spike', 'RouteRec', 0.117, 0.185),
            ],
            panel='Q5_feature_intervention',
            category_key='intervention',
            source_path='experiments/run/fmoe_n4/eval_exports/feature_intervention_scores.csv',
            notes='Placeholder until eval-time semantic intervention export is added.',
            status='placeholder_requires_logging',
        ),
    )
    write_csv(
        '05_feature_intervention_shift.csv',
        [
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'shuffle_all', 'expert_group': 'memory', 'delta_mass': -0.08, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'shuffle_all', 'expert_group': 'focus', 'delta_mass': -0.03, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'shuffle_all', 'expert_group': 'tempo', 'delta_mass': -0.06, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'shuffle_all', 'expert_group': 'exposure', 'delta_mass': -0.04, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'repeat_flatten', 'expert_group': 'memory', 'delta_mass': -0.19, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'repeat_flatten', 'expert_group': 'focus', 'delta_mass': 0.04, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'repeat_flatten', 'expert_group': 'tempo', 'delta_mass': 0.03, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'repeat_flatten', 'expert_group': 'exposure', 'delta_mass': 0.06, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'switch_boost', 'expert_group': 'memory', 'delta_mass': -0.04, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'switch_boost', 'expert_group': 'focus', 'delta_mass': 0.18, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'switch_boost', 'expert_group': 'tempo', 'delta_mass': 0.01, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'switch_boost', 'expert_group': 'exposure', 'delta_mass': 0.02, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'tempo_compress', 'expert_group': 'memory', 'delta_mass': -0.02, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'tempo_compress', 'expert_group': 'focus', 'delta_mass': 0.01, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'tempo_compress', 'expert_group': 'tempo', 'delta_mass': 0.17, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'tempo_compress', 'expert_group': 'exposure', 'delta_mass': 0.02, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'popularity_spike', 'expert_group': 'memory', 'delta_mass': 0.01, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'popularity_spike', 'expert_group': 'focus', 'delta_mass': 0.02, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'popularity_spike', 'expert_group': 'tempo', 'delta_mass': 0.01, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_feature_shift', 'intervention': 'popularity_spike', 'expert_group': 'exposure', 'delta_mass': 0.21, 'source_path': 'experiments/run/fmoe_n4/eval_exports/feature_intervention_shift.csv', 'notes': 'Placeholder until counterfactual family-mass deltas are exported.', 'status': 'placeholder_requires_logging'},
        ],
    )
    write_csv(
        '05_behavior_slice_quality.csv',
        variant_metric_rows(
            [
                ('repeat-heavy', 'best_baseline', 0.112, 0.181),
                ('repeat-heavy', 'shared_ffn', 0.114, 0.183),
                ('repeat-heavy', 'RouteRec', 0.120, 0.189),
                ('fast-tempo', 'best_baseline', 0.107, 0.171),
                ('fast-tempo', 'shared_ffn', 0.110, 0.176),
                ('fast-tempo', 'RouteRec', 0.119, 0.187),
                ('focused', 'best_baseline', 0.109, 0.174),
                ('focused', 'shared_ffn', 0.111, 0.178),
                ('focused', 'RouteRec', 0.117, 0.184),
                ('exploration-heavy', 'best_baseline', 0.101, 0.164),
                ('exploration-heavy', 'shared_ffn', 0.104, 0.169),
                ('exploration-heavy', 'RouteRec', 0.114, 0.181),
            ],
            panel='Q5_slice_quality',
            category_key='slice_name',
            source_path='experiments/run/fmoe_n4/diag_exports/behavior_slice_quality.csv',
            notes='Appendix slice summary placeholder.',
            status='placeholder_requires_logging',
        ),
    )
    write_csv(
        '05_behavior_slice_gain.csv',
        [
            {'paper_section': 'Q5', 'panel': 'Q5_slice_gain', 'slice_name': 'repeat-heavy', 'relative_gain': 0.018, 'route_concentration': 0.590, 'source_path': 'experiments/run/fmoe_n4/diag_exports/behavior_slice_gain.csv', 'notes': 'Appendix slice summary placeholder.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_slice_gain', 'slice_name': 'fast-tempo', 'relative_gain': 0.029, 'route_concentration': 0.660, 'source_path': 'experiments/run/fmoe_n4/diag_exports/behavior_slice_gain.csv', 'notes': 'Appendix slice summary placeholder.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_slice_gain', 'slice_name': 'focused', 'relative_gain': 0.021, 'route_concentration': 0.610, 'source_path': 'experiments/run/fmoe_n4/diag_exports/behavior_slice_gain.csv', 'notes': 'Appendix slice summary placeholder.', 'status': 'placeholder_requires_logging'},
            {'paper_section': 'Q5', 'panel': 'Q5_slice_gain', 'slice_name': 'exploration-heavy', 'relative_gain': 0.034, 'route_concentration': 0.710, 'source_path': 'experiments/run/fmoe_n4/diag_exports/behavior_slice_gain.csv', 'notes': 'Appendix slice summary placeholder.', 'status': 'placeholder_requires_logging'},
        ],
    )

    write_csv('A01_main_configuration.csv', [{'Aspect': 'Backbone layout', 'Main paper configuration': 'self-attention -> macro -> mid -> self-attention -> micro'}, {'Aspect': 'Stage execution', 'Main paper configuration': 'Serial'}, {'Aspect': 'Routing granularity', 'Main paper configuration': 'Macro/mid sequence-wise, micro position-wise'}, {'Aspect': 'Router composition', 'Main paper configuration': 'Shared scalar-group conditional router'}, {'Aspect': 'Expert mixing', 'Main paper configuration': 'Dense softmax mixture'}, {'Aspect': 'Macro history window', 'Main paper configuration': 'H=5 recent sessions'}, {'Aspect': 'Core auxiliary terms', 'Main paper configuration': 'Route consistency + z-loss'}])
    write_csv('A02_implementation_spec.csv', [{'Item': 'Semantic groups', 'Reported setting': 'G=4 (Tempo, Focus, Memory, Exposure)'}, {'Item': 'Experts per group', 'Reported setting': 'C in {2,3,4}; total E = 4C'}, {'Item': 'Cue dimensionality', 'Reported setting': '16 scalars per stage'}, {'Item': 'Router hidden width', 'Reported setting': '48-128'}, {'Item': 'Routing temperature', 'Reported setting': 'macro 1.0; mid/micro 1.2'}, {'Item': 'Consistency neighbors', 'Reported setting': 'k_nn = 1'}, {'Item': 'Missing-signal handling', 'Reported setting': 'broadcast if available, else zero fill'}])
    write_csv('A03_feature_families.csv', [{'Family': 'Tempo', 'Macro': 'Context-valid ratio; inter-session gap; pace trend', 'Mid': 'Prefix-valid ratio; mean/std interval; session age', 'Micro': 'Last gap; short-window pace; pace deviation'}, {'Family': 'Focus', 'Macro': 'Theme entropy; dominant mass; shift rate', 'Mid': 'Category entropy; top-1 mass; switch rate', 'Micro': 'Immediate switch; suffix entropy; suffix uniqueness'}, {'Family': 'Memory', 'Macro': 'Repeat intensity; overlap; repeat trend', 'Mid': 'Item uniqueness; repeat rate; novelty rate', 'Micro': 'Reconsumption flag; suffix repeat rate; longest run'}, {'Family': 'Exposure', 'Macro': 'Mean popularity; spread; entropy; trend', 'Mid': 'Popularity mean/std; entropy; drift', 'Micro': 'Last-item popularity; suffix spread; entropy'}])
    write_csv('A04_datasets_table.csv', [{'Dataset': 'Beauty', 'Domain': 'E-com.', 'Interactions': 33488, 'Sessions': 4243, 'Items': 3625, 'Avg_len': 7.9}, {'Dataset': 'Foursquare', 'Domain': 'POI', 'Interactions': 145238, 'Sessions': 25369, 'Items': 30588, 'Avg_len': 5.7}, {'Dataset': 'KuaiRec†', 'Domain': 'Video', 'Interactions': 287411, 'Sessions': 24458, 'Items': 6477, 'Avg_len': 11.8}, {'Dataset': 'LastFM†', 'Domain': 'Music', 'Interactions': 470408, 'Sessions': 25089, 'Items': 52510, 'Avg_len': 18.8}, {'Dataset': 'ML-1M', 'Domain': 'Movie', 'Interactions': 575281, 'Sessions': 14539, 'Items': 3533, 'Avg_len': 39.6}, {'Dataset': 'Retail Rocket', 'Domain': 'Browse', 'Interactions': 821243, 'Sessions': 153092, 'Items': 90211, 'Avg_len': 5.4}])

    full_rows = []
    seed_scores = {'Beauty': 0.082, 'Retail Rocket': 0.374, 'Foursquare': 0.171, 'ML-1M': 0.059, 'LastFM': 0.310, 'KuaiRec': 0.095}
    for dataset, route_mrr in seed_scores.items():
        for metric, mult in [('HR@5', 1.28), ('HR@10', 1.55), ('HR@20', 1.88), ('NDCG@5', 0.98), ('NDCG@10', 1.19), ('NDCG@20', 1.31), ('MRR@5', 0.88), ('MRR@10', 0.95), ('MRR@20', 1.0)]:
            route_score = round(route_mrr * mult, 3)
            full_rows.append({'dataset': dataset, 'metric': metric, 'SASRec': round(route_score * 0.92, 3), 'GRU4Rec': round(route_score * 0.72, 3), 'TiSASRec': round(route_score * 0.94, 3), 'FEARec': round(route_score * 0.89, 3), 'DuoRec': round(route_score * 0.91, 3), 'BSARec': round(route_score * 0.86, 3), 'FAME': round(route_score * 0.84, 3), 'DIF-SR': round(route_score * 0.87, 3), 'FDSA': round(route_score * 0.90, 3), 'RouteRec': route_score})
    full_rows.append({'dataset': 'All', 'metric': 'AvgRank↓', 'SASRec': 3.420, 'GRU4Rec': 9.060, 'TiSASRec': 3.720, 'FEARec': 5.920, 'DuoRec': 5.690, 'BSARec': 6.560, 'FAME': 6.860, 'DIF-SR': 6.420, 'FDSA': 5.140, 'RouteRec': 2.220})
    write_csv('A05_full_results_table.csv', full_rows)
    write_csv('A06_semantic_variants.csv', [{'variant': 'full_semantic', 'series': 'RouteRec', 'ndcg20': 0.118, 'hr10': 0.187}, {'variant': 'reduced_family', 'series': 'RouteRec', 'ndcg20': 0.116, 'hr10': 0.183}, {'variant': 'shuffled_family', 'series': 'RouteRec', 'ndcg20': 0.114, 'hr10': 0.179}, {'variant': 'flat_random', 'series': 'RouteRec', 'ndcg20': 0.112, 'hr10': 0.175}])
    write_csv('A06_scope_layout_variants.csv', [{'variant': 'original_scope', 'series': 'RouteRec', 'ndcg20': 0.118, 'hr10': 0.187}, {'variant': 'identical_scope', 'series': 'RouteRec', 'ndcg20': 0.113, 'hr10': 0.177}, {'variant': 'scope_swap', 'series': 'RouteRec', 'ndcg20': 0.114, 'hr10': 0.179}, {'variant': 'extra_attn', 'series': 'RouteRec', 'ndcg20': 0.116, 'hr10': 0.183}])
    write_csv('A06_order_router_variants.csv', [{'variant': 'base_order', 'series': 'RouteRec', 'ndcg20': 0.118, 'hr10': 0.187}, {'variant': 'macro_micro_mid', 'series': 'RouteRec', 'ndcg20': 0.116, 'hr10': 0.182}, {'variant': 'mid_macro_micro', 'series': 'RouteRec', 'ndcg20': 0.117, 'hr10': 0.184}, {'variant': 'flat_router', 'series': 'RouteRec', 'ndcg20': 0.114, 'hr10': 0.179}])
    write_csv('A07_expert_usage.csv', [{'stage': 'macro', 'expert_label': 'E1', 'usage': 0.12}, {'stage': 'macro', 'expert_label': 'E2', 'usage': 0.14}, {'stage': 'macro', 'expert_label': 'E3', 'usage': 0.11}, {'stage': 'macro', 'expert_label': 'E4', 'usage': 0.13}, {'stage': 'mid', 'expert_label': 'E1', 'usage': 0.18}, {'stage': 'mid', 'expert_label': 'E2', 'usage': 0.24}, {'stage': 'mid', 'expert_label': 'E3', 'usage': 0.20}, {'stage': 'mid', 'expert_label': 'E4', 'usage': 0.19}, {'stage': 'micro', 'expert_label': 'E1', 'usage': 0.26}, {'stage': 'micro', 'expert_label': 'E2', 'usage': 0.31}, {'stage': 'micro', 'expert_label': 'E3', 'usage': 0.24}, {'stage': 'micro', 'expert_label': 'E4', 'usage': 0.19}])
    write_csv('A07_entropy_effective.csv', [{'stage': 'macro', 'entropy': 1.72, 'effective_experts': 3.40}, {'stage': 'mid', 'entropy': 1.41, 'effective_experts': 2.80}, {'stage': 'micro', 'entropy': 1.12, 'effective_experts': 2.20}])
    write_csv('A07_stage_consistency.csv', [{'similarity_bucket': '0.0-0.2', 'stage': 'macro', 'consistency': 0.45}, {'similarity_bucket': '0.2-0.4', 'stage': 'macro', 'consistency': 0.54}, {'similarity_bucket': '0.4-0.6', 'stage': 'macro', 'consistency': 0.63}, {'similarity_bucket': '0.6-0.8', 'stage': 'macro', 'consistency': 0.71}, {'similarity_bucket': '0.8-1.0', 'stage': 'macro', 'consistency': 0.77}, {'similarity_bucket': '0.0-0.2', 'stage': 'mid', 'consistency': 0.39}, {'similarity_bucket': '0.2-0.4', 'stage': 'mid', 'consistency': 0.47}, {'similarity_bucket': '0.4-0.6', 'stage': 'mid', 'consistency': 0.56}, {'similarity_bucket': '0.6-0.8', 'stage': 'mid', 'consistency': 0.64}, {'similarity_bucket': '0.8-1.0', 'stage': 'mid', 'consistency': 0.71}, {'similarity_bucket': '0.0-0.2', 'stage': 'micro', 'consistency': 0.28}, {'similarity_bucket': '0.2-0.4', 'stage': 'micro', 'consistency': 0.35}, {'similarity_bucket': '0.4-0.6', 'stage': 'micro', 'consistency': 0.44}, {'similarity_bucket': '0.6-0.8', 'stage': 'micro', 'consistency': 0.53}, {'similarity_bucket': '0.8-1.0', 'stage': 'micro', 'consistency': 0.61}])
    write_csv('A07_feature_bucket_patterns.csv', [{'feature_family': 'Tempo', 'bucket_label': 'high', 'usage': 0.30}, {'feature_family': 'Tempo', 'bucket_label': 'low', 'usage': 0.20}, {'feature_family': 'Tempo', 'bucket_label': 'mid', 'usage': 0.25}, {'feature_family': 'Memory', 'bucket_label': 'high', 'usage': 0.36}, {'feature_family': 'Memory', 'bucket_label': 'low', 'usage': 0.26}, {'feature_family': 'Memory', 'bucket_label': 'mid', 'usage': 0.31}, {'feature_family': 'Focus', 'bucket_label': 'high', 'usage': 0.33}, {'feature_family': 'Focus', 'bucket_label': 'low', 'usage': 0.23}, {'feature_family': 'Focus', 'bucket_label': 'mid', 'usage': 0.28}, {'feature_family': 'Exposure', 'bucket_label': 'high', 'usage': 0.39}, {'feature_family': 'Exposure', 'bucket_label': 'low', 'usage': 0.29}, {'feature_family': 'Exposure', 'bucket_label': 'mid', 'usage': 0.34}])
    write_csv('A08_behavior_slice_quality.csv', pd.read_csv(DATA_DIR / '05_behavior_slice_quality.csv').to_dict('records'))
    write_csv('A08_behavior_slice_gain.csv', pd.read_csv(DATA_DIR / '05_behavior_slice_gain.csv').to_dict('records'))
    write_csv('A09_low_resource_transfer.csv', [{'shot_ratio': '5%', 'variant': 'full', 'ndcg20': 0.071}, {'shot_ratio': '10%', 'variant': 'full', 'ndcg20': 0.083}, {'shot_ratio': '20%', 'variant': 'full', 'ndcg20': 0.096}, {'shot_ratio': '5%', 'variant': 'sequence_only', 'ndcg20': 0.066}, {'shot_ratio': '10%', 'variant': 'sequence_only', 'ndcg20': 0.078}, {'shot_ratio': '20%', 'variant': 'sequence_only', 'ndcg20': 0.089}, {'shot_ratio': '5%', 'variant': 'shared_ffn', 'ndcg20': 0.061}, {'shot_ratio': '10%', 'variant': 'shared_ffn', 'ndcg20': 0.073}, {'shot_ratio': '20%', 'variant': 'shared_ffn', 'ndcg20': 0.084}])
    write_csv('A09_transfer_variants.csv', [{'variant': 'finetune_all', 'series': 'RouteRec', 'ndcg20': 0.101, 'hr10': 0.181}, {'variant': 'freeze_router', 'series': 'RouteRec', 'ndcg20': 0.096, 'hr10': 0.173}, {'variant': 'group_router', 'series': 'RouteRec', 'ndcg20': 0.098, 'hr10': 0.176}, {'variant': 'anchor_init', 'series': 'RouteRec', 'ndcg20': 0.100, 'hr10': 0.179}])
    write_csv('A09_multi_source_order.csv', [{'variant': 'Beauty→Retail', 'series': 'RouteRec', 'ndcg20': 0.099, 'hr10': 0.177}, {'variant': 'Retail→Beauty', 'series': 'RouteRec', 'ndcg20': 0.096, 'hr10': 0.171}, {'variant': 'Beauty→Foursquare', 'series': 'RouteRec', 'ndcg20': 0.101, 'hr10': 0.180}, {'variant': 'Foursquare→Beauty', 'series': 'RouteRec', 'ndcg20': 0.097, 'hr10': 0.174}])


def write_all_notebooks() -> None:
    write_notebook('01_main_results_table.ipynb', build_main_results_cells())
    write_notebook('02_q2_routing_control.ipynb', build_routing_cells())
    write_notebook('03_q3_stage_structure.ipynb', build_stage_cells())
    write_notebook('04_q4_lightweight_cues.ipynb', build_cue_cells())
    write_notebook('05_q5_behavior_regimes.ipynb', build_behavior_cells())
    write_notebook('A01_appendix_main_configuration.ipynb', build_plain_table_cells('A01 Appendix Main Configuration', 'A01_main_configuration.csv', 'Fixed main configuration used in the paper.', precision=3))
    write_notebook('A02_appendix_implementation_spec.ipynb', build_plain_table_cells('A02 Appendix Implementation Spec', 'A02_implementation_spec.csv', 'Implementation-level search controls used in the selected paper track.', precision=3))
    write_notebook('A03_appendix_feature_families.ipynb', build_plain_table_cells('A03 Appendix Feature Families', 'A03_feature_families.csv', 'Representative cue summaries used by RouteRec in the main configuration.', precision=3))
    write_notebook('A04_appendix_datasets.ipynb', build_plain_table_cells('A04 Appendix Datasets', 'A04_datasets_table.csv', 'Datasets used in the experiments.', precision=1))
    write_notebook('A05_appendix_full_results.ipynb', build_full_results_cells())
    write_notebook('A06_appendix_extended_structure.ipynb', build_extended_structure_cells())
    write_notebook('A07_appendix_routing_diagnostics.ipynb', build_diagnostics_cells())
    write_notebook('A08_appendix_behavior_slices.ipynb', build_appendix_behavior_cells())
    write_notebook('A09_appendix_transfer_variants.ipynb', build_transfer_cells())


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_all_csvs()
    write_all_notebooks()


if __name__ == '__main__':
    main()