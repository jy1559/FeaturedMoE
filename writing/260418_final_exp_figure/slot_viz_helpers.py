from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLOR_MAP = {
    "RouteRec": "#0F766E",
    "behavior_guided": "#0F766E",
    "shared_ffn": "#5B7C99",
    "hidden_only": "#F28E2B",
    "mixed": "#D36A6A",
    "mixed_hidden_behavior": "#D36A6A",
    "behavior_only": "#0F766E",
    "full": "#0F766E",
    "remove_category": "#5B7C99",
    "remove_time": "#DA8A35",
    "sequence_only": "#C96567",
    "sequence_only_portable": "#C96567",
    "best_baseline": "#5B7C99",
    "baseline": "#5B7C99",
    "final": "#0F766E",
    "remove_macro": "#4C78A8",
    "remove_mid": "#F58518",
    "remove_micro": "#E45756",
    "dense_ffn": "#5B7C99",
    "best_single_stage": "#72B7B2",
    "best_two_stage": "#F28E2B",
    "three_stage": "#0F766E",
    "macro_late": "#4C78A8",
    "micro_early": "#F58518",
    "mid_repeat": "#E45756",
    "order_swap": "#5B7C99",
    "flat_wrapper": "#DA8A35",
    "shuffle_all": "#4C78A8",
    "zero_all": "#E45756",
    "zero_memory": "#72B7B2",
    "zero_tempo": "#F2CF5B",
    "macro": "#0F766E",
    "mid": "#4C78A8",
    "micro": "#F58518",
    "retail_rocket": "#0F766E",
    "beauty": "#5B7C99",
    "kuairec": "#E45756",
}

FALLBACK_COLORS = [
    "#0F766E",
    "#5B7C99",
    "#F28E2B",
    "#D36A6A",
    "#8E6C8A",
    "#B279A2",
    "#4E9F3D",
    "#9D755D",
]

MARKERS = ["o", "s", "D", "^", "P", "X"]


def build_palette(labels: Iterable[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    fallback_index = 0
    for label in labels:
        if label in COLOR_MAP:
            palette[label] = COLOR_MAP[label]
        else:
            palette[label] = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
            fallback_index += 1
    return palette


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.08)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "axes.titlesize": 14.5,
            "axes.labelsize": 12,
            "axes.titlepad": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFB",
            "grid.color": "#D8D8D2",
            "grid.linewidth": 0.75,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "figure.dpi": 130,
        }
    )


def _resolve_ticklabels(values: list[str], label_map: dict[str, str] | list[str] | None) -> list[str]:
    if label_map is None:
        return values
    if isinstance(label_map, dict):
        return [label_map.get(value, value) for value in values]
    return label_map


def tight_limits(values: Iterable[float], padding_ratio: float = 0.12, min_span: float = 0.012) -> tuple[float, float]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return 0.0, 1.0
    lower = float(series.min())
    upper = float(series.max())
    span = max(upper - lower, min_span)
    padding = span * padding_ratio
    return lower - padding, upper + padding


def annotate_bars(ax: plt.Axes, fmt: str = "{:.3f}", fontsize: int = 9) -> None:
    for container in ax.containers:
        labels = []
        for patch in container:
            height = patch.get_height()
            labels.append("" if pd.isna(height) else fmt.format(height))
        ax.bar_label(container, labels=labels, padding=3, fontsize=fontsize)


def dual_metric_grouped_plot(
    data: pd.DataFrame,
    category_col: str,
    variant_col: str,
    bar_col: str,
    line_col: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    bar_label: str = "NDCG@20",
    line_label: str = "HR@10",
    category_order: list[str] | None = None,
    variant_order: list[str] | None = None,
    category_labels: dict[str, str] | list[str] | None = None,
    rotate: int = 20,
    line_on_secondary: bool = True,
    annotate_lines: bool = False,
    show_legend: bool | None = None,
    single_variant_category_colors: bool = True,
) -> tuple[plt.Axes, plt.Axes]:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    plot_df = data.copy()
    categories = category_order or plot_df[category_col].dropna().unique().tolist()
    variants = variant_order or plot_df[variant_col].dropna().unique().tolist()
    palette = build_palette(variants)
    category_palette = build_palette(categories)

    x = np.arange(len(categories), dtype=float)
    width = min(0.72 / max(len(variants), 1), 0.55)
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2.0) * width

    twin = ax.twinx() if line_on_secondary else ax

    for idx, variant in enumerate(variants):
        subset = plot_df[plot_df[variant_col] == variant].set_index(category_col)
        bar_values = [subset.loc[cat, bar_col] if cat in subset.index else np.nan for cat in categories]
        line_values = [subset.loc[cat, line_col] if cat in subset.index else np.nan for cat in categories]
        xpos = x + offsets[idx]
        if len(variants) == 1 and single_variant_category_colors:
            bar_color = [category_palette[cat] for cat in categories]
            line_color = "#243447"
        else:
            bar_color = palette[variant]
            line_color = palette[variant]
        ax.bar(
            xpos,
            bar_values,
            width=width * 0.92,
            color=bar_color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.8,
            label=variant,
            zorder=2,
        )
        twin.plot(
            xpos,
            line_values,
            color=line_color,
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=2.2,
            markersize=5.8,
            zorder=3,
            markerfacecolor="white" if len(variants) > 1 else None,
            markeredgewidth=1.2,
        )
        if annotate_lines:
            for x_point, y_point in zip(xpos, line_values):
                if pd.notna(y_point):
                    twin.annotate(f"{y_point:.3f}", (x_point, y_point), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5)

    ax.set_xticks(x)
    display_labels = _resolve_ticklabels(categories, category_labels)
    ha = "center" if rotate == 0 else "right"
    ax.set_xticklabels(display_labels, rotation=rotate, ha=ha)
    ax.margins(x=0.06)
    ax.set_ylabel(bar_label)
    twin.set_ylabel(line_label)
    bar_values_all = plot_df[bar_col].tolist()
    line_values_all = plot_df[line_col].tolist()
    ax.set_ylim(*tight_limits(bar_values_all, padding_ratio=0.18, min_span=0.01))
    if line_on_secondary:
        twin.set_ylim(*tight_limits(line_values_all, padding_ratio=0.14, min_span=0.01))
    if title:
        ax.set_title(title)
    should_show_legend = len(variants) > 1 if show_legend is None else show_legend
    if should_show_legend:
        legend = ax.legend(loc="upper left", ncol=1, title=variant_col)
        if legend is not None:
            legend._legend_box.align = "left"
    ax.grid(axis="y")
    twin.grid(False)
    return ax, twin


def single_metric_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    order: list[str] | None = None,
    hue_order: list[str] | None = None,
    rotate: int = 15,
    annotate: bool = True,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    palette = None
    if hue is not None:
        palette = build_palette(hue_order or data[hue].dropna().unique().tolist())
    sns.barplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, palette=palette, ax=ax)
    ax.set_ylim(*tight_limits(data[y].tolist()))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=rotate)
    if annotate:
        annotate_bars(ax)
    return ax


def category_bar_line_plot(
    data: pd.DataFrame,
    category_col: str,
    bar_col: str,
    line_col: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    line_label: str | None = None,
    order: list[str] | None = None,
    category_labels: dict[str, str] | list[str] | None = None,
    rotate: int = 15,
    bar_color: str = "#0F766E",
    line_color: str = "#D36A6A",
    annotate: bool = False,
) -> tuple[plt.Axes, plt.Axes]:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.4))

    categories = order or data[category_col].dropna().unique().tolist()
    indexed = data.set_index(category_col)
    bar_values = [indexed.loc[cat, bar_col] if cat in indexed.index else np.nan for cat in categories]
    line_values = [indexed.loc[cat, line_col] if cat in indexed.index else np.nan for cat in categories]
    x = np.arange(len(categories), dtype=float)

    ax.bar(
        x,
        bar_values,
        width=0.62,
        color=bar_color,
        alpha=0.74,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    twin = ax.twinx()
    twin.plot(
        x,
        line_values,
        color=line_color,
        marker="o",
        linewidth=2.1,
        markersize=5.8,
        zorder=3,
    )

    ax.set_xticks(x)
    display_labels = _resolve_ticklabels(categories, category_labels)
    ha = "center" if rotate == 0 else "right"
    ax.set_xticklabels(display_labels, rotation=rotate, ha=ha)
    ax.set_ylim(*tight_limits(bar_values, padding_ratio=0.16, min_span=0.01))
    twin.set_ylim(*tight_limits(line_values, padding_ratio=0.12, min_span=0.01))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    twin.set_ylabel(line_label or line_col)

    if annotate:
        for x_point, bar_value in zip(x, bar_values):
            if pd.notna(bar_value):
                ax.annotate(f"{bar_value:.3f}", (x_point, bar_value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5)
        for x_point, line_value in zip(x, line_values):
            if pd.notna(line_value):
                twin.annotate(f"{line_value:.3f}", (x_point, line_value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5)

    ax.grid(axis="y")
    twin.grid(False)
    return ax, twin


def line_panel(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    hue_order: list[str] | None = None,
    annotate: bool = False,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    palette = build_palette(hue_order or data[hue].dropna().unique().tolist())
    sns.lineplot(data=data, x=x, y=y, hue=hue, marker="o", linewidth=2.1, palette=palette, ax=ax)
    ax.set_ylim(*tight_limits(data[y].tolist(), padding_ratio=0.12, min_span=0.02))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if annotate:
        for _, row in data.iterrows():
            ax.annotate(f"{row[y]:.3f}", (row[x], row[y]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5)
    return ax


def scatter_panel(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    label_col: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    palette = build_palette(data[hue].dropna().unique().tolist())
    sns.scatterplot(data=data, x=x, y=y, hue=hue, s=140, palette=palette, ax=ax)
    ax.set_ylim(*tight_limits(data[y].tolist(), padding_ratio=0.08, min_span=0.012))
    ax.set_xlim(*tight_limits(data[x].tolist(), padding_ratio=0.08, min_span=0.01))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if label_col is not None:
        for _, row in data.iterrows():
            ax.annotate(str(row[label_col]), (row[x], row[y]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    return ax


def heatmap_panel(
    data: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str = "magma",
    fmt: str = ".2f",
    cbar: bool = True,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    pivot_df = data.pivot(index=index, columns=columns, values=values)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=fmt,
        linewidths=0.6,
        linecolor="#F5F4EF",
        cmap=cmap,
        cbar=cbar,
        cbar_kws={"shrink": 0.82, "pad": 0.02},
        ax=ax,
    )
    if title:
        ax.set_title(title)
    return ax


def style_ranked_table(
    data: pd.DataFrame,
    numeric_columns: list[str],
    lower_is_better: bool = False,
    caption: str | None = None,
) -> pd.io.formats.style.Styler:
    display_df = data.copy()

    def style_row(row: pd.Series) -> list[str]:
        styles = ["" for _ in row.index]
        values = pd.to_numeric(row[numeric_columns], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            return styles
        ranked = valid.sort_values(ascending=lower_is_better)
        best = ranked.index[0]
        second = ranked.index[1] if len(ranked) > 1 else None
        for index, column in enumerate(row.index):
            if column == best:
                styles[index] = "background-color: #d9efe8; font-weight: 700;"
            elif second is not None and column == second:
                styles[index] = "background-color: #eef3f8; font-weight: 600;"
        return styles

    styler = display_df.style
    if caption:
        styler = styler.set_caption(caption)
    styler = styler.format(precision=3)
    styler = styler.apply(style_row, axis=1)
    styler = styler.set_table_styles(
        [
            {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "13px"), ("font-weight", "600"), ("text-align", "left")]},
            {"selector": "th", "props": [("background-color", "#F3F4F6"), ("font-weight", "700"), ("border", "1px solid #D6D6D6")]},
            {"selector": "td", "props": [("border", "1px solid #E3E3E3"), ("padding", "6px 8px")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("font-size", "11px"), ("width", "100%")]}]
    )
    return styler


def style_plain_table(
    data: pd.DataFrame,
    caption: str | None = None,
    precision: int = 3,
) -> pd.io.formats.style.Styler:
    styler = data.style
    if caption:
        styler = styler.set_caption(caption)
    styler = styler.format(precision=precision)
    styler = styler.set_table_styles(
        [
            {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "13px"), ("font-weight", "600"), ("text-align", "left")]},
            {"selector": "th", "props": [("background-color", "#F3F4F6"), ("font-weight", "700"), ("border", "1px solid #D6D6D6")]},
            {"selector": "td", "props": [("border", "1px solid #E3E3E3"), ("padding", "6px 8px")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("font-size", "11px"), ("width", "100%")]}]
    )
    return styler