from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .paper_theme import build_palette


def resolve_bar_ylim(
    values: Iterable[float],
    include_zero: bool = False,
    padding_ratio: float = 0.12,
    min_span: float = 0.02,
) -> tuple[float, float]:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if numeric.empty:
        return (0.0, 1.0)

    lower = numeric.min()
    upper = numeric.max()
    if include_zero:
        lower = min(0.0, lower)
    span = max(upper - lower, min_span)
    padding = span * padding_ratio

    ymin = lower - padding
    ymax = upper + padding
    if include_zero:
        ymin = min(ymin, 0.0)
    return ymin, ymax


def annotate_bars(ax, fmt: str = "{:.3f}", fontsize: int = 8) -> None:
    for container in ax.containers:
        labels = []
        for patch in container:
            height = patch.get_height()
            if np.isnan(height):
                labels.append("")
            else:
                labels.append(fmt.format(height))
        ax.bar_label(container, labels=labels, padding=3, fontsize=fontsize)


def grouped_barplot(
    data: pd.DataFrame,
    x: str,
    hue: str,
    y: str,
    ax=None,
    order: list[str] | None = None,
    hue_order: list[str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    include_zero: bool = False,
    annotate: bool = True,
    rotate: int = 0,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    labels = hue_order or data[hue].dropna().unique().tolist()
    palette = build_palette(labels)
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )

    ax.set_ylim(*resolve_bar_ylim(data[y].tolist(), include_zero=include_zero))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if rotate:
        ax.tick_params(axis="x", rotation=rotate)
    if annotate:
        annotate_bars(ax)
    return ax


def lineplot_with_markers(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    ax=None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    marker: str = "o",
    annotate_points: bool = False,
    fmt: str = "{:.3f}",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    labels = data[hue].dropna().unique().tolist()
    palette = build_palette(labels)
    sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker, palette=palette, ax=ax)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if annotate_points:
        for _, row in data.iterrows():
            ax.annotate(fmt.format(row[y]), (row[x], row[y]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    return ax


def scatterplot_with_annotations(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    style: str | None = None,
    annotate_column: str | None = None,
    ax=None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    palette = None
    if hue is not None:
        palette = build_palette(data[hue].dropna().unique().tolist())
    sns.scatterplot(data=data, x=x, y=y, hue=hue, style=style, palette=palette, s=80, ax=ax)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if annotate_column is not None:
        for _, row in data.iterrows():
            ax.annotate(str(row[annotate_column]), (row[x], row[y]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    return ax


def heatmap_from_long(
    data: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    ax=None,
    title: str | None = None,
    cmap: str = "crest",
    fmt: str = ".3f",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    pivot_df = data.pivot(index=index, columns=columns, values=values)
    sns.heatmap(pivot_df, annot=True, fmt=fmt, cmap=cmap, linewidths=0.3, cbar=True, ax=ax)
    if title:
        ax.set_title(title)
    return ax