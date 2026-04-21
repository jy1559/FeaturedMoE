from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PALETTE = {
    "ink": "#20313B",
    "muted": "#66757D",
    "grid": "#D9D8D2",
    "paper": "#FCFCFA",
    "badge": "#F3EEE4",
    "route": "#0F766E",
    "blue": "#5B7C99",
    "orange": "#D28B36",
    "rose": "#C96567",
    "gold": "#D9B44A",
    "plum": "#8E6C8A",
}

DATASET_LABELS = {
    "beauty": "Beauty",
    "foursquare": "Foursquare",
    "KuaiRecLargeStrictPosV2_0.2": "KuaiRec",
    "lastfm0.03": "LastFM",
    "movielens1m": "ML-1M",
    "retail_rocket": "Retail Rocket",
}

FAMILY_COLORS = {
    "memory": PALETTE["blue"],
    "focus": PALETTE["orange"],
    "tempo": PALETTE["route"],
    "exposure": PALETTE["rose"],
}

SUBFIG_SIZE = (7.8, 5.8)
LEGEND_STRIP_SIZE = (15.6, 1.8)
LEGEND_STRIP_HALF_SIZE = (7.65, 1.8)
OURS_SUFFIX = "\n(Ours)"


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.12)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "axes.labelsize": 20.5,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 18.5,
            "ytick.labelsize": 18.5,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "legend.frameon": True,
            "legend.fontsize": 21.0,
            "legend.facecolor": "white",
            "legend.edgecolor": PALETTE["muted"],
            "figure.dpi": 130,
            "savefig.dpi": 220,
        }
    )


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing export: {path}")
    return pd.read_csv(path)


def preview_frame(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = list(df.columns)
    print(f"rows={len(df):,} cols={len(cols)}")
    print(cols)
    return df.head(n)


def dataset_label(raw: str) -> str:
    return DATASET_LABELS.get(raw, raw)


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> plt.Axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["muted"])
    ax.spines["bottom"].set_color(PALETTE["muted"])
    ax.grid(axis=grid_axis, color=PALETTE["grid"], alpha=0.7)
    return ax


def single_subfigure_axes(figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize or SUBFIG_SIZE)
    fig.subplots_adjust(left=0.18, right=0.835, bottom=0.19, top=0.91)
    clean_axes(ax)
    return fig, ax


def legend_strip_axes(figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize or LEGEND_STRIP_SIZE)
    ax.axis("off")
    return fig, ax


def half_legend_strip_axes(figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize or LEGEND_STRIP_HALF_SIZE)
    ax.axis("off")
    return fig, ax


def mark_ours_first(order: list[str]) -> list[str]:
    if not order:
        return order
    marked = list(order)
    marked[0] = f"{marked[0]}{OURS_SUFFIX}"
    return marked


def y_limits_like_q2(values: Iterable[float], lower_pad: float = 0.55, upper_pad: float = 0.18) -> tuple[float, float]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return (0.0, 1.0)
    lo = float(series.min())
    hi = float(series.max())
    span = max(hi - lo, 0.004)
    return lo - span * lower_pad, hi + span * upper_pad


def metric_legend_handles() -> list[object]:
    return [
        Patch(facecolor="#8FA6DE", edgecolor="#6178AE", alpha=0.92, label="NDCG@20"),
        Line2D(
            [0],
            [0],
            color="#C33245",
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            label="HR@10",
        ),
    ]


def add_metric_legend(ax: plt.Axes, loc: str = "lower right") -> None:
    ax.legend(
        handles=metric_legend_handles(),
        loc=loc,
        frameon=True,
        fancybox=False,
        borderpad=0.48,
        labelspacing=0.32,
        handlelength=2.0,
        handletextpad=0.62,
        prop={"size": plt.rcParams["legend.fontsize"] - 2},
    )


def add_category_legend(
    fig: plt.Figure,
    labels: list[str],
    colors: list[str],
    ncol: int | None = None,
    y: float = 1.02,
) -> None:
    handles = [Patch(facecolor=color, edgecolor="white", linewidth=0.8, label=label) for label, color in zip(labels, colors)]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(handles),
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
    )


def add_split_legends(
    ax: plt.Axes,
    category_labels: list[str],
    category_colors: list[str],
    metric_loc: str = "lower right",
) -> None:
    category_handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.8, label=label)
        for label, color in zip(category_labels, category_colors)
    ]
    legend_top = ax.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=max(2, min(len(category_handles), 3)),
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ax.add_artist(legend_top)
    add_metric_legend(ax, loc=metric_loc)


def add_legend_strip(
    ax: plt.Axes,
    category_labels: list[str],
    category_colors: list[str],
    ncol: int | None = None,
) -> None:
    category_handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.8, label=label)
        for label, color in zip(category_labels, category_colors)
    ]
    ax.legend(
        handles=category_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=ncol or len(category_handles),
        frameon=False,
        columnspacing=1.9,
        handletextpad=0.9,
        borderaxespad=0.0,
        prop={"size": plt.rcParams["legend.fontsize"]},
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    """Place a bold panel label inside the top-left corner with a white background
    so it remains readable regardless of what is drawn beneath it."""
    ax.text(
        0.02,
        0.97,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
        ),
        zorder=10,
    )


def annotate_demo(fig: plt.Figure, df: pd.DataFrame) -> None:
    if "data_status" not in df.columns or df.empty:
        return
    status = str(df["data_status"].iloc[0]).strip()
    if not status:
        return
    fig.text(
        0.012,
        0.988,
        f"data_status={status}",
        ha="left",
        va="top",
        fontsize=8.5,
        color=PALETTE["muted"],
        bbox={"boxstyle": "round,pad=0.22", "facecolor": PALETTE["badge"], "edgecolor": PALETTE["grid"]},
    )


def metric_limits(values: Iterable[float], padding: float = 0.12, floor: float | None = None) -> tuple[float, float]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return (0.0, 1.0)
    lo = float(series.min())
    hi = float(series.max())
    span = max(hi - lo, 0.01)
    pad = span * padding
    lower = lo - pad
    upper = hi + pad
    if floor is not None:
        lower = max(lower, floor)
    return lower, upper


def palette_for(labels: Iterable[str]) -> dict[str, str]:
    fallback = [PALETTE["route"], PALETTE["blue"], PALETTE["orange"], PALETTE["rose"], PALETTE["gold"], PALETTE["plum"]]
    mapping: dict[str, str] = {}
    for idx, label in enumerate(labels):
        label_text = str(label).lower()
        if label_text in FAMILY_COLORS:
            mapping[str(label)] = FAMILY_COLORS[label_text]
        elif "behavior" in label_text or "final" in label_text or "semantic" in label_text or "full" in label_text:
            mapping[str(label)] = PALETTE["route"]
        elif "shared" in label_text or "baseline" in label_text or "sasrec" in label_text:
            mapping[str(label)] = PALETTE["blue"]
        elif "hidden" in label_text or "single" in label_text or "consistency" in label_text:
            mapping[str(label)] = PALETTE["orange"]
        elif "mixed" in label_text or "shuffle" in label_text or "zero" in label_text:
            mapping[str(label)] = PALETTE["rose"]
        else:
            mapping[str(label)] = fallback[idx % len(fallback)]
    return mapping


def paper_axes(figsize: tuple[float, float] = (9.5, 4.2)) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    clean_axes(ax)
    return fig, ax


def _resolve_order(values: pd.Series, order_col: pd.Series | None = None) -> list[str]:
    if order_col is not None and len(values) == len(order_col):
        df = pd.DataFrame({"v": values.astype(str), "o": pd.to_numeric(order_col, errors="coerce")})
        df = df.sort_values("o", kind="stable")
        ordered = df["v"].drop_duplicates().tolist()
        if ordered:
            return ordered
    return values.astype(str).drop_duplicates().tolist()


def bar_line_panel(
    df: pd.DataFrame,
    category_col: str,
    ndcg_col: str,
    hr_col: str,
    ax: plt.Axes,
    order: list[str] | None = None,
    order_col: str | None = None,
    title: str | None = None,
    panel: str | None = None,
    xrotation: int = 20,
    palette_override: dict[str, str] | None = None,
    show_xticklabels: bool = True,
    add_metric_legend_box: bool = False,
) -> tuple[plt.Axes, plt.Axes]:
    work = df.copy()
    work = work.dropna(subset=[category_col])
    work[category_col] = work[category_col].astype(str)
    if order is None:
        order_values = _resolve_order(work[category_col], work[order_col] if order_col and order_col in work.columns else None)
    else:
        order_values = order

    palette = palette_override or palette_for(order_values)
    x = np.arange(len(order_values), dtype=float)
    ndcg_vals = []
    hr_vals = []
    for label in order_values:
        row = work[work[category_col] == label]
        ndcg_vals.append(float(pd.to_numeric(row[ndcg_col], errors="coerce").mean()) if not row.empty else np.nan)
        hr_vals.append(float(pd.to_numeric(row[hr_col], errors="coerce").mean()) if not row.empty else np.nan)

    ax.bar(
        x,
        ndcg_vals,
        width=0.62,
        color=[palette[label] for label in order_values],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    twin = ax.twinx()
    twin.plot(
        x,
        hr_vals,
        color="#C33245",
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        markeredgecolor=PALETTE["ink"],
        markeredgewidth=0.6,
        zorder=3,
    )

    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(order_values, rotation=xrotation, ha="right" if xrotation else "center")
    else:
        ax.set_xticklabels([""] * len(order_values))
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel("NDCG@20")
    twin.set_ylabel("HR@10")
    ax.set_ylim(*y_limits_like_q2(ndcg_vals))
    twin.set_ylim(*y_limits_like_q2(hr_vals))
    ax.margins(x=0.06)
    clean_axes(ax)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_color(PALETTE["muted"])
    if title:
        ax.set_title(title)
    if panel:
        panel_label(ax, panel)
    if add_metric_legend_box:
        add_metric_legend(ax, loc="lower right")
    return ax, twin
