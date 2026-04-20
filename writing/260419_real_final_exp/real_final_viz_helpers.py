from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
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
    "movielens1m": "ML-1M",
}

FAMILY_COLORS = {
    "memory": PALETTE["blue"],
    "focus": PALETTE["orange"],
    "tempo": PALETTE["route"],
    "exposure": PALETTE["rose"],
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.04)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.2,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
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
) -> tuple[plt.Axes, plt.Axes]:
    work = df.copy()
    work = work.dropna(subset=[category_col])
    work[category_col] = work[category_col].astype(str)
    if order is None:
        order_values = _resolve_order(work[category_col], work[order_col] if order_col and order_col in work.columns else None)
    else:
        order_values = order

    palette = palette_for(order_values)
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
        color=PALETTE["ink"],
        marker="o",
        linewidth=2.1,
        markersize=5.4,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(order_values, rotation=xrotation, ha="right" if xrotation else "center")
    ax.set_ylabel("NDCG@20")
    twin.set_ylabel("HR@10")
    ax.set_ylim(*metric_limits(ndcg_vals, padding=0.18, floor=0.0))
    twin.set_ylim(*metric_limits(hr_vals, padding=0.14, floor=0.0))
    ax.margins(x=0.06)
    clean_axes(ax)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_color(PALETTE["muted"])
    if title:
        ax.set_title(title)
    if panel:
        panel_label(ax, panel)
    return ax, twin
