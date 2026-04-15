from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def load_csv_or_demo(
    csv_path: str | Path,
    required_columns: list[str],
    demo_builder: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, str]:
    csv_path = Path(csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"{csv_path.name} is missing required columns: {missing}")
        if not df.empty:
            return df, "csv"

    if demo_builder is None:
        return pd.DataFrame(columns=required_columns), "empty"

    demo_df = demo_builder()
    missing = [column for column in required_columns if column not in demo_df.columns]
    if missing:
        raise ValueError(f"Demo data is missing required columns: {missing}")
    return demo_df, "demo"


def export_figure(
    fig,
    output_stem: str,
    results_root: str | Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    results_root = Path(results_root)
    output_dir = results_root / "generated_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for extension in formats:
        output_path = output_dir / f"{output_stem}.{extension}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)
    return saved_paths