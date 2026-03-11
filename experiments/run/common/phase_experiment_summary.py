#!/usr/bin/env python3
"""Build compact phase-local summary files from result JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _phase_parts(run_phase: str) -> dict[str, str]:
    parts = [p for p in str(run_phase or "").split("_") if p]
    out = {"family": "", "combo": "", "anchor": "", "mode": "", "k": ""}
    combo_idx = -1
    anchor_idx = -1
    k_idx = -1
    for i, tok in enumerate(parts):
        if combo_idx < 0 and re.fullmatch(r"C\d+", tok):
            combo_idx = i
        elif combo_idx >= 0 and anchor_idx < 0 and re.fullmatch(r"A\d+", tok):
            anchor_idx = i
        elif anchor_idx >= 0 and re.fullmatch(r"k\d+", tok):
            k_idx = i
            break

    if combo_idx >= 0:
        out["family"] = "_".join(parts[:combo_idx])
        out["combo"] = parts[combo_idx]
    if anchor_idx >= 0:
        out["anchor"] = parts[anchor_idx]
    if k_idx >= 0:
        out["k"] = parts[k_idx][1:]
    if anchor_idx >= 0:
        mode_end = k_idx if k_idx >= 0 else len(parts)
        out["mode"] = "_".join(parts[anchor_idx + 1 : mode_end])
    return out


def _merged_params(data: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("context_fixed", "fixed_search", "best_params"):
        block = data.get(key)
        if isinstance(block, dict):
            merged.update(block)
    return merged


def _best_trial(data: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_mrr: float | None = None
    trials = data.get("trials")
    if not isinstance(trials, list):
        return best

    for trial in trials:
        if not isinstance(trial, dict):
            continue
        status = str(trial.get("status", "")).strip().lower()
        if status not in {"", "ok", "success"}:
            continue
        mrr = _to_float(trial.get("mrr@20"))
        if mrr is None:
            valid_result = trial.get("valid_result")
            if isinstance(valid_result, dict):
                mrr = _to_float(valid_result.get("mrr@20"))
        if mrr is None:
            continue
        if best_mrr is None or mrr > best_mrr:
            best_mrr = mrr
            best = trial
    return best


def _fmt_float(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _fmt_sci(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.2e}"


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_No rows yet._"]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


@dataclass
class SummaryRow:
    run_phase: str
    combo: str
    anchor: str
    mode: str
    expert_top_k: str
    dims: str
    layout: str
    merge: str
    group_mode: str
    best_mrr20: float
    best_lr: float | None
    best_wd: float | None
    epochs_run: int | None
    early_stopped: bool
    n_completed: int
    max_evals: int
    track_note: str

    def csv_row(self) -> dict[str, Any]:
        return {
            "run_phase": self.run_phase,
            "combo": self.combo,
            "anchor": self.anchor,
            "mode": self.mode,
            "expert_top_k": self.expert_top_k,
            "dims": self.dims,
            "layout": self.layout,
            "merge": self.merge,
            "group_mode": self.group_mode,
            "best_mrr20": f"{self.best_mrr20:.6f}",
            "best_lr": _fmt_sci(self.best_lr),
            "best_wd": _fmt_sci(self.best_wd),
            "epochs_run": "" if self.epochs_run is None else str(self.epochs_run),
            "early_stopped": "yes" if self.early_stopped else "no",
            "n_completed": str(self.n_completed),
            "max_evals": str(self.max_evals),
            "track_note": self.track_note,
        }


def _collect_rows(results_dir: Path, dataset: str, phase_bucket: str) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for path in sorted(results_dir.glob("*.json")):
        data = _load_json(path)
        if not data:
            continue
        if str(data.get("dataset", "")).strip() != dataset:
            continue
        run_phase = str(data.get("run_phase", "")).strip()
        if not run_phase.startswith(phase_bucket):
            continue

        best_mrr = _to_float(data.get("best_mrr@20"))
        best_trial = _best_trial(data)
        trial_mrr = _to_float(best_trial.get("mrr@20")) if best_trial else None
        if trial_mrr is None and isinstance(best_trial.get("valid_result"), dict):
            trial_mrr = _to_float(best_trial["valid_result"].get("mrr@20"))
        if trial_mrr is not None:
            best_mrr = trial_mrr
        if best_mrr is None:
            continue

        merged = _merged_params(data)
        parts = _phase_parts(run_phase)
        dims = (
            f"{merged.get('embedding_size', '-')}/"
            f"{merged.get('d_feat_emb', '-')}/"
            f"{merged.get('d_expert_hidden', '-')}/"
            f"{merged.get('d_router_hidden', '-')}"
        )
        track_note = f"L{merged.get('arch_layout_id', '-')} {merged.get('stage_merge_mode', '-')} {merged.get('group_router_mode', '-')}"
        rows.append(
            SummaryRow(
                run_phase=run_phase,
                combo=parts["combo"] or "-",
                anchor=parts["anchor"] or "-",
                mode=parts["mode"] or str(merged.get("inner_rule_mode", "-")),
                expert_top_k=parts["k"] or str(merged.get("expert_top_k", "-")),
                dims=dims,
                layout=str(merged.get("arch_layout_id", "-")),
                merge=str(merged.get("stage_merge_mode", "-")),
                group_mode=str(merged.get("group_router_mode", "-")),
                best_mrr20=best_mrr,
                best_lr=_to_float((best_trial.get("params") or {}).get("learning_rate")) if best_trial else None,
                best_wd=_to_float((best_trial.get("params") or {}).get("weight_decay")) if best_trial else None,
                epochs_run=int(best_trial.get("epochs_run")) if best_trial and best_trial.get("epochs_run") is not None else None,
                early_stopped=bool(best_trial.get("early_stopped")) if best_trial else False,
                n_completed=int(_to_float(data.get("n_completed")) or 0),
                max_evals=int(_to_float(data.get("max_evals")) or 0),
                track_note=track_note,
            )
        )
    rows.sort(key=lambda row: row.best_mrr20, reverse=True)
    return rows


def _aggregate(rows: list[SummaryRow], key_fn) -> list[dict[str, Any]]:
    buckets: dict[str, list[SummaryRow]] = {}
    for row in rows:
        key = key_fn(row)
        buckets.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        vals = [it.best_mrr20 for it in items]
        out.append(
            {
                "key": key,
                "runs": len(items),
                "best": max(vals),
                "avg": sum(vals) / len(vals),
            }
        )
    out.sort(key=lambda item: item["best"], reverse=True)
    return out


def _write_csv(rows: list[SummaryRow], output_csv: Path) -> None:
    fieldnames = [
        "run_phase",
        "combo",
        "anchor",
        "mode",
        "expert_top_k",
        "dims",
        "layout",
        "merge",
        "group_mode",
        "best_mrr20",
        "best_lr",
        "best_wd",
        "epochs_run",
        "early_stopped",
        "n_completed",
        "max_evals",
        "track_note",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.csv_row())


def _write_md(
    rows: list[SummaryRow],
    output_md: Path,
    title: str,
    notes: str,
    dataset: str,
    phase_bucket: str,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- Updated: `{_utc_now()}`")
    lines.append(f"- Dataset: `{dataset}`")
    lines.append(f"- Phase bucket: `{phase_bucket}`")
    lines.append(f"- Completed result files: `{len(rows)}`")
    if notes.strip():
        lines.append(f"- Focus: {notes.strip()}")
    if rows:
        best = rows[0]
        lines.append(
            f"- Current best: `{best.best_mrr20:.4f}` with `{best.anchor} / {best.mode} / k{best.expert_top_k} / {best.dims}`"
        )
    lines.append("")

    mode_rows = _aggregate(rows, lambda row: row.mode)
    k_rows = _aggregate(rows, lambda row: f"k{row.expert_top_k}")
    anchor_rows = _aggregate(rows, lambda row: row.anchor)

    lines.append("## By Mode")
    lines.extend(
        _md_table(
            ["mode", "runs", "best", "avg"],
            [
                [str(item["key"]), str(item["runs"]), _fmt_float(item["best"]), _fmt_float(item["avg"])]
                for item in mode_rows
            ],
        )
    )
    lines.append("")

    lines.append("## By Expert Top-K")
    lines.extend(
        _md_table(
            ["k", "runs", "best", "avg"],
            [
                [str(item["key"]), str(item["runs"]), _fmt_float(item["best"]), _fmt_float(item["avg"])]
                for item in k_rows
            ],
        )
    )
    lines.append("")

    lines.append("## By Anchor")
    lines.extend(
        _md_table(
            ["anchor", "runs", "best", "avg"],
            [
                [str(item["key"]), str(item["runs"]), _fmt_float(item["best"]), _fmt_float(item["avg"])]
                for item in anchor_rows
            ],
        )
    )
    lines.append("")

    lines.append("## Combo Table")
    lines.extend(
        _md_table(
            ["rank", "combo", "anchor", "mode", "k", "dims", "best", "lr", "wd", "epoch", "stop", "setup"],
            [
                [
                    str(idx),
                    row.combo,
                    row.anchor,
                    row.mode,
                    f"k{row.expert_top_k}",
                    row.dims,
                    _fmt_float(row.best_mrr20),
                    _fmt_sci(row.best_lr),
                    _fmt_sci(row.best_wd),
                    "-" if row.epochs_run is None else str(row.epochs_run),
                    "Y" if row.early_stopped else "N",
                    row.track_note,
                ]
                for idx, row in enumerate(rows[:20], start=1)
            ],
        )
    )
    lines.append("")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact phase summary from result JSONs.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-bucket", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    title = args.title.strip() or f"{args.phase_bucket} Summary"

    rows = _collect_rows(results_dir, args.dataset, args.phase_bucket)
    _write_csv(rows, output_csv)
    _write_md(rows, output_md, title=title, notes=args.notes, dataset=args.dataset, phase_bucket=args.phase_bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
