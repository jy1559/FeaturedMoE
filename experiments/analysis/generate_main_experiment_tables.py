#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[2]
BASELINE_RESULTS = ROOT / "experiments" / "run" / "artifacts" / "results" / "baseline" / "normal"
FMOE_RESULTS = ROOT / "experiments" / "run" / "artifacts" / "results" / "fmoe_n3" / "normal" / "final_all_datasets"
OUT_DIR = ROOT / "writing" / "tables"

DATASETS = [
    ("amazon_beauty", "Amazon"),
    ("foursquare", "Foursquare"),
    ("KuaiRecLargeStrictPosV2_0.2", "KuaiRec"),
    ("lastfm0.03", "LastFM"),
    ("movielens1m", "ML1M"),
    ("retail_rocket", "Retail"),
]

BASELINE_MODELS = [
    ("GRU4Rec", "GRU4Rec"),
    ("SASRec", "SASRec"),
    ("TiSASRec", "TiSASRec"),
    ("DuoRec", "DuoRec"),
    ("SIGMA", "SIGMA"),
    ("BSARec", "BSARec"),
    ("FEARec", "FEARec"),
    ("DIFSR", "DIF-SR"),
    ("FAME", "FAME"),
]

METRICS = [
    ("hit@5", "HR@5"),
    ("hit@10", "HR@10"),
    ("hit@20", "HR@20"),
    ("ndcg@5", "NDCG@5"),
    ("ndcg@10", "NDCG@10"),
    ("ndcg@20", "NDCG@20"),
    ("mrr@5", "MRR@5"),
    ("mrr@10", "MRR@10"),
    ("mrr@20", "MRR@20"),
]


def _norm_model(raw: object) -> str:
    s = str(raw or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "gru4rec": "GRU4Rec",
        "sasrec": "SASRec",
        "tisasrec": "TiSASRec",
        "duorec": "DuoRec",
        "sigma": "SIGMA",
        "bsarec": "BSARec",
        "fearec": "FEARec",
        "difsr": "DIFSR",
        "fame": "FAME",
        "featuredmoen3": "FeaturedMoE_N3",
    }
    return aliases.get(s, str(raw or "").strip())


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metric_dict(section: object) -> Dict[str, float]:
    if not isinstance(section, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in section.items():
        k = str(key).strip().lower()
        try:
            out[k] = float(value)
        except Exception:
            continue
    return out


def _scope_from_path(path: Path) -> set[str]:
    name = path.name.lower()
    scopes = {"all"}
    if "_a10_" in name:
        scopes.add("A10")
    if "_a12_" in name:
        scopes.add("A12")
    return scopes


def _empty_record() -> dict:
    return {
        "test_best_test": None,
        "valid_best_valid": None,
        "valid_best_test": None,
    }


def _candidate_payload(path: Path) -> Optional[dict]:
    payload = _load_json(path)
    if not payload:
        return None
    model = _norm_model(payload.get("model"))
    dataset = str(payload.get("dataset") or "").strip()
    if not dataset:
        return None
    best_valid = _metric_dict(payload.get("best_valid_result"))
    test = _metric_dict(payload.get("test_result"))
    if not best_valid or not test:
        return None
    valid_sel = best_valid.get("mrr@20")
    test_sel = test.get("mrr@20")
    if valid_sel is None or test_sel is None:
        return None
    return {
        "path": str(path),
        "model": model,
        "dataset": dataset,
        "best_valid": best_valid,
        "test": test,
        "valid_sel": float(valid_sel),
        "test_sel": float(test_sel),
    }


def _update_best(dst: dict, key: str, score: float, payload: dict) -> None:
    prev = dst.get(key)
    if prev is None or float(prev["score"]) < float(score):
        dst[key] = {"score": float(score), "payload": payload}


def _collect() -> dict:
    data: dict = {
        "baseline": defaultdict(_empty_record),
        "fmoe": {scope: defaultdict(_empty_record) for scope in ["all", "A10", "A12"]},
    }

    for path in BASELINE_RESULTS.rglob("*.json"):
        payload = _candidate_payload(path)
        if not payload:
            continue
        dataset = payload["dataset"]
        model = payload["model"]
        if dataset not in {d for d, _ in DATASETS}:
            continue
        if model not in {m for m, _ in BASELINE_MODELS}:
            continue
        rec = data["baseline"][(dataset, model)]
        _update_best(rec, "test_best_test", payload["test_sel"], payload["test"])
        _update_best(rec, "valid_best_valid", payload["valid_sel"], payload["best_valid"])
        _update_best(rec, "valid_best_test", payload["valid_sel"], payload["test"])

    for path in FMOE_RESULTS.rglob("*.json"):
        payload = _candidate_payload(path)
        if not payload:
            continue
        dataset = payload["dataset"]
        model = payload["model"]
        if dataset not in {d for d, _ in DATASETS}:
            continue
        if model != "FeaturedMoE_N3":
            continue
        for scope in _scope_from_path(path):
            rec = data["fmoe"][scope][(dataset, model)]
            _update_best(rec, "test_best_test", payload["test_sel"], payload["test"])
            _update_best(rec, "valid_best_valid", payload["valid_sel"], payload["best_valid"])
            _update_best(rec, "valid_best_test", payload["valid_sel"], payload["test"])
    return data


def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    return f"{float(v):.4f}"


def _rank_values(values: Iterable[Optional[float]]) -> tuple[set[int], set[int]]:
    indexed = [(i, float(v)) for i, v in enumerate(values) if v is not None and math.isfinite(float(v))]
    if not indexed:
        return set(), set()
    uniq = sorted({v for _, v in indexed}, reverse=True)
    best_val = uniq[0]
    second_val = uniq[1] if len(uniq) > 1 else None
    best = {i for i, v in indexed if abs(v - best_val) <= 1e-12}
    second = set()
    if second_val is not None:
        second = {i for i, v in indexed if abs(v - second_val) <= 1e-12}
    return best, second


def _style_md(text: str, rank: str) -> str:
    if text == "--":
        return text
    if rank == "best":
        return f"**{text}**"
    if rank == "second":
        return f"<u>{text}</u>"
    return text


def _style_tex(text: str, rank: str) -> str:
    if text == "--":
        return text
    if rank == "best":
        return f"\\textbf{{{text}}}"
    if rank == "second":
        return f"\\underline{{{text}}}"
    return text


def _build_row_values(data: dict, dataset: str, metric_key: str, view_key: str, fmoe_scope: str) -> list[Optional[float]]:
    vals: list[Optional[float]] = []
    for model_key, _ in BASELINE_MODELS:
        rec = data["baseline"][(dataset, model_key)].get(view_key)
        vals.append(None if rec is None else rec["payload"].get(metric_key))
    fmoe_rec = data["fmoe"][fmoe_scope][(dataset, "FeaturedMoE_N3")].get(view_key)
    vals.append(None if fmoe_rec is None else fmoe_rec["payload"].get(metric_key))
    return vals


def _table_title(view_key: str, fmoe_scope: str) -> str:
    view_map = {
        "test_best_test": "Oracle Test Best",
        "valid_best_valid": "Best Valid",
        "valid_best_test": "Test At Best Valid",
    }
    scope_map = {"all": "FMoE_N3 All", "A10": "FMoE_N3 A10", "A12": "FMoE_N3 A12"}
    return f"{view_map[view_key]} / {scope_map[fmoe_scope]}"


def _render_markdown(data: dict, view_key: str, fmoe_scope: str) -> str:
    headers = ["Dataset", "Metric"] + [label for _, label in BASELINE_MODELS] + ["FMoE_N3"]
    lines = [f"## {_table_title(view_key, fmoe_scope)}", "", "| " + " | ".join(headers) + " |", "|" + "|".join([" --- " for _ in headers]) + "|"]
    for dataset, dataset_label in DATASETS:
        first = True
        for metric_key, metric_label in METRICS:
            vals = _build_row_values(data, dataset, metric_key, view_key, fmoe_scope)
            best, second = _rank_values(vals)
            rendered = []
            for idx, v in enumerate(vals):
                rank = "best" if idx in best else "second" if idx in second else ""
                rendered.append(_style_md(_fmt(v), rank))
            lines.append("| " + " | ".join([
                dataset_label if first else "",
                metric_label,
                *rendered,
            ]) + " |")
            first = False
    lines.append("")
    return "\n".join(lines)


def _render_latex(data: dict, view_key: str, fmoe_scope: str) -> str:
    cols = "ll" + "r" * (len(BASELINE_MODELS) + 1)
    headers = ["Dataset", "Metric"] + [label for _, label in BASELINE_MODELS] + ["FMoE\\_N3"]
    out = [
        f"% {_table_title(view_key, fmoe_scope)}",
        "\\begin{tabular}{" + cols + "}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    for dataset, dataset_label in DATASETS:
        first = True
        for metric_key, metric_label in METRICS:
            vals = _build_row_values(data, dataset, metric_key, view_key, fmoe_scope)
            best, second = _rank_values(vals)
            rendered = []
            for idx, v in enumerate(vals):
                rank = "best" if idx in best else "second" if idx in second else ""
                rendered.append(_style_tex(_fmt(v), rank))
            out.append(" & ".join([
                dataset_label if first else "",
                metric_label,
                *rendered,
            ]) + " \\\\")
            first = False
        out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("")
    return "\n".join(out)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_run_summary(data: dict) -> str:
    lines = ["# Main Experiment Table Artifacts", ""]
    for scope in ["all", "A10", "A12"]:
        lines.append(f"## {scope}")
        for view in ["test_best_test", "valid_best_valid", "valid_best_test"]:
            lines.append(f"- {view}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    data = _collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scope in ["all", "A10", "A12"]:
        for view in ["test_best_test", "valid_best_valid", "valid_best_test"]:
            stem = f"main_exp_{scope.lower()}_{view}"
            _write_text(OUT_DIR / f"{stem}.md", _render_markdown(data, view, scope))
            _write_text(OUT_DIR / f"{stem}.tex", _render_latex(data, view, scope))
    _write_text(OUT_DIR / "main_exp_tables_README.md", _render_run_summary(data))
    print(f"[done] wrote tables to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
