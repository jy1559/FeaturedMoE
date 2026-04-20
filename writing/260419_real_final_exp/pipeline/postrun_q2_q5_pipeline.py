#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
WRITING_ROOT = REPO_ROOT / "writing" / "260419_real_final_exp"
PIPELINE_ROOT = WRITING_ROOT / "pipeline"
DATA_ROOT = WRITING_ROOT / "data"
NOTEBOOK_ROOT = WRITING_ROOT
TEX_ROOT = REPO_ROOT / "writing" / "ACM_template"
FIGURE_ROOT = TEX_ROOT / "figures" / "appendix"
REAL_FINAL_ROOT = REPO_ROOT / "experiments" / "run" / "final_experiment" / "real_final_ablation"
LOG_ROOT = REPO_ROOT / "experiments" / "run" / "artifacts" / "logs" / "real_final_ablation"

DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "retail_rocket",
    "lastfm0.03",
    "foursquare",
]

Q2_VARIANT_MAP = {
    "shared_ffn": ("Shared FFN", 1),
    "hidden_only": ("Hidden only", 2),
    "feature_fusion_bias": ("Fusion bias", 3),
    "mixed_hidden_behavior": ("Mixed", 4),
    "behavior_guided": ("Behavior-guided", 5),
}

EXPECTED_DATA_FILES = [
    DATA_ROOT / "q2_quality.csv",
    DATA_ROOT / "q2_routing_profile.csv",
    DATA_ROOT / "q3_temporal_decomp.csv",
    DATA_ROOT / "q3_routing_org.csv",
    DATA_ROOT / "q5_case_heatmap.csv",
    DATA_ROOT / "q5_intervention_summary.csv",
    DATA_ROOT / "q_suite_manifest.json",
    DATA_ROOT / "q_suite_run_index.csv",
]

EXPECTED_FIGURES = [
    FIGURE_ROOT / "a03_objective_variants_a.pdf",
    FIGURE_ROOT / "a03_objective_variants_b.pdf",
    FIGURE_ROOT / "a02_structural_temporal.pdf",
    FIGURE_ROOT / "a02_structural_cue_org.pdf",
    FIGURE_ROOT / "a05_routing_profiles_a.pdf",
    FIGURE_ROOT / "a05_routing_profiles_b.pdf",
    FIGURE_ROOT / "a05_routing_profiles_c.pdf",
    FIGURE_ROOT / "a05_routing_profiles.pdf",
]

NOTEBOOKS = [
    NOTEBOOK_ROOT / "02_q2_routing_control.ipynb",
    NOTEBOOK_ROOT / "03_q3_design_justification.ipynb",
    NOTEBOOK_ROOT / "05_q5_behavior_semantics.ipynb",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_csv_arg(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[run]", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def assert_required_logs(required_questions: list[str]) -> None:
    missing: list[Path] = []
    for question in required_questions:
        summary = LOG_ROOT / question / "summary.csv"
        if not summary.exists():
            missing.append(summary)
            continue
        ok_rows = [row for row in read_csv_rows(summary) if str(row.get("status", "")).lower() == "ok"]
        if not ok_rows:
            raise RuntimeError(f"No status=ok rows found in {summary}")
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Missing required summary files:\n{joined}")


def export_bundle(python_bin: str, datasets: list[str]) -> None:
    cmd = [
        python_bin,
        str(REAL_FINAL_ROOT / "export_q2_q5_bundle.py"),
        "--output-dir",
        str(DATA_ROOT),
    ]
    if datasets:
        cmd.extend(["--datasets", ",".join(datasets)])
    run_checked(cmd, cwd=REPO_ROOT)


def rebuild_q2_routing_profile(datasets: list[str]) -> Path:
    summary_path = LOG_ROOT / "q2" / "summary.csv"
    rows = read_csv_rows(summary_path)
    dataset_filter = set(datasets)
    out: list[dict[str, Any]] = []

    for row in rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        dataset = str(row.get("dataset", ""))
        if dataset_filter and dataset not in dataset_filter:
            continue

        result_path = Path(str(row.get("result_path", "")).strip())
        if not result_path.exists():
            continue
        payload = load_json(result_path)
        diag_path_raw = str(payload.get("diag_raw_test_json", "")).strip()
        if not diag_path_raw:
            continue
        diag_path = Path(diag_path_raw)
        if not diag_path.exists():
            continue
        diag_payload = load_json(diag_path)
        stage_metrics = diag_payload.get("stage_metrics") or {}

        setting_key = str(row.get("setting_key", "")).strip()
        variant_label = str(row.get("variant_label", "")).strip()
        if not variant_label:
            variant_label = Q2_VARIANT_MAP.get(setting_key, (setting_key, 0))[0]

        for stage_name, stage_data in stage_metrics.items():
            family_names = stage_data.get("family_names") or []
            usage_sum = stage_data.get("usage_sum") or []
            if not family_names or not usage_sum:
                continue
            total = sum(safe_float(value) for value in usage_sum) or 1.0
            for family_name, usage in zip(family_names, usage_sum):
                out.append(
                    {
                        "dataset": dataset,
                        "setting_key": setting_key,
                        "variant_label": variant_label,
                        "stage_name": stage_name,
                        "routed_family": str(family_name).lower(),
                        "usage_share": round(safe_float(usage) / total, 4),
                        "base_rank": row.get("base_rank", ""),
                        "seed_id": row.get("seed_id", ""),
                        "data_status": "real",
                    }
                )

    if not out:
        raise RuntimeError("Failed to rebuild q2_routing_profile.csv from finished Q2 results")

    out_path = DATA_ROOT / "q2_routing_profile.csv"
    write_csv_rows(out_path, out)
    print(f"[saved] {out_path} ({len(out)} rows)")
    return out_path


def assert_generated_files(paths: list[Path], *, label: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Missing expected {label} files:\n{joined}")


def execute_notebook(notebook_path: Path) -> None:
    try:
        import nbformat  # type: ignore
        from nbclient import NotebookClient  # type: ignore
    except Exception:
        jupyter = shutil.which("jupyter")
        if not jupyter:
            raise RuntimeError(
                "Notebook execution requires nbformat+nbclient or the jupyter executable"
            )
        run_checked(
            [
                jupyter,
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(notebook_path),
            ],
            cwd=notebook_path.parent,
        )
        return

    print(f"[run] execute notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, notebook_path)


def execute_notebooks() -> None:
    for notebook in NOTEBOOKS:
        execute_notebook(notebook)


def compile_tex() -> None:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise RuntimeError("latexmk is not installed, so --compile-tex cannot run")
    run_checked([latexmk, "-pdf", "-interaction=nonstopmode", "sample-sigconf.tex"], cwd=TEX_ROOT)


def maybe_validate_q4() -> None:
    q4_summary = LOG_ROOT / "q4" / "summary.csv"
    if not q4_summary.exists():
        print("[note] q4 summary.csv is absent; skipping Q4 validation")
        return
    rows = read_csv_rows(q4_summary)
    ok_rows = [row for row in rows if str(row.get("status", "")).lower() == "ok"]
    if not ok_rows:
        print("[note] q4 summary.csv exists but has no ok rows; skipping Q4 validation")
        return
    q4_csv = DATA_ROOT / "q4_efficiency_table.csv"
    if q4_csv.exists():
        print(f"[ok] Q4 CSV present: {q4_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-run Q2/Q3/Q5 export and figure pipeline")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use for export commands",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset filter for the post-run export",
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Only rebuild CSV assets and skip notebook execution",
    )
    parser.add_argument(
        "--compile-tex",
        action="store_true",
        help="Compile sample-sigconf.tex after figure generation",
    )
    args = parser.parse_args()

    datasets = parse_csv_arg(args.datasets)

    print("[step] verify required experiment summaries")
    assert_required_logs(["q2", "q3", "q5"])

    print("[step] export notebook CSV bundle")
    export_bundle(args.python_bin, datasets)

    print("[step] rebuild q2_routing_profile.csv")
    rebuild_q2_routing_profile(datasets)

    print("[step] validate generated CSV files")
    assert_generated_files(EXPECTED_DATA_FILES, label="data")
    maybe_validate_q4()

    if not args.skip_notebooks:
        print("[step] execute Q2/Q3/Q5 notebooks")
        execute_notebooks()
        print("[step] validate generated figure PDFs")
        assert_generated_files(EXPECTED_FIGURES, label="figure")
    else:
        print("[note] notebook execution skipped by flag")

    if args.compile_tex:
        print("[step] compile paper tex")
        compile_tex()

    print("[done] post-run pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
