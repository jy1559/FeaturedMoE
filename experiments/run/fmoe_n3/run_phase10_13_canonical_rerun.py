#!/usr/bin/env python3
"""Rerun only corrected Phase10/13 settings with targeted artifact overwrite."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import run_phase10_feature_portability as phase10
import run_phase13_feature_sanity as phase13
from run_phase9_auxloss import ARTIFACT_ROOT
from run_phase_wide_common import sanitize_token


DEFAULT_PHASE10_SETTINGS = [
    "P10-22_NO_CATEGORY_NO_TIMESTAMP",
    "P10-23_COMMON_TEMPLATE_NO_CATEGORY",
]
DEFAULT_PHASE13_SETTINGS = [
    "P13-01_CATEGORY_ZERO_DATA",
    "P13-14_FEATURE_ROLE_SWAP",
]


@dataclass
class PhasePlan:
    name: str
    axis: str
    phase_id: str
    settings: list[str]
    rows: list[dict]
    log_files: list[Path]
    summary_path: Path
    manifest_out: Path
    command: list[str]


def _parse_csv(value: str) -> list[str]:
    return [tok.strip() for tok in str(value or "").split(",") if tok.strip()]


def _parse_with_module(module, cli_args: list[str]) -> argparse.Namespace:
    old_argv = sys.argv
    try:
        sys.argv = [getattr(module, "__file__", "runner.py"), *cli_args]
        return module.parse_args()
    finally:
        sys.argv = old_argv


def _phase13_log_path(row: dict, dataset: str) -> Path:
    phase = sanitize_token(phase13.PHASE_ID, upper=True)
    axis_id = sanitize_token(str(row.get("axis_id", "A")), upper=True)
    axis_desc = sanitize_token(str(row.get("axis_desc", "axis")), upper=False)
    setting_id = sanitize_token(str(row.get("setting_id", "00")), upper=True)
    setting_desc = sanitize_token(str(row.get("setting_desc", "setting")), upper=True)
    filename = f"{phase}_{axis_id}_{axis_desc}_{setting_id}_{setting_desc}.log"
    return phase13._phase_log_dir(dataset) / filename


def _build_phase10_plan(args: argparse.Namespace, python_bin: str) -> PhasePlan:
    settings = _parse_csv(args.phase10_settings)
    cli_args = [
        "--dataset", args.dataset,
        "--gpus", args.gpus,
        "--seeds", args.seeds,
        "--only-setting", ",".join(settings),
        "--include-extra-24",
        "--max-evals", str(int(args.max_evals)),
        "--tune-epochs", str(int(args.tune_epochs)),
        "--tune-patience", str(int(args.tune_patience)),
        "--no-resume-from-logs",
        "--verify-logging",
        "--manifest-out", str(phase10._phase10_axis_dataset_dir(args.dataset) / "phase10_canonical_rerun_matrix.json"),
    ]
    parsed = _parse_with_module(phase10, cli_args)
    settings_obj = phase10._build_settings(parsed)
    rows = phase10._build_rows(parsed, settings_obj)
    command = [
        python_bin,
        str(Path(phase10.__file__).resolve()),
        *cli_args,
    ]
    if args.dry_run:
        command.append("--dry-run")
    return PhasePlan(
        name="Phase10",
        axis=phase10.AXIS,
        phase_id=phase10.PHASE,
        settings=settings,
        rows=rows,
        log_files=[phase10._log_path(row, args.dataset) for row in rows],
        summary_path=phase10._phase10_summary_csv_path(args.dataset),
        manifest_out=Path(cli_args[cli_args.index("--manifest-out") + 1]),
        command=command,
    )


def _build_phase13_plan(args: argparse.Namespace, python_bin: str) -> PhasePlan:
    settings = _parse_csv(args.phase13_settings)
    cli_args = [
        "--dataset", args.dataset,
        "--gpus", args.gpus,
        "--hparams", args.phase13_hparams,
        "--seeds", args.seeds,
        "--only-setting", ",".join(settings),
        "--max-evals", str(int(args.max_evals)),
        "--tune-epochs", str(int(args.tune_epochs)),
        "--tune-patience", str(int(args.tune_patience)),
        "--no-resume-from-logs",
        "--verify-logging",
        "--manifest-out", str(phase13._phase_log_dir(args.dataset) / "phase13_canonical_rerun_matrix.json"),
    ]
    parsed = _parse_with_module(phase13, cli_args)
    settings_obj = phase13._build_settings(parsed)
    rows = phase13._build_rows(parsed, settings_obj)
    command = [
        python_bin,
        str(Path(phase13.__file__).resolve()),
        *cli_args,
    ]
    if args.dry_run:
        command.append("--dry-run")
    return PhasePlan(
        name="Phase13",
        axis=phase13.AXIS,
        phase_id=phase13.PHASE_ID,
        settings=settings,
        rows=rows,
        log_files=[_phase13_log_path(row, args.dataset) for row in rows],
        summary_path=phase13._summary_path(args.dataset),
        manifest_out=Path(cli_args[cli_args.index("--manifest-out") + 1]),
        command=command,
    )


def _remove_path(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        return True
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _remove_matching(root: Path, pattern: str, *, dry_run: bool) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in root.glob(pattern):
        if _remove_path(path, dry_run=dry_run):
            removed += 1
    return removed


def _prune_summary(summary_path: Path, *, run_phases: set[str], setting_keys: set[str], dry_run: bool) -> int:
    if not summary_path.exists():
        return 0
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    kept = [
        row for row in rows
        if str(row.get("run_phase", "")).strip() not in run_phases
        and str(row.get("setting_key", "")).strip() not in setting_keys
    ]
    removed = len(rows) - len(kept)
    if removed <= 0 or dry_run:
        return max(removed, 0)
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return removed


def _cleanup_phase_artifacts(plan: PhasePlan, dataset: str, *, dry_run: bool) -> dict:
    run_phases = {str(row.get("run_phase", "")).strip() for row in plan.rows}
    setting_keys = {str(row.get("setting_key", "")).strip() for row in plan.rows}
    tokens = {rp.lower() for rp in run_phases if rp}

    log_removed = sum(1 for path in plan.log_files if _remove_path(path, dry_run=dry_run))
    summary_pruned = _prune_summary(
        plan.summary_path,
        run_phases=run_phases,
        setting_keys=setting_keys,
        dry_run=dry_run,
    )

    logging_root = ARTIFACT_ROOT / "logging" / "fmoe_n3" / dataset / plan.phase_id
    normal_root = ARTIFACT_ROOT / "results" / "fmoe_n3" / "normal" / plan.axis / plan.phase_id / dataset / "FMoEN3"
    special_result_root = ARTIFACT_ROOT / "results" / "fmoe_n3" / "special" / plan.axis / plan.phase_id / dataset / "FMoEN3"
    special_log_root = ARTIFACT_ROOT / "logs" / "fmoe_n3" / "special" / plan.axis / plan.phase_id / dataset / "FMoEN3"

    logging_removed = 0
    normal_removed = 0
    special_result_removed = 0
    special_log_removed = 0
    for token in sorted(tokens):
        logging_removed += _remove_matching(logging_root, f"*{token}*", dry_run=dry_run)
        normal_removed += _remove_matching(normal_root, f"*{token}*", dry_run=dry_run)
        special_result_removed += _remove_matching(special_result_root, f"*{token}*", dry_run=dry_run)
        special_log_removed += _remove_matching(special_log_root, f"*{token}*", dry_run=dry_run)

    _remove_path(plan.manifest_out, dry_run=dry_run)

    return {
        "log_removed": log_removed,
        "summary_pruned": summary_pruned,
        "logging_removed": logging_removed,
        "normal_removed": normal_removed,
        "special_result_removed": special_result_removed,
        "special_log_removed": special_log_removed,
    }


def _run_command(cmd: list[str], *, cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun corrected canonical settings for Phase10/13 only")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--phase13-hparams", default="1")
    parser.add_argument("--phase10-settings", default=",".join(DEFAULT_PHASE10_SETTINGS))
    parser.add_argument("--phase13-settings", default=",".join(DEFAULT_PHASE13_SETTINGS))
    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--python-bin", default="/venv/FMoE/bin/python")
    parser.add_argument("--skip-phase10", action="store_true")
    parser.add_argument("--skip-phase13", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    plans: list[PhasePlan] = []
    if not args.skip_phase10 and _parse_csv(args.phase10_settings):
        plans.append(_build_phase10_plan(args, args.python_bin))
    if not args.skip_phase13 and _parse_csv(args.phase13_settings):
        plans.append(_build_phase13_plan(args, args.python_bin))
    if not plans:
        print("[rerun] No target phase/settings selected.")
        return 0

    print("[rerun] Target settings")
    for plan in plans:
        print(f"  - {plan.name}: {','.join(plan.settings)} (rows={len(plan.rows)})")

    if not args.no_clean:
        for plan in plans:
            stats = _cleanup_phase_artifacts(plan, args.dataset, dry_run=bool(args.dry_run))
            print(
                f"[cleanup:{plan.name}] logs={stats['log_removed']} summary_rows={stats['summary_pruned']} "
                f"logging_dirs={stats['logging_removed']} normal_results={stats['normal_removed']} "
                f"special_results={stats['special_result_removed']} special_logs={stats['special_log_removed']}"
            )
    else:
        print("[cleanup] skipped by --no-clean")

    for plan in plans:
        _run_command(plan.command, cwd=repo_root)

    print("[rerun] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
