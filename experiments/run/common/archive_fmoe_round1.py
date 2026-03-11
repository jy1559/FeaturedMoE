#!/usr/bin/env python3
"""Archive deprecated FMoE tracks into quarantine with a summary index."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
QUARANTINE_DAY = datetime.now(timezone.utc).strftime("%Y%m%d")
BUCKET_NAME = "fmoe_round1"
QUARANTINE_LEVEL = "L10"
DATE_ROOT = EXP_DIR / "_quarantine" / QUARANTINE_DAY
BUCKET_ROOT = DATE_ROOT / f"{QUARANTINE_LEVEL}_{BUCKET_NAME}"
MANIFEST_PATH = BUCKET_ROOT / "manifest.jsonl"
INDEX_MD = DATE_ROOT / "fmoe_archive_index.md"
INDEX_CSV = DATE_ROOT / "fmoe_archive_index.csv"
REPORT_PATH = EXP_DIR / "run" / "artifacts" / "reports" / "fmoe_experiment_report_20260310.md"
QUARANTINE_MANAGER = EXP_DIR / "run" / "common" / "quarantine_manager.sh"


@dataclass(frozen=True)
class TrackSpec:
    track: str
    model_dir: str
    config_globs: tuple[str, ...]
    test_globs: tuple[str, ...]
    summary: str
    decision: str


TRACKS = (
    TrackSpec(
        track="fmoe_hir",
        model_dir="experiments/models/FeaturedMoE_HiR",
        config_globs=("experiments/configs/model/featured_moe_hir.yaml", "experiments/configs/model/featured_moe_hir_tune.yaml"),
        test_globs=(),
        summary="README/phase skeleton remained but no scored results or included overview runs.",
        decision="archive",
    ),
    TrackSpec(
        track="fmoe_hir2",
        model_dir="experiments/models/FeaturedMoE_HiR2",
        config_globs=("experiments/configs/model/featured_moe_hir2.yaml", "experiments/configs/model/featured_moe_hir2_tune.yaml"),
        test_globs=("experiments/tests/test_hir2_modules.py", "experiments/tests/test_hir2_registration_and_scripts.py"),
        summary="Stage-first / serial_weighted probe stayed well below the current mainline.",
        decision="archive",
    ),
    TrackSpec(
        track="fmoe_protox",
        model_dir="experiments/models/FeaturedMoE_ProtoX",
        config_globs=("experiments/configs/model/featured_moe_protox.yaml", "experiments/configs/model/featured_moe_protox_tune.yaml"),
        test_globs=("experiments/tests/test_protox_modules.py", "experiments/tests/test_protox_registration_and_scripts.py"),
        summary="Prototype-first routing explored broadly but did not reach mainline quality.",
        decision="archive",
    ),
    TrackSpec(
        track="fmoe_individual",
        model_dir="experiments/models/FeaturedMoE_Individual",
        config_globs=("experiments/configs/model/featured_moe_individual.yaml", "experiments/configs/model/featured_moe_individual_tune.yaml"),
        test_globs=("experiments/tests/test_individual_modules.py", "experiments/tests/test_individual_registration.py"),
        summary="Individual-expert branch kept some raw results but never graduated into a maintained experiment line.",
        decision="archive",
    ),
    TrackSpec(
        track="fmoe_v2_hir",
        model_dir="experiments/models/FeaturedMoE_v2_HiR",
        config_globs=("experiments/configs/model/featured_moe_v2_hir.yaml", "experiments/configs/model/featured_moe_v2_hir_tune.yaml"),
        test_globs=(),
        summary="Hybrid v2+HiR branch was not kept warm and has no maintained scored result set.",
        decision="archive",
    ),
    TrackSpec(
        track="fmoe_hgr_v3",
        model_dir="experiments/models/FeaturedMoE_HGRv3",
        config_globs=("experiments/configs/model/featured_moe_hgr_v3.yaml", "experiments/configs/model/featured_moe_hgr_v3_tune.yaml"),
        test_globs=("experiments/tests/test_hgr_v3_modules.py", "experiments/tests/test_hgr_v3_registration.py"),
        summary="Inner-teacher architecture probe stayed below mainline and recent overviews captured only non-OOM error runs.",
        decision="archive",
    ),
)


_OVERVIEW_FIELD_RE = re.compile(r"^- ([a-zA-Z0-9_]+): (.+)$")


def _load_result_jsons(track: str):
    result_dir = EXP_DIR / "run" / "artifacts" / "results" / track
    if not result_dir.exists():
        return []
    out = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            out.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


def _parse_overview(track: str) -> dict:
    overview = EXP_DIR / "run" / "artifacts" / "logs" / track / "experiment_overview.md"
    parsed = {
        "included_runs": None,
        "included_oom_runs": None,
        "excluded_non_oom_error_runs": None,
        "summarized_experiments": None,
    }
    if not overview.exists():
        return parsed
    for raw_line in overview.read_text(encoding="utf-8").splitlines():
        match = _OVERVIEW_FIELD_RE.match(raw_line.strip())
        if not match:
            continue
        key, value = match.groups()
        if key in parsed:
            try:
                parsed[key] = int(value)
            except ValueError:
                parsed[key] = value
    return parsed


def _count_log_files(track: str) -> int:
    log_dir = EXP_DIR / "run" / "artifacts" / "logs" / track
    if not log_dir.exists():
        return 0
    return sum(1 for _ in log_dir.rglob("*.log"))


def _count_run_files(track: str) -> int:
    run_dir = EXP_DIR / "run" / track
    if not run_dir.exists():
        return 0
    return sum(1 for path in run_dir.rglob("*") if path.is_file())


def _format_num(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _best_result_summary(track: str) -> dict:
    results = _load_result_jsons(track)
    if not results:
        return {
            "result_json_count": 0,
            "recorded_trials": 0,
            "completed_trials": 0,
            "best_mrr20": None,
            "best_phase": "-",
            "best_dataset": "-",
            "best_model": "-",
            "best_path": "-",
        }

    best_path, best_payload = max(
        results,
        key=lambda pair: float(pair[1].get("best_mrr@20", float("-inf")) or float("-inf")),
    )
    return {
        "result_json_count": len(results),
        "recorded_trials": sum(int(payload.get("n_recorded_trials", 0) or 0) for _, payload in results),
        "completed_trials": sum(int(payload.get("n_completed", 0) or 0) for _, payload in results),
        "best_mrr20": float(best_payload.get("best_mrr@20", 0.0)),
        "best_phase": str(best_payload.get("run_phase", "-")),
        "best_dataset": str(best_payload.get("dataset", "-")),
        "best_model": str(best_payload.get("model", "-")),
        "best_path": str(best_path.relative_to(EXP_DIR)),
    }


def _paths_to_move(spec: TrackSpec) -> list[Path]:
    paths = [
        REPO_ROOT / spec.model_dir,
        EXP_DIR / "run" / spec.track,
        EXP_DIR / "run" / "artifacts" / "logs" / spec.track,
        EXP_DIR / "run" / "artifacts" / "results" / spec.track,
    ]
    for pattern in spec.config_globs:
        paths.extend(REPO_ROOT.glob(pattern))
    for pattern in spec.test_globs:
        paths.extend(REPO_ROOT.glob(pattern))
    dedup = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        dedup.append(path)
    return dedup


def _build_rows() -> list[dict]:
    rows = []
    for spec in TRACKS:
        overview = _parse_overview(spec.track)
        best = _best_result_summary(spec.track)
        tuning_parts = [
            f"result_json={best['result_json_count']}",
            f"recorded_trials={best['recorded_trials']}",
            f"completed_trials={best['completed_trials']}",
            f"raw_logs={_count_log_files(spec.track)}",
            f"run_files={_count_run_files(spec.track)}",
        ]
        if overview["included_runs"] is not None:
            tuning_parts.append(f"overview_included={overview['included_runs']}")
        if overview["included_oom_runs"] is not None:
            tuning_parts.append(f"overview_oom={overview['included_oom_runs']}")
        if overview["excluded_non_oom_error_runs"] is not None:
            tuning_parts.append(f"overview_non_oom_error={overview['excluded_non_oom_error_runs']}")
        rows.append(
            {
                "track": spec.track,
                "model_dir": spec.model_dir,
                "result_json_count": best["result_json_count"],
                "recorded_trials": best["recorded_trials"],
                "completed_trials": best["completed_trials"],
                "included_runs": overview["included_runs"],
                "oom_runs": overview["included_oom_runs"],
                "non_oom_error_runs": overview["excluded_non_oom_error_runs"],
                "best_mrr20": best["best_mrr20"],
                "best_phase": best["best_phase"],
                "best_dataset": best["best_dataset"],
                "best_model": best["best_model"],
                "best_path": best["best_path"],
                "tuning_footprint": ", ".join(tuning_parts),
                "summary": spec.summary,
                "decision": spec.decision,
            }
        )
    return rows


def _write_index(rows: list[dict]) -> None:
    DATE_ROOT.mkdir(parents=True, exist_ok=True)

    with INDEX_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "track",
                "model_dir",
                "result_json_count",
                "recorded_trials",
                "completed_trials",
                "included_runs",
                "oom_runs",
                "non_oom_error_runs",
                "best_mrr20",
                "best_phase",
                "best_dataset",
                "best_model",
                "best_path",
                "tuning_footprint",
                "summary",
                "decision",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# FMoE Archive Index",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- archive_bucket: `{QUARANTINE_LEVEL}_{BUCKET_NAME}`",
        "- scope: archive deprecated candidate tracks while keeping `fmoe_v2`, `fmoe_v3`, `fmoe_hgr`, `fmoe_rule`, `fmoe_v4_distillation`, and `FeaturedMoE_N` active",
        "",
        "| track | model dir | tuning footprint | best MRR@20 | best phase | decision | summary |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {track} | `{model_dir}` | {tuning_footprint} | {best_mrr20} | `{best_phase}` | {decision} | {summary} |".format(
                track=row["track"],
                model_dir=row["model_dir"],
                tuning_footprint=row["tuning_footprint"],
                best_mrr20=_format_num(row["best_mrr20"]),
                best_phase=row["best_phase"],
                decision=row["decision"],
                summary=row["summary"],
            )
        )

    INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _move_paths(paths: list[Path]) -> None:
    move_cmd = [
        "bash",
        str(QUARANTINE_MANAGER),
        "move",
        "--level",
        QUARANTINE_LEVEL,
        "--bucket",
        BUCKET_NAME,
        "--reason",
        "Archive deprecated FMoE candidate tracks after common-model consolidation.",
        "--manifest",
        str(MANIFEST_PATH),
    ]
    move_cmd.extend(str(path) for path in paths)
    subprocess.run(move_cmd, cwd=str(REPO_ROOT), check=True)


def _validate_archive(rows: list[dict]) -> None:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Missing manifest: {MANIFEST_PATH}")
    manifest_tracks = set()
    for raw_line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        rel_path = str(payload.get("path", ""))
        for spec in TRACKS:
            if spec.track in rel_path or spec.model_dir in rel_path:
                manifest_tracks.add(spec.track)
    expected_tracks = {row["track"] for row in rows}
    if manifest_tracks != expected_tracks:
        raise RuntimeError(f"Manifest track mismatch: expected={expected_tracks}, got={manifest_tracks}")

    untouched_v4 = EXP_DIR / "run" / "artifacts" / "logs" / "fmoe_v4_distillation"
    if not untouched_v4.exists():
        raise RuntimeError("v4_distillation log directory should remain active but is missing.")

    for spec in TRACKS:
        src_model_dir = REPO_ROOT / spec.model_dir
        dst_model_dir = BUCKET_ROOT / spec.model_dir
        if src_model_dir.exists():
            raise RuntimeError(f"Source model dir still exists after archive: {src_model_dir}")
        if not dst_model_dir.exists():
            raise RuntimeError(f"Archived model dir missing in quarantine: {dst_model_dir}")


def main() -> None:
    rows = _build_rows()
    _write_index(rows)

    move_paths = []
    for spec in TRACKS:
        move_paths.extend(_paths_to_move(spec))
    _move_paths(move_paths)
    _validate_archive(rows)

    print(f"Archive completed. Index: {INDEX_MD}")
    print(f"Archive CSV: {INDEX_CSV}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
