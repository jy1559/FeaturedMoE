#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 Phase9_2 verification runs (4 candidates x 4 hparams x 4 seeds)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from run_phase9_auxloss import (  # noqa: E402
    ARTIFACT_ROOT,
    AXIS as SOURCE_AXIS,
    EXP_DIR,
    LOG_ROOT,
    MODEL_TAG,
    PHASE as SOURCE_PHASE,
    TRACK,
    _apply_base_overrides,
    _aux_profiles,
    _base_definitions,
    _base_fixed_overrides,
    _dataset_tag,
    _extract_run_phase_from_log,
    _is_completed_log,
    _load_result_index,
    _metric_to_float,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)

AXIS = "phase9_2_verification_v2"
PHASE = "P9_2"

CONCEPT_ORDER = ("C0", "C1", "C2", "C3")
DEFAULT_CANDIDATE_RUN_PHASES: Dict[str, str] = {
    # C0 winner (best_valid top in Natural)
    "K1": "P9_B4_C0_N4_S1",
    # C1 winner (best_valid top in CanonicalBalance)
    "K2": "P9_B3_C1_B1_S1",
    # C2 winner (best_valid top in Specialization)
    "K3": "P9_B1_C2_S3_S1",
    # C3 winner (best_valid top in FeatureAlignment)
    "K4": "P9_B2_C3_F2_S1",
}


def _hparam_variants() -> Dict[str, Dict[str, float]]:
    return {
        "H1": {
            "embedding_size": 128,
            "d_ff": 256,
            "d_expert_hidden": 128,
            "d_router_hidden": 64,
            "fixed_weight_decay": 1e-6,
            "fixed_hidden_dropout_prob": 0.15,
        },
        "H2": {
            "embedding_size": 160,
            "d_ff": 320,
            "d_expert_hidden": 160,
            "d_router_hidden": 80,
            "fixed_weight_decay": 5e-7,
            "fixed_hidden_dropout_prob": 0.12,
        },
        "H3": {
            "embedding_size": 160,
            "d_ff": 320,
            "d_expert_hidden": 160,
            "d_router_hidden": 80,
            "fixed_weight_decay": 2e-6,
            "fixed_hidden_dropout_prob": 0.18,
        },
        "H4": {
            "embedding_size": 112,
            "d_ff": 224,
            "d_expert_hidden": 112,
            "d_router_hidden": 56,
            "fixed_weight_decay": 3e-6,
            "fixed_hidden_dropout_prob": 0.20,
        },
    }


def _profile_lookup() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for profile in _aux_profiles():
        key = f"{str(profile['concept_id']).upper()}_{str(profile['combo_id']).upper()}"
        out[key] = profile
    return out


def _phase92_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset) / MODEL_TAG


def _phase92_axis_dataset_dir(dataset: str) -> Path:
    root = LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _phase92_summary_csv_path(dataset: str) -> Path:
    return _phase92_axis_dataset_dir(dataset) / "summary.csv"


def _summary_fieldnames() -> list[str]:
    return [
        "run_phase",
        "run_id",
        "dataset",
        "candidate_id",
        "source_run_phase",
        "base_id",
        "concept_id",
        "combo_id",
        "main_aux",
        "support_aux",
        "hvar_id",
        "seed_id",
        "gpu_id",
        "status",
        "run_best_valid_mrr20",
        "test_mrr20",
        "n_completed",
        "interrupted",
        "result_path",
        "timestamp_utc",
    ]


def _write_summary_csv(path: Path, rows: list[Dict[str, Any]], result_index: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writeheader()
        for row in rows:
            rec = result_index.get(str(row["run_phase"]))
            status = "pending"
            best = ""
            test = ""
            n_completed = ""
            interrupted = ""
            result_path = ""
            if isinstance(rec, dict):
                best_v = _metric_to_float(rec.get("best_mrr"))
                test_v = _metric_to_float(rec.get("test_mrr"))
                best = "" if best_v is None else f"{best_v:.6f}"
                test = "" if test_v is None else f"{test_v:.6f}"
                n_completed = int(rec.get("n_completed", 0) or 0)
                interrupted = bool(rec.get("interrupted", False))
                result_path = str(rec.get("path", "") or "")
                if best_v is not None and n_completed > 0:
                    status = "completed"
                elif result_path:
                    status = "result_found"
            writer.writerow(
                {
                    "run_phase": row["run_phase"],
                    "run_id": row["run_id"],
                    "dataset": row["dataset"],
                    "candidate_id": row["candidate_id"],
                    "source_run_phase": row["source_run_phase"],
                    "base_id": row["base_id"],
                    "concept_id": row["concept_id"],
                    "combo_id": row["combo_id"],
                    "main_aux": row["main_aux"],
                    "support_aux": row["support_aux"],
                    "hvar_id": row["hvar_id"],
                    "seed_id": row["seed_id"],
                    "gpu_id": row.get("assigned_gpu", ""),
                    "status": status,
                    "run_best_valid_mrr20": best,
                    "test_mrr20": test,
                    "n_completed": n_completed,
                    "interrupted": interrupted,
                    "result_path": result_path,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )


def _scan_completed_run_phases(dataset: str) -> set[str]:
    done = set()
    root = _phase92_log_dir(dataset)
    if not root.exists():
        return done
    for log_path in sorted(root.glob("*.log")):
        run_phase = _extract_run_phase_from_log(log_path)
        if not run_phase:
            continue
        if _is_completed_log(log_path):
            done.add(run_phase)
    return done


def _completed_by_result(result_index: Dict[str, Dict[str, Any]]) -> set[str]:
    done = set()
    for run_phase, rec in result_index.items():
        if _metric_to_float(rec.get("best_mrr")) is not None and int(rec.get("n_completed", 0)) > 0:
            done.add(run_phase)
    return done


def _candidate_sort_key(candidate_id: str) -> tuple[str, int]:
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(candidate_id))
    if not m:
        return (str(candidate_id), 0)
    return (m.group(1), int(m.group(2)))


def _parse_phase9_run_phase(run_phase: str) -> tuple[str, str, str, int] | None:
    # Expected: P9_<BASE>_<CONCEPT>_<COMBO>_S1
    parts = str(run_phase).split("_")
    if len(parts) != 5:
        return None
    if parts[0] != "P9":
        return None
    base_id = str(parts[1]).upper()
    concept_id = str(parts[2]).upper()
    combo_id = str(parts[3]).upper()
    seed_token = str(parts[4]).upper()
    if base_id not in {"B1", "B2", "B3", "B4"}:
        return None
    if concept_id not in {"C0", "C1", "C2", "C3"}:
        return None
    if not seed_token.startswith("S"):
        return None
    try:
        seed_id = int(seed_token[1:])
    except Exception:
        return None
    return base_id, concept_id, combo_id, seed_id


def _load_candidate_map_json(path: str) -> Dict[str, str]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("candidate-map-json must be a JSON object")

    out: Dict[str, str] = {}
    for raw_candidate_id, raw_value in obj.items():
        candidate_id = str(raw_candidate_id).upper().strip()
        run_phase = str(raw_value).strip()
        if not candidate_id:
            raise ValueError(f"Invalid candidate id in candidate-map-json: {raw_candidate_id}")
        parsed = _parse_phase9_run_phase(run_phase)
        if parsed is None:
            raise ValueError(
                f"Invalid run_phase in candidate-map-json for {candidate_id}: {raw_value} "
                "(expected P9_<BASE>_<CONCEPT>_<COMBO>_S<seed>)"
            )
        out[candidate_id] = run_phase
    return out


def _build_candidate_record(candidate_id: str, source_run_phase: str, profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    parsed = _parse_phase9_run_phase(source_run_phase)
    if parsed is None:
        raise ValueError(f"Invalid source run_phase: {source_run_phase}")
    base_id, concept_id, combo_id, _ = parsed
    key = f"{concept_id}_{combo_id}"
    profile = profiles.get(key)
    if profile is None:
        raise ValueError(f"Unknown profile for source run_phase={source_run_phase} (key={key})")
    return {
        "candidate_id": str(candidate_id).upper(),
        "source_run_phase": source_run_phase,
        "base_id": base_id,
        "concept_id": concept_id,
        "combo_id": combo_id,
        "main_aux": str(profile.get("main_aux", "none")),
        "support_aux": str(profile.get("support_aux", "none")),
        "scenario": str(profile.get("scenario", "")),
        "overrides": copy.deepcopy(dict(profile.get("overrides", {}) or {})),
    }


def _validate_candidates_against_source(
    candidates: list[Dict[str, Any]],
    dataset: str,
    source_min_completed: int,
    allow_partial_source: bool,
) -> None:
    source_index = _load_result_index(dataset, SOURCE_AXIS)
    missing = []
    partial = []
    for cand in candidates:
        run_phase = str(cand["source_run_phase"])
        rec = source_index.get(run_phase)
        if not isinstance(rec, dict):
            missing.append(run_phase)
            continue
        best = _metric_to_float(rec.get("best_mrr"))
        n_completed = int(rec.get("n_completed", 0) or 0)
        interrupted = bool(rec.get("interrupted", False))
        cand["source_best_mrr"] = best
        cand["source_test_mrr"] = _metric_to_float(rec.get("test_mrr"))
        cand["source_n_completed"] = n_completed
        cand["source_interrupted"] = interrupted
        cand["source_result_path"] = str(rec.get("path", ""))
        if best is None or n_completed < int(source_min_completed) or interrupted:
            partial.append(run_phase)

    if missing:
        raise RuntimeError(
            "Missing source phase9 result(s) for candidate(s): "
            + ", ".join(sorted(missing))
            + f" (axis={SOURCE_AXIS})"
        )
    if partial and not allow_partial_source:
        raise RuntimeError(
            "Candidate source result is incomplete/partial (set --allow-partial-source to bypass): "
            + ", ".join(sorted(partial))
        )


def _resolve_candidates(args: argparse.Namespace) -> list[Dict[str, Any]]:
    profiles = _profile_lookup()

    candidate_map = dict(DEFAULT_CANDIDATE_RUN_PHASES)
    json_map = _load_candidate_map_json(args.candidate_map_json)
    if json_map:
        candidate_map = dict(json_map)

    only_candidate = {tok.upper() for tok in _parse_csv_strings(args.only_candidate)}
    out: list[Dict[str, Any]] = []
    for candidate_id in sorted(candidate_map.keys(), key=_candidate_sort_key):
        if only_candidate and candidate_id not in only_candidate:
            continue
        out.append(
            _build_candidate_record(
                candidate_id=candidate_id,
                source_run_phase=str(candidate_map[candidate_id]),
                profiles=profiles,
            )
        )
    if not out:
        raise RuntimeError("No candidates selected. Check --only-candidate / --candidate-map-json.")

    _validate_candidates_against_source(
        candidates=out,
        dataset=args.dataset,
        source_min_completed=int(args.source_min_completed),
        allow_partial_source=bool(args.allow_partial_source),
    )

    concept_set = {str(c["concept_id"]).upper() for c in out}
    expected = set(CONCEPT_ORDER)
    if len(out) == 4 and concept_set != expected:
        print(f"[phase9_2][warn] 4 candidates selected but concept coverage is {sorted(concept_set)} (expected {sorted(expected)}).")
    return out


def _run_phase_name(candidate: Dict[str, Any], hvar_id: str, seed_id: int) -> str:
    return (
        f"P9_2_{candidate['candidate_id']}_{candidate['base_id']}_{candidate['concept_id']}_"
        f"{candidate['combo_id']}_{hvar_id}_S{int(seed_id)}"
    )


def _run_id(candidate: Dict[str, Any], hvar_id: str, seed_id: int) -> str:
    return (
        f"{candidate['candidate_id']}_{candidate['base_id']}_{candidate['concept_id']}_"
        f"{candidate['combo_id']}_{hvar_id}_S{int(seed_id)}"
    )


def _build_rows(args: argparse.Namespace, candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")
    hvars = _hparam_variants()
    bases = _base_definitions()

    rows: list[Dict[str, Any]] = []
    seed_cursor = 0
    for candidate in candidates:
        base_id = str(candidate["base_id"]).upper()
        base_cfg = bases[base_id]
        for hvar_id in ("H1", "H2", "H3", "H4"):
            h = dict(hvars[hvar_id])
            for seed_id in seeds:
                overrides = _base_fixed_overrides()
                _apply_base_overrides(
                    overrides=overrides,
                    base_cfg=base_cfg,
                    feature_group_bias_lambda=float(args.feature_group_bias_lambda),
                    rule_bias_scale=float(args.rule_bias_scale),
                )
                for key, value in dict(candidate.get("overrides", {}) or {}).items():
                    overrides[str(key)] = value
                run_phase = _run_phase_name(candidate, hvar_id=hvar_id, seed_id=seed_id)
                run_id = _run_id(candidate, hvar_id=hvar_id, seed_id=seed_id)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "candidate_id": str(candidate["candidate_id"]),
                        "source_run_phase": str(candidate["source_run_phase"]),
                        "base_id": base_id,
                        "concept_id": str(candidate["concept_id"]).upper(),
                        "combo_id": str(candidate["combo_id"]).upper(),
                        "main_aux": str(candidate["main_aux"]),
                        "support_aux": str(candidate["support_aux"]),
                        "hvar_id": hvar_id,
                        "hparams": h,
                        "seed_id": int(seed_id),
                        "seed_offset": int(seed_cursor),
                        "run_id": run_id,
                        "run_phase": run_phase,
                        "overrides": overrides,
                    }
                )
                seed_cursor += 1
    return rows


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    h = dict(row["hparams"])
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(args.tune_epochs)),
        "--tune-patience",
        str(int(args.tune_patience)),
        "--seed",
        str(int(args.seed_base) + int(row["seed_offset"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        row["run_phase"],
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        "eval_mode=session",
        "feature_mode=full_v3",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_phase={hydra_literal(PHASE)}",
        f"MAX_ITEM_LIST_LENGTH={int(args.max_item_list_length)}",
        f"train_batch_size={int(args.batch_size)}",
        f"eval_batch_size={int(args.batch_size)}",
        f"embedding_size={int(h['embedding_size'])}",
        f"num_heads={int(args.num_heads)}",
        f"attn_dropout_prob={hydra_literal(float(args.attn_dropout_prob))}",
        f"d_ff={int(h['d_ff'])}",
        f"d_feat_emb={int(args.d_feat_emb)}",
        f"d_expert_hidden={int(h['d_expert_hidden'])}",
        f"d_router_hidden={int(h['d_router_hidden'])}",
        f"expert_scale={int(args.expert_scale)}",
        "++layer_layout=[macro,mid,micro]",
        f"++search.learning_rate={hydra_literal([float(args.search_lr_min), float(args.search_lr_max)])}",
        f"++search.weight_decay={hydra_literal([float(h['fixed_weight_decay'])])}",
        f"++search.hidden_dropout_prob={hydra_literal([float(h['fixed_hidden_dropout_prob'])])}",
        f"++search.lr_scheduler_type={hydra_literal(_parse_csv_strings(args.search_lr_scheduler))}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search_space_type_overrides.lr_scheduler_type=choice",
        f"++p9_2_candidate_id={hydra_literal(row['candidate_id'])}",
        f"++p9_2_source_run_phase={hydra_literal(row['source_run_phase'])}",
        f"++p9_2_base_id={hydra_literal(row['base_id'])}",
        f"++p9_2_concept_id={hydra_literal(row['concept_id'])}",
        f"++p9_2_combo_id={hydra_literal(row['combo_id'])}",
        f"++p9_2_main_aux={hydra_literal(row['main_aux'])}",
        f"++p9_2_support_aux={hydra_literal(row['support_aux'])}",
        f"++p9_2_hvar_id={hydra_literal(row['hvar_id'])}",
        f"++p9_2_run_id={hydra_literal(row['run_id'])}",
    ]
    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _log_path(row: Dict[str, Any], dataset: str) -> Path:
    root = _phase92_log_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{row['run_id']}.log"


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    lines = [
        "[PHASE9_2_SETTING_HEADER]",
        (
            f"run_id={row['run_id']} run_phase={row['run_phase']} candidate={row['candidate_id']} "
            f"base={row['base_id']} concept={row['concept_id']} combo={row['combo_id']} "
            f"hvar={row['hvar_id']} seed={row['seed_id']}"
        ),
        f"source_run_phase={row['source_run_phase']} main_aux={row['main_aux']} support_aux={row['support_aux']}",
        f"dataset={row['dataset']} gpu={gpu_id}",
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={int(args.seed_base) + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _write_matrix_manifest(rows: list[Dict[str, Any]], candidates: list[Dict[str, Any]], args: argparse.Namespace) -> Path:
    if args.manifest_out:
        out_path = Path(args.manifest_out)
    else:
        out_path = _phase92_axis_dataset_dir(args.dataset) / "verification_matrix.json"
    payload = {
        "track": TRACK,
        "source_axis": SOURCE_AXIS,
        "source_phase": SOURCE_PHASE,
        "axis": AXIS,
        "phase": PHASE,
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "source_run_phase": c["source_run_phase"],
                "base_id": c["base_id"],
                "concept_id": c["concept_id"],
                "combo_id": c["combo_id"],
                "main_aux": c["main_aux"],
                "support_aux": c["support_aux"],
                "source_best_mrr": c.get("source_best_mrr"),
                "source_test_mrr": c.get("source_test_mrr"),
                "source_n_completed": c.get("source_n_completed"),
            }
            for c in candidates
        ],
        "n_rows": len(rows),
        "rows": [
            {
                "run_id": r["run_id"],
                "run_phase": r["run_phase"],
                "candidate_id": r["candidate_id"],
                "base_id": r["base_id"],
                "concept_id": r["concept_id"],
                "combo_id": r["combo_id"],
                "hvar_id": r["hvar_id"],
                "seed_id": r["seed_id"],
            }
            for r in rows
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _launch_rows(rows: list[Dict[str, Any]], gpus: list[str], args: argparse.Namespace) -> int:
    if not rows:
        print("[phase9_2] no rows to run.")
        return 0

    for idx, row in enumerate(rows):
        row["assigned_gpu"] = gpus[idx % len(gpus)]
        row["assigned_order"] = idx + 1

    result_index = _load_result_index(args.dataset, AXIS)
    done_results = _completed_by_result(result_index)
    done_logs: set[str] = set()
    if args.resume_from_logs:
        done_logs = _scan_completed_run_phases(args.dataset)

    runnable: list[Dict[str, Any]] = []
    for row in rows:
        run_phase = str(row["run_phase"])
        if run_phase in done_results:
            continue
        lp = _log_path(row, args.dataset)
        if args.resume_from_logs and lp.exists() and _is_completed_log(lp):
            continue
        if args.resume_from_logs and run_phase in done_logs:
            continue
        runnable.append(row)

    summary_path = _phase92_summary_csv_path(args.dataset)
    _write_summary_csv(summary_path, rows, result_index)

    if not runnable:
        print("[phase9_2] all runs are already completed by result/log markers.")
        return 0

    if args.dry_run:
        for row in runnable:
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, row["assigned_gpu"], args)
            print(
                f"[dry-run] gpu={row['assigned_gpu']} run={row['run_id']} "
                f"candidate={row['candidate_id']} source={row['source_run_phase']} "
                f"hvar={row['hvar_id']} -> {lp}"
            )
            print("          " + " ".join(cmd))
        return 0

    gpu_bins: Dict[str, deque[Dict[str, Any]]] = {gpu: deque() for gpu in gpus}
    for row in runnable:
        gpu_bins[str(row["assigned_gpu"])].append(row)

    active: Dict[str, Dict[str, Any]] = {}
    while True:
        for gpu_id in gpus:
            if gpu_id in active:
                continue
            if not gpu_bins[gpu_id]:
                continue
            row = gpu_bins[gpu_id].popleft()
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, gpu_id, args)
            _write_log_preamble(lp, row, gpu_id, args, cmd)
            env = dict(os.environ)
            env["HYPEROPT_RESULTS_DIR"] = str(ARTIFACT_ROOT / "results")
            with lp.open("a", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, cwd=EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[gpu_id] = {"proc": proc, "row": row, "log_path": lp}
            print(
                f"[launch] gpu={gpu_id} run={row['run_id']} "
                f"(candidate={row['candidate_id']} source={row['source_run_phase']} hvar={row['hvar_id']})"
            )

        done_gpu = []
        for gpu_id, slot in active.items():
            proc = slot["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            row = slot["row"]
            lp = slot["log_path"]
            done_gpu.append(gpu_id)
            print(f"[done] gpu={gpu_id} run={row['run_id']} rc={rc} log={lp}")

            result_index = _load_result_index(args.dataset, AXIS)
            _write_summary_csv(summary_path, rows, result_index)

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = any(gpu_bins[g] for g in gpus)
        if not pending and not active:
            break
        time.sleep(3)

    result_index = _load_result_index(args.dataset, AXIS)
    _write_summary_csv(summary_path, rows, result_index)
    print(f"[phase9_2] summary updated: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Phase9_2 verification launcher (candidate-based)")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--seed-base", type=int, default=58000)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)

    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)
    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")

    parser.add_argument("--only-candidate", default="", help="Comma-separated subset of candidate ids (e.g. K1,K4)")
    parser.add_argument(
        "--candidate-map-json",
        default="",
        help='Optional JSON map like {"K1":"P9_B4_C0_N4_S1", "K2":"P9_B3_C1_B1_S1", ...}',
    )
    parser.add_argument("--source-min-completed", type=int, default=20)
    parser.add_argument("--allow-partial-source", action="store_true")

    parser.add_argument("--manifest-out", default="", help="Optional matrix JSON output path")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=4)
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals = 1
    args.tune_epochs = 1
    args.tune_patience = 1
    args.seeds = "1"
    args.gpus = _parse_csv_strings(args.gpus)[0] if _parse_csv_strings(args.gpus) else "0"


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided")

    candidates = _resolve_candidates(args)
    print(f"[phase9_2] selected_candidates={len(candidates)} axis={AXIS} phase={PHASE}")
    for cand in candidates:
        print(
            "[phase9_2][candidate] "
            f"{cand['candidate_id']} -> {cand['source_run_phase']} "
            f"(base={cand['base_id']} concept={cand['concept_id']} combo={cand['combo_id']} "
            f"best={cand.get('source_best_mrr')} test={cand.get('source_test_mrr')})"
        )

    rows = _build_rows(args, candidates)
    if args.smoke_test:
        rows = rows[: max(int(args.smoke_max_runs), 1)]
    if not rows:
        raise SystemExit("No rows matched filters")

    manifest_path = _write_matrix_manifest(rows, candidates, args)
    print(f"[phase9_2] dataset={args.dataset} total_rows={len(rows)} manifest={manifest_path}")
    return _launch_rows(rows=rows, gpus=gpus, args=args)


if __name__ == "__main__":
    raise SystemExit(main())

