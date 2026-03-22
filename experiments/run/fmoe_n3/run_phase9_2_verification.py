#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 Phase9_2 verification runs (4 winners x 4 hparams x 4 seeds)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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

AXIS = "phase9_2_verification_v1"
PHASE = "P9_2"


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
    for p in _aux_profiles():
        key = f"{str(p['concept_id']).upper()}_{str(p['combo_id']).upper()}"
        out[key] = p
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
        "base_id",
        "winner_concept_id",
        "winner_combo_id",
        "winner_main_aux",
        "winner_support_aux",
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
                    "base_id": row["base_id"],
                    "winner_concept_id": row["winner_concept_id"],
                    "winner_combo_id": row["winner_combo_id"],
                    "winner_main_aux": row["winner_main_aux"],
                    "winner_support_aux": row["winner_support_aux"],
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


def _parse_phase9_run_phase(run_phase: str) -> tuple[str, str, str] | None:
    # Expected: P9_<BASE>_<CONCEPT>_<COMBO>_S1
    parts = str(run_phase).split("_")
    if len(parts) < 5:
        return None
    if parts[0] != "P9":
        return None
    base_id = str(parts[1]).upper()
    concept_id = str(parts[2]).upper()
    combo_id = str(parts[3]).upper()
    seed_tok = str(parts[4]).upper()
    if base_id not in {"B1", "B2", "B3", "B4"}:
        return None
    if not seed_tok.startswith("S"):
        return None
    return base_id, concept_id, combo_id


def _load_winner_map_json(path: str) -> Dict[str, str]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("winner-map-json must be a JSON object")
    out: Dict[str, str] = {}
    for base_id, profile_key in obj.items():
        b = str(base_id).upper().strip()
        p = str(profile_key).upper().strip()
        if b not in {"B1", "B2", "B3", "B4"}:
            raise ValueError(f"Invalid base_id in winner-map-json: {base_id}")
        out[b] = p
    return out


def _select_winners(
    *,
    dataset: str,
    manual_map: Dict[str, str],
    allow_fallback_winner: bool,
) -> Dict[str, Dict[str, Any]]:
    profiles = _profile_lookup()
    winners: Dict[str, Dict[str, Any]] = {}

    for base_id, profile_key in manual_map.items():
        prof = profiles.get(profile_key)
        if prof is None:
            raise ValueError(f"Unknown profile key in winner-map-json: {profile_key}")
        winners[base_id] = {
            "concept_id": str(prof["concept_id"]).upper(),
            "combo_id": str(prof["combo_id"]).upper(),
            "main_aux": str(prof["main_aux"]),
            "support_aux": str(prof["support_aux"]),
            "scenario": str(prof["scenario"]),
            "overrides": copy.deepcopy(dict(prof.get("overrides", {}) or {})),
            "source": "manual_map",
        }

    if len(winners) == 4:
        return winners

    result_index = _load_result_index(dataset, SOURCE_AXIS)
    by_base: Dict[str, list[Dict[str, Any]]] = {b: [] for b in ("B1", "B2", "B3", "B4")}
    for run_phase, rec in result_index.items():
        parsed = _parse_phase9_run_phase(run_phase)
        if parsed is None:
            continue
        base_id, concept_id, combo_id = parsed
        best = _metric_to_float(rec.get("best_mrr"))
        if best is None:
            continue
        test = _metric_to_float(rec.get("test_mrr"))
        profile_key = f"{concept_id}_{combo_id}"
        prof = profiles.get(profile_key)
        if prof is None:
            continue
        by_base[base_id].append(
            {
                "base_id": base_id,
                "concept_id": concept_id,
                "combo_id": combo_id,
                "best_mrr": best,
                "test_mrr": -1e9 if test is None else float(test),
                "main_aux": str(prof["main_aux"]),
                "support_aux": str(prof["support_aux"]),
                "scenario": str(prof["scenario"]),
                "overrides": copy.deepcopy(dict(prof.get("overrides", {}) or {})),
                "source": f"phase9_result:{run_phase}",
            }
        )

    for base_id in ("B1", "B2", "B3", "B4"):
        if base_id in winners:
            continue
        candidates = by_base.get(base_id, [])
        if candidates:
            candidates.sort(
                key=lambda x: (
                    float(x["best_mrr"]),
                    float(x["test_mrr"]),
                    str(x["concept_id"]),
                    str(x["combo_id"]),
                ),
                reverse=True,
            )
            winners[base_id] = candidates[0]
            continue
        if allow_fallback_winner:
            fallback = profiles["C0_N1"]
            winners[base_id] = {
                "base_id": base_id,
                "concept_id": "C0",
                "combo_id": "N1",
                "main_aux": str(fallback["main_aux"]),
                "support_aux": str(fallback["support_aux"]),
                "scenario": str(fallback["scenario"]),
                "overrides": copy.deepcopy(dict(fallback.get("overrides", {}) or {})),
                "source": "fallback:C0_N1",
            }
            continue
        raise RuntimeError(
            f"No winner candidate found for {base_id}. "
            "Run phase9 first or provide --winner-map-json / --allow-fallback-winner."
        )

    return winners


def _run_phase_name(base_id: str, concept_id: str, combo_id: str, hvar_id: str, seed_id: int) -> str:
    return f"P9_2_{base_id}_{concept_id}{combo_id}_{hvar_id}_S{int(seed_id)}"


def _run_id(base_id: str, concept_id: str, combo_id: str, hvar_id: str, seed_id: int) -> str:
    return f"{base_id}_{concept_id}{combo_id}_{hvar_id}_S{int(seed_id)}"


def _build_rows(args: argparse.Namespace, winners: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")
    hvars = _hparam_variants()
    bases = _base_definitions()
    base_allow = {tok.upper() for tok in _parse_csv_strings(args.only_base)}

    rows: list[Dict[str, Any]] = []
    seed_cursor = 0
    for base_id in ("B1", "B2", "B3", "B4"):
        if base_allow and base_id not in base_allow:
            continue
        winner = winners[base_id]
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
                for key, value in dict(winner.get("overrides", {}) or {}).items():
                    overrides[str(key)] = value
                run_phase = _run_phase_name(
                    base_id=base_id,
                    concept_id=str(winner["concept_id"]),
                    combo_id=str(winner["combo_id"]),
                    hvar_id=hvar_id,
                    seed_id=seed_id,
                )
                run_id = _run_id(
                    base_id=base_id,
                    concept_id=str(winner["concept_id"]),
                    combo_id=str(winner["combo_id"]),
                    hvar_id=hvar_id,
                    seed_id=seed_id,
                )
                rows.append(
                    {
                        "dataset": args.dataset,
                        "base_id": base_id,
                        "hvar_id": hvar_id,
                        "hparams": h,
                        "seed_id": int(seed_id),
                        "seed_offset": int(seed_cursor),
                        "run_id": run_id,
                        "run_phase": run_phase,
                        "winner_concept_id": str(winner["concept_id"]).upper(),
                        "winner_combo_id": str(winner["combo_id"]).upper(),
                        "winner_main_aux": str(winner["main_aux"]),
                        "winner_support_aux": str(winner["support_aux"]),
                        "winner_source": str(winner["source"]),
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
        f"++p9_2_base_id={hydra_literal(row['base_id'])}",
        f"++p9_2_hvar_id={hydra_literal(row['hvar_id'])}",
        f"++p9_2_winner_concept_id={hydra_literal(row['winner_concept_id'])}",
        f"++p9_2_winner_combo_id={hydra_literal(row['winner_combo_id'])}",
        f"++p9_2_winner_main_aux={hydra_literal(row['winner_main_aux'])}",
        f"++p9_2_winner_support_aux={hydra_literal(row['winner_support_aux'])}",
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
            f"run_id={row['run_id']} run_phase={row['run_phase']} "
            f"base={row['base_id']} hvar={row['hvar_id']} seed={row['seed_id']}"
        ),
        (
            f"winner={row['winner_concept_id']}_{row['winner_combo_id']} "
            f"main_aux={row['winner_main_aux']} support_aux={row['winner_support_aux']}"
        ),
        f"winner_source={row['winner_source']}",
        f"dataset={row['dataset']} gpu={gpu_id}",
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={int(args.seed_base) + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _write_matrix_manifest(rows: list[Dict[str, Any]], winners: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> Path:
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
        "winners": {
            base_id: {
                "concept_id": str(rec["concept_id"]),
                "combo_id": str(rec["combo_id"]),
                "main_aux": str(rec["main_aux"]),
                "support_aux": str(rec["support_aux"]),
                "source": str(rec["source"]),
            }
            for base_id, rec in winners.items()
        },
        "n_rows": len(rows),
        "rows": [
            {
                "run_id": r["run_id"],
                "run_phase": r["run_phase"],
                "base_id": r["base_id"],
                "hvar_id": r["hvar_id"],
                "seed_id": r["seed_id"],
                "winner_concept_id": r["winner_concept_id"],
                "winner_combo_id": r["winner_combo_id"],
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
                f"base={row['base_id']} winner={row['winner_concept_id']}_{row['winner_combo_id']} hvar={row['hvar_id']} -> {lp}"
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
                f"({row['base_id']}/{row['winner_concept_id']}{row['winner_combo_id']}/{row['hvar_id']})"
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
    parser = argparse.ArgumentParser(description="FMoE_N3 Phase9_2 verification launcher")
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

    parser.add_argument("--only-base", default="", help="Comma-separated subset of {B1,B2,B3,B4}")
    parser.add_argument("--winner-map-json", default="", help='Optional JSON map like {"B1":"C0_N1",...}')
    parser.add_argument("--allow-fallback-winner", action="store_true")

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

    manual_map = _load_winner_map_json(args.winner_map_json)
    winners = _select_winners(
        dataset=args.dataset,
        manual_map=manual_map,
        allow_fallback_winner=bool(args.allow_fallback_winner),
    )
    rows = _build_rows(args, winners)
    if args.smoke_test:
        rows = rows[: max(int(args.smoke_max_runs), 1)]
    if not rows:
        raise SystemExit("No rows matched filters")

    manifest_path = _write_matrix_manifest(rows, winners, args)
    print(f"[phase9_2] dataset={args.dataset} total_rows={len(rows)} manifest={manifest_path}")
    return _launch_rows(rows=rows, gpus=gpus, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
