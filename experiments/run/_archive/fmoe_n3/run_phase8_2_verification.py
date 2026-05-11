#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 Phase8_2 verification runs (4x4x4 grid)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path("/workspace/jy1559/FMoE")
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"

TRACK = "fmoe_n3"
AXIS = "phase8_2_verification_v1"
PHASE = "P8_2"
MODEL_TAG = "FMoEN3"

STAGES = ("macro", "mid", "micro")
PRIMITIVES = ("a_joint", "b_group", "c_shared", "d_cond", "e_scalar")


def hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Invalid float for hydra literal: {value}")
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = [f"{k}:{hydra_literal(v)}" for k, v in value.items()]
        return "{" + ",".join(items) + "}"
    raise TypeError(f"Unsupported hydra literal type: {type(value).__name__}")


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _parse_csv_ints(text: str) -> list[int]:
    return [int(tok.strip()) for tok in str(text or "").split(",") if tok.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [tok.strip() for tok in str(text or "").split(",") if tok.strip()]


def _dataset_tag(dataset: str) -> str:
    return str(dataset).replace("/", "_")


def _primitive_default_sources() -> Dict[str, str]:
    return {
        "a_joint": "both",
        "b_group": "both",
        "c_shared": "both",
        "d_cond": "feature",
        "e_scalar": "feature",
    }


def _build_stage_router_primitives(
    *,
    sources: Optional[Dict[str, str]] = None,
    temperatures: Optional[Dict[str, float]] = None,
    top_ks: Optional[Dict[str, int]] = None,
    wrapper_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    base_sources = _primitive_default_sources()
    if isinstance(sources, dict):
        for key, val in sources.items():
            if key in PRIMITIVES and val is not None:
                base_sources[key] = str(val)

    base_temps = {key: 1.0 for key in PRIMITIVES}
    if isinstance(temperatures, dict):
        for key, val in temperatures.items():
            if key in PRIMITIVES and val is not None:
                base_temps[key] = float(val)

    base_topk = {key: 0 for key in PRIMITIVES}
    if isinstance(top_ks, dict):
        for key, val in top_ks.items():
            if key in PRIMITIVES and val is not None:
                base_topk[key] = int(val)

    wrapper_cfg = {"alpha_d": 1.0, "alpha_struct": 1.0, "alpha_a": 1.0}
    if isinstance(wrapper_params, dict):
        for key, val in wrapper_params.items():
            wrapper_cfg[str(key)] = float(val)

    out: Dict[str, Dict[str, Any]] = {}
    for stage_name in STAGES:
        stage_cfg: Dict[str, Any] = {}
        for primitive in PRIMITIVES:
            stage_cfg[primitive] = {
                "source": str(base_sources[primitive]),
                "temperature": float(base_temps[primitive]),
                "top_k": int(base_topk[primitive]),
            }
        stage_cfg["wrapper"] = dict(wrapper_cfg)
        out[stage_name] = stage_cfg
    return out


def _copy_stage_router_primitives(primitives: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(primitives or {}))


def _set_stage_specific_wrapper(overrides: Dict[str, Any], wrapper_map: Dict[str, str]) -> None:
    stage_wrapper = {}
    for stage in STAGES:
        stage_wrapper[stage] = str(wrapper_map.get(stage, "w1_flat"))
    overrides["stage_router_wrapper"] = stage_wrapper


def _update_primitive_sources(primitives: Dict[str, Any], source_map: Dict[str, str]) -> Dict[str, Any]:
    out = _copy_stage_router_primitives(primitives)
    for stage in STAGES:
        stage_cfg = dict(out.get(stage, {}))
        for primitive, source in source_map.items():
            if primitive not in PRIMITIVES:
                continue
            raw = dict(stage_cfg.get(primitive, {}))
            raw["source"] = str(source)
            stage_cfg[primitive] = raw
        out[stage] = stage_cfg
    return out


def _update_primitive_top_k(primitives: Dict[str, Any], top_k_map: Dict[str, int]) -> Dict[str, Any]:
    out = _copy_stage_router_primitives(primitives)
    for stage in STAGES:
        stage_cfg = dict(out.get(stage, {}))
        for primitive in PRIMITIVES:
            raw = dict(stage_cfg.get(primitive, {}))
            raw["top_k"] = int(top_k_map.get(primitive, 0))
            stage_cfg[primitive] = raw
        out[stage] = stage_cfg
    return out


def _base_fixed_overrides() -> Dict[str, Any]:
    return {
        "layer_layout": ["macro", "mid", "micro"],
        "stage_compute_mode": _all_stage_map("moe"),
        "stage_router_mode": _all_stage_map("learned"),
        "stage_router_source": _all_stage_map("both"),
        "stage_feature_injection": _all_stage_map("none"),
        "topk_scope_mode": "global_flat",
        "moe_top_k": 0,
        "balance_loss_lambda": 0.002,
        "z_loss_lambda": 1e-4,
        "route_smoothness_lambda": 0.01,
        "route_consistency_lambda": 0.0,
        "route_sharpness_lambda": 0.0,
        "route_monopoly_lambda": 0.0,
        "route_monopoly_tau": 0.25,
        "route_prior_lambda": 0.0,
        "group_prior_align_lambda": 0.0,
        "factored_group_balance_lambda": 0.0,
        "feature_group_bias_lambda": 0.0,
        "rule_bias_scale": 0.0,
        "stage_router_primitives": _build_stage_router_primitives(),
    }


def _source_profiles() -> Dict[str, Dict[str, str]]:
    return {
        "src_base": {
            "a_joint": "both",
            "b_group": "both",
            "c_shared": "both",
            "d_cond": "feature",
            "e_scalar": "feature",
        },
        "src_abc_feature": {
            "a_joint": "feature",
            "b_group": "feature",
            "c_shared": "feature",
            "d_cond": "feature",
            "e_scalar": "feature",
        },
    }


def _wrapper_map_for_base(base_id: str) -> Dict[str, str]:
    if base_id == "A":
        return {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"}
    if base_id == "B":
        return {"macro": "w4_bxd", "mid": "w6_bxd_plus_a", "micro": "w1_flat"}
    if base_id == "C":
        return {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"}
    if base_id == "D":
        return {"macro": "w2_a_plus_d", "mid": "w2_a_plus_d", "micro": "w2_a_plus_d"}
    raise ValueError(f"Unknown base_id: {base_id}")


def _source_profile_for_base(base_id: str) -> str:
    if base_id == "A":
        return "src_abc_feature"
    return "src_base"


def _bias_for_base(base_id: str, feature_group_bias_lambda: float, rule_bias_scale: float) -> tuple[float, float, str]:
    if base_id == "A":
        return 0.0, float(rule_bias_scale), "bias_rule"
    if base_id == "B":
        return float(feature_group_bias_lambda), 0.0, "bias_group_feat"
    if base_id in {"C", "D"}:
        return 0.0, 0.0, "bias_off"
    raise ValueError(f"Unknown base_id: {base_id}")


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


def _base_descriptions() -> Dict[str, str]:
    return {
        "A": "B1_phasewise_combo(all_w5+bias_rule+src_abc_feature+tk_dense)",
        "B": "B2_best_learned_mixed2(mixed_2+bias_group_feat+src_base+tk_dense)",
        "C": "B3_clean_all_w5(all_w5+bias_off+src_base+tk_dense)",
        "D": "B4_clean_all_w2(all_w2+bias_off+src_base+tk_dense)",
    }


def _phase82_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset) / MODEL_TAG


def _run_phase_name(base_id: str, hvar_id: str, seed_id: int) -> str:
    return f"P8_2_{base_id}_{hvar_id}_S{int(seed_id)}"


def _run_id(base_id: str, hvar_id: str, seed_id: int) -> str:
    return f"{base_id}_{hvar_id}_S{int(seed_id)}"


def _is_completed_log(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "[RUN_STATUS]" in text


def _log_path(row: Dict[str, Any], dataset: str) -> Path:
    root = _phase82_log_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{row['run_id']}.log"


def _build_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")
    hvars = _hparam_variants()
    base_desc = _base_descriptions()
    source_profiles = _source_profiles()

    base_order = ["A", "B", "C", "D"]  # A/B first, then C/D.
    seed_cursor = 0
    for base_id in base_order:
        for hvar_id in ["H1", "H2", "H3", "H4"]:
            for seed_id in seeds:
                overrides = _base_fixed_overrides()
                _set_stage_specific_wrapper(overrides, _wrapper_map_for_base(base_id))

                feat_lambda, rule_scale, bias_mode = _bias_for_base(
                    base_id=base_id,
                    feature_group_bias_lambda=float(args.feature_group_bias_lambda),
                    rule_bias_scale=float(args.rule_bias_scale),
                )
                overrides["feature_group_bias_lambda"] = float(feat_lambda)
                overrides["rule_bias_scale"] = float(rule_scale)

                source_profile = _source_profile_for_base(base_id)
                source_map = source_profiles[source_profile]
                primitives = _update_primitive_sources(overrides.get("stage_router_primitives", {}), source_map)
                primitives = _update_primitive_top_k(primitives, {p: 0 for p in PRIMITIVES})
                overrides["stage_router_primitives"] = primitives
                overrides["moe_top_k"] = 0

                run_id = _run_id(base_id=base_id, hvar_id=hvar_id, seed_id=seed_id)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "base_id": base_id,
                        "hvar_id": hvar_id,
                        "seed_id": int(seed_id),
                        "seed_offset": int(seed_cursor),
                        "run_id": run_id,
                        "run_phase": _run_phase_name(base_id=base_id, hvar_id=hvar_id, seed_id=seed_id),
                        "setting_desc": base_desc[base_id],
                        "source_profile": source_profile,
                        "bias_mode": bias_mode,
                        "topk_profile": "tk_dense",
                        "hparams": dict(hvars[hvar_id]),
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
        f"++p8_2_base_id={hydra_literal(row['base_id'])}",
        f"++p8_2_hvar_id={hydra_literal(row['hvar_id'])}",
        f"++p8_2_run_id={hydra_literal(row['run_id'])}",
        f"++p8_2_setting_desc={hydra_literal(row['setting_desc'])}",
        f"++p8_2_bias_mode={hydra_literal(row['bias_mode'])}",
        f"++p8_2_source_profile={hydra_literal(row['source_profile'])}",
    ]
    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    lines = [
        "[PHASE8_2_SETTING_HEADER]",
        f"run_id={row['run_id']} run_phase={row['run_phase']} base={row['base_id']} hvar={row['hvar_id']} seed={row['seed_id']}",
        f"desc={row['setting_desc']}",
        f"bias_mode={row['bias_mode']} source_profile={row['source_profile']} topk={row['topk_profile']}",
        f"dataset={row['dataset']} gpu={gpu_id}",
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={int(args.seed_base) + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    # Overwrite stale/incomplete log by design.
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _write_matrix_manifest(rows: list[Dict[str, Any]], args: argparse.Namespace) -> None:
    out_path = _phase82_log_dir(args.dataset).parent / "verification_matrix.json"
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE,
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_rows": len(rows),
        "rows": [
            {
                "run_id": r["run_id"],
                "run_phase": r["run_phase"],
                "base_id": r["base_id"],
                "hvar_id": r["hvar_id"],
                "seed_id": r["seed_id"],
                "bias_mode": r["bias_mode"],
                "source_profile": r["source_profile"],
                "topk_profile": r["topk_profile"],
                "hparams": r["hparams"],
            }
            for r in rows
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _launch_rows(rows: list[Dict[str, Any]], gpus: list[str], args: argparse.Namespace) -> int:
    if not rows:
        return 0

    for idx, row in enumerate(rows):
        row["assigned_gpu"] = gpus[idx % len(gpus)]
        row["assigned_order"] = idx + 1

    runnable: list[Dict[str, Any]] = []
    for row in rows:
        lp = _log_path(row, args.dataset)
        if lp.exists() and _is_completed_log(lp):
            print(f"[skip] completed log exists: {lp.name}")
            continue
        runnable.append(row)

    if not runnable:
        print("[phase8_2] all runs are already completed by log marker.")
        return 0

    if args.dry_run:
        for row in runnable:
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, row["assigned_gpu"], args)
            print(f"[dry-run] gpu={row['assigned_gpu']} run={row['run_id']} -> {lp}")
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
            print(f"[launch] gpu={gpu_id} run={row['run_id']}")

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

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = any(gpu_bins[g] for g in gpus)
        if not pending and not active:
            break
        time.sleep(3)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Phase8_2 verification launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--seed-base", type=int, default=38000)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)

    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals = 1
    args.tune_epochs = 1
    args.tune_patience = 1
    args.seeds = "1"


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided")

    rows = _build_rows(args)
    _write_matrix_manifest(rows, args)
    print(f"[phase8_2] dataset={args.dataset} total_rows={len(rows)}")
    return _launch_rows(rows=rows, gpus=gpus, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
