#!/usr/bin/env python3
"""AI502 group1 best-hparam transfer launcher.

기존 FMoE/RouteRec 로그에서 target dataset별로 잘 나온 A12 설정과 LR을
그대로 가져와서 group1 transfer를 다시 돈다. 핵심 원칙은 source와 target을
같은 target 설정으로 학습해 tensor shape를 맞추는 것이다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import run_ai502_transfer as base


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = base.EXP_DIR
REPO_ROOT = base.REPO_ROOT
HYPEROPT = base.HYPEROPT
PYTHON_BIN = base.PYTHON_BIN

ARTIFACT_DIR = THIS_DIR / "artifacts_group1_best"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
LOG_DIR = ARTIFACT_DIR / "logs"
MANIFEST_DIR = ARTIFACT_DIR / "manifests"
RESULTS_DIR = ARTIFACT_DIR / "hyperopt_results"
LOGGING_DIR = ARTIFACT_DIR / "logging"
ANALYSIS_DIR = ARTIFACT_DIR / "analysis"
DATA_ROOT = Path(os.environ.get("AI502_DATA_ROOT", "/workspace/RouteRec/Datasets/processed/feature_added_v4"))


TARGET_SETTINGS: dict[str, dict[str, dict[str, Any]]] = {
    "beauty": {
        "beauty_ab_h13_low_feat_dropout": {
            "source_log": "fmoe_n3/Final_tuning_A12/amazon_beauty/AB_H13_low_feat_dropout",
            "prior_best_mrr20": 0.1232,
            "prior_test_mrr20": 0.0851,
            "params": {
                "embedding_size": 176,
                "hidden_size": 176,
                "d_ff": 352,
                "d_expert_hidden": 176,
                "d_router_hidden": 88,
                "d_feat_emb": 8,
                "MAX_ITEM_LIST_LENGTH": 40,
                "num_heads": 4,
                "expert_scale": 3,
                "hidden_dropout_prob": 0.16,
                "attn_dropout_prob": 0.10,
                "weight_decay": 1.0e-6,
                "learning_rate": 0.00135,
                "lr_scheduler_type": "warmup_cosine",
                "route_consistency_lambda": 0.0008,
                "z_loss_lambda": 0.0002,
                "stage_family_dropout_prob": {"macro": 0.04, "mid": 0.04, "micro": 0.04},
                "stage_feature_dropout_prob": {"macro": 0.02, "mid": 0.02, "micro": 0.02},
            },
        },
        "beauty_h13_final": {
            "source_log": "fmoe_n3/Final_all_datasets/amazon_beauty/A12/H13",
            "prior_best_mrr20": 0.1189,
            "prior_test_mrr20": 0.0877,
            "params": {
                "embedding_size": 176,
                "hidden_size": 176,
                "d_ff": 352,
                "d_expert_hidden": 176,
                "d_router_hidden": 88,
                "d_feat_emb": 16,
                "MAX_ITEM_LIST_LENGTH": 20,
                "num_heads": 4,
                "expert_scale": 3,
                "hidden_dropout_prob": 0.10,
                "attn_dropout_prob": 0.10,
                "weight_decay": 1.0e-6,
                "learning_rate": 0.000964234,
                "lr_scheduler_type": "warmup_cosine",
                "route_consistency_lambda": 0.0008,
                "z_loss_lambda": 0.0002,
                "stage_family_dropout_prob": {"macro": 0.10, "mid": 0.10, "micro": 0.10},
                "stage_feature_dropout_prob": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
            },
        },
    },
    "retail_rocket": {
        "retail_r15_h13_width_lr_validate": {
            "source_log": "fmoe_n4/CrossDataset_A12_Portfolio/retail_rocket/R15_h13_width_lr_validate",
            "prior_best_mrr20": 0.3730,
            "prior_test_mrr20": 0.3730,
            "params": {
                "embedding_size": 176,
                "hidden_size": 176,
                "d_ff": 352,
                "d_expert_hidden": 176,
                "d_router_hidden": 88,
                "d_feat_emb": 24,
                "MAX_ITEM_LIST_LENGTH": 20,
                "num_heads": 4,
                "expert_scale": 3,
                "hidden_dropout_prob": 0.16,
                "attn_dropout_prob": 0.08,
                "weight_decay": 1.0e-6,
                "learning_rate": 0.000702904,
                "lr_scheduler_type": "warmup_cosine",
                "route_consistency_lambda": 0.0005,
                "z_loss_lambda": 0.0001,
                "stage_family_dropout_prob": {"macro": 0.02, "mid": 0.02, "micro": 0.02},
                "stage_feature_dropout_prob": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
            },
        },
        "retail_r10_h13_width_refine": {
            "source_log": "fmoe_n4/CrossDataset_A12_Portfolio/retail_rocket/R10_h13_width_refine",
            "prior_best_mrr20": 0.3726,
            "prior_test_mrr20": 0.3737,
            "params": {
                "embedding_size": 176,
                "hidden_size": 176,
                "d_ff": 352,
                "d_expert_hidden": 176,
                "d_router_hidden": 88,
                "d_feat_emb": 24,
                "MAX_ITEM_LIST_LENGTH": 20,
                "num_heads": 4,
                "expert_scale": 3,
                "hidden_dropout_prob": 0.19,
                "attn_dropout_prob": 0.10,
                "weight_decay": 5.0e-7,
                "learning_rate": 0.000692007,
                "lr_scheduler_type": "warmup_cosine",
                "route_consistency_lambda": 0.0005,
                "z_loss_lambda": 0.0001,
                "stage_family_dropout_prob": {"macro": 0.02, "mid": 0.02, "micro": 0.02},
                "stage_feature_dropout_prob": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
            },
        },
    },
    "KuaiRecLargeStrictPosV2_0.2": {
        "kuairec_h14_feature_strong": {
            "source_log": "fmoe_n3/Final_tuning_A12/KuaiRecLargeStrictPosV2_0.2/KU_H14_feature_strong",
            "prior_best_mrr20": 0.1721,
            "prior_test_mrr20": 0.1695,
            "params": {
                "embedding_size": 256,
                "hidden_size": 256,
                "d_ff": 512,
                "d_expert_hidden": 256,
                "d_router_hidden": 128,
                "d_feat_emb": 16,
                "MAX_ITEM_LIST_LENGTH": 20,
                "num_heads": 4,
                "expert_scale": 4,
                "hidden_dropout_prob": 0.05,
                "attn_dropout_prob": 0.10,
                "weight_decay": 5.0e-7,
                "learning_rate": 0.00035,
                "lr_scheduler_type": "warmup_cosine",
                "route_consistency_lambda": 0.0008,
                "z_loss_lambda": 0.0002,
                "stage_family_dropout_prob": {"macro": 0.02, "mid": 0.02, "micro": 0.02},
                "stage_feature_dropout_prob": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
            },
        },
        "kuairec_h10_long_context": {
            "source_log": "fmoe_n3/Final_tuning_A12/KuaiRecLargeStrictPosV2_0.2/KU_H10_long_context",
            "prior_best_mrr20": 0.1706,
            "prior_test_mrr20": 0.1684,
            "params": {
                "embedding_size": 192,
                "hidden_size": 192,
                "d_ff": 384,
                "d_expert_hidden": 192,
                "d_router_hidden": 96,
                "d_feat_emb": 16,
                "MAX_ITEM_LIST_LENGTH": 30,
                "num_heads": 4,
                "expert_scale": 3,
                "hidden_dropout_prob": 0.14,
                "attn_dropout_prob": 0.10,
                "weight_decay": 8.0e-7,
                "learning_rate": 0.0008,
                "lr_scheduler_type": "warmup_cosine",
                "route_consistency_lambda": 0.0008,
                "z_loss_lambda": 0.0002,
                "stage_family_dropout_prob": {"macro": 0.06, "mid": 0.06, "micro": 0.06},
                "stage_feature_dropout_prob": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
            },
        },
    },
}

GROUP1_PAIRS = [
    {
        "pair_id": "retail_rocket_to_beauty",
        "source": "retail_rocket",
        "target": "beauty",
        "settings": ["beauty_ab_h13_low_feat_dropout", "beauty_h13_final"],
    },
    {
        "pair_id": "beauty_to_retail_rocket",
        "source": "beauty",
        "target": "retail_rocket",
        "settings": ["retail_r15_h13_width_lr_validate", "retail_r10_h13_width_refine"],
    },
    {
        "pair_id": "foursquare_to_KuaiRec",
        "source": "foursquare",
        "target": "KuaiRecLargeStrictPosV2_0.2",
        "settings": ["kuairec_h14_feature_strong", "kuairec_h10_long_context"],
    },
    {
        "pair_id": "lastfm_to_KuaiRec",
        "source": "lastfm0.03",
        "target": "KuaiRecLargeStrictPosV2_0.2",
        "settings": ["kuairec_h14_feature_strong", "kuairec_h10_long_context"],
    },
]

MODES = [
    "feature_encoder_init",
    "group_router_init",
    "feature_encoder_group_router_init",
    "feature_encoder_a12_router_init",
    "full_except_feature_router_init",
    "full_model_init",
]

POLICIES = {
    "std": {"loaded_lr_scale": 1.0, "freeze_loaded": False},
    "loaded_lr_0.35": {"loaded_lr_scale": 0.35, "freeze_loaded": False},
    "loaded_lr_0.05": {"loaded_lr_scale": 0.05, "freeze_loaded": False},
    "freeze_loaded": {"loaded_lr_scale": 1.0, "freeze_loaded": True},
}


def token(value: str) -> str:
    return base.token(value)


def setting_for(target: str, setting_id: str) -> dict[str, Any]:
    return TARGET_SETTINGS[target][setting_id]


def setting_params(target: str, setting_id: str) -> dict[str, Any]:
    return dict(setting_for(target, setting_id)["params"])


def setting_lr(target: str, setting_id: str) -> float:
    return float(setting_params(target, setting_id)["learning_rate"])


def native_checkpoint(dataset: str, target: str, setting_id: str, seed: int) -> Path:
    return CHECKPOINT_DIR / "native" / base.dataset_short(dataset) / base.dataset_short(target) / setting_id / f"seed_{seed}" / "best.pth"


def transfer_checkpoint(pair_id: str, setting_id: str, seed: int, mode: str, policy: str) -> Path:
    return CHECKPOINT_DIR / "transfer" / pair_id / setting_id / f"seed_{seed}" / mode / policy / "best.pth"


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(value: str | None) -> list[int]:
    return [int(item) for item in parse_csv(value, ["1", "2", "3", "4", "5"])]


def pair_specs(pair_filter: set[str] | None = None) -> list[dict[str, Any]]:
    if not pair_filter:
        return list(GROUP1_PAIRS)
    return [spec for spec in GROUP1_PAIRS if spec["pair_id"] in pair_filter]


def native_requirements(seeds: list[int], pairs: list[dict[str, Any]]) -> list[tuple[str, str, str, int]]:
    req: set[tuple[str, str, str, int]] = set()
    for spec in pairs:
        for setting_id in spec["settings"]:
            for dataset in (spec["source"], spec["target"]):
                for seed in seeds:
                    req.add((dataset, spec["target"], setting_id, seed))
    return sorted(req, key=lambda x: (base.dataset_short(x[1]), x[2], base.dataset_short(x[0]), x[3]))


def build_native_rows(seeds: list[int], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, target, setting_id, seed in native_requirements(seeds, pairs):
        ckpt = native_checkpoint(dataset, target, setting_id, seed)
        rows.append(
            {
                "phase": "native",
                "dataset": dataset,
                "target_dataset": target,
                "source_dataset": "",
                "pair_id": "",
                "setting_id": setting_id,
                "seed": seed,
                "transfer_mode": "none",
                "backend_transfer_mode": "none",
                "policy": "scratch",
                "source_checkpoint": "",
                "baseline_native_checkpoint": str(ckpt),
                "checkpoint_export": str(ckpt),
                "lr_values": [setting_lr(target, setting_id)],
                "params": setting_params(target, setting_id),
                "job_key": f"native/{base.dataset_short(dataset)}/as_{base.dataset_short(target)}/{setting_id}/seed_{seed}",
                "run_phase": f"AI502_G1_NATIVE_{token(base.dataset_short(dataset))}_AS_{token(base.dataset_short(target))}_{token(setting_id)}_s{seed}",
            }
        )
    return rows


def build_transfer_rows(
    seeds: list[int],
    pairs: list[dict[str, Any]],
    modes: list[str],
    policies: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for spec in pairs:
        source = spec["source"]
        target = spec["target"]
        for setting_id in spec["settings"]:
            for seed in seeds:
                for mode in modes:
                    for policy in policies:
                        ckpt = transfer_checkpoint(spec["pair_id"], setting_id, seed, mode, policy)
                        rows.append(
                            {
                                "phase": "transfer",
                                "dataset": target,
                                "target_dataset": target,
                                "source_dataset": source,
                                "pair_id": spec["pair_id"],
                                "setting_id": setting_id,
                                "seed": seed,
                                "transfer_mode": mode,
                                "backend_transfer_mode": base.MODE_BACKEND[mode],
                                "policy": policy,
                                "source_checkpoint": str(native_checkpoint(source, target, setting_id, seed)),
                                "baseline_native_checkpoint": str(native_checkpoint(target, target, setting_id, seed)),
                                "checkpoint_export": str(ckpt),
                                "lr_values": [setting_lr(target, setting_id)],
                                "params": setting_params(target, setting_id),
                                "job_key": f"transfer/{spec['pair_id']}/{setting_id}/seed_{seed}/{mode}/{policy}",
                                "run_phase": f"AI502_G1_T_{token(spec['pair_id'])}_{token(setting_id)}_{token(mode)}_{token(policy)}_s{seed}",
                            }
                        )
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seeds(args.seeds)
    pairs = pair_specs(set(parse_csv(args.pairs, [])) or None)
    modes = parse_csv(args.modes, MODES)
    policies = parse_csv(args.policies, ["std", "loaded_lr_0.35", "loaded_lr_0.05"])
    unknown_modes = [mode for mode in modes if mode not in MODES]
    unknown_policies = [policy for policy in policies if policy not in POLICIES]
    if unknown_modes:
        raise SystemExit(f"알 수 없는 transfer mode: {unknown_modes}")
    if unknown_policies:
        raise SystemExit(f"알 수 없는 policy: {unknown_policies}")
    if args.phase == "native":
        return build_native_rows(seeds, pairs)
    if args.phase == "transfer":
        return build_transfer_rows(seeds, pairs, modes, policies)
    raise SystemExit(f"지원하지 않는 phase: {args.phase}")


def write_manifest(rows: list[dict[str, Any]], phase: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{phase}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    fields = [
        "phase",
        "dataset",
        "source_dataset",
        "target_dataset",
        "pair_id",
        "setting_id",
        "seed",
        "transfer_mode",
        "backend_transfer_mode",
        "policy",
        "source_checkpoint",
        "baseline_native_checkpoint",
        "checkpoint_export",
        "lr_values",
        "job_key",
        "run_phase",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key, "") for key in fields}
            item["lr_values"] = json.dumps(row.get("lr_values", []), ensure_ascii=False)
            writer.writerow(item)
    return path


def result_root() -> Path:
    return RESULTS_DIR / "ai502_transfer_group1_best"


def completed_run_phases() -> set[str]:
    out: set[str] = set()
    for path in sorted(result_root().glob("*.json")):
        if path.name.endswith("_special_metrics.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trials = payload.get("trials") if isinstance(payload, dict) else []
        if payload.get("run_phase") and any((trial or {}).get("status") == "ok" for trial in trials):
            out.add(str(payload["run_phase"]))
    return out


def log_completed_cleanly(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    if any(x in text for x in ("Traceback", "RuntimeError", "NameError", "ModuleNotFoundError", "ERROR", "Error:")):
        return False
    tail = "\n".join(text.splitlines()[-8:])
    return "[RUN_STATUS] END status=normal" in tail and "[RUN_STATUS] rc=0" in tail


def build_command(row: dict[str, Any], gpu: str, args: argparse.Namespace) -> list[str]:
    gpu_value: Any = int(gpu) if str(gpu).isdigit() else str(gpu)
    params = dict(row["params"])
    lr = float(row["lr_values"][0])
    overrides: dict[str, Any] = {
        **base.COMMON_FIXED,
        **params,
        "dataset": row["dataset"],
        "gpu_id": gpu_value,
        "seed": int(row["seed"]),
        "data_path": str(DATA_ROOT),
        "run_group": "ai502_transfer_group1_best",
        "run_axis": "group1_best_hparam_transfer",
        "run_phase": row["run_phase"],
        "fmoe_logging_output_root": str(LOGGING_DIR),
        "__artifact_combo_best_export_path": row["checkpoint_export"],
        "baseline_native_checkpoint": row["baseline_native_checkpoint"],
        "search_space_type_overrides": {"learning_rate": "choice"},
        "search": {"learning_rate": [lr]},
        "stage_router_source": base.all_stage_map("both"),
    }
    if row["transfer_mode"] == "none":
        overrides["transfer"] = {"enable": False, "mode": "none"}
    else:
        policy = POLICIES[row["policy"]]
        overrides["transfer"] = {
            "enable": True,
            "mode": row["backend_transfer_mode"],
            "source_checkpoint": row["source_checkpoint"],
            "source_architecture": "A12",
            "strict_shape": False,
            "freeze_loaded": bool(policy["freeze_loaded"]),
            "allow_empty": False,
            "allow_router_fallback": False,
            "loaded_lr_scale": float(policy["loaded_lr_scale"]),
        }

    cmd = [
        str(PYTHON_BIN),
        str(HYPEROPT),
        "--config-name",
        "config",
        "--max-evals",
        "1",
        "--tune-epochs",
        str(args.epochs),
        "--tune-patience",
        str(args.patience),
        "--search-algo",
        "random",
        "--seed",
        str(row["seed"]),
        "--run-group",
        "ai502_transfer_group1_best",
        "--run-axis",
        "group1_best_hparam_transfer",
        "--run-phase",
        row["run_phase"],
    ]
    plain = {"model", "dataset", "gpu_id", "feature_mode", "eval_mode"}
    for key, value in overrides.items():
        cmd.append(f"{'' if key in plain else '++'}{key}={base.hydra_literal(value)}")
    return cmd


def validate_rows(rows: list[dict[str, Any]], check_missing: bool) -> None:
    seen = set()
    for row in rows:
        if row["job_key"] in seen:
            raise SystemExit(f"중복 job_key: {row['job_key']}")
        seen.add(row["job_key"])
        if check_missing and row["source_checkpoint"] and not Path(row["source_checkpoint"]).exists():
            raise SystemExit(f"source checkpoint 없음: {row['source_checkpoint']}")


def write_log_preamble(path: Path, row: dict[str, Any], gpu: str, cmd: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in row.items() if k != "params"}
    with path.open("w", encoding="utf-8") as fh:
        fh.write("[AI502_GROUP1_BEST_ROW] " + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        fh.write(f"[GPU] {gpu}\n")
        fh.write("[COMMAND] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")


def run_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> int:
    if not DATA_ROOT.exists():
        raise SystemExit(f"data root 없음: {DATA_ROOT}")
    validate_rows(rows, check_missing=(not args.dry_run and args.phase == "transfer"))
    manifest = write_manifest(rows, args.phase)
    print(f"[manifest] phase={args.phase} rows={len(rows)} path={manifest}")
    if not rows:
        return 0
    gpus = parse_csv(args.gpus, ["0"])
    queue = deque(rows)
    active: dict[str, dict[str, Any]] = {}
    completed = completed_run_phases() if args.resume else set()
    env = dict(os.environ)
    env["HYPEROPT_RESULTS_DIR"] = str(RESULTS_DIR)
    env["PYTHONPATH"] = os.pathsep.join([str(EXP_DIR), str(REPO_ROOT), env.get("PYTHONPATH", "")])
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            row = queue.popleft()
            log_path = LOG_DIR / row["phase"] / f"{token(row['run_phase'])}.log"
            row["log_path"] = str(log_path)
            checkpoint_exists = Path(row["checkpoint_export"]).exists()
            if args.resume and row["run_phase"] in completed and log_completed_cleanly(log_path) and checkpoint_exists:
                print(f"[skip] clean result/log/checkpoint exists {row['job_key']}")
                continue
            cmd = build_command(row, gpu, args)
            if args.dry_run:
                print(f"[dry-run] gpu={gpu} {row['job_key']} -> {log_path}")
                print("          " + " ".join(shlex.quote(x) for x in cmd))
                continue
            write_log_preamble(log_path, row, gpu, cmd)
            with log_path.open("a", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[gpu] = {"proc": proc, "row": row, "log": log_path}
            print(f"[launch] gpu={gpu} phase={row['phase']} job={row['job_key']}")
        if args.dry_run:
            continue
        done = []
        for gpu, slot in active.items():
            rc = slot["proc"].poll()
            if rc is None:
                continue
            row = slot["row"]
            with Path(slot["log"]).open("a", encoding="utf-8") as fh:
                fh.write(f"\n[RUN_STATUS] rc={rc} job_key={row['job_key']}\n")
            print(f"[done] gpu={gpu} rc={rc} job={row['job_key']} log={slot['log']}")
            done.append(gpu)
            if rc != 0 and not args.keep_going:
                raise SystemExit(rc)
        for gpu in done:
            active.pop(gpu, None)
        if queue or active:
            time.sleep(5)
    return 0


def iter_results() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(result_root().glob("*.json")):
        if path.name.endswith("_special_metrics.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trials = payload.get("trials") if isinstance(payload, dict) else []
        if payload.get("run_phase") and any((trial or {}).get("status") == "ok" for trial in trials):
            payload["_result_path"] = str(path)
            out[str(payload["run_phase"])] = payload
    return out


def load_manifest_rows() -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in sorted(MANIFEST_DIR.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows[row["run_phase"]] = dict(row)
    return list(rows.values())


def result_metric(result: dict[str, Any], name: str) -> float:
    if name == "valid_mrr20":
        return float(result.get("best_mrr@20") or 0.0)
    aliases = {
        "test_mrr20": "mrr@20",
        "test_ndcg20": "ndcg@20",
        "test_recall20": "recall@20",
        "test_hr10": "hit@10",
    }
    test = result.get("test_result") or {}
    return float(test.get(aliases[name], 0.0) or 0.0)


def summarize() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results = iter_results()
    rows = load_manifest_rows()
    enriched = []
    native_mrr: dict[tuple[str, str, str], float] = {}
    for row in rows:
        result = results.get(row["run_phase"])
        if not result:
            continue
        trial = (result.get("trials") or [{}])[0]
        tr = result.get("transfer_report") or {}
        opt = tr.get("optimizer_report") or {}
        item = dict(row)
        item.update(
            valid_mrr20=result_metric(result, "valid_mrr20"),
            test_mrr20=result_metric(result, "test_mrr20"),
            test_ndcg20=result_metric(result, "test_ndcg20"),
            test_recall20=result_metric(result, "test_recall20"),
            test_hr10=result_metric(result, "test_hr10"),
            epochs_run=trial.get("epochs_run", ""),
            early_stopped=trial.get("early_stopped", ""),
            avg_epoch_time_sec=result.get("avg_epoch_time_sec", ""),
            loaded_tensors=tr.get("loaded_tensors", ""),
            skipped_tensors=tr.get("skipped_tensors", ""),
            init_changed_tensors=(tr.get("init_delta_from_target") or {}).get("changed_tensors", ""),
            train_changed_tensors=(tr.get("train_delta_from_init") or {}).get("changed_tensors", ""),
            loaded_lr_scale=opt.get("loaded_lr_scale", ""),
            loaded_lr=opt.get("loaded_lr", ""),
            result_path=result.get("_result_path", ""),
        )
        if item["phase"] == "native":
            native_mrr[(item["dataset"], item["setting_id"], item["seed"])] = float(item["test_mrr20"])
        enriched.append(item)
    for item in enriched:
        if item["phase"] != "transfer":
            item["baseline_test_mrr20"] = ""
            item["gain_mrr20"] = ""
            continue
        key = (item["target_dataset"], item["setting_id"], item["seed"])
        baseline = native_mrr.get(key)
        item["baseline_test_mrr20"] = baseline if baseline is not None else ""
        item["gain_mrr20"] = (float(item["test_mrr20"]) - baseline) if baseline is not None else ""

    full_path = ANALYSIS_DIR / "group1_best_full.csv"
    fields = [
        "phase",
        "source_dataset",
        "target_dataset",
        "pair_id",
        "setting_id",
        "seed",
        "transfer_mode",
        "policy",
        "valid_mrr20",
        "test_mrr20",
        "baseline_test_mrr20",
        "gain_mrr20",
        "test_ndcg20",
        "test_recall20",
        "test_hr10",
        "epochs_run",
        "early_stopped",
        "avg_epoch_time_sec",
        "loaded_tensors",
        "skipped_tensors",
        "init_changed_tensors",
        "train_changed_tensors",
        "loaded_lr_scale",
        "loaded_lr",
        "run_phase",
        "result_path",
    ]
    with full_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in enriched:
            writer.writerow({key: item.get(key, "") for key in fields})

    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for item in enriched:
        if item["phase"] == "transfer" and item.get("gain_mrr20") != "":
            key = (item["pair_id"], item["setting_id"], item["transfer_mode"], item["policy"])
            groups[key].append(float(item["gain_mrr20"]))
    summary_path = ANALYSIS_DIR / "group1_best_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pair_id", "setting_id", "transfer_mode", "policy", "n", "mean_gain_mrr20", "win_rate"])
        writer.writeheader()
        for key, values in sorted(groups.items()):
            writer.writerow(
                {
                    "pair_id": key[0],
                    "setting_id": key[1],
                    "transfer_mode": key[2],
                    "policy": key[3],
                    "n": len(values),
                    "mean_gain_mrr20": sum(values) / len(values),
                    "win_rate": sum(v > 0 for v in values) / len(values),
                }
            )

    md_path = THIS_DIR / "result_group1_best.md"
    best_rows = []
    for key, values in sorted(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)[:20]:
        best_rows.append((key, len(values), sum(values) / len(values), sum(v > 0 for v in values) / len(values)))
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# AI502 group1 best-hparam 결과 요약\n\n")
        fh.write("- 기준: target native scratch와 같은 `setting_id`, 같은 seed의 test MRR@20 비교\n")
        fh.write("- full CSV: `artifacts_group1_best/analysis/group1_best_full.csv`\n")
        fh.write("- 집계 CSV: `artifacts_group1_best/analysis/group1_best_summary.csv`\n\n")
        fh.write("## 상위 gain 조합\n\n")
        fh.write("| pair | setting | mode | policy | n | mean gain | win-rate |\n")
        fh.write("|---|---|---|---|---:|---:|---:|\n")
        for key, n, mean_gain, win_rate in best_rows:
            fh.write(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {n} | {mean_gain:.6f} | {win_rate:.2f} |\n")
    print(f"[summary] full={full_path}")
    print(f"[summary] grouped={summary_path}")
    print(f"[summary] md={md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI502 group1 best-hparam transfer launcher")
    parser.add_argument("--phase", choices=["native", "transfer", "summary", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--policies", default="std,loaded_lr_0.35,loaded_lr_0.05")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.phase == "summary":
        summarize()
        return 0
    if args.phase == "all":
        for phase in ("native", "transfer"):
            phase_args = argparse.Namespace(**vars(args))
            phase_args.phase = phase
            rows = build_rows(phase_args)
            run_rows(rows, phase_args)
            if not args.dry_run:
                summarize()
        return 0
    return run_rows(build_rows(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
