#!/usr/bin/env python3
"""AI502 transfer follow-up 실험 launcher.

기존 `artifacts/checkpoints/native`의 native checkpoint bank를 재사용하고,
필요한 seed/hparam native만 top-up한 뒤 해석 가능한 transfer 축을 더
신뢰성 있게 확인한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import run_ai502_transfer as base


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = base.EXP_DIR
REPO_ROOT = base.REPO_ROOT
HYPEROPT = base.HYPEROPT
PYTHON_BIN = base.PYTHON_BIN

SOURCE_ARTIFACT_DIR = THIS_DIR / "artifacts"
SOURCE_NATIVE_CKPT_DIR = SOURCE_ARTIFACT_DIR / "checkpoints" / "native"
REAL_ARTIFACT_DIR = THIS_DIR / "artifacts_real"
REAL_CHECKPOINT_DIR = REAL_ARTIFACT_DIR / "checkpoints"
REAL_LOG_DIR = REAL_ARTIFACT_DIR / "logs"
REAL_MANIFEST_DIR = REAL_ARTIFACT_DIR / "manifests"
REAL_RESULTS_DIR = REAL_ARTIFACT_DIR / "hyperopt_results"
REAL_LOGGING_DIR = REAL_ARTIFACT_DIR / "logging"
REAL_ANALYSIS_DIR = REAL_ARTIFACT_DIR / "analysis"

REAL_PAIRS = [
    {
        "pair_id": "retail_rocket_to_beauty",
        "source": "retail_rocket",
        "target": "beauty",
        "hparams": ["shared_3", "shared_6"],
        "note": "commerce target에서 cue/router transfer가 가장 선명했던 축",
    },
    {
        "pair_id": "beauty_to_retail_rocket",
        "source": "beauty",
        "target": "retail_rocket",
        "hparams": ["shared_3", "shared_6"],
        "note": "commerce close pair, full/full-except와 cue transfer 비교",
    },
    {
        "pair_id": "foursquare_to_KuaiRec",
        "source": "foursquare",
        "target": "KuaiRecLargeStrictPosV2_0.2",
        "hparams": ["shared_4", "shared_5"],
        "note": "rich-context target에서 partial transfer 안정성 확인",
    },
    {
        "pair_id": "lastfm_to_KuaiRec",
        "source": "lastfm0.03",
        "target": "KuaiRecLargeStrictPosV2_0.2",
        "hparams": ["shared_3", "shared_5"],
        "note": "rich-context negative/contrast pair",
    },
]

REAL_TRIPLETS = [
    {
        "triplet_id": "beauty_to_retail_rocket_to_KuaiRec",
        "a": "beauty",
        "b": "retail_rocket",
        "c": "KuaiRecLargeStrictPosV2_0.2",
        "hparams": ["shared_3", "shared_5"],
    },
]

REAL_MODES = [
    "feature_encoder_init",
    "group_router_init",
    "feature_encoder_group_router_init",
    "feature_encoder_a12_router_init",
    "full_except_feature_router_init",
    "full_model_init",
]

FULL_POLICY_MODES = [
    "feature_encoder_a12_router_init",
    "full_except_feature_router_init",
    "full_model_init",
]

POLICIES = {
    "std": {"lr_scale": 1.0, "loaded_lr_scale": 1.0, "modes": REAL_MODES},
    "loaded_lr_0.35": {"lr_scale": 1.0, "loaded_lr_scale": 0.35, "modes": REAL_MODES},
    "global_lr_0.5": {"lr_scale": 0.5, "loaded_lr_scale": 1.0, "modes": FULL_POLICY_MODES},
}

MULTIHOP_MODES = ["feature_encoder_a12_router_init", "full_model_init"]
MULTIHOP_POLICIES = ["std", "loaded_lr_0.35"]


def native_checkpoint(dataset: str, hparam: str, seed: int) -> Path:
    return SOURCE_NATIVE_CKPT_DIR / base.dataset_short(dataset) / hparam / f"seed_{seed}" / "best.pth"


def real_transfer_checkpoint(kind: str, *parts: str) -> Path:
    return REAL_CHECKPOINT_DIR.joinpath(kind, *parts, "best.pth")


def lr_values(dataset: str, policy: str) -> list[float]:
    scale = float(POLICIES.get(policy, {}).get("lr_scale", 1.0))
    return [base.LR_CENTER[dataset] * scale]


def row_common(dataset: str, hparam: str, seed: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_short": base.dataset_short(dataset),
        "hparam_id": hparam,
        "seed": int(seed),
        "preset": base.HPARAM_PRESETS[hparam],
    }


def collect_native_requirements(seeds: list[int]) -> list[tuple[str, str, int]]:
    req: set[tuple[str, str, int]] = set()
    for spec in REAL_PAIRS:
        for dataset in (spec["source"], spec["target"]):
            for hparam in spec["hparams"]:
                for seed in seeds:
                    req.add((dataset, hparam, seed))
    for spec in REAL_TRIPLETS:
        for dataset in (spec["a"], spec["b"], spec["c"]):
            for hparam in spec["hparams"]:
                for seed in seeds:
                    req.add((dataset, hparam, seed))
    return sorted(req, key=lambda x: (base.dataset_short(x[0]), x[1], x[2]))


def build_native_topup_rows(seeds: list[int], *, include_existing: bool = False) -> list[dict[str, Any]]:
    rows = []
    for dataset, hparam, seed in collect_native_requirements(seeds):
        ckpt = native_checkpoint(dataset, hparam, seed)
        if ckpt.exists() and not include_existing:
            continue
        row = row_common(dataset, hparam, seed)
        row.update(
            phase="native_topup",
            source_dataset="",
            target_dataset=dataset,
            pair_id="",
            triplet_id="",
            comparison_role="",
            transfer_mode="none",
            backend_transfer_mode="none",
            policy="scratch",
            job_key=f"native_topup/{base.dataset_short(dataset)}/{hparam}/seed_{seed}",
            run_phase=f"AI502_REAL_NATIVE_{base.token(base.dataset_short(dataset))}_{hparam}_s{seed}",
            checkpoint_export=str(ckpt),
            source_checkpoint="",
            baseline_native_checkpoint=str(ckpt),
            baseline_native_result=base.native_result_key(dataset, hparam, seed),
            lr_values=[base.LR_CENTER[dataset]],
        )
        rows.append(row)
    return rows


def build_pair_rows(seeds: list[int], policies: list[str]) -> list[dict[str, Any]]:
    rows = []
    for spec in REAL_PAIRS:
        source = spec["source"]
        target = spec["target"]
        pid = spec["pair_id"]
        for hparam in spec["hparams"]:
            for seed in seeds:
                for policy in policies:
                    policy_cfg = POLICIES[policy]
                    for mode in policy_cfg["modes"]:
                        row = row_common(target, hparam, seed)
                        ckpt = real_transfer_checkpoint("pair", pid, hparam, f"seed_{seed}", mode, policy)
                        row.update(
                            phase="transfer",
                            source_dataset=source,
                            target_dataset=target,
                            pair_id=pid,
                            triplet_id="",
                            comparison_role="pair",
                            transfer_mode=mode,
                            backend_transfer_mode=base.MODE_BACKEND[mode],
                            policy=policy,
                            job_key=f"transfer/{pid}/{hparam}/seed_{seed}/{mode}/{policy}",
                            run_phase=f"AI502_REAL_T_{base.token(pid)}_{hparam}_{base.token(mode)}_{base.token(policy)}_s{seed}",
                            checkpoint_export=str(ckpt),
                            source_checkpoint=str(native_checkpoint(source, hparam, seed)),
                            baseline_native_checkpoint=str(native_checkpoint(target, hparam, seed)),
                            baseline_native_result=base.native_result_key(target, hparam, seed),
                            lr_values=lr_values(target, policy),
                        )
                        rows.append(row)
    return rows


def build_multihop_rows(seeds: list[int], policies: list[str]) -> list[dict[str, Any]]:
    rows = []
    for spec in REAL_TRIPLETS:
        a, b, c = spec["a"], spec["b"], spec["c"]
        tid = spec["triplet_id"]
        for hparam in spec["hparams"]:
            for seed in seeds:
                for policy in policies:
                    for mode in MULTIHOP_MODES:
                        bridge_ckpt = real_transfer_checkpoint(
                            "multihop_bridge",
                            base.pair_id(a, b),
                            hparam,
                            f"seed_{seed}",
                            mode,
                            policy,
                        )
                        bridge = row_common(b, hparam, seed)
                        bridge.update(
                            phase="multihop_bridge",
                            source_dataset=a,
                            target_dataset=b,
                            pair_id=base.pair_id(a, b),
                            triplet_id=tid,
                            comparison_role="a_to_b_bridge",
                            transfer_mode=mode,
                            backend_transfer_mode=base.MODE_BACKEND[mode],
                            policy=policy,
                            job_key=f"multihop_bridge/{tid}/{hparam}/seed_{seed}/{mode}/{policy}",
                            run_phase=f"AI502_REAL_BRIDGE_{base.token(tid)}_{hparam}_{base.token(mode)}_{base.token(policy)}_s{seed}",
                            checkpoint_export=str(bridge_ckpt),
                            source_checkpoint=str(native_checkpoint(a, hparam, seed)),
                            baseline_native_checkpoint=str(native_checkpoint(b, hparam, seed)),
                            baseline_native_result=base.native_result_key(b, hparam, seed),
                            lr_values=lr_values(b, policy),
                        )
                        rows.append(bridge)
                        for source, role, source_ckpt in [
                            (a, "a_to_c_direct", native_checkpoint(a, hparam, seed)),
                            (b, "b_to_c_direct", native_checkpoint(b, hparam, seed)),
                            (b, "a_to_b_to_c", bridge_ckpt),
                        ]:
                            row = row_common(c, hparam, seed)
                            row.update(
                                phase="multihop",
                                source_dataset=source,
                                target_dataset=c,
                                pair_id=base.pair_id(source, c),
                                triplet_id=tid,
                                comparison_role=role,
                                transfer_mode=mode,
                                backend_transfer_mode=base.MODE_BACKEND[mode],
                                policy=policy,
                                job_key=f"multihop/{tid}/{role}/{hparam}/seed_{seed}/{mode}/{policy}",
                                run_phase=f"AI502_REAL_MH_{base.token(tid)}_{base.token(role)}_{hparam}_{base.token(mode)}_{base.token(policy)}_s{seed}",
                                checkpoint_export=str(real_transfer_checkpoint("multihop", tid, role, hparam, f"seed_{seed}", mode, policy)),
                                source_checkpoint=str(source_ckpt),
                                baseline_native_checkpoint=str(native_checkpoint(c, hparam, seed)),
                                baseline_native_result=base.native_result_key(c, hparam, seed),
                                lr_values=lr_values(c, policy),
                            )
                            rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]], *, check_missing: bool = True) -> None:
    dup = [key for key, count in Counter(row["job_key"] for row in rows).items() if count > 1]
    if dup:
        raise SystemExit("중복 job_key가 있습니다:\n" + "\n".join(dup[:20]))
    planned_exports = {str(row.get("checkpoint_export", "")) for row in rows}
    if not check_missing:
        return
    missing = []
    for row in rows:
        if row["phase"] == "native_topup":
            continue
        for field in ("source_checkpoint", "baseline_native_checkpoint"):
            path = Path(str(row.get(field, "")))
            if not path.exists() and str(path) not in planned_exports:
                missing.append(f"{field} missing: {path} ({row['job_key']})")
    if missing:
        raise SystemExit("필요한 native/bridge checkpoint가 없습니다. native_topup 또는 선행 phase를 먼저 실행하세요.\n" + "\n".join(missing[:30]))


def write_manifest(rows: list[dict[str, Any]], phase: str) -> Path:
    REAL_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = REAL_MANIFEST_DIR / f"{phase}_{int(time.time())}.csv"
    fields = [
        "phase", "job_key", "run_phase", "dataset", "source_dataset", "target_dataset",
        "pair_id", "triplet_id", "comparison_role", "hparam_id", "seed",
        "transfer_mode", "backend_transfer_mode", "policy", "source_checkpoint",
        "baseline_native_checkpoint", "checkpoint_export", "lr_values", "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if key == "lr_values" else row.get(key, "") for key in fields})
    return path


def build_command(row: dict[str, Any], gpu: str, args: argparse.Namespace) -> list[str]:
    gpu_value: Any = int(gpu) if str(gpu).isdigit() else str(gpu)
    overrides: dict[str, Any] = {
        **base.COMMON_FIXED,
        **row["preset"],
        "dataset": row["dataset"],
        "gpu_id": gpu_value,
        "seed": int(row["seed"]),
        "data_path": str(base.DATA_ROOT),
        "run_group": "ai502_transfer_real",
        "run_axis": "transfer_real",
        "run_phase": row["run_phase"],
        "fmoe_logging_output_root": str(REAL_LOGGING_DIR),
        "__artifact_combo_best_export_path": row["checkpoint_export"],
        "baseline_native_checkpoint": row["baseline_native_checkpoint"],
        "baseline_native_result": row["baseline_native_result"],
        "search_space_type_overrides": {"learning_rate": "choice"},
        "search": {"learning_rate": row["lr_values"]},
    }
    if row["transfer_mode"] != "none":
        policy_cfg = POLICIES.get(row["policy"], {})
        overrides["transfer"] = {
            "enable": True,
            "mode": row["backend_transfer_mode"],
            "source_checkpoint": row["source_checkpoint"],
            "source_architecture": "A12",
            "strict_shape": False,
            "freeze_loaded": False,
            "allow_empty": False,
            "allow_router_fallback": False,
            "loaded_lr_scale": float(policy_cfg.get("loaded_lr_scale", 1.0)),
        }
    else:
        overrides["transfer"] = {"enable": False, "mode": "none"}

    cmd = [
        str(PYTHON_BIN), str(HYPEROPT),
        "--config-name", "config",
        "--max-evals", "1",
        "--tune-epochs", str(args.epochs),
        "--tune-patience", str(args.patience),
        "--search-algo", "random",
        "--seed", str(row["seed"]),
        "--run-group", "ai502_transfer_real",
        "--run-axis", "transfer_real",
        "--run-phase", row["run_phase"],
    ]
    plain = {"model", "dataset", "gpu_id", "feature_mode", "eval_mode"}
    for key, value in overrides.items():
        cmd.append(f"{'' if key in plain else '++'}{key}={base.hydra_literal(value)}")
    return cmd


def parse_csv_arg(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(value: str | None) -> list[int]:
    return [int(x) for x in parse_csv_arg(value, ["1", "2", "3", "4", "5"])]


def parse_gpus(value: str | None) -> list[str]:
    return parse_csv_arg(value, ["0"])


def write_log_preamble(log_path: Path, row: dict[str, Any], gpu: str, cmd: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in row.items() if k != "preset"}
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("[AI502_REAL_ROW] " + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        fh.write(f"[GPU] {gpu}\n")
        fh.write("[COMMAND] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")


def log_completed_cleanly(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    bad_patterns = (
        "Traceback",
        "RuntimeError",
        "NameError",
        "ModuleNotFoundError",
        "Error:",
        "ERROR",
        "[RUN_STATUS] rc=1",
        "[RUN_STATUS] rc=2",
        "[RUN_STATUS] rc=3",
        "[RUN_STATUS] rc=4",
        "[RUN_STATUS] rc=5",
        "[RUN_STATUS] rc=6",
        "[RUN_STATUS] rc=7",
        "[RUN_STATUS] rc=8",
        "[RUN_STATUS] rc=9",
    )
    if any(pattern in text for pattern in bad_patterns):
        return False
    tail = "\n".join(text.splitlines()[-8:])
    return "[RUN_STATUS] END status=normal" in tail and "[RUN_STATUS] rc=0" in tail


def completed_run_phases() -> set[str]:
    root = REAL_RESULTS_DIR / "ai502_transfer_real"
    completed: set[str] = set()
    for path in sorted(root.glob("*.json")):
        if path.name.endswith("_special_metrics.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trials = payload.get("trials") if isinstance(payload, dict) else None
        has_ok_trial = any((trial or {}).get("status") == "ok" for trial in (trials or []))
        run_phase = payload.get("run_phase") if isinstance(payload, dict) and has_ok_trial else None
        if run_phase:
            completed.add(str(run_phase))
    return completed


def run_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> int:
    validate_rows(rows, check_missing=not args.dry_run)
    manifest = write_manifest(rows, args.phase)
    print(f"[manifest] phase={args.phase} rows={len(rows)} path={manifest}")
    if not rows:
        return 0
    gpus = parse_gpus(args.gpus)
    queue = deque(rows)
    active: dict[str, dict[str, Any]] = {}
    completed = completed_run_phases() if args.resume else set()
    required_sources = {
        str(row.get("source_checkpoint", ""))
        for row in rows
        if str(row.get("source_checkpoint", "")).strip()
    }
    planned_exports = {
        str(row.get("checkpoint_export", ""))
        for row in rows
        if str(row.get("checkpoint_export", "")).strip()
    }
    env = dict(os.environ)
    env["HYPEROPT_RESULTS_DIR"] = str(REAL_RESULTS_DIR)
    env["PYTHONPATH"] = os.pathsep.join([str(EXP_DIR), str(REPO_ROOT), env.get("PYTHONPATH", "")])
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            row = None
            for _ in range(len(queue)):
                candidate = queue.popleft()
                source_path = str(candidate.get("source_checkpoint", "")).strip()
                waits_for_planned_source = bool(source_path and source_path in planned_exports and not Path(source_path).exists())
                if waits_for_planned_source:
                    queue.append(candidate)
                    continue
                row = candidate
                break
            if row is None:
                if args.dry_run:
                    blocked = [item["job_key"] for item in list(queue)[:20]]
                    print("[dry-run-blocked] 선행 checkpoint 생성 뒤 실행될 job:")
                    for job_key in blocked:
                        print(f"  {job_key}")
                    return 0
                if not active:
                    blocked = [item["job_key"] for item in list(queue)[:10]]
                    raise SystemExit("선행 checkpoint를 기다리는 job만 남았습니다:\n" + "\n".join(blocked))
                break
            checkpoint_path = str(row["checkpoint_export"])
            checkpoint_needed_later = checkpoint_path in required_sources
            checkpoint_exists = Path(checkpoint_path).exists()
            row["assigned_gpu"] = gpu
            log_path = REAL_LOG_DIR / row["phase"] / f"{base.token(row['run_phase'])}.log"
            row["log_path"] = str(log_path)
            if (
                args.resume
                and row["run_phase"] in completed
                and log_completed_cleanly(log_path)
                and (checkpoint_exists or not checkpoint_needed_later)
            ):
                print(f"[skip] clean result/log exists {row['job_key']}")
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


def iter_result_jsons() -> list[dict[str, Any]]:
    root = REAL_RESULTS_DIR / "ai502_transfer_real"
    out = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith("_special_metrics.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trials = payload.get("trials") if isinstance(payload, dict) else None
        has_ok_trial = any((trial or {}).get("status") == "ok" for trial in (trials or []))
        if isinstance(payload, dict) and payload.get("run_phase") and has_ok_trial:
            payload["_result_path"] = str(path)
            out.append(payload)
    return out


def load_manifest_rows() -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(REAL_MANIFEST_DIR.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows[row["run_phase"]] = dict(row)
    return list(rows.values())


def result_metric(result: dict[str, Any], metric_name: str) -> float:
    if metric_name == "valid_mrr20":
        return float(result.get("best_mrr@20") or 0.0)
    test = result.get("test_result") or {}
    aliases = {
        "test_mrr20": "mrr@20",
        "test_ndcg20": "ndcg@20",
        "test_recall20": "recall@20",
        "test_hit10": "hit@10",
    }
    return float(test.get(aliases.get(metric_name, metric_name), result.get(metric_name, 0.0)) or 0.0)


def summarize() -> None:
    REAL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results = {row["run_phase"]: row for row in iter_result_jsons()}
    manifests = load_manifest_rows()
    native_rows = []
    transfer_rows = []
    native_metric: dict[tuple[str, str, str], float] = {}
    for row in manifests:
        result = results.get(row["run_phase"])
        if not result:
            continue
        item = dict(row)
        trial = (result.get("trials") or [{}])[0]
        tr = result.get("transfer_report") or {}
        opt = tr.get("optimizer_report") or {}
        item.update(
            result_path=result.get("_result_path", ""),
            valid_mrr20=result_metric(result, "valid_mrr20"),
            test_mrr20=result_metric(result, "test_mrr20"),
            test_ndcg20=result_metric(result, "test_ndcg20"),
            test_recall20=result_metric(result, "test_recall20"),
            test_hit10=result_metric(result, "test_hit10"),
            epochs_run=trial.get("epochs_run", ""),
            early_stopped=trial.get("early_stopped", ""),
            avg_epoch_time_sec=result.get("avg_epoch_time_sec", ""),
            loaded_tensors=tr.get("loaded_tensors", ""),
            init_changed_tensors=(tr.get("init_delta_from_target") or {}).get("changed_tensors", ""),
            train_changed_tensors=(tr.get("train_delta_from_init") or {}).get("changed_tensors", ""),
            loaded_lr_scale=opt.get("loaded_lr_scale", ""),
            loaded_lr=opt.get("loaded_lr", ""),
            route_entropy_macro=trial.get("macro_1.entropy_mean", ""),
            route_neff_macro=trial.get("macro_1.n_eff", ""),
            route_top1_macro=trial.get("macro_1.top1_max_frac", ""),
            route_knn_js_macro=trial.get("macro_1.route_consistency_knn_js", ""),
            group_knn_js_macro=trial.get("macro_1.route_consistency_feature_group_knn_mean_js", ""),
        )
        if row["phase"] == "native_topup":
            native_rows.append(item)
        else:
            transfer_rows.append(item)
        if row["phase"] == "native_topup":
            native_metric[(row["target_dataset"], row["hparam_id"], row["seed"])] = float(item["test_mrr20"])

    # 기존 native bank의 seed 1~3 result까지 baseline으로 읽는다.
    source_results = SOURCE_ARTIFACT_DIR / "hyperopt_results" / "ai502_transfer"
    for path in sorted(source_results.glob("*.json")):
        if path.name.endswith("_special_metrics.json"):
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        phase = str(result.get("run_phase", ""))
        if not phase.startswith("AI502_P0_NATIVE_"):
            continue
        # run_phase는 AI502_P0_NATIVE_<dataset>_<shared_i>_s<seed>
        match = re.search(r"_(shared_\d+)_s(\d+)$", phase.lower())
        if not match:
            continue
        hparam = match.group(1)
        seed = match.group(2)
        dataset = result.get("dataset_raw") or result.get("dataset")
        if dataset:
            native_metric[(str(dataset), hparam, seed)] = result_metric(result, "test_mrr20")

    for row in transfer_rows:
        base_value = native_metric.get((row["target_dataset"], row["hparam_id"], row["seed"]))
        row["baseline_test_mrr20"] = base_value if base_value is not None else ""
        try:
            row["gain_mrr20"] = float(row["test_mrr20"]) - float(base_value)
        except Exception:
            row["gain_mrr20"] = ""

    fields = [
        "phase", "pair_id", "triplet_id", "comparison_role", "source_dataset", "target_dataset",
        "hparam_id", "seed", "transfer_mode", "policy", "valid_mrr20", "test_mrr20",
        "baseline_test_mrr20", "gain_mrr20", "test_ndcg20", "test_recall20", "test_hit10",
        "epochs_run", "early_stopped", "avg_epoch_time_sec", "loaded_tensors",
        "init_changed_tensors", "train_changed_tensors", "loaded_lr_scale", "loaded_lr",
        "route_entropy_macro", "route_neff_macro", "route_top1_macro", "route_knn_js_macro",
        "group_knn_js_macro", "result_path",
    ]
    with (REAL_ANALYSIS_DIR / "real_transfer_rows.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in transfer_rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in transfer_rows:
        if row.get("gain_mrr20") == "":
            continue
        groups[(row["phase"], row["pair_id"] or row["triplet_id"], row["comparison_role"], row["transfer_mode"], row["policy"])].append(row)
    agg_fields = ["phase", "unit", "comparison_role", "transfer_mode", "policy", "n", "mean_gain", "min_gain", "max_gain", "wins", "mean_epochs"]
    with (REAL_ANALYSIS_DIR / "real_transfer_agg.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields)
        writer.writeheader()
        for key, values in sorted(groups.items()):
            gains = [float(row["gain_mrr20"]) for row in values]
            epochs = [float(row["epochs_run"]) for row in values if row.get("epochs_run") not in {"", None}]
            writer.writerow(
                {
                    "phase": key[0],
                    "unit": key[1],
                    "comparison_role": key[2],
                    "transfer_mode": key[3],
                    "policy": key[4],
                    "n": len(values),
                    "mean_gain": sum(gains) / len(gains),
                    "min_gain": min(gains),
                    "max_gain": max(gains),
                    "wins": sum(1 for gain in gains if gain > 0),
                    "mean_epochs": sum(epochs) / len(epochs) if epochs else "",
                }
            )
    print(f"[summary] transfer_rows={len(transfer_rows)} out={REAL_ANALYSIS_DIR}")


def build_rows_for_phase(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seeds(args.seeds)
    policies = parse_csv_arg(args.policies, list(POLICIES))
    for policy in policies:
        if policy not in POLICIES:
            raise SystemExit(f"알 수 없는 policy: {policy}")
    if args.phase == "native_topup":
        return build_native_topup_rows(seeds, include_existing=args.include_existing_native)
    if args.phase == "transfer":
        return build_pair_rows(seeds, policies)
    if args.phase == "multihop":
        return build_multihop_rows(seeds, [p for p in policies if p in MULTIHOP_POLICIES])
    raise SystemExit(f"Unsupported phase: {args.phase}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI502 real transfer follow-up launcher")
    parser.add_argument("--phase", choices=["native_topup", "transfer", "multihop", "summary", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--policies", default="std,loaded_lr_0.35,global_lr_0.5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--include-existing-native", action="store_true")
    args = parser.parse_args()

    if args.phase == "summary":
        summarize()
        return 0
    if args.phase == "all":
        for phase in ("native_topup", "transfer", "multihop"):
            phase_args = argparse.Namespace(**vars(args))
            phase_args.phase = phase
            rows = build_rows_for_phase(phase_args)
            run_rows(rows, phase_args)
            if not args.dry_run:
                summarize()
        return 0
    rows = build_rows_for_phase(args)
    return run_rows(rows, args)


if __name__ == "__main__":
    raise SystemExit(main())
