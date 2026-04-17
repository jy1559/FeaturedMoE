#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "run_full_history_campaign.py"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("baseline3_full_history_base", str(BASE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load baseline_3 base module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()

DEFAULT_AXIS = "BEAUTY_3MODEL_AGGR16"
DEFAULT_DATASETS = ["beauty"]
DEFAULT_MODELS = ["bsarec", "fame", "difsr"]
COMBOS_PER_MODEL = 16


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def align_hidden(value: int, num_heads: int) -> int:
    heads = max(1, int(num_heads))
    hidden = max(32, int(round(value / 8.0) * 8))
    if hidden % heads == 0:
        return hidden
    hidden = int(round(hidden / heads) * heads)
    if hidden < heads * 8:
        hidden = heads * 8
    return hidden


def top_existing_pool(
    dataset: str,
    model_spec: Dict[str, str],
    baseline_mod: Any,
    baseline2_sources: List[Tuple[str, List[Dict[str, str]]]],
) -> List[Any]:
    model_option = str(model_spec["model_option"])
    existing = BASE.build_existing_baseline_candidates(dataset, model_option, baseline2_sources)
    fallback = [
        BASE.Candidate(
            params=BASE.normalize_baseline_params(model_option, cand.params),
            source=cand.source,
            source_axis=cand.source.split(":", 1)[0],
            source_phase="",
            best_valid_mrr20=float(cand.score),
            seen_test_mrr20=float(cand.score),
            overall_test_mrr20=float(cand.score),
            selection_score=float(cand.score),
            hparam_id=cand.hparam_id,
            architecture_id=cand.architecture_id,
        )
        for cand in (
            BASE.PAIR60.stage_a_candidate_pool(BASE.read_csv(BASE.PAIR60.BASELINE2_STAGEA_SUMMARY), dataset, model_option)
            + BASE.PAIR60.baseline_history_pool(dataset, model_spec, baseline_mod)
            + BASE.PAIR60.default_baseline_candidates(dataset, model_option, baseline_mod)
        )
    ]
    pool = BASE.select_top_unique_candidates(model_option, existing, 8)
    seen = {BASE.candidate_dedup_key(model_option, cand.params) for cand in pool}
    for cand in fallback:
        sig = BASE.candidate_dedup_key(model_option, cand.params)
        if sig in seen:
            continue
        pool.append(cand)
        seen.add(sig)
        if len(pool) >= 6:
            break
    if not pool:
        raise RuntimeError(f"No candidate pool for dataset={dataset} model={model_option}")
    return pool


def mutate_params(model_option: str, source_params: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    params = BASE.normalize_baseline_params(model_option, source_params)
    num_heads = clamp_int(spec.get("num_heads", params.get("num_heads", 2)), 1, 8)
    hidden = params.get("hidden_size", params.get("embedding_size", 128))
    hidden = int(round(hidden * float(spec.get("hidden_mult", 1.0))))
    hidden = clamp_int(hidden, 64, 320)
    hidden = align_hidden(hidden, num_heads)

    params["hidden_size"] = hidden
    if "embedding_size" in params:
        params["embedding_size"] = hidden
    if "num_heads" in params:
        params["num_heads"] = num_heads
    if "inner_size" in params:
        inner_mult = float(spec.get("inner_mult", 2.0))
        params["inner_size"] = clamp_int(int(round(hidden * inner_mult)), hidden, 1024)
    if "attribute_hidden_size" in params:
        attr_mult = float(spec.get("attr_mult", 1.0))
        params["attribute_hidden_size"] = clamp_int(int(round(hidden * attr_mult)), 64, 384)

    params["max_len"] = clamp_int(spec.get("max_len", params.get("max_len", 20)), 20, 96)
    params["num_layers"] = clamp_int(
        spec.get("num_layers", params.get("num_layers", 2)) + int(spec.get("layer_delta", 0)),
        1,
        4,
    )
    params["dropout"] = round(
        clamp_float(params.get("dropout", 0.15) + float(spec.get("dropout_delta", 0.0)), 0.05, 0.45),
        4,
    )
    params["weight_decay"] = round(
        clamp_float(params.get("weight_decay", 1e-4) * float(spec.get("wd_mult", 1.0)), 1e-7, 5e-3),
        8,
    )

    model_name = str(model_option).lower()
    if model_name == "fame":
        params["num_experts"] = clamp_int(
            spec.get("num_experts", params.get("num_experts", 3)) + int(spec.get("expert_delta", 0)),
            2,
            8,
        )
    if model_name == "difsr":
        params["fusion_type"] = str(spec.get("fusion_type", params.get("fusion_type", "gate")))
        params["use_attribute_predictor"] = bool(spec.get("use_attribute_predictor", True))
        params["lambda_attr"] = round(
            clamp_float(params.get("lambda_attr", 0.1) + float(spec.get("lambda_attr_delta", 0.0)), 0.02, 0.4),
            4,
        )
    return params


def aggressive_variant_specs(model_option: str) -> List[Dict[str, Any]]:
    common = [
        {"kind": "existing_primary", "theme": "carry_best_seed", "source_index": 0, "existing": True},
        {"kind": "existing_secondary", "theme": "carry_alt_seed", "source_index": 1, "existing": True},
        {"kind": "existing_tertiary", "theme": "carry_third_seed", "source_index": 2, "existing": True},
        {"kind": "existing_quaternary", "theme": "carry_fourth_seed", "source_index": 3, "existing": True},
        {"kind": "long_context_stable", "theme": "len32_regup", "source_index": 0, "max_len": 32, "dropout_delta": 0.03, "wd_mult": 1.5},
        {"kind": "long_context_extreme", "theme": "len48_regup", "source_index": 0, "max_len": 48, "dropout_delta": 0.05, "wd_mult": 1.9},
        {"kind": "wide_context", "theme": "len32_hidden150", "source_index": 0, "max_len": 32, "hidden_mult": 1.5, "inner_mult": 2.5, "dropout_delta": 0.02, "wd_mult": 1.2},
        {"kind": "wide_context_extreme", "theme": "len48_hidden175", "source_index": 0, "max_len": 48, "hidden_mult": 1.75, "inner_mult": 3.0, "dropout_delta": 0.05, "wd_mult": 1.35},
        {"kind": "compact_long", "theme": "len64_hidden075", "source_index": 0, "max_len": 64, "hidden_mult": 0.75, "inner_mult": 1.75, "dropout_delta": 0.06, "wd_mult": 2.2, "layer_delta": -1},
        {"kind": "alt_long", "theme": "alt_len48", "source_index": 1, "max_len": 48, "dropout_delta": 0.03, "wd_mult": 1.6},
        {"kind": "alt_wide", "theme": "alt_hidden150", "source_index": 1, "max_len": 40, "hidden_mult": 1.5, "inner_mult": 2.5, "dropout_delta": 0.02, "wd_mult": 1.25},
        {"kind": "head_shift", "theme": "head_relayout", "source_index": 2, "max_len": 40, "hidden_mult": 1.25, "num_heads": 4, "dropout_delta": 0.02, "wd_mult": 1.3},
        {"kind": "lowreg_bigdim", "theme": "len48_lowreg", "source_index": 2, "max_len": 48, "hidden_mult": 1.6, "inner_mult": 2.75, "dropout_delta": -0.01, "wd_mult": 0.65},
        {"kind": "highreg_short", "theme": "len24_highreg", "source_index": 3, "max_len": 24, "hidden_mult": 1.1, "dropout_delta": 0.07, "wd_mult": 2.5},
    ]

    model_name = str(model_option).lower()
    if model_name == "bsarec":
        common.extend(
            [
                {"kind": "bsarec_freq_wide", "theme": "len64_hidden256", "source_index": 0, "max_len": 64, "hidden_mult": 2.0, "inner_mult": 3.0, "num_heads": 4, "dropout_delta": 0.04, "wd_mult": 1.7, "layer_delta": -1},
                {"kind": "bsarec_freq_compact", "theme": "len80_hidden096", "source_index": 1, "max_len": 80, "hidden_mult": 0.75, "inner_mult": 2.0, "num_heads": 2, "dropout_delta": 0.08, "wd_mult": 2.8, "layer_delta": -1},
            ]
        )
    elif model_name == "fame":
        common.extend(
            [
                {"kind": "fame_more_experts", "theme": "len48_expert6", "source_index": 0, "max_len": 48, "hidden_mult": 1.5, "inner_mult": 2.5, "dropout_delta": 0.04, "wd_mult": 1.5, "expert_delta": 2},
                {"kind": "fame_router_pressure", "theme": "len64_expert8", "source_index": 1, "max_len": 64, "hidden_mult": 1.25, "inner_mult": 2.25, "dropout_delta": 0.07, "wd_mult": 2.1, "expert_delta": 4},
            ]
        )
    elif model_name == "difsr":
        common.extend(
            [
                {"kind": "difsr_attr_wide", "theme": "len48_attr150", "source_index": 0, "max_len": 48, "hidden_mult": 1.4, "inner_mult": 2.4, "attr_mult": 1.5, "dropout_delta": 0.03, "wd_mult": 1.4, "fusion_type": "gate", "lambda_attr_delta": 0.06},
                {"kind": "difsr_attr_pressure", "theme": "len64_attr200", "source_index": 1, "max_len": 64, "hidden_mult": 1.2, "inner_mult": 2.2, "attr_mult": 2.0, "dropout_delta": 0.06, "wd_mult": 2.0, "fusion_type": "sum", "lambda_attr_delta": 0.12},
            ]
        )
    else:
        common.extend(
            [
                {"kind": "model_special_a", "theme": "len48_hidden140", "source_index": 0, "max_len": 48, "hidden_mult": 1.4, "inner_mult": 2.4, "dropout_delta": 0.04, "wd_mult": 1.5},
                {"kind": "model_special_b", "theme": "len64_hidden090", "source_index": 1, "max_len": 64, "hidden_mult": 0.9, "inner_mult": 2.0, "dropout_delta": 0.07, "wd_mult": 2.0},
            ]
        )
    return common


def build_beauty_model_combos(
    dataset: str,
    model_spec: Dict[str, str],
    baseline_mod: Any,
    baseline2_sources: List[Tuple[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    model_option = str(model_spec["model_option"])
    pool = top_existing_pool(dataset, model_spec, baseline_mod, baseline2_sources)
    while len(pool) < 4:
        pool.append(pool[-1])

    combos: List[Dict[str, Any]] = []
    seen = set()
    for rank, spec in enumerate(aggressive_variant_specs(model_option), start=1):
        source = pool[min(int(spec.get("source_index", 0)), len(pool) - 1)]
        if spec.get("existing"):
            params = BASE.normalize_baseline_params(model_option, source.params)
            combo_source = source.source
            selection_score = source.selection_score
        else:
            params = mutate_params(model_option, source.params, spec)
            combo_source = f"variant_from:{source.source}"
            selection_score = source.selection_score

        sig = BASE.candidate_dedup_key(model_option, params)
        if sig in seen:
            params["weight_decay"] = round(clamp_float(params.get("weight_decay", 1e-4) * 1.18, 1e-7, 5e-3), 8)
            sig = BASE.candidate_dedup_key(model_option, params)
        seen.add(sig)
        combos.append(
            {
                "combo_id": f"C{rank:02d}",
                "combo_rank": rank,
                "combo_kind": spec["kind"],
                "combo_theme": spec["theme"],
                "combo_source": combo_source,
                "source_axis": source.source_axis,
                "source_phase": source.source_phase,
                "selection_score": selection_score,
                "hparam_id": source.hparam_id,
                "architecture_id": source.architecture_id,
                "params": params,
            }
        )

    if len(combos) != COMBOS_PER_MODEL:
        raise RuntimeError(f"Expected {COMBOS_PER_MODEL} combos for {model_option}, got {len(combos)}")
    return combos


def choose_batch_sizes(dataset: str, model_option: str, params: Dict[str, Any]) -> Tuple[int, int]:
    max_len = BASE.safe_int(params.get("max_len", 20), 20)
    hidden = BASE.safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
    pressure = max_len * hidden
    model_name = str(model_option).lower()
    if pressure >= 14000:
        return (1024, 2048) if model_name == "fame" else (1536, 3072)
    if pressure >= 10000:
        return (2048, 4096)
    if pressure >= 7000:
        return (3072, 6144)
    return (4096, 8192)


def compute_lr_bounds(dataset: str, model_option: str, combo_rank: int, params: Dict[str, Any]) -> Tuple[float, float]:
    lo, hi = BASE.PAIR60.compute_lr_bounds(dataset, model_option, 2, params)
    if combo_rank >= 13:
        lo *= 0.55
        hi *= 1.6
    elif combo_rank >= 9:
        lo *= 0.65
        hi *= 1.45
    elif combo_rank >= 5:
        lo *= 0.75
        hi *= 1.3
    lo = max(1e-5, float(lo))
    hi = min(1.2e-2, float(hi))
    if hi <= lo:
        hi = min(1.2e-2, lo * 2.0)
    return lo, hi


def compute_wd_bounds(model_option: str, combo_rank: int, params: Dict[str, Any]) -> Tuple[float, float]:
    default_wd = 1e-4
    base_wd = max(BASE.safe_float(params.get("weight_decay", default_wd), default_wd), 1e-7)
    if combo_rank >= 13:
        lo = base_wd * 0.25
        hi = base_wd * 3.5
    elif combo_rank >= 9:
        lo = base_wd * 0.35
        hi = base_wd * 3.0
    elif combo_rank >= 5:
        lo = base_wd * 0.5
        hi = base_wd * 2.4
    else:
        lo = base_wd * 0.7
        hi = base_wd * 1.8
    lo = max(1e-7, float(lo))
    hi = min(5e-3, float(hi))
    if hi <= lo:
        hi = min(5e-3, lo * 2.0)
    return lo, hi


def build_matrix_rows(
    datasets: List[str],
    model_options: List[str],
    baseline_mod: Any,
    _fmoe_mod: Any,
    baseline2_sources: List[Tuple[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    matrix_rows: List[Dict[str, Any]] = []
    pair_cursor = 0
    for dataset in datasets:
        for model_option in model_options:
            pair_cursor += 1
            model_spec = BASE.MODEL_SPEC_BY_OPTION[model_option]
            pair_id = f"P{pair_cursor:03d}"
            combos = build_beauty_model_combos(dataset, model_spec, baseline_mod, baseline2_sources)
            for combo in combos:
                lr_lo, lr_hi = compute_lr_bounds(dataset, model_option, int(combo["combo_rank"]), combo["params"])
                wd_lo, wd_hi = compute_wd_bounds(model_option, int(combo["combo_rank"]), combo["params"])
                matrix_rows.append(
                    {
                        "dataset": dataset,
                        "model_option": model_option,
                        "model_label": model_spec["model_label"],
                        "pair_id": pair_id,
                        "combo_id": combo["combo_id"],
                        "combo_rank": int(combo["combo_rank"]),
                        "combo_kind": combo["combo_kind"],
                        "combo_theme": combo["combo_theme"],
                        "combo_source": combo["combo_source"],
                        "source_axis": combo["source_axis"],
                        "source_phase": combo["source_phase"],
                        "selection_score": float(combo["selection_score"]),
                        "hparam_id": combo.get("hparam_id", ""),
                        "architecture_id": combo.get("architecture_id", ""),
                        "lr_lo": float(lr_lo),
                        "lr_hi": float(lr_hi),
                        "wd_lo": float(wd_lo),
                        "wd_hi": float(wd_hi),
                        "params_json": json.dumps(combo["params"], ensure_ascii=False, sort_keys=True),
                    }
                )
    return matrix_rows


def build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> List[str]:
    dataset = str(row["dataset"])
    model_option = str(row["model_option"])
    params = json.loads(str(row["params_json"]))
    train_batch_size, eval_batch_size = choose_batch_sizes(dataset, model_option, params)

    python_bin = BASE.os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        BASE.dataset_config_name(dataset),
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(args.tune_epochs)),
        "--tune-patience",
        str(int(args.tune_patience)),
        "--search-algo",
        str(args.search_algo),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        BASE.TRACK,
        "--run-axis",
        str(args.axis),
        "--run-phase",
        str(row["run_phase"]),
        f"model={model_option}",
        f"dataset={dataset}",
        "eval_mode=session_fixed",
        "feature_mode=full_v4",
        f"train_batch_size={int(train_batch_size)}",
        f"eval_batch_size={int(eval_batch_size)}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        f"++seed={int(row['runtime_seed'])}",
        "++history_input_mode=full_history_session_targets",
        "++history_group_field=user_id",
        "++target_group_field=session_id",
        "++history_eval_policy=strict_train_prefix",
    ]

    cmd.extend(BASE.PAIR60.base_runtime_overrides(model_option, params))

    wd_key = BASE.weight_search_key(model_option)
    search_dict = {
        "learning_rate": [float(row["lr_lo"]), float(row["lr_hi"])],
        wd_key: [float(row["wd_lo"]), float(row["wd_hi"])],
    }
    search_types = {
        "learning_rate": "loguniform",
        wd_key: "loguniform",
    }
    cmd.append(f"++search={BASE.PAIR60.hydra_literal(search_dict)}")
    cmd.append(f"++search_space_type_overrides={BASE.PAIR60.hydra_literal(search_types)}")

    fixed = BASE.PAIR60.fixed_search_entries(model_option, params)
    for key, value in fixed.items():
        if key == wd_key:
            continue
        cmd.append(f"++search.{key}={BASE.PAIR60.hydra_literal([value])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", type=str, default=DEFAULT_AXIS)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seeds", type=str, default="1")
    parser.add_argument("--runtime-seed-base", type=int, default=3100)
    parser.add_argument("--max-evals", type=int, default=16)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--search-algo", type=str, default="random")
    parser.add_argument("--resume-from-logs", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--slack-progress-step", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    BASE.STOP_EVENT.clear()
    BASE.install_signal_handlers()

    datasets = BASE.parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    model_options = [m.lower() for m in BASE.parse_csv_list(args.models)] or list(DEFAULT_MODELS)
    for dataset in datasets:
        if dataset != "beauty":
            raise RuntimeError(f"This runner is beauty-only. Unsupported dataset: {dataset}")
    for model in model_options:
        if model not in set(DEFAULT_MODELS):
            raise RuntimeError(f"This runner supports only {DEFAULT_MODELS}; got {model}")
    gpus = BASE.parse_csv_list(args.gpus) or ["0"]
    seeds = BASE.parse_csv_ints(args.seeds) or [1]

    baseline_mod, fmoe_mod, slack_mod = BASE.PAIR60.load_external_modules()
    SlackProgressNotifier = getattr(slack_mod, "SlackProgressNotifier")

    baseline2_sources = BASE.load_baseline2_sources()
    matrix_rows = build_matrix_rows(datasets, model_options, baseline_mod, fmoe_mod, baseline2_sources)

    axis_root = BASE.LOG_ROOT / str(args.axis)
    axis_root.mkdir(parents=True, exist_ok=True)
    matrix_csv = axis_root / "combo_matrix.csv"
    BASE.write_csv(matrix_csv, matrix_rows, BASE.MATRIX_FIELDS)

    run_rows = BASE.build_run_rows(matrix_rows, str(args.axis), seeds, int(args.runtime_seed_base))
    if int(args.max_runs) > 0:
        run_rows = run_rows[: int(args.max_runs)]

    summary_csv = axis_root / "summary.csv"
    existing_by_run_phase = BASE.read_existing_summary(summary_csv) if args.resume_from_logs else {}

    all_rows: List[Dict[str, Any]] = []
    precompleted_rows: List[Dict[str, Any]] = []
    pending_jobs: List[Dict[str, Any]] = []
    for row in run_rows:
        run_phase = str(row["run_phase"])
        if args.resume_from_logs:
            prev = existing_by_run_phase.get(run_phase)
            if prev is not None and str(prev.get("status", "")) == "ok":
                all_rows.append(dict(prev))
                precompleted_rows.append(dict(prev))
                continue
            resumed = BASE.build_resumed_row(row, axis_root)
            if resumed is not None:
                all_rows.append(resumed)
                precompleted_rows.append(resumed)
                continue
        pending_jobs.append(dict(row))

    notifier = SlackProgressNotifier(
        phase_label=f"{args.axis}",
        rows=run_rows,
        progress_step=int(args.slack_progress_step),
    )
    notifier.notify_plan(precompleted_rows=precompleted_rows)

    print(f"[launch] pairs={len(datasets) * len(model_options)} combos={len(matrix_rows)} runs={len(run_rows)}")
    print(f"[launch] precompleted={len(precompleted_rows)} pending={len(pending_jobs)}")
    print(f"[launch] matrix={matrix_csv}")

    if args.dry_run:
        leaderboard = BASE.build_leaderboard(all_rows)
        BASE.write_csv(summary_csv, all_rows, BASE.SUMMARY_FIELDS)
        BASE.write_csv(
            axis_root / "leaderboard.csv",
            leaderboard,
            [
                "dataset",
                "model",
                "model_label",
                "pair_id",
                "combo_id",
                "combo_kind",
                "best_valid_mrr20",
                "seen_test_mrr20",
                "test_mrr20",
                "avg_epoch_time_sec",
                "test_inference_time_sec",
                "result_path",
            ],
        )
        manifest = {
            "track": BASE.TRACK,
            "axis": args.axis,
            "datasets": datasets,
            "models": model_options,
            "pairs": len(datasets) * len(model_options),
            "combos": len(matrix_rows),
            "combos_per_model": COMBOS_PER_MODEL,
            "seeds": seeds,
            "runs": len(run_rows),
            "precompleted": len(precompleted_rows),
            "pending": len(pending_jobs),
            "dry_run": True,
            "timestamp_utc": BASE.now_utc(),
            "matrix_csv": str(matrix_csv),
            "summary_csv": str(summary_csv),
        }
        (axis_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[launch] dry_run_completed=true")
        return

    if pending_jobs:
        job_queue: Queue = Queue()
        for row in pending_jobs:
            job_queue.put(row)

        lock = threading.Lock()

        def worker(gpu_id: str) -> None:
            while True:
                if BASE.STOP_EVENT.is_set():
                    return
                try:
                    job = job_queue.get_nowait()
                except Empty:
                    return
                if BASE.STOP_EVENT.is_set():
                    return
                print(
                    "[launch] "
                    f"dataset={job['dataset']} model={job['model_option']} combo={job['combo_id']} "
                    f"seed={job['seed_id']} gpu={gpu_id} status=start"
                )
                try:
                    result_row = BASE.run_one(job, gpu_id, args, axis_root)
                except Exception as exc:
                    result_row = BASE.build_summary_row(
                        row=job,
                        gpu_id=gpu_id,
                        status="fail",
                        metrics={},
                        result_path="",
                        log_path=BASE.resolve_log_path(axis_root, job),
                        elapsed_sec=0.0,
                        error=f"worker_exception={exc}",
                    )
                with lock:
                    all_rows.append(result_row)
                    notifier.mark_complete(result_row)
                print(
                    "[launch] "
                    f"dataset={job['dataset']} model={job['model_option']} combo={job['combo_id']} "
                    f"seed={job['seed_id']} gpu={gpu_id} status={result_row['status']}"
                )
                job_queue.task_done()

        original_build_command = BASE.build_command
        try:
            BASE.build_command = build_command
            threads: List[threading.Thread] = []
            for gpu_id in gpus:
                thread = threading.Thread(target=worker, args=(str(gpu_id),), daemon=False)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        finally:
            BASE.build_command = original_build_command

    all_rows.sort(
        key=lambda r: (
            BASE.DATASET_ORDER_INDEX.get(str(r.get("dataset", "")), 999),
            BASE.MODEL_ORDER_INDEX.get(str(r.get("model", "")), 999),
            str(r.get("pair_id", "")),
            str(r.get("combo_id", "")),
            BASE.safe_int(r.get("runtime_seed", 0), 0),
        )
    )

    leaderboard = BASE.build_leaderboard(all_rows)
    BASE.write_csv(summary_csv, all_rows, BASE.SUMMARY_FIELDS)
    BASE.write_csv(
        axis_root / "leaderboard.csv",
        leaderboard,
        [
            "dataset",
            "model",
            "model_label",
            "pair_id",
            "combo_id",
            "combo_kind",
            "best_valid_mrr20",
            "seen_test_mrr20",
            "test_mrr20",
            "avg_epoch_time_sec",
            "test_inference_time_sec",
            "result_path",
        ],
    )

    ok_rows = [row for row in all_rows if str(row.get("status", "")) == "ok"]
    manifest = {
        "track": BASE.TRACK,
        "axis": args.axis,
        "datasets": datasets,
        "models": model_options,
        "pairs": len(datasets) * len(model_options),
        "combos": len(matrix_rows),
        "combos_per_model": COMBOS_PER_MODEL,
        "seeds": seeds,
        "runs": len(run_rows),
        "ok": len(ok_rows),
        "fail": len(all_rows) - len(ok_rows),
        "precompleted": len(precompleted_rows),
        "pending_executed": len(pending_jobs),
        "interrupted": bool(BASE.STOP_EVENT.is_set()),
        "timestamp_utc": BASE.now_utc(),
        "matrix_csv": str(matrix_csv),
        "summary_csv": str(summary_csv),
        "leaderboard_csv": str(axis_root / "leaderboard.csv"),
    }
    (axis_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[launch] ok={manifest['ok']} fail={manifest['fail']} "
        f"interrupted={str(manifest['interrupted']).lower()} summary={summary_csv}"
    )


if __name__ == "__main__":
    main()