#!/usr/bin/env python3
"""Run a targeted revised campaign for problematic PAIR60_V4 runs.

This launcher reuses original run commands recorded in PAIR60_V4 logs, then applies
small, explicit overrides for:
- OOM-prone runs: lower train/eval batch sizes.
- Underperforming runs: nudge fixed architecture/search knobs using stronger settings
  from better-performing baselines in the same dataset family.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"

TRACK = "baseline_2"
BASE_AXIS = "PAIR60_V4"
DEFAULT_AXIS = "PAIR60_V4_REVISED"

BASE_LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK / BASE_AXIS
REVISED_LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RevisionSpec:
    run_phase: str
    reason: str
    ref_note: str
    train_batch_size: int | None = None
    eval_batch_size: int | None = None
    lr_range: tuple[float, float] | None = None
    fixed_overrides: Dict[str, Any] | None = None


# Target set derived from current PAIR60_V4 analysis (OOM + strong underperformance outliers).
REVISED_TARGETS: List[RevisionSpec] = [
    RevisionSpec(
        run_phase="PAIR60_V4_DLASTFM0_03_MTISASREC_P053_C2_S1",
        reason="OOM in 3/4 trials",
        ref_note="Keep TiSASRec shape, only reduce memory pressure.",
        train_batch_size=1024,
        eval_batch_size=2048,
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DMOVIELENS1M_MTISASREC_P043_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Use smaller batch and slightly shorter time window for stability.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={"time_span": 256},
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MTISASREC_P013_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Retail Rocket TiSASRec memory cut + softer model width.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={"hidden_size": 96, "embedding_size": 96, "inner_size": 192},
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MDIFSR_P017_C1_S1",
        reason="OOM in 6/6 trials and zero score",
        ref_note="Reference DIFSR stable profile: smaller width/layers + lower batch.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "fusion_type": "gate",
            "use_attribute_predictor": True,
            "lambda_attr": 0.12,
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MDIFSR_P017_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Same as C1 revised to remove memory failures.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "fusion_type": "gate",
            "use_attribute_predictor": True,
            "lambda_attr": 0.12,
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MFDSA_P019_C1_S1",
        reason="OOM in 6/6 trials and zero score",
        ref_note="Reference FDSA stable shape with lower width + lower batch.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "attribute_hidden_size": 128,
            "fusion_type": "gate",
            "use_attribute_predictor": True,
            "lambda_attr": 0.12,
            "selected_features": ["category"],
            "pooling_mode": "mean",
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DRETAIL_ROCKET_MFDSA_P019_C2_S1",
        reason="OOM in 6/6 trials",
        ref_note="Same FDSA revised profile as C1.",
        train_batch_size=1024,
        eval_batch_size=2048,
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.15,
            "hidden_dropout_prob": 0.15,
            "attn_dropout_prob": 0.15,
            "weight_decay": 2e-4,
            "attribute_hidden_size": 128,
            "fusion_type": "gate",
            "use_attribute_predictor": True,
            "lambda_attr": 0.12,
            "selected_features": ["category"],
            "pooling_mode": "mean",
        },
        lr_range=(1.2e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MBSAREC_P005_C2_S1",
        reason="Severe underperformance vs dataset peers",
        ref_note="Reference SASRec-like stable transformer width/dropout.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "n_heads": 2,
            "num_heads": 2,
            "inner_size": 256,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-5,
        },
        lr_range=(1.5e-4, 7.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MDIFSR_P007_C2_S1",
        reason="Near-zero score compared to same-dataset models",
        ref_note="Reference DIFSR gate fusion setting used in stronger runs.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-5,
            "attribute_hidden_size": 128,
            "fusion_type": "gate",
            "use_attribute_predictor": True,
            "lambda_attr": 0.1,
        },
        lr_range=(1.2e-4, 6.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MFAME_P008_C1_S1",
        reason="Very low score outlier",
        ref_note="Reference stronger FAME setup with more regularized transformer block.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1.5e-4,
            "num_experts": 3,
        },
        lr_range=(1.8e-4, 5.5e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DBEAUTY_MGRU4REC_P002_C1_S1",
        reason="Low score vs transformer baselines",
        ref_note="Reference stronger GRU4Rec profile from better datasets.",
        fixed_overrides={
            "hidden_size": 192,
            "embedding_size": 192,
            "num_layers": 1,
            "dropout_prob": 0.25,
            "hidden_dropout_prob": 0.25,
            "weight_decay": 1e-5,
            "MAX_ITEM_LIST_LENGTH": 30,
        },
        lr_range=(2.0e-4, 1.0e-3),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DFOURSQUARE_MBSAREC_P035_C2_S1",
        reason="Underperforming BSARec combo",
        ref_note="Reference BSARec C1-style balanced transformer settings.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "n_heads": 2,
            "num_heads": 2,
            "inner_size": 256,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-5,
        },
        lr_range=(1.2e-4, 6.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DFOURSQUARE_MFAME_P038_C1_S1",
        reason="Underperforming FAME combo",
        ref_note="Reference stronger FAME settings used in lastfm/foursquare families.",
        fixed_overrides={
            "hidden_size": 128,
            "embedding_size": 128,
            "n_layers": 2,
            "num_layers": 2,
            "inner_size": 256,
            "dropout_ratio": 0.1,
            "hidden_dropout_prob": 0.1,
            "attn_dropout_prob": 0.1,
            "weight_decay": 1e-4,
            "num_experts": 3,
        },
        lr_range=(1.5e-4, 5.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DFOURSQUARE_MGRU4REC_P032_C2_S1",
        reason="Underperforming GRU4Rec combo",
        ref_note="Reference GRU4Rec high-LR profile that performed better in similar runs.",
        fixed_overrides={
            "hidden_size": 192,
            "embedding_size": 192,
            "num_layers": 1,
            "dropout_prob": 0.2,
            "hidden_dropout_prob": 0.2,
            "weight_decay": 1e-5,
            "MAX_ITEM_LIST_LENGTH": 30,
        },
        lr_range=(2.0e-4, 9.0e-4),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MGRU4REC_P002_C1_S1",
        reason="Low score on KuaiRec strict",
        ref_note="Reference Kuai-friendly GRU4Rec with larger hidden and broader LR.",
        fixed_overrides={
            "hidden_size": 192,
            "embedding_size": 192,
            "num_layers": 1,
            "dropout_prob": 0.25,
            "hidden_dropout_prob": 0.25,
            "weight_decay": 1e-5,
        },
        lr_range=(2.2e-4, 1.1e-3),
    ),
    RevisionSpec(
        run_phase="PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MTISASREC_P023_C1_S1",
        reason="Lower-than-expected TiSASRec score",
        ref_note="Reference TiSASRec stable setting from lastfm, with conservative width/time_span.",
        train_batch_size=1536,
        eval_batch_size=3072,
        fixed_overrides={
            "hidden_size": 96,
            "embedding_size": 96,
            "n_layers": 2,
            "num_layers": 2,
            "n_heads": 2,
            "num_heads": 2,
            "inner_size": 192,
            "dropout_ratio": 0.12,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "weight_decay": 1e-6,
            "time_span": 512,
        },
        lr_range=(1.2e-4, 4.5e-4),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", type=str, default=DEFAULT_AXIS)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--search-algo", type=str, default="random")
    parser.add_argument("--max-evals", type=int, default=6)
    parser.add_argument("--tune-epochs", type=int, default=70)
    parser.add_argument("--tune-patience", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--python-bin", type=str, default=os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python"))
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def find_log_map() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in BASE_LOG_ROOT.rglob("*.log"):
        out[p.stem] = p
    return out


def parse_cmd_from_log(log_path: Path) -> List[str]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for ln in lines[:8]:
        if ln.startswith("# cmd="):
            return shlex.split(ln[len("# cmd=") :].strip())
    raise RuntimeError(f"No '# cmd=' line found in log: {log_path}")


def token_key(token: str) -> str:
    if token.startswith("++") and "=" in token:
        return token[2:].split("=", 1)[0]
    if token.startswith("--"):
        return token
    return ""


def set_arg_pair(tokens: List[str], flag: str, value: str) -> List[str]:
    out: List[str] = []
    i = 0
    found = False
    while i < len(tokens):
        t = tokens[i]
        if t == flag and i + 1 < len(tokens):
            out.extend([flag, value])
            i += 2
            found = True
            continue
        out.append(t)
        i += 1
    if not found:
        out.extend([flag, value])
    return out


def drop_keys(tokens: List[str], keys: List[str]) -> List[str]:
    keyset = set(keys)
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        k = token_key(t)
        if k in keyset:
            i += 1
            continue
        if t in keyset and i + 1 < len(tokens):
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{hydra_literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Unsupported type: {type(value)}")


def build_revised_command(base_cmd: List[str], spec: RevisionSpec, axis: str, gpu: str, args: argparse.Namespace) -> List[str]:
    run_phase_new = spec.run_phase.replace(BASE_AXIS, axis, 1) + "_REV1"

    # Replace core positional/script and global args.
    cmd = list(base_cmd)
    if cmd:
        cmd[0] = args.python_bin
    cmd = set_arg_pair(cmd, "--run-axis", axis)
    cmd = set_arg_pair(cmd, "--run-phase", run_phase_new)
    cmd = set_arg_pair(cmd, "--search-algo", str(args.search_algo))
    cmd = set_arg_pair(cmd, "--max-evals", str(int(args.max_evals)))
    cmd = set_arg_pair(cmd, "--tune-epochs", str(int(args.tune_epochs)))
    cmd = set_arg_pair(cmd, "--tune-patience", str(int(args.tune_patience)))

    # Remove stale gpu_id/model runtime args that we will append cleanly.
    cmd = drop_keys(
        cmd,
        [
            "gpu_id",
            "search",
            "train_batch_size",
            "eval_batch_size",
            "search.train_batch_size",
            "search.eval_batch_size",
            "search_space_type_overrides.train_batch_size",
            "search_space_type_overrides.eval_batch_size",
        ],
    )

    # Ensure gpu override is last-wins.
    cmd.append(f"gpu_id={gpu}")

    if spec.train_batch_size is not None:
        bs = int(spec.train_batch_size)
        cmd.extend(
            [
                f"++train_batch_size={bs}",
                f"++search.train_batch_size={hydra_literal([bs])}",
                "++search_space_type_overrides.train_batch_size=choice",
            ]
        )
    if spec.eval_batch_size is not None:
        ebs = int(spec.eval_batch_size)
        cmd.extend(
            [
                f"++eval_batch_size={ebs}",
                f"++search.eval_batch_size={hydra_literal([ebs])}",
                "++search_space_type_overrides.eval_batch_size=choice",
            ]
        )

    if spec.fixed_overrides:
        for k, v in spec.fixed_overrides.items():
            cmd.extend(
                [
                    f"++{k}={hydra_literal(v)}",
                    f"++search.{k}={hydra_literal([v])}",
                    f"++search_space_type_overrides.{k}=choice",
                ]
            )

    if spec.lr_range is not None:
        lo, hi = float(spec.lr_range[0]), float(spec.lr_range[1])
        cmd.extend(
            [
                f"++search={hydra_literal({'learning_rate': [lo, hi]})}",
                "++search_space_type_overrides={learning_rate:\"loguniform\"}",
            ]
        )

    # Make seed different but deterministic for revised runs.
    # Keep original ++seed if present, and append last-wins +10000.
    seed_val = None
    for t in cmd:
        if t.startswith("++seed="):
            try:
                seed_val = int(t.split("=", 1)[1].strip().strip('"'))
            except Exception:
                seed_val = None
    if seed_val is not None:
        cmd.append(f"++seed={seed_val + 10000}")

    return cmd


def relay(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def run_one(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# revised_ts={now_utc()}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(EXP_DIR),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    return int(rc)
                time.sleep(0.2)
        except KeyboardInterrupt:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            time.sleep(0.4)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            return 130


def main() -> None:
    args = parse_args()
    gpus = parse_csv_list(args.gpus) or ["0"]

    targets = list(REVISED_TARGETS)
    if int(args.max_runs) > 0:
        targets = targets[: int(args.max_runs)]

    log_map = find_log_map()
    axis_root = REVISED_LOG_ROOT / str(args.axis)
    axis_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    relay(f"[revised] axis={args.axis} targets={len(targets)} gpus={gpus}")

    for idx, spec in enumerate(targets):
        base_log = log_map.get(spec.run_phase)
        if base_log is None:
            relay(f"[revised][skip] run_phase={spec.run_phase} reason=base_log_not_found")
            manifest.append(
                {
                    "run_phase": spec.run_phase,
                    "status": "skip",
                    "error": "base_log_not_found",
                    "reason": spec.reason,
                    "ref_note": spec.ref_note,
                }
            )
            continue

        gpu = gpus[idx % len(gpus)]
        cmd_base = parse_cmd_from_log(base_log)
        cmd = build_revised_command(cmd_base, spec, args.axis, gpu, args)

        run_phase_new = spec.run_phase.replace(BASE_AXIS, args.axis, 1) + "_REV1"
        log_path = axis_root / "revised_runs" / f"{run_phase_new}.log"

        relay(f"[revised][plan] {spec.run_phase}")
        relay(f"  reason: {spec.reason}")
        relay(f"  ref: {spec.ref_note}")
        if spec.train_batch_size is not None or spec.eval_batch_size is not None:
            relay(
                "  batch: "
                f"train={spec.train_batch_size if spec.train_batch_size is not None else 'unchanged'} "
                f"eval={spec.eval_batch_size if spec.eval_batch_size is not None else 'unchanged'}"
            )
        if spec.lr_range is not None:
            relay(f"  lr_range: [{spec.lr_range[0]}, {spec.lr_range[1]}]")
        if spec.fixed_overrides:
            relay(f"  fixed_overrides: {json.dumps(spec.fixed_overrides, ensure_ascii=False)}")

        if args.dry_run:
            manifest.append(
                {
                    "run_phase": spec.run_phase,
                    "run_phase_revised": run_phase_new,
                    "status": "dry-run",
                    "gpu": gpu,
                    "reason": spec.reason,
                    "ref_note": spec.ref_note,
                    "base_log": str(base_log),
                    "log_path": str(log_path),
                    "cmd": cmd,
                }
            )
            continue

        relay(f"[revised][start] {run_phase_new} gpu={gpu}")
        rc = run_one(cmd, log_path)
        status = "ok" if rc == 0 else "fail"
        relay(f"[revised][done] {run_phase_new} rc={rc} status={status}")

        manifest.append(
            {
                "run_phase": spec.run_phase,
                "run_phase_revised": run_phase_new,
                "status": status,
                "rc": rc,
                "gpu": gpu,
                "reason": spec.reason,
                "ref_note": spec.ref_note,
                "base_log": str(base_log),
                "log_path": str(log_path),
            }
        )

    manifest_path = axis_root / "revised_manifest.json"
    manifest_path.write_text(json.dumps({"axis": args.axis, "generated_at": now_utc(), "runs": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    relay(f"[revised] manifest={manifest_path}")


if __name__ == "__main__":
    main()
