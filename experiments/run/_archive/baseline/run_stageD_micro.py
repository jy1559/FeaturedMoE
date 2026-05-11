#!/usr/bin/env python3
"""Stage D runner (micro + wide) built on top of Stage C engine.

This reuses the robust scheduling/logging/result pipeline from run_stageC_focus.py
and swaps profiles + parent-axis to Stage C candidates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stageC_focus as base  # noqa: E402

# Stage identity
base.AXIS = "StageD_MicroWide_anchor2_core5"
base.PHASE_ID = "P19"
base.PHASE_NAME = "STAGED_MICROWIDE_ANCHOR2_CORE5"
base.AXIS_DESC = "staged_microwide_anchor2_core5"
base.STAGE_ID = "D"

# Parent candidates from Stage C
base.STAGEB_AXIS = "StageC_Focus_anchor2_core5"

# D profiles: aggressive regularization/local-knob expansion
base.C_PROFILES = {
    "D1": {
        "name": "micro_balanced",
        "max_len": 10,
        "dropout_delta": 0.00,
        "wd_mult": 1.00,
        "lr_mult": 1.00,
        "lr_span_mult": 0.85,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "su", "tau": 0.40, "lmd": 0.02, "lmd_sem": 0.06},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.12},
        "fame": {"num_experts": 3},
    },
    "D2": {
        "name": "micro_lowreg",
        "max_len": 15,
        "dropout_delta": -0.03,
        "wd_mult": 0.70,
        "lr_mult": 1.08,
        "lr_span_mult": 0.90,
        "sasrec": {"inner_ratio": 3, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "us_x", "tau": 0.28, "lmd": 0.03, "lmd_sem": 0.10},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.18},
        "fame": {"num_experts": 4},
    },
    "D3": {
        "name": "micro_highreg",
        "max_len": 20,
        "dropout_delta": 0.05,
        "wd_mult": 1.60,
        "lr_mult": 0.90,
        "lr_span_mult": 0.95,
        "sasrec": {"inner_ratio": 2, "heads_mode": "half_if_small"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.60, "lmd": 0.04, "lmd_sem": 0.02},
        "difsr": {"fusion_type": "concat", "use_attribute_predictor": False, "lambda_attr": 0.00},
        "fame": {"num_experts": 2},
    },
    "D4": {
        "name": "ultra_lowreg_outlier",
        "max_len": 20,
        "dropout_delta": -0.06,
        "wd_mult": 0.45,
        "lr_mult": 1.20,
        "lr_span_mult": 1.25,
        "sasrec": {"inner_ratio": 4, "heads_mode": "double_if_fit"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "us_x", "tau": 0.18, "lmd": 0.07, "lmd_sem": 0.15},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.28},
        "fame": {"num_experts": 5},
    },
    "D5": {
        "name": "ultra_highreg_outlier",
        "max_len": 15,
        "dropout_delta": 0.10,
        "wd_mult": 2.20,
        "lr_mult": 0.75,
        "lr_span_mult": 1.10,
        "sasrec": {"inner_ratio": 1, "heads_mode": "half"},
        "gru4rec": {"layer_delta": -1},
        "duorec": {"contrast": "un", "tau": 1.00, "lmd": 0.12, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "sum", "use_attribute_predictor": False, "lambda_attr": 0.00},
        "fame": {"num_experts": 2},
    },
    "D6": {
        "name": "duorec_fame_bias_outlier",
        "max_len": 10,
        "dropout_delta": -0.01,
        "wd_mult": 0.85,
        "lr_mult": 1.10,
        "lr_span_mult": 1.05,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "su", "tau": 0.22, "lmd": 0.05, "lmd_sem": 0.12},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.15},
        "fame": {"num_experts": 6},
    },
    "D7": {
        "name": "long_context_dense",
        "max_len": 20,
        "dropout_delta": -0.02,
        "wd_mult": 0.80,
        "lr_mult": 1.06,
        "lr_span_mult": 1.10,
        "sasrec": {"inner_ratio": 3, "heads_mode": "double_if_fit"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "su", "tau": 0.35, "lmd": 0.03, "lmd_sem": 0.10},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.22},
        "fame": {"num_experts": 5},
    },
    "D8": {
        "name": "sparse_robust",
        "max_len": 10,
        "dropout_delta": 0.06,
        "wd_mult": 1.80,
        "lr_mult": 0.88,
        "lr_span_mult": 0.95,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.75, "lmd": 0.08, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "concat", "use_attribute_predictor": False, "lambda_attr": 0.05},
        "fame": {"num_experts": 2},
    },
}

base.PROFILE_ORDER = list(base.C_PROFILES.keys())
base.PROFILE_COST_WEIGHT.update({
    "D1": 1.00,
    "D2": 1.08,
    "D3": 1.05,
    "D4": 1.25,
    "D5": 1.18,
    "D6": 1.15,
    "D7": 1.20,
    "D8": 1.10,
})


def _load_parent_candidates(dataset: str, model_option: str, model_label: str, topk: int):
    """Load parent candidates from Stage C output first, then Stage B fallback.

    Stage C/D/E runners share the same candidate schema but file names differ
    by original stage implementation, so we probe both.
    """
    candidate_files = ("stageC_candidates.json", "stageB_candidates.json")
    selected = []
    for cand_name in candidate_files:
        path = base.LOG_ROOT / base.STAGEB_AXIS / base._dataset_tag(dataset) / cand_name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            node = payload.get("models", {}).get(str(model_label), {})
            if not isinstance(node, dict):
                continue
            picked = node.get("selected_profiles") or []
            if not picked:
                picked = (node.get("profiles") or [])[: max(1, int(topk))]
            for it in picked[: max(1, int(topk))]:
                cfg_raw = dict(it.get("config", {}) or {})
                cfg = base._normalize_parent_config(model_option, cfg_raw)
                lo = base._metric_to_float(it.get("lr_lo"))
                hi = base._metric_to_float(it.get("lr_hi"))
                best_lr = base._metric_to_float(it.get("best_lr"))
                if lo is None or hi is None or hi <= lo:
                    if best_lr is not None:
                        lo = max(base.LR_CLAMP_MIN, best_lr / 1.8)
                        hi = min(base.LR_CLAMP_MAX, best_lr * 1.8)
                    else:
                        lo, hi, _ = base._load_stagea_lr_window(dataset, model_label)
                selected.append(
                    {
                        "parent_profile_id": str(it.get("profile_id", "P?")),
                        "config": cfg,
                        "lr_lo": float(lo),
                        "lr_hi": float(hi),
                        "lr_band_id": str(it.get("lr_band_id", "PSEL")),
                        "parent_run_phase": str(it.get("representative_run_phase", "")),
                    }
                )
            if selected:
                return selected[: max(1, int(topk))]
        except Exception:
            continue

    lo, hi, band = base._load_stagea_lr_window(dataset, model_label)
    return [
        {
            "parent_profile_id": "P_FALLBACK",
            "config": base._default_parent_config(model_option),
            "lr_lo": float(lo),
            "lr_hi": float(hi),
            "lr_band_id": str(band),
            "parent_run_phase": "",
        }
    ]


base._load_stageb_parent_candidates = _load_parent_candidates


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
