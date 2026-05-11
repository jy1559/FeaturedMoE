#!/usr/bin/env python3
"""Stage E runner (re-LR + light seed check) built on top of Stage C engine.

This reuses run_stageC_focus.py and sets parent-axis to Stage D candidates.
Profiles here are LR-local variants with minimal structural perturbation.
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
base.AXIS = "StageE_ReLRSeed_anchor2_core5"
base.PHASE_ID = "P20"
base.PHASE_NAME = "STAGEE_RELRSEED_ANCHOR2_CORE5"
base.AXIS_DESC = "stagee_relrseed_anchor2_core5"
base.STAGE_ID = "E"

# Parent candidates from Stage D
base.STAGEB_AXIS = "StageD_MicroWide_anchor2_core5"

# E profiles: local LR windows with mild/no structure movement
base.C_PROFILES = {
    "E1": {
        "name": "relr_low",
        "max_len": 10,
        "dropout_delta": 0.00,
        "wd_mult": 1.00,
        "lr_mult": 0.82,
        "lr_span_mult": 0.75,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "su", "tau": 0.40, "lmd": 0.02, "lmd_sem": 0.06},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.15},
        "fame": {"num_experts": 3},
    },
    "E2": {
        "name": "relr_mid",
        "max_len": 10,
        "dropout_delta": 0.00,
        "wd_mult": 1.00,
        "lr_mult": 1.00,
        "lr_span_mult": 0.70,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "su", "tau": 0.40, "lmd": 0.02, "lmd_sem": 0.06},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.15},
        "fame": {"num_experts": 3},
    },
    "E3": {
        "name": "relr_high",
        "max_len": 10,
        "dropout_delta": 0.00,
        "wd_mult": 1.00,
        "lr_mult": 1.18,
        "lr_span_mult": 0.75,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "su", "tau": 0.40, "lmd": 0.02, "lmd_sem": 0.06},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.15},
        "fame": {"num_experts": 3},
    },
    "E4": {
        "name": "relr_vlow_outlier",
        "max_len": 10,
        "dropout_delta": 0.01,
        "wd_mult": 1.10,
        "lr_mult": 0.68,
        "lr_span_mult": 0.82,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.55, "lmd": 0.03, "lmd_sem": 0.02},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.10},
        "fame": {"num_experts": 3},
    },
    "E5": {
        "name": "relr_vhigh_outlier",
        "max_len": 10,
        "dropout_delta": -0.01,
        "wd_mult": 0.92,
        "lr_mult": 1.36,
        "lr_span_mult": 0.82,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "us_x", "tau": 0.28, "lmd": 0.04, "lmd_sem": 0.10},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.22},
        "fame": {"num_experts": 4},
    },
}

base.PROFILE_ORDER = list(base.C_PROFILES.keys())
base.PROFILE_COST_WEIGHT.update({
    "E1": 0.95,
    "E2": 1.00,
    "E3": 1.00,
    "E4": 1.05,
    "E5": 1.05,
})


def _load_parent_candidates(dataset: str, model_option: str, model_label: str, topk: int):
    """Load parent candidates from Stage D output first, then Stage B-format fallback."""
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
