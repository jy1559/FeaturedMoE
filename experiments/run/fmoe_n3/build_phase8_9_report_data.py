#!/usr/bin/env python3
"""Build unified phase8/8_2/9(+9_2) report tables for KuaiRec FMoE_N3."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


# -------- constants --------

HVAR_CONFIG: Dict[str, Dict[str, float]] = {
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

P9_PROFILE_MAP: Dict[Tuple[str, str], Tuple[str, str, str]] = {
    ("C0", "N1"): ("Natural", "none", "none"),
    ("C0", "N2"): ("Natural", "route_smoothness", "none"),
    ("C0", "N3"): ("Natural", "balance", "z"),
    ("C0", "N4"): ("Natural", "z", "none"),
    ("C1", "B1"): ("CanonicalBalance", "balance", "z"),
    ("C1", "B2"): ("CanonicalBalance", "balance_strong", "z"),
    ("C1", "B3"): ("CanonicalBalance", "factored_group_balance", "z"),
    ("C1", "B4"): ("CanonicalBalance", "primitive_balance", "z"),
    ("C2", "S1"): ("Specialization", "route_sharpness", "none"),
    ("C2", "S2"): ("Specialization", "route_sharpness_strong", "none"),
    ("C2", "S3"): ("Specialization", "route_sharpness", "route_monopoly"),
    ("C2", "S4"): ("Specialization", "route_smoothness", "route_sharpness"),
    ("C3", "F1"): ("FeatureAlignment", "route_prior", "none"),
    ("C3", "F2"): ("FeatureAlignment", "route_prior_strong", "z"),
    ("C3", "F3"): ("FeatureAlignment", "group_prior_align", "none"),
    ("C3", "F4"): ("FeatureAlignment", "wrapper_group_feature_align", "group_prior_align"),
}

P9_BASE_META: Dict[str, Dict[str, str]] = {
    "B1": {"wrapper_combo": "all_w2", "bias_mode": "bias_rule", "source_profile": "src_base"},
    "B2": {"wrapper_combo": "mixed_2", "bias_mode": "bias_group_feat", "source_profile": "src_base"},
    "B3": {"wrapper_combo": "mixed_2", "bias_mode": "bias_both", "source_profile": "src_base"},
    "B4": {"wrapper_combo": "mixed_2", "bias_mode": "bias_both", "source_profile": "src_abc_feature"},
}


# -------- helpers --------


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def to_int(value: Any, default: int = 0) -> int:
    v = to_float(value)
    if v is None:
        return default
    return int(v)


def parse_bool(text: str) -> bool:
    v = str(text).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool: {text}")


def parse_timestamp(text: str, fallback_mtime: float) -> float:
    raw = str(text or "").strip()
    if raw:
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            pass
    return float(fallback_mtime)


def infer_status(n_completed: int, min_completed: int) -> str:
    if n_completed >= min_completed:
        return "completed"
    if n_completed > 0:
        return "partial"
    return "pending"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        rows = [{"note": "no rows"}]
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -------- loading --------


@dataclass
class RunEntry:
    run_phase: str
    run_dir: Path
    run_summary: Dict[str, Any]
    ts: float


class ReportBuilder:
    def __init__(self, repo_root: Path, dataset: str, min_completed: int, include_phase9_2: bool, out_dir: Path):
        self.repo_root = repo_root
        self.dataset = dataset
        self.min_completed = int(min_completed)
        self.include_phase9_2 = include_phase9_2
        self.out_dir = out_dir

        self.logging_root = (
            self.repo_root
            / "experiments"
            / "run"
            / "artifacts"
            / "logging"
            / "fmoe_n3"
            / self.dataset
        )
        self.logs_root = (
            self.repo_root
            / "experiments"
            / "run"
            / "artifacts"
            / "logs"
            / "fmoe_n3"
        )

        self.phase82_matrix_map = self._load_phase82_matrix_map()
        self.phase92_candidate_map = self._load_phase92_candidate_map()
        self.phase92_summary_map = self._load_phase92_summary_map()

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_phase82_matrix_map(self) -> Dict[str, Dict[str, Any]]:
        path = (
            self.logs_root
            / "phase8_2_verification_v1"
            / "P8_2"
            / self.dataset
            / "verification_matrix.json"
        )
        obj = self._load_json(path)
        out: Dict[str, Dict[str, Any]] = {}
        if not obj:
            return out
        for row in obj.get("rows", []):
            rid = str(row.get("run_id", "")).strip()
            if rid:
                out[rid] = dict(row)
        return out

    def _load_phase92_candidate_map(self) -> Dict[str, Dict[str, Any]]:
        path = (
            self.logs_root
            / "phase9_2_verification_v2"
            / "P9_2"
            / self.dataset
            / "verification_matrix.json"
        )
        obj = self._load_json(path)
        out: Dict[str, Dict[str, Any]] = {}
        if not obj:
            return out
        for row in obj.get("candidates", []):
            cid = str(row.get("candidate_id", "")).strip()
            if cid:
                out[cid] = dict(row)
        return out

    def _load_phase92_summary_map(self) -> Dict[str, Dict[str, Any]]:
        path = (
            self.logs_root
            / "phase9_2_verification_v2"
            / "P9_2"
            / self.dataset
            / "summary.csv"
        )
        out: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return out
        try:
            with path.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            for row in rows:
                run_phase = str(row.get("run_phase", "")).strip()
                if run_phase:
                    out[run_phase] = dict(row)
        except Exception:
            return {}
        return out

    def load_latest_runs(self) -> Dict[str, RunEntry]:
        latest: Dict[str, RunEntry] = {}
        for phase_dir in (self.logging_root / "P8", self.logging_root / "P9"):
            if not phase_dir.exists():
                continue
            for run_dir in sorted(phase_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                summary_path = run_dir / "run_summary.json"
                if not summary_path.exists():
                    continue
                payload = self._load_json(summary_path)
                if not payload:
                    continue
                run_phase = str(payload.get("run_phase", "")).strip()
                if not run_phase:
                    continue
                ts = parse_timestamp(payload.get("timestamp", ""), run_dir.stat().st_mtime)
                entry = RunEntry(run_phase=run_phase, run_dir=run_dir, run_summary=payload, ts=ts)

                prev = latest.get(run_phase)
                if prev is None:
                    latest[run_phase] = entry
                    continue

                keep_new = False
                if entry.ts > prev.ts:
                    keep_new = True
                elif entry.ts == prev.ts:
                    n_new = to_int(entry.run_summary.get("n_completed", 0), 0)
                    n_prev = to_int(prev.run_summary.get("n_completed", 0), 0)
                    if n_new > n_prev:
                        keep_new = True
                    elif n_new == n_prev:
                        b_new = to_float(entry.run_summary.get("best_mrr@20"))
                        b_prev = to_float(prev.run_summary.get("best_mrr@20"))
                        if (b_new or -1e9) > (b_prev or -1e9):
                            keep_new = True
                        elif (b_new or -1e9) == (b_prev or -1e9) and str(entry.run_dir) > str(prev.run_dir):
                            keep_new = True
                if keep_new:
                    latest[run_phase] = entry
        return latest

    @staticmethod
    def phase_tag(run_phase: str) -> Optional[str]:
        rp = str(run_phase)
        if rp.startswith("P8_SCR_") or rp.startswith("P8_CFM_"):
            return "phase8"
        if rp.startswith("P8_2_"):
            return "phase8_2"
        if rp.startswith("P9_B"):
            return "phase9"
        if rp.startswith("P9_2_"):
            return "phase9_2"
        return None

    @staticmethod
    def parse_phase8_run_phase(run_phase: str) -> Dict[str, Any]:
        parts = run_phase.split("_")
        out: Dict[str, Any] = {
            "pass_tag": "",
            "stage": "",
            "axis_group": "",
            "setting_id": "",
            "seed_id": None,
        }
        if len(parts) < 5:
            return out
        pass_tag = parts[1]
        stage = parts[2]
        setting_tokens = parts[3:-1]
        seed_tok = parts[-1]
        seed_id = None
        if seed_tok.startswith("S"):
            seed_id = to_int(seed_tok[1:], default=0)
        setting_id = "_".join(setting_tokens)

        out.update(
            {
                "pass_tag": pass_tag,
                "stage": stage,
                "axis_group": "CFM" if pass_tag == "CFM" else stage,
                "setting_id": setting_id,
                "seed_id": seed_id,
            }
        )
        return out

    @staticmethod
    def parse_phase8_setting_tokens(setting_id: str, stage: str) -> Dict[str, str]:
        sid = str(setting_id or "").lower()

        wrapper = ""
        for key in ["all_w1", "all_w2", "all_w3", "all_w4", "all_w5", "all_w6", "mixed_1", "mixed_2", "mixed_3"]:
            if key in sid:
                wrapper = key
                break

        bias = ""
        bias_checks = [
            ("bias_group_feat_rule", "bias_group_feat_rule"),
            ("bias_group_feat", "bias_group_feat"),
            ("bias_rule", "bias_rule"),
            ("bias_both", "bias_both"),
            ("bias_feat", "bias_feat"),
            ("bias_off", "bias_off"),
        ]
        for pat, name in bias_checks:
            if pat in sid:
                bias = name
                break

        source = ""
        source_checks = [
            ("src_a_hidden_b_d_feature", "src_a_hidden_b_d_feature"),
            ("src_all_both", "src_all_both"),
            ("src_abc_feature", "src_abc_feature"),
            ("src_base", "src_base"),
        ]
        for pat, name in source_checks:
            if pat in sid:
                source = name
                break

        topk = ""
        topk_checks = [
            ("tk_a3_d1_final4", "tk_a3_d1_final4"),
            ("tk_d1_final4", "tk_d1_final4"),
            ("tk_d1", "tk_d1"),
            ("tk_dense", "tk_dense"),
        ]
        for pat, name in topk_checks:
            if pat in sid:
                topk = name
                break

        if stage == "A":
            if not bias:
                bias = "bias_off"
            if not source:
                source = "src_default"
            if not topk:
                topk = "tk_dense"
        return {
            "wrapper_combo": wrapper,
            "bias_mode": bias,
            "source_profile": source,
            "topk_profile": topk,
        }

    @staticmethod
    def parse_phase82_run_phase(run_phase: str) -> Dict[str, Any]:
        # P8_2_A_H1_S1
        parts = run_phase.split("_")
        out: Dict[str, Any] = {
            "base_id": "",
            "hvar_id": "",
            "seed_id": None,
            "setting_id": "",
            "axis_group": "",
            "run_id": "",
        }
        if len(parts) != 5:
            return out
        base_id, hvar_id, seed_tok = parts[2], parts[3], parts[4]
        seed_id = None
        if seed_tok.startswith("S"):
            seed_id = to_int(seed_tok[1:], default=0)
        run_id = f"{base_id}_{hvar_id}_S{seed_id}" if seed_id is not None else ""
        out.update(
            {
                "base_id": base_id,
                "hvar_id": hvar_id,
                "seed_id": seed_id,
                "setting_id": f"{base_id}_{hvar_id}",
                "axis_group": base_id,
                "run_id": run_id,
            }
        )
        return out

    @staticmethod
    def parse_phase9_run_phase(run_phase: str) -> Dict[str, Any]:
        # P9_B1_C0_N4_S1
        parts = run_phase.split("_")
        out: Dict[str, Any] = {
            "base_id": "",
            "concept_id": "",
            "combo_id": "",
            "seed_id": None,
            "setting_id": "",
            "axis_group": "",
        }
        if len(parts) != 5:
            return out
        base_id, concept_id, combo_id, seed_tok = parts[1], parts[2], parts[3], parts[4]
        seed_id = None
        if seed_tok.startswith("S"):
            seed_id = to_int(seed_tok[1:], default=0)
        out.update(
            {
                "base_id": base_id,
                "concept_id": concept_id,
                "combo_id": combo_id,
                "seed_id": seed_id,
                "setting_id": f"{base_id}_{concept_id}_{combo_id}",
                "axis_group": concept_id,
            }
        )
        return out

    @staticmethod
    def parse_phase92_run_phase(run_phase: str) -> Dict[str, Any]:
        # P9_2_K1_B4_C0_N4_H1_S1
        parts = run_phase.split("_")
        out: Dict[str, Any] = {
            "candidate_id": "",
            "base_id": "",
            "concept_id": "",
            "combo_id": "",
            "hvar_id": "",
            "seed_id": None,
            "setting_id": "",
            "axis_group": "",
            "run_id": "",
        }
        if len(parts) != 8:
            return out
        candidate_id, base_id, concept_id, combo_id, hvar_id, seed_tok = (
            parts[2],
            parts[3],
            parts[4],
            parts[5],
            parts[6],
            parts[7],
        )
        seed_id = None
        if seed_tok.startswith("S"):
            seed_id = to_int(seed_tok[1:], default=0)
        out.update(
            {
                "candidate_id": candidate_id,
                "base_id": base_id,
                "concept_id": concept_id,
                "combo_id": combo_id,
                "hvar_id": hvar_id,
                "seed_id": seed_id,
                "setting_id": f"{candidate_id}_{hvar_id}",
                "axis_group": candidate_id,
                "run_id": f"{candidate_id}_{base_id}_{concept_id}_{combo_id}_{hvar_id}_S{seed_id}",
            }
        )
        return out

    @staticmethod
    def extract_special(run_dir: Path) -> Dict[str, Any]:
        path = run_dir / "special_metrics.json"
        out: Dict[str, Any] = {
            "special_test_overall_mrr20": None,
            "special_test_overall_hit20": None,
            "cold_item_mrr20": None,
            "cold_item_count": None,
            "sess_1_2_mrr20": None,
            "sess_1_2_count": None,
            "sess_3_5_mrr20": None,
            "sess_3_5_count": None,
            "special_available": 0,
        }
        if not path.exists():
            return out
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            test = obj.get("test_special_metrics", {}) or {}
            overall = test.get("overall", {}) or {}
            slices = test.get("slices", {}) or {}
            counts = test.get("counts", {}) or {}

            cold = ((slices.get("target_popularity_abs", {}) or {}).get("<=5", {}) or {})
            sess12 = ((slices.get("session_len", {}) or {}).get("1-2", {}) or {})
            sess35 = ((slices.get("session_len", {}) or {}).get("3-5", {}) or {})
            count_cold = ((counts.get("target_popularity_abs", {}) or {}).get("<=5", None))
            count_s12 = ((counts.get("session_len", {}) or {}).get("1-2", None))
            count_s35 = ((counts.get("session_len", {}) or {}).get("3-5", None))

            out.update(
                {
                    "special_test_overall_mrr20": to_float(overall.get("mrr@20")),
                    "special_test_overall_hit20": to_float(overall.get("hit@20")),
                    "cold_item_mrr20": to_float(cold.get("mrr@20")),
                    "cold_item_count": to_int(count_cold, default=0) if count_cold is not None else None,
                    "sess_1_2_mrr20": to_float(sess12.get("mrr@20")),
                    "sess_1_2_count": to_int(count_s12, default=0) if count_s12 is not None else None,
                    "sess_3_5_mrr20": to_float(sess35.get("mrr@20")),
                    "sess_3_5_count": to_int(count_s35, default=0) if count_s35 is not None else None,
                    "special_available": 1,
                }
            )
        except Exception:
            return out
        return out

    @staticmethod
    def extract_diag(run_dir: Path, run_summary: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer explicit pointer, fallback to canonical path.
        path_text = str(run_summary.get("diag_overview_table_csv", "") or "").strip()
        path = Path(path_text) if path_text else (run_dir / "diag" / "raw" / "overview_table.csv")
        out: Dict[str, Any] = {
            "diag_available": 0,
            "diag_n_eff": None,
            "diag_cv_usage": None,
            "diag_top1_max_frac": None,
            "diag_entropy_mean": None,
            "diag_route_jitter_adjacent": None,
            "diag_route_consistency_knn_score": None,
            "diag_route_consistency_knn_js": None,
            "diag_route_consistency_group_knn_score": None,
            "diag_route_consistency_group_knn_js": None,
            "diag_route_consistency_intra_group_knn_mean_score": None,
            "diag_route_consistency_intra_group_knn_mean_js": None,
            "diag_family_top_expert_mean_share": None,
            "diag_source_row_count": 0,
        }
        if not path.exists():
            return out
        try:
            with path.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            if not rows:
                return out
            row = rows[0]
            out.update(
                {
                    "diag_available": 1,
                    "diag_source_row_count": len(rows),
                    "diag_n_eff": to_float(row.get("n_eff")),
                    "diag_cv_usage": to_float(row.get("cv_usage")),
                    "diag_top1_max_frac": to_float(row.get("top1_max_frac")),
                    "diag_entropy_mean": to_float(row.get("entropy_mean")),
                    "diag_route_jitter_adjacent": to_float(row.get("route_jitter_adjacent")),
                    "diag_route_consistency_knn_score": to_float(row.get("route_consistency_knn_score")),
                    "diag_route_consistency_knn_js": to_float(row.get("route_consistency_knn_js")),
                    "diag_route_consistency_group_knn_score": to_float(row.get("route_consistency_group_knn_score")),
                    "diag_route_consistency_group_knn_js": to_float(row.get("route_consistency_group_knn_js")),
                    "diag_route_consistency_intra_group_knn_mean_score": to_float(
                        row.get("route_consistency_intra_group_knn_mean_score")
                    ),
                    "diag_route_consistency_intra_group_knn_mean_js": to_float(
                        row.get("route_consistency_intra_group_knn_mean_js")
                    ),
                    "diag_family_top_expert_mean_share": to_float(row.get("family_top_expert_mean_share")),
                }
            )
        except Exception:
            return out
        return out

    @staticmethod
    def extract_heatmap_vectors(run_dir: Path, run_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        path_text = str(run_summary.get("diag_raw_best_valid_json", "") or "").strip()
        path = Path(path_text) if path_text else (run_dir / "diag" / "raw" / "best_valid_diag.json")
        if not path.exists():
            return []
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            stage_metrics = obj.get("stage_metrics", {}) or {}
            stage = stage_metrics.get("macro@1")
            if not isinstance(stage, dict):
                if stage_metrics:
                    # fallback: first available stage
                    stage = stage_metrics[next(iter(stage_metrics.keys()))]
                else:
                    return []
            families = ((stage.get("feature_family_expert_heatmap") or {}).get("family_names") or [])
            values = ((stage.get("feature_family_expert_heatmap") or {}).get("values") or [])
            experts = stage.get("expert_names") or []
            if not (isinstance(families, list) and isinstance(values, list) and isinstance(experts, list)):
                return []
            if not families or not experts or len(values) != len(families):
                return []

            rows: List[Dict[str, Any]] = []
            for i, fam in enumerate(families):
                vec = values[i] if i < len(values) else []
                if not isinstance(vec, list) or len(vec) != len(experts):
                    continue
                row_sum = float(sum(float(x) for x in vec))
                if row_sum <= 0:
                    norm = [0.0 for _ in vec]
                else:
                    norm = [float(x) / row_sum for x in vec]
                rows.append(
                    {
                        "family": str(fam),
                        "experts": [str(e) for e in experts],
                        "weights_norm": norm,
                    }
                )
            return rows
        except Exception:
            return []

    def build_rows(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        latest = self.load_latest_runs()
        all_rows: List[Dict[str, Any]] = []
        heatmap_points_raw: List[Dict[str, Any]] = []
        existing_phase92: Set[str] = set()

        for run_phase, entry in sorted(latest.items(), key=lambda kv: kv[0]):
            phase_tag = self.phase_tag(run_phase)
            if phase_tag is None:
                continue
            if phase_tag == "phase9_2" and not self.include_phase9_2:
                continue

            rs = entry.run_summary
            n_completed = to_int(rs.get("n_completed", 0), 0)
            best_valid = to_float(rs.get("best_mrr@20"))
            test_mrr = to_float(rs.get("test_mrr@20"))

            row: Dict[str, Any] = {
                "dataset": self.dataset,
                "run_phase": run_phase,
                "phase_tag": phase_tag,
                "status": infer_status(n_completed, self.min_completed),
                "is_main_eligible": 1 if n_completed >= self.min_completed else 0,
                "n_completed": n_completed,
                "best_valid_mrr20": best_valid,
                "test_mrr20": test_mrr,
                "test_hr10": to_float(rs.get("test_hr@10")),
                "timestamp": rs.get("timestamp", ""),
                "run_dir": str(entry.run_dir),
                "axis_group": "",
                "setting_id": "",
                "pass_tag": "",
                "stage": "",
                "wrapper_combo": "",
                "bias_mode": "",
                "source_profile": "",
                "topk_profile": "",
                "base_id": "",
                "concept_id": "",
                "concept_name": "",
                "combo_id": "",
                "candidate_id": "",
                "hvar_id": "",
                "seed_id": None,
                "main_aux": "",
                "support_aux": "",
                "embedding_size": None,
                "d_ff": None,
                "d_expert_hidden": None,
                "d_router_hidden": None,
                "fixed_weight_decay": None,
                "fixed_hidden_dropout_prob": None,
            }

            if phase_tag == "phase8":
                p8 = self.parse_phase8_run_phase(run_phase)
                row.update(p8)
                row.update(self.parse_phase8_setting_tokens(p8.get("setting_id", ""), p8.get("stage", "")))

            elif phase_tag == "phase8_2":
                p82 = self.parse_phase82_run_phase(run_phase)
                row.update(p82)
                run_id = p82.get("run_id", "")
                mrow = self.phase82_matrix_map.get(str(run_id), {})
                row["bias_mode"] = str(mrow.get("bias_mode", "") or "")
                row["source_profile"] = str(mrow.get("source_profile", "") or "")
                row["topk_profile"] = str(mrow.get("topk_profile", "") or "")
                hp = mrow.get("hparams", {}) or {}
                row["embedding_size"] = to_int(hp.get("embedding_size"), 0) if hp else None
                row["d_ff"] = to_int(hp.get("d_ff"), 0) if hp else None
                row["d_expert_hidden"] = to_int(hp.get("d_expert_hidden"), 0) if hp else None
                row["d_router_hidden"] = to_int(hp.get("d_router_hidden"), 0) if hp else None
                row["fixed_weight_decay"] = to_float(hp.get("fixed_weight_decay"))
                row["fixed_hidden_dropout_prob"] = to_float(hp.get("fixed_hidden_dropout_prob"))
                # wrapper by base definition
                base_wrapper = {
                    "A": "all_w5",
                    "B": "mixed_2",
                    "C": "all_w5",
                    "D": "all_w2",
                }
                row["wrapper_combo"] = base_wrapper.get(str(row.get("base_id", "")), "")

            elif phase_tag == "phase9":
                p9 = self.parse_phase9_run_phase(run_phase)
                row.update(p9)
                concept = str(row.get("concept_id", ""))
                combo = str(row.get("combo_id", ""))
                profile = P9_PROFILE_MAP.get((concept, combo))
                if profile:
                    row["concept_name"], row["main_aux"], row["support_aux"] = profile
                base_meta = P9_BASE_META.get(str(row.get("base_id", "")), {})
                row["wrapper_combo"] = base_meta.get("wrapper_combo", "")
                row["bias_mode"] = base_meta.get("bias_mode", "")
                row["source_profile"] = base_meta.get("source_profile", "")

            elif phase_tag == "phase9_2":
                p92 = self.parse_phase92_run_phase(run_phase)
                row.update(p92)
                candidate = self.phase92_candidate_map.get(str(row.get("candidate_id", "")), {})
                row["main_aux"] = str(candidate.get("main_aux", "") or "")
                row["support_aux"] = str(candidate.get("support_aux", "") or "")
                if not row.get("base_id"):
                    row["base_id"] = str(candidate.get("base_id", "") or "")
                if not row.get("concept_id"):
                    row["concept_id"] = str(candidate.get("concept_id", "") or "")
                if not row.get("combo_id"):
                    row["combo_id"] = str(candidate.get("combo_id", "") or "")
                profile = P9_PROFILE_MAP.get((str(row.get("concept_id", "")), str(row.get("combo_id", ""))))
                if profile:
                    row["concept_name"] = profile[0]
                base_meta = P9_BASE_META.get(str(row.get("base_id", "")), {})
                row["wrapper_combo"] = base_meta.get("wrapper_combo", "")
                row["bias_mode"] = base_meta.get("bias_mode", "")
                row["source_profile"] = base_meta.get("source_profile", "")
                h = HVAR_CONFIG.get(str(row.get("hvar_id", "")), {})
                if h:
                    row["embedding_size"] = to_int(h.get("embedding_size"), 0)
                    row["d_ff"] = to_int(h.get("d_ff"), 0)
                    row["d_expert_hidden"] = to_int(h.get("d_expert_hidden"), 0)
                    row["d_router_hidden"] = to_int(h.get("d_router_hidden"), 0)
                    row["fixed_weight_decay"] = to_float(h.get("fixed_weight_decay"))
                    row["fixed_hidden_dropout_prob"] = to_float(h.get("fixed_hidden_dropout_prob"))
                existing_phase92.add(run_phase)

            # attach special + diag
            row.update(self.extract_special(entry.run_dir))
            row.update(self.extract_diag(entry.run_dir, rs))
            all_rows.append(row)

            # collect heatmap vectors for PCA from main-eligible runs
            if row["is_main_eligible"] == 1:
                heat_rows = self.extract_heatmap_vectors(entry.run_dir, rs)
                for hrow in heat_rows:
                    heatmap_points_raw.append(
                        {
                            "run_phase": run_phase,
                            "phase_tag": phase_tag,
                            "axis_group": row.get("axis_group", ""),
                            "setting_id": row.get("setting_id", ""),
                            "family": hrow["family"],
                            "experts": hrow["experts"],
                            "weights_norm": hrow["weights_norm"],
                            "best_valid_mrr20": row.get("best_valid_mrr20"),
                            "test_mrr20": row.get("test_mrr20"),
                        }
                    )

        # phase9_2 pending runs may not have run_summary.json yet; restore from summary.csv
        if self.include_phase9_2 and self.phase92_summary_map:
            for run_phase, srow in sorted(self.phase92_summary_map.items(), key=lambda kv: kv[0]):
                if run_phase in existing_phase92:
                    continue

                n_completed = to_int(srow.get("n_completed"), 0)
                status_hint = str(srow.get("status", "")).strip().lower()
                inferred = infer_status(n_completed, self.min_completed)
                if status_hint == "pending" and n_completed <= 0:
                    inferred = "pending"

                best_valid = to_float(srow.get("run_best_valid_mrr20"))
                test_mrr = to_float(srow.get("test_mrr20"))
                p92 = self.parse_phase92_run_phase(run_phase)
                candidate = self.phase92_candidate_map.get(str(p92.get("candidate_id", "")), {})
                profile = P9_PROFILE_MAP.get((str(p92.get("concept_id", "")), str(p92.get("combo_id", ""))))
                base_meta = P9_BASE_META.get(str(p92.get("base_id", "")), {})
                h = HVAR_CONFIG.get(str(p92.get("hvar_id", "")), {})

                row: Dict[str, Any] = {
                    "dataset": self.dataset,
                    "run_phase": run_phase,
                    "phase_tag": "phase9_2",
                    "status": inferred,
                    "is_main_eligible": 1 if n_completed >= self.min_completed else 0,
                    "n_completed": n_completed,
                    "best_valid_mrr20": best_valid,
                    "test_mrr20": test_mrr,
                    "test_hr10": None,
                    "timestamp": srow.get("timestamp_utc", ""),
                    "run_dir": "",
                    "axis_group": p92.get("axis_group", ""),
                    "setting_id": p92.get("setting_id", ""),
                    "pass_tag": "",
                    "stage": "",
                    "wrapper_combo": base_meta.get("wrapper_combo", ""),
                    "bias_mode": base_meta.get("bias_mode", ""),
                    "source_profile": base_meta.get("source_profile", ""),
                    "topk_profile": "",
                    "base_id": p92.get("base_id", ""),
                    "concept_id": p92.get("concept_id", ""),
                    "concept_name": profile[0] if profile else "",
                    "combo_id": p92.get("combo_id", ""),
                    "candidate_id": p92.get("candidate_id", ""),
                    "hvar_id": p92.get("hvar_id", ""),
                    "seed_id": p92.get("seed_id"),
                    "run_id": p92.get("run_id", ""),
                    "main_aux": str(candidate.get("main_aux", "") or ""),
                    "support_aux": str(candidate.get("support_aux", "") or ""),
                    "embedding_size": to_int(h.get("embedding_size"), 0) if h else None,
                    "d_ff": to_int(h.get("d_ff"), 0) if h else None,
                    "d_expert_hidden": to_int(h.get("d_expert_hidden"), 0) if h else None,
                    "d_router_hidden": to_int(h.get("d_router_hidden"), 0) if h else None,
                    "fixed_weight_decay": to_float(h.get("fixed_weight_decay")) if h else None,
                    "fixed_hidden_dropout_prob": to_float(h.get("fixed_hidden_dropout_prob")) if h else None,
                    "special_test_overall_mrr20": None,
                    "special_test_overall_hit20": None,
                    "cold_item_mrr20": None,
                    "cold_item_count": None,
                    "sess_1_2_mrr20": None,
                    "sess_1_2_count": None,
                    "sess_3_5_mrr20": None,
                    "sess_3_5_count": None,
                    "special_available": 0,
                    "diag_available": 0,
                    "diag_n_eff": None,
                    "diag_cv_usage": None,
                    "diag_top1_max_frac": None,
                    "diag_entropy_mean": None,
                    "diag_route_jitter_adjacent": None,
                    "diag_route_consistency_knn_score": None,
                    "diag_route_consistency_knn_js": None,
                    "diag_route_consistency_group_knn_score": None,
                    "diag_route_consistency_group_knn_js": None,
                    "diag_route_consistency_intra_group_knn_mean_score": None,
                    "diag_route_consistency_intra_group_knn_mean_js": None,
                    "diag_family_top_expert_mean_share": None,
                    "diag_source_row_count": 0,
                }
                all_rows.append(row)

        return all_rows, heatmap_points_raw

    @staticmethod
    def sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stage_order = {"A": 0, "B": 1, "C": 2, "D": 3, "CFM": 4}
        concept_order = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}

        def key(row: Dict[str, Any]) -> Tuple:
            phase = str(row.get("phase_tag", ""))
            if phase == "phase8":
                return (
                    0,
                    stage_order.get(str(row.get("axis_group", "")), 99),
                    str(row.get("setting_id", "")),
                    to_int(row.get("seed_id"), 0),
                )
            if phase == "phase8_2":
                return (
                    1,
                    str(row.get("base_id", "")),
                    str(row.get("hvar_id", "")),
                    to_int(row.get("seed_id"), 0),
                )
            if phase == "phase9":
                return (
                    2,
                    str(row.get("base_id", "")),
                    concept_order.get(str(row.get("concept_id", "")), 99),
                    str(row.get("combo_id", "")),
                    to_int(row.get("seed_id"), 0),
                )
            if phase == "phase9_2":
                return (
                    3,
                    str(row.get("candidate_id", "")),
                    str(row.get("hvar_id", "")),
                    to_int(row.get("seed_id"), 0),
                )
            return (99, str(row.get("run_phase", "")))

        return sorted(rows, key=key)

    def build_phase_tables(self, all_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        by_phase: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in all_rows:
            by_phase[str(row.get("phase_tag", ""))].append(row)

        phase8_all = self.sort_rows(by_phase.get("phase8", []))
        phase82_all = self.sort_rows(by_phase.get("phase8_2", []))
        phase9_all = self.sort_rows(by_phase.get("phase9", []))
        phase92_all = self.sort_rows(by_phase.get("phase9_2", []))

        phase8_main = [r for r in phase8_all if int(r.get("is_main_eligible", 0)) == 1]
        phase82_main = [r for r in phase82_all if int(r.get("is_main_eligible", 0)) == 1]
        phase9_main = [r for r in phase9_all if int(r.get("is_main_eligible", 0)) == 1]
        phase92_main = [r for r in phase92_all if int(r.get("is_main_eligible", 0)) == 1]

        phase_nonmain = [
            r
            for r in self.sort_rows(all_rows)
            if int(r.get("is_main_eligible", 0)) == 0 and str(r.get("phase_tag", "")) in {"phase8", "phase8_2", "phase9", "phase9_2"}
        ]
        phase92_pending = [r for r in phase92_all if int(r.get("is_main_eligible", 0)) == 0]

        return {
            "phase8_main": phase8_main,
            "phase8_2_main": phase82_main,
            "phase9_main": phase9_main,
            "phase9_2_main": phase92_main,
            "phase9_2_pending": phase92_pending,
            "phase_nonmain": phase_nonmain,
            "diag_special_join": self.sort_rows(all_rows),
        }

    def build_pca_points(self, heat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not heat_rows:
            return []

        # matrix rows are (run_phase,family) vectors over experts
        ids: List[Tuple[str, str]] = []
        meta: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        expert_names_ref: Optional[List[str]] = None
        for row in heat_rows:
            experts = list(row.get("experts", []))
            weights = list(row.get("weights_norm", []))
            if not experts or len(experts) != len(weights):
                continue
            if expert_names_ref is None:
                expert_names_ref = experts
            if experts != expert_names_ref:
                # align to first expert order where possible
                idx = {e: i for i, e in enumerate(experts)}
                aligned = [weights[idx[e]] if e in idx else 0.0 for e in expert_names_ref]
                weights = aligned

            vec = np.array(weights, dtype=float)
            if vec.size == 0:
                continue
            ids.append((str(row.get("run_phase", "")), str(row.get("family", ""))))
            meta.append(row)
            vectors.append(vec)

        if not vectors or expert_names_ref is None:
            return []

        X = np.vstack(vectors)
        # PCA by SVD on centered vectors
        Xc = X - X.mean(axis=0, keepdims=True)
        if Xc.shape[0] >= 2 and Xc.shape[1] >= 2:
            u, s, vt = np.linalg.svd(Xc, full_matrices=False)
            k = min(2, vt.shape[0])
            components = vt[:k]
            coords = Xc.dot(components.T)
            var = (s**2) / max(Xc.shape[0] - 1, 1)
            denom = var.sum() if var.sum() > 0 else 1.0
            explained = (var / denom).tolist()
            pc1_var = explained[0] if len(explained) >= 1 else 0.0
            pc2_var = explained[1] if len(explained) >= 2 else 0.0
        else:
            coords = np.zeros((X.shape[0], 2), dtype=float)
            components = np.zeros((2, X.shape[1]), dtype=float)
            pc1_var = 0.0
            pc2_var = 0.0

        out_rows: List[Dict[str, Any]] = []
        for i, row in enumerate(meta):
            weights = list(row.get("weights_norm", []))
            pc1 = float(coords[i, 0]) if coords.shape[1] >= 1 else 0.0
            pc2 = float(coords[i, 1]) if coords.shape[1] >= 2 else 0.0
            for j, expert in enumerate(expert_names_ref):
                out_rows.append(
                    {
                        "run_phase": row.get("run_phase", ""),
                        "phase_tag": row.get("phase_tag", ""),
                        "axis_group": row.get("axis_group", ""),
                        "setting_id": row.get("setting_id", ""),
                        "family": row.get("family", ""),
                        "expert": expert,
                        "weight_norm": float(weights[j]) if j < len(weights) else 0.0,
                        "best_valid_mrr20": row.get("best_valid_mrr20"),
                        "test_mrr20": row.get("test_mrr20"),
                        "pc1": pc1,
                        "pc2": pc2,
                        "pc1_explained_var_ratio": pc1_var,
                        "pc2_explained_var_ratio": pc2_var,
                        "axis_label_x": f"PC1 ({pc1_var * 100.0:.2f}%)",
                        "axis_label_y": f"PC2 ({pc2_var * 100.0:.2f}%)",
                        "loading_pc1": float(components[0, j]) if components.shape[0] >= 1 else 0.0,
                        "loading_pc2": float(components[1, j]) if components.shape[0] >= 2 else 0.0,
                    }
                )

        return out_rows

    def write_tables(self, tables: Dict[str, List[Dict[str, Any]]], pca_rows: List[Dict[str, Any]]) -> None:
        write_csv(self.out_dir / "phase8_main.csv", tables["phase8_main"])
        write_csv(self.out_dir / "phase8_2_main.csv", tables["phase8_2_main"])
        write_csv(self.out_dir / "phase9_main.csv", tables["phase9_main"])
        write_csv(self.out_dir / "phase9_2_main.csv", tables["phase9_2_main"])
        write_csv(self.out_dir / "phase9_2_pending.csv", tables["phase9_2_pending"])
        write_csv(self.out_dir / "phase_nonmain.csv", tables["phase_nonmain"])
        write_csv(self.out_dir / "diag_special_join.csv", tables["diag_special_join"])
        write_csv(self.out_dir / "family_expert_pca_points.csv", pca_rows)

    def print_validation(self, tables: Dict[str, List[Dict[str, Any]]]) -> None:
        def count_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
            out: Dict[str, int] = defaultdict(int)
            for r in rows:
                out[str(r.get(key, ""))] += 1
            return dict(sorted(out.items(), key=lambda kv: kv[0]))

        p8 = tables["phase8_main"]
        p82 = tables["phase8_2_main"]
        p9 = tables["phase9_main"]
        p92 = tables["phase9_2_main"]
        p92_pending = tables["phase9_2_pending"]
        nonmain = tables["phase_nonmain"]

        print("[validation] phase8 main rows:", len(p8), "axis_group:", count_by(p8, "axis_group"))

        print(
            "[validation] phase8_2 main rows:",
            len(p82),
            "base:",
            count_by(p82, "base_id"),
            "hvar:",
            count_by(p82, "hvar_id"),
        )

        print("[validation] phase9 main rows:", len(p9), "base:", count_by(p9, "base_id"), "concept:", count_by(p9, "concept_id"))

        print("[validation] phase9_2 main rows:", len(p92), "candidate:", count_by(p92, "candidate_id"), "hvar:", count_by(p92, "hvar_id"))
        print("[validation] phase9_2 pending rows:", len(p92_pending), "candidate:", count_by(p92_pending, "candidate_id"))
        print("[validation] non-main rows:", len(nonmain), "phase_tag:", count_by(nonmain, "phase_tag"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase8/8_2/9(+9_2) report tables.")
    parser.add_argument("--repo-root", default="/workspace/jy1559/FMoE")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--min-completed", type=int, default=20)
    parser.add_argument("--include-phase9-2", type=parse_bool, default=True)
    parser.add_argument(
        "--out-dir",
        default="/workspace/jy1559/FMoE/experiments/run/fmoe_n3/docs/data/phase8_9",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = ReportBuilder(
        repo_root=Path(args.repo_root),
        dataset=args.dataset,
        min_completed=int(args.min_completed),
        include_phase9_2=bool(args.include_phase9_2),
        out_dir=Path(args.out_dir),
    )

    all_rows, heat_raw = builder.build_rows()
    tables = builder.build_phase_tables(all_rows)
    pca_rows = builder.build_pca_points(heat_raw)
    builder.write_tables(tables, pca_rows)
    builder.print_validation(tables)

    print(f"[done] wrote tables to: {builder.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
