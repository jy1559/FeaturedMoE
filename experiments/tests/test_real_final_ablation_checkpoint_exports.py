from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from run.final_experiment.real_final_ablation import common as real_common  # noqa: E402


def _base_row(question: str, setting_key: str) -> dict:
    return {
        "question": question,
        "run_axis": "appendix_cases",
        "run_phase": "CASES_BEAUTY_BEHAVIOR_GUIDED_R02_S1",
        "job_id": "CASES_BEAUTY_BEHAVIOR_GUIDED_R02_S1",
        "dataset": "beauty",
        "max_evals": 2,
        "max_run_hours": 1.0,
        "tune_epochs": 10,
        "tune_patience": 3,
        "runtime_seed": 123,
        "seed_id": 1,
        "base_rank": 2,
        "setting_key": setting_key,
        "search_space": {"learning_rate": [0.001]},
        "search_space_types": {"learning_rate": "choice"},
        "fixed_context": {},
    }


def test_build_route_command_includes_checkpoint_export_for_case_eval_rows():
    cmd = real_common.build_route_command(_base_row("cases", "behavior_guided"), "0", search_algo="random")
    joined = " ".join(cmd)
    assert "++artifact_export_final_checkpoint=true" in joined
    assert "++__artifact_combo_best_export_path=" in joined
    assert "CASES_BEAUTY_BEHAVIOR_GUIDED_R02_S1_best_model_state.pth" in joined


def test_build_route_command_skips_checkpoint_export_for_non_postprocess_rows():
    cmd = real_common.build_route_command(_base_row("objective", "balance_only"), "0", search_algo="random")
    joined = " ".join(cmd)
    assert "++artifact_export_final_checkpoint=false" in joined
    assert "++__artifact_combo_best_export_path=" not in joined