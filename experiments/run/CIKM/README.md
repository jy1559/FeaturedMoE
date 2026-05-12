# CIKM 2026 – RouteRec Experiments

**Paper**: RouteRec: Behavior-Guided Expert Routing for Sequential Recommendation  
**Submission deadline**: Full paper May 23 2026 AoE  
**Datasets**: KuaiRec (full, 3.2M train, 8,966 items) · lastfm (full, 11M train, 547K items)  
**Feature mode**: `final` → `Datasets/processed/final_dataset/`  
**Eval mode**: `session_fixed` (pre-split `.train/.valid/.test.inter`)

---

## Folder Structure

```
CIKM/
├── common.py            ← shared infrastructure (search spaces, job runner)
│
├── results/             ← CSV summaries written here automatically
│   ├── run_full_result.md          ← 실험 결과 현황 요약
│   ├── main_baselines_summary.csv
│   ├── main_routerec_summary.csv
│   ├── resume_safe_summary.csv
│   └── cue_perturb_*.csv          ← exp_cue_perturb 결과 (실행 후 생성)
│
├── logs/                ← per-job outer logs (outer shell, not hyperopt internals)
│   ├── cue_perturb_eval/          ← eval_perturb.py 실행 로그
│   ├── cue_perturb_train/         ← train_perturb.py 실행 로그
│   └── cikm_resume_safe_parallel/ ← 이전 main 실험 로그
│
├── artifacts/           ← CIKM 실험 결과 JSON + 모델 체크포인트 (CIKM 전용)
│   ├── results/         ← exp_cue_perturb 결과 JSON (HYPEROPT_RESULTS_DIR)
│   └── logs/            ← hyperopt 내부 special logs (RUN_LOGS_DIR)
│
├── scripts/             ← main 실험 실행 스크립트 (P0 완료, 히스토리용)
│   ├── run_all.sh              ← baselines + RouteRec 전체 실행
│   ├── run_full_kuai_lfm.sh    ← KuaiRec + lastfm 전체 실행
│   ├── run_lfm_stable.sh       ← lastfm 안정적 재실행
│   └── run_resume_safe.sh      ← 실패/미완료 job 선별 재실행
│
├── exp_main/            ← P0: main comparison table
│   ├── main_baselines.py / .sh   (9 baselines × KuaiRec + lastfm)
│   └── main_routerec.py / .sh    (RouteRec × KuaiRec + lastfm)
│
├── exp_cue_perturb/     ← P1-A: cue-perturbation ablation
├── exp_capacity/        ← P1-B: capacity-matched SASRec-wide baseline
├── exp_seed/            ← P2-A: multi-seed robustness
└── exp_analysis/        ← P2-B: BRD analysis & routing visualisation
```

> **artifact 경로 정책**  
> - `scripts/` 및 `exp_main/` 실행 시: 결과 JSON → `experiments/run/artifacts/results/cikm/`  
> - `exp_cue_perturb/` 실행 시: 결과 JSON → `CIKM/artifacts/results/` (HYPEROPT_RESULTS_DIR 자동 설정)

---

## 실험 현황

→ 상세 결과: [`results/run_full_result.md`](results/run_full_result.md)

| 데이터셋 | 상태 |
|---------|------|
| KuaiRec | ✅ 완료 (10개 모델) |
| lastfm  | 🔄 진행 중 (6/10 완료, FAME/FDSA 실행 중) |

---

## Quick Start

### exp_cue_perturb (P1-A ablation) 실행

```bash
cd experiments/run/CIKM/exp_cue_perturb

# dry-run (실제 실행 없이 조건 확인)
bash demo_dry_run.sh

# eval-only (P0 checkpoint 재사용, ~1h)
python eval_perturb.py --gpu 0 --datasets KuaiRec

# train-time (새 학습 필요, ~30h full / ~13h 핵심만)
python train_perturb.py --gpus 0 --conditions hidden_only both_zero

# 전체 실행
bash run_all_perturb.sh 0 1
```

### main 실험 재실행 (필요 시)

```bash
# lastfm 미완료 job 재실행
cd experiments/run/CIKM
bash scripts/run_resume_safe.sh 0 1 2 3
```

---

## Configuration Details

| Setting | Value |
|---------|-------|
| Hyperopt algo | TPE |
| Max evals per job | 5 |
| Tune epochs | 100 |
| Early stopping patience | 10 |
| Valid metric | MRR@20 |
| KuaiRec eval | full (8,966 items) |
| lastfm eval | sampled (1,000 negatives, pop strategy) |
| Seed | 42 |

### Tuned params
Only `learning_rate` and `weight_decay` are tuned, in a narrow range around best configs from sampled-dataset experiments.  
All structural hyperparams are **fixed** to their best known values from `experiments/run/final_experiment/`.

---

## Adding a New Experiment

1. Add your runner script to the relevant `exp_*/` subfolder.
2. Import from `../common.py` (already on `sys.path` when the script is in an `exp_*` subfolder).
3. Call `run_jobs_queued(jobs, gpus=..., summary_path=..., run_axis=..., run_phase=...)`.
4. Results land in `CIKM/results/` automatically.
5. hyperopt 결과 JSON을 CIKM/artifacts에 집중하려면 subprocess env에 `HYPEROPT_RESULTS_DIR` 설정.
