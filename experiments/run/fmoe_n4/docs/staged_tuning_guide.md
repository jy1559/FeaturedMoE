# FMoE_N4 staged tuning 가이드

## 개요
- `fmoe_n4`는 `baseline_2/run_staged_tuning.py` 인프라를 재사용한다.
- track만 `fmoe_n4`로 분리되어 로그/결과가 별도 경로에 저장된다.
- 기본 모델은 `featured_moe_n3` 1개다.
- stage 예산/승급/변이 정책도 baseline_2와 동일하게 동작한다.
  - 승급: `A->10`, `B->4`, `C(변이12)->2`, `D(변이6)->1`
  - balanced 예산: `B(30/50/5)`, `C(20/80/8)`, `D(12/100/10)` (`max_evals/epochs/patience`)

## 실행
- Stage별:
  - `stageA_fmoe_n4.sh`
  - `stageB_fmoe_n4.sh`
  - `stageC_fmoe_n4.sh`
  - `stageD_fmoe_n4.sh`
- 통합:
  - `run_all_stages.sh`

## 저장 경로
- `experiments/run/artifacts/logs/fmoe_n4/ABCD_v1/stages/stageA|B|C|D/`
- stage별 공통 출력:
  - `summary.csv`
  - `promotion.csv`
  - `leaderboard.csv`
  - `manifest.json`

## 평가 규칙
- `feature_mode=full_v4`
- `exclude_unseen_target_from_main_eval=true`
- `log_unseen_target_metrics=true`
- seen-main metric + unseen(cold) metric 동시 저장
