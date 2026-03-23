# Phase 13 — Feature Sanity Wide Search (P13)

## 1) 목표
- 성능 향상이 feature-aware routing 효과인지(정렬된 hint 사용), 단순 파라미터 증가 효과인지 분리 검증한다.
- eval perturb / train corruption / semantic mismatch를 통해 feature alignment 민감도를 확인한다.

## 2) 실행 스펙 (고정)
- 실행 타입: `wide`
- setting 수: `24`
- hparam 수: `1 (H1)`
- seed 수: `1 (S1)`
- 총 run 수: `24 x 1 x 1 = 24`
- 실행 엔트리:
  - `experiments/run/fmoe_n3/run_phase13_feature_sanity.py`
  - `experiments/run/fmoe_n3/phase_13_feature_sanity.sh`

## 3) 로그/요약 규칙
- 로그 루트: `experiments/run/artifacts/logs/fmoe_n3/phase13_feature_sanity_v1/<dataset>/`
- 로그 파일명: `P13_<axis_id>_<axis_desc>_<setting_id>_<setting_desc>.log`
- summary: 동일 경로의 `summary.csv`
- summary append 이벤트(강제):
  - `trigger=trial_new_best`
  - `trigger=run_complete`
- resume 정책:
  - 완료 로그(`"[RUN_STATUS] END status=normal"`)는 skip
  - 미완료 로그는 preamble overwrite 후 재실행

## 4) Feature Perturb 코어 확장 (필수)
- N3 config 확장:
  - `feature_perturb_mode` (default: `none`)
  - `feature_perturb_apply` (default: `none`, `train|eval|both|none`)
  - `feature_perturb_family` (default: empty)
  - `feature_perturb_shift` (default: `1`)
- 지원 모드:
  - `zero`, `shuffle`, `global_permute`, `batch_permute`, `family_permute`, `position_shift`, `role_swap`, `stage_mismatch`
- 기존 `set_feature_ablation_mode` 경로와 충돌 없이 병행 동작하도록 유지.

## 5) 세팅 매트릭스 (24)

### 5.1 Data condition (2)
- `P13-00_FULL_DATA`
- `P13-01_CATEGORY_ZERO_DATA`

### 5.2 Eval perturb (6)
- `P13-02_EVAL_ALL_ZERO`
- `P13-03_EVAL_ALL_SHUFFLE`
- `P13-04_EVAL_SHUFFLE_TEMPO`
- `P13-05_EVAL_SHUFFLE_FOCUS`
- `P13-06_EVAL_SHUFFLE_MEMORY`
- `P13-07_EVAL_SHUFFLE_EXPOSURE`

### 5.3 Train corruption (6)
- `P13-08_TRAIN_GLOBAL_PERMUTE_ALL`
- `P13-09_TRAIN_BATCH_PERMUTE_ALL`
- `P13-10_TRAIN_PERMUTE_TEMPO`
- `P13-11_TRAIN_PERMUTE_FOCUS`
- `P13-12_TRAIN_PERMUTE_MEMORY`
- `P13-13_TRAIN_PERMUTE_EXPOSURE`

### 5.4 Semantic mismatch (3)
- `P13-14_FEATURE_ROLE_SWAP`
- `P13-15_STAGE_MISMATCH_ASSIGN`
- `P13-16_POSITION_SHIFT_FEATURE`

### 5.5 8배수 보강 (+7)
- `P13-17_EVAL_ZERO_TEMPO`
- `P13-18_EVAL_ZERO_FOCUS`
- `P13-19_EVAL_ZERO_MEMORY`
- `P13-20_EVAL_ZERO_EXPOSURE`
- `P13-21_TRAIN_POSITION_SHIFT_PLUS1`
- `P13-22_TRAIN_POSITION_SHIFT_PLUS2`
- `P13-23_TRAIN_POSITION_SHIFT_PLUS3`

## 6) 구현 메모
- `CATEGORY_ZERO_DATA`는 category/theme 계열 feature drop keyword 기반 proxy로 운영한다.
- phase runner는 perturb 설정(`feature_perturb_*`)만 세팅별로 주입하고 공통 실행/저장/summary는 wide 공용 유틸을 사용한다.
