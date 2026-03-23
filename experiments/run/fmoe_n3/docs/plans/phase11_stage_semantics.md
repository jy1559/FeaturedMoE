# Phase 11 — Stage Semantics Wide Search (P11)

## 1) 목표
- `macro / mid / micro` stage 분해의 의미를 검증한다.
- stage 필요성, stage 순서 민감도, macro/mid routing granularity(session vs token)를 분리해서 본다.
- dense prepend가 stage decomposition을 대체하는지 확인한다.

## 2) 실행 스펙 (고정)
- 실행 타입: `wide`
- setting 수: `24`
- hparam 수: `1 (H1)`
- seed 수: `1 (S1)`
- 총 run 수: `24 x 1 x 1 = 24`
- 실행 엔트리:
  - `experiments/run/fmoe_n3/run_phase11_stage_semantics.py`
  - `experiments/run/fmoe_n3/phase_11_stage_semantics.sh`

## 3) 로그/요약 규칙
- 로그 루트: `experiments/run/artifacts/logs/fmoe_n3/phase11_stage_semantics_v1/<dataset>/`
- 로그 파일명: `P11_<axis_id>_<axis_desc>_<setting_id>_<setting_desc>.log`
- summary: 동일 경로의 `summary.csv`
- summary append 이벤트(강제):
  - `trigger=trial_new_best`
  - `trigger=run_complete`
- resume 정책:
  - 완료 로그(`"[RUN_STATUS] END status=normal"`)는 skip
  - 미완료 로그는 preamble overwrite 후 재실행

## 4) 세팅 매트릭스 (24)

### 4.1 Base stage ablation (7)
- `P11-00_MACRO_MID_MICRO`
- `P11-01_MID_MICRO`
- `P11-02_MACRO_MICRO`
- `P11-03_MACRO_MID`
- `P11-04_MACRO_ONLY`
- `P11-05_MID_ONLY`
- `P11-06_MICRO_ONLY`

### 4.2 Prepend dense layer (7)
- `P11-07_LAYER_MACRO_MID_MICRO`
- `P11-08_LAYER_MID_MICRO`
- `P11-09_LAYER_MACRO_MICRO`
- `P11-10_LAYER_MACRO_MID`
- `P11-11_LAYER_MACRO`
- `P11-12_LAYER_MID`
- `P11-13_LAYER_MICRO`

### 4.3 Stage order permutation (5)
- `P11-14_MACRO_MICRO_MID`
- `P11-15_MID_MACRO_MICRO`
- `P11-16_MID_MICRO_MACRO`
- `P11-17_MICRO_MACRO_MID`
- `P11-18_MICRO_MID_MACRO`

### 4.4 Routing granularity (3)
- `P11-19_TOKEN_TOKEN_TOKEN`
- `P11-20_SESSION_TOKEN_TOKEN`
- `P11-21_TOKEN_SESSION_TOKEN`

### 4.5 8배수 보강 (+2)
- `P11-22_LAYER_ONLY_BASELINE`
- `P11-23_LAYER2_MACRO_MID_MICRO`

## 5) 구현 메모
- 이 phase는 config override 중심으로 구현한다.
- 핵심 override 축:
  - `layer_layout`
  - `stage_router_granularity`
- 새로운 연산자 추가는 없음 (executor 기존 serial path 사용).
