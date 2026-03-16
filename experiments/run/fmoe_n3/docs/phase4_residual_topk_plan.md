# FMoE_N3 Phase 4 Plan: Residual First + Top-k 10 Variants

작성일: 2026-03-16  
대상 데이터셋: KuaiRecLargeStrictPosV2_0.2 우선, lastfm0.03 후속 검증  
핵심 변경: residual 7종을 먼저 완료하고, 이후 top-k 10종으로 넘어간다.

---

## 1. 이번 수정의 핵심 결정

- 실험 순서: residual 전체 완료 후 top-k 전체 수행
- variation 수: residual 7종 + top-k 10종 = 17종
- combo lane: 4개 고정 (C1~C4)
- 실행 적용: residual은 4개 lane 전체, top-k는 router family 호환 lane에만 적용
- 총 run 수: residual 28 + top-k 20 = 48 runs
- GPU 4개 사용 시 lane별 run 수: C1/C2는 13, C3/C4는 11

예산 고정:

- max_evals: 8 (고정)
- tune_epochs: 100 (고정)
- tune_patience: 10 (고정)
- lr-space: 기존보다 좁힘 (Phase 4 전용 narrow range)

---

## 2. Top-k 축을 10종으로 재정의 (router family 제약 반영)

요청 반영 기준:

- 4 experts 기반(= expert_scale=1) 실험 추가
- 12 experts 기반(= expert_scale=3)은 dense/top3/top6 중심
- group 라우팅은 per-group top-k와 group 무시 top-k를 모두 포함
- top-k variation마다 필요한 router family를 고정한다

Top-k variation ID는 `K0`~`K9`로 둔다.

### A. Flat router 전용: 4 experts 계열 (global)

- K0: 4expert-dense
  - expert_scale=1, top_k=0
- K1: 4expert-top1
  - expert_scale=1, top_k=1
- K2: 4expert-top2
  - expert_scale=1, top_k=2

### B. Flat router 전용: 12 experts 계열 (global)

- K3: 12expert-dense
  - expert_scale=3, top_k=0
- K4: 12expert-top3
  - expert_scale=3, top_k=3
- K5: 12expert-top6
  - expert_scale=3, top_k=6

### C. Factored router 전용: 4 groups x 3 experts 계열

- K6: group-dense
  - group route 사용, group 내부 dense
- K7: group-top1-pergroup
  - 선택된 group 내부에서 top1
- K8: group-top2-pergroup
  - 선택된 group 내부에서 top2
- K9: group-top6-global-ignore-group
  - factored를 쓰더라도 최종 expert 선택은 group 제약 없이 전 expert에서 global top6

주의:

- K0~K5는 flat(standard) 라우터 전용으로 실행한다.
- K6~K9는 factored 라우터 전용으로 실행한다.
- flat에서 group-wise top-k를 억지로 돌리지 않는다.
- factored에서 per-group top-k와 group 무시 global top-k를 동시에 둬서, group 제약 자체의 효과를 분리한다.

---

## 3. Residual 7종(변경 없음, 우선 실행)

Residual variation ID: `R0`~`R6`

- R0: current baseline (`h + MoE`)
- R1: shared only
- R2: shared + 0.3*MoE
- R3: shared + 0.5*MoE
- R4: shared + learnable global alpha
- R5: shared + stage-wise alpha
- R6: shared + stage-wise alpha + warmup gate

실행 순서 원칙:

1. 각 combo lane에서 R0→R6 순서로 먼저 실행
2. residual 7종 완료 후 같은 lane에서 K0→K9 실행

---

## 4. Combo lane 설계 (Residual용 / Top-k용 분리)

Residual과 Top-k는 목적이 다르므로 combo 고정값을 분리한다.

### 4-1. Residual용 combo lane 4종

각 variation에서는 관련 축 파라미터만 바꾸고, 나머지는 combo lane 고정값 유지.

### C1. Standard default

- router_type: standard
- injection: gated_bias
- layout: [macro, mid, micro]
- balance_loss_lambda: 0.004
- z_loss_lambda: 0.0001

### C2. Standard stabilized

- router_type: standard
- injection: gated_bias
- layout: [macro, mid, micro]
- balance_loss_lambda: 0.004
- z_loss_lambda: 0.001
- gate_entropy_lambda: very weak on (필요 시)

### C3. Factored default

- router_type: factored
- injection: group_gated_bias
- layout: [macro, mid, micro]
- balance_loss_lambda: 0.004
- z_loss_lambda: 0.0001
- group_prior_align_lambda: weak

### C4. Factored stabilized

- router_type: factored
- injection: group_gated_bias
- layout: [macro, mid, micro] 기본, 필요 시 미세 조정
- balance_loss_lambda: 0.004~0.006
- z_loss_lambda: 0.001
- group_prior_align_lambda: weak
- factored_group_balance_lambda: weak

### 4-2. Top-k용 combo lane 4종

Top-k는 router family 제약을 강하게 걸기 위해 별도 lane으로 운용한다.

#### C1. Flat default (C1 재사용)

- router_type: standard 고정
- injection: gated_bias
- balance_loss_lambda: 0.004
- z_loss_lambda: 0.0001

#### C2. Flat stabilized (C2 재사용)

- router_type: standard 고정
- injection: gated_bias
- balance_loss_lambda: 0.004
- z_loss_lambda: 0.001
- gate_entropy_lambda: weak

#### C3. Factored strong-reg (신규)

- router_type: factored 고정
- injection: group_gated_bias
- balance_loss_lambda: 0.006
- z_loss_lambda: 0.001
- group_prior_align_lambda: weak~mid
- factored_group_balance_lambda: weak

#### C4. Factored aggressive-reg (신규)

- router_type: factored 고정
- injection: group_gated_bias
- balance_loss_lambda: 0.008
- z_loss_lambda: 0.002
- group_prior_align_lambda: mid
- factored_group_balance_lambda: weak~mid

Top-k lane 적용 범위:

- C1/C2: K0~K5 (flat 전용)
- C3/C4: K6~K9 (factored 전용)

즉, top-k에서는 lane별로 가능한 variation만 실행한다.

---

## 5. 예산/탐색 범위 고정

이번 phase는 시간이 길어지는 것을 막기 위해 budget를 아래로 고정한다.

- max_evals = 8
- tune_epochs = 100
- tune_patience = 10

lr-space는 narrow 설정으로 제한한다.

권장 narrow 범위:

- learning_rate: [2.0e-4, 2.0e-3]
- hidden_dropout_prob: 기존 후보 유지 (0.10/0.15/0.20/0.25) 또는 3개로 축소
- weight_decay: [1e-7, 1e-6, 1e-5, 5e-5] 중심

원칙:

- Phase 4에서는 lr 대탐색 금지
- 축 비교에 필요한 최소 탐색만 수행

---

## 6. 로깅/결과 파일 정리 방식 개선

요구사항: 기존 .log/.result/.summary는 유지하되, special/diag/feature_ablation은 에이전트가 읽기 쉽게 묶어서 관리.

### 6-1. 저장 구조

기본 로그는 기존처럼 저장:

- run log
- result json/csv
- axis summary csv

추가로 unified 분석 아티팩트를 만든다:

- 파일 1개 방식(권장): `*_analysis_bundle.json`
  - section: `summary`
  - section: `diag`
  - section: `special`
  - section: `feature_ablation`
  - section: `missing_sections` (미생성 항목 명시)

또는 폴더 방식(대안):

- 동일 basename 폴더 생성 후
  - `summary.json`
  - `diag.json`
  - `special.json`
  - `feature_ablation.json`
  - `index.json` (어떤 파일이 비었는지/왜 없는지)

이번 phase는 에이전트 분석 중심이므로, 파일 1개 방식이 우선이다.

### 6-2. 부분 로깅 고려

실험/모델에 따라 일부 로그가 없을 수 있으므로, 다음 정책을 강제한다.

- 없는 섹션은 빈 dict로 두지 말고 `missing_sections`에 명시
- `not_applicable` / `not_enabled` / `failed_to_compute` 상태코드 추가
- summary CSV에는 `diag_available`, `special_available`, `fablation_available` 플래그 추가

---

## 7. 파일명 규칙

요청한 형식으로 로그 제목을 통일한다.

형식:

`인덱스번호_축_간단설명_combo_날짜시간.log`

예시:

- `001_R_base_C1_20260316_231500.log`
- `017_R_stagealpha_warmup_C3_20260316_233200.log`
- `018_K_4e_dense_C1_20260316_233500.log`
- `027_K_groupignore_global6_C4_20260317_001200.log`

설명:

- 축은 `R`/`K`만 사용한다.
- 파일명에는 `R0`, `K9` 같은 variation ID를 넣지 않는다.
- variation ID는 summary metadata 필드로만 관리한다.

같은 basename으로 result/analysis도 맞춘다:

- `.result.json`
- `.summary.json`
- `.analysis_bundle.json`

---

## 8. 실행 순서 및 GPU lane 배치

GPU lane 원칙:

- GPU0: C1 lane
- GPU1: C2 lane
- GPU2: C3 lane
- GPU3: C4 lane

각 GPU 실행 순서:

1. residual R0~R6 (7 runs)
2. top-k는 lane 호환 variation만 실행

lane별 top-k 실행 수:

- C1/C2 (flat): K0~K5 = 6 runs
- C3/C4 (factored): K6~K9 = 4 runs

lane별 총 실행 수:

- C1/C2: 7 + 6 = 13 runs
- C3/C4: 7 + 4 = 11 runs

전체 48 runs.

중요 실행 정책:

- 각 GPU lane은 자기 큐를 순서대로 끝까지 실행한다.
- 다른 GPU의 같은 순번 run이 끝날 때까지 기다리지 않는다.
- cross-lane barrier/synchronization 없이 독립 진행한다.

---

## 9. 스크립트 산출물 계획

Python:

- `run_phase4_residual_topk.py`
  - 48-run manifest 생성
  - axis/residual/topk variation 메타데이터 기록
  - combo lane별 순차 큐 실행
  - lane 간 동기화 없는 독립 worker 실행
  - summary refresh 및 analysis bundle 저장

Shell:

- `phase_4_residual_topk.sh`
  - 기본값 고정: `--max-evals 8 --tune-epochs 100 --tune-patience 10`
  - dataset/gpus/seed/only/axis/combo 필터 지원

Phase key 제안:

- axis: `phase4_residual_topk_v2`
- phase: `P4`

---

## 10. 이번 플랜의 확인 질문

1. residual 재설계(R0~R6) 중 어떤 구조가 안정성과 성능을 동시에 개선하는가?
2. top-k는 12expert top3/top6과 4expert top1/top2 중 어디가 더 유리한가?
3. group 제약 top-k와 group 무시 global top-k(K9)의 차이는 무엇인가?
4. flat/factored 각각에서 aux/reg 강도 최적점은 어떻게 다른가?
5. unified analysis bundle이 에이전트 자동 분석 품질을 실제로 높이는가?

---

## 11. 결정 요약

- top-k 축은 7종에서 10종으로 확장했다.
- residual 7종을 먼저 끝내고 top-k 10종으로 진행한다.
- 예산은 `max_evals=8, tune_epochs=100, patience=10`으로 고정한다.
- lr-space는 좁힌다.
- top-k는 flat/factored router 요구사항을 강제해 lane별 적용 variation을 분리한다.
- special/diag/feature_ablation은 unified bundle로 묶어 저장한다.
- 로그 파일명은 `인덱스_축_설명_combo_날짜시간` 규칙으로 통일한다.