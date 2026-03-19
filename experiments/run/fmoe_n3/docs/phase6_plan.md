# Phase 6 Plan: Candidate Reinforcement + Evidence Strengthening

작성일: 2026-03-17
대상: FeaturedMoE_N3 (KuaiRecLargeStrictPosV2_0.2 우선, 필요 시 lastfm0.03 확장)

## 0) 목적
- A/B/C 최종 후보를 공정하게 재검증한다.
- 논문 근거 보강용으로 인과성 높은 ablation(구조/정규화/feature)을 추가한다.
- 결과 저장은 아래 원칙으로 통일한다.
  - logs: 실행 로그 텍스트
  - results: 각 run 결과 json(기본 결과 파일)
  - logging: special/diag/feature ablation/요약 카드 등 추가 산출물

## 1) 실행 기본값
- 공통 기본값
  - `max-evals=10`
  - `tune-epochs=100`
  - `tune-patience=10`
  - `gpus=0,1,2,3`
- 후보 확인(cand3x)만 예외
  - 각 run `max-evals = 30` (기본의 3배)
- LR 탐색
  - 기본: `[2e-4, 2e-3]`
  - 후보 확인(cand3x): `[8e-5, 8e-3]` (넓게 확장)

## 2) 실험 묶음과 콤보 수

### 2.1 후보 재확인 (cand3x)
- 목적: A/B/C 3개 후보를 seed 3개로 공정 비교
- 설정: A/B/C 문서 정의(260317_summary) 기반
- 개수: `3 후보 x 3 seed = 9`
- run_phase 규칙: `P6_CAND_{A|B|C}_S{1..3}`

### 2.2 Baseline bridge 확장 (baseline_bridge)
- 목적: candidate 직후 즉시 SASRec/MoE 기준선과 성능-진단 축 비교
- 8개 설정:
  - B0: sasrec-equivalent len20
  - B1: sasrec-equivalent window10
  - B2: hidden-only MoE (standard)
  - B3: hidden-only MoE (factored)
  - B4: feature-only MoE (standard)
  - B5: feature-only MoE (factored)
    - B6: both-source MoE (standard)
    - B7: both-source MoE (factored)
  - 주의: B5는 d_feat_emb → d_router_hidden 자동 projection으로 지원 가능
  - 개수: `8`
  - B6: both-source MoE (standard)
  - B7: both-source MoE (factored)
- 개수: `8`
- run_phase 규칙: `P6_BASE_{B0..B7}`

### 2.3 Router x Injection 교차 (router2x2)
- 목적: standard/factored 와 gated/group_gated의 인과 보강
- 구성: `2x2` 교차를 context 2종으로 반복
  - Context X1: base residual + topk=1 + global_flat
  - Context X2: warmup residual + topk=2 + group_top2_pergroup
- 개수: `2(router) x 2(injection) x 2(context) = 8`
- run_phase 규칙: `P6_RXI_{X1|X2}_{ROU}_{INJ}`

### 2.4 Specialization 정규화 ablation (spec_ablation)
- 목적: specialization 유도 정규화의 단독/조합 기여도 분해
- 후보: A/B 두 기준점에서 동일 방법 비교
- 방법(5):
  - M0: hard off (smooth/consistency/sharp/monopoly/prior/group_balance 전부 0)
  - M1: high smoothness only (`route_smoothness_lambda=0.04`)
  - M2: high consistency only (`route_consistency_lambda=0.04`, `pairs=8`)
  - M3: high sharp+monopoly (`sharp=0.01`, `monopoly=0.04`, `tau=0.25`)
  - M4: strict mixed prior+consistency (`smooth=0.02`, `consistency=0.03`, `pairs=8`, `route_prior=0.01`, `group_prior=1e-3`, `group_balance=2e-3`)
- 개수: `5 방법 x 2 후보(A/B) = 10`
- run_phase 규칙: `P6_SPEC_{A|B}_{M0..M4}`

#### Phase5 대비 차이점
- phase5는 주로 후보 탐색/구조 조합을 넓게 보는 목적이라 정규화 강도가 비교적 보수적이었고, 다른 축(router/injection/residual) 변화와 섞여 있었다.
- phase6 spec_ablation은 A/B 구조를 고정해 정규화 축만 분해하며, 단일 축 강정규화(M1~M3)와 강한 혼합 정규화(M4)를 별도로 둬서 "정규화가 얼마나 강해야 실제 specialization 신호로 이어지는지"를 직접 본다.
- 즉, phase5의 재실행이 아니라 phase5 결과 위에 "정규화 강도/방식"만 더 공격적으로 눌러 보는 후속 인과 검증이다.

### 2.5 Feature ablation sweep (feature_ablation)
- 목적: feature 축 근거 강화 (macro window + family mask)
- 기준: 후보 B 고정
- 변수:
  - `macro_history_window`: `{5, 10}`
  - family mask: 4개 family 중 1개 또는 2개 선택
    - family: Tempo, Memory, Focus, Exposure
    - 1개 조합 4개 + 2개 조합 6개 = 10개
- 개수: `2 window x 10 mask = 20`
- run_phase 규칙: `P6_FEAT_W{5|10}_{MASK}`

## 3) 총 실험 수
- 전체: `9 + 8 + 8 + 10 + 20 = 55 runs`

## 3.1 실행 순서(고정)
- `candidate 9 -> baseline 8 -> router x injection 8 -> specialization 10 -> feature ablation 20`
- GPU 할당은 전체 큐를 이 순서로 미리 round-robin 배치한다.
  - 예시(4 GPU): candidate 1..9는 `0->1->2->3->0...->0`, 이어서 baseline은 다음 GPU부터 시작해 `1->2->3->0...`로 계속된다.
- 각 GPU는 자기 큐에서 현재 run 종료 즉시 다음 run을 바로 launch한다(다른 GPU 종료 대기 없음).

## 4) 실행 스크립트
- python launcher:
  - `experiments/run/fmoe_n3/run_phase6_candidate_reinfor.py`
- shell wrapper:
  - `experiments/run/fmoe_n3/phase_6_candidate_reinfor.sh`

### 4.1 전체 실행 예시
- `bash experiments/run/fmoe_n3/phase_6_candidate_reinfor.sh --datasets KuaiRecLargeStrictPosV2_0.2 --gpus 0,1,2,3 --suites all --max-evals 10 --tune-epochs 100 --tune-patience 10`

### 4.2 묶음별 실행 예시
- 후보 재확인만:
  - `bash experiments/run/fmoe_n3/phase_6_candidate_reinfor.sh --suites candidate`
- 구조 교차만:
  - `bash experiments/run/fmoe_n3/phase_6_candidate_reinfor.sh --suites router`
- 정규화 ablation만:
  - `bash experiments/run/fmoe_n3/phase_6_candidate_reinfor.sh --suites spec`
- feature sweep만:
  - `bash experiments/run/fmoe_n3/phase_6_candidate_reinfor.sh --suites feature`
- baseline bridge만:
  - `bash experiments/run/fmoe_n3/phase_6_candidate_reinfor.sh --suites base`

## 5) 로그/결과 경로 규칙
- 실행 로그 텍스트:
  - `experiments/run/artifacts/logs/fmoe_n3/phase6_candidate_reinfor_v2/P6/{dataset}/FMoEN3/*.log`
- run result json:
  - `experiments/run/artifacts/results/fmoe_n3/*.json` (또는 설정에 따른 기본 result 경로)
- 추가 산출물(logging bundle):
  - `experiments/run/artifacts/logging/fmoe_n3/{dataset}/P6/{run_id}/`
  - 포함 파일(존재 시): `result.json`, `special_metrics.json`, `diag_*.json/csv`, `feature_ablation.json`, `run_summary.json`, `analysis_card.json`

## 6) 네이밍/가독성 규칙
- 로그 파일 stem:
  - `{index}_{category}_{combo_id}_{timestamp}.log`
- 핵심 식별자:
  - `category` = `cand3x|router2x2|spec_ablation|feature_ablation|baseline_bridge`
  - `combo_id`는 실험 의도를 직접 드러내는 문자열 사용
- run_phase는 짧고 검색 가능하게 고정 prefix `P6_` 사용

## 7) 해석 우선순위(요약 문서용)
- 1순위: best MRR@20
- 2순위: test MRR@20
- 3순위: micro route jitter / consistency(knn js, knn score)
- 4순위: feature ablation delta (global + family)
- 5순위: special slice(new/cold/short-long) 동반 변화

## 8) 위험/주의
- feature sweep(20 runs)은 총량이 커서 장시간 소요 가능
- candidate 묶음은 3x max-evals라 먼저 실행해도 큐 점유가 길다
- 4 GPU 순차 큐는 GPU별 길이 편차가 생길 수 있으므로 category별 분할 실행도 고려
