# Phase 8 Plan: Router/Wrapper + Diagnostics (KuaiRec 중심)

## 1. 목적
- N3 routing을 primitive(a~e) + wrapper(w1~w6)로 정리한 구현을 KuaiRec에서 우선 검증한다.
- diagnostics 저장을 run-dir 기반 단일 schema(`.../logging/fmoe_n3/<dataset>/P8/<run_id>/diag/...`)로 고정한다.
- 성능 해석은 `final routing`과 `internal routing`을 같은 node schema로 비교한다.

## 2. 고정 실험 운영 정책
- dataset 우선순위: `KuaiRecLargeStrictPosV2_0.2` 중심.
- run budget: 2-step.
- Step 1: 1-seed screening으로 후보 wrapper/stage 조합을 빠르게 압축.
- Step 2: 상위 후보만 3-seed 재검증.
- anchor별 seed 고정 규칙 유지(phase7 방식 재사용), 단 라우터 조합만 교체.
- 이번 phase에서는 launcher/sh/py 신규 작성 없이, 수동/기존 runner로 설정만 주입해 실행.

## 3. 실험 축 (A->D 순차)
- A. Wrapper core 비교 (bias OFF)
- B. Bias augmentation 비교
- C. Primitive source 비교 (hidden/feature/both)
- D. Top-k refinement (primitive/final 분리)

### A. Wrapper core (최우선)
- 공통 고정:
- `feature_group_bias_lambda=0`
- `rule_bias_scale=0`
- `route_prior_lambda=0`
- 비교 후보:
- `all_w1`: macro/mid/micro = w1
- `all_w4`: macro/mid/micro = w4
- `all_w6`: macro/mid/micro = w6
- `mixed_1`: macro=w4, mid=w6, micro=w1
- `mixed_2`: macro=w6, mid=w1, micro=w1
- 성공 기준:
- valid MRR@20 상위권 + test MRR@20 하락 없음
- final.expert `top1_monopoly_norm` 과도 증가 없음
- final/group/intra `knn_consistency_score` 붕괴 없음

### B. Bias augmentation
- A 단계 상위 2개만 진행.
- 켜는 순서:
- `feature_group_bias_lambda`
- `rule_bias_scale`
- 각각 단독 -> 병행 순서로 비교.
- 목표:
- 성능 개선이 있더라도 internal node가 한 primitive로 collapse되는지 확인.

### C. Primitive source
- B 단계 상위 1~2개 wrapper 고정.
- `a_joint,b_group,c_shared,d_cond,e_scalar`의 source를 stage별 독립 조정.
- 기본 baseline: `both`.
- 비교군:
- `group-only-feature` (d/e feature 고정, a/b/c both)
- `all-both`
- `a-hidden + b/d-feature` 혼합형

### D. Top-k refinement
- primitive top-k와 final top-k를 분리 탐색.
- 우선순위:
- `d_cond top_k=1` (group당 1 expert 강제)
- `final top_k`는 baseline 유지 후 필요 시만 변경.
- 체크 포인트:
- 활성 expert 수, sparse 안정성, 성능/diag trade-off.

## 4. P8 Diagnostics 운영 규칙

### 저장 경로
- 기준: `.../logging/fmoe_n3/<dataset>/P8/<run_id>/diag/`
- Tier A: `tier_a_final/final_metrics.csv`
- Tier B: `tier_b_internal/internal_metrics.csv`
- Tier C: `tier_c_viz/*`
- Raw 참조: `raw/*`
- 메타 인덱스: `meta.json`

### 필수 컬럼
- 공통: `stage_name, split, aggregation_level, node_kind, node_name, route_space, support_size, wrapper_name`
- 핵심: `entropy_norm, n_eff_norm, top1_monopoly_norm, jitter_adj_norm, knn_consistency_score`
- primitive 전용: `source_type, temperature, top_k`

### 분석 리더 가이드 (읽는 순서)
1. `tier_a_final/final_metrics.csv`에서 stage별 `final.expert`, `final.group`, `final.intra.*`를 먼저 확인.
2. 같은 run의 `tier_b_internal/internal_metrics.csv`에서 `wrapper.*`, `primitive.*`로 원인 추적.
3. 필요 시 `raw/best_valid_overview.md`와 `raw/test_diag.json`으로 상세 drill-down.
4. 시각화가 필요하면 `tier_c_viz/viz_manifest.json`에서 row_count 확인 후 PCA csv.gz 사용.

## 5. 모델/로깅 변경 검증 체크리스트
- wrapper별 required primitive만 계산되는지 확인.
- `router_aux.primitive_outputs`에 사용 primitive만 들어가는지 확인.
- load-balance/z/route 정규화가 final gate 기준으로만 계산되는지 확인.
- run root에 legacy `diag_*.json/csv` 파일이 새로 생기지 않는지 확인.
- 모든 diag 포인터가 `result.json`, `run_summary.json`에서 `diag/*` 하위 파일을 가리키는지 확인.

## 6. 실패 기준과 즉시 중단 조건
- 학습 불안정(Non-finite loss, grad explode) 연속 2회.
- test MRR@20가 baseline 대비 유의미하게 하락하고, final/internal 모두 collapse 지표 증가.
- diag 파일 누락/깨짐이 run의 20% 이상에서 발생.

## 7. 산출물
- 코드: router/wrapper + diag logging schema 전환.
- 문서: 본 `phase8_plan.md`.
- notebook: architecture 폴더의 sanity notebook으로 primitive/wrapper 동작 시각 검증.
