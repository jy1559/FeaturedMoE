# Phase Core Summary + Phase1 Plan

## 1) 문서 목적
- core_ablation_v2 이전 실험 결과를 한 번에 정리한다.
- 무엇이 실제로 좋았는지(조합/성능/안정성)를 분리해서 본다.
- LR 범위를 실제 로그 기반으로 정리한다.
- special/diag/feature_ablation 신호를 함께 해석한다.
- phase1(phase1_upgrade_v1, P1)에서 무엇을 검증할지 목적/변수/기대효과를 명확히 둔다.

## 2) 이전 실험 총정리 (core_ablation_v2)

### 2.1 핵심 성능 요약 (KuaiRecLargeStrictPosV2_0.2)
- FMoE 상위 best MRR@20
  - C70: 0.0801 (test MRR@20 0.1614)
  - R33: 0.0801 (test MRR@20 0.1606)
  - E40: 0.0799 (test MRR@20 0.1610)
  - X62: 0.0798 (test MRR@20 0.1619)
  - M22: 0.0798 (test MRR@20 0.1603)
- SASRec baseline (KuaiRec) best MRR@20: 0.0785 (test MRR@20 0.1597)
- 결론
  - FMoE best MRR@20는 SASRec 대비 +0.0016.
  - test MRR@20 관점에서도 X62(0.1619), C70(0.1614), T50(0.1613) 등으로 SASRec 0.1597보다 우위.

### 2.2 조합별 판단 (좋았던 것 / 약했던 것)

| combo | MRR@20(best) | test MRR@20 | 판단 | 코멘트 |
|---|---:|---:|---|---|
| C70 | 0.0801 | 0.1614 | strong | expert_scale=3 계열이 상위권 유지 |
| R33 | 0.0801 | 0.1606 | strong | hidden+gated_bias 계열 안정적인 상위권 |
| X62 | 0.0798 | 0.1619 | strong | len=30 확장 + both 라우팅 조합이 매우 우수 |
| E40 | 0.0799 | 0.1610 | strong | complex encoder(all) 계열 성능 상승 확인 |
| T50 | 0.0793 | 0.1613 | good | token granularity 확장으로 test 지표 좋음 |
| X61 | 0.0788 | 0.1611 | mixed | 상위 trial 좋으나 변동성/붕괴 trial 동반 |
| R32 | 0.0611 | 0.1243 | weak | feature_only routing은 명확한 붕괴 구간 |

### 2.3 baseline 비교 (필수 컨텍스트)

| dataset | baseline model | best MRR@20 | test MRR@20 | 비고 |
|---|---|---:|---:|---|
| KuaiRecLargeStrictPosV2_0.2 | SASRec | 0.0785 | 0.1597 | FMoE core 최상위가 소폭 우위 |
| lastfm0.03 | SASRec | 0.4020 | 0.3801 | 현재 core_ablation_v2 FMoE 비교 데이터 없음 |

- 주의
  - core_ablation_v2 로그에는 lastfm0.03 행이 없다.
  - 따라서 lastfm은 현재 baseline 기준선만 존재하고, phase1에서 FMoE 직접 비교를 채워야 한다.

## 3) LR 범위 정리 (실제 trial_knobs 기반)

### 3.1 전반
- 상위권 조합(best MRR@20 >= 0.0795)에서 관측된 LR 분포
  - min: 4.6e-05
  - median: 3.095e-04
  - max: 1.16e-03
- 실전 권장 시작점
  - 안정 구간: 1.5e-04 ~ 7.0e-04
  - 공격 구간: 7.0e-04 ~ 1.2e-03 (성능 상승 가능하지만 변동성 증가)
  - 회피 구간: 5e-05 이하 (R32/X61 류 붕괴 trial과 동행 빈도 증가)

### 3.2 상위 조합별 LR 스냅샷

| combo | lr min | lr median | lr max | 해석 |
|---|---:|---:|---:|---|
| C70 | 1.62e-04 | 2.83e-04 | 1.16e-03 | 저~중 LR에서 안정적, 고LR도 일부 성공 |
| R33 | 2.20e-04 | 4.17e-04 | 7.30e-04 | 중LR 중심으로 안정 |
| X62 | 7.01e-05 | 2.2855e-04 | 3.87e-04 | len30는 중저LR에서 강함 |
| E40 | 3.75e-04 | 3.75e-04 | 3.75e-04 | 단일 성공점(추가 탐색 필요) |
| T50 | 5.86e-04 | 6.13e-04 | 6.40e-04 | token granularity는 중고LR 쪽 성공 |
| X61 | 4.87e-05 | 3.7185e-04 | 6.95e-04 | 저LR 붕괴/중LR 회복 패턴 공존 |

## 4) special/diag/feature_ablation 해석

### 4.1 feature_ablation 핵심

| 항목 | 최대/최소 | combo | 값 | 해석 |
|---|---|---|---:|---|
| route_change_under_feature_zero | max | R32 | 0.9131 | feature 제거 시 라우팅 급변, 구조 의존 과다 |
| route_change_under_feature_shuffle | max | X62 | 0.1881 | 셔플 민감도 존재하나 성능은 상위권 유지 |
| feature_zero_delta_mrr | min | R32 | -0.0007 | feature_zero에서 성능 저하(취약) |
| feature_zero_delta_mrr | max | R33 | +0.0039 | feature_zero에도 상대적 내성/보상 가능 |

### 4.2 diag 핵심 (라우팅 쏠림/붕괴 시그널)

| 지표 | max combo | 값 | 해석 |
|---|---|---:|---|
| mid_1.top1_max_frac | R32 | 0.8646 | mid stage expert 쏠림 매우 큼 |
| micro_1.top1_max_frac | R32 | 0.7547 | micro stage도 심한 쏠림 |

- 요약
  - R32는 성능/ablation/diag 모두에서 붕괴 시그널이 일관적이다.
  - both 라우팅 + balanced 설계(C70/X62/R33 계열)는 성능과 안정성 균형이 상대적으로 우수하다.

## 5) 그래서 phase1에서 뭐 하냐 (P1 계획)

### 5.1 목적/변수/기대효과 V표

| 그룹 | 목적 | 변수 축 | 기대효과 | keep | drop/주의 |
|---|---|---|---|---|---|
| G1 Core anchor | core 승자 재검증 | C70/X62/T50/E41 계열 anchor | 재현성 + 기준선 확보 | V |  |
| G2 Length expansion | max_seq_length 확장 | len=30 다수 + len=50 1개 | 장시퀀스 대응, generalization 확인 | V | 과도한 batch로 OOM 주의 |
| G3 Layout topology | 구조 다양성 검증 | layer/macro_ffn/mid_ffn/micro 조합 | 성능상승 + routing 안정 구조 탐색 | V | 무의미한 깊이 확장은 축소 |
| G4 Feature mode | feature 인코딩/주입 검증 | linear/complex, gated_bias 위치 | feature 활용 효율/강건성 상승 | V | feature-only routing 패턴 회피 |
| G5 Scheduler/LR | 학습 안정화 | warmup_cosine/plateau + LR envelope | early collapse 완화, 최적점 확대 | V | scheduler 없는 고LR 단독 탐색 축소 |
| G6 Aux+Embed | capacity/regularization 균형 | embed 64/256, aux lambda 강도 | 과소/과대 파라미터 구간 경계 파악 | V | 한쪽 극단만 반복 금지 |

### 5.2 phase1 실행 의사결정 규칙
- 유지(keep)
  - both 라우팅 중심 실험
  - len=30 확장축
  - C70/R33/X62 계열의 중LR(약 1.5e-04~7e-04) 탐색
- 축소/중단(drop)
  - feature_only routing(R32류)
  - 저LR 극단(약 5e-05 이하) 중심 탐색
  - baseline 스타일의 광범위 retune 반복

### 5.3 phase1 성공 기준 (기본)
- 1차: MRR@20에서 KuaiRec 기준 SASRec 0.0785 상회 유지
- 2차: test MRR@20 0.161대 재현(X62/C70/T50권)
- 3차: diag 쏠림 완화
  - top1_max_frac 과도 상승(특히 0.75+) 조합은 우선순위 하향
- 4차: ablation 강건성
  - feature_zero/shuffle에서 급격한 route_change(예: 0.6+) 조합은 리스크 태깅

## 6) 현재 phase1 상태
- axis: phase1_upgrade_v1
- phase: P1
- 현 시점 요약 파일 상태
  - summary/special/diag/feature_ablation 모두 no rows (실행 결과 미적재)
- 의미
  - 문서의 phase1 항목은 사전 계획이며, 실제 평가는 실행 후 업데이트 필요.

## 7) 업데이트 체크리스트 (실행 후 문서 갱신용)
- P1 KuaiRec 결과 입력
  - top-5 combo, best/test MRR@20, LR 범위 재추정
- P1 lastfm 결과 입력
  - SASRec(0.4020/0.3801) 대비 증감
- special/diag/feature_ablation 재평가
  - route_change, top1_max_frac, delta_mrr 기준으로 keep/drop 라벨 업데이트

## 8) 참고 소스
- experiments/run/artifacts/logs/fmoe_n3/core_ablation_v2/core_ablation_v2_summary.csv
- experiments/run/artifacts/logs/fmoe_n3/core_ablation_v2/core_ablation_v2_special_summary.csv
- experiments/run/artifacts/logs/fmoe_n3/core_ablation_v2/core_ablation_v2_diag_summary.csv
- experiments/run/artifacts/logs/fmoe_n3/core_ablation_v2/core_ablation_v2_feature_ablation_summary.csv
- experiments/run/artifacts/logs/baseline/KuaiRecLargeStrictPosV2_0.2/P0_summary.csv
- experiments/run/artifacts/logs/baseline/lastfm0.03/P0_summary.csv
- experiments/run/artifacts/logs/fmoe_n3/phase1_upgrade_v1/phase1_upgrade_v1_summary.csv
- experiments/run/artifacts/logs/fmoe_n3/phase1_upgrade_v1/phase1_upgrade_v1_special_summary.csv
- experiments/run/artifacts/logs/fmoe_n3/phase1_upgrade_v1/phase1_upgrade_v1_diag_summary.csv
- experiments/run/artifacts/logs/fmoe_n3/phase1_upgrade_v1/phase1_upgrade_v1_feature_ablation_summary.csv
