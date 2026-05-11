# Phase9 AuxLoss Plan (Documentation-Only, v1)

작성일: 2026-03-21  
대상: `FeaturedMoE_N3` on `KuaiRecLargeStrictPosV2_0.2`  
범위 고정: **코드/스크립트 수정 없이 문서만 작성**

---

## 0) Summary

- 이번 턴의 산출물은 이 문서 1개:
  - `/workspace/jy1559/FMoE/experiments/run/fmoe_n3/docs/plans/phase9_auxloss.md`
- 문서 목표:
1. 현재 N3 aux loss를 **타깃 레벨별(final/wrapper/primitive/router-logit)**로 정리한다.
2. Phase9 실험을 **복잡도 제한(main 1개 + support 0~1개)** 규칙으로 설계한다.
3. 향후 `sh/py` 구현 시 누락이 없도록 naming, summary, winner selection rule을 확정한다.

---

## 1) Phase9 목적/원칙

### 1.1 목적
- Phase9는 “aux를 많이 넣는 탐색”보다, **aux의 동작-성능 인과를 설명 가능하게 분리**하는 phase다.
- Phase8 상위 wrapper/bias/source 조합을 고정하고 aux만 바꿔서 비교한다.

### 1.2 복잡도 제어 원칙
- 각 조합의 active aux는 최대 2개만 허용:
  - `main_aux`: 반드시 1개 또는 none
  - `support_aux`: 0개 또는 1개
- 즉, 한 조합에서 동시 활성 aux 수(`n_active_aux`)는 `0,1,2`만 허용한다.
- 목적: 논문 서술/원인 해석/실험 운영 난이도 동시 관리.

---

## 2) Aux Loss Inventory (코드 기준)

기준 코드:
- `experiments/models/FeaturedMoE_N3/featured_moe_n3.py`
- `experiments/models/FeaturedMoE_N/featured_moe_n.py`
- `experiments/models/FeaturedMoE_v3/losses.py`

### 2.1 현재 사용 가능 aux (N3 코드 경로 존재)

| aux key | target level | 계산 대상 텐서(핵심) | 유도 동작 | Phase9 권장 |
|---|---|---|---|---|
| `balance_loss_lambda` | final gate | `aux_data.gate_weights` | expert usage 균형(평준화) | 사용 |
| `z_loss_lambda` | router-logit | `router_aux.learned_gate_logits` (fallback: `gate_logits`) | logit 폭주 억제/stability | support 위주 사용 |
| `gate_entropy_lambda` | final gate | `aux_data.gate_weights` | entropy 증가(균일 분포 쪽) | 보류 |
| `route_smoothness_lambda` | final gate | 토큰 인접 시점 `gate_weights` 변화량 | 시간축 route 급변 완화 | 사용 |
| `route_consistency_lambda` | final gate + feature-neighbor | session gate prob 간 KNN JS | 이웃 feature간 route 일관성 강제 | 분석용/보류 |
| `route_sharpness_lambda` | final gate | `gate_weights` entropy | route 집중(특화) 유도 | 사용 |
| `route_monopoly_lambda` (+`tau`) | final gate | top1 expert usage | 과점 제어/또는 집중 유도 | 사용 |
| `route_prior_lambda` | final gate + feature prior | session gate prob vs family prior KL | feature-family aligned routing | 사용 |
| `group_prior_align_lambda` | wrapper/group | `router_aux.group_weights` vs `group_prior` KL | group 분포를 feature-derived prior에 정렬 | 사용 |
| `factored_group_balance_lambda` | wrapper/group-logit | `router_aux.factored_group_logits` | group-level collapse 완화 | 사용(조건부) |
| `rule_agreement_lambda` | router-logit | `gate_logits` vs `rule_target_logits` KL | learned router를 rule target에 정렬 | 분석용/보류 |
| `group_coverage_lambda` | wrapper/group | `router_aux.group_weights` entropy reward | group usage 다양성 보상 | 분석용/보류 |

### 2.2 target-level 요약

- `final gate` 중심: `balance/z/smoothness/sharpness/monopoly/route_prior`
- `wrapper/group` 중심: `group_prior_align/factored_group_balance`
- `router-logit` 직접: `z/rule_agreement`
- `primitive` 직접: 현재 N3에서 전용 lambda 없음(아래 7절의 후보 키로 확장 예정)

### 2.3 Phase9 메인에서 제외(보류)하는 항목과 이유

- `route_consistency_lambda`: Phase7 관찰상 성능 상관이 단조롭지 않아(일관성↑가 성능↑를 보장하지 않음) 메인 조합에서 제외.
- `rule_agreement_lambda`: rule target 유도 효과는 해석이 어려워 main/support 2개 제한 원칙에 맞지 않음.
- `group_coverage_lambda`: reward형 항으로 다른 aux와 상호작용이 커서 초기 비교에서 변수 과다.
- `gate_entropy_lambda`: balance/monopoly/sharpness와 의미 중복이 커서 v1에서는 제외.

---

## 3) Phase9 Base 고정 (4개)

Phase8 결과에서 선택한 base 4개(모두 `S1`, `topk=dense`):

| base_id | run_phase | wrapper/bias/source | run_best_valid_mrr20 | test_mrr20 |
|---|---|---|---:|---:|
| `B1` | `P8_SCR_B_ALL_W2_BIAS_RULE_S1` | `all_w2 + bias_rule + src_base` | 0.0826 | 0.1596 |
| `B2` | `P8_SCR_B_MIXED_2_BIAS_GROUP_FEAT_S1` | `mixed_2 + bias_group_feat + src_base` | 0.0823 | 0.1570 |
| `B3` | `P8_SCR_B_MIXED_2_BIAS_BOTH_S1` | `mixed_2 + bias_both + src_base` | 0.0821 | 0.1594 |
| `B4` | `P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_S1` | `mixed_2 + bias_both + src_abc_feature` | 0.0822 | 0.1596 |

---

## 4) Phase9 실험 매트릭스 (4x4x4, S1)

## 4.1 구조
- `base`: 4개 (`B1~B4`)
- `concept`: 4개 (`C0~C3`)
- `combo`: concept별 4개 (`*_1~*_4`)
- seed: `S1` 고정
- 총 run: `4 x 4 x 4 x 1 = 64`

## 4.2 Concept/Combo 프로파일 (16개)

모든 combo는 `main_aux` 1개(또는 none) + `support_aux` 0~1개만 허용.

| concept | combo_id | main_aux (lambda) | support_aux (lambda) | n_active_aux | 기대 시나리오 |
|---|---|---|---|---:|---|
| `C0_Natural` | `N1` | none | none | 0 | 완전 baseline. aux 없는 기준선 확보 |
| `C0_Natural` | `N2` | `route_smoothness_lambda=0.01` | none | 1 | route jitter만 완화한 자연형 |
| `C0_Natural` | `N3` | `balance_loss_lambda=0.001` | `z_loss_lambda=5e-5` | 2 | 아주 약한 canonical 안정화 |
| `C0_Natural` | `N4` | `z_loss_lambda=1e-4` | none | 1 | logit 안정화 단독 효과 확인 |
| `C1_CanonicalBalance` | `B1` | `balance_loss_lambda=0.002` | `z_loss_lambda=1e-4` | 2 | P7 baseline급 균형 유도 |
| `C1_CanonicalBalance` | `B2` | `balance_loss_lambda=0.006` | `z_loss_lambda=3e-4` | 2 | 강한 균형 유도(성능-균형 trade-off 확인) |
| `C1_CanonicalBalance` | `B3` | `factored_group_balance_lambda=1e-3` | `z_loss_lambda=1e-4` | 2 | group-level collapse 방지(조건부 stage 효과) |
| `C1_CanonicalBalance` | `B4` | `primitive_balance_lambda=8e-4` | `z_loss_lambda=1e-4` | 2 | primitive 분포 균형(신규 키 필요) |
| `C2_Specialization` | `S1` | `route_sharpness_lambda=0.004` | none | 1 | 약한 특화 유도 |
| `C2_Specialization` | `S2` | `route_sharpness_lambda=0.008` | none | 1 | 강한 특화 유도 |
| `C2_Specialization` | `S3` | `route_sharpness_lambda=0.008` | `route_monopoly_lambda=0.02, route_monopoly_tau=0.25` | 2 | 집중+과점 제어 동시 확인 |
| `C2_Specialization` | `S4` | `route_smoothness_lambda=0.03` | `route_sharpness_lambda=0.004` | 2 | 시계열 안정성 안에서 특화 유도 |
| `C3_FeatureAlignment` | `F1` | `route_prior_lambda=5e-4` | none | 1 | session-level feature prior 정렬 |
| `C3_FeatureAlignment` | `F2` | `route_prior_lambda=1e-3` | `z_loss_lambda=1e-4` | 2 | 정렬 강도 증가 + 안정화 |
| `C3_FeatureAlignment` | `F3` | `group_prior_align_lambda=5e-4` | none | 1 | group 분포 정렬 효과 분리 |
| `C3_FeatureAlignment` | `F4` | `wrapper_group_feature_align_lambda=1e-3` | `group_prior_align_lambda=2e-4` | 2 | wrapper-group 특화 정렬(신규 키 필요) |

### 4.3 `active aux <= 2` 검증 규칙
- 위 16개 프로파일에서 `n_active_aux`는 `0/1/2`만 존재한다.
- 전체 64런은 `base` cross-product이므로 동일 규칙을 자동 만족한다.

---

## 5) 기대 동작/해석 프레임

### 5.1 Concept별 기대 결과

- `C0_Natural`  
  성능 기대: 과규제 없는 baseline 대비 보수적 개선/유지.  
  diag 기대: entropy/n_eff의 큰 붕괴 없이 자연 분포 유지.

- `C1_CanonicalBalance`  
  성능 기대: valid 안정성 개선 가능, 과도한 강도(B2)는 test 손실 위험.  
  diag 기대: `top1_monopoly_norm` 감소, `n_eff_norm` 증가 방향.

- `C2_Specialization`  
  성능 기대: 특정 base에서 고점 가능, 과도 특화 시 일반화 하락 가능.  
  diag 기대: entropy 감소(집중), 단 `top1_monopoly_norm` 과상승은 위험 신호.

- `C3_FeatureAlignment`  
  성능 기대: feature-rich base(B2~B4)에서 상대적 이득 가능.  
  diag 기대: wrapper/group/primitive 노드에서 feature-aligned 패턴 강화.

### 5.2 지표 매핑(읽기 순서)

1. final node:
   - `entropy_norm`
   - `n_eff_norm`
   - `top1_monopoly_norm`
2. wrapper/primitive node:
   - `wrapper.group`
   - `wrapper.intra.*`
   - `primitive.*`
3. 해석 원칙:
   - 성능이 같으면 collapse 리스크가 낮은 조합을 우선.

---

## 6) 실행 규약 (향후 sh/py 구현용 사양)

### 6.1 Naming

- `AXIS=phase9_auxloss_v1`
- `PHASE=P9`
- `run_phase` 규칙:
  - `P9_<BASE>_<CONCEPT>_<COMBO>_S1`
  - 예: `P9_B3_C2_S3_S1`

### 6.2 summary.csv 필수 컬럼(최소)

- `base_id`
- `concept_id`
- `combo_id`
- `main_aux`
- `support_aux`
- `run_best_valid_mrr20`
- `test_mrr20`

### 6.3 Best 선별 규칙

- 1순위: `run_best_valid_mrr20` 최대
- 2순위: `test_mrr20` 최대
- 동률 guardrail(default):
  - `top1_monopoly_norm <= 0.60`
  - `n_eff_norm >= 0.25`
- guardrail까지 동률이면 active aux가 더 적은 조합 우선(`n_active_aux` 낮은 쪽).

---

## 7) 공용 인터페이스/타입 변경 사양 (문서 명시만, 미구현)

### 7.1 신규 config 키 후보

1. `primitive_balance_lambda`
2. `wrapper_group_feature_align_lambda`

### 7.2 기존 키 재사용 우선 목록

1. `balance_loss_lambda`
2. `z_loss_lambda`
3. `route_smoothness_lambda`
4. `route_sharpness_lambda`
5. `route_monopoly_lambda`
6. `route_prior_lambda`
7. `group_prior_align_lambda`
8. `factored_group_balance_lambda`

---

## 8) Phase9_2 연결 규칙

- Phase9 종료 후 base별 winner 1개 선출(총 4개).
- 확장 검증은 `4(hparam) x 4(seed)`로 수행:
  - 총 `4 x 4 x 4 = 64` run.
- 포맷은 `phase8_2` 검증 방식을 계승:
  - matrix 기반
  - run_id/run_phase 역파싱 가능
  - 완료 로그 기준 resume 정책 유지.

---

## 9) Test Plan (문서 검증 체크리스트)

- [ ] `4 x 4 x 4 = 64` 조합 정의가 닫혀 있는가?
- [ ] 모든 조합이 `main/support` 규칙으로 `active aux <= 2`를 만족하는가?
- [ ] 각 조합에 기대 시나리오가 최소 1줄 이상 있는가?
- [ ] run naming, summary, best-selection이 추가 의사결정 없이 구현 가능한가?
- [ ] 신규 키가 필요한 조합(`C1_B4`, `C3_F4`)이 명시되어 있는가?

---

## 10) Assumptions / Defaults

- 이번 턴은 `phase9_auxloss.md` 설계 확정만 수행한다.
- `sh/py/코드/테스트 실행`은 수행하지 않는다.
- seed는 Phase9 screening에서 `S1` 고정.
- Phase9의 목적은 “넓은 탐색”보다 “설명 가능한 비교”다.

