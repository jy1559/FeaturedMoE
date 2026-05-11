# FMoE_N3 모델 아키텍처 요약 v2

작성일: 2026-03-16  
기준 실험 축: core_ablation_v2 / phase1_upgrade_v1 / phase2_router_v1 / phase3_focus_v1  
우선 데이터셋: KuaiRecLargeStrictPosV2_0.2 (탐색용), lastfm0.03 (일반화 확인용)  
기본 지표: MRR@20 (test_mrr@20 / HR@10 병행)  
비교 기준: SASRec (P01) mrr@20 = 0.0785 (KuaiRec) / 0.4020 (lastfm)

---

## 1. 모델 요약 및 변경 가능한 축 정리

### 1-1. 모델 구조 요약

FMoE_N3는 SASRec 기반의 시퀀스 추천 모델이다.  
핵심 아이디어는 "attention 이후 FFN을 stage별 MoE로 교체하고, engineered feature로 라우팅과 표현 보정을 동시에 조건화"하는 것이다.

```
입력 아이템 시퀀스
   → Embedding
   → Layer 스택 (layer_layout에 따라 결정)
       각 레이어는 다음 중 하나:
       - layer   : standard SASRec attn + ffn
       - macro   : macro-attention + macro MoE FFN
       - mid     : mid-attention + mid MoE FFN
       - micro   : micro-attention + micro MoE FFN
       - *_ffn   : attention 없이 해당 stage MoE FFN만
   → 최종 hidden → 아이템 점수
```

stage(macro/mid/micro)별로 독립적으로 설정 가능한 축:

| 구성요소 | 설정 파라미터 | 선택지 |
|---|---|---|
| FFN 계산 방식 | stage_compute_mode | none / dense_plain / moe |
| 라우터 신호 | stage_router_source | hidden / feature / both |
| 라우터 구조 | stage_router_type | standard / factored |
| 라우터 학습 방식 | stage_router_mode | none / learned / rule_soft |
| 라우팅 단위 | stage_router_granularity | session / token |
| feature 주입 방식 | stage_feature_injection | none / film / gated_bias / group_gated_bias |
| feature 인코더 | stage_feature_encoder_mode | linear / complex |

전역 설정:

| 파라미터 | 의미 |
|---|---|
| layer_layout | stage 배치 순서 및 조합 |
| expert_scale | expert 수 = base * expert_scale |
| moe_top_k | 0 = dense routing, k>0 = top-k sparse |
| macro_history_window | macro feature 통계 시간 범위 |
| stage_feature_family_mask | 사용할 feature group 제한 (Tempo/Memory/Focus/Exposure) |
| MAX_ITEM_LIST_LENGTH | 입력 시퀀스 길이 |

### 1-2. 변경 가능한 핵심 축 상세

#### A. Stage 계산 방식 (stage_compute_mode)

- **none**: stage FFN 완전 비활성화. 구조 비교/디버깅용
- **dense_plain**: MoE 없이 단일 고정 FFN. 안정적 baseline
- **moe**: expert mixture로 계산. 현재 메인 트랙

#### B. 라우터 신호 (stage_router_source)

- **hidden**: sequence hidden state만으로 라우팅. 상대적으로 안정적
- **feature**: engineered feature 기반 라우팅. feature에 의존적 routing이 명시적으로 일어남
- **both**: hidden + feature 결합. 현재 상위권 다수

#### C. 라우터 구조 (stage_router_type)

- **standard**: expert logits를 직접 생성하는 단층 선형 라우터
  - 단순하고 안정적, 최고점 많이 배출
- **factored**: group logits(어떤 expert group?) + intra-group logits(그 안에서 어떤 expert?) 를 분리해서 합성
  - DeepSeekMoE의 expert segmentation 개념과 유사
  - group_gated_bias injection과 시너지가 있음
  - PD04/P3S2에서 경쟁력 확인

#### D. Feature 주입 방식 (stage_feature_injection)

- **none**: feature 보정 없이 hidden만 사용
- **film**: FiLM(scale + shift)으로 hidden을 변조
  - 표현 공간 전체를 바꾸는 global conditioning
- **gated_bias**: gate 가중치 × feature bias를 hidden에 더함
  - 덧셈 형태의 조건부 shift. 현재 가장 강한 단일 옵션
- **group_gated_bias**: factored 라우터의 expert group별로 다른 feature bias 적용
  - HyMoERec의 shared+specialized branch 구분 아이디어와 유사
  - factored router와 함께 쓸 때 가장 유효

#### E. 라우팅 단위 (stage_router_granularity)

- **session**: 한 샘플 내 모든 위치가 같은 라우팅 경향을 공유
- **token**: 위치별로 라우팅이 다름. 계산량 증가, 변동성 증가
- micro stage는 항상 token 고정 (디자인 상)

#### F. Feature 인코더 (stage_feature_encoder_mode)

- **linear**: 단순 선형 투영. 안정적 baseline
- **complex**: 2층 비선형 인코더. 경우에 따라 도움이 되나 항상은 아님

#### G. Dense vs Sparse routing (moe_top_k)

- **moe_top_k=0 (dense)**: 모든 expert를 계산하고 gate 가중합
  - 현재 모든 실험의 기본값
  - 안정적이고 gradient가 모든 expert를 통과
- **moe_top_k=1 or 2 (sparse)**: 상위 k개 expert만 선택·계산
  - Switch Transformer 방식 (top-1), ST-MoE 방식 (top-2)
  - 조건부 계산 효율 이점, 대신 routing collapse 리스크 증가
  - X63 (top_k=2) 실험: mrr=0.0795 (M22=0.0798 대비 소폭 하락, 유의미한 차이는 아님)
  - ST-MoE 권고처럼 sparse 시에는 z_loss 등 weak stabilization 필수

#### H. Residual 구조 (현재 + 변경 후보)

현재 구조:
```
out_stage = norm(h) → route&compute → h + stage_output
```
즉, `h + MoE(norm(h))` 형태의 단순 residual.

문제: shared 공통 표현과 routed 특화 표현의 역할이 명시적으로 분리되지 않음.  
DeepSpeed-MoE, HyMoERec의 설계를 참조하면:
```
out_stage = h + SharedFFN(norm(h)) + alpha * MoE(norm(h))
```
가 더 명시적인 역할 분리 구조.

#### I. Aux/Reg 축

- **balance_loss_lambda**: expert 사용 균형 유도 (현재 기본 0.004)
- **z_loss_lambda**: router logit 안정화 (현재 기본 0.0001)
  - ST-MoE에서 training instability의 핵심 해결책으로 제안한 손실
  - sparse routing 시 더 중요해짐
- **gate_entropy_lambda**: gate entropy 조절
- **rule_agreement_lambda**: learned router와 rule prior 정렬
- **group_prior_align_lambda**: factored 라우터의 group prior 정렬
- **factored_group_balance_lambda**: factored group 분배 균형

---

## 2. 현재까지의 결과

### 2-1. KuaiRec 기준 최고 성능 조합

**공동 1위권 (mrr@20 ≈ 0.0811)**

| combo | mrr@20 | test_mrr | 핵심 차이 |
|---|---|---|---|
| A05 (phase1) | 0.0811 | 0.1622 | standard + gated_bias + both |
| PA05 (phase2) | 0.0811 | 0.1622 | A05 재현 (phase2 anchor) |
| P3S2_01 (phase3) | 0.0811 | 0.1621 | factored + group_gated_bias + both |

vs SASRec (P01): mrr@20 = 0.0785 → **+3.3% 상대 개선**

**A05 / PA05 고정 설정 (현재 anchor)**:
```
layer_layout: ["macro", "mid", "micro"]
stage_compute_mode: {all: moe}
stage_router_source: {all: both}
stage_router_type: {all: standard}
stage_router_mode: {all: learned}
stage_feature_injection: {all: gated_bias}
stage_router_granularity: {macro: session, mid: session, micro: token}
stage_feature_encoder_mode: {all: linear}
embedding_size: 128, expert_scale: 3, moe_top_k: 0 (dense)
balance_loss_lambda: 0.004, z_loss_lambda: 0.0001
```

**P3S2_01 설정 (factored 계열)**:
```
(A05와 동일, 아래만 다름)
stage_router_type: {all: factored}
stage_feature_injection: {all: group_gated_bias}
```

### 2-2. 단계별 개선 경로 (KuaiRec mrr@20)

| 단계 | 대표 combo | mrr@20 | 핵심 변경 |
|---|---|---|---|
| SASRec baseline | P01 | 0.0785 | - |
| stage wrapper 도입 | D10/D11 | 0.0785 | dense_plain, feature 없음 |
| gated_bias 도입 | D14/D15 | 0.0785 | dense + gated_bias |
| MoE 최초 도입 | M22 | 0.0798 | moe + both + no injection |
| MoE + injection 결합 | A05/PA05 | 0.0811 | moe + both + gated_bias |
| router 구조 확장 | P3S2_01 | 0.0811 | factored + group_gated_bias |

### 2-3. 축별 실험 결과 분석

#### stage_compute_mode
- moe가 dense_plain 대비 일관되게 우위 (+0.001~+0.003)
- dense_plain은 SASRec 수준에 머물렀음 (stage wrapper 비용만 추가)
- **결론**: 메인 트랙은 moe

#### stage_router_source
- R32 (feature only): mrr=0.0611, micro top1_max_frac=0.747, mid top1_max_frac=0.865 → **심각한 routing collapse**
  - feature ablation route_change_under_feature_zero = 0.91 (라우팅이 feature에 완전 종속)
  - test_mrr@20 = 0.1243 vs M22 0.1603 → 큰 폭 열화
- R31 (hidden only): mrr=0.0783, feature ablation 지표 모두 0 (정상, feature 미사용)
- M22/A05 (both): 상위권. feature ablation route_change ≈ 0.08~0.13 (feature가 라우팅에 의미있게 기여하지만 과의존 아님)
- **결론**: both가 기본값. feature-only는 collapse 위험성 명확하게 확인됨

#### stage_router_type
- standard (A05/PA05): mrr=0.0811
- factored (P3S2_01): mrr=0.0811으로 동등
- factored의 diag 패턴이 standard보다 group entropy가 고르게 분산
- **결론**: 둘 다 생존. 공정 비교 필요

#### stage_feature_injection
- none (M22): mrr=0.0798, feature_zero_delta_mrr ≈ -0.004 (injection 없어도 라우팅에 feature 영향)
- gated_bias (A05): mrr=0.0811 (+0.0013 vs M22)
- group_gated_bias (P3S2): mrr=0.0811, factored와 조합 시 dynamics 더 풍부
- film (D12/D13): dense에서 역할 제한됨, moe와 조합 실험 제한적
- **결론**: gated_bias 계열이 현재 최강

#### moe_top_k (dense vs sparse)
- X63 (top_k=2): mrr=0.0795 vs M22(dense)=0.0798
  - diag: X63에서 n_eff가 top_k=2에 맞게 낮아짐 (당연)
  - routing entropy 감소, expert concentration 증가
- **결론**: 현재 dense가 더 안정적. sparse는 추가 검증 필요

#### Aux/Reg
- PC (강한 balance/z_loss): PC01~PC08 모두 A05/PA05 대비 열세
- PB (wd 고정): 특별한 이점 없음, test 안정성 소폭 개선
- PD08/PD10 (weak group_prior_align): 중상위 수준 보조
- PD11 (factored_strong_group_balance): 성능 하락
- **결론**: 약한 규제만 효과적. 현재 기본값(balance=0.004, z=0.0001)이 적절

#### layer_layout
- phase1 L08 (deep layer prefix): 상위권 진입 기록 있음
- phase3 S4 (deep prefix family): S1/S2 대비 현재 열세
- L05 (macro repeated): mrr=0.0811 도달 사례 있으나 불안정
- **결론**: layout 효과는 다른 축과 결합 의존성이 커서 격리 판단 어려움

### 2-4. lastfm 결과

| combo | mrr@20 | 비고 |
|---|---|---|
| SASRec (P01) baseline | 0.4020 | - |
| M21 (macro+mid moe) | 0.4034 | lastfm 최고점 |
| M22 (full moe) | 0.4024 | |
| R30 (rule_soft) | 0.4027 | |
| R31 (hidden only) | 0.4026 | |
| R32 (feature only) | 0.4029 | lastfm에선 collapse 없음! KuaiRec와 다른 패턴 |
| T50 (mid token) | 0.4006 | |

- lastfm은 feature-only router(R32)가 collapse 없이 정상 작동 (top1_max_frac ≈ 1.0 but test_mr이 정상)
- 상위 20개 MRR@20이 0.401~0.403으로 촘촘함. 구조 차이 효과가 KuaiRec보다 작음

### 2-5. 모델 동작 분석 (diag 지표 해석)

현재 diag 지표로 확인 가능한 것들:

| 지표 | 의미 | 위험 신호 |
|---|---|---|
| top1_max_frac | 가장 많이 쓰인 expert가 전체의 몇 % | > 0.7 (특정 expert 과집중) |
| n_eff | 유효 expert 수 (entropy 기반) | 너무 낮음 = collapse |
| cv_usage | expert 사용 분포의 변동계수 | 너무 높음 = 불균형 |
| route_jitter_adjacent | 인접 시퀀스 위치 간 route 변화 | 너무 높음 = 불안정 |
| route_jitter_session | 세션 내 route 일관성 | - |
| stage_delta_norm | stage 출력의 크기 | 너무 작음 = stage 비기여 |
| condition_norm | conditioning signal 크기 | 0에 가까움 = feature 미작동 |
| feature_ablation.route_change_under_shuffle/zero | feature가 라우팅 결정에 미치는 영향 | - |
| family-wise delta_mrr | 각 feature family의 성능 기여 | - |

**R32 collapse 패턴** (feature-only, KuaiRec):
- micro top1_max_frac = 0.747, mid top1_max_frac = 0.865 → severe collapse
- feature_ablation.route_change_under_feature_zero = 0.91 (라우팅이 feature에 완전 종속)
- mrr@20 = 0.0611 (SASRec의 78% 수준으로 크게 열화)

---

## 3. 추가/변경/제거 후보

### 3-1. 추가/변경: Residual 재설계 (최우선)

**배경 (관련 논문)**:  
- DeepSpeed-MoE: "dense fallback branch + routed expert branch"를 공식 구현에서 지원
- HyMoERec (SeqRec): shared + specialized branch + adaptive alpha fusion이 직접 참고 사례
- DeepSeekMoE: 공통 지식은 shared experts가, 특화는 routed experts가 담당

**현재 구조 문제**:  
MoE가 라우팅으로 specialization을 만들려 하는데, residual 경로에서 shared branch(공통 feedforward)와 역할이 구분되지 않아 specialization이 약해질 수 있음.

**제안 실험 (R0~R6 순서로)**:

```
R0 (현재 baseline): out = h + MoE(norm(h))
R1:                  out = h + SharedFFN(norm(h))                     ← MoE 없이 shared만
R2:                  out = h + SharedFFN(norm(h)) + 0.3 * MoE(norm(h)) ← alpha 고정 0.3
R3:                  out = h + SharedFFN(norm(h)) + 0.5 * MoE(norm(h)) ← alpha 고정 0.5
R4:                  out = h + SharedFFN(norm(h)) + alpha * MoE(norm(h)) ← alpha scalar 학습
R5:                  out = h + SharedFFN(norm(h)) + alpha_s * MoE(norm(h)) ← stage별 alpha 학습
R6:                  out = h + SharedFFN(norm(h)) + warmup(t)*sigmoid(a_s)*MoE(norm(h)) ← alpha scheduling
```

R6 구체 설계:
- `a_s`: macro/mid/micro 각각 독립 학습 파라미터
- warmup(t): epoch 기반 스케줄 (0→1로 선형 증가)
- 해석: 초반에 SharedFFN이 공통 지식 흡수 → 후반에 MoE specialization이 점진적으로 개입

초기값 권장:
- macro: a_s 작게 시작 (macro는 넓은 context, shared 역할 크게)
- micro: a_s 크게 시작 (micro는 세밀한 specialization이 더 유리)

**예상 효과**:
- Initialization 안정화 → 초반 collapse 위험 감소
- Shared/routed 역할 명시적 분리 → 해석 가능성 향상
- z_loss 의존도 감소 (alpha warmup이 자연스러운 stabilizer 역할)

### 3-2. 추가/변경: Dense vs Sparse (top-k) 체계적 비교

**배경 (관련 논문)**:  
- Switch Transformer: top-1 routing으로 같은 FLOPs에서 최대 7x pretraining 효율
  - "정확도 자체보다 조건부 계산 효율"이 핵심 메시지
- ST-MoE: sparse routing의 training instability 문제를 z_loss로 해결

**제안 실험 순서**:

```
1. Dense (현재)    : moe_top_k=0          ← baseline 유지
2. Global top-1    : moe_top_k=1          ← Switch Transformer 방식
3. Global top-2    : moe_top_k=2          ← ST-MoE 방식
(factored winner가 결정된 후)
4. Group-wise top-1: factored + group내 top-1
5. Group-wise top-2: factored + group내 top-2
```

**제약 조건**:
- 처음 sparse 실험에서는 capacity/drop_token 없이 "select + renorm"만 구현
- sparse 쓸 때는 z_loss_lambda를 0에서 시작해 0.0001 → 0.001로 단계적 확인
- expert collapse 체크를 diag로 필수 확인

### 3-3. 추가: Router family 정면 비교

Residual winner를 고정한 뒤:

```
Round A: standard + gated_bias (=A05 anchor)
           vs
           factored + group_gated_bias (=P3S2 anchor)
(동일 budget / 동일 seed / 동일 lr search space)

Round B: winner에 대해 source 비교
         - both (현재 기본값)
         - hidden (order: both 다음 안전한 옵션)
         - feature (제한적, 위험 신호 있음)
```

### 3-4. 추가: Weak z-loss 안정성 확인

Winner 구조에 대해서만:
- no z-loss (z_loss_lambda=0)
- weak z-loss (z_loss_lambda=0.0001, 현재 기본)
- mid z-loss (z_loss_lambda=0.001)

특히 sparse top-k 사용 시 z_loss 필요성이 커짐 (ST-MoE 결과 참조).

### 3-5. 모델 동작 분석 로깅 설계

**현재 기록 중 (diag)**:  
- top1_max_frac, n_eff, cv_usage, route_jitter (adjacent/session), stage_delta_norm, condition_norm

**현재 기록 중 (special_metrics)**:  
- popularity slice (<=5, 6-20, 21-100, >100)  
- session_len slice (1-2, 3-5, 6-10)

**현재 기록 중 (feature_ablation)**:  
- family별 route_change_under_shuffle/zero  
- family별 delta_mrr  
- overall feature shuffle/zero delta

**Residual 실험 시 추가 필요 지표**:

| 지표 | 목적 |
|---|---|
| shared_branch_norm (per stage) | SharedFFN 출력 크기 |
| moe_branch_norm (per stage) | MoE 출력 크기 |
| alpha_value_trajectory (per stage) | α 학습 방향 추적 |
| shared_vs_moe_weight_ratio | 두 branch의 실질 기여 비율 |

**Sparse routing 시 추가 필요 지표**:

| 지표 | 목적 |
|---|---|
| active_expert_count (per stage) | 실제로 활성화된 expert 수 |
| token_drop_fraction | capacity overflow로 버려진 token 비율 |

**현재 지표 중 통합/변경 후보**:

- route_jitter_adjacent와 route_jitter_session: 현재 두 개를 모두 기록하지만, 기본 리포트에선 adjacent만 표시하고 session은 supplementary로 이동 가능
- feature_ablation은 비용이 높아 best trial에만 수행 중. 이 정책은 유지

**Special slices 확장 후보**:

- 현재: popularity, session_len
- 추가 검토: user_activity_tier (신규/중간/헤비), repeat_item_rate (재방문 비율)
  - 단, 현재 dataset에 user feature가 충분히 있는지 확인 필요

### 3-6. 제거/축소 후보

| 항목 | 근거 |
|---|---|
| feature-only 라우팅 대규모 재탐색 | R32 KuaiRec collapse 패턴 명확. 제한적 확인으로 충분 |
| 강한 aux/reg 반복 탐색 | PC01~PC11 실험 결과 대부분 PA05 열세. 추가 탐색 비효율 |
| deep layout 계열 광범위 확장 | P3S4가 현재 S1/S2 대비 성능 열세 |
| L02/L03 (broken layouts) 재실험 | 초기 trial에서 0.06 수준 collapse, 재실험 가치 없음 |
| Global top-k 조기 capacity/drop 도입 | sparse routing 초기에는 "select+renorm"으로 충분, drop/overflow는 후순위 |
| head-wise/facet-wise MoE (FAME 방식) | 현재 단계에서 복잡도 대비 이득 불확실. 후순위 |

---

## 4. Ablation study 충분성 확인 및 실험 설계

### 4-1. 현재 ablation 충분성 체크

| 축 | 현재 비교 | 충분성 | 부족한 부분 |
|---|---|---|---|
| MoE vs Dense | M22 vs D10~D15 | ✓ 충분 | - |
| source: both vs hidden vs feature | M22/A05 vs R31 vs R32 | △ 부분 충분 | feature의 routing vs injection 역할 분리 필요 |
| router type: standard vs factored | PA05 vs P3S2_01 | △ 공정 비교 미완 | 동일 budget/seed 정면 비교 필요 |
| injection: none vs gated_bias | M22 vs A05 | ✓ 충분 | - |
| injection: gated_bias vs group_gated_bias | (각기 다른 router type과 묶임) | △ 미분리 | standard + group_gated_bias 실험 없음 |
| dense vs sparse (top-k) | X63 (top_k=2 only) | ✗ 부족 | top-1 / global vs group-wise 추가 필요 |
| residual 구조 | 없음 | ✗ 미실험 | R0~R6 전체 필요 |
| alpha/shared+moe 분리 | 없음 | ✗ 미실험 | shared FFN 단독 효과 필요 |
| aux/reg: 없음 vs 약 vs 강 | PC 블록 (규제 위주) | △ 부분 | z-loss 단독 효과 미분리 |
| family mask | X61 (Tempo+Memory only) | ✗ 부족 | 반복 검증과 다른 조합 필요 |
| lastfm에서 구조 일반화 | core 일부만 | △ KuaiRec 대비 부족 | phase2 winner 이식 필요 |

### 4-2. 앞으로의 실험 순서

#### Phase 4A: Residual redesign (최우선)

목표: shared FFN + alpha MoE 구조가 현재 baseline을 실제로 개선하는가?

```
R0 = 현재 anchor (PA05 or P3S2_01)
R1 = anchor + SharedFFN (MoE 제거, SharedFFN만)
R2 = anchor + SharedFFN + 0.3*MoE
R3 = anchor + SharedFFN + 0.5*MoE
R4 = anchor + SharedFFN + learnable_alpha * MoE
R5 = anchor + SharedFFN + stage_alpha * MoE (stage별 각자)
R6 = anchor + SharedFFN + warmup(epoch)*stage_alpha * MoE
```

기록 추가: shared_branch_norm, moe_branch_norm, alpha_trajectory  
데이터셋: KuaiRec (탐색) → best 1~2개를 lastfm 이식

#### Phase 4B: Router family 정면 비교

전제: Phase 4A에서 residual winner 확정 후 진행

```
B1: residual_winner + standard + gated_bias   (= S1 anchor 재확인)
B2: residual_winner + factored + group_gated_bias (= S2 anchor)
(동일 seed / lr search space)
```

그 다음 winner에 대해서만:
```
B3: winner + source=both    ← 현재 기본값 확인
B4: winner + source=hidden
B5: winner + source=feature (주의: collapse 모니터링)
```

#### Phase 4C: Sparse routing 실험

전제: Phase 4B winner 고정 후 진행

```
C1: top_k=0 (dense)   ← PA05/P3S2 수준 재확인용
C2: top_k=1 (global)  ← Switch Transformer 방식
C3: top_k=2 (global)  ← ST-MoE 방식
(factored winner면):
C4: group_top_k=1     ← group-wise sparse
C5: group_top_k=2
```

sparse 시 diag 추가: active_expert_count, token_drop_fraction, routing_entropy 감소 추적

#### Phase 4D: Weak z-loss 체계적 확인

```
D1: z_loss=0                ← 없음
D2: z_loss=0.0001 (현재 기본) ← 약
D3: z_loss=0.001            ← 중간
```

특히 Phase 4C(sparse)에서 D3가 필요할 가능성이 높음.

#### Phase 4E: Special performance 확인 (비교용 실험 아님)

Phase 4A~4D best config에 대해:
- Popularity slice 비교 (<=5 / 6-20 / 21-100 / >100)
- Session length slice 비교 (1-2 / 3-5 / 6-10)
- Feature ablation (family별 route_change 및 delta_mrr)

이 단계는 "더 좋은 config를 뽑는 실험"이 아니라  
"왜 좋아졌는지 설명하는 분석"이다.

#### Phase 5: 성능 끌어올리기 + 파라미터 연구

Phase 4 best config를 anchor로:
- len30 (X62 스타일)
- family mask 반복 검증 (Tempo/Memory vs full)
- expert_scale 3 vs 4
- encoder complex
- lastfm 이식 (K1/K2 결과 → lastfm 검증)

#### 크로스 데이터셋 검증

Phase 4 이후 KuaiRec top 2~3 config를 lastfm에 이식:
- 목표: "KuaiRec 특화 현상인가, 일반 패턴인가" 판단
- lastfm에서도 residual alpha 효과가 나오면 논문 클레임 가능

---

## 5. 코드/심볼 참조 부록 (에이전트용)

### 모델 구현 핵심 파일

```
experiments/models/FeaturedMoE_N3/
  featured_moe_n3.py     ← 전체 설정 파싱, forward, loss 합산
  stage_executor.py      ← layer_layout 토큰 파싱, 실행 그래프 컴파일
  stage_modules.py       ← StageBlock: routing/injection/factored/rule_soft 구현
```

Residual 구조 수정 위치:
- `stage_modules.py`의 `StageBlock.forward()` 내부
- SharedFFN branch 추가 시 `__init__`에 `nn.Linear` 추가 필요
- alpha 파라미터는 `nn.Parameter` 로 선언

### 실험 정의 파일

```
experiments/run/fmoe_n3/
  run_core_28.py      ← core_ablation_v2 (28 combos: P00~C72)
  run_phase1_28.py    ← phase1_upgrade_v1 (28 combos: A01~X65)
  run_phase2_40.py    ← phase2_router_v1 (40 combos: PA01~PD12)
  run_phase3_20.py    ← phase3_focus_v1 (20 combos: P3S1_01~P3S4_05)
```

### 결과/로그 입력 소스

```
experiments/run/artifacts/logs/fmoe_n3/
  {phase}/
    {phase}_summary.csv              ← 전체 trial mrr/hr
    {phase}_diag_summary.csv         ← diag 지표 (best trial의 routing dynamics)
    {phase}_special_summary.csv      ← popularity/session slice 성능
    {phase}_feature_ablation_summary.csv ← feature family별 contribution

experiments/run/artifacts/logs/fmoe_n3/special/
  {phase}/{combo}/{dataset}/FMoEN3/*_special_metrics.json  ← 상세 slice JSON

experiments/run/artifacts/logs/baseline/
  KuaiRecLargeStrictPosV2_0.2/P0_summary.csv
  lastfm0.03/P0_summary.csv
```

### 핵심 config 파라미터 심볼

```python
# stage-wise 설정 (dict: macro/mid/micro 각각)
stage_compute_mode       # none | dense_plain | moe
stage_router_source      # hidden | feature | both
stage_router_type        # standard | factored
stage_router_mode        # none | learned | rule_soft
stage_router_granularity # session | token
stage_feature_injection  # none | film | gated_bias | group_gated_bias
stage_feature_encoder_mode # linear | complex
stage_feature_family_mask  # ["Tempo","Memory","Focus","Exposure"] 등

# 전역 설정
moe_top_k                # 0=dense, k>0=sparse
expert_scale             # expert 수 multiplier
macro_history_window     # macro feature 시간 창
MAX_ITEM_LIST_LENGTH     # sequence 길이

# aux/reg
balance_loss_lambda      # expert 균형 손실
z_loss_lambda            # router logit 안정화 (ST-MoE)
gate_entropy_lambda      # entropy 정책
rule_agreement_lambda    # rule prior 정렬
group_prior_align_lambda # factored group prior 정렬
factored_group_balance_lambda  # factored group 균형

# 학습 설정
lr_scheduler_type        # warmup_cosine | plateau
lr_scheduler_warmup_ratio
lr_scheduler_min_lr_ratio
```

### Best combo 빠른 참조

```
A05  → P1/KuaiRecLargeStrictPosV2_0.2/FMoEN3/005_A05_both_plus_gated_bias.log
PA05 → P2/KuaiRecLargeStrictPosV2_0.2/FMoEN3/003_PA05_all_gated_bias.log
P3S2_01 → P3/KuaiRecLargeStrictPosV2_0.2/FMoEN3/006_P3S2_01_s2_base_factored_groupinj.log
M21 (lastfm best) → CORE28/lastfm0.03/FMoEN3/010_M21_macro_mid_moe_both.log
R32 (collapse 참고) → CORE28/KuaiRecLargeStrictPosV2_0.2/FMoEN3/014_R32_full_moe_feature_only.log
```
