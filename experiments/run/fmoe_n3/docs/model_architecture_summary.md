# FMoE_N3 통합 모델 설명서 (실험 설계/해석용)

작성일: 2026-03-16
대상 축: core_ablation_v2, phase1_upgrade_v1, phase2_router_v1, phase3_focus_v1
우선 데이터셋: KuaiRecLargeStrictPosV2_0.2, lastfm0.03
기본 지표: MRR@20
비교 기준: SASRec baseline

---

## 1) 이 문서를 어떻게 보면 좋은가

- 먼저 2~5장을 읽어서 "무엇을 바꿀 수 있고, 각 옵션이 무엇을 의미하는지"를 잡는다.
- 그 다음 6~8장을 읽어서 "지금까지 결과상 무엇이 먹혔고, 무엇을 줄일지"를 결정한다.
- 마지막 9~11장으로 "발표용 ablation 커버리지"와 "다음 실험 우선순위"를 확정한다.
- 코드 경로/심볼은 맨 아래 부록에만 모아두었다.

---

## 2) 모델 한 줄 요약

FMoE_N3는 SASRec 기반 시퀀스 모델 위에 stage 단위(Macro/Mid/Micro) MoE 라우팅과 feature 조건부 주입을 올린 구조다.

핵심 직관:

- stage마다 같은 방식으로 돌지 않는다.
- 라우터 입력을 hidden/feature/both로 바꿀 수 있다.
- MoE를 켜거나(dense_plain 대비) 끌 수 있다.
- feature를 "라우팅에만" 쓸지, "표현 보정(gated bias/FiLM)"에도 쓸지 고를 수 있다.

---

## 3) 결과 스냅샷 (지금까지)

### 3-1. KuaiRecLargeStrictPosV2_0.2

- FMoE_N3 관측 최고 mrr@20: 0.0811
  - 대표: A05, PA05, P3S2
- SASRec baseline mrr@20: 0.0785
- 상대 개선: 약 +0.0026

### 3-2. lastfm0.03

- FMoE_N3 관측 최고 mrr@20: 0.4034 (core M21)
- SASRec baseline mrr@20: 0.4020
- 상대 개선: 약 +0.0014

해석:

- KuaiRec에서는 구조/라우팅 축 변경의 효과가 더 선명하게 보인다.
- lastfm은 절대 점수가 높고 상위 조합이 촘촘해 "큰 폭 개선"보다 "소폭 안정 개선" 형태가 많다.

---

## 4) 변경 가능한 축 정리 (옵션 의미 + 현재까지 결과)

아래는 축별로
- 무엇을 의미하는지
- 옵션별 의미
- 지금까지 관측 결과
를 같이 정리했다.

### A. 전역 최적화 축 (lr/wd/dropout/scheduler)

- 의미:
  - 구조는 고정하고 학습 안정성과 수렴 지점을 조정하는 축

- 주요 파라미터:
  - lr_min/lr_max
    - 의미: 하이퍼옵트가 탐색할 LR 구간
    - 관측: KuaiRec 상위권은 대체로 중간 LR 대역에서 안정
  - search_weight_decay
    - 의미: 가중치 규제 세기
    - 관측: phase2 PB 블록에서 wd 고정 탐색은 큰 점프는 없지만 test 안정성은 좋았음
  - search_hidden_dropout_prob
    - 의미: 모델 regularization 강도
    - 관측: 과도하게 낮거나 높으면 변동성 증가
  - lr_scheduler_type
    - warmup_cosine
      - 의미: 초반 워밍업 후 코사인 감쇠
      - 관측: S02 계열에서 상단권 유지
    - plateau
      - 의미: 개선 정체 시 단계적 LR 감소
      - 관측: S03도 상위권, 두 스케줄러 모두 경쟁력 있음

- 현재 결론:
  - "스케줄러 없는 공격적 LR"보다 warmup_cosine/plateau 계열이 안정적으로 유리

---

### B. Layer 구성 축 (layer_layout)

- 의미:
  - attention/ffn/stage 연산 순서와 개수를 직접 정의

- 옵션/토큰:
  - layer
    - 의미: SASRec 스타일 attn+ffn 한 블록
  - attn
    - 의미: attention만 단독 수행
  - ffn
    - 의미: ffn만 단독 수행
  - macro/mid/micro
    - 의미: 해당 stage attention + stage 전용 ffn/moe 수행
  - macro_ffn/mid_ffn/micro_ffn
    - 의미: stage attention 없이 해당 stage ffn/moe만 수행

- 현재 관측:
  - L08(deep layer prefix)가 phase1에서 꽤 괜찮게 나왔지만,
  - phase3 S4(깊은 prefix family)는 현재 S1/S2 대비 열세

- 현재 결론:
  - layer_layout은 "완전한 승자"가 아니라 데이터셋/다른 축과 상호작용이 큼

---

### C. Stage compute 축 (stage_compute_mode)

- 의미:
  - stage ffn을 어떤 계산 방식으로 돌릴지 선택

- 옵션:
  - none
    - 의미: stage ffn 경로 비활성화
    - 해석: 거의 ablation/디버깅 용도
  - dense_plain
    - 의미: MoE 없이 단일 dense ffn
    - 해석: 라우팅 비용 없이 단순/안정 비교 기준
  - moe
    - 의미: expert mixture로 ffn 계산
    - 해석: 표현력 증가, 대신 collapse/불안정 리스크 관리 필요

- 현재 관측:
  - 상위권은 대부분 moe 기반
  - dense_plain은 비교군으로 의미는 있으나 최고점 경쟁력은 낮음

- 현재 결론:
  - 메인 트랙은 moe 유지가 타당

---

### D. Router 입력 축 (stage_router_source)

- 의미:
  - 라우터가 어떤 신호로 expert를 고를지 정의

- 옵션:
  - hidden
    - 의미: sequence hidden 상태만으로 라우팅
    - 관측: 중간~양호, 극단 붕괴는 상대적으로 적음
  - feature
    - 의미: engineered feature 기반 라우팅
    - 관측: S3 family에서 중상위 가능하나, pure feature 조합은 리스크 존재
  - both
    - 의미: hidden + feature 결합 라우팅
    - 관측: A05/PA05 등 상위권 빈도가 가장 높음

- 현재 결론:
  - 기본값은 both가 가장 안전
  - feature-only는 제한적으로 검증

---

### E. Router 형태 축 (stage_router_type, stage_router_mode)

- 의미:
  - expert 선택 로직의 구조를 결정

- 옵션:
  - stage_router_type = standard
    - 의미: 단일 expert logits를 직접 생성
    - 관측: A05/PA05 등 최고점 다수
  - stage_router_type = factored
    - 의미: group logits + intra-group logits을 합성
    - 관측: PD04/P3S2가 상단권, 구조적으로 유망

  - stage_router_mode = learned
    - 의미: 학습 가능한 라우터 네트워크
    - 관측: 현재 메인 상위권의 중심
  - stage_router_mode = rule_soft
    - 의미: 규칙 기반 prior를 활용한 소프트 라우팅
    - 관측: core에서 의미 있는 비교축이었으나, phase2/3 주축은 learned
  - stage_router_mode = none
    - 의미: 라우팅 비활성 (compute_mode와 결합 제약)

- 현재 결론:
  - standard vs factored는 둘 다 살려둘 가치가 있음
  - learned가 기본, rule_soft는 보조/해석 실험용

---

### F. Feature 주입 축 (stage_feature_injection)

- 의미:
  - feature를 표현 보정(residual branch)에 어떻게 주입할지 결정

- 옵션:
  - none
    - 의미: feature 보정 없음
  - film
    - 의미: scale/shift 형태로 hidden 변조
  - gated_bias
    - 의미: gate와 bias로 hidden 보정
    - 관측: A05/PA05 핵심 패턴, 상위권에서 반복적으로 강함
  - group_gated_bias
    - 의미: feature group별 다른 gated bias를 expert group에 연결
    - 관측: PD04/P3S2에서 강함, factored와 조합 시 특히 유효

- 현재 결론:
  - gated_bias 계열이 현재까지 가장 실전적

---

### G. Router granularity 축 (stage_router_granularity)

- 의미:
  - 라우팅 결정을 세션 단위로 할지, 토큰 단위로 할지

- 옵션:
  - session
    - 의미: 한 세션/샘플에서 공유된 라우팅 경향
    - 관측: 안정적
  - token
    - 의미: 위치별로 라우팅을 다르게 적용
    - 관측: T50/T51에서 test 지표가 나쁘지 않음, 대신 계산량/변동성 증가

- 제약:
  - micro stage는 token 고정

- 현재 결론:
  - 기본은 session + micro token
  - all-token은 선택적 고비용 실험 축

---

### H. Feature 표현 축 (d_feat_emb, encoder mode, family mask, window)

- 의미:
  - feature를 어떤 표현으로 만들고 어떤 그룹을 쓸지 결정

- 옵션:
  - stage_feature_encoder_mode
    - linear
      - 의미: 단순 선형 투영
      - 관측: 안정성 좋고 baseline 역할 우수
    - complex
      - 의미: 2층 비선형 인코더
      - 관측: E40~E42에서 경쟁력, 항상 이득은 아님

  - stage_feature_family_mask
    - 의미: Tempo/Memory/Focus/Exposure 중 사용할 그룹 제한
    - 관측: X61류 제한 실험은 의미 있으나, 광범위 일반화 증거는 아직 부족

  - macro_history_window
    - 의미: macro stage 과거 통계 반영 범위
    - 관측: X60(window 10) vs 기본(window 5)에서 일관 우위는 미확정

- 현재 결론:
  - linear/complex 둘 다 유지
  - family mask는 진짜 선택 근거를 위해 반복 실험이 더 필요

---

### I. 용량 축 (embedding/expert_scale/len)

- 의미:
  - 모델 표현력과 비용을 직접 바꾸는 축

- 옵션:
  - embedding_size
    - 관측: embed256(X65)가 test 상단 근처 사례는 있으나 일관 우위 증거 부족
  - expert_scale
    - 관측: C70(expert_scale=3)가 KuaiRec 상단
  - MAX_ITEM_LIST_LENGTH
    - 관측: len30(X62/N30*)는 유의미한 개선 사례 존재
    - len50은 케이스별 편차 큼

- 현재 결론:
  - len30은 유지 가치 높음
  - 대형화는 비용 대비 이득이 불확실하므로 제한적으로

---

### J. Aux/Reg 축

- 의미:
  - collapse 방지와 라우팅 품질을 위한 보조 손실

- 주요 파라미터와 해석:
  - balance_loss_lambda
    - 의미: expert 사용 균형 유도
    - 관측: 너무 강하면 성능 저하 사례
  - z_loss_lambda
    - 의미: router logits 안정화
    - 관측: 약하게 둘 때 유리한 경우가 많음
  - gate_entropy_lambda
    - 의미: 게이트 엔트로피 조절
    - 관측: 과강도는 noise 유발 가능
  - rule_agreement_lambda
    - 의미: learned router와 rule prior 정렬
  - group_prior_align_lambda
    - 의미: group prior와 라우팅 정렬
    - 관측: PD08/PD10에서 중상위권 보조
  - factored_group_balance_lambda
    - 의미: factored group 분배 균형
    - 관측: PD11류 강규제는 오히려 불리

- 현재 결론:
  - aux/reg는 "약한 규제"가 대체로 유리

---

## 5) 축별로 지금까지 보이는 결과 요약 (한눈표)

- stage_compute_mode
  - moe: 메인 상위권 대부분
  - dense_plain/none: 비교군 역할, 최고점은 제한적
- stage_router_source
  - both: 가장 일관적으로 강함
  - feature: 가능성은 있으나 조합 의존 큼
  - hidden: 중간 안정권
- stage_router_type
  - standard: 최고점 많이 배출
  - factored: PD04/P3S2로 경쟁력 확인
- stage_feature_injection
  - gated_bias: 현재 베스트 패턴
  - group_gated_bias: factored와 결합 시 강함
  - film/none: 메인 상위권 빈도는 상대적으로 낮음
- layer_layout
  - 효과는 있으나 다른 축과 결합 영향이 큼
- length
  - len30는 유지 가치 높음
  - len50는 비용 대비 불확실
- aux/reg
  - 약한 정규화는 도움, 강한 규제는 역효과 가능

---

## 6) 추가/변경/제거 후보 (코드/실험 관점)

### 6-1. 추가/변경: Residual 재설계 (요청사항 반영)

현재 문제 인식:

- N3 residual이 "생 hidden + MoE 출력" 형태에 가까워, shared FFN과 MoE 역할 분리가 약함

제안 실험안:

1. Shared FFN + MoE 가중합 residual
- 구조:
  - h_shared = SharedFFN(hidden)
  - h_moe = MoE(hidden or conditioned hidden)
  - out = hidden + beta * h_shared + alpha * h_moe
- 실험 포인트:
  - alpha 고정 vs 학습 가능
  - beta 고정(1.0) vs 학습 가능
  - beta = 1-alpha 결합형 vs 독립형

2. Alpha learnable + scheduling
- 구조:
  - alpha_stage in {macro, mid, micro} 각각 학습 파라미터
  - epoch 진행에 따라 alpha 상한/하한 또는 target curve 부여
- 실험 포인트:
  - stage별 초기 alpha 다르게 설정
  - warmup 구간에서는 alpha를 낮게 시작 후 점증

3. Stage별 다른 residual mixing 정책
- 예:
  - macro: shared 비중 높게
  - mid: shared/moe 균형
  - micro: moe 비중 높게

예상 효과:

- 초반 학습 안정화
- collapse 완화(특정 expert 과의존 방지)
- 구조 설명력 향상(발표시 "왜 좋아졌는지" 설명 가능)

---

### 6-2. 추가 실험 가치가 높은 항목

- standard vs factored를 동일 예산/동일 seed로 정면 비교
- feature_source(feature)와 both의 순수 비교(다른 축 고정)
- group_gated_bias 단독 효과 분리 실험
- family mask(Tempo/Memory/Focus/Exposure) 반복 검증
- residual alpha/beta 설계(위 6-1)

---

### 6-3. 제거/축소 후보 (시간 가성비 기준)

- feature_only 라우팅 대규모 재탐색
  - 근거: core R32 붕괴 사례가 명확
- 과도한 대형화(embedding/expert scale) 무차별 탐색
  - 근거: 계산비 증가 대비 일관 이득 미확정
- 강한 aux/reg 조합 반복
  - 근거: PD11류처럼 성능 하락/경직화 가능
- deep layout 계열의 광범위 확장
  - 근거: phase3 S4가 현재 상대 열세

---

## 7) 발표 관점: ablation study 충분성 점검

질문: "각 축 선택을 실험 결과로 설명 가능한가?"

현재 충족되는 부분:

- 라우터 구조 축
  - standard/factored 비교 신호 있음 (PD04/P3S2)
- injection 축
  - gated_bias/group_gated_bias 강점 근거 있음
- source 축
  - both 우세 신호 있음
- scheduler 축
  - S02/S03 근거 있음

현재 부족한 부분:

- residual 설계 축
  - 아직 체계적 ablation 없음
- family mask 축
  - 반복/통계량 부족
- window/length/large capacity 축
  - 일부 점 실험은 있으나 축 전체 결론은 약함
- lastfm에서 phase2/3 구조 일반화 검증
  - KuaiRec 대비 데이터 포인트 부족

발표용으로 필요한 최소 보강:

1. Residual alpha/beta ablation (필수)
2. standard vs factored 공정 비교
3. both vs feature source 공정 비교
4. family mask 2~3개 대표 조합 반복
5. KuaiRec와 lastfm에 동일 축 최소 1회 교차 검증

---

## 8) 다음 실험 우선순위 (실행 제안)

1. Residual mixing 실험 세트 (최우선)
- 목적: 구조 설명력 + 성능 개선 동시 노림
- 설정:
  - A: alpha 고정(0.3/0.5/0.7), beta=1
  - B: alpha 학습, beta=1
  - C: alpha/beta 둘 다 학습
  - D: beta=1-alpha 제약형

2. PA05 vs PD04 vs P3S2 공정 비교
- 목적: 현재 상위 3구조 우열 확정
- 조건: seed/예산/스케줄 동일

3. lastfm 이식 검증 (소규모)
- 목적: KuaiRec 승자 구조의 데이터셋 일반화 확인
- 후보: PA05, PD04, residual best 1개

---

## 9) 운영 체크리스트

- 한 번에 한 축만 변경했는가?
- 같은 예산/seed로 비교했는가?
- MRR@20 + test_mrr@20 둘 다 기록했는가?
- diag(top1_max_frac, n_eff, jitter) 확인했는가?
- feature ablation route change를 확인했는가?
- 성능이 비슷하면 안정성 지표로 선택했는가?

---

## 10) 코드/심볼 참조 부록 (에이전트용)

아래는 본문에서 언급한 축의 코드 기준점이다.

- 메인 모델 초기화/파라미터 파싱
  - experiments/models/FeaturedMoE_N3/featured_moe_n3.py
- layer layout 실행 컴파일
  - experiments/models/FeaturedMoE_N3/stage_executor.py
- stage block 구현(라우팅/주입/factored/rule-soft)
  - experiments/models/FeaturedMoE_N3/stage_modules.py
- phase별 combo 정의
  - experiments/run/fmoe_n3/run_core_28.py
  - experiments/run/fmoe_n3/run_phase1_28.py
  - experiments/run/fmoe_n3/run_phase2_40.py
  - experiments/run/fmoe_n3/run_phase3_20.py
- 튜닝 엔진
  - experiments/hyperopt_tune.py
- 결과 요약 입력 소스
  - experiments/run/artifacts/results/fmoe_n3/diag/*/*/*/FMoEN3/trial_summary.csv
  - experiments/run/artifacts/logs/fmoe_n3/core_ablation_v2/core_ablation_v2_diag_summary.csv
  - experiments/run/artifacts/logs/fmoe_n3/core_ablation_v2/core_ablation_v2_feature_ablation_summary.csv
  - experiments/run/artifacts/logs/fmoe_n3/core_ablation_v2/core_ablation_v2_special_summary.csv
  - experiments/run/artifacts/logs/fmoe_n3/phase1_upgrade_v1/phase1_upgrade_v1_diag_summary.csv
- baseline 비교 입력
  - experiments/run/artifacts/logs/baseline/KuaiRecLargeStrictPosV2_0.2/P0_summary.csv
  - experiments/run/artifacts/logs/baseline/lastfm0.03/P0_summary.csv

심볼 키워드 빠른 검색:

- layer_layout
- stage_compute_mode
- stage_router_source
- stage_feature_injection
- stage_router_type
- rule_agreement_lambda
- group_prior_align_lambda
- factored_group_balance_lambda

