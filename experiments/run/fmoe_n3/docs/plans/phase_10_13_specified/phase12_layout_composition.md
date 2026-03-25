# Phase 12 — Layout / Attention Placement / Stage Composition

## 1. Phase 목표

이 phase의 핵심 질문은 다음이다.

> **같은 macro / mid / micro stage를 쓰더라도, 그것들을 어떤 layout과 composition으로 배치하느냐가 중요한가?**

즉 이 phase는 stage의 존재 여부를 묻는 것이 아니라,

- attention을 어디에 두는가,
- attn과 moe_ffn을 어떻게 섞는가,
- 특정 stage를 반복하는 것이 유리한가,
- 여러 stage를 한 번에 parallel / bundle로 쓰는 것이 serial stage보다 나은가,
- 여러 stage를 같이 쓸 때 aggregation을 어떻게 하는가

를 보는 phase다.

논문에서는 이 phase를 통해 다음 메시지를 노린다.

1. **serial stage decomposition이 단순 implementation artifact가 아니라 meaningful choice일 수 있다**
2. **attention placement가 routing quality에 영향을 준다**
3. **여러 horizon을 한 번에 섞는 것보다 분리해서 적용하는 것이 유리할 수도 있다**

---

## 2. 이 phase에서 주장하고 싶은 것

### main claim

> **The benefit of feature-aware stage routing depends on how stage computations are composed, not merely on which stages are present.**

### 이 phase가 잘 나오면 하고 싶은 말

- serial layout가 bundle/parallel보다 안정적이면:
  - “horizon separation이 중요하다”
- 특정 attention placement가 consistently 좋으면:
  - “routing 직전 contextualization 위치가 중요하다”
- all-stage bundle이 약하면:
  - “모든 horizon을 한 번에 섞는 것보다 structured decomposition이 중요하다”

### 기대와 다르게 나와도 쓸 수 있는 말

- bundle setting이 더 좋으면:
  - “strict serial decomposition보다 selected horizon interaction이 더 중요할 수 있다”
- shared attention 1회만으로 충분하면:
  - “compact composition이 더 낫다”
- 특정 stage 반복이 좋으면:
  - “그 stage는 더 많은 routing refinement depth를 필요로 한다”

---

## 3. 이 phase의 경계

이 phase는 **layout / composition**을 보는 phase다.

즉 여기서는
- stage가 왜 필요한가
- stage 순서가 왜 중요한가
같은 질문을 다시 다루지 않는다.

그건 Phase 11에서 이미 검증하는 축이다.

이 phase에서 다루는 질문은 오직 다음과 같다.

- 같은 stage set을 유지한 채 **배치 방식**을 바꾸면 어떻게 되는가?
- stage를 하나씩 따로 쓰는 것이 좋은가, 같이 묶는 것이 좋은가?
- attention을 어느 위치에 둘 때 routing이 잘 작동하는가?

---

## 4. Phase 구성 요약

총 **32개 세팅**.

- 4.1 Attention / layout variants: **10개**
- 4.2 Parallel / bundle stage composition: **15개**
- 4.3 8배수 보강: **7개**

---

## 4.1 Attention / layout variants (10개)

### 목적

이 축은 다음 질문에 답한다.

- stage-specific attention이 꼭 필요한가?
- shared attention 하나면 충분한가?
- 특정 stage만 extra refinement를 한 번 더 받으면 좋은가?
- attention 없이 moe_ffn만 연속으로 두는 것이 가능한가?

즉, 같은 stage set을 쓰더라도 **attn과 moe_ffn의 배치가 routing 품질에 어떤 영향을 주는지**를 보는 실험이다.

### 세팅 목록과 의미

- `P12-00_ATTN_ONESHOT`
  - `[attn, macro_ffn, mid_ffn, micro_ffn]`
  - attention 한 번만 하고 이후는 FFN-stage만 연속

- `P12-01_ATTN_MACRO_ONLY`
  - `[attn, macro_ffn, attn, mid_ffn, micro_ffn]`
  - macro 이후 한 번 더 attention으로 hidden 정리

- `P12-02_ATTN_MICRO_BEFORE`
  - `[attn, macro_ffn, mid_ffn, attn, micro_ffn]`
  - micro 진입 직전에 attention 추가

- `P12-03_NO_ATTN_ONLY_MOEFFN`
  - `[macro_ffn, mid_ffn, micro_ffn]`
  - attention 없이 stage FFN만 연속

- `P12-04_LAYER_PLUS_MOEFFN`
  - `[layer, macro_ffn, mid_ffn, micro_ffn]`
  - shared dense layer 하나 + stage FFN 연속

- `P12-05_MACRO_REPEATED`
  - `[macro, macro_ffn, mid, micro]`
  - macro stage를 한 번 더 refinement

- `P12-06_MID_REPEATED`
  - `[macro, mid, mid_ffn, micro]`
  - mid stage를 한 번 더 refinement

- `P12-07_MICRO_REPEATED`
  - `[macro, mid, micro, micro_ffn]`
  - micro stage를 한 번 더 refinement

- `P12-08_MACRO_NOLOCAL_ATTN`
  - `[macro_ffn, mid, micro]`
  - macro stage만 local attention 제거

- `P12-09_MID_NOLOCAL_ATTN`
  - `[macro, mid_ffn, micro]`
  - mid stage만 local attention 제거

### 각 세팅에서 보고 싶은 것

- `ATTN_ONESHOT`이 강하면:
  - attention은 꼭 stage마다 없어도 된다
- repeated-stage가 강하면:
  - 특정 horizon에서 extra refinement depth가 필요하다
- `NO_ATTN_ONLY_MOEFFN`가 너무 약하면:
  - routing 전에 context mixing이 꼭 필요하다
- 특정 stage만 no-local-attn이어도 괜찮으면:
  - 그 stage는 attention보다 feature-conditioned routing이 더 중요할 수 있다

### 이 축이 잘 나왔을 때 해석

- 특정 compact layout가 강함:
  - “feature-aware routing은 복잡한 stacking보다 적절한 placement가 중요하다”
- repeated-stage만 강함:
  - “특정 horizon은 더 깊은 routing refinement를 필요로 한다”

---

## 4.2 Parallel / bundle stage composition (15개)

### 목적

이 축은 현재 serial stage decomposition을 넘어서,
**여러 stage를 한 번에 묶어서 쓰면 어떻게 되는가**를 보는 실험이다.

핵심 질문은 두 가지다.

1. stage를 하나씩 따로 쓰는 것이 좋은가?
2. 특정 horizon pair는 joint하게 묶는 것이 더 나은가?

즉 이 실험은 “strict serial vs selected horizon interaction” 비교다.

### 구성 방식

bundle은 한 slot에서 여러 stage block 출력을 동시에 만든 뒤,
그 출력을 aggregation해서 다음 hidden으로 넘긴다.

aggregation 방법은 다음을 기본 후보로 둔다.

- `sum`: 단순 합
- `mean`: 단순 평균
- `learned`: learnable scalar 또는 projection 기반 조합
- `router_conditioned`: 입력/feature에 따라 조합 비중을 동적으로 조절

### 세팅 목록

#### (A) Two-stage bundle then one-stage follow-up — 9개

- `P12-10_BUNDLE_MACROMID_SUM`
- `P12-11_BUNDLE_MACROMID_MEAN`
- `P12-12_BUNDLE_MACROMID_LEARNED`
- `P12-13_BUNDLE_MIDMICRO_SUM`
- `P12-14_BUNDLE_MIDMICRO_MEAN`
- `P12-15_BUNDLE_MIDMICRO_LEARNED`
- `P12-16_BUNDLE_MACROMICRO_SUM`
- `P12-17_BUNDLE_MACROMICRO_MEAN`
- `P12-18_BUNDLE_MACROMICRO_LEARNED`

의미:
- 먼저 두 stage를 같이 쓰고,
- 남은 stage를 뒤에 serial하게 붙이는 방식

#### (B) All-stage bundle — 3개

- `P12-19_BUNDLE_ALL_SUM`
- `P12-20_BUNDLE_ALL_MEAN`
- `P12-21_BUNDLE_ALL_LEARNED`

의미:
- macro / mid / micro를 한 번에 동시에 계산해 합치는 방식
- 가장 공격적인 “all horizons at once” 비교

#### (C) Multi-bundle chain / conditioned aggregation — 3개

- `P12-22_BUNDLE_MACROMID_THEN_MIDMICRO_LEARNED`
- `P12-23_BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED`
- `P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED`

의미:
- 한 번의 bundle로 끝내지 않고 bundle을 두 번 이어 붙이는 구조
- 또는 aggregation 자체를 router-conditioned하게 두는 구조

### 이 축에서 보고 싶은 것

- serial이 강하면:
  - horizon separation 자체가 중요하다
- 특정 2-stage bundle이 좋으면:
  - 그 두 horizon은 joint interaction이 유익하다
- all-stage bundle이 약하면:
  - 모든 horizon을 한 번에 섞는 건 오히려 inductive bias를 흐린다
- router-conditioned aggregation이 강하면:
  - 조합 방식도 input-dependent해야 한다는 서사를 만들 수 있다

### 이 축이 잘 나왔을 때 해석

- serial > bundle:
  - “structured serial composition이 core contribution”
- bundle 일부 > serial:
  - “strict separation보다 selected horizon interaction이 더 중요하다”
- all-stage bundle strongest:
  - “현재 serial design보다 horizon interaction이 더 중요한 방향일 수 있다”

---

## 4.3 8배수 보강 (7개)

GPU 8개 병렬 매트릭스(32=8x4)를 맞추기 위한 composition 보강 세팅이다.

### 세팅 목록

- `P12-25_BUNDLE_ALL_ROUTER_CONDITIONED`
- `P12-26_BUNDLE_MACROMID_THEN_MIDMICRO_SUM`
- `P12-27_BUNDLE_MACROMID_THEN_MIDMICRO_MEAN`
- `P12-28_BUNDLE_MACROMID_THEN_MIDMICRO_ROUTER_CONDITIONED`
- `P12-29_BUNDLE_MACROMICRO_THEN_MIDMICRO_SUM`
- `P12-30_BUNDLE_MACROMICRO_THEN_MIDMICRO_MEAN`
- `P12-31_BUNDLE_MACROMICRO_THEN_MIDMICRO_ROUTER_CONDITIONED`

### 보강 의도

- `ALL_ROUTER_CONDITIONED`:
  - all-stage 동시 결합에서 input-conditioned aggregation까지 포함해 상한 확인
- `THEN_*` 체인 6개:
  - 2단 bundle chain에서 aggregation 선택(sum/mean/router)의 민감도 비교

---

## 5. 이 phase에서 꼭 보고 싶은 결과 패턴

### 가장 이상적인 패턴

1. compact한 attention placement 하나가 base보다 비슷하거나 약간 좋음
2. repeated-stage는 일부만 유효
3. serial 또는 limited bundle이 all-stage bundle보다 안정적

이 패턴이면 논문에서는 이렇게 쓸 수 있다.

> **The gain of FeaturedMoE depends not only on stage-wise routing itself, but also on preserving a structured composition across temporal horizons.**

### 차선의 패턴

1. 특정 bundle 조합이 serial보다 좋음
2. all-stage bundle은 여전히 약함

이 경우에는 다음처럼 쓸 수 있다.

> **Selected horizon interaction can be beneficial, but indiscriminately merging all stages is less effective than controlled composition.**

---

## 6. 추천 분석 포인트

### main table

- overall metric
- cold-item metric
- 가능하면 parameter / complexity proxy

### figures

1. **layout family별 bar plot**
2. **serial vs bundle 비교 plot**
3. **aggregation method 비교 plot**

### supplementary

- bundle 시 top1 concentration 변화
- bundle 시 route consistency 변화
- repeated-stage에서 routing entropy 변화

---

## 7. 구현 메모

### 7.1 Attention / layout variants

이 부분은 대부분 `layer_layout` override로 가능하다.
즉 현재 executor 구조를 크게 안 바꿔도 된다.

### 7.2 Bundle / parallel composition

이 부분은 새 코드가 필요하다.

현재 executor는 serial op를 좌→우로 실행하므로,
한 slot에서 여러 stage를 동시에 실행하려면 다음 중 하나가 필요하다.

1. `bundle(macro,mid)` 같은 새 token grammar 추가
2. bundle op에서 여러 stage block을 동시에 호출하는 executor path 추가
3. aggregation mode(`sum / mean / learned / router-conditioned`) 지원

### 구현 시 주의

- stage block parameter는 share하지 않는 것이 기본
- aggregation만 새로 추가
- stage별 gate / aux logging이 유지되는지 확인
- complexity 증가가 너무 크지 않게 관리

---

## 8. 최종적으로 논문에 남기고 싶은 문장

가장 이상적인 결과가 나왔을 때:

> **The effectiveness of FeaturedMoE depends not only on which temporal stages are used, but also on how they are composed; preserving a structured serial composition is often more effective than indiscriminately mixing all stages.**

대안 결과가 나왔을 때:

> **Joint composition across selected horizons can sometimes outperform strict serial decomposition, suggesting that feature-aware routing benefits from controlled horizon interaction rather than purely isolated stages.**
