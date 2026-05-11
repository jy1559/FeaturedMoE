# Phase 10 — Feature Portability / Compactness

## 1. Phase 목표

이 phase의 핵심 질문은 다음이다.

> **FMoE는 많은 handcrafted feature에 의존하는 모델인가, 아니면 적은 수의 흔한 feature만으로도 충분히 작동하는 모델인가?**

즉 이 phase는 단순히 “어떤 feature를 빼면 성능이 떨어지는가”를 보는 것이 아니라,

- 어떤 **feature family 자체**가 중요한지,
- family 개수와 family 내부 feature 수를 줄여도 성능이 유지되는지,
- category / timestamp 같은 흔한 정보가 빠졌을 때도 framework가 유지되는지,
- training 시 일부 feature를 랜덤하게 꺼도 오히려 generalization에 도움이 되는지

를 한 번에 확인하는 phase다.

논문에서 이 phase가 담당하는 메시지는 분명하다.

1. **feature-rich metadata가 없어도 된다**
2. **few, common, portable features만으로도 충분하다**
3. **feature bank가 커서 좋아진 것이 아니라, feature family 구조와 그 조합이 중요했다**

---

## 2. 이 phase에서 주장하고 싶은 것

### main claim

> **A small, portable feature template is already sufficient to make feature-aware MoE work well.**

### 이 phase가 잘 나오면 하고 싶은 말

- full feature가 최고더라도 compact subset이 근접하면:
  - “많은 feature engineering이 아니라 compact family template가 핵심이었다”
- `NO_CATEGORY`가 꽤 버티면:
  - “category가 있으면 좋지만 필수는 아니다”
- `NO_TIMESTAMP`가 꽤 버티면:
  - “tempo/time signal이 중요해도 전체가 그 하나에만 의존하지는 않는다”
- `TOP2_PER_GROUP`, `COMMON_TEMPLATE`가 강하면:
  - “large handcrafted bank보다 small reusable template를 제안한다”
- dropout 계열이 도움이 되면:
  - “feature availability variation에 robust한 학습이 가능하다”

### 기대와 다르게 나와도 쓸 수 있는 말

- full만 압도적으로 좋고 compact가 많이 밀리면:
  - “feature-aware MoE는 의미 있는 feature 신호 자체에 크게 의존하며, compactness에는 trade-off가 있다”
- 특정 family 하나만 매우 중요하면:
  - “모든 family가 equally useful하지 않고, effective inductive bias는 sparse subset에서 온다”
- `NO_CATEGORY`가 크게 무너지면:
  - “category-derived behavioral abstraction이 routing prior의 중요한 축이다”
- `NO_TIMESTAMP`가 크게 무너지면:
  - “tempo / recency / pace 기반 요약이 routing prior의 핵심 축이다”
- dropout이 별로면:
  - “randomly dropping features보다 stable feature structure 자체가 더 중요했다”

---

## 3. 이 phase의 구현 철학

이 phase의 핵심은 **0 padding이 아니라 structural removal**이다.

즉,

- feature를 runtime에서 0으로 masking하는 실험이 main이 아니라,
- 애초에 `build_stage_feature_spec()` 단계에서 feature column 자체를 제거해서,
- encoder input 차원, stage별 feature list, family별 local index가 함께 줄어들도록 구현하는 것이 핵심이다.

이렇게 해야 “그 feature가 값은 0인데 구조는 남아있는 상태”가 아니라,
**처음부터 그 feature를 모르는 모델**에 가까운 비교가 된다.

따라서 이 phase의 main 실험은 전부 **spec-level removal / reduction**으로 가고,
현재 모델에 있는 eval-time `zero` / `shuffle` 계열은 Phase 13 sanity 실험으로 넘긴다.

---

## 4. Phase 구성 요약

총 **24개 세팅(기본)**.

- 4.1 Group subset lattice: **15개**
- 4.2 Intra-group reduction: **3개**
- 4.3 Availability ablation: **2개**
- 4.4 Stochastic feature usage: **2개**
- 4.5 8배수 보강: **2개**

운영 정책:
- 기본 실행은 24개(`--include-extra-24` 기본)
- 필요 시 `--no-extra-24`로 22개 축소 fallback 허용

---

## 4.1 Group subset lattice (15개)

### 목적

이 축은 가장 중요하다.

보고 싶은 것은 다음이다.

- Tempo / Focus / Memory / Exposure 중 어떤 family가 truly necessary한가
- 두 family, 세 family만으로도 full에 가까운가
- 특정 family가 빠질 때만 급격히 무너지나
- feature-aware routing의 inductive bias가 “많은 feature”가 아니라 “어떤 family 조합”에서 오는가

### 구현 방식

`stage_feature_family_mask` 또는 그에 대응하는 profile을 사용해서,
각 stage에서 선택된 family만 남긴다.

중요:
- 끈 family는 해당 family에 속한 column을 **아예 spec에서 제거**
- `0-padding`이나 dummy tensor로 두지 않음
- 결과적으로 feature encoder input과 family grouping 구조가 모두 같이 줄어듦

### 세팅 목록

- `P10-00_FULL`
- `P10-01_TEMPO`
- `P10-02_FOCUS`
- `P10-03_MEMORY`
- `P10-04_EXPOSURE`
- `P10-05_TEMPO_FOCUS`
- `P10-06_TEMPO_MEMORY`
- `P10-07_TEMPO_EXPOSURE`
- `P10-08_FOCUS_MEMORY`
- `P10-09_FOCUS_EXPOSURE`
- `P10-10_MEMORY_EXPOSURE`
- `P10-11_TEMPO_FOCUS_MEMORY`
- `P10-12_TEMPO_FOCUS_EXPOSURE`
- `P10-13_TEMPO_MEMORY_EXPOSURE`
- `P10-14_FOCUS_MEMORY_EXPOSURE`

### 각 family를 직관적으로 어떻게 볼 것인가

- **Tempo**: 시간 간격, pace, recency 같은 시간적 진행 속도 신호
- **Focus**: category/theme 집중도, 전환 정도 같은 주제 응집 신호
- **Memory**: 반복/재방문/novelty 같은 과거 기억 관련 신호
- **Exposure**: popularity, exposure diversity, frequency 같은 노출 편향 신호

### 이 축이 잘 나왔을 때 해석

- 2~3 family만으로 full에 근접하면:
  - “FMoE는 large feature bank가 아니라 compact family subset으로도 충분하다”
- 특정 family 하나만 빠졌을 때만 크게 하락하면:
  - “그 family가 routing hint의 핵심 축”
- 단일 family로도 baseline 대비 이득이 남으면:
  - “각 family가 독립적인 behavioral prior로도 쓸모가 있다”

### 다른 결과가 나왔을 때 해석

- full만 강하고 subset이 모두 약하면:
  - “feature-aware routing의 이득은 여러 behavioral view의 결합에서 온다”
- 특정 단일 family가 가장 강하면:
  - “실질적으로 중요한 건 sparse한 몇 개 신호이며, 그 family를 중심으로 compact variant를 재정의할 수 있다”

---

## 4.2 Intra-group feature reduction (3개)

### 목적

4.1이 “몇 개 family가 필요한가”라면,
4.2는 “family 내부에 feature가 몇 개나 있어야 하는가”를 본다.

즉,
- family는 유지하되,
- 각 family 안의 feature 수를 줄였을 때도 성능이 유지되는지
를 확인한다.

이 축은 논문에서 **compact reusable template**를 주장하기 위한 핵심 보조 증거다.

### 구현 방식

family 자체는 유지하되,
각 family 안의 column list를 profile별로 축소한다.

예를 들면,
- `TOP2_PER_GROUP`: 각 family에서 대표 2개만
- `TOP1_PER_GROUP`: 각 family에서 대표 1개만
- `COMMON_TEMPLATE`: stage 간 의미적으로 공통되는 feature만 남김

중요:
이것도 runtime masking이 아니라 **column selection 자체를 축소**해야 한다.

### 세팅 목록

- `P10-15_TOP2_PER_GROUP`
- `P10-16_TOP1_PER_GROUP`
- `P10-17_COMMON_TEMPLATE`

### 이 축이 잘 나왔을 때 해석

- `TOP2_PER_GROUP`가 좋으면:
  - “few representative signals per family suffice”
- `TOP1_PER_GROUP`도 꽤 버티면:
  - “router hint로는 매우 sparse한 summary도 충분하다”
- `COMMON_TEMPLATE`가 좋으면:
  - “dataset-specific feature bank보다 reusable template가 핵심이다”

### 다른 결과가 나왔을 때 해석

- `TOP1_PER_GROUP`가 크게 약하면:
  - “family 내부에도 최소한의 redundancy가 필요하다”
- `COMMON_TEMPLATE`가 약하면:
  - “stage별 역할 차이를 반영하는 feature 설계가 중요하다”

---

## 4.3 Availability ablation (2개)

### 목적

이 축은 “특정 종류의 흔한 정보가 아예 없는 상황”을 본다.

여기서 중요한 건 category-only나 timestamp-only처럼 너무 extreme한 setting보다,
실제 반박에 바로 대응되는 두 setting을 먼저 보는 것이다.

- category-derived signal이 없으면 얼마나 버티는가
- timestamp-derived signal이 없으면 얼마나 버티는가

### 구현 방식

- `NO_CATEGORY`: category / theme 기반 컬럼 전체 제거
- `NO_TIMESTAMP`: gap / interval / recency / pace 기반 컬럼 전체 제거

이 역시 structural removal로 구현한다.

### 세팅 목록

- `P10-18_NO_CATEGORY`
- `P10-19_NO_TIMESTAMP`

### 이 축이 잘 나왔을 때 해석

- `NO_CATEGORY`가 꽤 버티면:
  - “category가 없어도 framework는 유지된다”
- `NO_TIMESTAMP`가 꽤 버티면:
  - “tempo signal이 중요하더라도 전체가 그것에만 의존하지 않는다”
- 둘 다 full 대비 moderately 유지되면:
  - “heterogeneous feature availability에 robust하다”

### 다른 결과가 나왔을 때 해석

- `NO_CATEGORY`에서 크게 하락:
  - “category-derived behavior abstraction이 매우 중요하다”
- `NO_TIMESTAMP`에서 크게 하락:
  - “tempo / recency / pace 요약이 routing prior의 핵심이다”

---

## 4.4 Stochastic feature usage (2개)

### 목적

이 축은 main thesis라기보다 robustness regularization에 가깝다.

즉,
- 학습 중 일부 family나 feature를 랜덤하게 꺼주면,
- feature availability variation에 더 잘 견디는 모델이 되는지
를 보는 실험이다.

이건 “이 setting이 제일 좋다”를 찾기보다,
**feature-aware routing이 training-time stochastic masking에도 유연한가**를 보는 보조 실험으로 둔다.

### 구현 방식

train-time에서만 적용.

- `FAMILY_DROPOUT`: family 단위로 일정 확률 off
- `FEATURE_DROPOUT`: feature 단위로 일정 확률 off

주의:
이 축은 4.1~4.3과 달리 runtime masking 성격이 있으므로,
본문에서는 “robustness regularization” 정도로 해석하는 것이 자연스럽다.

### 세팅 목록

- `P10-20_FAMILY_DROPOUT`
- `P10-21_FEATURE_DROPOUT`

### 이 축이 잘 나왔을 때 해석

- full보다 비슷하거나 더 좋으면:
  - “feature availability variation에 robust한 학습이 가능하다”
- family_dropout만 좋으면:
  - “coarser regularization이 더 자연스럽다”
- feature_dropout만 좋으면:
  - “fine-grained redundancy가 실제로 작동한다”

### 다른 결과가 나왔을 때 해석

- 둘 다 약하면:
  - “randomly dropping features보다 stable structured feature signal이 더 중요하다”

---

## 4.5 8배수 보강 (2개)

GPU 8개 병렬 실행 매트릭스를 맞추기 위한 보강 세팅이다.

### 세팅 목록

- `P10-22_NO_CATEGORY_NO_TIMESTAMP`
- `P10-23_COMMON_TEMPLATE_NO_CATEGORY`

### 구현 의도

- `NO_CATEGORY_NO_TIMESTAMP`:
  - availability 축의 강한 교차 제거 조건을 추가해 portability 하한 확인
- `COMMON_TEMPLATE_NO_CATEGORY`:
  - compact template 유지 상태에서 category/theme 의존성을 한 번 더 분리 확인

---

## 5. 이 phase에서 꼭 보고 싶은 결과 패턴

### 가장 이상적인 패턴

1. `FULL` 최고
2. 2~3 family subset, `TOP2_PER_GROUP`, `COMMON_TEMPLATE`가 근접
3. `NO_CATEGORY` 또는 `NO_TIMESTAMP` 중 적어도 하나는 꽤 버팀
4. dropout 계열은 적어도 catastrophic drop은 아님

이 패턴이면 논문에서 가장 하고 싶은 말은 다음이다.

> **FeaturedMoE does not require a large handcrafted feature bank; a compact, portable subset of common behavioral features is already sufficient.**

### 차선의 패턴

1. compact는 약간 밀림
2. 특정 family가 핵심적으로 중요
3. no-category / no-timestamp 중 하나는 많이 무너짐

이 경우 논문에서는 이렇게 정리 가능하다.

> **The gain of feature-aware routing comes from a sparse but meaningful subset of behavioral cues, rather than from indiscriminately adding many features.**

---

## 6. 추천 분석 포인트

### main table

- overall ranking metric
- cold-item slice
- 가능하면 parameter / input feature dimension

### figures

1. **family subset heatmap**
   - row: subset setting
   - col: dataset / metric
2. **compactness vs performance plot**
   - x: feature count or active family count
   - y: performance
3. **availability bar plot**
   - full / no_category / no_timestamp

### supplementary

- family importance ranking
- compact template trade-off 표
- cold-item에서 어느 family가 가장 큰 역할을 하는지

---

## 7. 구현 메모

### 추천 config axis

- `feature_profile` 또는 이에 준하는 축 추가
- 예시 value:
  - `full`
  - `tempo_only`, `focus_only`, ...
  - `tempo_focus`, ...
  - `top2_per_group`
  - `top1_per_group`
  - `common_template`
  - `no_category`
  - `no_timestamp`

### 구현 위치

가장 자연스러운 곳은 `feature_config.py`.

- family mask 적용
- family 내부 column selection 적용
- availability profile 적용

이렇게 해서 `build_stage_feature_spec()` 출력 자체가 달라지게 만든다.

### 주의

- 이 phase의 main 실험은 zero masking 금지
- eval-time zero/shuffle은 Phase 13으로 분리
- dropout 계열은 train-time only regularization으로 처리

---

## 8. 최종적으로 논문에 남기고 싶은 문장

가장 이상적인 결과가 나왔을 때:

> **FeaturedMoE does not rely on large handcrafted feature banks; a compact, portable set of common behavioral features already provides a strong routing prior.**

다른 결과가 나왔을 때:

> **Feature-aware routing is most effective when built on a sparse but carefully chosen subset of behavioral signals, revealing that not all feature families contribute equally to expert specialization.**
