# Phase 11 — Stage Semantics / Stage Necessity / Routing Granularity

## 1. Phase 목표

이 phase의 핵심 질문은 다음이다.

> **macro / mid / micro라는 stage 분해가 정말 의미 있는가? 그리고 각 stage는 어떤 granularity에서 routing되어야 하는가?**

즉 이 phase는 단순한 구조 ablation이 아니라,

- stage별 horizon이 실제로 필요한지,
- dense contextualization layer를 앞에 하나 넣는 것이 stage 효과를 대체하는지,
- stage 순서가 중요한지,
- macro와 mid는 session-level routing이 맞는지 token-level routing이 맞는지

를 검증하는 phase다.

논문에서는 이 phase를 통해 아래 서사를 만들고 싶다.

1. **temporal horizon별 decomposition은 임의적이지 않다**
2. **각 stage는 서로 다른 behavioral regime를 담당한다**
3. **macro / mid는 session summary를 보는 stage이고, micro는 token-local variation을 보는 stage일 수 있다**

---

## 2. 이 phase에서 주장하고 싶은 것

### main claim

> **FMoE works not just because it uses MoE, but because it decomposes routing across meaningful temporal stages.**

### additional claim

> **For macro and mid stages, session-level routing may be more natural than token-level routing because their feature signals are session-consistent summaries rather than token-varying cues.**

### 이 phase가 잘 나오면 하고 싶은 말

- `[macro, mid, micro]`가 가장 안정적이면:
  - “3-stage decomposition 자체가 타당하다”
- 일부 stage 제거 시 성능 하락이 명확하면:
  - “각 horizon이 다른 역할을 가진다”
- `[layer, macro, mid, micro]`가 base를 크게 못 이기면:
  - “그냥 dense layer를 더 쌓아서 좋아진 것이 아니다”
- 기본 granularity `[session, session, token]`가 strongest이면:
  - “macro/mid는 세션-level regime routing이 자연스럽다”

### 기대와 다르게 나와도 쓸 수 있는 말

- 특정 single-stage가 강하면:
  - “가장 유효한 horizon이 어디인지 드러났다”
- 순서에 민감하지 않으면:
  - “stage set 자체는 중요하지만 strict ordering은 덜 중요하다”
- macro/mid token routing이 더 좋으면:
  - “session summary보다 token-conditioned hidden variation이 더 유용할 수 있다”
- prepend layer가 consistently 좋으면:
  - “light contextualization before stage routing helps form cleaner routing inputs”

---

## 3. 이 phase의 경계

이 phase는 **stage semantics**를 보는 phase다.

따라서 여기서 답하려는 질문은 전부 다음 축 안에 있어야 한다.

- stage가 몇 개 필요한가?
- 어떤 stage가 중요한가?
- stage 순서가 중요한가?
- 각 stage는 session 단위로 routing해야 하나, token 단위로 routing해야 하나?

반대로,
- attention을 어디에 둘지,
- stage를 반복할지,
- 여러 stage를 한 번에 parallel하게 합칠지
같은 질문은 **layout / composition** 성격이므로 Phase 12로 넘긴다.

---

## 4. Phase 구성 요약

총 **24개 세팅**.

- 4.1 Base stage ablation: **7개**
- 4.2 Prepend dense layer: **7개**
- 4.3 Stage order permutation: **5개**
- 4.4 Routing granularity: **3개**
- 4.5 8배수 보강: **2개**

---

## 4.1 Base stage ablation (7개)

### 목적

가장 기본적인 질문이다.

- macro, mid, micro 각각이 실제로 필요한가?
- 3-stage가 정말 필요한가, 아니면 특정 stage만 있으면 되는가?

이 축은 논문에서 “왜 3-stage로 갔는가”를 직접적으로 방어하는 실험이다.

### 세팅 목록

- `P11-00_MACRO_MID_MICRO`
- `P11-01_MID_MICRO`
- `P11-02_MACRO_MICRO`
- `P11-03_MACRO_MID`
- `P11-04_MACRO_ONLY`
- `P11-05_MID_ONLY`
- `P11-06_MICRO_ONLY`

### 각 세팅이 의미하는 것

- `MACRO_MID_MICRO`: 기본 full stage decomposition
- `MID_MICRO`: 장기/이전 세션 관점을 제거
- `MACRO_MICRO`: 현재 세션 요약 stage 없이 coarse + local만 사용
- `MACRO_MID`: local token-level refinement 없이 session-summary 중심
- `*_ONLY`: 특정 horizon 하나만으로도 충분한지 확인

### 이 축이 잘 나왔을 때 해석

- full 3-stage strongest:
  - “coarse-to-fine horizon decomposition이 의미 있다”
- 2-stage가 full과 비슷:
  - “완전한 3단 구조는 아니어도 특정 두 horizon의 조합이 핵심이다”
- single-stage도 surprisingly strong:
  - “가장 유효한 horizon이 특정 stage에 집중되어 있다”

---

## 4.2 Prepend dense layer (7개)

### 목적

이 축은 반박 대응용으로 중요하다.

리뷰어 입장에서는 이렇게 볼 수 있다.

> “그냥 앞에 layer 하나 더 둬서 hidden을 정제하면 좋아지는 것 아닌가?”

그래서 같은 stage ablation을 유지하되,
**앞에 일반 SASRec-style dense layer 하나를 추가**해서 비교한다.

핵심 질문:
- stage 효과가 정말 stage decomposition 때문인가?
- 아니면 just extra contextualization depth 때문인가?

### 세팅 목록

- `P11-07_LAYER_MACRO_MID_MICRO`
- `P11-08_LAYER_MID_MICRO`
- `P11-09_LAYER_MACRO_MICRO`
- `P11-10_LAYER_MACRO_MID`
- `P11-11_LAYER_MACRO`
- `P11-12_LAYER_MID`
- `P11-13_LAYER_MICRO`

### 어떻게 바뀌는가

기존 setting 앞에 `layer` 하나만 prepend한다.
즉,
- `[macro, mid, micro]` → `[layer, macro, mid, micro]`
- `[mid, micro]` → `[layer, mid, micro]`
같은 형태다.

### 이 축이 잘 나왔을 때 해석

- prepend layer가 base를 크게 못 이기면:
  - “단순히 hidden을 더 정제해서 생긴 이득이 아니다”
- prepend layer가 consistently 좋으면:
  - “routing 전에 한 번 더 contextualization하는 것이 stage routing 표현을 안정화한다”

### 다른 결과가 나왔을 때 해석

- prepend layer가 특정 stage 조합에서만 좋으면:
  - “그 조합은 routing 전에 extra context formation이 특히 중요했다”

---

## 4.3 Stage order permutation (5개)

### 목적

이 축은 `macro -> mid -> micro`라는 순서가 정말 의미 있는지 보는 실험이다.

즉,
- coarse-to-fine이 중요한가?
- 아니면 stage set만 있으면 순서는 크게 중요하지 않은가?

이 실험은 “왜 그 순서로 쌓았는가?”라는 질문에 직접 답한다.

### 세팅 목록

- `P11-14_MACRO_MICRO_MID`
- `P11-15_MID_MACRO_MICRO`
- `P11-16_MID_MICRO_MACRO`
- `P11-17_MICRO_MACRO_MID`
- `P11-18_MICRO_MID_MACRO`

### 각 세팅이 의미하는 것

- `MACRO_MICRO_MID`: coarse 후 바로 local, 그 다음 current summary
- `MID_MACRO_MICRO`: 현재 세션 요약을 먼저 보고 장기 컨텍스트를 뒤에 반영
- `MID_MICRO_MACRO`: 현재/로컬부터 보고 macro를 나중에 반영
- `MICRO_*`: 가장 local한 routing을 먼저 수행

### 이 축이 잘 나왔을 때 해석

- 기본 순서가 strongest:
  - “macro → mid → micro의 coarse-to-fine decomposition이 자연스럽다”
- 순서 차이가 작음:
  - “stage set 자체는 중요하지만 strict order는 덜 중요하다”
- 특정 alternative order가 더 좋음:
  - “현재 stage 역할 해석을 조금 수정해야 한다”

---

## 4.4 Routing granularity (3개)

### 목적

이 축은 이번 phase에서 매우 중요하다.

현재 기본 설정은 다음과 같다.

- macro: `session`
- mid: `session`
- micro: `token`

이렇게 둔 이유는,
macro와 mid의 feature가 session 내부에서는 거의 동일하기 때문이다.

- macro: 과거 세션 요약 기반 feature
- mid: 현재 세션 전체 요약 기반 feature
- micro: token-local feature / local transition 기반

즉,
macro와 mid는 **세션 전체의 regime를 나타내는 feature**에 가깝고,
micro는 **token별 local variation**을 보는 쪽에 가깝다.

그래서 여기서는
- macro/mid를 session-level routing해야 하는가
- 아니면 token-level hidden variation을 활용하는 것이 더 좋은가
를 본다.

### 세팅 목록

기본값은 비교 기준으로 두고, 추가 3개만 실험한다.

- 기본: `[session, session, token]`
- `P11-19_TOKEN_TOKEN_TOKEN`
- `P11-20_SESSION_TOKEN_TOKEN`
- `P11-21_TOKEN_SESSION_TOKEN`

### 각 세팅이 의미하는 것

- `TOKEN_TOKEN_TOKEN`
  - macro와 mid도 token별 hidden으로 gate를 계산
  - 같은 session 안에서도 token마다 다른 expert 사용 가능
- `SESSION_TOKEN_TOKEN`
  - macro는 세션-level, mid는 token-level
  - 장기 요약은 세션 단위, 현재 세션은 token 변화 반영
- `TOKEN_SESSION_TOKEN`
  - macro는 token-level, mid는 세션-level
  - 장기 컨텍스트를 token별 hidden으로 세분화해 쓰는 실험

### 이 축이 중요한 이유

이 실험은 단순한 hyperparam tweak가 아니라,
**“macro/mid stage가 무엇을 보는 stage인가”**에 대한 해석 실험이다.

- session-level이 강하면:
  - “macro/mid는 session regime routing stage”
- token-level이 강하면:
  - “summary feature 위에 hidden variation을 추가로 보는 것이 더 expressive”
- mixed가 강하면:
  - “macro와 mid의 성격이 서로 다르다”

### 이 축이 잘 나왔을 때 해석

- `[session, session, token]` strongest:
  - “macro/mid는 session summary로 route하는 것이 자연스럽다”
- token routing이 macro/mid에서도 강함:
  - “session-consistent feature 위에 token-wise hidden variation을 결합하는 것이 더 유리하다”
- mixed one-stage만 강함:
  - “macro와 mid의 기능적 역할이 완전히 같지 않다”

---

## 4.5 8배수 보강 (2개)

GPU 8개 병렬 배치를 맞추기 위한 실행 보강 세팅이다.

### 세팅 목록

- `P11-22_LAYER_ONLY_BASELINE`
- `P11-23_LAYER2_MACRO_MID_MICRO`

### 보강 의도

- `LAYER_ONLY_BASELINE`:
  - stage 분해 없이 dense contextualization만 둔 순수 baseline으로 하한점 확인
- `LAYER2_MACRO_MID_MICRO`:
  - stage 앞단 dense depth를 한 단계 더 늘렸을 때의 대조군 확보

---

## 5. 이 phase에서 꼭 보고 싶은 결과 패턴

### 가장 이상적인 패턴

1. `[macro, mid, micro]` strongest or most stable
2. single-stage / 일부 제거 setting은 의미 있게 약함
3. prepend dense layer는 base를 크게 못 이김
4. 기본 granularity `[session, session, token]`가 가장 자연스러운 결과를 줌

이 패턴이면 논문에서 다음을 강하게 주장할 수 있다.

> **The gain of FeaturedMoE comes from decomposing routing across meaningful temporal stages, with session-level routing for macro/mid and token-level routing for micro forming the most natural configuration.**

### 차선의 패턴

1. 특정 2-stage 조합이 가장 강함
2. prepend layer가 도움이 됨
3. macro/mid의 token routing이 유리함

이 경우에도 논문은 충분히 쓸 수 있다.
다만 메시지가
“3-stage full이 정답”에서
“어떤 temporal regimes가 실제로 중요한가를 밝혀냈다”
로 바뀐다.

---

## 6. 추천 분석 포인트

### main table

- overall ranking metric
- cold-item slice
- 가능하면 stage별 routing stats 요약

### figures

1. **stage ablation bar plot**
2. **prepend layer vs no prepend 비교 plot**
3. **order sensitivity plot**
4. **granularity comparison plot**

### supplementary

- macro/mid/micro별 routing entropy, top1 concentration
- granularity에 따른 route consistency 변화
- order permutation에 따른 stage별 usage 변화

---

## 7. 구현 메모

### 대부분 config-level로 가능

이 phase는 현재 executor 기준으로 거의 config-level 실험이다.

- `layer_layout` 변경
- `stage_router_granularity` 변경

위 두 축만 잘 override하면 대부분 된다.

### 주의

- 이 phase에서는 attention placement / repeated-stage layout / bundle composition까지 섞지 않는다
- 그건 Phase 12에서 다룬다
- 즉 이 phase는 철저히 **stage 의미**를 보는 phase로 유지하는 것이 좋다

---

## 8. 최종적으로 논문에 남기고 싶은 문장

가장 이상적인 결과가 나왔을 때:

> **FeaturedMoE benefits from decomposing routing across temporal stages, and the most natural configuration routes macro and mid at the session level while reserving token-level routing for the micro stage.**

대안 결과가 나왔을 때:

> **The stage ablations reveal that not all temporal horizons contribute equally, and that the best routing granularity depends on whether a stage primarily encodes session-level regime summaries or token-level variation.**
