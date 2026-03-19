# FMoE Router Specialization: Logging 추가 항목 + Aux/Reg 실험 계획 (간소화 버전)

## 목적
현재 관찰은 다음을 시사한다.

- balanced routing이 목표가 아님
- 어느 정도 specialization/imbalance는 오히려 성능에 도움될 수 있음
- 하지만 한 expert 독식(monopoly)은 또 좋지 않을 수 있음
- session 내 routing jitter는 해로울 가능성이 큼

따라서 다음 단계는
**(1) specialization이 실제로 어떻게 생기는지 더 잘 보이도록 logging을 추가하고,**
**(2) load balancing이 아니라 stable semantic specialization을 유도하는 aux/reg를 실험하는 것**이다.

---

## 1. Logging: 무엇을 추가/수정할지

아래는 "어떤 값을 넣을지"와 "왜 필요한지"만 정리한다.
저장 구조나 빈도는 기존 코드베이스 방식에 맞춰 적용하면 된다.

### 1) Expert usage / top-1 assignment 관련
추가할 것:
- expert별 usage mass
- expert별 top-1 assignment fraction
- usage CV (imbalance 정도)
- top-1 max fraction
- effective number of experts (`n_eff`)

왜 필요한가:
- 지금 결과상 "균등하게 쓰는 것"이 목표가 아닌 것으로 보임
- usage CV는 specialization 정도를, top-1 max fraction은 monopoly 위험을, `n_eff`는 실질적으로 몇 개 expert가 살아 있는지를 보여줌
- 특히 usage mass와 top-1 assignment를 같이 봐야 dense routing에서 실제 dominant expert가 누구인지 보임

---

### 2) Routing entropy
추가할 것:
- route entropy 평균/분산/중간값 정도
- stage별 entropy

왜 필요한가:
- router가 너무 퍼져서 expert averaging만 하는지,
  아니면 지나치게 뾰족하게 한 expert만 고르는지를 보려는 것
- 현재 원하는 상태는 "무조건 sharp"도 아니고 "무조건 balanced"도 아니므로 entropy를 함께 봐야 함

---

### 3) Session jitter / route smoothness
추가할 것:
- session 내 인접 token 간 routing 변화량
- stage별 jitter (macro / mid / micro)
- 필요하면 short/long session별 jitter

왜 필요한가:
- 지금까지 나온 진단 중 가장 중요한 신호
- specialization이 있어도 세션 안에서 route가 계속 흔들리면 noisy routing일 가능성이 큼
- 특히 macro/mid는 비교적 안정적이어야 하고, micro만 좀 더 가변적일 수 있음

---

### 4) Feature-route consistency
추가할 것:
- 비슷한 feature state를 가진 샘플들끼리 route 분포가 얼마나 비슷한지 보는 consistency 지표
- 간단히는 feature embedding 기준 kNN을 잡아서 route JS divergence 평균을 계산

왜 필요한가:
- 지금 네가 원하는 specialization은 "expert를 무조건 다르게 쓰는 것"이 아니라
  "비슷한 feature regime는 비슷한 expert로, 다른 regime는 다른 expert로" 가는 것
- 이걸 직접 보는 핵심 지표

---

### 5) Feature bucket -> expert 경향
추가할 것:
- 일부 대표 feature에 대해 bucket별 expert 사용 경향
- heatmap 만들 수 있을 정도의 집계값만 저장

권장 feature 예시:
- macro: session gap, popularity avg, category entropy, new user 여부
- mid: repeat ratio, switch ratio or entropy drift, session progress
- micro: interval std, short-term switch rate, window validity

왜 필요한가:
- "어떤 feature regime가 어떤 expert로 가는가"를 가장 직관적으로 보여줌
- monotonic/해석 가능한 specialization이 있는지 확인 가능

---

### 6) Expert purity / expert별 담당 regime 요약
추가할 것:
- expert별로 주로 맡는 샘플의 feature 평균/비율 요약
- 예: repeat ratio 평균, switch ratio 평균, popularity 평균, new user 비율, short session 비율 등

왜 필요한가:
- expert가 실제로 의미 있는 역할을 나눠 갖는지 해석하기 쉬워짐
- 예를 들어 "이 expert는 repeat-heavy 세션에서 주로 켜진다" 같은 설명이 가능해짐

---

### 7) Cold / sparse / session slice 성능 + 일부 라우팅 지표
이미 special logging을 일부 하고 있다면, 아래 축만 선택적으로 추가하면 충분함.

추천 slice:
- new user vs old user
- cold item vs warm item
- short session vs long session
- high repeat vs high switch
- low validity micro vs high validity micro

이 slice들에 대해 같이 보면 좋은 것:
- MRR/HR
- session jitter
- usage CV
- top-1 max fraction

왜 필요한가:
- 이 모델은 전체 평균보다 특정 regime에서 더 강할 수 있음
- 특히 cold/sparse/short session에서 specialization이 실제 도움이 되는지 보기 좋음

---

## 2. Logging 추가의 핵심 메시지

추가 로깅은 결국 아래 질문에 답하기 위한 것이다.

1. specialization이 실제로 생기고 있는가?
2. 그 specialization이 stable한가?
3. semantic한가? (비슷한 feature면 비슷한 expert로 가는가)
4. monopoly만 심해진 것은 아닌가?
5. cold/sparse regime에서 실제로 도움이 되는가?

즉, 이제는 balanced routing 여부보다
**stable semantic specialization**을 보여주는 로그가 더 중요하다.

---

## 3. Aux/Reg 방법 5개

여기서는 baseline 포함 5개 방법을 비교한다.

### Method 0. Baseline
정의:
- no aux/reg
  또는
- 현재까지 best 결과에서 사용하던 기존 aux/reg 유지

실험에서는 둘 중 하나를 baseline으로 고정.

의미:
- 새로운 regularization들이 실제로 더 낫는지 비교하는 기준점

---

### Method 1. Smoothness regularization
목적:
- session jitter 감소
- 특히 macro/mid routing을 덜 흔들리게 하기

핵심 아이디어:
- 세션 내에서 인접 token들의 route가 너무 달라지지 않도록 penalty
- macro > mid >> micro 강도로 두는 것이 자연스러움

왜 중요한가:
- 지금 관찰과 가장 직접적으로 연결됨
- 구현도 쉽고 해석도 쉬움

주의:
- micro에 너무 강하게 걸면 genuine shift까지 막을 수 있음

---

### Method 2. Feature-consistency regularization
목적:
- 비슷한 feature regime이면 비슷한 route를 타게 하기

핵심 아이디어:
- feature space에서 가까운 샘플쌍에 대해 route 분포도 가깝게 유지
- macro/mid에 더 적합

왜 중요한가:
- specialization을 "semantic alignment" 관점에서 직접 강화
- load balancing과 달리 균등화는 강제하지 않음

주의:
- feature 자체가 noisy하거나 잘못 설계되었으면 consistency를 잘못 강제할 수 있음

---

### Method 3. Sharp-but-not-monopoly
목적:
- sample-wise로는 sharper routing을 유도하되,
- global하게 한 expert 독점은 방지

핵심 아이디어:
- route entropy를 약간 낮춰 specialization을 장려
- usage/top-1이 특정 threshold를 지나치게 넘으면 monopoly penalty 추가

왜 중요한가:
- 현재 진단과 가장 잘 맞는 방향
- "balanced routing" 대신
  "specialized but not monopolized routing"을 유도할 수 있음

주의:
- sharpness만 강하면 collapse 위험
- monopoly penalty가 너무 약하면 결국 한 expert 독식 가능

---

### Method 4. Soft prior / family prior
목적:
- feature group/family semantic을 약하게 보존
- hand-crafted prior와 learnable routing을 절충

핵심 아이디어:
- 특정 feature family/regime에서 선호할 expert prior를 soft하게 주고,
  router가 그 prior를 참고하되 완전히 묶이진 않게 함

예시 intuition:
- repeat-heavy -> repeater 성격 expert prior
- switch-heavy -> local/hopper 성격 expert prior
- new user/high session gap -> retention/rhythm 성격 expert prior

왜 중요한가:
- "비슷한 부류의 feature가 비슷한 expert로 가게" 하려는 목적과 가장 직접적으로 연결됨
- semantic 해석이 쉬움

주의:
- prior 설계가 틀리면 bias가 생길 수 있음

---

## 4. 8개 combo 설계

aux/reg 외 나머지 설정을 8개 조합으로 고정한다.
축은 3개로 두는 것을 추천한다.

### 축 1. Temperature
- base temperature
- low temperature

이유:
- routing sharpness와 직접 상호작용
- 어떤 aux/reg가 sharp/soft routing 조건에서 잘 듣는지 보기 좋음

### 축 2. Expert count
- base experts
- fewer experts

이유:
- specialization granularity가 달라짐
- experts가 적을 때와 많을 때 regularization이 다르게 작동할 수 있음

### 축 3. Shared fallback
- off
- on

이유:
- fallback/shared path가 monopoly와 instability를 얼마나 완충하는지 확인 가능
- 특히 routing이 흔들릴 때 안정성을 줄 수 있음

즉 2 x 2 x 2 = 8 combo.

---

## 5. 실험 구조: 5 x 8

총 40 runs:
- Method 0. Baseline
- Method 1. Smoothness
- Method 2. Consistency
- Method 3. Sharp+Monopoly
- Method 4. Soft Prior

각각에 대해 8개 combo 적용.

이 방식의 장점:
- aux/reg 효과와 backbone/router 설정 효과를 분리해서 볼 수 있음
- 어떤 regularization이 어떤 routing condition에서 잘 듣는지 비교 가능

---

## 6. 각 method의 튜닝 원칙

조합 폭발을 막기 위해 각 method는 1~2개 정도의 핵심 변수만 연다.

### Baseline
- 튜닝 없음

### Smoothness
- overall strength 1개
- stage별 비율은 고정 (예: macro 크게, mid 중간, micro 0 또는 작게)

### Consistency
- consistency strength 1개
- kNN 개수나 pair 구성 방식은 고정

### Sharp+Monopoly
- sharpness strength 1개
- monopoly strength는 sharpness와 연동시키거나
  threshold는 expert 수에 따라 rule-based로 고정

### Soft Prior
- prior strength 1개
- prior mapping 자체는 고정

즉, method별로 너무 많은 hyperparameter를 열지 않는 것이 중요하다.

---

## 7. 실제 실행 순서 추천

### Step 1. Baseline + logging sanity check
먼저 baseline 설정 하나에서 로그가 제대로 나오는지 확인.
이 단계에서는
- usage/top1/jitter/consistency가 정상 계산되는지
- heatmap용 집계값이 해석 가능하게 나오는지
만 보면 됨.

### Step 2. Baseline combo 먼저 확인
8개 combo에 대해 baseline만 먼저 돌려서
- 원래 routing이 어떤 성향인지
- 어떤 combo가 원래 jitter가 큰지
- 어떤 combo가 monopoly에 취약한지
를 먼저 본다.

### Step 3. Full 5 x 8
그 다음 40 runs 실행.

### Step 4. 상위 후보 재확인
상위 조합 몇 개만 seed를 바꿔 재실험.

---

## 8. 결과를 어떻게 읽을까

단순히 MRR 최고만 고르지 말고 아래를 같이 본다.

1. MRR/HR이 올랐는가?
2. session jitter가 줄었는가?
3. usage CV는 적당히 유지되는가? (specialization 유지)
4. top-1 max fraction이 너무 높아지진 않았는가? (anti-monopoly)
5. feature-route consistency가 좋아졌는가?
6. cold/sparse/session slice에서 실제 이득이 있는가?

즉 최종적으로는 아래 형태가 가장 좋다.

- balanced는 아님
- monopoly도 아님
- semantically aligned specialization
- 낮은 jitter
- cold/sparse regime에서 도움

---

## 9. 이후 heterogeneous expert는 어떻게 볼까

현재 단계의 메인은 아님.
우선은 logging + aux/reg 실험이 먼저.

그 다음,
가장 좋은 method+combo 하나를 고른 뒤에만 hetero expert branch로 가는 것을 추천.

이유:
- 지금 바로 heterogeneous expert를 넣으면
  개선 원인이 router stabilization인지 expert 구조 차이인지 분해가 어려워짐
- 먼저 specialization을 잘 만드는 routing 조건을 찾은 뒤,
  그 위에 heterogeneous expert를 얹는 게 훨씬 해석 가능함

추천 방향:
- 먼저 homogeneous expert에서 best routing condition 확보
- 그 다음 macro-only heterogeneous 같은 제한된 branch부터 테스트

---

## 10. 최종 요약

지금 단계에서 가장 중요한 것은 아래 두 가지다.

### A. Logging 추가/수정
추가해야 할 핵심:
- expert usage / top-1 / n_eff / monopoly
- route entropy
- session jitter
- feature-route consistency
- feature bucket -> expert 경향
- expert purity summary
- cold/user/session slice에서 성능 + 핵심 specialization 지표

### B. Aux/Reg 실험
우선 비교할 5개 방법:
1. baseline
2. smoothness
3. consistency
4. sharp-but-not-monopoly
5. soft prior

그리고 이걸 8개 combo에 대해 비교.

핵심 목표는
**load balancing**이 아니라
**stable semantic specialization**이다.
