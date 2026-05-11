# Cross-Dataset Transfer Learning Plan for FeaturedMoE_N3

작성일: 2026-04-12  
대상: `FeaturedMoE_N3` on cross-dataset transfer settings  
기본 아키텍처: `A10_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W4`  
범위: **문서 기준 설계 우선, 이후 구현/실행으로 바로 이어질 수 있는 계획**

---

## 0) Summary

- 이번 문서의 목적은 `FeaturedMoE_N3`에서 cross-dataset transfer를 “막연한 pretrain-finetune”이 아니라, **router 단위의 선택적 전이 실험**으로 정리하는 것이다.
- 이번 라운드의 핵심 가설은 다음과 같다.

> **transferable한 것은 expert FFN 자체가 아니라, coarse behavioral family 수준의 group router prior다.**

- 더 정확히는 다음을 검증한다.
1. `macro` stage의 `group router`인 `router_b`, 즉 `p(g|z)`는 dataset을 넘어 재사용 가능할 수 있다.
2. `within-group router`인 `router_d`, 즉 `p(c|g,z_g)`는 더 dataset-specific할 수 있다.
3. 따라서 가장 예쁜 결론은 다음 형태다.

> **coarse behavioral family 수준은 공유되지만, finer expert allocation은 target adaptation이 필요하다.**

- 이번 문서의 기본 설계는 다음을 고정한다.
  - source architecture 우선순위: `A10`, checkpoint 부족 시 shape-compatible `A8` fallback 허용
  - main story: `macro` stage 중심
  - main transfer object: `router_b`
  - expert transfer: v1 범위에서 제외
  - wrapper family: `w4_bxd`를 중심으로 해석

---

## 1) 왜 지금 transfer를 하는가

현재 `fmoe_n3` 실험은 다음 정도까지는 이미 말했다.

- compact behavioral feature만으로도 MoE routing이 의미 있게 작동할 수 있다
- stage 구성과 wrapper 구조가 성능/해석 가능성에 모두 영향을 준다
- `A8/A10/A11/A12` 계열 비교는 단순 cosmetic 차이가 아니라 routing factorization 차이다

그 다음 단계에서 가장 자연스러운 질문은 이것이다.

> **이 routing prior는 dataset마다 완전히 새로 배워야 하는가, 아니면 일부는 재사용 가능한가?**

이 질문은 논문적으로도 가치가 있다.

- 단순 성능 비교가 아니라 **feature-aware MoE의 transferability**를 말할 수 있다.
- 추천 도메인이 달라도, 행동 패턴 수준의 family prior는 공유될 수 있다는 메시지를 줄 수 있다.
- “많은 external metadata가 없어도 portable한 behavioral prior를 만들 수 있다”는 주장으로 확장할 수 있다.

즉 이번 transfer 실험은 성능을 조금 더 올리는 데만 목적이 있는 것이 아니라,

1. FMoE router가 **무엇을 배우는지**
2. 그 중 무엇이 **재사용 가능한 inductive bias인지**
3. 무엇이 **target-specific adaptation을 요구하는지**

를 분리해서 보여주는 데 목적이 있다.

---

## 2) 왜 `A10`을 기본 transfer backbone으로 쓰는가

이번 문서에서 transfer backbone은 `A10_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W4`를 기본으로 둔다.

이유는 다음과 같다.

1. `A10`은 `w4_bxd`를 macro/mid/micro 전체에 일관되게 적용한다.
2. `w4_bxd`는 구조적으로 `group router`와 `within-group router`를 가장 직접적으로 해석할 수 있다.
3. 즉 `router_b`와 `router_d`를 분리해서 load/init/freeze/anchor 하기에 가장 적합하다.
4. `no_bias` 조건이라 rule/group bias 부가항 없이도 전이 효과를 해석하기 쉽다.

반면 `A8`은 강한 baseline이지만 stage별 wrapper가 섞여 있어 transfer unit을 문서상으로 설명할 때 `A10`보다 덜 깔끔하다.  
그래서 source architecture 우선순위는 다음처럼 둔다.

- 1순위: `A10`
- 2순위: 동일 shape의 `A8`

`A8` fallback은 checkpoint availability를 위한 운영적 장치이지, main story 자체를 바꾸는 것은 아니다.

---

## 3) 모델 분해와 transfer 단위

이번 실험에서 `FeaturedMoE_N3`는 다음 단위로 나누어 본다.

### 3.1 feature encoder

- 경로 예시:
  - `stage_executor.stage_blocks.<stage>.feature_encoder.*`
- 역할:
  - raw behavioral feature를 router 입력 공간으로 바꾼다.
- transfer 해석:
  - 이 부분만 옮겼을 때 좋아지면 router prior보다 **feature normalization/representation alignment** 효과일 수 있다.

### 3.2 group router

- 경로 예시:
  - `stage_executor.stage_blocks.<stage>.router_b.*`
- 의미:
  - `p(g|z)`에 해당하는 coarse group selection
- 해석:
  - 이번 문서에서 가장 핵심적으로 transferable하다고 가정하는 대상이다.
  - behavioral family 수준 prior가 dataset을 넘어 공유되는지 직접 시험할 수 있다.

### 3.3 within-group router

- 경로 예시:
  - `stage_executor.stage_blocks.<stage>.router_d.*`
- 의미:
  - `p(c|g,z_g)`에 해당하는 group 내부 expert allocation
- 해석:
  - group-level prior보다 더 dataset-specific할 가능성이 높다.
  - `Full-router-init <= Group-init`이면 이 가설이 강하게 지지된다.

### 3.4 experts

- 경로 예시:
  - `stage_executor.stage_blocks.<stage>.experts.*`
- 의미:
  - 실제 expert FFN
- 이번 범위에서 제외하는 이유:
1. expert transfer까지 넣으면 “router prior transfer”와 “expert representation reuse”가 섞여 해석이 흐려진다.
2. source-target 간 item/domain 차이가 큰 경우 expert는 dataset-specific 표현을 더 많이 가질 가능성이 높다.
3. 논문 1차 메시지는 **router transferability**만으로도 충분히 강하다.

정리하면, 이번 문서의 중심 질문은 이것이다.

> **feature encoder / group router / within-group router 중 무엇이 실제로 transferable한가?**

---

## 4) 실험 원칙

### 4.1 main result는 `macro`부터 본다

- 첫 결과는 반드시 `macro`만 transfer한다.
- 이유:
  - `macro`는 coarse preference/prior session 수준 신호를 다루므로 transfer story가 가장 자연스럽다.
  - `mid`와 `micro`는 target-specific intent/transition에 더 민감해 negative transfer가 날 가능성이 높다.

### 4.2 dataset pair 수는 적게, 해석은 강하게

- 처음부터 모든 조합을 돌리지 않는다.
- 2~4개 pair에서 해석이 강한 결과를 먼저 만든다.
- 이후 positive signal이 확인되면 확장한다.

### 4.3 shape compatibility는 엄격히 관리한다

기본 허용 규칙:

- `embedding_size` 동일
- `d_router_hidden` 동일

권장 추가 일치:

- `d_feat_emb`
- `expert_scale`
- stage wrapper 구조

운영 원칙:

- source-target 간 shape mismatch가 나면 그 checkpoint는 제외한다.
- checkpoint availability 때문에 story를 바꾸지 않는다.
- 가능하면 동일 hparam shape를 우선 사용한다.

### 4.4 동일 hparam shape 우선

source checkpoint 선택 우선순위:

1. target과 `embedding_size`, `d_router_hidden`, `d_feat_emb`가 모두 같은 source
2. 최소한 `embedding_size`, `d_router_hidden`이 같은 source
3. 없으면 해당 pair를 보류하거나 fallback pair로 대체

즉, “가장 높은 성능 checkpoint”보다 “shape-compatible하고 해석 가능한 checkpoint”를 우선한다.

### 4.5 experts는 항상 target에서 새로 학습

- `Native`, `FeatureEncoder-init`, `Group-init`, `Full-router-init` 모두 experts는 fresh initialization으로 통일한다.
- 이렇게 해야 transfer 효과를 router 쪽에 귀속시킬 수 있다.

---

## 5) Stage 1 — What Is Transferable?

### 5.1 목적

Stage 1의 목적은 가장 단순한 비교로 “무엇이 transferable한지”를 분리하는 것이다.

즉,

- feature encoder만 옮겨도 충분한가
- group router가 핵심인가
- full router까지 옮기면 오히려 나빠지는가

를 최소 비용으로 본다.

### 5.2 고정 조건

- architecture: `A10_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W4`
- transfer 범위: `macro`만
- experts: 항상 target에서 새로 학습
- seed: `3`
- source checkpoint 수: pair당 `1`

### 5.3 Stage 1 비교군

#### `Native`

- target dataset에서 모든 파라미터를 원래대로 처음부터 학습
- transfer 없음

#### `FeatureEncoder-init`

- `macro.feature_encoder`만 source checkpoint로 초기화
- router와 experts는 target에서 새로 학습

#### `Group-init`

- `macro.router_b`만 source checkpoint로 초기화
- `macro.router_d`와 experts는 target에서 새로 학습

#### `Full-router-init`

- `macro.router_b + macro.router_d`를 source checkpoint로 초기화
- experts는 target에서 새로 학습

### 5.4 dataset pair

#### `KuaiRecLargeStrictPosV2_0.2 -> lastfm0.03`

- domain 차이가 커서 transferable prior가 진짜로 남는지 보기 좋다.
- 성능이 좋아지면 “behavioral family prior is not domain-identity-specific” 메시지가 강해진다.

#### `lastfm0.03 -> KuaiRecLargeStrictPosV2_0.2`

- 역방향을 같이 봐야 asymmetric transfer를 확인할 수 있다.
- source richness / target sparsity 방향 차이가 있으면 해석 포인트가 생긴다.

#### `amazon_beauty -> retail_rocket`

- 둘 다 commerce 계열이라 coarse shopping behavior prior가 공유될 가능성이 있다.
- domain similarity가 어느 정도 있는 positive control 역할도 한다.

#### `foursquare -> movielens1m`

- sequence/novelty/transition 구조가 달라 fine router의 dataset specificity를 보기 좋다.
- `Group-init`만 좋고 `Full-router-init`이 나빠지면 특히 설득력이 높다.

### 5.5 기대 결과와 해석

- `Group-init > Native`
  - main hypothesis 지지
  - coarse behavioral group prior는 transferable

- `Full-router-init <= Group-init`
  - finer routing은 dataset-specific
  - 논문 핵심 문장으로 쓰기 좋다

- `FeatureEncoder-init`만 좋음
  - router prior transfer보다 feature scaling/alignment 효과 가능성
  - 이 경우 transfer story를 “representation portability” 쪽으로 조정해야 한다

- 모든 transfer가 비슷하거나 약간 나쁨
  - negative result이지만, “explicit routing even with compact features still requires target-specific adaptation”으로 쓸 수 있다

### 5.6 Stage 1에서 보고 싶은 메인 그림

- pair별 4-way bar plot
  - `Native`
  - `FeatureEncoder-init`
  - `Group-init`
  - `Full-router-init`

이 그림 하나로도 논문 본문에서 상당히 많은 말을 할 수 있어야 한다.

---

## 6) Stage 2 — Init, Anchor, or Freeze?

### 6.1 목적

Stage 1에서 `Group-init`이 유망했다면, 그 이득이 다음 중 무엇인지 분리해야 한다.

1. 단순 initialization boost
2. source prior를 계속 유지해야 하는 효과
3. source prior와 target adaptation 사이의 trade-off

즉, 이번 stage는 “transfer prior를 얼마나 유지해야 하는가”를 보는 stage다.

### 6.2 대상

- Stage 1에서 `Group-init`이 `Native`보다 좋거나 최소한 동률에 가깝게 유지된 pair만 진행

### 6.3 비교군

#### `Group-init`

- Stage 1의 기준선

#### `Group-init + weak anchor`

- `macro.router_b`를 source에서 초기화
- 학습은 허용하되 source params에서 너무 멀어지지 않도록 약한 L2 anchor 부여

#### `Group-init + stronger anchor`

- 같은 방식이지만 anchor를 stronger하게 설정

#### `Group-freeze`

- `macro.router_b`를 source에서 초기화 후 freeze

#### `FeatEnc+Group-init + anchor`

- Stage 1에서 feature encoder도 도움이 있었을 때만 추가

### 6.4 기본 값

- `anchor_lambda_group_router = 1e-5, 5e-5`
- `lr_scale_group_router = 0.1, 0.3`

### 6.5 기대 결과와 해석

- `weak anchor`가 최고
  - 가장 좋은 시나리오
  - source prior는 useful하지만 target adaptation도 필요
  - 논문 메시지가 가장 매끄럽다

- `freeze`가 최고
  - source prior가 매우 강하게 transferable
  - 매우 인상적이지만 pair-specific인지 확인이 필요

- anchor/freeze가 모두 나쁨
  - init only benefit 또는 negative transfer
  - 그래도 “portable prior는 약하지만 initialization bias는 존재”라는 식으로 정리 가능

### 6.6 Stage 2에서 보고 싶은 그림

- drift norm vs performance scatter
- anchor strength별 bar plot

이 stage는 성능표 하나보다 **drift와 성능의 관계**를 보여줄 때 훨씬 논문에 쓰기 좋다.

---

## 7) Stage 3 — How Far Can Transfer Go?

### 7.1 목적

Stage 1~2의 결과가 positive였다면, 이제 질문은 다음으로 이동한다.

> **transferable한 것이 정말 macro group router에만 국한되는가, 아니면 더 넓은 router까지 확장 가능한가?**

### 7.2 대상

- Stage 1~2에서 signal이 좋았던 pair만
- 모든 pair를 확장하지 않는다

### 7.3 비교 축

#### `macro group-only`

- `macro.router_b`만 transfer

#### `macro full-router`

- `macro.router_b + macro.router_d` transfer

#### `macro+mid group-only`

- `macro.router_b + mid.router_b` transfer

#### `macro+mid full-router`

- `macro.router_b + macro.router_d + mid.router_b + mid.router_d` transfer

### 7.4 제외 항목

- expert transfer는 계속 제외
- micro transfer는 메인 결과표에서 제외
- micro는 appendix 후보로만 소량 확인

### 7.5 기대 결과와 해석

- `macro`만 좋고 `mid/full-router`에서 이득이 줄어듦
  - 가장 깔끔한 메인 메시지
  - transferable prior는 coarse level에 집중

- `mid group`까지 약간 좋음
  - “coarse + intermediate intent prior”까지 확장 가능
  - 메인 문장을 살짝 넓혀 쓸 수 있다

- `macro full-router`가 `macro group-only`보다 consistently 나쁨
  - fine allocation이 dataset-specific이라는 결론 강화

### 7.6 Stage 3에서 보고 싶은 그림

- scope comparison table
  - `macro group-only`
  - `macro full-router`
  - `macro+mid group-only`
  - `macro+mid full-router`

이 stage는 표와 짧은 해석이 가장 잘 맞는다.

---

## 8) Stage 4 — Multi-Hop Transfer and Order Effect

### 8.1 목적

Stage 4는 단순히 pair를 더 늘리는 stage가 아니다.  
이번 stage의 목적은 **prior accumulation**과 **transfer order dependence**를 보는 것이다.

이 stage가 잘 나오면 논문적으로 가장 새롭다.

### 8.2 기본 설계

target dataset `C`를 하나 정하고 다음을 비교한다.

- `A -> C`
- `B -> C`
- `A -> B -> C`
- `B -> A -> C`

### 8.3 기본 transfer 범위

- `macro.router_b` only

이유:

- Stage 4의 main message는 accumulation/order effect이지, 범위 확장이 아니다
- 따라서 transfer object는 가장 clean한 `macro group router`로 고정하는 것이 맞다

### 8.4 추천 target

- `retail_rocket`
- 또는 `movielens1m`

둘 다 Stage 1~3에서 상대적으로 해석 포인트를 만들기 좋다.

### 8.5 추천 source 조합

- `Kuai`, `LastFM`
- 또는 `amazon_beauty`, `foursquare`

### 8.6 기대 결과와 해석

- two-hop > one-hop
  - coarse behavioral prior accumulates across datasets

- `A -> B -> C`와 `B -> A -> C`가 다름
  - transfer prior is path-dependent

- 차이가 거의 없음
  - accumulation보다는 target adaptation이 우세
  - 그래도 negative result로 충분히 의미 있음

### 8.7 Stage 4에서 보고 싶은 그림

- order-effect line plot 또는 grouped bar chart

이 결과는 절대 규모보다 “순서에 따라 바뀌는지”가 핵심이다.

---

## 9) 구현 계획

이번 문서는 문서 설계이지만, 실제 구현으로 이어질 수 있도록 필요한 변경 단위를 명시한다.

### 9.1 selective checkpoint load target

다음 prefix를 기준으로 selective load 가능해야 한다.

- `stage_executor.stage_blocks.<stage>.router_b.*`
- `stage_executor.stage_blocks.<stage>.router_d.*`
- `stage_executor.stage_blocks.<stage>.feature_encoder.*`
- `stage_executor.stage_blocks.<stage>.experts.*`

현재 `FeaturedMoE_N3` 구조상 `N3StageBlock` 내부 모듈 분해가 이미 되어 있으므로,
새 모델을 만들기보다 **state_dict prefix matching**으로 구현하는 것이 가장 자연스럽다.

### 9.2 transfer config

최소한 다음 키가 필요하다.

- `transfer.enable`
- `transfer.scope`
- `transfer.source_checkpoint`
- `transfer.source_arch`
- `transfer.source_dataset`
- `transfer.target_dataset`
- `transfer.load_feature_encoder`
- `transfer.freeze_group_router`
- `transfer.freeze_intra_router`
- `transfer.lr_scale_group_router`
- `transfer.lr_scale_intra_router`
- `transfer.lr_scale_feature_encoder`
- `transfer.anchor_lambda_group_router`
- `transfer.anchor_lambda_intra_router`

권장 scope 예시:

- `none`
- `macro_group_only`
- `macro_full_router`
- `macro_mid_group_only`
- `macro_mid_full_router`

### 9.3 optimizer group / lr scale

transferred params와 fresh params를 optimizer param group 수준에서 분리해야 한다.

예시:

- transferred group router
- transferred intra router
- transferred feature encoder
- fresh parameters

이렇게 해야 “transfer param만 lr를 낮춘다”는 실험이 깔끔해진다.

### 9.4 anchor regularization

anchor는 source-loaded parameter의 frozen copy를 잡아두고, 현재 target 학습 중 파라미터와의 L2 penalty를 더하는 방식이 가장 단순하다.

권장 원칙:

- v1의 anchor 대상은 `router_b` 우선
- `freeze=true`면 anchor 대신 `requires_grad=False`
- `freeze=false`면 anchor + reduced lr 조합을 허용

### 9.5 logging 추가

기존 diag 위에 다음을 추가한다.

- group usage distribution
- feature bucket별 group probability
- transferred parameter drift norm
- source-target group JS divergence
- anchor loss value

이 logging은 성능보다도 논문 해석에 중요하다.

### 9.6 shape compatibility rule

기본 허용 규칙:

- `embedding_size` 동일
- `d_router_hidden` 동일

불일치 checkpoint는 transfer 대상에서 제외한다.  
shape mismatch 상태에서 억지로 load하는 것은 v1의 논문 메시지를 흐린다.

---

## 10) 결과가 나왔을 때 논문에서 어떻게 쓸 것인가

### 10.1 가장 좋은 시나리오

- `Group-init > Native`
- `Group-init + weak anchor`가 최고
- `Full-router-init <= Group-init`

이 경우 논문에서 가장 예쁘게 쓸 수 있는 문장은 다음이다.

> **Cross-dataset transfer is effective at the coarse behavioral family level, but transferring finer expert allocation is less helpful and often harmful, indicating that fine routing remains dataset-specific.**

즉,

- coarse family prior는 shared
- fine allocation은 target adaptation 필요

라는 메시지가 완성된다.

### 10.2 두 번째로 좋은 시나리오

- `Group-init` 이득은 pair-specific
- commerce 계열 또는 일부 비슷한 pair에서만 반복

이 경우:

> **transferability is conditional on dataset similarity**

라는 식으로 쓸 수 있다.

이 또한 충분히 논문 가치가 있다.

### 10.3 덜 좋은 시나리오

- transfer가 대체로 비슷하거나 약간 나쁨

이 경우에도 다음처럼 쓸 수 있다.

> **Even explicit feature-aware routing learned from compact behavioral cues still requires target-specific adaptation, suggesting that the learned routing policies are not trivially portable across datasets.**

즉 “안 됐다”가 아니라,

- transferable하지 않은 이유도 구조적으로 설명할 수 있고
- 이는 router가 실제로 dataset-specific policy를 배우고 있다는 반증이 된다

는 방향으로 정리할 수 있다.

### 10.4 figure 후보

- Stage 1: 4-way bar plot
- Stage 2: drift-vs-performance scatter
- Stage 3: scope comparison table
- Stage 4: order-effect line/bar chart

이 4개만 있어도 transfer 섹션의 narrative가 충분히 구성된다.

---

## 11) 실행 우선순위와 중단 기준

### 11.1 Stage 1 stop rule

- `Group-init`이 어떤 pair에서도 `Native`를 넘지 못하면:
  - Stage 2는 축소
  - Stage 3, 4는 보류

이 경우 feature encoder portability 쪽으로만 짧게 정리하거나, negative transfer 결과로 appendix화한다.

### 11.2 Stage 2 stop rule

- `weak anchor`가 한 번도 이득을 못 주면:
  - anchor/freeze 라인은 축소
  - `Group-init` 단독 결과만 main text에 남기는 쪽이 낫다

### 11.3 Stage 3 stop rule

- `macro+mid`가 전반적으로 악화되면:
  - 최종 메인 story는 `macro-only transfer`
  - 이 자체가 오히려 더 깔끔한 결론일 수 있다

### 11.4 Stage 4 실행 조건

- Stage 1~3에서 최소 `2개 pair` 이상 positive signal이 있을 때만 진행

Stage 4는 가장 흥미롭지만, positive signal이 없는 상태에서 무리하게 확장하면 계산량만 커지고 메시지는 흐려진다.

---

## 12) 최종 권장 메시지

이번 transfer 라인의 최종 목표는 성능을 조금 올리는 것보다,

1. **무엇이 transfer되는지**
2. **무엇이 transfer되지 않는지**
3. **그 경계가 왜 생기는지**

를 `FeaturedMoE_N3`의 구조와 연결해서 설명하는 것이다.

가장 바람직한 결론은 다음 중 하나다.

- **coarse behavioral family prior는 transferable하다**
- **fine expert allocation은 target adaptation이 필요하다**
- **transferability는 dataset similarity와 stage granularity에 의존한다**

즉, 이번 문서의 성패는 “모든 실험이 잘 되는가”가 아니라,

> **router transferability의 경계를 얼마나 설득력 있게 그리느냐**

에 달려 있다.

---

## 13) Assumptions

- 문서 경로는 `experiments/run/fmoe_n3/docs/plans/transfer_learning.md`
- 기본 source architecture는 `A10`
- checkpoint 부족 시 shape-compatible `A8` fallback 허용
- main story는 `macro`와 `router_b` 중심
- expert transfer는 v1 범위 밖
- 문서는 바로 구현 가능한 수준의 설계서를 목표로 하며, 이후 `sh/py/config/model` 수정 시 기준 문서로 사용한다
