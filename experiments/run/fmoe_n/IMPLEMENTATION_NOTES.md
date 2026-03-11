# FMoE_N Implementation Notes

## 1. 현재 `FeaturedMoE_N` 구현 요약

### 목표
- `v2`의 stage/layout 실행기와 실험 인프라는 유지
- router / feature encoder / logging은 더 단순하게 재구성
- teacher / distillation / 복잡한 aux를 기본 경로에서 제거
- 향후 rule-based, lightweight regularization, group-aware router를 붙이기 쉽게 유지

### 현재 포함된 것
- 모델명: `FeaturedMoE_N`
- config alias: `featured_moe_n`, `featuredmoe_n`
- run track: `fmoe_n`
- 기본 feature encoder:
  - `linear`
  - `sinusoidal_selected`
- 기본 router:
  - `router_design=simple_flat`
  - `router_impl=learned`
  - stage별 `rule_soft` 교체 가능
  - learned logits에 `rule_bias_scale`로 additive rule prior 추가 가능
- 기본 loss:
  - next-item CE
  - small load-balance loss
  - router temperature schedule
- 기본 특별 logging:
  - `special_metrics.json` 하나만 생성
  - popularity / session length / new user slice metric 집계

### 현재 제외한 것
- `teacher_design`, `teacher_delivery`, `teacher_*`
- distillation 관련 loss
- feature specialization aux
- stage merge aux 기본 사용
- z-loss 실제 반영
- sampled softmax / InfoNCE / gBCE 기본 경로 반영


## 2. 현재 구조 상세

### 2.1 전체 흐름
1. item embedding + position embedding으로 시퀀스 hidden 생성
2. optional global pre-transformer 수행
3. `macro / mid / micro` stage를 layout에 따라 serial 또는 parallel 실행
4. 각 stage의 MoE block은 shared feature bank + stage router + stage expert aggregation으로 구성
5. optional global post-transformer 수행
6. final hidden으로 full-sort scoring

### 2.2 feature encoder
- raw feature는 먼저 `[0,1]` ratio space로 정규화
- `linear`:
  - feature 하나당 1차원 bank만 유지
- `sinusoidal_selected`:
  - 선택된 feature만 `ratio + sin/cos basis`로 확장
  - 기본 대상: `*time*`, `*gap*`, `*int*`, `*pop*`, `*valid_r*`
- 중요한 점:
  - router와 expert가 같은 transformed feature bank를 공유
  - 이전 계열처럼 router용 projection과 expert용 projection을 따로 무겁게 두지 않음

### 2.3 stage / expert 구조
- stage마다 base group 4개를 유지
  - macro: 4 groups
  - mid: 4 groups
  - micro: 4 groups
- 현재 구현에서는 이 base group이 사실상 stage expert slot 역할도 같이 함
- `expert_scale > 1`이면 동일 feature subset을 갖는 clone expert를 추가
- 각 expert는 자기 group에 대응되는 feature subset만 입력으로 사용
- expert는 shared feature bank에서 자기 subset만 gather해서 shallow MLP 수행

### 2.4 현재 router 구조
- 현재 learned router는 stage별로 하나
- 입력:
  - optional hidden
  - optional stage 전체 union feature bank
- 출력:
  - stage의 모든 expert logits를 한 번에 생성
- 즉, 현재는 `group별 local router`가 아니라 `stage-flat router`다

정리하면 현재 learned router는 아래와 같다.

```text
router_input(stage)
  = concat(hidden, stage_union_features)
router_logits(stage)
  -> [all experts in this stage]
softmax/top-k
  -> expert mixture
```

### 2.5 `rule_soft`의 현재 의미
- `rule_soft`는 selected stage feature subset을 `[0,1]` ratio로 바꾸고
- feature별 bin/center 기반 fixed logit을 expert별로 만듦
- 현재 common 의미는 `ratio_bins` 고정
- 사용 방식:
  - `router_impl=rule_soft`: learned router를 완전히 대체
  - `router_impl=learned` + `rule_bias_scale>0`: learned logits에 prior로 더함

### 2.6 특별 logging
- 기본 `on`
- 파일은 run당 `special_metrics.json` 하나만 생성
- 현재 포함:
  - overall valid/test
  - `target_popularity_abs`
    - `<=5`, `6-20`, `21-100`, `>100`
  - `session_len`
    - `1-2`, `3-5`, `6-10`, `11+`
  - `new_user`
    - `new`, `existing`


## 3. 지금 구조의 핵심 장점

- 기존 `v2` 실험 자산을 거의 그대로 활용 가능
- feature encoder가 가벼워져서 해석이 쉬움
- router / expert 모두 feature bank를 공유해서 중복 projection이 줄어듦
- `rule_soft`를 replacement와 prior 둘 다로 쓸 수 있음
- slice metric이 기본으로 남아서 long-tail / short-session narrative를 만들기 쉬움


## 4. 지금 구조의 핵심 한계

### 4.1 지금 router는 group 의미를 강하게 보장하지 않음
질문한 포인트가 맞다.

현재 learned router는:
- stage 전체 feature union을 입력으로 받고
- stage 전체 expert logits를 한 번에 출력한다

그래서 현재 group 의미는 아래처럼만 남아 있다.
- expert body가 자기 group feature subset만 사용한다
- `rule_soft`는 expert별 selected feature를 사용한다
- logging에서 expert/group 이름이 남는다

하지만 learned router 자체는:
- `M1`을 고를 때 `M4` feature를 같이 볼 수 있고
- `m2_3`를 고를 때 `m2_1/m2_2/m2_4` feature 전체를 같이 볼 수 있고
- hidden이 강하면 feature group 구분 없이 expert 선택을 해버릴 수 있다

즉 현재 구조에서 group은
- expert input 관점에서는 의미가 있지만
- router decision 관점에서는 약한 semantic prior일 뿐이다

그래서 "feature를 group별 의미에 맞게 나눠서 라우팅한다"는 원래 의도와는 완전히 일치하지 않는다.

### 4.2 결과적으로 생길 수 있는 문제
- group별 feature를 나눠둔 의미가 router에서 희석됨
- 해석할 때 "왜 이 group이 선택됐는지"가 불명확해짐
- hidden dominance가 생기면 feature-driven routing이라는 주장도 약해짐
- rule_soft가 좋게 나오는 이유가 오히려 여기서 설명될 수 있음
  - rule_soft는 group-feature 대응을 더 강하게 유지하기 때문


## 5. 이 문제를 어떻게 푸는 게 좋은가

## 추천: `flat_stage`를 baseline으로 남기고, 다음 우선순위는 `group_local` router

### 5.1 Option A: Group-local router
가장 먼저 해볼 만한 방향이다.

핵심 아이디어:
- 각 group마다 자기 feature subset만 본다
- group별 score를 따로 계산한다
- 마지막에 group logits를 concat해서 softmax한다

예시:

```text
for each group g in stage:
  group_input_g = concat(hidden, group_features_g)
  group_logit_g = head_g(group_input_g)

router_logits = concat(group_logit_1, ..., group_logit_4)
```

장점:
- 구현이 가장 단순
- 지금 expert 구조를 거의 안 바꿔도 됨
- group feature leakage를 바로 줄일 수 있음
- "이 group은 이 feature로 골랐다"는 해석이 쉬워짐

단점:
- group 간 상호작용은 약해질 수 있음
- hidden이 여전히 너무 강하면 group semantics가 다시 약해질 수 있음

### 5.2 Option B: Hierarchical router
의도를 가장 잘 살리는 구조다.

1. group router가 4개 group 중 어디로 갈지 결정
2. group 내부 router가 그 group의 clone/expert 중 어디를 쓸지 결정

예시:

```text
group_weight = softmax(group_router(hidden, group_summaries))
within_group_weight_g = softmax(local_router_g(hidden, group_features_g))
final_weight[g, e] = group_weight[g] * within_group_weight_g[e]
```

장점:
- group 의미가 명확해짐
- 논문/미팅에서 말한 macro-mid-micro, group semantic story와 가장 잘 맞음
- clone expert 또는 group 내부 다양한 expert family를 붙이기 좋음

단점:
- 구현량이 증가
- hyperparameter가 늘어남
- 잘못 설계하면 학습이 불안정할 수 있음

### 5.3 Option C: Factorized logits
계산 효율과 의미 보존의 타협안이다.

아이디어:
- 최종 expert logit을 `group logit + intra-group logit`으로 분해

```text
logit(group, expert) = logit_group(group_features)
                     + logit_inner(hidden, group_features)
```

장점:
- flat보다 의미가 강함
- hierarchical보다 구현이 가벼움
- group과 clone 역할을 분리하기 좋음

단점:
- 설계가 애매하면 flat과 비슷한 문제를 다시 가질 수 있음


## 6. 내 추천

### 바로 다음 구현 우선순위
1. 현재 `simple_flat`은 baseline으로 유지
2. `router_design=group_local` 추가
3. 그 다음 필요하면 `router_design=hierarchical` 추가

이유:
- 지금 질문한 핵심 의문은 타당하고, flat만으로는 group story가 약하다
- 그렇다고 바로 hierarchical로 가면 구현/튜닝 비용이 커진다
- `group_local`은 의미를 살리면서도 가장 싸게 검증 가능하다

### `group_local`에서 권장하는 기본 형태
- group당 1개 head
- 입력은 `hidden + group-local transformed features`
- `expert_scale=1`이면 group logits 4개만 바로 출력
- `expert_scale>1`이면 group마다 clone logits 추가
- `rule_soft`는 group-local prior로 유지

즉 첫 단계에서는 아래면 충분하다.

```yaml
routing:
  router_design: group_local
  router_impl: learned
  rule_bias_scale: 0.2
  router_use_hidden: true
  router_use_feature: true
  group_router_feature_scope: local
```


## 7. 앞으로 넣을 수 있는 loss / aux / SSL / reg

## 7.1 바로 붙여도 되는 가벼운 것

### 1. z-loss
- softmax partition(`logsumexp`)가 과도하게 커지는 것을 누르는 regularizer
- 보통 logit scale 폭주를 막는 안정화 용도
- router logits이나 item logits에 작게 걸 수 있음
- 추천:
  - 기본 off
  - `1e-4 ~ 1e-3` 수준 작은 lambda만 탐색

### 2. gate entropy regularization
- router가 너무 빨리 collapse하지 않게 entropy 하한을 주는 방식
- load-balance loss보다 직접적으로 routing sharpness를 제어 가능
- 추천:
  - early epoch에만 약하게
  - 후반엔 temperature annealing과 함께 줄이기

### 3. group coverage / usage regularization
- 각 stage에서 특정 group만 계속 선택되는 것을 막음
- 특히 `group_local`이나 `hierarchical`로 갈 때 중요
- 의미:
  - "expert balance"보다 한 단계 위의 "group balance"

### 4. rule agreement penalty
- learned router와 `rule_soft` prior가 너무 멀어지지 않게 KL이나 cosine penalty를 줄 수 있음
- full distillation보다 가볍고 목적이 명확함
- 추천:
  - hard teacher가 아니라 soft prior alignment 정도만

## 7.2 조건부로 볼 만한 것

### 5. popularity-aware weighting / gBCE
- low-pop item 성능을 밀고 싶을 때 고려
- popularity slice narrative를 가져갈 때 맞음
- 추천:
  - baseline 이후
  - 우선은 item popularity inverse weight나 clipped weighting부터

### 6. sampled softmax
- item 수가 클 때 계산량 절감용
- 지금 FMoE_N의 핵심 문제는 router semantics 쪽이라 1순위는 아님
- 정말 큰 dataset에서 학습 비용이 병목일 때만

### 7. InfoNCE / contrastive main branch
- SSL branch를 명확히 붙일 때 의미가 있음
- 그냥 추천 loss 대체재로 바로 넣는 건 우선순위가 낮음

## 7.3 SSL 아이디어

### 1. session crop consistency
- 같은 session에서 두 개 crop/view를 만들고
- final representation뿐 아니라 router decision도 비슷해지게 유도

추천 형태:
- rep-level InfoNCE는 약하게
- router/group distribution KL consistency를 더 핵심으로

### 2. feature masking consistency
- 일부 engineered feature를 mask/drop한 view와 원본 view 사이에서
- group 선택이 과도하게 바뀌지 않도록 regularize

장점:
- "feature-driven routing이 robust한가?"를 직접 다룸

### 3. short/long window agreement
- 최근 짧은 window와 조금 긴 window에서
- micro / mid routing이 지나치게 불안정하지 않게 제약


## 8. 특별 logging에서 다음으로 추가하면 좋은 것

현재 `special_metrics.json`은 evaluation slice 성능만 남긴다.
다음으로는 같은 파일 안에 아래를 추가하는 게 좋다.

### 8.1 router slice summary
- stage별
- slice별
- 평균 gate entropy
- top1 expert share
- active expert count
- group usage distribution

중요:
- 파일을 늘리지 말고 `special_metrics.json` 안에 section 추가

### 8.2 low-pop 개선량 직접 기록
- overall 대비 `<=5` pop bin 성능 차이
- short session(`1-2`) 대비 overall 차이
- new user 대비 existing 차이

이렇게 해두면 나중에 보고서에서 바로 쓸 수 있다.


## 9. 구현 로드맵 제안

### Phase 1
- 현재 `simple_flat` 유지
- `group_local` router 추가
- 특별 logging에 `group usage / entropy by slice` 추가

### Phase 2
- `rule_soft + learned` hybrid를 group-local 기준으로 정리
- `rule_bias_scale` sweep
- `group coverage` / `entropy` regularization 소량 추가

### Phase 3
- 필요하면 `hierarchical` router 추가
- clone expert 또는 group 내부 expert family 차등화
- SSL consistency 실험


## 10. 결론

현재 `FeaturedMoE_N`은:
- 가볍고
- 실험하기 쉽고
- rule prior를 붙이기 쉬운 baseline으로는 적절하다

하지만 지금 learned router는 stage 전체 union feature로 stage 전체 expert를 한 번에 라우팅하기 때문에,
원래 말하던 "group semantic routing"을 강하게 구현한 구조는 아니다.

그래서 다음 핵심 작업은 loss보다 먼저:
- `group_local` 또는 `hierarchical` router로
- group feature와 routing decision의 연결을 구조적으로 강하게 만드는 것

이게 맞다.
