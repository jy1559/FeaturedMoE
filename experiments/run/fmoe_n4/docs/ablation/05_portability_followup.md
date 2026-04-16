# FMoE_N4 Beauty And Retail Portability Follow-Up

## 1. 목적

이 문서는 main KuaiRec ablation을 다 끝낸 뒤, selected variant 몇 개만 `beauty`와 `retail_rocket`로 옮겨 보는 후속 계획이다.

이 follow-up은 main paper 필수는 아니다. 다만 사용자가 명시적으로 `kuairec 고정, beauty나 retail까지 가능`이라고 했기 때문에, implementation-ready한 후속 명세를 남겨 둔다.

핵심 목적은 transfer learning 자체가 아니라 **variant portability** 확인이다.

- KuaiRec에서 중요했던 구조/feature/objective 차이가 다른 도메인에서도 살아남는가
- category-rich인 `beauty`와 sparse anonymous stream인 `retail_rocket`에서 반응이 달라지는가

## 2. 나중에 만들 파일

- `experiments/run/fmoe_n4/ablation_portability_followup.py`
- `experiments/run/fmoe_n4/ablation_portability_followup.sh`

권장 axis / phase:

- `AXIS = ablation_portability_followup_v1`
- `PHASE_ID = P4E`
- `PHASE_NAME = N4_PORTABILITY_FOLLOWUP`

## 3. 어떤 variant만 옮길지

새로운 setting matrix를 만들지 말고, 앞 study에서 이미 정의한 row를 재사용한다.

반드시 가져갈 row:

- `FULL_BASE`
  - later selected base row
- `SHARED_FFN`
  - `RC-01` 또는 `ST-07`
- `SEQUENCE_ONLY_PORTABLE`
  - `CF-03`

조건부로 가져갈 row:

- routing control best challenger 1개
  - 예: `ROUTER_SOURCE_FEATURE`가 base가 아니면 이것
  - 예: `ROUTER_SOURCE_BOTH`가 base와 가장 근접한 대조군이면 이것
- objective simplification best 1개
  - 조건: KuaiRec confirm에서 full base 대비 test MRR 차이가 `0.002` 이내일 것

즉 portability follow-up은 최대 5개 row만 들고 간다.

## 4. dataset 순서

권장 순서는 아래다.

1. `beauty`
   - category / time cue가 비교적 살아 있으므로 cue portability를 보기 좋다.
2. `retail_rocket`
   - sparse한 로그 조건에서 `sequence_only`가 얼마나 버티는지 보기 좋다.

## 5. 실행 정책

### 5.1 stage 1: 1-seed screen

- dataset: `beauty`, `retail_rocket`
- variant 수: 최대 5
- seed: `1`
- `max_evals=5`
- `tune_epochs=30`
- `tune_patience=4`
- LR choice는 target dataset 기준으로 base LR에서 상대배율만 적용

### 5.2 stage 2: 3-seed confirm

아래 조건을 만족하는 row만 confirm으로 올린다.

- `sequence_only`가 `shared_ffn` 대비 gain을 유지함
- 또는 full base와의 gap이 target dataset에서 `0.003` 이내임
- 또는 routing challenger가 target dataset에서 KuaiRec와 동일한 방향성 차이를 보임

confirm 예산:

- seeds: `1,2,3`
- `max_evals=3`
- `tune_epochs=50`
- `tune_patience=6`

## 6. 구현 방식

### 6.1 기존 결과를 읽어 variant 목록을 만들 것

이 launcher는 setting을 코드에 다시 적지 말고, 앞선 study manifest 또는 summary에서 가져오는 방식이 더 낫다.

권장 방식:

1. `routing_control` summary에서 `SHARED_FFN`와 best challenger를 읽는다.
2. `cue_family` summary에서 `SEQUENCE_ONLY_PORTABLE`를 읽는다.
3. `objective_variants` summary에서 simple-but-close winner를 읽는다.
4. 위 row의 `delta_overrides`를 target dataset용으로 다시 적용한다.

즉 portability launcher는 `selected_setting_refs.json` 같은 작은 입력 파일을 받아도 좋다.

### 6.2 direct rerun과 transfer learning을 분리할 것

이번 문서에서 정의하는 것은 direct rerun이다.

- source checkpoint를 warm-start하지 않는다.
- target dataset에서 scratch 또는 base policy로 다시 튠한다.

만약 이후 실제 source-target transfer까지 하고 싶다면, 그때는 `experiments/run/fmoe_n3/transfer_learning/stageA.py`를 참고해서 별도 문서를 다시 만드는 편이 맞다.

## 7. 결과를 어디에 둘지

이 follow-up은 두 가지 방식 중 하나를 택한다.

### 7.1 권장 방식

우선은 `experiments/run/artifacts/logs/fmoe_n4/ablation_portability_followup_v1/...` 아래에만 summary를 두고, appendix 반영 여부는 나중에 정한다.

이유:

- direct rerun은 `A05_transfer`의 엄밀한 의미의 transfer와 다르다.
- 논문 막판에 appendix에서 빼도 정리하기 쉽다.

### 7.2 appendix에 억지로 넣어야 한다면

`writing/results/A05_transfer/A05c_transfer_variants.csv`에 아래 식으로 넣을 수는 있다.

- `source_dataset = KuaiRecLargeStrictPosV2_0.2`
- `target_dataset = beauty` 또는 `retail_rocket`
- `variant_group = direct_rerun_variant`

단, 실제 transfer가 아니므로 notebook caption이나 notes에 direct rerun임을 명시해야 한다.

## 8. 이 follow-up을 언제 생략해도 되는지

아래 조건이면 portability follow-up은 생략해도 된다.

- main KuaiRec ablation만으로도 논문 메시지가 충분함
- beauty / retail 예산이 없거나
- `sequence_only`와 `shared_ffn` 차이가 KuaiRec에서조차 너무 작아 target으로 옮길 가치가 없음

## 9. 구현 시 주의사항

- `beauty`와 `retail_rocket`는 target domain 특성이 달라서, KuaiRec winner를 그대로 고정값으로 이식하지 말고 LR만은 좁게 다시 본다.
- portability follow-up의 목적은 full sweep이 아니다. variant 수를 최대 5개로 자르는 것이 맞다.
- 결과가 애매하면 appendix note로만 쓰고 main claim에 섞지 않는다.
