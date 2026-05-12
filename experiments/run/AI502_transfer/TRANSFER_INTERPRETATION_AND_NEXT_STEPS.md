# Transfer 결과 해석과 다음 실험 제안

## 현재 상태

- 최종 실험 위치: `artifacts_group1_best`
- 결과 row: native 70개, transfer 720개
- 정상성 확인:
  - 모든 로그가 `[RUN_STATUS] END status=normal`로 종료
  - `Traceback`, `RuntimeError`, `ValueError`, `ERROR` 계열 에러 없음
  - 모든 transfer row에서 `loaded_tensors > 0`
  - 모든 transfer row에서 `train_changed_tensors > 0`
- 정리:
  - transfer checkpoint 24GB 삭제
  - 예전 shared-preset 기반 `artifacts`, `artifacts_real` 삭제
  - 현재 남은 큰 파일은 group1 native checkpoint bank 약 3GB

## 핵심 결과 요약

이번 결과는 “transfer가 target 성능을 확실히 올린다”보다는 “full-model transfer는 위험하고, feature/router 중심 partial transfer는 훨씬 안정적이다”에 가깝다.

| mode | mean gain | win-rate | 해석 |
|---|---:|---:|---|
| `feature_encoder_init` | +0.000317 | 0.56 | 평균 gain은 작지만 가장 안정적인 partial 축 |
| `feature_encoder_a12_router_init` | +0.000260 | 0.53 | feature encoder + A12 router 전체, 평균 gain은 작음 |
| `feature_encoder_group_router_init` | +0.000101 | 0.65 | 평균 gain은 작지만 win-rate가 가장 좋음 |
| `group_router_init` | -0.000769 | 0.51 | router만으로는 충분하지 않음 |
| `full_except_feature_router_init` | -0.006655 | 0.29 | feature/router 제외 full transfer도 negative |
| `full_model_init` | -0.006852 | 0.22 | 명확한 negative transfer |

가장 중요한 비교는 partial vs full이다. partial transfer mode들은 full model 대비 평균적으로 약 `+0.006~+0.007` 높고, full 대비 win-rate도 `0.73~0.76` 수준이다. 따라서 “partial transfer가 성능을 크게 올린다”는 주장보다, “item/domain mismatch 상황에서 full transfer의 negative transfer를 피하면서 feature/router-local knowledge만 상대적으로 안정적으로 가져온다”는 주장이 더 안전하다.

## 절대 gain에 대한 판단

partial transfer의 절대 gain은 매우 작다. mode 전체 평균 기준 Cohen's d도 대략 다음 수준이다.

| mode | Cohen's d 대략값 | 판단 |
|---|---:|---|
| `feature_encoder_init` | 0.07 | 거의 noise 수준 |
| `feature_encoder_group_router_init` | 0.02 | 거의 noise 수준 |
| `feature_encoder_a12_router_init` | 0.05 | 거의 noise 수준 |
| `full_model_init` | -0.74 | 꽤 뚜렷한 negative transfer |
| `full_except_feature_router_init` | -0.70 | 꽤 뚜렷한 negative transfer |

즉, 현재 결과만으로 “partial transfer가 성능을 유의하게 개선한다”고 쓰기는 어렵다. 다만 “full transfer는 일관되게 손해이고, partial transfer는 그 손해를 피한다”는 결과는 꽤 선명하다.

## target별 관찰

| target | 관찰 |
|---|---|
| `retail_rocket` | partial transfer가 가장 긍정적이다. `feature_encoder_group_router_init` win-rate 0.77, `feature_encoder_a12_router_init` win-rate 0.73 |
| `KuaiRec` | `feature_encoder_init`, `feature_encoder_group_router_init`가 약한 양수. full transfer는 음수 |
| `beauty` | 불안정하다. 특히 `retail_rocket→beauty`에서 full 및 router-only 계열이 자주 손해 |

논문에 넣는다면 target 평균만 크게 넣기보다, target별 breakdown을 같이 보여줘야 한다.

## LR policy 해석

| policy | mean gain | win-rate | 해석 |
|---|---:|---:|---|
| `std` | -0.000915 | 0.50 | 전체적으로 가장 무난 |
| `loaded_lr_0.35` | -0.001033 | 0.48 | std 대비 뚜렷한 개선 없음 |
| `loaded_lr_0.05` | -0.004851 | 0.39 | 전체적으로 너무 보수적 |

`loaded_lr_0.05`는 일부 조합에서는 좋지만 전체적으로는 나쁘다. 거의 freeze에 가까운 loaded LR이 source representation을 보존해 주는 경우가 있더라도, 일반 설정으로 주장하기는 어렵다.

## 논문에 넣을 수 있는 주장

현재 결과로 가능한 안전한 문장:

> Under cross-domain item-space mismatch, full-model transfer consistently suffers from negative transfer. In contrast, feature/router-local initialization is substantially more stable and avoids most of the degradation, although its absolute performance gain over scratch is small.

피해야 할 문장:

> Transfer learning improves recommendation accuracy.

이 문장은 현재 결과로는 너무 강하다. 평균 gain이 작고, 오차범위 내 noise로 볼 수 있는 축이 많다.

## 더 검증한다면 가장 필요한 실험

### 1. Low-data target transfer

target train data를 `10% / 25% / 50% / 100%`로 줄여 비교한다.

기대:
- full-data에서는 transfer gain이 작아도, low-data에서는 source feature/router 초기화가 더 뚜렷하게 작동할 수 있다.
- transfer learning 논문 주장에는 full-data보다 low-data setting이 더 자연스럽다.

필수 비교:
- scratch
- `feature_encoder_init`
- `feature_encoder_group_router_init`
- `feature_encoder_a12_router_init`
- `full_model_init`

### 2. Random-source control

real source checkpoint가 진짜 의미 있는지 확인해야 한다.

비교:
- real source checkpoint
- random source checkpoint
- shuffled source checkpoint
- scratch

판단:
- real source가 random/shuffled보다 낫지 않으면 transfer 효과가 아니라 초기화 noise일 가능성이 크다.
- 이 실험이 있어야 “transfer가 실제 source knowledge를 가져온다”고 말할 수 있다.

### 3. Early convergence curve

최종 MRR gain이 작아도, transfer가 초반 수렴을 빠르게 할 수 있다.

저장할 것:
- epoch별 valid MRR@20
- epoch별 test MRR@20 또는 best-valid checkpoint 기준 test
- early stop epoch
- best epoch

판단:
- final metric이 비슷해도 epoch 5/10/20에서 transfer가 앞서면 “sample/compute efficiency” 주장 가능
- 현재 `epochs_run`만으로는 부족하다.

### 4. Router behavior 분석

metric gain이 작을 때는 routing behavior가 중요하다.

볼 것:
- route entropy
- effective expert count
- top-1 expert concentration
- route consistency
- feature-group alignment

판단:
- `feature_encoder_group_router_init`가 metric gain은 작아도 routing collapse를 줄이거나 consistency를 높이면 논문상 의미가 있다.

### 5. Source similarity 축

source-target pair를 더 체계적으로 잡는다.

예시:
- 같은 target `KuaiRec`에 대해 `foursquare`, `lastfm`, `beauty`, `retail_rocket` source 비교
- dataset profile similarity와 transfer gain 상관 분석

판단:
- “어떤 source가 언제 도움이 되는가”를 보이면 결과가 훨씬 논문스럽다.

## 다음 실험 우선순위

1. `retail_rocket` target low-data 실험
   - 현재 partial transfer signal이 가장 좋다.
2. `KuaiRec` target low-data 실험
   - feature-rich target이라 feature/router transfer 주장과 잘 맞는다.
3. random-source control
   - transfer 효과가 source knowledge인지 초기화 noise인지 분리한다.
4. convergence curve 저장
   - final gain이 작을 때 가장 중요한 보조 증거다.

## 현재 결과를 표로 쓸 때 추천 구성

1. Full vs partial transfer table
   - mode별 mean gain, win-rate, full 대비 delta
2. Target-specific table
   - target별 partial transfer gain
3. Negative transfer table
   - full_model, full_except_feature_router의 일관된 손해
4. Sanity table
   - loaded tensor count, train changed tensor count
5. Optional appendix
   - pair x setting x policy full table

핵심 메시지는 “partial transfer가 강하게 이긴다”가 아니라 “cross-domain transfer에서 무엇을 가져오면 망가지고, 무엇만 가져오면 비교적 안전한가”로 잡는 편이 좋다.
