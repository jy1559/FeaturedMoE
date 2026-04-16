# FMoE_N4 Routing Control Study

## 1. 목적

이 study는 논문 본문 `Routing Signal Comparison` 패널을 직접 만드는 실험이다.

비교의 핵심은 네 가지다.

- shared FFN만 썼을 때
- router가 hidden만 볼 때
- router가 hidden + feature를 같이 볼 때
- router가 feature만 볼 때

즉, "expert를 넣었다"가 아니라 **router source가 무엇이냐** 가 핵심인지 먼저 확인하는 실험이다.

## 2. 나중에 만들 파일

- `experiments/run/fmoe_n4/ablation_routing_control.py`
- `experiments/run/fmoe_n4/ablation_routing_control.sh`

권장 axis / phase:

- `AXIS = ablation_kuairec_routing_control_v1`
- `PHASE_ID = P4A`
- `PHASE_NAME = N4_ROUTING_CONTROL`

## 3. 직접 재사용할 코드 anchor

- runner 뼈대: `experiments/run/fmoe_n4/stage1_a12_broad_templates.py`
- setting matrix 스타일: `experiments/run/fmoe_n3/ablation/ablation_3_12.py`
- wrapper primitive 관련 참고: `experiments/models/FeaturedMoE_N3/router_wrapper.py`
- wrapper 설명 참고: `experiments/run/fmoe_n3/docs/architecture/plan_router_wrapper.md`

## 4. 반드시 유지할 공통 규칙

- `fixed_values`는 base result에서 읽어온 값을 그대로 사용한다.
- 이 study에서 바꾸는 것은 `layer_layout`, `stage_compute_mode`, `stage_router_mode`, `stage_router_source`, `stage_router_granularity`, `stage_feature_injection` 뿐이다.
- LR만 아주 좁게 다시 본다.
- batch size와 eval batch size는 base result에서 상속한다.

## 5. setting matrix

### 5.1 main paper 4-setting

| setting_id | setting_key | delta override | 메모 |
| --- | --- | --- | --- |
| `RC-01` | `SHARED_FFN` | `layer_layout=["macro","mid","micro"]`, `stage_compute_mode={macro:dense_plain,mid:dense_plain,micro:dense_plain}`, `stage_router_mode={macro:none,mid:none,micro:none}`, `stage_feature_injection={macro:none,mid:none,micro:none}` | behavior router 없이 dense FFN만 두는 하한선 |
| `RC-02` | `ROUTER_SOURCE_HIDDEN` | `stage_router_source={macro:hidden,mid:hidden,micro:hidden}` | hidden-only control |
| `RC-03` | `ROUTER_SOURCE_BOTH` | `stage_router_source={macro:both,mid:both,micro:both}` | mixed control |
| `RC-04` | `ROUTER_SOURCE_FEATURE` | `stage_router_source={macro:feature,mid:feature,micro:feature}` | behavior-guided control |

### 5.2 appendix용 확장 4-setting

| setting_id | setting_key | delta override | 메모 |
| --- | --- | --- | --- |
| `RC-05` | `GRANULARITY_ALL_SESSION` | `stage_router_granularity={macro:session,mid:session,micro:session}` | granularity confound 확인 |
| `RC-06` | `GRANULARITY_ALL_TOKEN` | `stage_router_granularity={macro:token,mid:token,micro:token}` | fully token-level router |
| `RC-07` | `INJECTION_ALL_GATED_BIAS` | `stage_feature_injection={macro:gated_bias,mid:gated_bias,micro:gated_bias}` | feature reinjection control |
| `RC-08` | `INJECTION_ALL_GROUP_GATED_BIAS` | `stage_feature_injection={macro:group_gated_bias,mid:group_gated_bias,micro:group_gated_bias}` | stronger reinjection control |

`RC-05`~`RC-08`는 main figure 필수는 아니지만, 만약 hidden/feature 결과가 비슷하게 나오면 이 네 개로 confound를 분리한다.

## 6. 구현 방식

### 6.1 `build_settings()`가 만들어야 할 구조

`fmoe_n3/ablation/ablation_3_12.py`와 같은 setting dict를 만든다.

필수 필드:

- `setting_idx`
- `setting_id`
- `setting_key`
- `setting_desc`
- `setting_group`
- `setting_detail`
- `delta_overrides`

여기서 `delta_overrides`는 base override 위에 얹는 값만 담는다.

### 6.2 row 생성 시 고정할 것

- `track = fmoe_n4`
- `dataset = KuaiRecLargeStrictPosV2_0.2`
- `search_algo = random`
- `feature_mode`는 base result에서 읽고, 없으면 `full_v3`
- `fmoe_diag_logging = true`
- `fmoe_special_logging = true`

### 6.3 shared FFN row에서 주의할 점

`SHARED_FFN`은 dense control이므로 아래 세 가지를 같이 내려야 한다.

- `stage_compute_mode = dense_plain`
- `stage_router_mode = none`
- `stage_feature_injection = none`

`stage_router_source`와 `stage_router_granularity`는 남아 있어도 실제로는 쓰이지 않지만, parser 일관성을 위해 base 값을 남겨도 괜찮다.

## 7. 실행 순서

### 7.1 KuaiRec scout

- 먼저 `RC-01`~`RC-04`만 실행한다.
- settings 수: 4
- seeds: `1`
- `max_evals=5`
- `tune_epochs=30`
- `tune_patience=4`

### 7.2 KuaiRec confirm

- `RC-01`은 항상 유지한다.
- `RC-02`~`RC-04` 중 상위 2개만 confirm으로 올린다.
- seeds: `1,2,3`
- `max_evals=3`
- `tune_epochs=50`
- `tune_patience=6`

### 7.3 appendix extension

- core 4개가 정리된 뒤 `RC-05`~`RC-08`을 1-seed scout로 추가한다.
- 이 4개는 main narrative보다 appendix와 diagnostics용이다.

## 8. 결과를 어떤 CSV에 넣을지

### 8.1 `writing/results/02_routing_control/02a_routing_control_quality.csv`

필수 row:

- `shared_ffn` -> `RC-01`
- `hidden_only` -> `RC-02`
- `mixed_hidden_behavior` -> `RC-03`
- `behavior_guided` -> `RC-04`

권장 metric:

- `metric = MRR`
- `cutoff = 20`
- `split = test`

### 8.2 `writing/results/02_routing_control/02b_routing_control_consistency.csv`

동일 4개 row에 대해 아래 metric 하나를 고정해서 넣는다.

- 1순위: `route_consistency_knn`
- 2순위 fallback: `route_consistency_group_knn`

metric 이름은 CSV에 명시적으로 넣는다. 이후 plot notebook가 metric명을 그대로 라벨링하게 두는 편이 안전하다.

## 9. 구현 시 주의사항

- base가 이미 `feature` router라면 `RC-04`는 중복 row가 되므로 생성하지 않는다.
- base가 이미 `both` router라면 `RC-03`도 중복 row다.
- wrapper를 바꾸지 않는 실험이므로 `stage_router_wrapper`는 base 값을 그대로 둔다.
- 이 study는 구조를 바꾸는 실험이 아니므로 `layer_layout`은 `RC-01` 이외에는 base 그대로 둔다.

## 10. 이 study가 끝나면 다음 문서로 넘어갈 기준

아래가 확인되면 `02_stage_structure.md`로 넘어간다.

- feature-only, both, hidden-only 중 하나가 분명히 우세하거나
- 최소한 shared FFN 대비 behavior router 계열이 이득을 보이거나
- 어느 축이 confound인지 appendix 4-setting으로 더 분리할 필요가 명확해졌거나
