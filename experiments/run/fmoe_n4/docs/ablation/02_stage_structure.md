# FMoE_N4 Stage Structure Study

## 1. 목적

이 study는 논문 본문 `Stage-wise Architecture Analysis`와 appendix `Extended Stage and Layout Ablations`를 동시에 커버한다.

핵심 질문은 세 가지다.

- macro / mid / micro 중 어떤 stage가 실제로 중요한가
- dense FFN을 staged routing으로 바꾼 것 자체가 중요한가, 아니면 stage 수가 중요한가
- wrapper와 stage order가 임의의 구현 artifact가 아닌가

## 2. 나중에 만들 파일

- `experiments/run/fmoe_n4/ablation_stage_structure.py`
- `experiments/run/fmoe_n4/ablation_stage_structure.sh`

권장 axis / phase:

- `AXIS = ablation_kuairec_stage_structure_v1`
- `PHASE_ID = P4B`
- `PHASE_NAME = N4_STAGE_STRUCTURE`

## 3. 직접 재사용할 코드 anchor

- `experiments/run/fmoe_n3/ablation/run_ablation_12.py`
- `experiments/run/fmoe_n3/ablation/run_ablation_12_specify.py`
- `experiments/run/fmoe_n3/docs/plans/phase_10_13_specified/phase12_layout_composition.md`
- `experiments/run/fmoe_n3/docs/architecture/plan_router_wrapper.md`

## 4. 이 study에서 바꾸는 키

- `layer_layout`
- `stage_compute_mode`
- `stage_router_mode`
- `stage_router_wrapper`

그 외 `stage_router_source`, `stage_router_granularity`, `stage_feature_injection`, aux lambda는 base에서 그대로 가져간다. 이 study의 목적은 구조 비교이지 cue/regularizer 비교가 아니다.

## 5. setting matrix

### 5.1 stage removal / stage count 핵심 7-setting

이 7개는 main paper에 직접 쓰인다.

| setting_id | setting_key | delta override | 주 용도 |
| --- | --- | --- | --- |
| `ST-01` | `REMOVE_MACRO` | `layer_layout=["attn","mid_ffn","attn","micro_ffn"]`, `stage_compute_mode={macro:none,mid:moe,micro:moe}`, `stage_router_mode={macro:none,mid:learned,micro:learned}` | panel (a) remove macro, panel (b) two-stage |
| `ST-02` | `REMOVE_MID` | `layer_layout=["attn","macro_ffn","attn","micro_ffn"]`, `stage_compute_mode={macro:moe,mid:none,micro:moe}`, `stage_router_mode={macro:learned,mid:none,micro:learned}` | panel (a) remove mid, panel (b) two-stage |
| `ST-03` | `REMOVE_MICRO` | `layer_layout=["attn","macro_ffn","mid_ffn"]`, `stage_compute_mode={macro:moe,mid:moe,micro:none}`, `stage_router_mode={macro:learned,mid:learned,micro:none}` | panel (a) remove micro, panel (b) two-stage |
| `ST-04` | `SINGLE_STAGE_MACRO` | `layer_layout=["attn","macro_ffn"]`, `stage_compute_mode={macro:moe,mid:none,micro:none}`, `stage_router_mode={macro:learned,mid:none,micro:none}` | panel (b) one-stage 후보 |
| `ST-05` | `SINGLE_STAGE_MID` | `layer_layout=["attn","mid_ffn"]`, `stage_compute_mode={macro:none,mid:moe,micro:none}`, `stage_router_mode={macro:none,mid:learned,micro:none}` | panel (b) one-stage 후보 |
| `ST-06` | `SINGLE_STAGE_MICRO` | `layer_layout=["attn","micro_ffn"]`, `stage_compute_mode={macro:none,mid:none,micro:moe}`, `stage_router_mode={macro:none,mid:none,micro:learned}` | panel (b) one-stage 후보 |
| `ST-07` | `DENSE_FULL_ONLY` | `layer_layout=["macro","mid","micro"]`, `stage_compute_mode={macro:dense_plain,mid:dense_plain,micro:dense_plain}`, `stage_router_mode={macro:none,mid:none,micro:none}`, `stage_feature_injection={macro:none,mid:none,micro:none}` | panel (b) dense baseline |

`ST-07`은 `01_routing_control.md`의 `RC-01_SHARED_FFN`와 완전히 동일한 row다. 이미 routing control에서 돌렸다면 재실행하지 말고 summary를 재사용한다.

### 5.2 layout appendix 2-setting

| setting_id | setting_key | delta override | 메모 |
| --- | --- | --- | --- |
| `ST-08` | `ATTN_BEFORE_MID` | `layer_layout=["attn","macro_ffn","attn","mid_ffn","attn","micro_ffn"]` | attention placement sensitivity |
| `ST-09` | `NO_ATTN_ONLY_MOEFFN` | `layer_layout=["macro_ffn","mid_ffn","micro_ffn"]` | attention 없이 routing만 남기는 실험 |

`ST-09`은 성능이 매우 낮을 수 있지만 appendix에서 "routing 전에 contextualization이 필요한가"를 보여주는 역할이 있다.

### 5.3 wrapper / stage-order 5-setting

| setting_id | setting_key | delta override | 메모 |
| --- | --- | --- | --- |
| `ST-10` | `ORDER_MACRO_MICRO_MID` | `layer_layout=["attn","macro_ffn","micro_ffn","attn","mid_ffn"]` | order variant 1 |
| `ST-11` | `ORDER_MID_MACRO_MICRO` | `layer_layout=["attn","mid_ffn","macro_ffn","attn","micro_ffn"]` | order variant 2 |
| `ST-12` | `WRAPPER_ALL_W1_FLAT` | `stage_router_wrapper={macro:w1_flat,mid:w1_flat,micro:w1_flat}` | flat joint wrapper |
| `ST-13` | `WRAPPER_ALL_W4_BXD` | `stage_router_wrapper={macro:w4_bxd,mid:w4_bxd,micro:w4_bxd}` | hierarchical group-conditional wrapper |
| `ST-14` | `WRAPPER_ALL_W5_EXD` | `stage_router_wrapper={macro:w5_exd,mid:w5_exd,micro:w5_exd}` | scalar-group conditional wrapper |

### 5.4 wrapper 변경 시 추가 검증

wrapper만 바꿔도 primitive requirement가 달라질 수 있다. 따라서 `ST-12`~`ST-14`를 구현할 때는 아래 순서를 따른다.

1. base override를 deepcopy한다.
2. `stage_router_wrapper`만 먼저 patch한다.
3. base에 `stage_router_primitives`가 있으면 wrapper requirement와 충돌하지 않는지 확인한다.
4. 필요 시 `router_wrapper.required_primitives_for_wrapper` 기준으로 primitive config를 보정한다.
5. `--dry-run --smoke-test`를 반드시 먼저 돈다.

## 6. 실행 순서

### 6.1 KuaiRec scout 1차

먼저 아래 7개만 돌린다.

- `ST-01`~`ST-07`

이유:

- 본문 panel (a), (b)를 가장 직접적으로 만든다.
- stage count 메시지가 먼저 서야 wrapper/order 실험 해석이 쉬워진다.

예산:

- settings 수: 7
- seed: `1`
- `max_evals=5`
- `tune_epochs=30`
- `tune_patience=4`

### 6.2 KuaiRec scout 2차

1차 결과를 본 뒤 appendix 성격의 7개를 추가한다.

- `ST-08`~`ST-14`

예산:

- settings 수: 7
- seed: `1`
- `max_evals=5`
- `tune_epochs=30`
- `tune_patience=4`

### 6.3 confirm

confirm으로 올릴 것은 아래만 남긴다.

- panel (a)에서 가장 의미 있는 2개
- panel (b)에서 best one-stage 1개
- panel (c)에서 best wrapper/order 2개

총 5개 이하로 제한한다.

예산:

- seeds: `1,2,3`
- `max_evals=3`
- `tune_epochs=50`
- `tune_patience=6`

## 7. 결과를 어떤 CSV에 넣을지

### 7.1 `writing/results/03_stage_structure/03a_stage_ablation.csv`

필수 row:

- `remove_macro` -> `ST-01`
- `remove_mid` -> `ST-02`
- `remove_micro` -> `ST-03`
- `full` -> later selected base row

### 7.2 `writing/results/03_stage_structure/03b_dense_vs_staged.csv`

권장 구성:

- `dense_ffn` -> `ST-07`
- `single_stage` -> `ST-04`~`ST-06` 중 best test MRR 하나
- `two_stage` -> `ST-01`~`ST-03` 중 best test MRR 하나
- `three_stage` -> later selected base row

즉 launcher는 single-stage와 two-stage를 여러 개 돌리지만, figure용 CSV에는 각 stage count의 best row만 넣는다.

### 7.3 `writing/results/03_stage_structure/03c_wrapper_order.csv`

필수 row 후보:

- `order_macro_micro_mid` -> `ST-10`
- `order_mid_macro_micro` -> `ST-11`
- `wrapper_all_w1_flat` -> `ST-12`
- `wrapper_all_w4_bxd` -> `ST-13`
- `wrapper_all_w5_exd` -> `ST-14`

여기서 later selected base와 동일한 row는 export 시 제거한다.

## 8. 구현 시 주의사항

- `REMOVE_*`는 dense replacement가 아니라 stage 자체를 뺀 skip variant다.
- `DENSE_FULL_ONLY`만 dense baseline으로 두고, 나머지 구조 비교는 MoE stage만 켜는 것이 깔끔하다.
- order variant는 attention block 수를 너무 많이 늘리지 말고, base와 같은 수준의 attn 수를 유지한다.
- wrapper variant는 source / granularity / feature injection을 동시에 바꾸지 않는다. 그래야 panel (c)가 wrapper/order 비교로 남는다.

## 9. 이 study가 끝나면 다음 문서로 넘어갈 기준

아래가 정리되면 `03_cue_ablation.md`로 넘어간다.

- three-stage가 dense / single / two-stage 대비 유리한지
- 어느 stage 제거가 가장 치명적인지
- wrapper/order가 단순 artifact인지, 아니면 실제로 차이를 내는지
