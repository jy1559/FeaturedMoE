# FMoE_N4 Objective And Regularization Variants

## 1. 목적

이 study는 appendix `Objective and Regularization Variants`와 diagnostics 일부를 만들기 위한 실험이다.

핵심 질문은 아래다.

- route consistency가 실제로 필요한가
- z-loss가 안정성에 얼마나 기여하는가
- balance loss가 성능/안정성/활성 expert 수에 어떤 영향을 주는가

중요한 점은 이 실험이 구조 비교가 아니라 **objective delta 실험**이라는 것이다. 따라서 layout, wrapper, cue family는 base 그대로 둔다.

## 2. 나중에 만들 파일

- `experiments/run/fmoe_n4/ablation_objective_variants.py`
- `experiments/run/fmoe_n4/ablation_objective_variants.sh`

권장 axis / phase:

- `AXIS = ablation_kuairec_objective_v1`
- `PHASE_ID = P4D`
- `PHASE_NAME = N4_OBJECTIVE_VARIANTS`

## 3. 직접 재사용할 코드 anchor

- `experiments/run/fmoe_n3/ablation/ablation_2_12.py`
- `experiments/run/fmoe_n3/run_phase9_auxloss.py`
- `writing/results/A02_objective_variants/A02_objective_variants.csv`

## 4. 이 study에서 바꾸는 키

- `route_consistency_lambda`
- `z_loss_lambda`
- `balance_loss_lambda`

그 외 키는 base에서 그대로 상속한다.

## 5. base lambda 처리 원칙

base가 나중에 정해질 것이므로, 이 study는 절대 lambda 절대값을 새로 설계하지 않는다. 아래 규칙만 따른다.

- `*_OFF`: 해당 lambda를 `0.0`으로 만든다.
- `*_ONLY`: 해당 lambda만 base 값을 유지하고 나머지는 `0.0`으로 만든다.
- `*_X2`: 해당 lambda를 `base_lambda x 2`로 키운다.

단, base에서 이미 어떤 lambda가 `0.0`이면 해당 `*_OFF` row는 중복이므로 만들지 않는다.

## 6. setting matrix

### 6.1 core 8-setting

| setting_id | setting_key | delta override | 메모 |
| --- | --- | --- | --- |
| `OBJ-01` | `NO_CONSISTENCY` | `route_consistency_lambda=0.0` | consistency 제거 |
| `OBJ-02` | `NO_ZLOSS` | `z_loss_lambda=0.0` | z-loss 제거 |
| `OBJ-03` | `NO_BALANCE` | `balance_loss_lambda=0.0` | balance 제거 |
| `OBJ-04` | `ALL_AUX_OFF` | `route_consistency_lambda=0.0`, `z_loss_lambda=0.0`, `balance_loss_lambda=0.0` | 완전 제거 하한선 |
| `OBJ-05` | `CONSISTENCY_ONLY` | `z_loss_lambda=0.0`, `balance_loss_lambda=0.0` | consistency만 유지 |
| `OBJ-06` | `ZLOSS_ONLY` | `route_consistency_lambda=0.0`, `balance_loss_lambda=0.0` | z-loss만 유지 |
| `OBJ-07` | `BALANCE_ONLY` | `route_consistency_lambda=0.0`, `z_loss_lambda=0.0` | balance만 유지 |
| `OBJ-08` | `CONSISTENCY_PLUS_ZLOSS` | `balance_loss_lambda=0.0` | 본문 objective 핵심 pair |

### 6.2 optional strong 2-setting

| setting_id | setting_key | delta override | 메모 |
| --- | --- | --- | --- |
| `OBJ-09` | `CONSISTENCY_X2` | `route_consistency_lambda=base_consistency_lambda * 2` | 너무 약해 보일 때만 |
| `OBJ-10` | `ZLOSS_X2` | `z_loss_lambda=base_z_loss_lambda * 2` | sharpness 안정성 확인 |

`OBJ-09` / `OBJ-10`은 appendix 심화용이다. core 8개에서 메시지가 충분하면 생략 가능하다.

## 7. 구현 방식

### 7.1 runner에서 꼭 켤 logging

- `fmoe_diag_logging = true`
- `fmoe_special_logging = true`
- `log_unseen_target_metrics = true` 가능하면 유지

### 7.2 비교 metric을 미리 고정할 것

이 study는 단순 MRR만 보면 해석이 약하므로, summary export 시 아래 세 축을 같이 뽑는다.

- quality metric
  - `MRR@20`
- consistency metric
  - 1순위: `route_consistency_knn`
  - 2순위: `route_consistency_group_knn`
- stability metric
  - 1순위: `effective_experts_mean`
  - 2순위: `router_entropy_mean`
  - 3순위: `train_divergence_count`

필드명이 diag csv마다 다를 수 있으므로, runner 구현 후 aggregation helper에서 fallback 순서를 같이 구현하는 편이 좋다.

## 8. 실행 순서

### 8.1 KuaiRec scout core

- `OBJ-01`~`OBJ-08`
- seed: `1`
- `max_evals=5`
- `tune_epochs=30`
- `tune_patience=4`

### 8.2 confirm

confirm으로 올릴 것은 아래만 남긴다.

- `ALL_AUX_OFF`
- `NO_CONSISTENCY`, `NO_ZLOSS`, `NO_BALANCE` 중 성능 차이가 큰 2개
- `CONSISTENCY_PLUS_ZLOSS`

총 4개 이하로 제한한다.

예산:

- seeds: `1,2,3`
- `max_evals=3`
- `tune_epochs=50`
- `tune_patience=6`

### 8.3 optional strong variant

- base objective가 너무 약해 보여 차이가 안 나는 경우에만 `OBJ-09`, `OBJ-10`을 1-seed로 추가한다.

## 9. 결과를 어떤 CSV에 넣을지

### 9.1 `writing/results/A02_objective_variants/A02_objective_variants.csv`

필수 컬럼은 이미 템플릿에 있다.

- `objective_variant`
- `quality_metric`
- `quality_value`
- `consistency_metric`
- `consistency_value`
- `stability_metric`
- `stability_value`

권장 export row:

- `full_objective` -> later selected base row
- `no_consistency` -> `OBJ-01`
- `no_zloss` -> `OBJ-02`
- `no_balance` -> `OBJ-03`
- `all_aux_off` -> `OBJ-04`
- `consistency_plus_zloss` -> `OBJ-08`

## 10. diagnostics와 연결할 것

`A03_routing_diagnostics`용 csv는 objective study에서 따로 새 변형을 더 만들 필요는 없다. 아래 조합만 별도로 export하면 충분하다.

- later selected base row
- `OBJ-04_ALL_AUX_OFF`
- objective study에서 가장 불안정했던 1개
- objective study에서 가장 안정적이었던 1개

이 네 개만으로도 expert usage / entropy / consistency plot을 만들 수 있다.

## 11. 구현 시 주의사항

- base에서 balance loss가 원래 `0.0`이면 `OBJ-03`, `OBJ-07`은 자동 제거한다.
- objective 실험은 cue family나 wrapper를 절대 같이 바꾸지 않는다.
- MRR 차이가 작더라도 stability가 크게 좋아지면 appendix value가 있으므로 summary에 남긴다.
