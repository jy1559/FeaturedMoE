# FMoE_N3 Core Ablation V3 정리

## 개요

- 기준 로그: `core_ablation_v2/CORE28/KuaiRecLargeStrictPosV2_0.2`
- combo 정의 원본: `core_ablation_v2` builder in `experiments/run/fmoe_n3/run_core_28.py`
- 아래 시간은 **현재 final_only 정책에 맞춘 runtime hint**를 기준으로 잡았다.
- 기존에 남아 있는 예전 smoke 로그는 `observed_*` 컬럼으로만 참고하고, budget 계산에는 직접 쓰지 않는다.
- 현재 기본 실행 정책은 `fmoe_eval_logging_timing=final_only`, `fmoe_feature_ablation_logging=false`다.
- 즉 학습 중 매 epoch마다 diag를 쌓지 않고, best-valid와 test 시점에서만 diag/special을 수집한다.
- 이번 수정 기준 추천 예산은 `모든 combo 최소 3 eval`이다.
- 실제 실행 스크립트 `phase_core_28.sh`도 기본값으로 이 추천 예산을 사용한다.

## 이번 버전에서 보는 핵심 질문

- plain SASRec 형태를 최대한 맞춘 control이 현재 데이터셋에서 어디까지 재현되는가
- stage wrapper만 넣었을 때도 성능이나 학습 양상이 어떻게 달라지는가
- feature를 routing으로 쓰는 것이 더 중요한지, 아니면 FiLM/gated bias 같은 주입만으로도 충분한지
- macro-only, macro+mid, full macro+mid+micro 중 어디까지가 실제 이득이고 어디부터 비용만 커지는지
- token-level routing이 session-level 기본값보다 의미 있는 개선을 주는지

## combo family 의미

- `P` (2개): plain control. SASRec 형태를 최대한 그대로 따라가는 기준선이다. diag 없음.
- `D` (6개): dense wrapper/control. stage 구조는 쓰지만 expert routing은 끄고, plain/FiLM/gated bias만 비교한다. diag 없음.
- `M` (3개): learned MoE의 핵심 anchor. macro only, macro+mid, full까지 단계적으로 늘린다. diag 있음.
- `R` (5개): routing 방식 비교. rule_soft, hidden-only, feature-only, hidden+gated bias, hybrid를 본다. diag 있음.
- `E` (3개): feature encoder 복잡도 비교. linear 대신 complex MLP를 stage별로 넣어본다. diag 있음.
- `T` (2개): routing granularity 비교. 기본 session routing에서 token routing으로 바꿨을 때를 본다. diag 있음.
- `X` (4개): 구조/입력 조건 ablation. macro window, feature family subset, len 30, top-k 2 같은 단일 변화다. diag 있음.
- `C` (3개): capacity/control 비교. expert_scale을 키운 모델과 이에 대응하는 dense control을 맞춘다. `C70`만 diag 있고 `C71/C72`는 diag 없음.

## baseline C3 기준에서 주로 바뀌는 축

- layout 축: `[layer]`, `[layer,layer,layer]`, `[macro]`, `[macro,mid]`, `[macro,mid,micro]`
- compute 축: plain dense / dense + FiLM / dense + gated bias / learned MoE / rule_soft
- router 입력 축: hidden-only / feature-only / hidden+feature
- feature encoder 축: linear / complex
- granularity 축: 기본 `macro=session, mid=session, micro=token`, 일부 combo에서 token routing override
- 구조 ablation 축: macro history 5 -> 10, feature family 전체 -> Tempo+Memory만, len 10 -> 30, top-k 0 -> 2

## 왜 뒤쪽 combo가 더 느렸는가

- `plain/dense` 계열은 diag가 없어서 빠르다.
- `MoE/rule` 계열은 최종 best-valid/test에서 routing diag를 같이 모으기 때문에 고정비가 있다.
- `[macro, mid, micro]` full layout은 `[macro]`, `[macro, mid]`보다 attention + stage block 수가 많아서 느리다.
- `T50/T51`은 mid/macro를 token routing으로 바꿔 router 계산량이 늘어난다.
- `X62`는 `len=30`이라 attention cost가 커진다.
- `C70`은 `expert_scale=3`이라 expert MLP 비용이 증가한다.
- 반대로 `C71/C72`는 param-match는 하지만 expert dispatch가 없어서 훨씬 빠르다.
- 지금 표의 시간은 `final_only` 기준이라, 학습 중간에는 metric 위주로 보고 마지막 best-valid/test에서만 상세 기록을 남긴다.
- 나중에 `per_eval`로 바꾸면 시간은 다시 커지지만, epoch별 router 변화 추적은 더 자세히 볼 수 있다.

## 시간 해석 방법

- `smoke_total_min`: 현재 runner 기준 1 eval / 1 epoch smoke 예상 총시간
- `smoke_trial_min`: 현재 runner 기준 trial 핵심 시간 추정
- `observed_smoke_*`: 예전에 남아 있던 실제 로그 기준 참고값
- `fixed_overhead_min`: 데이터 로드, 초기화, final eval/저장 같은 고정비 추정
- `est_50ep_single_eval_min`: `fixed_overhead + 50 * smoke_trial_min`
- 이 값은 대략적인 비교용이다. 실제 wall time은 early stopping, cache hit, GPU 상태에 따라 달라진다.
- 특히 현재는 early stopping patience가 10이라, 실제 평균 epoch가 50보다 낮으면 아래 총시간도 함께 내려간다.

## 실행 기본값

- 기본 환경: `/venv/FMoE/bin/python`
- 기본 데이터셋: `KuaiRecLargeStrictPosV2_0.2`
- 기본 recipe: `C3` 계열 (`d_model=128`, `num_heads=4`, `attn_dropout=0.15`, `len=10`)
- 기본 search: `lr=5e-5~3e-3 (loguniform)`, `wd`/`dropout`은 choice
- 기본 budget: `전 combo 3 eval`, `tune_epochs=100`, `tune_patience=10`
- 기본 logging: `special on`, `diag final_only`, `feature ablation logging off`

## logging 운영 모드 선택지

- `기본값`: `final_only` + `feature_ablation off`
  - 전체 28 combo 3-eval sweep 기준 약 `3470.4 GPU-min`, 4 GPU wall `14.46 h`
  - 가장 추천되는 기본선이다. 결과를 넓게 보고 싶을 때 쓴다.
- `feature 민감도만 추가`: `final_only` + `feature_ablation on`
  - 예측 배수: 약 `1.10x`
  - 전체 기준 약 `3817.4 GPU-min`, 4 GPU wall `15.91 h`
  - best-valid에서 `zero/shuffle` 비교만 더 보고 싶을 때 적당하다.
- `epoch별 router 추적`: `per_eval` + `feature_ablation off`
  - 예측 배수: 약 `1.23x`
  - 전체 기준 약 `4268.6 GPU-min`, 4 GPU wall `17.79 h`
  - feature zero/shuffle은 빼고, epoch마다 router/diag 변화를 보고 싶을 때 좋다.
- `최대 상세`: `per_eval` + `feature_ablation on`
  - 예측 배수: 약 `1.33x`
  - 전체 기준 약 `4615.6 GPU-min`, 4 GPU wall `19.23 h`
  - 가장 자세하지만 가장 비싸다. 소수 combo 재실행용으로만 권장한다.

### 왜 이렇게 차이나는가

- `per_eval`은 validation pass 자체를 더 추가하는 건 아니고, **매 epoch validation에 diag/special 집계 작업을 붙인다.**
- `feature_ablation on`은 trial 끝에서 `valid_zero`, `valid_shuffle` 두 번의 추가 eval pass를 더 수행한다.
- 현재 총예산의 약 92%가 diag 있는 combo에 몰려 있어서, logging 정책 변화가 전체 wall time에 꽤 크게 반영된다.

### 실전 추천 순서

- 1차: 기본값(`final_only`, feature ablation off)으로 전 combo 3 eval
- 2차: 상위 4~6개 combo만 `final_only + feature_ablation on`으로 재실행
- 3차: 그중 상위 1~2개만 `per_eval`로 다시 돌려 epoch별 router 변화 확인
- 즉, `per_eval`을 전체 sweep에 거는 것보다 `--only`로 소수 combo만 다시 돌리는 쪽이 효율적이다.

## 현재 추천 budget

- 이번 버전 기본 추천은 **전 combo 3 eval 통일**이다.
- 이유:
  - 1 eval이나 2 eval은 운이 너무 크게 작용한다.
  - 그렇다고 빠른 combo만 과하게 늘리면 전체 실험 시간이 급격히 길어진다.
  - 일단 전 combo를 3 eval로 한 번 훑고, 다음 라운드에서 상위 combo만 추가 확장하는 편이 해석이 깔끔하다.
- 추정 총 예산: `3470.4 GPU-min`
- 4 GPU 기준 추정 wall time: `14.46 h`
- 즉, 이전의 12시간 목표보다는 조금 늘어나지만, `최소 3회` 조건을 만족시키는 가장 단순한 기본선이다.
- 대신 이건 꽤 안전한 보수 추정이다. early stop이 빨리 걸리거나 일부 fast combo가 실제 더 빨리 끝나면 실제 wall time은 이보다 내려갈 수 있다.

## 이번 1차 실행 해석 가이드

- `P00/P01`이 baseline과 크게 어긋나면 backbone parity부터 다시 봐야 한다.
- `D10~D15`가 `P01`보다 안정적이거나 좋아지면, stage wrapper/feature injection 자체의 효과가 있다는 뜻이다.
- `M20 -> M21 -> M22` 순으로 좋아지면 stage를 늘릴 가치가 있고, 비슷하면 macro 중심 설계가 더 효율적일 수 있다.
- `R30~R34`는 routing 정보원이 실제로 hidden인지 feature인지, 혹은 hybrid/rule이 더 나은지 확인하는 묶음이다.
- `E40~E42`는 feature encoder를 복잡하게 할 가치가 있는지 판단하는 묶음이다.
- `T50/T51`은 token routing이 비용 증가만 주는지, 실제 gain도 있는지 확인하는 묶음이다.
- `X60~X63`, `C70~C72`는 anchor(M22)를 기준으로 단일 변화가 얼마나 민감한지 보는 확인용이다.

## 다음 라운드에서 바꿔볼 만한 parameter

- `mid_router_temperature`, `micro_router_temperature`
  - 현재 기본은 `1.2`다. `0.9 / 1.0 / 1.4` 정도를 보면 routing sharpness에 따른 안정성 차이를 보기 좋다.
- `balance_loss_lambda`
  - 현재 기본은 `0.002`다. `0.001 / 0.005`를 보면 expert usage 쏠림 억제 강도를 비교할 수 있다.
- `z_loss_lambda`, `gate_entropy_lambda`
  - 지금은 `0.0`이다. router logit 폭주나 지나친 확신을 누르고 싶을 때 소량(`1e-4`, `5e-4`, `1e-3`)부터 시작하기 좋다.
- `rule_agreement_lambda`, `group_coverage_lambda`
  - hybrid/rule 계열에서만 우선 볼 만하다. rule을 약하게 regularize할지, feature group coverage를 넓힐지 판단할 때 쓴다.
- `d_feat_emb`
  - 현재 `16`이다. `32`까지 올리면 feature encoder 표현력이 늘고, 복잡 encoder(E40~E42)와의 상호작용도 보기 좋다.
- `d_router_hidden`
  - 현재 `64`다. `96`이나 `128`로 올리면 learned router capacity 부족인지 확인할 수 있다.
- `macro_session_pooling`
  - 현재 `mean`이다. 필요하면 `last` 또는 `mean+last`류를 다음 구현 후보로 볼 만하다. macro/mid가 session routing인 만큼 pooled hidden 정의가 중요하다.
- `moe_top_k`, `moe_top_k_ratio`
  - 지금 core에서는 `0`과 `2`를 본다. 추가로 sparse routing을 더 세게 보고 싶으면 `top_k=1`도 다음 후보가 된다.
- `dense_hidden_scale`, `expert_scale`
  - dense/MoE capacity 비교를 더 치밀하게 하려면 capacity matched control 쪽에서 같이 조절하는 게 좋다.
- `lr_scheduler_type`, `temperature_warmup_until`, `moe_top_k_warmup_until`
  - 이번 코어 sweep에서는 꺼뒀지만, 상위 combo 재실행 단계에서는 warmup이나 scheduler를 다시 켜볼 가치가 있다.

## 4 GPU 분배안

- 아래 분배는 combo별 추정 시간 합이 한 GPU에 몰리지 않도록 greedy balancing으로 나눈 것이다.

- GPU 0: `867.0 GPU-min` 예상
  - `X62` `m22_len_30` (추천 3 eval, 50epoch 환산 70.4분)
  - `E42` `full_moe_both_complex_mid` (추천 3 eval, 50epoch 환산 60.4분)
  - `R31` `full_moe_hidden_only` (추천 3 eval, 50epoch 환산 60.4분)
  - `X61` `m22_tempo_memory_only` (추천 3 eval, 50epoch 환산 60.4분)
  - `C72` `dense_film_param_match_c70` (추천 3 eval, 50epoch 환산 11.2분)
  - `D15` `dense_gated_macro_mid_micro` (추천 3 eval, 50epoch 환산 9.6분)
  - `D11` `dense_plain_macro_mid_micro` (추천 3 eval, 50epoch 환산 8.7분)
  - `P00` `plain_c2_one_layer` (추천 3 eval, 50epoch 환산 8.0분)
- GPU 1: `867.0 GPU-min` 예상
  - `E40` `full_moe_both_complex_all` (추천 3 eval, 50epoch 환산 65.4분)
  - `T51` `full_moe_both_all_token` (추천 3 eval, 50epoch 환산 65.4분)
  - `R33` `full_moe_hidden_gated_bias` (추천 3 eval, 50epoch 환산 60.4분)
  - `X63` `m22_topk_2` (추천 3 eval, 50epoch 환산 60.4분)
  - `C71` `dense_plain_param_match_c70` (추천 3 eval, 50epoch 환산 10.4분)
  - `D13` `dense_film_macro_mid_micro` (추천 3 eval, 50epoch 환산 9.6분)
  - `P01` `plain_c3_three_layer` (추천 3 eval, 50epoch 환산 8.8분)
  - `D12` `dense_film_macro_mid` (추천 3 eval, 50epoch 환산 8.7분)
- GPU 2: `868.4 GPU-min` 예상
  - `R32` `full_moe_feature_only` (추천 3 eval, 50epoch 환산 65.4분)
  - `C70` `m22_expert_scale_3` (추천 3 eval, 50epoch 환산 60.4분)
  - `M22` `full_moe_both_anchor` (추천 3 eval, 50epoch 환산 60.4분)
  - `R34` `hybrid_macro_learn_mid_micro_rule` (추천 3 eval, 50epoch 환산 60.4분)
  - `M21` `macro_mid_moe_both` (추천 3 eval, 50epoch 환산 42.9분)
- GPU 3: `868.0 GPU-min` 예상
  - `T50` `full_moe_both_mid_token` (추천 3 eval, 50epoch 환산 65.4분)
  - `E41` `full_moe_both_complex_macro` (추천 3 eval, 50epoch 환산 60.4분)
  - `R30` `full_rule_soft` (추천 3 eval, 50epoch 환산 60.4분)
  - `X60` `m22_macro_window_10` (추천 3 eval, 50epoch 환산 60.4분)
  - `M20` `macro_only_moe_both` (추천 3 eval, 50epoch 환산 25.4분)
  - `D10` `dense_plain_macro_mid` (추천 3 eval, 50epoch 환산 8.7분)
  - `D14` `dense_gated_macro_mid` (추천 3 eval, 50epoch 환산 8.7분)

## 파일

- combo 상세 표: `experiments/run/artifacts/logs/fmoe_n3/core_ablation_v3/combo_reference.csv`
- 4 GPU 분배표: `experiments/run/artifacts/logs/fmoe_n3/core_ablation_v3/gpu_4way_plan.csv`

## 실행 팁

- 기본 실행:
  `bash experiments/run/fmoe_n3/phase_core_28.sh`
- 이 기본 실행은 이미 `추천 budget(전 combo 3 eval)`을 사용한다.
- 소수 combo만 feature 민감도를 더 보고 싶을 때:
  `python3 experiments/run/fmoe_n3/run_core_28.py --dataset KuaiRecLargeStrictPosV2_0.2 --gpus 0 --only M22,R30,T50 --use-recommended-budget --feature-ablation-logging`
- 모든 epoch에서 diag를 보고 싶을 때:
  `bash experiments/run/fmoe_n3/phase_core_28.sh --eval-logging-timing per_eval --feature-ablation-logging`
- 다만 `per_eval`은 시간이 다시 크게 늘어나므로, 우선은 `final_only`로 결과를 본 뒤 필요한 combo만 다시 돌리는 편이 좋다.
