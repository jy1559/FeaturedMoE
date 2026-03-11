# FeaturedMoE_v3 Flat Router Plan

`FeaturedMoE_v3`는 flat-router 메인 트랙이다.  
`FeaturedMoE_v2_HiR`는 hierarchical 실험 보조 트랙으로 두고, 넓은 튜닝은 당분간 하지 않는다.

## Router Family
- `flat_legacy`
  - 기존 flat learned router 재현용.
- `flat_hidden_only`
  - hidden만으로 12-way expert logits 생성.
- `flat_global_interaction`
  - `h || s || h*s || |h-s|` 기반 global interaction.
- `flat_hidden_group_clone12`
  - hidden-only 12-way base + group-local clone residual.
- `flat_group_bias12`
  - global 12-way logits + group-local bias broadcast.
- `flat_clone_residual12`
  - global 12-way logits + group-local clone residual.
- `flat_group_clone_combo`
  - group bias와 clone residual을 함께 추가.

모든 설계는 `expert_scale=3`이면 stage마다 `12 logits`를 한 번에 만들고, 마지막에만 `12-way softmax/top-k`를 적용한다.

## Distill Family
- `none`
- `group_only`
  - 4-group rule-like teacher로 coarse group preference를 가르침.
- `clone_only`
  - group-local feature 상태로 group 내부 clone 분류를 가르침.
- `group_plus_clone`

기본 distill schedule:
- `router_distill_temperature=1.5`
- `router_distill_until=0.2`
- `group_only`: `lambda_group=0.002`
- `clone_only`: `lambda_clone=0.003`
- `group_plus_clone`: `lambda_group=0.0015`, `lambda_clone=0.003`

## Phase Order
1. `phase_a_ml1_legacy_repro.sh`
   - ML1 `0.0982` non-rule baseline 재현.
2. `phase_b_ml1_router_structure.sh`
   - 입력/구조만 비교.
3. `phase_c_ml1_distill_modes.sh`
   - `group_only` vs `clone_only` vs `group_plus_clone`.
4. `phase_d_ml1_routing_semantics.sh`
   - `moe_top_k=0/2/4` 비교.
5. `phase_e_ml1_dim_robustness.sh`
   - anchor 밖 dim 3종 확인.
6. `phase_f_rr_transfer.sh`
   - ML1 winner를 RR로 전이.
7. `phase_rr_rule_sanity.sh`
   - RR hybrid/rule comparator quick sanity.

## Tuning Policy
- LR 중심 탐색.
- `wd`는 `{0, 1e-6, 1e-4}` 고정 choice.
- `dropout=0.10` 고정.
- `balance_loss_lambda=0.005` 고정.
- 예외:
  - `phase_a_ml1_legacy_repro.sh`는 legacy parity를 위해 `balance_loss_lambda=0.01`.
