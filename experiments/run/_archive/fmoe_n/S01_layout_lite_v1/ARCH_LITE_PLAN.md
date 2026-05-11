# FeaturedMoE_N S01 ARCH3 Plan

## Purpose

- `S01_layout_lite_v1`는 더 이상 light layout만 비교하는 phase가 아니다.
- `ARCH3`의 목표는 아래를 한 번에 분리하는 것이다.
  - backbone 자체가 약한가
  - MoE가 오히려 해를 끼치는가
  - rule-based router가 current layout에서만 안 맞는가
  - regularizer / group-aware router input이 collapse를 줄일 수 있는가

## Dataset Mode

- 한 번 실행할 때 dataset 하나만 돈다.
- 지원:
  - `KuaiRecSmall0.1`
  - `lastfm0.03`
- combo topology는 동일하고, batch / LR band만 dataset별로 다르다.

## Core Definitions

### `MoE-off`

- 같은 layout skeleton을 유지한다.
- 단, 각 `moe_block`을 `dense_ffn`으로 바꾼다.
- `stage_inter_layer_style=attn`와 같이 쓰면 실질적으로:
  - `attn -> dense FFN`
  - `attn -> dense FFN`
  같은 transformer/SASRec형 block에 가까워진다.
- 그래서 `MoE-off`는 "같은 backbone에서 routing만 뺀 control"에 가깝다.

### `pure_attention`

- 같은 layout skeleton을 유지한다.
- 단, 각 `moe_block`을 `identity`로 바꾼다.
- `stage_inter_layer_style=attn`면 각 slot이 attention-only block처럼 동작한다.
- 즉 `pure_attention`은 SASRec과 같지는 않다.
- 오히려 "routing도 없고 dense FFN도 없는 더 빡센 attention-only ablation"이다.

### `full rule`

- `router_impl=rule_soft`
- `router_impl_by_stage={}`
- 즉 eligible stage 전체를 rule router로 본다.

### `group-feature router`

- 기존 `router_use_feature=true`는 stage feature bank 전체를 router에 넣는다.
- `ARCH3`에서는 여기에 expert-group 단위 summary를 추가로 넣는 mode를 본다.
- mode:
  - `none`
  - `mean`
  - `mean_std`

## Relevant Layouts

- `L8`
  - repeated macro heavy
  - `macro 0/2`, `mid 0/1`, `micro 0/1`
  - 현재 `ARCH2`에서 가장 강한 축 중 하나
- `L30`
  - light 2-stage tail
  - `global_pre=2`, `macro 0/1`, `mid 0/1`
  - light anchor
- `L7`
  - old strong classic serial anchor
- `L16`
  - global-pre가 들어간 heavy classic serial anchor
- `L19`
  - 3-stage full heavy serial
  - 이번에는 소수 sentry로만 둔다

## Common Fixed Params

- `execution=serial`
- `router_design=simple_flat`
- `embedding_size=128`
- `num_heads=8`
- `d_feat_emb=16`
- `d_expert_hidden=128`
- `d_router_hidden=64`
- `expert_scale=3`
- `moe_top_k=0`
- `feature_encoder_mode=linear`
- `fmoe_schedule_enable=false`
- `weight_decay=5e-5`
- `hidden_dropout_prob=0.10`

## ARCH3 Budget

- combos: `28 / dataset`
- waves: `7`
- GPUs: `4`
- default phase prefix: `ARCH3`
- default budget:
  - `max_evals=4`
  - `epochs=100`
  - `patience=10`

이렇게 잡은 이유:

- `ARCH2`에서 많은 combo가 `early_stop@30~50`에 몰렸다.
- 그래서 이번엔 epoch는 유지하고, 대신 구조를 더 공격적으로 넓힌다.
- `MoE-off` / `pure_attention`은 LR upper를 더 높게 연다.
  - 빨리 망할 구조는 빨리 정리하기 위함이다.

## Combo Families

### Wave 1. Plain anchor re-check

- `A01` `L8 plain attn moe`
- `A02` `L30 plain attn moe`
- `A03` `L7 plain attn moe`
- `A04` `L16 plain attn moe`

역할:

- `ARCH1/2`에서 괜찮았던 skeleton을 plain learned router로 다시 anchor화

### Wave 2. `MoE-off` control

- `A05` `L8 plain attn dense_ffn`
- `A06` `L30 plain attn dense_ffn`
- `A07` `L7 plain attn dense_ffn`
- `A08` `L16 plain attn dense_ffn`

역할:

- "같은 layout인데 routing만 빼면 더 낫나?" 확인

### Wave 3. `pure_attention` control

- `A09` `L8 plain attn identity`
- `A10` `L30 plain attn identity`
- `A11` `L7 plain attn identity`
- `A12` `L16 plain attn identity`

역할:

- attention-only ablation
- backbone 자체가 attention mixing만으로 어느 정도까지 버티는지 확인

### Wave 4. `full rule`

- `A13` `L8 full rule`
- `A14` `L30 full rule`
- `A15` `L7 full rule`
- `A16` `L16 full rule`

역할:

- current weak hybrid를 rule failure로 오해하지 않기 위한 truth-table branch

### Wave 5. hybrid + heavy sentry

- `A17` `L8 hybrid`
- `A18` `L30 hybrid`
- `A19` `L19 plain balance=0`
- `A20` `L19 hybrid balance=0.004`

역할:

- hybrid를 유지
- 동시에 heavier full 3-stage line도 완전히 버리지 않음
- `balance_loss_lambda` 극단값도 같이 본다

### Wave 6. router regularizer

- `A21` `L8 plain + z-loss`
- `A22` `L30 plain + z-loss`
- `A23` `L8 plain + gate entropy`
- `A24` `L30 plain + gate entropy`

역할:

- router logits scale 문제인지
- early collapse 문제인지
를 plain anchor에서 먼저 본다

### Wave 7. group-aware router input

- `A25` `L8 plain + group mean + z-loss`
- `A26` `L30 plain + group mean + z-loss`
- `A27` `L8 bias + group mean_std + entropy`
- `A28` `L16 hybrid + group mean_std + z-loss + entropy + balance=0.004`

역할:

- router에 raw stage feature bank 외에 group summary를 같이 넣어서
  분기 정보가 조금 더 구조적으로 들어가게 한다

## LR Read

현재 가정은 이렇다.

- plain/hybrid/rule `moe` 계열:
  - low-mid LR가 여전히 main
  - Kuai 대체로 `2e-4 ~ 6e-3`
  - lastfm 대체로 `1e-4 ~ 1.8e-3`
- `MoE-off dense_ffn`:
  - routing noise가 없으므로 upper를 더 높게 연다
  - Kuai `~1e-2` 이상까지 허용
- `pure_attention`:
  - 빨리 끝나는 쪽도 감수하고 upper를 더 열어 둔다
  - "높은 LR로 빨리 죽는 구조"도 정보로 본다
- regularizer / group-feature combo:
  - base combo보다 upper를 약간 낮춰 안정성 쪽을 본다

## Current Optimizer / Scheduler Note

- 현재 run config에는 explicit optimizer scheduler가 없다.
- `RecBole` trainer default optimizer는 `Adam`이다.
- 즉 지금은 사실상:
  - `Adam`
  - no LR scheduler
  - tuned `learning_rate`, `weight_decay`
  로 보고 있다.

다음에 별도 축으로 열 만한 건:

- `AdamW`
- cosine decay
- one-cycle 또는 short warmup + decay

다만 `ARCH3`에서는 구조/aux/control 분해가 먼저다.

