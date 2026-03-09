# EXPERIMENT_PLAN_hgr

실행은 별도 세션에서 진행하고, 이 문서는 phase 설계와 기본값의 근거를 고정합니다.

## 왜 HGR는 routing-first인가
HGR의 새로움은 `4-group outer router + group-internal expert router`입니다.
그래서 초반 탐색 우선순위는 다음 순서가 맞습니다.

1. `group_router_mode`
2. `stage_merge_mode`
3. `arch_layout_id`
4. optimizer / regularization
5. capacity / batch
6. routing stabilization (`group_top_k`, `moe_top_k`, temp, feature dropout, optional warmup)

`embedding_size`나 `expert_scale`를 너무 일찍 크게 흔들면,
실제로는 routing bias 차이인데 capacity 차이로 묻히기 쉽습니다.

## 기본 anchor
- dataset anchor: `movielens1m`
- base layout: `arch_layout_id=0 = [1,1,1,1,0]`
- base group router: `per_group`
- base stage merge: `serial`
- base top-k: `group_top_k=0`, `moe_top_k=0`
- base dims:
  - `embedding_size=128`
  - `d_feat_emb=16`
  - `d_expert_hidden=160`
  - `d_router_hidden=64`
  - `expert_scale=3`
- base regularization:
  - `hidden_dropout_prob=0.12`
  - `balance_loss_lambda=0.01`
  - `mid_router_temperature=1.3`
  - `micro_router_temperature=1.3`
  - `mid_router_feature_dropout=0.1`
  - `micro_router_feature_dropout=0.1`
  - `use_valid_ratio_gating=true`
- base batch:
  - ML1M: `4096 / 8192`
  - RR: `3072 / 6144`

이 anchor를 쓰는 이유:
- `per_group`가 HGR의 핵심 아이디어라 기본 control로 두기 좋음
- `serial`이 `parallel`보다 variance가 낮고 초반 스크리닝에 유리함
- `group_top_k=0`은 4개 group 모두를 soft하게 보게 해서 초반 collapse를 줄임
- `expert_scale=3`은 `4 x 3 = 12 experts/stage`라 충분히 expressive하면서 OOM risk가 낮음
- `d_expert_hidden=160`은 `128`보다 약간 여유가 있고 `256`보다 안전함

## Phase 0: Smoke
목표: 경로/shape/registration 확인
- `serial + per_group + layout0`
- `parallel + hybrid + layout8`
- dry-run 2개, 실제 1 epoch or short patience 1개

예시:
```bash
bash experiments/run/fmoe_hgr/train_single.sh --dataset movielens1m --layout-id 0 --stage-merge-mode serial --group-router-mode per_group --gpu 0 --dry-run
bash experiments/run/fmoe_hgr/train_single.sh --dataset movielens1m --layout-id 8 --stage-merge-mode parallel --group-router-mode hybrid --gpu 0 --dry-run
```

## Phase 1: Wide-Shallow routing/layout screen
목표: routing family를 빨리 줄이기

원칙:
- 총 `80 combos`
- 절반(`40`)은 base layout `[1,1,1,1,0]`에 고정해서 routing/capacity 영향 확인
- 절반(`40`)은 diverse layout 20종을 고정 capacity로 훑어서 layout 영향 확인
- combo 내부 search는 `learning_rate`, `weight_decay`만 넓고 얕게
- `dropout`, `balance_loss_lambda`는 고정

base-layout half:
- routing profiles 8개: `serial/per_group`, `serial/hybrid`, `serial/stage_wide`, `parallel/per_group`, `parallel/hybrid`, `parallel/stage_wide`
- 일부는 `group_top_k=2`, 마지막 stress row는 `moe_top_k=2`까지 포함
- capacity profiles 5개: anchor, medium+, width+, under-capacity outlier, heavy outlier

diverse-layout half:
- layout 20종
- 각 layout당 2개 routing view:
  - `serial + per_group + group_top_k=0`
  - `parallel + hybrid + group_top_k=2`
- capacity는 anchor 고정

layout 쪽 휴리스틱:
- macro-heavy 다수
- mid는 대부분 `0~1`, 일부 outlier만 `2~3`
- micro는 상당수 `0/-1`
- post layer는 거의 안 쓰고, `post-only` outlier 1개만 유지

권장 search:
- `lr=[2e-4,5e-4,1e-3,2e-3,4e-3,8e-3,1.5e-2]`
- `wd=[0,1e-6,1e-5,1e-4]`
- `dropout=[0.12]`
- `balance=[0.01]`

권장 budget:
- `max_evals=10`
- `epochs=20`
- `patience=5`

## Phase 2: Same-combo confirmation
목표: P1 top result가 우연치인지 확인

원칙:
- P1 top-N을 그대로 다시 돌림
- combo는 고정
- search space만 좁힘
- 이 phase에서 `dropout`과 `balance_loss_lambda`를 처음 열어줌
- `layout / merge / group router / group_top_k / moe_top_k / capacity / batch`는 parent result에서 그대로 상속

권장 search:
- `lr=[2.5e-4,5e-4,7.5e-4,1e-3,1.5e-3,2e-3,3e-3]`
- `wd=[0,5e-6,1e-5,5e-5]`
- `dropout=[0.08,0.12,0.16]`
- `balance=[0.003,0.01,0.03]`

권장 budget:
- `max_evals=16~24`
- `epochs=100`
- `patience=10`

채택 기준:
- best `mrr@20`
- trial 분산이 과도하지 않은지
- OOM / 실패율이 낮은지

## Phase 3: Capacity / batch refinement
목표: best routing combo 위에서 구조적 capacity를 맞춤

원칙:
- `stage_merge_mode`, `group_router_mode`, `group_top_k`, `layout_id`는 parent result에서 고정
- combo간 차이는 `embedding_size`, `d_feat_emb`, `d_expert_hidden`, `d_router_hidden`, `expert_scale`, `batch`
- combo 내부에서 optimizer를 다시 튜닝

권장 combo family:
- `128,16,160,64,3,4096`
- `128,16,192,64,3,4096`
- `128,24,192,80,3,3072`
- `160,16,192,80,3,3072`
- `160,24,224,96,3,3072`
- `128,16,160,64,4,2048`

이 순서로 가는 이유:
- HGR는 outer/intra router 둘 다 있어서 `d_router_hidden`이 너무 작으면 표현력이 부족할 수 있음
- 반대로 `expert_scale=4`는 성능 이득보다 variance/OOM을 먼저 올릴 가능성이 있어 late phase로 미룸
- batch는 capacity와 같이 조절해야 LR sweet spot이 같이 바뀜

## Phase 4: Routing stabilization / optional schedule
목표: top-k / temperature / feature dropout / warmup을 마지막에만 분리 탐색

권장 routing search:
- `group_top_k=[0,2]`
- `moe_top_k=[0,2]`
- `mid_router_temperature=[1.0,1.15,1.3,1.5]`
- `micro_router_temperature=[1.0,1.15,1.3,1.5]`
- `mid_router_feature_dropout=[0.05,0.1,0.15]`
- `micro_router_feature_dropout=[0.05,0.1,0.15]`
- `parallel_stage_gate_top_k=[0,2]` only if parallel

권장 schedule search:
- `alpha_warmup_until=[0.1,0.2,0.3]`
- `alpha_warmup_start=[0.0,0.1,0.2]`
- `temperature_warmup_until=[0.1,0.2,0.3]`
- `mid_router_temperature_start=[1.3,1.6,1.9]`
- `micro_router_temperature_start=[1.3,1.6,1.9]`
- `moe_top_k_warmup_until=[0.1,0.2,0.3]`

권장 원칙:
- `routing` 먼저
- 정체가 있을 때만 `schedule`
- `combined`는 마지막

## num_layers / layout 해석
- HGR는 이제 `num_layers=-1`로 두고, 실제 attention budget은 선택한 `arch_layout_id`가 직접 결정합니다.
- 즉 `num_layers`가 layout을 제한하지 않고, layout의 합이 곧 실제 layer budget입니다.
- 그래서 P1에서는 `2-layer only`가 아니라 `2~4+ layer` layout이 같이 섞입니다.
- 의도는 “좋은 routing bias가 layer budget 때문에 가려지지 않도록” 하는 것입니다.

## RetailRocket transfer
RR는 ML1M best를 그대로 가져가되, 처음에는 아래만 다시 튜닝
- `learning_rate`
- `weight_decay`
- `hidden_dropout_prob`
- `train_batch_size`

RR에서는 `group_router_mode`, `stage_merge_mode`, `layout_id`를 먼저 흔들지 않습니다.
HGR에서는 optimizer/batch 영향이 ML1보다 더 커질 가능성이 높기 때문입니다.
