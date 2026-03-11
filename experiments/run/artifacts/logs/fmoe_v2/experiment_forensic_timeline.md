# FMoE_v2 Experiment Forensic Timeline

이 문서는 `FMoE_v2` 실험을 나중에 다시 봐도 흐름을 복원할 수 있게 정리한 포렌식 타임라인이다.  
핵심 질문은 네 가지다.

1. 언제 어떤 phase를 돌렸는가
2. 그 phase의 목표는 무엇이었는가
3. 당시 모델 구조와 기본 설정은 어떤 상태였는가
4. 그 결과 왜 다음 phase로 넘어갔는가

## 읽는 법

- 기준 소스:
  - [experiment_overview.md](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/experiment_overview.md)
  - [experiment_summary.md](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE_v2/experiment_summary.md)
  - [README.md](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/README.md)
  - [featured_moe_v2_tune.yaml](/workspace/jy1559/FMoE/experiments/configs/model/featured_moe_v2_tune.yaml)
  - [moe_stages.py](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE/moe_stages.py)
- 증거 레벨:
  - `explicit`: 실험 로그, `exp_desc`, result JSON, timeline에 직접 남아 있는 내용
  - `inferred`: 현재 스크립트 기본값, focus vars, 시점, diff에서 복원한 내용
- 구조 변경 경계는 `git commit boundary`가 아니라 `snapshot: pre-fmoe-v2-router-overhaul-20260309` 시점과 현재 uncommitted working tree의 차이로 본다.

## 모델 상태 경계

### Snapshot / legacy state

기준: 2026-03-09 02:07 UTC 스냅샷 이전에 실행된 실험 (`P0`, `P1`, `P1S`, `P2DB`, `P2RRF`)  
증거 레벨: `explicit + inferred`

- learned flat router
- `router_design` 없음
- raw stage feature를 `Linear` 한 번 태운 뒤 hidden/feature concat 기반 learned gate
- `moe_top_k` 탐색 허용
- `balance_loss_lambda=0.01`
- `fmoe_v2_feature_spec_aux_enable=false`
- `router_distill_*` 없음

대표 근거:
- [snapshot config view](/workspace/jy1559/FMoE/experiments/configs/model/featured_moe_v2_tune.yaml)
- [snapshot commit](/workspace/jy1559/FMoE/.git/COMMIT_EDITMSG)

### Current factorized state

기준: 스냅샷 이후 현재 working tree에서 실행된 실험 (`P1RFI`, `P1RGI2`, `P2RGI`, `P3RRT`)  
증거 레벨: `explicit`

- `router_design=group_factorized_interaction`
- `group_top_k=0`, `expert_top_k=1` 기본
- feature encoder + hidden encoder + `f || h || f*h || |f-h|`
- `fmoe_v2_feature_spec_aux_enable=true`
- `fmoe_v2_feature_spec_stages=[mid]`
- `fmoe_v2_feature_spec_aux_lambda=3e-4`
- `balance_loss_lambda=0.003`
- `router_distill_enable`, `router_distill_lambda`, `router_distill_temperature`, `router_distill_until` 추가

대표 근거:
- [featured_moe_v2_tune.yaml](/workspace/jy1559/FMoE/experiments/configs/model/featured_moe_v2_tune.yaml)
- [moe_stages.py](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE/moe_stages.py)
- [README.md](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/README.md)

## Phase Index

| 시기 | phase / experiment | dataset | 목표 | 모델 상태 | best_mrr@20 | 증거 |
|---|---|---|---|---|---:|---|
| 2026-03-05 | `P0`, `P1` bootstrap | ML1 | env / report / wandb / basic hyperopt path 확인 | `legacy-flat` | 0.0151 / 0.0243 | explicit + inferred |
| 2026-03-05 | `P1S` | ML1 | layout / execution 1차 스크리닝 | `legacy-flat` | 0.0975 | explicit + inferred |
| 2026-03-05~06 | `P2DB` | ML1 | best layout 근처에서 dim / batch refinement | `legacy-flat` | 0.0982 | explicit |
| 2026-03-08 | `P1S (ReR)` | RR | RR로 transfer한 layout screening | `legacy-flat + RR narrow` | 0.2699 | explicit |
| 2026-03-08 | `P2DB (ReR)` | RR | RR에서 serial L7 기준 dim / batch refinement | `legacy-flat + RR narrow` | 0.2661 | explicit |
| 2026-03-08 | `P2RRF` | RR | RR top layout 주변 large combo 재탐색 | `legacy-flat + RR narrow` | 0.2720 | explicit |
| 2026-03-09 | `P1RFI` | RR | factorized router 전환 직후 blocked-joint probe | `factorized-router + feature-spec` | 0.2617 | explicit |
| 2026-03-09 | `P1RGI2` | RR | factorized router layout-first recovery P1 | `factorized-router + feature-spec` | 0.2618 | explicit |
| 2026-03-09 | `P2RGI` | RR | factorized router narrow P2 dim / small top-k probe | `factorized-router + feature-spec` | 0.2621 | explicit |
| 2026-03-09 | `P3RRT` | RR | router teaching (`expert_top_k`, distill, feature-spec`) 장기 검증 | `factorized-router + router-teach` | 0.2644 | explicit |
| 2026-03-10~ | `P3RRTS` | RR | `expert_top_k=0/1/2` semantics 비교, distill 제거 | `factorized-router + semantics probe planned` | - | explicit + inferred |

## 2026-03-10 Track Split

- `FeaturedMoE_v2` working tree에 flat lineage와 factorized lineage가 함께 섞이기 시작해, 코드/runner/log track을 분리했다.
- flat lineage는 [FeaturedMoE_v3](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE_v3) / [run/fmoe_v3](/workspace/jy1559/FMoE/experiments/run/fmoe_v3)로 이동했다.
- factorized lineage는 [FeaturedMoE_v2_HiR](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE_v2_HiR) / [run/fmoe_v2_hir](/workspace/jy1559/FMoE/experiments/run/fmoe_v2_hir)로 분리했다.
- 운영 기준:
  - `FeaturedMoE_v3`는 flat baseline 복구/개선용 main track
  - `FeaturedMoE_v2_HiR`는 hierarchical routing hypothesis 검증용 side track
  - `FeaturedMoE_v2`는 historical mixed track으로만 본다

## Phase Dossiers

### 1. Bootstrap / ML1 smoke

#### `P0`

- 기간: 2026-03-05 08:02~08:29 UTC
- 대상: `movielens1m`
- 목표: 환경, report, hyperopt path, tiny-batch fallback 확인
- 사용 흔적:
  - [events.jsonl](/workspace/jy1559/FMoE/experiments/run/artifacts/timeline/events.jsonl)
  - [P0 train envcheck log](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/train/P0/movielens1m/FeaturedMoE_v2_serial/movielens1m_FeaturedMoE_v2_serial_train_P0_envcheck_gpu0_20260305_080239_853.log)
  - [P0 result JSON](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/movielens1m_FeaturedMoE_v2_20260305_082854.json)
- 당시 모델 상태: `legacy-flat`
- 탐색 축: `learning_rate`, `weight_decay`, `hidden_dropout_prob`, `balance_loss_lambda`를 nominal하게만 체크
- 결과 해석:
  - 성능 자체는 의미가 거의 없고 (`MRR@20=0.0151`)
  - 중요한 건 `P0_envcheck`, `P0_report_check`, `smallbs` fallback이 timeline에 남아 있어 실행 파이프라인이 열렸다는 점이다.
- 비고:
  - `P0_report_check`의 hparam 로그는 timeline에는 잡히지만 현재 workspace에서 파일 자체는 보이지 않는다.
  - 따라서 이 phase는 [events.jsonl](/workspace/jy1559/FMoE/experiments/run/artifacts/timeline/events.jsonl) + result JSON 기준으로 복원한다.

#### `P1`

- 기간: 2026-03-05 08:48~08:50 UTC
- 대상: `movielens1m`
- 목표: report / wandb / hyperopt summary가 제대로 남는지 한 번 더 확인
- 사용 로그: [P1_log_check](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1/ML1/FMoEv2/20260305_084825_719_hparam_P1_log_check.log)
- 당시 모델 상태: `legacy-flat`
- 탐색 축: `learning_rate`, `weight_decay`
- 대표 결과:
  - best `MRR@20=0.0243`
  - `full@1`
- 다음 phase로 넘어간 이유:
  - 목적이 성능이 아니라 logging/report sanity였기 때문
  - 이후부터 본격적인 layout screening으로 이동

### 2. Legacy layout screening

#### `P1S` on ML1

- 기간: 2026-03-05 오전~오후 UTC
- 대상: `movielens1m`
- 목표: serial / parallel layout 다건 스크리닝
- 사용 runner: [p1_wide_shallow.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p1_wide_shallow.sh)
- 당시 모델 상태: `legacy-flat`
- 탐색 축:
  - layout / execution 고정 다건
  - `lr`, `wd` 위주
  - fixed-dim shallow budget
- 대표 결과:
  - best `L7 serial`, `MRR@20=0.0975`
  - serial 쪽 상위는 `L7`, `L18`, `L16`
  - parallel 최고는 `L28`, `MRR@20=0.0968`
- 해석:
  - ML1에서는 `L7`이 가장 강했고, serial이 우세
  - 이 단계가 이후 `P2DB`에서 `serial/L7`을 anchor로 고정하게 만든 근거다.
- 증거:
  - [experiment_overview.md](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/experiment_overview.md)
  - [experiment_summary.md](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE_v2/experiment_summary.md)
  - 목적 설명은 현재 runner 기본값에서 복원했으므로 `(inferred from runner defaults + focus vars)`

### 3. Legacy dim-batch refinement

#### `P2DB` on ML1

- 기간: 2026-03-05~06 UTC
- 대상: `movielens1m`
- 목표: `P1S` best anchor 근처에서 dim / router / batch refinement
- 사용 runner: [p2_dim_batch_combo.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p2_dim_batch_combo.sh)
- 당시 모델 상태: `legacy-flat`
- 탐색 축:
  - `embedding_size`, `d_feat_emb`, `d_expert_hidden`, `d_router_hidden`, batch
  - `lr`, `wd`
- 대표 결과:
  - best `MRR@20=0.0982`
  - anchor는 계속 `serial / L7`
- 해석:
  - ML1에서는 legacy-flat 구조가 어느 정도 정착했고
  - 이후 RetailRocket 전이는 `L7` 또는 그 근처 anchor를 중심으로 시작하게 된다.

### 4. RR legacy transfer / focus

#### `P1S (ReR)`

- 기간: 2026-03-08 06:05~08:04 UTC
- 대상: `retail_rocket`
- 목표: RR로 transfer한 첫 layout screening
- 사용 로그: [P1S RR L16 best log](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ReR/FMoEv2/20260308_060500_257_hparam_P1S_G1_C7_serial_L16.log)
- 당시 모델 상태: `legacy-flat`
- 대표 결과:
  - best `L16 serial`, `MRR@20=0.2699`
  - winning trial은 `early_stop@34`
  - 같은 run 안에서도 `full@50` trial이 있었지만 peak는 early-stop trial에서 나왔다.
- 해석:
  - ML1 best였던 `L7`보다 RR에서는 `L16`이 더 강했다.
  - 즉 dataset transfer 시 layout optimum이 바뀐다는 첫 강한 신호였다.

#### `P2DB (ReR)`

- 기간: 2026-03-08 10:11~13:50 UTC
- 대상: `retail_rocket`
- 목표: RR에서 `serial/L7` 고정 dim-batch refinement
- 사용 로그: [P2DB RR serial L7](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ReR/FMoEv2/20260308_101133_510_hparam_P2DB_G1_C1_serial_L7_E160_R96_B4096.log)
- 당시 모델 상태: `legacy-flat`
- 대표 결과:
  - best `MRR@20=0.2661`
  - best trial은 `early_stop@38`
  - 일부 trial은 `full@80`까지 갔지만 peak를 넘지 못했다.
- 해석:
  - RR에서는 `L7` anchor 자체가 이미 suboptimal이었다.
  - 즉 P2를 많이 돌려도 anchor layout이 어긋나면 ceiling이 낮다는 점이 드러났다.

#### `P2RRF`

- 기간: 2026-03-08 19:14~22:04 UTC
- 대상: `retail_rocket`
- 목표: `L16` anchor 중심 RR-focused P2
- 사용 runner: [p2_rr_focus.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p2_rr_focus.sh)
- 사용 로그: [P2RRF L16 best log](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RRF/ReR/FMoEv2/20260308_191442_338_hparam_P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096.log)
- 사용 결과: [P2RRF best JSON](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rrf_l16_g4_c04_e128_f24_h160_r64_b4096_20260308_191445_359236_pid382700.json)
- 당시 모델 상태: `legacy-flat + RR narrow`
- 대표 결과:
  - best `MRR@20=0.2720`
  - winning combo는 `L16 / E128 / F24 / H160 / R64 / B4096`
  - best trial은 `full@50`
- 해석:
  - 현재까지 legacy RR 최고점은 여기서 나왔다.
  - `L16 + moderate dim + F24`가 RR에서 가장 설득력 있는 anchor로 굳어진 시점이다.

### 5. Factorized-router transition

#### `P1RFI`

- 기간: 2026-03-09 04:32~06:23 UTC
- 대상: `retail_rocket`
- 목표: factorized router 전환 직후 blocked-joint P1
- 사용 runner: [p1_rr_factorized_probe.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p1_rr_factorized_probe.sh)
- 사용 로그: [P1RFI L16 medium](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_267_hparam_P1RFI_G5_C01_serial_L16_E160_F16_H160_R80_B4096.log)
- 당시 모델 상태: `factorized-router + feature-spec`
- 탐색 축:
  - layout anchor + 일부 dim probe
  - `router_design=group_factorized_interaction`
  - `group_top_k=0`, `expert_top_k=1`
  - `feature_spec_aux=true`
- 대표 결과:
  - best `MRR@20=0.2617`
  - best trial은 `full@25`
  - 같은 run에 `0.003~0.007` 붕괴 trial도 다수 존재
- 해석:
  - 구조 전환 직후 LR/space가 너무 넓어 budget을 많이 낭비했다.
  - 이 단계는 “factorized router가 당장 legacy를 넘었다”기보다 “layout / LR band를 다시 찾는 초기 probe”로 보는 게 맞다.

#### `P1RGI2`

- 기간: 2026-03-09 07:00~11:24 UTC
- 대상: `retail_rocket`
- 목표: factorized router layout-first recovery P1
- 사용 로그: [P1RGI2 L16 base](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_807_hparam_P1RGI2_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log)
- 사용 결과: [P1RGI2 best JSON](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rgi2_g4_c00_serial_l16_e128_f16_h128_r64_b4096_20260309_070028_753184_pid473259.json)
- 당시 모델 상태: `factorized-router + feature-spec`
- 대표 결과:
  - best `MRR@20=0.2618`
  - winning combo는 `L16 + E128/F16/H128/R64`
  - best trial은 `full@25`
- 해석:
  - factorized router에서도 RR best layout은 여전히 `L16`이었다.
  - 따라서 layout 쪽은 빠르게 정리됐고, 이후 P2는 dim / capacity / small top-k probe로 넘어가게 된다.

#### `P2RGI`

- 기간: 2026-03-09 12:33~15:26 UTC
- 대상: `retail_rocket`
- 목표: factorized router narrow P2
- 사용 runner: [p2_rr_factorized_dim.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p2_rr_factorized_dim.sh)
- 사용 로그:
  - [P2RGI L16 base](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_123317_232_hparam_P2RGI_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log)
  - [P2RGI L16 F24](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_123317_233_hparam_P2RGI_G6_C02_serial_L16_E128_F24_H160_R64_B4096.log)
- 사용 결과:
  - [P2RGI L16 base JSON](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rgi_g4_c00_serial_l16_e128_f16_h128_r64_b4096_20260309_123320_144536_pid506475.json)
  - [P2RGI L16 F24 JSON](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rgi_g6_c02_serial_l16_e128_f24_h160_r64_b4096_20260309_123320_184609_pid506474.json)
- 당시 모델 상태: `factorized-router + feature-spec`
- 대표 결과:
  - global best `MRR@20=0.2621` on `L16 base`
  - exact P3 seed best `MRR@20=0.2619` on `L16 F24`
  - `L16 base` best trial은 `early_stop@30`
  - `L16 F24`는 한 번 `full@40`이 있었고 최종 best는 `early_stop@35`
- 해석:
  - factorized router가 아직 legacy RR best `0.2720`을 못 넘고 있다.
  - 그래도 `L16 F24` seed는 장기 horizon P3 router-teaching 실험의 시작점으로는 가장 타당했다.

### 6. Router-teaching phase

#### `P3RGI`

- 상태: script exists, but no executed logs/results found in current workspace
- 사용 runner: [p3_rr_factorized_router.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p3_rr_factorized_router.sh)
- 분류: `planned, not yet evidenced by logs/results`
- 의도:
  - best layout+dim seed 위에서 `expert_top_k`, feature-spec, distill을 짧게 ablation

#### `P3RRT`

- 기간: 2026-03-09 15:31~21:08 UTC
- 사용 runner: [p3_rr_router_teach.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p3_rr_router_teach.sh)
- 사용 로그 / 결과:
  - [P3RRT C02 log](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P3RRT/ReR/FMoEv2/20260309_153111_053_hparam_P3RRT_ROUTER_G6_C02_L16F24_L16_E128_F24_H160_R64_B4096_K2_D0_SMID_U20.log)
  - [P3RRT C10 log](/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P3RRT/ReR/FMoEv2/20260309_210319_092_hparam_P3RRT_ROUTER_G6_C10_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMID_U20.log)
  - [P3RRT best JSON (C11)](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p3rrt_router_g7_c11_l16f24_l16_e128_f24_h160_r64_b4096_k2_d5_smm_u20_20260309_205815_537261_pid528926.json)
- 당시 모델 상태: `factorized-router + router-teach`
- 대표 결과:
  - best `MRR@20=0.2644` (`C11`, `K2 + D5 + SMM + U20`)
  - next `0.2641` (`C10`), `0.2638` (`C09`, `C02`)
  - exact seed baseline `P2RGI L16 F24 = 0.2619` 대비는 개선, legacy RR best `0.2720` 대비는 여전히 미달
  - 12개 combo 전부 `full@100`은 못 갔고, best권도 `early_stop@47~61`에서 멈춤
- 해석:
  - `expert_top_k=2`가 가장 강한 양의 신호였다.
  - `distill`은 `K1`에서는 약한 보조 효과만 있었고, `K2` 위에 얹힐 때만 약간 도움이 됐다.
  - `distill_until=0.35`와 `feature_spec_lambda=7e-4`는 좋지 않았다.
  - 이때 teacher는 hidden을 보지 않고 stage feature만으로 만든 `4-group rule_soft` logits를 학생의 `group_logits_raw`에 맞추는 coarse supervision이었다.
  - 즉 “teacher를 더 세게”보다 “router semantics 자체를 다시 보자”는 결론이 나왔다.

#### `P3RRTS`

- 상태: runner 추가 완료, 아직 result JSON 없음
- 사용 runner: [p3_rr_router_teach.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p3_rr_router_teach.sh)
- 분류:
  - `planned, not yet evidenced by logs/results`
  - `not yet summarized by experiment_overview`
- 의도:
  - `P3RRT`에서 `K2 > K1`는 확인됐지만, 원래 의도였던 dense-within-group (`expert_top_k=0`)은 실제로 한 번도 비교하지 않았기 때문에 semantics 자체를 다시 본다.
  - 그래서 `distill`을 전부 빼고 `expert_top_k=0/1/2`와 약한 feature-spec만 비교하는 `semantics12` profile을 추가했다.
- 왜 바꿨는가:
  - `P3RRT`의 핵심 교훈은 `distill`보다 `top-k semantics` 영향이 컸다는 점이다.
  - 따라서 다음 단계는 teacher loss를 더 세게 거는 것이 아니라, “group은 전부 유지하면서 clone을 dense하게 섞을지 (`K0`) / 하나만 고를지 (`K1`) / 둘을 고를지 (`K2`)”를 직접 비교하는 쪽으로 수정됐다.
- 어떻게 바꿨는가:
  - [p3_rr_router_teach.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v2/p3_rr_router_teach.sh)에 `--catalog-profile semantics12` 추가
  - `K0/K1/K2` 각각 4 combo씩 총 12 combo
  - `distill=false` 고정
  - `feature_spec_lambda`는 `1e-4` 또는 `3e-4`
  - `feature_spec_stages`는 `[mid]` 또는 `[mid,micro]`
  - 즉 stage feature 기반 teacher 자체를 바꾸지 않고, student routing semantics만 다시 비교하는 방향으로 수정했다.
  - 이 단계는 “router teaching”보다 “router semantics probe”에 가깝다.

## 현재 시점 해석

- legacy RR 최고점은 [P2RRF best JSON](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rrf_l16_g4_c04_e128_f24_h160_r64_b4096_20260308_191445_359236_pid382700.json) 의 `MRR@20=0.2720`
- factorized-router RR 최고점은 현재 [P2RGI L16 base JSON](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rgi_g4_c00_serial_l16_e128_f16_h128_r64_b4096_20260309_123320_144536_pid506475.json) 의 `MRR@20=0.2621`
- router-teach 포함 factorized 최고점은 [P3RRT best JSON (C11)](/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p3rrt_router_g7_c11_l16f24_l16_e128_f24_h160_r64_b4096_k2_d5_smm_u20_20260309_205815_537261_pid528926.json) 의 `MRR@20=0.2644`
- 즉 현재 해석은 다음과 같다.
  - `layout`은 이미 큰 문제가 아니다. RR에서는 legacy와 factorized 모두 `L16`이 강하다.
  - 병목은 `layout`보다 `router learning / routing semantics` 자체다.
  - `K2`는 유의미한 개선 신호였지만, `distill`은 아직 main driver가 아니다.
  - 그래서 다음 phase는 “teacher를 더 강하게”보다 “`K0/K1/K2` semantics를 직접 비교”하는 방향으로 이동했다.

## 메모

- 이 문서는 현재 workspace 상태를 기준으로 썼다.
- 특히 구조 변경 설명은 `snapshot 커밋 이후의 working tree`를 기준으로 한다.
- 추후 `P3RRTS` result JSON과 `experiment_overview` 갱신이 생기면, 마지막 두 섹션만 다시 갱신하면 된다.
