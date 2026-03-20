# FMoE N3 KuaiRec Strict Master Handoff

Updated: 2026-03-19
Scope: `KuaiRecLargeStrictPosV2_0.2` + `FeaturedMoE_N3` (`core -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7`)
Primary metric: `best valid MRR@20`
Secondary metrics: `test MRR@20`, `cold_item_mrr@20`, routing diagnostics (`n_eff`, `cv_usage`, `top1_max_frac`, `entropy`, `route_consistency_*`)

이 문서는 다음 모델/연구자가 코드베이스를 직접 열지 않아도 현재 상태를 이해하고, 다음 실험과 논문 서술을 바로 시작할 수 있도록 만든 canonical handoff입니다.

Legend:
- `Observed fact`: 로그/결과 파일에서 직접 확인된 사실
- `Interpretation`: 사실을 설명하기 위한 가설적 해석

---

## 0) Executive Summary

- `Observed fact`: KuaiRec 기준 FMoE_N3는 baseline(SASRec `0.0785`)에서 phase 진행과 함께 `~0.0814` 권역까지 상승했고, test는 `~0.162x` 영역을 안정적으로 형성했다.
- `Observed fact`: P7에서 최고 valid는 `AUX_R0_STD_BAL_B` / `AUX_R0_STD_SPEC_A` (`valid_best_mrr20_max=0.0814`)이고, test 고점은 `R5_FAC_GROUP` (`test_mrr20_max=0.1624`)에서 나왔다.
- `Observed fact`: P6에서 `feature-only MoE(B4)`는 `B0(SASRec-eq)` 대비 `cold_item_mrr@20 +0.0126` 개선을 보였다.
- `Observed fact`: P7에서 `route_consistency_knn_score`가 높을수록 valid가 좋아지는 패턴이 아니며, 오히려 전체 상관은 음수(`pearson r=-0.4874`)였다.
- `Observed fact`: aux balance가 router entropy/balance를 단조롭게 개선하지 않았고, R0 anchor에서는 `none`이 `balance`보다 entropy mean이 더 높고(`1.958 > 1.912`) test mean도 더 높았다(`0.162125 > 0.162013`).
- `Interpretation`: 현재 데이터에서는 “balance 강제”보다 “적당한 specialization + 과도한 일관성 강제 회피 + jitter 제어”가 더 설명력이 높다.

---

## 1) Current Model Configuration and Tunable Axes

### 1.1 Current stack (high level)

- Base backbone: stage layout 중심 MoE (`macro/mid/micro`) + feature-aware routing.
- Router variants: `standard`, `factored`, `hir`, `fac_group` 계열.
- Router source axis: `hidden`, `feature`, `both`.
- Injection axis: `gated_bias`, `group_gated_bias`, `none`.
- Top-k scope axis: `global_flat`, `group_dense`, `group_top2_pergroup`.
- Residual axis: `base`, `shared_moe_learned_warmup`, 기타 residual variants.
- Aux/regularization axis: `balance/z`, `smoothness`, `consistency`, `sharpness+monopoly`, `prior/group-balance` 조합.

### 1.2 Axis usage and quality signal

Usage label rule: `Frequent >=40%`, `Occasional 15~39%`, `Rare <15%`.

Reference for usage ratio:
- P6 observed combos: 55 (launcher-defined settings + latest dedup)
- P7 observed runs: 64 (16 settings x 4 seeds, latest dedup)

| Axis | Values | Usage label | Signal on performance | Notes |
| --- | --- | --- | --- | --- |
| `stage_router_type` | `standard`, `factored` | P7: `factored` Frequent(68.8), `standard` Occasional(31.2) | P7 top-quartile valid의 95.2%가 `standard` | 탐색 비중은 factored가 높았지만, P7 고성능 구간은 standard dominance |
| `stage_router_source` | `both`, `feature`, `hidden` | P6: `both` Frequent(89.1), `hidden` Rare(7.3), `feature` Rare(3.6). P7: `both` Frequent(50.0), `feature` Occasional(37.5), `hidden` Rare(12.5) | P7 mean(valid): `both(0.080956) > feature(0.080771) > hidden(0.080275)` | `hidden-only`는 일관되게 약세. 질문 예시와 동일하게 `both`가 실전 기본값 |
| `stage_feature_injection` | `gated_bias`, `group_gated_bias`, `none` | P6: `group_gated_bias` Frequent(65.5), `gated_bias` Occasional(23.6), `none` Rare(10.9). P7: `group_gated_bias` Frequent(68.8), `gated_bias` Occasional(31.2) | P7 top-quartile의 95.2%가 `gated_bias` | P7에서 standard+gated_bias가 강함 |
| `topk_scope_mode` | `global_flat`, `group_dense`, `group_top2_pergroup` | P6: `group_dense` Frequent(61.8), `global_flat` Occasional(30.9), `group_top2_pergroup` Rare(7.3). P7: `group_dense` Frequent(68.8), `global_flat` Occasional(31.2) | P7 mean(valid): `global_flat(0.081225) > group_dense(0.080609)` | P7 고성능군은 global_flat 쏠림 |
| `stage_residual_mode` | `base`, `shared_moe_learned_warmup`, 기타 | P6: `base` Frequent(72.7), `warmup` Rare(12.7) | P4/P6 모두 `base`가 안정 기준선, warmup은 특정 컨텍스트에서 보완재 | `shared_only`는 P4에서 명확 열화 |
| `macro_history_window` | `NA(default)`, `5`, `10` | P6: `NA` Frequent(49.1), `5` Occasional(30.9), `10` Occasional(20.0) | P6 mean(valid): `NA(0.080719) > 5(0.079865) > 10(0.079509)` | feature ablation 구간에서만 창 길이 실험이 집중됨 |
| `stage_feature_family_mask` size | `1-family`, `2-family`, `NA` | P6: `NA` Frequent(63.6), `2-family` Occasional(21.8), `1-family` Rare(14.5) | feature_ablation 내부에선 `2-family`가 `1-family` 대비 우세 | 전역 최고점은 mask 실험 외 구간에서 생성됨 |
| `aux family` | `none`, `balance`, `specialization` | P7 anchor-conditional 실험에서 본격 비교 | R0 anchor: balance/spec이 valid를 소폭 올리지만 test는 `none`이 가장 높음. R2 anchor: balance/spec 효과 약하고 불안정 | aux는 “전역 1등”이 아니라 anchor-conditional 해석 필수 |

### 1.3 Practical defaults now

- Default router source는 `both`.
- Default weak anchor는 `hidden-only` (특히 high-quality zone 진입 어려움).
- P7 high-valid zone default recipe는 `standard + both + gated_bias + global_flat`.

---

## 2) Recommended Settings (5 candidates)

Selection rule:
- 1순위 `valid_best_mrr20_mean/max`
- 2순위 `test_mrr20_mean/max`
- 3순위 안정성(분산/seed 일관성)

### 2.1 Candidate leaderboard snapshot (P7)

| Candidate | Group | valid_best_mrr20_mean | valid_best_mrr20_max | test_mrr20_mean | test_mrr20_max | When to use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `AUX_R0_STD_BAL_B` | aux_reg | 0.081325 | 0.0814 | 0.162000 | 0.1622 | 현재 valid 우선 기본 추천 |
| `AUX_R0_STD_SPEC_A` | aux_reg | 0.081325 | 0.0814 | 0.161950 | 0.1621 | specialization 내러티브 유지 + valid high |
| `R0_STD` | router_core | 0.081150 | 0.0812 | 0.162125 | 0.1622 | aux 없는 clean baseline anchor |
| `R5_FAC_GROUP` | router_core | 0.080600 | 0.0806 | 0.162125 | 0.1624 | test 고점/리스크 테이킹 실험용 |
| `R2_FAC_HEAVY` | router_core | 0.080975 | 0.0811 | 0.161500 | 0.1616 | factored-heavy family의 비교 기준점 |

### 2.2 Key overrides per candidate

#### `AUX_R0_STD_BAL_B`

- `stage_router_type={macro,mid,micro: standard}`
- `stage_router_source={macro,mid,micro: both}`
- `stage_feature_injection={macro,mid,micro: gated_bias}`
- `topk_scope_mode=global_flat`, `moe_top_k=0`
- `balance_loss_lambda=0.006`, `z_loss_lambda=3e-4`
- `route_smoothness/consistency/sharpness/monopoly/prior = 0`

#### `AUX_R0_STD_SPEC_A`

- Router/injection/topk는 `AUX_R0_STD_BAL_B`와 동일
- `balance_loss_lambda=0.002`, `z_loss_lambda=1e-4`
- `route_smoothness_lambda=0.04`
- 기타 specialization loss 항은 off

#### `R0_STD`

- Router/injection/topk는 동일 (`standard+both+gated_bias+global_flat`)
- base aux: `balance=0.002`, `z=1e-4`, `route_smoothness=0.01`
- 기타 aux 항은 off

#### `R5_FAC_GROUP`

- `stage_router_type=factored`, `stage_router_source=both`
- `stage_feature_injection=group_gated_bias`
- `stage_factored_group_router_source=feature`
- `stage_factored_combine_mode=fac_group`
- `stage_factored_group_logit_scale=1.0`, `intra_logit_scale=1.0`
- `topk_scope_mode=group_dense`, `moe_top_k=0`
- `group_prior_align_lambda=5e-4`, `factored_group_balance_lambda=1e-3`

#### `R2_FAC_HEAVY`

- `stage_router_type=factored`, `stage_router_source=feature`
- `stage_feature_injection=group_gated_bias`
- `stage_factored_group_router_source=both`
- `stage_factored_group_logit_scale=1.6`, `intra_logit_scale=1.0`
- `stage_factored_combine_mode=add`
- `topk_scope_mode=group_dense`, `moe_top_k=0`
- `group_prior_align_lambda=5e-4`, `factored_group_balance_lambda=1e-3`

### 2.3 Why these 5

- `Observed fact`: high-valid 관점에서는 R0-standard anchor가 가장 강함.
- `Observed fact`: test absolute peak는 factored (`R5_FAC_GROUP`)에서 나와, 논문/후속 실험에서 “high-risk high-reward” 비교축 가치가 있음.
- `Interpretation`: 실전 운영은 `R0` 계열 2~3개를 mainline으로 두고, `R5/R2`를 contrastive branch로 유지하는 것이 합리적.

---

## 3) Timeline by Phase (Goal -> Experiment -> Result -> Interpretation)

| Phase | Goal | What changed | Key result (KuaiRec) | Interpretation |
| --- | --- | --- | --- | --- |
| Baseline | 절대 기준선 확보 | SASRec best 확보 | `best 0.0785`, `test 0.1597`, `HR@10 0.1859` | 이후 모든 개선폭의 기준점 |
| Core (`core_ablation_v2`) | 기본 지형 파악 | moe on/off, router source, injection, top-k 등 | `C70 best 0.0801`, `test 0.1614` | `feature-only` collapse, `both`가 안전한 상위권 |
| P1 (`phase1_upgrade_v1`) | anchor 재현 + 확장 | A/F/L/N/S/X 계열 확장 | `A05 best 0.0811`, `test 0.1620~0.1622` | core 신호가 실제 재현됨 |
| P2 (`phase2_router_v1`) | router/feature/aux 분해 | A/B/C/D block | `PA05 best 0.0811`, `test 0.1620` | 구조 단독보다 “구조 x hyper” 상호작용 큼 |
| P3 (`phase3_focus_v1`) | 유망 구조 집중 | S1/S2/S3/S4 집중 탐색 | `P3S1_05 best 0.0813` | standard/factored 둘 다 경쟁 가능 |
| P4 (`phase4_residual_topk_v2`) | residual/K/F 분해 | R/K/F axis 분리 실험 | valid 최고 `0.0814`, test 최고 `0.1623(group_dense C3)` | `shared_only` 열화, `base` 안정. K/F 축은 specialization 근거 강화 |
| P5 (`phase5_specialization_v1`) | specialization reg 탐색 | M0~M4 x combo | 방법별 최고 test `M1=0.1623`, jitter와 test 음상관 | balance 강제보다 jitter/specialization trade-off가 중요. 단, 일부 집계/ablation 결손 주의 |
| P6 (`phase6_candidate_reinfor_v2`) | 후보 재강화 + bridge + axis 확장 | candidate/base/router/spec/feature suites | valid 최고 `0.0818(cand_c_s1)` but test `0.1586`; B4 cold +0.0126 | valid 최고=실전 최고 아님. cold/item 관점에서 feature specialization의 가치 확인 |
| P7 (`phase7_router_aux_v1`) | router vs aux 조건부 해석 | router 8종 + aux 8종 (64 runs) | valid 최고 `0.0814`, test 최고 `0.1624` | aux 효과는 anchor-conditional. high KNN consistency가 항상 좋은 것 아님 |

---

## 4) Feature Ablation / Special Logging / Diag Consolidation

### 4.1 Cold-item improvement evidence (strong)

From P6 baseline bridge (`B0/B2/B4`):

| Model | best val MRR@20 | test MRR@20 | cold_item_mrr@20 |
| --- | ---: | ---: | ---: |
| `B0` (SASRec-eq) | 0.0782 | 0.1588 | 0.1087 |
| `B2` (hidden-only MoE) | 0.0793 | 0.1601 | 0.1110 |
| `B4` (feature-only MoE) | 0.0807 | 0.1617 | 0.1213 |

Delta vs `B0`:
- `B4`: `+0.0025` (best val), `+0.0029` (test), `+0.0126` (cold item)

`Observed fact`: feature 기반 MoE는 특히 cold item slice에서 강한 개선을 보임.

### 4.2 KNN/consistency metric definitions (code-grounded)

Code base: `experiments/models/FeaturedMoE_N3/diagnostics.py`

- `route_consistency_knn_score`
  - 세션 표현(`feat` 평균)으로 KNN neighbor를 만든 뒤,
  - full expert routing distribution 간 JS divergence 계산,
  - score는 `exp(-JS)`로 변환.
- `route_consistency_group_knn_score`
  - 같은 neighbor graph를 사용하되,
  - expert 전체가 아니라 group-level distribution(JS)으로 계산.
- `route_consistency_intra_group_knn_mean_score`
  - group별로 intra-group expert distribution JS를 계산,
  - group 평균을 사용.
- `route_consistency_feature_group_knn_*`
  - family별 feature subset으로 neighbor graph를 다시 만들고,
  - full expert routing JS를 계산.

요약 차이:
- `knn_score`: 전체 expert routing의 local smoothness.
- `group_knn_score`: group assignment level smoothness.
- `intra_group_knn_mean_score`: group 내부 expert 선택 smoothness.

### 4.3 P7 diag-metric vs valid relationship

From `results_phase7.md` (`n=168 run-stage samples`):

| Metric | Pearson r with valid_best_mrr20 |
| --- | ---: |
| `route_consistency_knn_score` | -0.4874 |
| `top1_max_frac` | -0.3289 |
| `n_eff` | -0.1181 |
| `cv_usage` | +0.1355 |
| `route_jitter_adjacent` | +0.0404 |
| `family_top_expert_mean_share` | +0.1830 |

`Observed fact`: consistency score를 높인다고 자동으로 성능이 오르지 않음.

### 4.4 Aux vs entropy/balance paradox (macro anchor view)

P7 macro-stage, anchor-conditional aggregation (`diag_best_valid_overview` + run summaries):

| Anchor | Aux family | best_mean | test_mean | entropy_mean | n_eff | cv_usage | top1_max_frac | knn_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard(R0)` | `none` | 0.081150 | 0.162125 | 1.958166 | 11.4049 | 0.2263 | 0.2136 | 0.9597 |
| `standard(R0)` | `balance` | 0.081275 | 0.162013 | 1.911705 | 11.2427 | 0.2577 | 0.2435 | 0.9535 |
| `standard(R0)` | `specialization` | 0.081213 | 0.161875 | 1.291759 | 7.6014 | 0.8504 | 0.3134 | 0.9462 |
| `factored-heavy(R2)` | `none` | 0.080975 | 0.161500 | 1.673392 | 11.3347 | 0.2417 | 0.1611 | 0.9520 |
| `factored-heavy(R2)` | `balance` | 0.080937 | 0.161612 | 1.664222 | 11.3777 | 0.2334 | 0.1588 | 0.9521 |

핵심:
- `Observed fact`: R0에서 balance가 valid는 소폭 올리지만 test는 none보다 낮다.
- `Observed fact`: entropy는 none이 오히려 더 높다(R0 기준).
- `Observed fact`: specialization은 entropy/n_eff를 크게 낮추고(top1 concentration 증가), test 개선으로 직결되지 않는다.
- `Interpretation`: 단순 balance 또는 단순 sharpening이 아니라, task-relevant specialization quality를 직접 겨냥해야 한다.

### 4.5 Phase5 logging caveat

- `Observed fact`: P5는 `feature_ablation_file` 공백/불일치 이슈가 있어, 해당 phase의 ablation 결론은 “경향 증거”로만 사용해야 함.
- 직접 원인 정리는 `260317_summary.md`의 `1.7-b` 참고.

---

## 5) Initial Intuition vs Actual Observation: Hypothesis Revision

### 5.1 What was expected

- 초기 의도: group-feature를 잘 분류할수록(expert weight가 feature-neighbor에 대해 일관적일수록) 성능이 올라야 한다.
- 기대 부가효과: aux balance가 usage imbalance를 완화하고 성능/안정성을 함께 올려야 한다.

### 5.2 What actually happened

- `Observed fact`: P7에서 `knn_score`, `group_knn_score`, `intra_group_knn_mean_score`가 높을수록 valid가 좋아지는 패턴이 아님.
- `Observed fact`: aux balance가 entropy/balance 지표를 단조롭게 개선하지 않음.
- `Observed fact`: aux 없음(`none`)에서도 entropy가 더 높은 경우가 있고, test가 더 좋은 경우도 존재.

### 5.3 Alternative hypotheses (for next experiments)

- `H1 (soft specialization > hard consistency)`:
  - KuaiRec에서는 “유사 feature면 동일 expert” 강제가 과도하면 정보 병목이 생길 수 있음.
- `H2 (metric-target mismatch)`:
  - 현재 consistency는 neighbor-smoothness를 측정하지만, 추천 품질과 직접 연결되는 구조(예: rank-sensitive gating quality)를 반영하지 못할 수 있음.
- `H3 (anchor-dependent aux dynamics)`:
  - 같은 aux라도 router anchor(standard vs factored-heavy)에 따라 작동점이 달라짐.
- `H4 (useful diversity)`:
  - 일정 수준의 routing diversity(낮은 consistency)가 오히려 multi-intent session에 유리할 수 있음.

### 5.4 Next-step experiment design (decision-ready)

1. `Anchor-conditional aux sweep`:
- R0, R2를 분리해 aux를 튜닝하고 절대 비교를 금지.

2. `Consistency objective redesign`:
- 현재 KNN consistency를 loss로 직접 올리는 대신, rank-aware or slice-aware objective(예: cold-item weighted objective)로 전환.

3. `Dual target`:
- global metric(valid/test) + slice metric(cold/session) 동시 최적화로 목적함수 재설계.

4. `Jitter-aware regularization`:
- 과도한 static consistency 대신 session-level harmful jitter만 억제하는 방향(현재 P5/P6 신호와 정합).

5. `Paper message framing`:
- “Consistency maximization”이 아니라 “useful specialization under controlled dynamics”로 메시지 전환.

### 5.5 Paper-writing skeleton (recommended)

- Claim 1: Feature-aware MoE improves hard slices (cold item) with clear quantitative gains.
- Claim 2: Strong routing consistency is not a universal proxy for recommendation quality.
- Claim 3: Aux effects are anchor-conditional; balance-specialization trade-off must be modeled conditionally.
- Claim 4: Best practice is controlled specialization (not maximal equalization, not maximal clustering).

---

## Appendix A) Data provenance and source map

Primary docs:
- `experiments/run/fmoe_n3/docs/260317_summary.md`
- `experiments/run/fmoe_n3/docs/260319_results.md`
- `experiments/run/fmoe_n3/docs/results_phase1.md`
- `experiments/run/fmoe_n3/docs/results_phase2.md`
- `experiments/run/fmoe_n3/docs/results_phase4.md`
- `experiments/run/fmoe_n3/docs/results_phase5.md`
- `experiments/run/fmoe_n3/docs/results_phase6.md`
- `experiments/run/fmoe_n3/docs/results_phase6_v2.md`
- `experiments/run/fmoe_n3/docs/results_phase7.md`
- `experiments/run/fmoe_n3/docs/plan_phase6.md`
- `experiments/run/fmoe_n3/docs/plan_phase7.md`

Key code paths for metrics/definitions:
- `experiments/models/FeaturedMoE_N3/diagnostics.py`
- `experiments/hyperopt_tune.py`
- `experiments/models/FeaturedMoE/special_metrics.py`
- `experiments/run/fmoe_n3/run_phase6_candidate_reinfor.py`
- `experiments/run/fmoe_n3/run_phase7_router_aux.py`

Operational artifacts used:
- P6/P7 logging bundles under `experiments/run/artifacts/logging/fmoe_n3/...`
- P6/P7 normal results under `experiments/run/artifacts/results/fmoe_n3/normal/...`
- Notebook logic reference: `experiments/run/fmoe_n3/docs/visualization_phase7.ipynb`

---

## Appendix B) Aggregation and labeling rules

- P7 dedup: 동일 `run_phase`의 최신 완료 run 우선.
- P6 dedup: launcher-defined combo/run_phase 기준 최신 완료 run 우선.
- `frequent/occasional/rare` 라벨:
  - `frequent >= 40%`
  - `occasional 15~39%`
  - `rare < 15%`
- 본문에서 상충되는 지표는 반드시 `Observed fact`와 `Interpretation`으로 분리 기술.

---

## Appendix C) Caveats

- P5는 일부 집계 불일치 및 feature-ablation 로그 공백이 있으므로, 강한 결론은 P4/P6/P7 근거 중심으로 해석.
- P7 diag overview 누락 run(8개)이 있어, diag 기반 결론은 `n=56` 기준임을 명시.
- `valid best`와 `test best`가 분리되는 케이스(P6 `cand_c_s1`)가 있으므로, 운영 채택은 multi-criteria로 수행.

