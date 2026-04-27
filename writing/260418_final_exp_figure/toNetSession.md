# RouteRec Q2-Q5 Handoff Note

## 이 문서의 역할

이 문서는 `Q2~Q5 figure를 채우기 위해 다음 세션이 바로 이어서 무엇을 해야 하는지`를 설명하는 handoff note다.

읽는 순서는 아래처럼 잡았다.

1. Q2~Q5가 각각 무엇을 증명하려는지
2. 지금까지 그 목적을 위해 무엇을 준비했는지
3. case subset이 왜 필요했고, pure / permissive가 정확히 무엇인지
4. 만들어 둔 dataset / manifest / 통계가 어디에 있는지
5. 아직 부족한 logging / wrapper가 무엇인지

즉 이 문서는 subset 자체를 설명하려는 문서가 아니라, `Q2~Q5 실험 계획 문서`이고, case subset 설명은 그 계획을 실행하기 위한 준비물로 들어간다.

기본 전제:

- 주 실험 계열은 `fmoe_n4`
- 기본 baseline 비교는 `baseline_2`
- 기본 데이터 루트는 `Datasets/processed/feature_added_v4`
- 기본 selection / 해석 우선순위는 `overall_seen_target`

핵심 참고 문서:

- [writing/ACM_template/sample-sigconf.tex](/workspace/FeaturedMoE/writing/ACM_template/sample-sigconf.tex)
- [writing/260418_final_exp_figure/q2_q5_main_body_strategy.md](/workspace/FeaturedMoE/writing/260418_final_exp_figure/q2_q5_main_body_strategy.md)
- [writing/260418_final_exp_figure/experiment_figure_plan.md](/workspace/FeaturedMoE/writing/260418_final_exp_figure/experiment_figure_plan.md)
- [experiments/tools/build_case_eval_subsets.py](/workspace/FeaturedMoE/experiments/tools/build_case_eval_subsets.py)
- [experiments/tools/finalize_case_eval_subsets.py](/workspace/FeaturedMoE/experiments/tools/finalize_case_eval_subsets.py)

## 먼저 큰 그림: Q2~Q5에서 무엇을 하려는가

현재 main-body figure 질문은 아래 네 개다.

1. Q2: routing을 무엇으로 제어해야 하는가
2. Q3: staged structure가 왜 필요한가
3. Q4: lightweight cue만으로도 충분한가
4. Q5: learned routing이 실제 behavioral regime와 align하는가

이 중 Q2~Q4는 대부분 기존 ablation 결과와 aggregate diagnostic으로 채울 수 있다. 반면 Q5와 Q2의 case-oriented panel은 `특정 behavioral regime만 따로 모아 extra eval`하는 흐름이 필요하다.

그래서 지금까지 한 준비는 크게 두 묶음이다.

1. Q2~Q4용 ablation / logging 경로 정리
2. Q5 및 case study용 case-eval subset 구축

## 지금까지 실제로 해 둔 일

이번 작업에서 한 일은 아래 순서다.

1. Q2~Q5의 본문 역할을 먼저 다시 정리했다.
핵심 방향은 [writing/260418_final_exp_figure/q2_q5_main_body_strategy.md](/workspace/FeaturedMoE/writing/260418_final_exp_figure/q2_q5_main_body_strategy.md) 에 있다.

2. six datasets의 held-out session에서 behavioral case를 실제로 mining했다.
단순 아이디어가 아니라, `memory / focus / tempo / exposure` 네 family와 `plus / minus` polarity를 기준으로 8개 그룹을 만들었다.

3. 원본 학습을 건드리지 않는 case-only eval 방식을 선택했다.
즉 원본 데이터셋으로 학습하고 원본 valid/test도 그대로 평가한 뒤, 마지막에 case subset으로 extra inference만 더 하는 구조다.

4. 이를 위해 case-eval용 dataset root를 새로 materialize했다.
처음에는 unbalanced subset을 만들었고, 그 다음 group size를 맞추는 balanced subset을 만들었다.

5. 마지막으로 permissive가 너무 커지는 문제를 줄이기 위해 final capped subset을 만들었다.
지금 문서가 참조하는 최종 artefact는 이 final capped subset이다.

## case subset이 왜 필요한가

Q5는 “RouteRec이 특정 behavior regime에서 실제로 gain을 내는가”를 봐야 한다. Q2도 aggregate 성능만이 아니라 “이런 prefix에서는 실제로 routing이 어떻게 달라지는가”를 보여주려면 regime별 케이스가 필요하다.

원본 valid/test 전체에서 이걸 바로 보려 하면 다음 문제가 있다.

- 원하는 regime를 명시적으로 분리해 비교하기 어렵다.
- group간 크기 차이가 너무 크다.
- 어떤 session이 왜 그 regime에 들어갔는지 설명하기 어렵다.

그래서 held-out session 중에서 `behaviorally 설명 가능한 case들만 골라 놓은 별도 subset`을 만들었다.

## pure / permissive는 정확히 무엇인가

질문에 대한 짧은 답부터 적으면 이렇다.

- `pure`는 원본 데이터가 아니다.
- `permissive`도 8개 case를 따로 분리한 다른 종류의 데이터라는 뜻이 아니다.
- 둘 다 `같은 8개 behavioral group`을 대상으로 한 case subset이다.
- 차이는 `그 그룹에 얼마나 강하게 속한다고 보느냐`다.

즉 정확한 의미는 아래다.

### 공통점

pure와 permissive는 둘 다 held-out session에서 뽑은 `case-only subset`이다. 둘 다 같은 8개 group을 쓴다.

- `memory_plus`
- `memory_minus`
- `focus_plus`
- `focus_minus`
- `tempo_plus`
- `tempo_minus`
- `exposure_plus`
- `exposure_minus`

### pure의 의미

`pure`는 해당 family/polarity가 더 강하고, 다른 비핵심 축의 섞임이 상대적으로 적은 session들이다.

쉽게 말하면:

- 더 극단적이다.
- 더 설명하기 쉽다.
- 본문 figure에서 “이 regime는 이런 행동이다”라고 말하기 좋다.

예를 들어 `memory_plus` pure는 repeat/reconstruction 성향이 강하고, 동시에 다른 family가 너무 튀지 않는 session 쪽에 가깝다.

### permissive의 의미

`permissive`는 같은 family/polarity 방향은 맞지만, pure만큼 극단적이지는 않은 session들이다. 그래도 다른 축이 완전히 섞여 있지는 않도록 어느 정도 balance 조건은 둔다.

쉽게 말하면:

- 방향은 같다.
- 덜 극단적이다.
- 더 일반적인 군이다.
- robustness check에 적합하다.

즉 사용 의도는 `pure에서 보인 경향이 너무 cherry-pick은 아닌가`를 permissive로 한 번 더 확인하는 것이다.

### 잘못 이해하면 안 되는 점

아래 해석은 둘 다 틀리다.

1. `pure = 원본 데이터`, `permissive = case subset`
틀리다. 둘 다 case subset이다.

2. `pure는 8개 case`, `permissive는 그 외 나머지`
틀리다. 둘 다 같은 8개 case group을 가진다.

정확한 구분은 이것뿐이다.

- `pure = 더 sharp한 regime subset`
- `permissive = 같은 방향의 더 넓은 regime subset`

## 지금 확정된 최종 case-eval 정책

처음 balanced permissive는 dataset에 따라 너무 커졌다.

- Foursquare permissive quota는 `302` 또는 `338`
- MovieLens permissive quota는 `458` 또는 `441`
- Retail permissive quota는 `1637` 또는 `1606`

이건 “학습은 원본으로 하고, 마지막에 extra inference만 한 번 더”라는 목적과 안 맞았다. 그래서 최종 정책을 아래처럼 고정했다.

- `pure`: balanced quota 유지
- `permissive`: `dataset x split x group`마다 상위 `128` session까지만 유지
- raw quota가 `128`보다 작은 dataset은 그대로 유지

즉 permissive는 크게 남아 있던 dataset만 줄였다.

실질적으로 cap이 들어간 곳:

- Foursquare permissive: `302/338 -> 128`
- MovieLens permissive: `458/441 -> 128`
- Retail permissive: `1637/1606 -> 128`

그대로 유지된 곳:

- KuaiRec permissive: test `113`, valid `62`
- Beauty permissive: test `24`, valid `23`
- LastFM permissive: test/valid `13`

## 지금 만들어 둔 artefact

### 원본 데이터

- [Datasets/processed/feature_added_v4](/workspace/FeaturedMoE/Datasets/processed/feature_added_v4)

### 최종 case-eval dataset root

- [Datasets/processed/feature_added_v4_case_eval_final_v1](/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_final_v1)

이 아래에는 두 종류의 실제 eval path가 있다.

1. tier union dataset
같은 tier 안의 8개 group을 합친 dataset

- `Datasets/processed/feature_added_v4_case_eval_final_v1/pure/<dataset>`
- `Datasets/processed/feature_added_v4_case_eval_final_v1/permissive/<dataset>`

2. tier/group dataset
특정 family/polarity group만 따로 뽑은 dataset

- `Datasets/processed/feature_added_v4_case_eval_final_v1/by_tier_group/<tier>/<group>/<dataset>`

예시 경로:

- [Datasets/processed/feature_added_v4_case_eval_final_v1/pure](/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_final_v1/pure)
- [Datasets/processed/feature_added_v4_case_eval_final_v1/permissive](/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_final_v1/permissive)
- [Datasets/processed/feature_added_v4_case_eval_final_v1/by_tier_group/pure](/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_final_v1/by_tier_group/pure)
- [Datasets/processed/feature_added_v4_case_eval_final_v1/by_tier_group/permissive](/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_final_v1/by_tier_group/permissive)

### subset 통계 / manifest

- [outputs/case_mining_v2_final/final_case_session_manifest.csv](/workspace/FeaturedMoE/outputs/case_mining_v2_final/final_case_session_manifest.csv)
- [outputs/case_mining_v2_final/final_case_group_stats.csv](/workspace/FeaturedMoE/outputs/case_mining_v2_final/final_case_group_stats.csv)
- [outputs/case_mining_v2_final/final_case_tier_summary.csv](/workspace/FeaturedMoE/outputs/case_mining_v2_final/final_case_tier_summary.csv)
- [outputs/case_mining_v2_final/final_case_tier_subset_sizes.csv](/workspace/FeaturedMoE/outputs/case_mining_v2_final/final_case_tier_subset_sizes.csv)
- [outputs/case_mining_v2_final/final_case_group_subset_sizes.csv](/workspace/FeaturedMoE/outputs/case_mining_v2_final/final_case_group_subset_sizes.csv)
- [outputs/case_mining_v2_final/final_case_all_tiers_subset_sizes.csv](/workspace/FeaturedMoE/outputs/case_mining_v2_final/final_case_all_tiers_subset_sizes.csv)
- [outputs/case_mining_v2_final/final_case_summary.md](/workspace/FeaturedMoE/outputs/case_mining_v2_final/final_case_summary.md)

## descriptor는 무엇인가

descriptor는 `이 session이 왜 이 regime에 들어갔는지 설명하는 외부 라벨`이다. learned routing output이 아니라, held-out row에서 계산한 external behavior feature 기반 label이다.

현재 descriptor 정보는 별도 파일로 분리하지 않고 manifest 안에 같이 있다.

핵심 descriptor field:

- `tier`: `pure` 또는 `permissive`
- `group`: `memory_plus`, `memory_minus`, `focus_plus`, `focus_minus`, `tempo_plus`, `tempo_minus`, `exposure_plus`, `exposure_minus`
- `family`: `memory`, `focus`, `tempo`, `exposure`
- `polarity`: `plus`, `minus`
- `selection_score`
- `core_score`
- `balance_score`
- `contamination_score`
- family score columns: `memory_plus`, `focus_plus`, `tempo_plus`, `exposure_plus`

논문에서 descriptor를 쓰는 방식:

1. case study caption
2. Q5 slice definition note
3. appendix regime summary table

즉 descriptor는 routing 결과 설명문이 아니라, routing을 검증하기 위한 외부 regime label이다.

## 지금 바로 쓸 수 있는 숫자

대표적인 최종 tier-level 크기는 아래와 같다.

- KuaiRec permissive test: `709` sessions, split의 `19.3%`
- LastFM permissive test: `98` sessions, split의 `2.6%`
- Beauty permissive test: `150` sessions, split의 `23.5%`
- Foursquare permissive test: `770` sessions, split의 `20.2%`
- MovieLens permissive test: `823` sessions, split의 `37.7%`
- Retail permissive test: `817` sessions, split의 `3.6%`

pure는 더 sharp한 regime evidence라 여전히 더 큰 편도 있고, dataset에 따라 상당히 큰 경우도 있다.

- KuaiRec pure test: `594` sessions
- LastFM pure test: `882` sessions
- Beauty pure test: `228` sessions
- Foursquare pure test: `1383` sessions
- MovieLens pure test: `2010` sessions
- Retail pure test: `8812` sessions

해석:

- pure는 main regime evidence에 적합하다.
- permissive는 robustness tier에 적합하다.
- MovieLens / Retail은 pure 전체 union보다는 group-level eval이 더 해석하기 좋다.

## Q2~Q5 notebook별 실험 설계

이 섹션의 목적은 “각 notebook이 어떤 claim을 증명하는가”, “무엇을 바꾸는 실험인가”, “어떤 로그를 읽어야 하는가”를 고정하는 것이다.

### Q2: What Should Control Routing?

Notebook:

- [writing/260418_final_exp_figure/02_q2_routing_control.ipynb](/workspace/FeaturedMoE/writing/260418_final_exp_figure/02_q2_routing_control.ipynb)

핵심 claim:

- behavior-guided routing이 hidden-only / mixed / shared FFN보다 낫다.
- 그리고 그 routing은 실제 behavior regime와 연결된다.

필요 실험:

- `RC-01_SHARED_FFN`
- `RC-02_ROUTER_SOURCE_HIDDEN`
- `RC-03_ROUTER_SOURCE_BOTH`
- `RC-04_ROUTER_SOURCE_FEATURE`

실제로 바꾸는 setting:

- `stage_router_source`
- `stage_router_mode`
- `stage_feature_injection`
- dense control의 경우 `stage_compute_mode`

기본 정의는 [experiments/run/fmoe_n4/docs/ablation/01_routing_control.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/01_routing_control.md) 기준으로 맞춘다.

panel별 입력:

1. panel (a) routing control performance
입력은 기존 ablation 결과로 충분하다.

- source: routing control ablation result table
- metric priority: `overall_seen_target`
- notebook input: dataset별 `NDCG@20`, `HR@10`, 보조로 `MRR@20`

2. panel (b) case-oriented routing evidence
이건 pure/permissive case subset과 연결된다.

- main text에는 `pure` 중심으로 representative group을 고른다.
- robustness나 appendix에는 같은 family/polarity의 `permissive`를 같이 보여준다.

중요한 점은, Q2 panel (b)를 정말 강하게 만들려면 group-level 평균만으로는 부족할 수 있다는 것이다. 대표 사례를 넣으려면 아직 아래 export가 필요하다.

- `session_id`, `item_id`, `timestamp` 기준으로 stage별 expert probability를 row-level로 남기는 per-case routing trace export

이건 현재 구현되어 있지 않다.

### Q3: Why Is the Staged Structure Effective?

Notebook:

- [writing/260418_final_exp_figure/03_q3_stage_structure.ipynb](/workspace/FeaturedMoE/writing/260418_final_exp_figure/03_q3_stage_structure.ipynb)

핵심 claim:

- gain은 단순 expert capacity가 아니라 temporal role separation에서 온다.

필요 실험:

- stage removal
- dense vs staged
- reduced-stage alternatives
- order / wrapper compressed variants

실제로 바꾸는 setting:

- `layer_layout`
- `stage_compute_mode`
- `stage_router_mode`
- 경우에 따라 `wrapper` / `order`

기본 정의는 [experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md) 기준.

panel별 입력:

1. panel (a) stage removal
2. panel (b) dense vs staged
3. panel (c) compressed order / wrapper

이건 기존 ablation 로그만으로 채울 수 있다. case subset은 필수는 아니다.

### Q4: Are Lightweight Cues Sufficient in Practice?

Notebook:

- [writing/260418_final_exp_figure/04_q4_lightweight_cues.ipynb](/workspace/FeaturedMoE/writing/260418_final_exp_figure/04_q4_lightweight_cues.ipynb)

핵심 claim:

- RouteRec의 routing control은 heavy metadata 의존이 아니라, ordinary logs 기반 cue만으로도 유지된다.

필요 실험:

- full
- `DROP_CATEGORY_DERIVED`
- `DROP_TIMESTAMP_DERIVED`
- `SEQUENCE_ONLY_PORTABLE`

실제로 바꾸는 setting:

- `stage_feature_family_mask`
- `stage_feature_drop_keywords`

기본 정의는 [experiments/run/fmoe_n4/docs/ablation/03_cue_ablation.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/03_cue_ablation.md) 기준.

panel별 입력:

1. panel (a) cue reduction performance
기존 ablation 결과로 충분하다.

2. panel (b) routing footprint without metadata
이건 `router_diag.json`이 필요하다. 여기서 stage별 `entropy_mean`, `n_eff`, `group_n_eff`, `group_entropy_mean` 등을 읽어 notebook에서 tidy하게 재정리하면 된다.

즉 Q4는 새로운 모델 로그가 꼭 필요한 것은 아니고, 현재 있는 diag를 figure-friendly CSV로 정리하는 작업이 필요하다.

### Q5: Do Routing Patterns Align with Behavioral Regimes?

Notebook:

- [writing/260418_final_exp_figure/05_q5_behavior_regimes.ipynb](/workspace/FeaturedMoE/writing/260418_final_exp_figure/05_q5_behavior_regimes.ipynb)

핵심 claim:

- RouteRec gain이 behaviorally meaningful한 regime에서 커지고,
- 그 regime에서 routing도 더 decisive하다.

여기서 pure/permissive가 직접 쓰인다.

추천 사용 원칙:

1. main text primary evidence는 `pure`
2. same family/polarity robustness는 `permissive`
3. 가능한 경우 `tier/group` dataset으로 group별 extra eval을 직접 돈다.

실험 구조:

1. base model은 원본 dataset으로 학습한다.
2. 원본 valid/test도 그대로 평가한다.
3. 마지막에 같은 checkpoint로 `by_tier_group/<tier>/<group>/<dataset>` subset에 대해 extra eval만 수행한다.

여기서 모아야 하는 값은 두 종류다.

1. slice-wise ranking quality
- `special_metrics`의 `overall_seen_target` 계열 metric
- 최소한 `NDCG@20`, `HR@10`, `MRR@20`

2. slice-wise routing concentration / selectivity
- `router_diag`의 stage별 `entropy_mean`, `n_eff`, `group_n_eff`, top-share 계열

Q5에서 pure/permissive를 쓰는 방법:

- `pure`: 본문 panel용. sharper하고 descriptor가 명확하다.
- `permissive`: appendix 또는 robustness line. 같은 방향의 일반적 regime에서도 경향이 유지되는지 확인한다.

즉 Q5는 pure/permissive를 합쳐 하나로 보지 말고, 같은 group의 두 tier를 나란히 보는 구조가 가장 자연스럽다.

## 지금 구현되어 있는 로그

### 이미 있음

1. run-level special slice metrics
- file: `special_metrics.json`
- writer: [experiments/models/FeaturedMoE/run_logger.py](/workspace/FeaturedMoE/experiments/models/FeaturedMoE/run_logger.py)
- source collector: [experiments/models/FeaturedMoE/special_metrics.py](/workspace/FeaturedMoE/experiments/models/FeaturedMoE/special_metrics.py)

현재 저장되는 것:

- `valid.overall`
- `test.overall`
- `slice_metrics.valid`
- `slice_metrics.test`
- `counts`

2. run-level router diagnostics
- file: `router_diag.json`
- writer: [experiments/models/FeaturedMoE/run_logger.py](/workspace/FeaturedMoE/experiments/models/FeaturedMoE/run_logger.py)
- compact diagnostic logic: [experiments/models/FeaturedMoE_N3/diagnostics.py](/workspace/FeaturedMoE/experiments/models/FeaturedMoE_N3/diagnostics.py)

현재 사용 가능한 요약 예시:

- `entropy_mean`
- `n_eff`
- `group_n_eff`
- `group_entropy_mean`
- `consistency_score`
- `group_consistency_score`
- `family top-share` 계열 요약

3. best-only final valid/test re-eval logging
- [experiments/recbole_train.py](/workspace/FeaturedMoE/experiments/recbole_train.py) 에서 best model state를 다시 로드해 final valid/test special/diag를 남긴다.
- 따라서 best checkpoint 기준 split-level aggregate는 이미 확보 가능하다.

## 아직 구현되지 않았거나, 바로 figure에 쓰기 어려운 부분

### 1. `overall_seen_target` / `overall_unseen_target`의 run-level direct export

현재 collector는 내부적으로 `overall_seen_target`과 `overall_unseen_target`을 계산한다. 하지만 `run_logger.log_special_metrics`는 실제 저장 시 `overall`만 남기고 seen/unseen aggregate는 별도 top-level로 저장하지 않는다.

즉 notebook에서 가장 자주 쓰는 `overall_seen_target`을 run-level JSON에서 바로 읽을 수 있도록 저장 구조를 보강하는 것이 필요하다.

우선순위: 매우 높음

### 2. checkpoint-only alternate-dataset eval wrapper

현재 파이프라인은 학습이 끝난 뒤 best model state로 원본 valid/test를 재평가하는 흐름은 있다. 하지만 “이미 학습된 checkpoint 하나를 여러 case-eval dataset root에 순차적으로 태워서 extra eval만 수행하는 wrapper”는 아직 없다.

Q5와 case analysis를 빠르게 반복하려면 이 wrapper가 필요하다.

필요 기능:

- checkpoint path 입력
- base config snapshot 복원
- `data_path` / dataset root만 case-eval root로 교체
- valid/test special + diag만 수행
- 결과를 notebook-friendly manifest로 저장

우선순위: 매우 높음

### 3. per-case routing trace export

Q2 panel (b)에서 representative case를 정말 설득력 있게 보여주려면, split-level 평균이 아니라 개별 prefix의 stage별 routing probability가 필요하다.

현재 diag는 aggregate 중심이라 아래가 없다.

- `session_id`, `item_id`, `timestamp`별 stage routing probability
- case별 top expert / top group trajectory

즉 “이 repeat-heavy case에서 macro는 memory group으로 갔고 micro는 exposure로 이동했다” 같은 그림은 아직 직접 못 그린다.

우선순위: 높음

### 4. notebook용 tidy export layer

지금도 raw JSON/CSV는 많지만, notebook에서 바로 쓰기 쉬운 long-format CSV는 아직 부족하다. 특히 Q4/Q5는 아래 tidy export가 있으면 반복 작업이 크게 줄어든다.

- `run x dataset x split x stage x metric` long table
- `run x tier x group x metric` long table
- `group descriptor summary` table

우선순위: 중간

## 지금 기준 권장 실행 순서

1. Q2, Q3, Q4의 ablation figure는 기존 `fmoe_n4` ablation 결과를 먼저 정리한다.
2. 그와 별개로 Q5용 checkpoint-only case-eval wrapper를 만든다.
3. wrapper로 `pure`와 `permissive`의 `tier/group` dataset을 모두 extra eval한다.
4. 그 결과에서 quality와 router footprint를 같이 뽑아 Q5 notebook을 채운다.
5. 마지막으로, Q2 panel (b)를 representative case figure로 갈지, group-level aggregate로 갈지 결정한다.

## 현재 판단 요약

- pure/permissive subset 자체는 이제 paper용으로 쓸 수 있는 수준으로 정리됐다.
- permissive는 `128` cap으로 큰 dataset의 과도한 재평가 문제를 줄였다.
- tier union dataset은 footprint / tier-wide sanity check용으로 바로 쓸 수 있다.
- tier/group dataset은 Q5와 case panel용 direct extra eval path로 쓰는 것이 맞다.
- Q2~Q4는 대부분 기존 ablation + existing special/diag로 채울 수 있다.
- 가장 큰 남은 구현은 `checkpoint-only alternate-dataset eval wrapper`와 `per-case routing trace export`다.