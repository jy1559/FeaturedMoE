# RouteRec Final Figure Experiment Plan

## 목적

이 문서는 [writing/ACM_template/sample-sigconf.tex](/workspace/FeaturedMoE/writing/ACM_template/sample-sigconf.tex)의 placeholder figure와 appendix panel을 실제 실험 항목으로 풀어 쓴 실행 계획서다.

지금 단계의 목표는 다음 두 가지다.

1. 논문에 아직 비어 있는 figure가 무엇인지 정리한다.
2. 각 figure를 채우기 위해 어떤 실험을 돌리고, 어떤 결과를 어떤 CSV/plot으로 정리해야 하는지 미리 고정한다.

이 문서는 값이 들어가기 전의 설계 문서다. 실제 숫자 확정이나 LaTeX 반영은 후속 단계에서 한다.

## 기본 원칙

- 기본 실험 계열은 `fmoe_n4`를 사용한다.
- 기본 baseline 비교는 `baseline_2` 결과를 사용한다.
- 기본 데이터 루트는 `Datasets/processed/feature_added_v4`다.
- 기본 해석 지표와 selection rule은 `overall_seen_target` 기준이다.
- short summary scalar는 기본적으로 `MRR@20`을 쓴다.
- main paper의 전체 성능 비교표는 이미 상당 부분 준비되어 있으므로, 이번 문서의 우선순위는 Q2-Q5와 appendix placeholder figure들이다.
- 가능하면 모든 main/appendix figure는 `writing/results` 아래의 기존 CSV 템플릿과 notebook 구조를 그대로 활용한다.

## 참고한 현재 레포 기준점

- paper draft: [writing/ACM_template/sample-sigconf.tex](/workspace/FeaturedMoE/writing/ACM_template/sample-sigconf.tex)
- active RouteRec runner: [experiments/run/fmoe_n4/stageA_fmoe_n4.sh](/workspace/FeaturedMoE/experiments/run/fmoe_n4/stageA_fmoe_n4.sh), [experiments/run/fmoe_n4/stageB_fmoe_n4.sh](/workspace/FeaturedMoE/experiments/run/fmoe_n4/stageB_fmoe_n4.sh), [experiments/run/fmoe_n4/stageC_fmoe_n4.sh](/workspace/FeaturedMoE/experiments/run/fmoe_n4/stageC_fmoe_n4.sh), [experiments/run/fmoe_n4/stageD_fmoe_n4.sh](/workspace/FeaturedMoE/experiments/run/fmoe_n4/stageD_fmoe_n4.sh)
- baseline summary: [experiments/run/baseline_2/docs/baseline_2_best_valid_test_tables_v4.md](/workspace/FeaturedMoE/experiments/run/baseline_2/docs/baseline_2_best_valid_test_tables_v4.md)
- ablation master plan: [experiments/run/fmoe_n4/docs/ablation/plan.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/plan.md)
- study-specific specs:
  - [experiments/run/fmoe_n4/docs/ablation/01_routing_control.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/01_routing_control.md)
  - [experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md)
  - [experiments/run/fmoe_n4/docs/ablation/03_cue_ablation.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/03_cue_ablation.md)
  - [experiments/run/fmoe_n4/docs/ablation/04_objective_variants.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/04_objective_variants.md)
  - [experiments/run/fmoe_n4/docs/ablation/05_portability_followup.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/05_portability_followup.md)
- figure/result workspace guide: [writing/results/00_results_guide.md](/workspace/FeaturedMoE/writing/results/00_results_guide.md)

## 논문에서 채워야 하는 figure 목록

### Main paper

1. Q2 routing control figure
2. Q3 stage structure figure
3. Q4 lightweight cue figure
4. Q5 behavior regime figure

### Appendix

1. extended structural ablation figure
2. objective and regularization comparison
3. routing diagnostics figure
4. appendix behavior-slice figure
5. transfer or portability figure

## Figure별 필요 실험 정리

## 1. Q2. Routing Control

### 논문에서 답하려는 질문

- router를 무엇으로 제어해야 하는가
- hidden-only gate보다 behavior-guided gate가 실제로 더 낫고 더 일관적인가

### 필요한 실험

- shared FFN baseline
- hidden-only router
- hidden + behavior mixed router
- behavior-only router

이는 [experiments/run/fmoe_n4/docs/ablation/01_routing_control.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/01_routing_control.md)의 core 4-setting으로 그대로 정의되어 있다.

### 핵심 비교군

- `RC-01` `SHARED_FFN`
- `RC-02` `ROUTER_SOURCE_HIDDEN`
- `RC-03` `ROUTER_SOURCE_BOTH`
- `RC-04` `ROUTER_SOURCE_FEATURE`

### figure panel 설계

- panel (a): routing control별 average ranking quality
  - 기본 metric은 dataset별 `test MRR@20` 또는 dataset-average `MRR@20`
  - 추천 표시는 dataset별 막대 + 오른쪽에 overall average 요약
- panel (b): feature similarity bucket 대비 routing consistency
  - route consistency 또는 route similarity를 bucket별 선 그래프로 표시
  - behavior-guided가 similarity가 높아질수록 더 일관적인 routing을 보이는지 확인

### 필요한 산출 CSV

- [writing/results/02_routing_control/02a_routing_control_quality.csv](/workspace/FeaturedMoE/writing/results/02_routing_control/02a_routing_control_quality.csv)
- [writing/results/02_routing_control/02b_routing_control_consistency.csv](/workspace/FeaturedMoE/writing/results/02_routing_control/02b_routing_control_consistency.csv)

### 원천 로그 / 집계 소스

- `experiments/run/artifacts/logs/fmoe_n4/.../ablation_routing_control...`
- diag logging output에서 consistency metric 추출

### 우선순위

- 매우 높음
- 이유: Q3-Q5보다 먼저 router signal 자체가 유효하다는 것을 세워야 이후 ablation 해석이 깔끔하다.

## 2. Q3. Stage Structure

### 논문에서 답하려는 질문

- RouteRec의 이득이 단순히 expert를 많이 넣어서 생긴 것인가
- 아니면 macro/mid/micro의 staged design이 실제로 중요했는가
- 현재 wrapper와 stage order는 임의의 구현 artifact가 아닌가

### 필요한 실험

- stage removal 3종
- single-stage / two-stage / three-stage 비교
- dense FFN replacement baseline
- order variant
- wrapper variant

이는 [experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md)에 이미 구조화돼 있다.

### 핵심 비교군

- `ST-01` `REMOVE_MACRO`
- `ST-02` `REMOVE_MID`
- `ST-03` `REMOVE_MICRO`
- `ST-04` `SINGLE_STAGE_MACRO`
- `ST-05` `SINGLE_STAGE_MID`
- `ST-06` `SINGLE_STAGE_MICRO`
- `ST-07` `DENSE_FULL_ONLY`
- `ST-10` `ORDER_MACRO_MICRO_MID`
- `ST-11` `ORDER_MID_MACRO_MICRO`
- `ST-12` `WRAPPER_ALL_W1_FLAT`
- `ST-13` `WRAPPER_ALL_W4_BXD`
- `ST-14` `WRAPPER_ALL_W5_EXD`

### figure panel 설계

- panel (a): stage removal
  - full model과 remove macro/mid/micro 3개 비교
  - dataset별 또는 overall `MRR@20`
- panel (b): dense vs staged
  - dense FFN, best single-stage, best two-stage, three-stage full 비교
  - stage count가 늘 때 성능이 어떻게 올라가는지 보여주는 요약형 bar plot 권장
- panel (c): wrapper/order variants
  - best order 2개 + wrapper 3개 정도로 압축
  - main text에서는 너무 많은 variant를 한 패널에 넣지 말고 4~5개로 제한하는 것이 좋다.

### 필요한 산출 CSV

- [writing/results/03_stage_structure/03a_stage_ablation.csv](/workspace/FeaturedMoE/writing/results/03_stage_structure/03a_stage_ablation.csv)
- [writing/results/03_stage_structure/03b_dense_vs_staged.csv](/workspace/FeaturedMoE/writing/results/03_stage_structure/03b_dense_vs_staged.csv)
- [writing/results/03_stage_structure/03c_wrapper_order.csv](/workspace/FeaturedMoE/writing/results/03_stage_structure/03c_wrapper_order.csv)

### 원천 로그 / 집계 소스

- `experiments/run/artifacts/logs/fmoe_n4/.../ablation_stage_structure...`

### 우선순위

- 매우 높음
- Q2 다음 바로 진행

## 3. Q4. Lightweight Cues

### 논문에서 답하려는 질문

- category/time 같은 richer signal을 빼도 RouteRec의 이득이 남는가
- sequence-derived portable cue만으로도 full model gain의 상당 부분을 보존하는가

### 필요한 실험

- remove category-derived cue
- remove timestamp-derived cue
- sequence-only portable cue
- 필요하면 appendix용 family subset 실험

이는 [experiments/run/fmoe_n4/docs/ablation/03_cue_ablation.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/03_cue_ablation.md)에 정의돼 있다.

### 핵심 비교군

- `CF-01` `DROP_CATEGORY_DERIVED`
- `CF-02` `DROP_TIMESTAMP_DERIVED`
- `CF-03` `SEQUENCE_ONLY_PORTABLE`
- full base
- retention 계산용 `SHARED_FFN` 또는 `DENSE_FULL_ONLY`

### figure panel 설계

- panel (a): reduced cue availability에서의 average `MRR@20`
  - `full`, `remove_category`, `remove_time`, `sequence_only`
- panel (b): gain retention
  - 공식은 spec 문서대로
  - `relative_gain = (variant - shared_ffn) / (full - shared_ffn)`
  - dataset별 retention bar 또는 overall retention dot/bar

### 필요한 산출 CSV

- [writing/results/04_cue_ablation/04a_cue_ablation.csv](/workspace/FeaturedMoE/writing/results/04_cue_ablation/04a_cue_ablation.csv)
- [writing/results/04_cue_ablation/04b_cue_retention.csv](/workspace/FeaturedMoE/writing/results/04_cue_ablation/04b_cue_retention.csv)

### 원천 로그 / 집계 소스

- `experiments/run/artifacts/logs/fmoe_n4/.../ablation_cue_family...`
- retention 계산 시 Q2/Q3의 shared FFN 기준값 재사용

### 우선순위

- 높음
- Q2, Q3 결과가 나온 뒤 수행

## 4. Q5. Behavior Regime Analysis

### 논문에서 답하려는 질문

- RouteRec 이득이 behaviorally heterogeneous regime에서 더 크게 나타나는가
- gain이 큰 regime에서 routing concentration도 함께 높아지는가

### 필요한 실험

이 항목은 현재 `fmoe_n4/docs/ablation`에 직접 runner spec이 별도로 정리돼 있지는 않다. 따라서 새 집계 스크립트 또는 분석 notebook용 전처리가 필요하다.

필요 작업은 두 단계다.

1. held-out instance를 외부 통계 기준으로 behavior slice에 배정
2. slice별로 baseline vs RouteRec 성능과 routing concentration을 집계

### 추천 slice 정의

- repeat-heavy
  - 최근 prefix 내 repeat ratio 상위 구간
- fast-tempo
  - 평균 inter-event gap 하위 구간
- narrow-focus
  - category entropy 또는 focus entropy 하위 구간
- exploration-heavy
  - switching rate, novelty rate, low-repeat/high-switch 조합 상위 구간

slice는 반드시 learned routing score가 아니라 외부 통계로 정의해야 한다.

### 핵심 비교군

- best `RouteRec(valid)` 또는 final base run
- 대표 baseline 1개 또는 2개
  - 가장 안전한 선택은 strongest baseline 1개와 shared FFN 1개
  - 논문 narrative상 권장 비교는 `RouteRec` vs `shared_ffn` vs `best baseline`

### figure panel 설계

- panel (a): slice-wise ranking quality
  - x축은 behavior slice
  - hue는 `best baseline`, `shared_ffn`, `RouteRec`
  - y축은 `MRR@20`
- panel (b): relative gain and routing concentration
  - slice별 RouteRec gain over baseline
  - 같은 plot 또는 paired plot에 routing entropy inverse, effective expert count inverse, concentration score 추가

### 필요한 산출 CSV

- [writing/results/A04_behavior_slices/A04a_slice_metrics.csv](/workspace/FeaturedMoE/writing/results/A04_behavior_slices/A04a_slice_metrics.csv)
- [writing/results/A04_behavior_slices/A04b_slice_gain_concentration.csv](/workspace/FeaturedMoE/writing/results/A04_behavior_slices/A04b_slice_gain_concentration.csv)

### 새로 필요한 구현

- behavior slice assignment script
- per-slice evaluation aggregator
- routing concentration summary extractor

### 우선순위

- 높음
- main paper에 매우 중요하지만, 앞선 ablation runner가 먼저 정리된 뒤 시작하는 것이 효율적이다.

## Appendix figure별 필요 실험

## 5. Appendix Extended Structural Ablations

### 목적

- main Q3보다 넓은 structural search를 appendix에 싣기 위함

### 포함할 후보

- reduced / shuffled / flat / random-group cue variants
- scope swap / identical-scope / layout variants
- stage-placement / ordering / router variants

### 주 소스

- [experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/02_stage_structure.md)
- Q3에서 안 실은 variant를 appendix로 이동

### 산출 방식

- 새 appendix 전용 CSV를 만들기보다, `03c_wrapper_order.csv`와 추가 summary를 합치는 방식이 효율적이다.
- 단, figure를 깔끔하게 하려면 appendix 전용 aggregate CSV를 별도 생성해도 된다.

## 6. Appendix Objective and Regularization Variants

### 목적

- route consistency, z-loss, balance loss의 역할을 분리

### 필요한 실험

- `NO_CONSISTENCY`
- `NO_ZLOSS`
- `NO_BALANCE`
- `ALL_AUX_OFF`
- `CONSISTENCY_ONLY`
- `ZLOSS_ONLY`
- `BALANCE_ONLY`
- `CONSISTENCY_PLUS_ZLOSS`

이는 [experiments/run/fmoe_n4/docs/ablation/04_objective_variants.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/04_objective_variants.md)에 정의돼 있다.

### 필요한 산출 CSV

- [writing/results/A02_objective_variants/A02_objective_variants.csv](/workspace/FeaturedMoE/writing/results/A02_objective_variants/A02_objective_variants.csv)

### figure/table 표현

- quality, consistency, stability 세 축을 나란히 비교
- appendix에서는 table + compact point plot 조합이 적합

## 7. Appendix Routing Diagnostics

### 목적

- RouteRec이 실제로 expert를 어떻게 쓰는지 보여주기 위함

### 필요한 집계

- expert usage heatmap by stage
- entropy and effective expert count summary
- stage-wise route consistency
- feature-bucket routing pattern

### 필요한 산출 CSV

- [writing/results/A03_routing_diagnostics/A03a_expert_usage.csv](/workspace/FeaturedMoE/writing/results/A03_routing_diagnostics/A03a_expert_usage.csv)
- [writing/results/A03_routing_diagnostics/A03b_entropy_effective_experts.csv](/workspace/FeaturedMoE/writing/results/A03_routing_diagnostics/A03b_entropy_effective_experts.csv)
- [writing/results/A03_routing_diagnostics/A03c_stage_consistency.csv](/workspace/FeaturedMoE/writing/results/A03_routing_diagnostics/A03c_stage_consistency.csv)
- [writing/results/A03_routing_diagnostics/A03d_feature_bucket_patterns.csv](/workspace/FeaturedMoE/writing/results/A03_routing_diagnostics/A03d_feature_bucket_patterns.csv)

### 원천 실험

- routing control
- cue ablation
- objective variants

diag logging을 켠 대표 run 몇 개만 뽑아서 appendix figure를 만들면 된다. 모든 variant를 다 실을 필요는 없다.

### 추천 포함 run

- full base
- shared FFN
- behavior-guided best
- `ALL_AUX_OFF`
- 필요하면 one challenging variant 추가

## 8. Appendix Behavior Slice Figure

### 목적

- main Q5를 더 큰 패널과 더 많은 dataset/slice로 확장

### 구현 방식

- main Q5와 같은 aggregator를 사용
- main text는 압축판, appendix는 full dataset 또는 더 세밀한 slice threshold 버전 사용

## 9. Appendix Transfer / Portability Figure

### 현재 판단

현재 paper draft는 transfer placeholder를 유지하고 있지만, 현재 레포의 `fmoe_n4` 워크플로에서는 엄밀한 transfer보다 portability follow-up이 더 현실적이다.

### 추천 해석

- appendix figure를 strict transfer로 밀기보다, direct rerun portability figure로 먼저 채우는 쪽이 안전하다.

### 필요한 실험

- source 역할은 KuaiRec-derived main result
- target은 `beauty`, `retail_rocket`
- variant는 최대 5개만 재사용
  - full base
  - shared FFN
  - sequence-only portable
  - best routing challenger 1개
  - objective simplification best 1개 선택 가능

이 계획은 [experiments/run/fmoe_n4/docs/ablation/05_portability_followup.md](/workspace/FeaturedMoE/experiments/run/fmoe_n4/docs/ablation/05_portability_followup.md)에 있다.

### 필요한 산출 CSV

- 기존 template 상으로는 [writing/results/A05_transfer/A05c_transfer_variants.csv](/workspace/FeaturedMoE/writing/results/A05_transfer/A05c_transfer_variants.csv) 사용 가능
- 다만 direct rerun임을 notes/caption에 반드시 표기

## 데이터셋 권장 사용 방식

## Main overall/Q1

- 6개 dataset 전체 유지
- `Beauty`, `Foursquare`, `KuaiRecLargeStrictPosV2_0.2`, `LastFM`, `ML-1M`, `Retail Rocket`

## Q2-Q4 ablation

- scout/confirm는 KuaiRec 우선
- 최종 figure에는 가능하면 KuaiRec만 쓰지 말고 1~3개 추가 dataset 요약도 포함하는 것이 좋다.
- 최소 추천:
  - KuaiRec: main search/testbed
  - Beauty: rich cue availability
  - Retail Rocket: sparse anonymous stream

## Q5 behavior regime

- KuaiRec, Beauty, Retail Rocket 우선
- 이유: regime variability와 cue availability 대비가 잘 난다.

## 실험 실행 우선순위

1. routing control
2. stage structure
3. cue ablation
4. objective variants
5. behavior slice aggregator 구축
6. routing diagnostics export
7. beauty / retail portability follow-up

이 순서가 좋은 이유는 다음과 같다.

- router signal이 먼저 정리돼야 stage/cue ablation 해석이 가능하다.
- stage/cue가 정리돼야 behavior regime 분석에서 무엇을 baseline으로 둘지 명확해진다.
- diagnostics는 대표 run만 뽑아도 되므로 가장 마지막에 export해도 늦지 않다.

## 지금 기준에서 바로 필요한 구현 항목

### 이미 계획이 있는 것

- routing control runner
- stage structure runner
- cue ablation runner
- objective variants runner
- portability follow-up runner

### 아직 별도 구현이 필요한 것

- behavior slice assignment / aggregation script
- routing diagnostics extraction helper
- appendix extended structural summary exporter

즉, runner 관점에서 가장 비어 있는 부분은 Q5와 appendix diagnostics 쪽이다.

## 추천 파일/산출물 구조

### 실험 로그

- `experiments/run/artifacts/logs/fmoe_n4/...`

### plot input CSV

- `writing/results/02_routing_control/...`
- `writing/results/03_stage_structure/...`
- `writing/results/04_cue_ablation/...`
- `writing/results/A02_objective_variants/...`
- `writing/results/A03_routing_diagnostics/...`
- `writing/results/A04_behavior_slices/...`
- `writing/results/A05_transfer/...`

### 최종 figure export

- `writing/results/generated_figures/...`

## paper narrative 기준으로 남아 있는 핵심 메시지

현재 main table 외에 figure로 반드시 확보해야 하는 메시지는 아래 네 가지다.

1. behavior-guided routing이 hidden-only보다 낫다.
2. staged macro/mid/micro 구조가 dense expert insertion보다 낫다.
3. lightweight cue만으로도 gain의 상당 부분이 남는다.
4. RouteRec gain과 routing sharpness가 특정 behavior regime에 집중된다.

이 네 가지가 채워지면 main paper의 placeholder는 대부분 의미 있게 대체된다.

## 최종 체크리스트

- Q2용 4-setting routing control 결과 확보
- Q3용 stage removal / stage count / wrapper-order 결과 확보
- Q4용 category-drop / time-drop / sequence-only 결과 확보
- Q5용 behavior slice aggregator 구현 및 결과 확보
- appendix용 objective variant 집계 확보
- appendix용 diagnostics export 확보
- portability follow-up을 appendix에 넣을지 여부 결정

## 권장 다음 단계

문서 작성 이후 실제 구현은 아래 순서가 가장 자연스럽다.

1. `fmoe_n4` ablation runner 4종을 먼저 만들고
2. `writing/results` CSV를 채우는 exporter를 붙이고
3. 마지막에 Q5 behavior slice 전처리/집계 스크립트를 만든다.

특히 Q5는 지금 문서 기준으로도 가장 구현 공백이 큰 항목이므로, 다음 작업의 첫 후보는 `behavior_slice_analysis` 계열 스크립트 설계다.