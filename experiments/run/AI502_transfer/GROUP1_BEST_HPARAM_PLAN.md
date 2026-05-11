# AI502 group1 best-hparam 재실험 계획

## 목표

기존 group1은 공통 preset과 추정 LR로 transfer를 비교해서, gain이 0에 가깝거나 해석이 흐린 경우가 있었다. 이번 재실험은 기존 FMoE/RouteRec 로그에서 dataset별로 성능이 좋았던 A12 설정과 `learning_rate`를 그대로 가져오고, source와 target을 같은 target 설정으로 다시 학습한 뒤 transfer한다.

핵심 비교는 `feature_encoder`, `router_e` 중심 group router, A12 전체 router, full model transfer가 scratch 대비 어떤 차이를 만드는지 확인하는 것이다.

## 기본 원칙

- 데이터는 `/workspace/RouteRec/Datasets/processed/feature_added_v4`를 사용한다. `AI502_DATA_ROOT` 환경변수로 바꿀 수 있다.
- 모델은 A12 계열 `featured_moe_n3_tune`을 사용한다.
- LR은 기존 로그의 best params에 찍힌 값을 그대로 고정한다.
- transfer에서 source와 target은 같은 `setting_id` shape를 쓴다.
- source/native checkpoint는 한 번만 만들고 모든 transfer mode/policy에서 재사용한다.
- baseline은 같은 target, 같은 `setting_id`, 같은 seed의 native scratch 결과다.
- router 입력은 기존 우수 실험을 따라 `stage_router_source={macro:"both",mid:"both",micro:"both"}`로 둔다. 단 `stage_router_primitives` 내부 source는 A12 W5 설정 그대로 feature cue를 유지한다.

## group1 pair

| pair | target 설정 기준 | 이유 |
|---|---|---|
| `retail_rocket→beauty` | beauty best 2개 | commerce close pair에서 feature/router transfer 확인 |
| `beauty→retail_rocket` | retail best 2개 | 반대 방향에서 target LR/shape 영향 분리 |
| `foursquare→KuaiRec` | KuaiRec best 2개 | rich-context source에서 KuaiRec target transfer |
| `lastfm→KuaiRec` | KuaiRec best 2개 | rich-context/sequence 성격이 다른 source 비교 |

## 선택한 설정

| target | setting_id | prior valid MRR@20 | prior test MRR@20 | LR | 주요 shape |
|---|---|---:|---:|---:|---|
| beauty | `beauty_ab_h13_low_feat_dropout` | 0.1232 | 0.0851 | 0.00135 | emb176, router88, feat8, len40 |
| beauty | `beauty_h13_final` | 0.1189 | 0.0877 | 0.000964234 | emb176, router88, feat16, len20 |
| retail_rocket | `retail_r15_h13_width_lr_validate` | 0.3730 | 0.3730 | 0.000702904 | emb176, router88, feat24, len20 |
| retail_rocket | `retail_r10_h13_width_refine` | 0.3726 | 0.3737 | 0.000692007 | emb176, router88, feat24, len20 |
| KuaiRec | `kuairec_h14_feature_strong` | 0.1721 | 0.1695 | 0.00035 | emb256, router128, feat16, len20 |
| KuaiRec | `kuairec_h10_long_context` | 0.1706 | 0.1684 | 0.0008 | emb192, router96, feat16, len30 |

## transfer mode

| mode | 의미 |
|---|---|
| `feature_encoder_init` | cue feature encoder만 init |
| `group_router_init` | A12의 group router, 특히 `router_e` 축 확인 |
| `feature_encoder_group_router_init` | feature encoder + group router |
| `feature_encoder_a12_router_init` | feature encoder + A12 active router 전체 |
| `full_except_feature_router_init` | item embedding, feature encoder, router 제외 compatible tensor |
| `full_model_init` | compatible tensor 기준 full init |

## LR policy

| policy | 의미 |
|---|---|
| `std` | target best LR 그대로 모든 trainable param에 적용 |
| `loaded_lr_0.35` | loaded parameter만 target LR의 0.35배 |
| `loaded_lr_0.05` | loaded parameter만 target LR의 0.05배, 거의 freeze에 가까운 보수적 fine-tune |
| `freeze_loaded` | 코드에서는 지원하지만 기본 실행에서는 제외 |

기본 실행 row 수는 native 70개, transfer 720개다. 4 GPU 기준으로 먼저 native bank가 만들어진 뒤 pair별 transfer 결과가 계속 쌓인다.

## 실행

```bash
cd /workspace/FeaturedMoE/experiments/run/AI502_transfer
bash run_ai502_group1_best.sh --clean --gpus 0,1,2,3
```

실행 전 row와 명령만 확인하려면:

```bash
bash run_ai502_group1_best.sh --dry-run --gpus 0,1,2,3
```

중간에 끊겼으면 `--clean` 없이 같은 명령을 다시 실행한다. 정상 종료 로그와 result/checkpoint가 모두 있으면 skip한다.

## 결과 파일

- full row 결과: `artifacts_group1_best/analysis/group1_best_full.csv`
- pair/mode/policy 집계: `artifacts_group1_best/analysis/group1_best_summary.csv`
- markdown 요약: `result_group1_best.md`

full CSV에는 baseline 대비 gain뿐 아니라 `epochs_run`, `early_stopped`, `avg_epoch_time_sec`, `loaded_tensors`, `skipped_tensors`, `init_changed_tensors`, `train_changed_tensors`, `loaded_lr`, `loaded_lr_scale`을 남긴다. 따라서 metric gain뿐 아니라 수렴 속도와 실제 transfer 적용 여부도 같이 확인한다.

## 주의

`/workspace/FeaturedMoE/Datasets/processed` 아래에는 현재 `feature_added_v4` 원본 디렉터리가 없고, case-eval/final dataset 계열만 있다. group1 재실험은 실제 v4가 있는 `/workspace/RouteRec/Datasets/processed/feature_added_v4`를 명시적으로 사용한다.
