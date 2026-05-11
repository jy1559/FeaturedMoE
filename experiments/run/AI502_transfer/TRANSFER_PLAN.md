# AI502 Transfer Learning 실험 계획

## 목표

`feature_added_v4`의 6개 데이터셋에서 FeaturedMoE N3의 어떤 부분을 transfer하는 것이 효과적인지 확인한다. 기본 실행은 fast profile이며, `shared_3`~`shared_6` 4개 preset, seed 3개, learning rate center 1점만 사용한다. 이번 재실행은 이전 버그 산출물을 섞지 않기 위해 `artifacts`를 비운 뒤 다시 시작하고, native를 먼저 전부 만든 다음 transfer를 3개 그룹으로 나눠서 진행한다. 각 그룹은 `init -> freeze -> multihop`까지 닫고 summarize를 남기므로, 전체 종료를 기다리지 않아도 그룹 단위로 중간 결과를 바로 확인할 수 있다.

## 공통 설정

- 실행 위치: `/workspace/FeaturedMoE/experiments/run/AI502_transfer`
- 데이터 위치: `/workspace/FeaturedMoE/Datasets/processed/feature_added_v4`
- 데이터셋: `beauty`, `foursquare`, `KuaiRecLargeStrictPosV2_0.2`, `lastfm0.03`, `movielens1m`, `retail_rocket`
- 모델: `featured_moe_n3_tune`
- 기본 feature/eval: `feature_mode=full_v4`, `eval_mode=session_fixed`
- fast budget: `epochs=100`, `patience=10`, `max_evals=1`
- full budget: `epochs=100`, `patience=10`, `max_evals=3`
- fast LR: dataset별 중심값 1점
- full LR: dataset별 중심값에 `0.75x`, `1.0x`, `1.25x`
- fast seed: `1,2,3`
- full seed: `1,2,3,4,5`
- checkpoint bank: `artifacts/checkpoints/native/<dataset>/<shared_id>/seed_<s>/best.pth`
- 재실행 기본 동작: `run_ai502_transfer_all.sh` 시작 시 `artifacts/{analysis,checkpoints,hyperopt_results,logging,logs,manifests,summaries}`를 삭제한다. 기존 결과를 남기려면 `--keep-artifacts`를 붙인다.

## 재실행 순서

기본 재실행 순서는 아래와 같다.

1. `artifacts` 정리
2. native 전체 1회
3. native summarize
4. group1 `init -> summarize -> freeze -> summarize -> multihop -> summarize`
5. group2 `init -> summarize -> freeze -> summarize -> multihop -> summarize`
6. group3 `init -> summarize -> freeze -> summarize -> multihop -> summarize`

이 구조의 목적은 `native를 전부 끝낸 뒤 transfer를 pair/triplet 묶음 단위로 닫아서`, 각 그룹이 끝날 때마다 `pair_transfer_gain.csv`, `freeze_gain_loss.csv`, `multihop_direct_vs_sequential.csv`를 다시 읽을 수 있게 만드는 것이다.

## 3개 그룹 구성

이전 fast 전체 실행이 약 26.5시간이었기 때문에, 이번엔 group별 row 수를 비슷하게 맞춰서 4 GPU 기준 대략 10~12시간 안쪽으로 첫 결과가 나오도록 잡는다. row 기준 추정치라 실제 wall time은 dataset별 차이로 달라질 수 있다.

### Group 1

- 목표: 가장 가능성 높았던 high-signal pair를 먼저 보고, commerce low-context 축도 바로 확인
- pair: `lastfm→KuaiRec`, `foursquare→KuaiRec`, `beauty→retail_rocket`, `retail_rocket→beauty`
- triplet: `beauty→retail_rocket→KuaiRec`
- 예상: native 포함 약 11~12시간

### Group 2

- 목표: reciprocal rich-context transfer와 movielens target을 중간 배치로 확인
- pair: `KuaiRec→foursquare`, `KuaiRec→lastfm`, `KuaiRec→movielens1m`, `lastfm→movielens1m`
- triplet: `foursquare→KuaiRec→lastfm`
- 예상: 약 10~11시간

### Group 3

- 목표: target coverage 보강 pair와 나머지 exploratory triplet 마무리
- pair: `lastfm→foursquare`, `KuaiRec→beauty`, `retail_rocket→KuaiRec`, `beauty→lastfm`
- triplet: `lastfm→KuaiRec→movielens1m`, `KuaiRec→foursquare→retail_rocket`
- 예상: 약 11~12시간

## Shared Hparam Preset

`shared_1`~`shared_8`은 dataset 간 transfer에서 tensor shape가 충돌하지 않도록 한 run 안에서는 source와 target이 같은 shape를 쓰게 한다. preset 사이 shape는 달라도 된다. preset은 기존 좋은 결과 근처의 차원을 묶었다. 기본 fast profile은 `shared_3`, `shared_4`, `shared_5`, `shared_6`만 사용한다.

| preset | embedding | d_ff | expert | router | feat_emb | max_len | 의도 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| shared_1 | 128 | 256 | 128 | 32 | 8 | 20 | 작은 공통 baseline |
| shared_2 | 160 | 320 | 160 | 32 | 8 | 30 | foursquare 계열 길이 보강 |
| shared_3 | 192 | 384 | 192 | 32 | 16 | 20 | beauty 계열 안정형 |
| shared_4 | 192 | 384 | 192 | 64 | 8 | 20 | retail 계열 router 보강 |
| shared_5 | 224 | 448 | 224 | 64 | 20 | 20 | KuaiRec 강한 preset |
| shared_6 | 224 | 448 | 224 | 96 | 12 | 30 | lastfm/long context preset |
| shared_7 | 128 | 256 | 128 | 96 | 16 | 10 | movielens target 확인용 |
| shared_8 | 256 | 512 | 256 | 64 | 16 | 20 | 큰 capacity 확인용 |

## Phase 0: Native Checkpoint Bank

목적은 모든 이후 phase가 공유할 scratch baseline과 source checkpoint를 한 번만 만드는 것이다.

- 단위: `dataset × shared_hparam × seed`
- fast profile은 LR center 1점만 사용한다.
- full profile은 LR 3점을 비교한다.
- 산출 checkpoint: `artifacts/checkpoints/native/<dataset>/<shared_id>/seed_<s>/best.pth`
- 이 결과가 target scratch baseline이다. Phase 1~3에서 scratch를 다시 돌리지 않는다.

실행:

```bash
./run_ai502_transfer.sh native 0,1,2,3
```

## Phase 1: Init-only Transfer Discovery

목적은 freeze 없이 어떤 transfer 범위가 좋은지 target별로 찾는 것이다.

중요한 전제는 A12 architecture를 실제 학습 config에 명시해서 checkpoint 안에 `router_d`, `router_e`가 생기도록 하는 것이다. 이전처럼 A12 이름만 metadata로 넘기고 실제 router wrapper 설정을 주지 않으면 checkpoint가 기본 router만 가진 상태가 되어 feature/router transfer가 no-op처럼 보일 수 있다. 현재 launcher는 A12의 `w5_exd` wrapper와 feature-source primitive 설정을 공통 override로 넘긴다.

A12 `w5_exd`의 `d_cond/e_scalar` router는 `_StageFeatureEncoder` 출력이 아니라 `group_feature_projections`로 만든 group별 feature context를 사용한다. 따라서 이 계획에서 `feature_encoder` transfer는 stage별 `feature_encoder`와 `group_feature_projections`를 함께 의미한다. 이 정의가 feature/cue encoder transfer의 실제 학습 경로와 맞다.

- `feature_encoder_init`: feature/cue encoder 초기화. A12에서는 `feature_encoder + group_feature_projections`
- `group_router_init`: A12 group router인 `router_e`만 초기화
- `feature_encoder_group_router_init`: feature/cue encoder와 A12 group router `router_e` 초기화
- `all_router_init`: checkpoint에 존재하는 모든 router 초기화
- `feature_encoder_router_init`: feature encoder와 모든 router 초기화
- `feature_encoder_a12_router_init`: feature/cue encoder와 A12 active router `router_d + router_e` 초기화
- `full_model_init`: shape가 맞는 tensor 전체 초기화. item embedding shape mismatch는 자동 skip
- `full_except_feature_router_init`: item embedding, feature encoder, router를 제외한 compatible tensor만 초기화

대표 pair는 grouped rerun에서 12개를 3개 묶음으로 나눠 실행한다. 이전 fast 10개에 `KuaiRec→beauty`, `lastfm→foursquare`를 다시 넣어서 target coverage를 복구하고, 각 그룹이 끝날 때마다 바로 후속 판단을 할 수 있게 순서를 바꾼다.

실행:

```bash
python3 run_ai502_transfer.py --phase init --profile fast --gpus 0,1,2,3 --pairs lastfm_to_KuaiRec,foursquare_to_KuaiRec,beauty_to_retail_rocket,retail_rocket_to_beauty
python3 summarize_ai502_transfer.py
```

## Phase 2: Freeze Follow-up

목적은 Phase 1 target별 top2 mode에 대해 `freeze_loaded=true`가 도움이 되는지 확인하는 것이다.

- source checkpoint, hparam, seed, pair는 같은 group의 Phase 1 결과를 재사용한다.
- 실제로 checkpoint에서 load된 parameter만 `requires_grad=False` 처리한다.
- result JSON에는 `transfer_report`와 별도 `freeze_report`가 남는다.
- fast profile은 `feature_encoder_group_router_init`, `feature_encoder_a12_router_init`, `full_model_init` 3개 mode를 freeze 확인한다.

실행:

```bash
python3 run_ai502_transfer.py --phase freeze --profile fast --gpus 0,1,2,3 --pairs lastfm_to_KuaiRec,foursquare_to_KuaiRec,beauty_to_retail_rocket,retail_rocket_to_beauty
python3 summarize_ai502_transfer.py
```

## Phase 3: Multi-hop Transfer

목적은 단순 `A→C`, `B→C`와 순차 fine-tune `A→B→C`를 비교하는 것이다.

- 비교군: `scratch C`, `A→C`, `B→C`, `A→B→C`
- `A→B` checkpoint가 같은 group의 Phase 1에 있으면 같은 경로를 재사용한다.
- Phase 1에 없는 bridge는 `multihop_bridge` row로 필요한 최소 조합만 추가한다.
- fast profile은 mode를 `feature_encoder_a12_router_init`, `full_model_init` 2개로 두고, triplet은 2개만 사용한다.
- full profile 또는 직접 지정에서는 `--multihop-modes ...`, `--triplets ...`로 늘릴 수 있다.

실행:

```bash
python3 run_ai502_transfer.py --phase multihop --profile fast --gpus 0,1,2,3 --triplets beauty_to_retail_rocket_to_KuaiRec
python3 summarize_ai502_transfer.py
```

## Pair 선택 근거

Table 9 계열 축을 압축해서 보면 데이터셋마다 다른 transfer 가설이 있다.

| dataset | session volatility | branching | repeat variability | context availability | popularity concentration | RouteRec gain/win-rate | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| beauty | 0.694 | 0.843 | 0.000 | 0.059 | 0.509 | +0.0020 / 0.556 | context가 약한 commerce target |
| foursquare | 0.944 | 0.688 | 1.961 | 0.988 | 0.647 | +0.0009 / 0.556 | rich context, session 변동 큼 |
| KuaiRec | 0.887 | 0.903 | 2.214 | 0.933 | 0.671 | +0.0182 / 1.000 | router/feature transfer가 가장 기대되는 source |
| lastfm | 0.739 | 0.630 | 2.036 | 0.990 | 0.456 | +0.0049 / 1.000 | rich context지만 popularity 편중은 낮음 |
| movielens1m | 0.381 | 0.839 | 0.000 | 0.264 | 0.699 | -0.0031 / 0.000 | challenging target, transfer 손실도 중요 |
| retail_rocket | 0.945 | 0.744 | 0.858 | 0.056 | 0.674 | +0.0055 / 0.667 | context 약하지만 session volatility 큼 |

대표 pair 전체 pool:

- rich-context 유사/교차: `foursquare→KuaiRec`, `KuaiRec→foursquare`, `KuaiRec→lastfm`, `lastfm→KuaiRec`
- low-context retail/beauty: `beauty→retail_rocket`, `retail_rocket→beauty`
- low-context에서 rich-context로: `retail_rocket→KuaiRec`, `beauty→lastfm`
- challenging ML target: `KuaiRec→movielens1m`, `lastfm→movielens1m`
- target coverage 보강: `KuaiRec→beauty`, `lastfm→foursquare`

Triplet pool:

- `foursquare→KuaiRec→lastfm`
- `beauty→retail_rocket→KuaiRec`
- `lastfm→KuaiRec→movielens1m`
- `KuaiRec→foursquare→retail_rocket`

## 중복 방지 규칙

- job key는 `phase/dataset_or_pair/shared_id/seed/transfer_mode/freeze_policy/lr_idx` 의미를 고정한다.
- native checkpoint가 있으면 scratch/source 용도로 재학습하지 않는다.
- `A→C`와 `A→B→C`가 같은 A native checkpoint를 필요로 할 때 native A는 한 번만 학습한다.
- `A→B→C`의 bridge는 Phase 1 checkpoint 경로를 우선 사용한다.
- launcher는 manifest 생성 시 같은 `job_key`가 두 번 나오면 즉시 중단한다.

## Dry-run 및 Smoke Test

Manifest와 command만 확인:

```bash
./run_ai502_transfer_all.sh --gpus 0,1 --dry-run
```

기본 grouped rerun 실행:

```bash
./run_ai502_transfer_all.sh --gpus 0,1,2,3
```

기본 스크립트는 내부적으로 `artifacts cleanup → native 전체 → summarize → group1 → group2 → group3` 순서로 barrier를 둔다. 각 group 내부는 `init → summarize → freeze → summarize → multihop → summarize` 순서다. 따라서 source/native checkpoint가 만들어지기 전에 다음 phase가 먼저 시작되지 않고, group이 끝날 때마다 분석 파일이 갱신된다.

grouped rerun 기본 row 수:

- native: 72
- group1 init/freeze/multihop: 624
- group2 init/freeze/multihop: 624
- group3 init/freeze/multihop: 720
- total: 2040

더 넓게 돌릴 때:

```bash
./run_ai502_transfer_all.sh \
  --gpus 0,1,2,3 \
  --profile full \
  --lr-mode tight3 \
  --hparams shared_1,shared_2,shared_3,shared_4,shared_5,shared_6,shared_7,shared_8 \
  --transfer-modes feature_encoder_init,group_router_init,feature_encoder_group_router_init,all_router_init,feature_encoder_router_init,feature_encoder_a12_router_init,full_model_init,full_except_feature_router_init \
  --freeze-modes feature_encoder_group_router_init,feature_encoder_a12_router_init,full_model_init \
  --multihop-modes feature_encoder_a12_router_init,full_model_init
```

작은 smoke:

```bash
python3 run_ai502_transfer.py --phase native --datasets beauty --hparams shared_1 --seeds 1 --dry-run --smoke-test
python3 run_ai502_transfer.py --phase init --datasets beauty,retail_rocket --pairs beauty_to_retail_rocket --hparams shared_1 --seeds 1 --transfer-modes feature_encoder_init,feature_encoder_group_router_init,feature_encoder_a12_router_init,full_model_init --dry-run
python3 run_ai502_transfer.py --phase freeze --datasets beauty,retail_rocket --pairs beauty_to_retail_rocket --hparams shared_1 --seeds 1 --freeze-modes feature_encoder_group_router_init,feature_encoder_a12_router_init,full_model_init --dry-run
```

## 결과 해석 기준

`summarize_ai502_transfer.py`는 다음 파일을 만든다.

- `native_scratch_mean.csv`: dataset/hparam별 scratch 평균
- `pair_transfer_gain.csv`: pair/mode별 native 대비 gain
- `target_top2_modes.csv`: target별 top2 transfer mode
- `freeze_gain_loss.csv`: freeze가 native 대비 얻거나 잃은 값
- `multihop_direct_vs_sequential.csv`: `A→C`, `B→C`, `A→B→C` 비교

핵심 판단은 test `MRR@20` 기준으로 한다. target별 평균 gain이 양수이고 seed 간 min이 크게 음수로 흔들리지 않는 mode를 후속 후보로 본다.
