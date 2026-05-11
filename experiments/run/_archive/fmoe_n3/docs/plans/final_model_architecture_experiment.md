# Final Model Architecture Experiment (Final_all_datasets)

작성일: 2026-03-26 (UTC)

## 1) 최종 결정
- 메인 layout: `MACRO_MID_MICRO` (`["macro","mid","micro"]`)
- 축/로그 이름: `Final_all_datasets`
- 평가 모드: `eval_mode=session_fixed` (세션 단위 시계열 train/valid/test)
- 최종 비교군: `A1~A6`
- `A1`: 기본(Final Main, plain `macro->mid->micro`)
  - `A2`: intra-feature NN 강화 + sub aux=`z-loss`
  - `A3`: `A1 + no_category` (구조적 drop)
  - `A4`: rule/group bias off + direct intra-group bias(`gls_stats12`)
  - `A5`: `A1 + no_category_no_timestamp` (구조적 drop)
  - `A6`: `A1 + no_bias` (bias_mode=none, rule/group bias off)

### 1.1 Layout 근거 (phase9~13 집계)
| Setting | valid mrr20 (mean±std) | test mrr20 (mean±std) | diag top1 | diag cv |
| --- | --- | --- | --- | --- |
| `P12-02_ATTN_MICRO_BEFORE` | `0.08085 ± 0.00006` | `0.16210 ± 0.00014` | `0.25884` | `0.12286` |
| `P11-00_MACRO_MID_MICRO` | `0.08130 ± 0.00008` | `0.16160 ± 0.00012` | `0.17024` | `0.14278` |

- 해석:
  - 두 설정 차이가 작고(절대값 기준 근소), 재현성과 해석 용이성을 위해 최종 main은 기본 `MACRO_MID_MICRO`로 고정한다.
  - `ATTN_MICRO_BEFORE`는 과거 비교군 결과로만 문서에 유지한다.

## 2) 최종 메인 아키텍처 (A1)
- Core: `B4 + C0-N4`
  - `B4`: mixed_2 + bias_both + src_abc_feature
  - `C0-N4`: `z_loss_lambda=1e-4`, balance 계열 off
- Feature policy:
  - `stage_family_dropout_prob={macro:0.10,mid:0.10,micro:0.10}`
  - `stage_feature_dropout_prob={macro:0.0,mid:0.0,micro:0.0}`
- Layout:
  - `layer_layout=[macro,mid,micro]`
- 실행 고정:
  - `feature_mode=full_v3`
  - `eval_mode=session_fixed`
  - `fmoe_special_logging=true`
  - `fmoe_diag_logging=true`
  - `verify_logging=true`

## 3) A2~A6 변형 정의
- `A2` (엄격 NN + z-loss)
  - `route_consistency_pairs=1`
  - `route_consistency_min_sim=0.995`
  - `route_consistency_lambda=8e-4`
  - `z_loss_lambda=2e-4`
  - `route_monopoly_lambda=0`, `balance_loss_lambda=0`
- `A3` (no category)
  - `stage_feature_drop_keywords=["cat","theme"]`
- `A4` (bias 전용 변경)
  - `rule_bias_scale=0`, `feature_group_bias_lambda=0`
  - `intra_group_bias_mode=gls_stats12`
  - `intra_group_bias_scale=0.12`
  - 적용 stage: `macro,mid,micro`
- `A5` (no category + no timestamp)
  - `stage_feature_drop_keywords=["cat","theme","timestamp","gap","pace","int_","_int","sess_age","ctx_valid_r","valid_r","delta_vs_mid"]`
- `A6` (no bias)
  - `bias_mode=none`
  - `rule_bias_scale=0`
  - `feature_group_bias_lambda=0`

## 4) 실험 예산 및 hparam 정책
- 데이터셋: `KuaiRecLargeStrictPosV2_0.2,lastfm0.03,amazon_beauty,foursquare,movielens1m,retail_rocket`
- 예산: `A1(kuai만) + dataset별 A2~A5` 기준으로 launcher 스케줄 진행
- seed: `1,2,3,4`
- hparam:
  - 공통: `H1,H3`
  - dataset별 outlier 1개:
    - `KuaiRecLargeStrictPosV2_0.2 -> H4`
    - `lastfm0.03 -> H5`
    - `amazon_beauty -> H8`
    - `foursquare -> H2`
    - `movielens1m -> H5`
    - `retail_rocket -> H6`

## 5) Special logging 개편 (session_fixed 정합)
- train 기준 item count를 사용해 slice를 집계한다.
  - `cold_0`: train count == 0
  - `rare_1_5`: train count in [1, 5]
  - `6_20`, `21_100`, `101+`
- session length slice:
  - `<=7`, `8-12`, `13+`
- 호환성:
  - legacy slice(`target_popularity_abs_legacy`, `session_len_legacy`) 병행 저장
  - 기존 분석 코드와의 호환을 유지한다.

## 6) 실행 인터페이스
- Python:
  - `experiments/run/fmoe_n3/run_final_all_datasets.py`
- Shell:
  - `experiments/run/fmoe_n3/phase_14_final_all_datasets.sh`

### 5.1 기본 실행 예시
```bash
bash experiments/run/fmoe_n3/phase_14_final_all_datasets.sh \
  --gpus 0,1,2,3 \
  --seeds 1,2,3,4
```

### 5.2 Dry-run / Smoke
```bash
bash experiments/run/fmoe_n3/phase_14_final_all_datasets.sh --dry-run
bash experiments/run/fmoe_n3/phase_14_final_all_datasets.sh --smoke-test --smoke-max-runs 2
```
