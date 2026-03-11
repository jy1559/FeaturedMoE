# FeaturedMoE_N P0 / P0.5 Plan

## 1. Scope

- 메인 트랙은 `FeaturedMoE_N`이다.
- 이번 단계에서는 모델 구조를 더 복잡하게 만들지 않는다.
- 산출물은 `P0` 운영안, `P0.5` 초안, 그리고 실제로 돌릴 수 있는 스크립트 초안이다.
- 고정 스크립트 이름:
  - `tune_hparam.sh`
  - `phase_p0_anchor.sh`
  - `phase_p0_5_lr_narrow.sh`


## 2. Why This P0

- `simple_flat`과 non-learnable prior를 먼저 본다.
- router 구조 복잡화는 뒤로 미룬다.
- 짧은 예산에서 epoch-based schedule 효과를 보기 어렵기 때문에 `fmoe_schedule_enable=false`로 고정한다.
- `special_metrics.json` 기반 run-group 비교를 기본 분석 단위로 둔다.


## 3. P0 Common Fixed Values

- `router_design=simple_flat`
- `embedding_size=128`
- `d_router_hidden=64`
- `hidden_dropout_prob=0.10`
- `weight_decay=5e-5`
- `mid_router_temperature=1.2`
- `micro_router_temperature=1.2`
- `fmoe_schedule_enable=false`
- `feature_encoder_sinusoidal_n_freqs=4`
- `rule_router.variant=ratio_bins`
- `rule_router.n_bins=5`
- `rule_router.feature_per_expert=4`
- `router_use_hidden=true`
- `router_use_feature=true`
- `fmoe_special_logging=true`


## 4. 24-Combo Matrix

### 4.1 Kuai 18

`Q01-Q12`는 triplet layout 비교다.

| Combo | Layout | Family | train/eval bs | Key diff |
| --- | --- | --- | --- | --- |
| Q01 | L7 | plain | 6144 / 12288 | base |
| Q02 | L7 | hybrid | 6144 / 12288 | `router_impl_by_stage={mid:rule_soft,micro:rule_soft}` |
| Q03 | L7 | bias | 6144 / 12288 | `rule_bias_scale=0.15` |
| Q04 | L16 | plain | 6144 / 12288 | base |
| Q05 | L16 | hybrid | 6144 / 12288 | hybrid |
| Q06 | L16 | bias | 6144 / 12288 | bias |
| Q07 | L19 | plain | 6144 / 12288 | base |
| Q08 | L19 | hybrid | 6144 / 12288 | hybrid |
| Q09 | L19 | bias | 6144 / 12288 | bias |
| Q10 | L15 | plain | 6144 / 12288 | base |
| Q11 | L15 | hybrid | 6144 / 12288 | hybrid |
| Q12 | L15 | bias | 6144 / 12288 | bias |
| Q13 | L7 | plain | 6144 / 12288 | `expert_scale=1` |
| Q14 | L7 | plain | 6144 / 12288 | `expert_scale=5` |
| Q15 | L7 | plain | 6144 / 12288 | `moe_top_k=2`, `moe_top_k_policy=fixed` |
| Q16 | L7 | plain | 6144 / 12288 | `d_feat_emb=64`, OOM 시 4096 / 8192 1회 fallback |
| Q17 | L7 | plain | 6144 / 12288 | `feature_encoder_mode=sinusoidal_selected` |
| Q18 | L7 | plain | 6144 / 12288 | `balance_loss_lambda=0.0` |

`Q01-Q12` 공통:

- `d_feat_emb=16`
- `d_expert_hidden=128`
- `expert_scale=3`
- `moe_top_k=0`
- `moe_top_k_policy=auto`
- `feature_encoder_mode=linear`
- `balance_loss_lambda=0.002`

`Q13-Q18` 공통:

- `Q01 plain`을 base anchor로 보고 한 축만 바꾼다.
- 즉 `L7`, `plain`, `train/eval bs=6144/12288`를 유지한다.

### 4.2 lastfm 6

| Combo | Layout | Family | train/eval bs |
| --- | --- | --- | --- |
| F01 | L7 | plain | 4096 / 4096 |
| F02 | L7 | hybrid | 4096 / 4096 |
| F03 | L7 | bias | 4096 / 4096 |
| F04 | L16 | plain | 4096 / 4096 |
| F05 | L16 | hybrid | 4096 / 4096 |
| F06 | L16 | bias | 4096 / 4096 |

lastfm 공통:

- `d_feat_emb=16`
- `d_expert_hidden=128`
- `expert_scale=3`
- `moe_top_k=0`
- `moe_top_k_policy=auto`
- `feature_encoder_mode=linear`
- `balance_loss_lambda=0.002`


## 5. Wave / GPU Assignment

- 총 `6 wave x 4 GPU`
- 각 wave는 `Kuai 3개 + lastfm 1개`
- lastfm GPU slot rotation은 `0 -> 1 -> 2 -> 3 -> 0 -> 1`
- 실제 GPU 번호는 실행 시 `--gpus` 순서를 따른다.

| Wave | Combos | lastfm GPU slot |
| --- | --- | --- |
| 1 | Q01 Q02 Q03 F01 | 0 |
| 2 | Q04 Q05 Q06 F02 | 1 |
| 3 | Q07 Q08 Q09 F03 | 2 |
| 4 | Q10 Q11 Q12 F04 | 3 |
| 5 | Q13 Q14 Q15 F05 | 0 |
| 6 | Q16 Q17 Q18 F06 | 1 |


## 6. P0 Search Budget

- 목적: 넓은 LR band에서 빠른 ranking
- `epochs=18`
- `patience=3`
- `max_evals=8`
- LR distribution은 `loguniform`

LR band:

- Kuai `Q01-Q15`: `[4e-4, 1.0e-2]`
- Kuai `Q16`: `[2e-4, 6e-3]`
- Kuai `Q17-Q18`: `[4e-4, 1.0e-2]`
- lastfm `F01-F06`: `[2e-4, 5e-3]`


## 7. P0.5 Draft

중요:

- `P0.5`는 구현은 해두되, 실제 기준은 첫 `P0` 결과를 보고 수정하는 전제다.
- `phase_p0_5_lr_narrow.sh` 안에도 draft note를 남긴다.

기본 budget:

- `epochs=45`
- `patience=6`
- `max_evals=4`

기본 LR narrow:

- `lower = max(orig_lower, best_lr * 0.6)`
- `upper = min(orig_upper, best_lr * 1.5)`

기본 auto-prune:

- 상대평가만 지원한다.
- Kuai 최대 `3개` drop
- lastfm 최대 `1개` drop
- candidate rule:
  - dataset별 `best_mrr@20`
  - `score < mean - 1.0 * std`

스크립트 flag:

- `--auto-prune-relative`
- `--sigma-threshold 1.0`
- `--max-drop-kuai 3`
- `--max-drop-lfm 1`

coverage floor:

- Kuai layout `L7/L15/L16/L19` 최소 1개 유지
- Kuai family `plain/hybrid/bias` 최소 1개 유지
- lastfm layout `L7/L16` 최소 1개 유지
- lastfm family floor는 soft rule이고 layout floor가 우선

특수 규칙:

- outlier가 cap보다 많아도 worst-first로 cap까지만 drop
- outlier가 없으면 drop 없이 전부 P0.5로 넘김
- `Q16`은 sentry이므로 같은 score면 마지막에 drop


## 8. Generated Artifacts

권장 manifest:

- `artifacts/inventory/fmoe_n/p0_manifest_latest.json`
- `artifacts/inventory/fmoe_n/p05_manifest_latest.json`

manifest에는 최소한 아래를 남긴다.

- combo 정의
- wave / gpu slot
- result JSON path
- best `MRR@20`
- fallback 여부
- P0.5 parent linkage


## 9. Run-Group Logging Rule

- epoch-level router logging은 기본으로 보지 않는다.
- 기본 해석 단위는 run-level `special_metrics.json`이다.
- phase 종료 후 combo/group 단위로 아래 slice를 묶어 본다.
  - overall
  - popularity
  - short session
  - new user


## 10. Immediate Interpretation Rule

- `P0`는 구조 승부가 아니라 anchor ranking 단계다.
- `P0.5`는 상위권 재확인과 LR band 정리 단계다.
- loss/aux 실험은 `P1`로 넘긴다.
