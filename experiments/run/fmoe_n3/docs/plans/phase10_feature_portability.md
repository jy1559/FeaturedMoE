# Phase 10 — Feature Portability / Compactness (Execution Plan)

## 1) 의도 요약

Phase10의 핵심 질문은 아래 3가지다.

1. `FeaturedMoE_N3`가 feature bank가 커서만 좋은가, 아니면 **적은 family/적은 feature 템플릿**으로도 유지되는가?
2. family 중 무엇이 핵심 축인가? (Tempo/Focus/Memory/Exposure)
3. category/timestamp availability 변화와 학습 중 stochastic usage에 얼마나 robust한가?

논문 메시지 목표:

> A compact, widely available feature template is often sufficient for feature-aware routing.

---

## 2) 구현 원칙 (이번 실행 기준)

- 메인 비교는 **spec-level structural removal**로 수행한다.
- 즉 `stage_feature_family_mask`, `stage_feature_family_topk`, `stage_feature_family_custom`, `stage_feature_drop_keywords`로
  stage feature spec 자체를 줄인다.
- `FAMILY_DROPOUT`, `FEATURE_DROPOUT` 2개는 training-time stochastic usage 확인용으로, 학습 중에만 적용한다.

---

## 3) 공통 고정 세팅

### 3.1 Anchor (P9/P9_2 기반)

- Base anchor: `B4` 계열 (mixed_2 wrapper + bias_both + src_abc_feature)
- 안정화: `z_loss_lambda=1e-4`
- `macro_history_window=5`

### 3.2 고정 hparam

- `embedding_size=160`
- `d_ff=320`
- `d_expert_hidden=160`
- `d_router_hidden=80`
- `expert_scale=3`
- `batch_size(train/eval)=4096`
- `fixed_weight_decay=2e-6`
- `fixed_hidden_dropout_prob=0.18`
- `attn_dropout_prob=0.1`
- `d_feat_emb=16`

### 3.3 탐색 축

- Search는 LR만 유지:
  - `learning_rate ∈ [1.5e-4, 8e-3]`
  - `lr_scheduler_type=warmup_cosine`

---

## 4) 실험 매트릭스

기본 22개 + 확장 2개(옵션) = 최대 24개.

- 기본 실행은 22개(`--no-extra-24` 기본)이며, 필요 시 `--include-extra-24`로 24개 확장한다.

### 4.1 Group subset lattice (15)

- `P10-00_FULL`
- `P10-01_Tempo`
- `P10-02_Focus`
- `P10-03_Memory`
- `P10-04_Exposure`
- `P10-05_Tempo_Focus`
- `P10-06_Tempo_Memory`
- `P10-07_Tempo_Exposure`
- `P10-08_Focus_Memory`
- `P10-09_Focus_Exposure`
- `P10-10_Memory_Exposure`
- `P10-11_Tempo_Focus_Memory`
- `P10-12_Tempo_Focus_Exposure`
- `P10-13_Tempo_Memory_Exposure`
- `P10-14_Focus_Memory_Exposure`

구현 키:

- `stage_feature_family_mask`

### 4.2 Intra-group reduction (3)

- `P10-15_TOP2_PER_GROUP`
- `P10-16_TOP1_PER_GROUP`
- `P10-17_COMMON_TEMPLATE`

구현 키:

- `stage_feature_family_topk` (TOP2/TOP1)
- `stage_feature_family_custom` (COMMON_TEMPLATE)

세팅 상세(keep/drop):

- `TOP2_PER_GROUP`:
  - 각 stage/family의 원본 4개 중 **앞 2개만 keep**, 뒤 2개 drop
  - 예) macro Tempo: keep=`mac5_ctx_valid_r`,`mac5_gap_last`; drop=`mac5_pace_mean`,`mac5_pace_trend`
  - 예) mid Focus: keep=`mid_cat_ent`,`mid_cat_top1`; drop=`mid_cat_switch_r`,`mid_cat_uniq_r`
  - 예) micro Exposure: keep=`mic_last_pop`,`mic_suffix_pop_std`; drop=`mic_suffix_pop_ent`,`mic_pop_delta_vs_mid`

- `TOP1_PER_GROUP`:
  - 각 stage/family의 원본 4개 중 **첫 1개만 keep**, 나머지 3개 drop
  - 예) macro Focus: keep=`mac5_theme_ent_mean`; drop=`mac5_theme_top1_mean`,`mac5_theme_repeat_r`,`mac5_theme_shift_r`
  - 예) mid Tempo: keep=`mid_valid_r`; drop=`mid_int_mean`,`mid_int_std`,`mid_sess_age`
  - 예) micro Memory: keep=`mic_is_recons`; drop=`mic_suffix_recons_r`,`mic_suffix_uniq_i`,`mic_suffix_max_run_i`

- `COMMON_TEMPLATE`:
  - `_common_template_custom()`에 명시된 2개만 keep, 그 외 drop
  - macro keep:
    - Tempo: `mac5_ctx_valid_r`, `mac5_gap_last`
    - Focus: `mac5_theme_top1_mean`, `mac5_theme_repeat_r`
    - Memory: `mac5_repeat_mean`, `mac5_adj_item_overlap_mean`
    - Exposure: `mac5_pop_mean`, `mac5_pop_ent_mean`
  - mid keep:
    - Tempo: `mid_valid_r`, `mid_int_mean`
    - Focus: `mid_cat_top1`, `mid_cat_switch_r`
    - Memory: `mid_repeat_r`, `mid_item_uniq_r`
    - Exposure: `mid_pop_mean`, `mid_pop_ent`
  - micro keep:
    - Tempo: `mic_valid_r`, `mic_last_gap`
    - Focus: `mic_cat_switch_now`, `mic_last_cat_mismatch_r`
    - Memory: `mic_is_recons`, `mic_suffix_recons_r`
    - Exposure: `mic_last_pop`, `mic_suffix_pop_ent`
  - 위 keep 리스트에 없는 원본 feature는 drop

### 4.3 Availability ablation (2)

- `P10-18_NO_CATEGORY`
- `P10-19_NO_TIMESTAMP`

구현 키:

- `stage_feature_drop_keywords`

세팅 상세(keep/drop):

- `NO_CATEGORY` (`["cat","theme"]` substring match):
  - drop 예시:
    - macro Focus 전체: `mac5_theme_*` 4개
    - macro Memory 일부: `mac5_adj_cat_overlap_mean`
    - mid Focus 전체: `mid_cat_*` 4개
    - micro Focus 전체: `mic_*cat*` 4개
  - keep 예시:
    - macro Tempo/Exposure 대부분
    - mid Tempo/Memory/Exposure(단, 이름에 `cat/theme` 없는 항목)
    - micro Tempo/Memory/Exposure(단, 이름에 `cat/theme` 없는 항목)

- `NO_TIMESTAMP` (`["timestamp","gap","pace","int_","_int","sess_age","ctx_valid_r","valid_r","delta_vs_mid"]` substring match):
  - drop 예시:
    - macro Tempo 전체: `mac5_ctx_valid_r`,`mac5_gap_last`,`mac5_pace_mean`,`mac5_pace_trend`
    - mid Tempo 전체: `mid_valid_r`,`mid_int_mean`,`mid_int_std`,`mid_sess_age`
    - micro Tempo 전체: `mic_valid_r`,`mic_last_gap`,`mic_gap_mean`,`mic_gap_delta_vs_mid`
    - micro Exposure 일부: `mic_pop_delta_vs_mid`
  - keep 예시:
    - macro Focus/Memory/Exposure 대부분
    - mid Focus/Memory/Exposure 대부분
    - micro Focus/Memory와 `mic_last_pop`,`mic_suffix_pop_std`,`mic_suffix_pop_ent`

### 4.4 Stochastic feature usage (2)

- `P10-20_FAMILY_DROPOUT`
- `P10-21_FEATURE_DROPOUT`

구현 키:

- `stage_family_dropout_prob`
- `stage_feature_dropout_prob`
- `stage_feature_dropout_scope`

### 4.5 확장 2개 (옵션, 총 24개 구성 시 사용)

- `P10-22_NO_CATEGORY_NO_TIMESTAMP`
- `P10-23_COMMON_TEMPLATE_NO_CATEGORY`

---

## 5) 실행 파일

- Python launcher:
  - `experiments/run/fmoe_n3/run_phase10_feature_portability.py`
- Shell wrapper:
  - `experiments/run/fmoe_n3/phase_10_feature_portability.sh`

예시:

```bash
bash experiments/run/fmoe_n3/phase_10_feature_portability.sh \
  --datasets KuaiRecLargeStrictPosV2_0.2 \
  --gpus 4,5,6,7 \
  --seeds 1 \
  --include-extra-24
```

스모크 테스트:

```bash
bash experiments/run/fmoe_n3/phase_10_feature_portability.sh \
  --gpus 4 \
  --smoke-test \
  --smoke-max-runs 2
```

---

## 6) 스케줄/큐/재실행 규칙

### 6.1 GPU 할당

- 실험 row를 생성 순서대로 `gpus[idx % len(gpus)]`로 사전 할당
- GPU별 queue를 만들고 비는 GPU 슬롯마다 즉시 다음 run launch

### 6.2 재실행 skip 기준 (요청 반영)

동일 실험(`run_id`)의 log 파일이 존재하고,

- 마지막 non-empty line이
  - `[RUN_STATUS] END status=normal`
  로 시작(뒤에 pid/start/end 메타 허용)

이면 skip.

그 외(파일 없음 / 마지막 줄 불일치 / 비정상 종료)는

- 같은 경로 log를 **preamble부터 덮어쓰기** 후 재실행.

---

## 7) summary.csv 정책

출력 위치:

- `experiments/run/artifacts/logs/fmoe_n3/phase10_feature_portability_v1/P10/<dataset>/summary.csv`

스키마 핵심:

- 첫 두 컬럼 고정:
  - `global_best_valid_mrr20`
  - `run_best_valid_mrr20`
- run 설명 컬럼 포함:
  - `exp_brief`, `setting_id`, `setting_key`, `setting_desc`

append 트리거:

1. `trial_new_best`: trial 진행 중 global best 갱신 시
2. `run_complete`: run 종료 시

즉 "run 종료" 또는 "global best 갱신" 시마다 한 줄 추가.

---

## 8) logging(special/diag) 검증 정책

- phase10 launcher는 run 완료 후 result json을 읽어
  - special 경로 (`special_log_file` / `special_result_file`)
  - diag 경로 (`diag_raw_trial_summary_file` / `diag_tier_a_final_file` / `diag_meta_file`)
  존재 여부를 확인한다.
- `--verify-logging` 기본 활성화.
- 검증 실패 시 런처는 즉시 에러를 발생시켜 조용한 누락을 방지한다.

---

## 9) 기대 해석 프레임

- subset(2~3 family)이 FULL 근접:
  - "large bank"보다 "meaningful family composition"이 중요.
- TOP2/TOP1/COMMON_TEMPLATE 유지:
  - reusable compact template 주장 강화.
- NO_CATEGORY/NO_TIMESTAMP 내성:
  - heterogeneous availability 대응 가능성 강화.
- dropout이 비슷하거나 개선:
  - feature-aware routing의 robustness regularization 시사.
