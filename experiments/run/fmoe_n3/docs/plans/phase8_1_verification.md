# Phase8_2 Verification Plan

## Summary
- 목적: `phase8` 결과 기반 재검증(verification) 중심의 `4(base) x 4(hparam) x 4(seed)=64` 실험.
- 데이터 소스:
  - `summary.csv`: `/workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n3/phase8_router_wrapper_diag_v1/P8/KuaiRecLargeStrictPosV2_0.2/summary.csv`
  - 로그 기준 스냅샷: `A/B/C` 완료, `D` 진행 중.
- 선호 반영:
  - rule-bias 포함은 4개 base 중 1개
  - 변주 스타일은 보수+균형

## 1) Phase8 결과 요약 (전체 행)

### Stage A: `wrapper_combo`
| name | 설명 | mean valid MRR@20 | mean test MRR@20 |
|---|---|---:|---:|
| all_w1 | w1 flat을 macro/mid/micro 모두 사용 | 0.080100 | 0.161500 |
| all_w2 | w2(a+d residual)을 macro/mid/micro 모두 사용 | 0.081900 | 0.160300 |
| all_w3 | w3(bxc product)을 macro/mid/micro 모두 사용 | 0.080500 | 0.161300 |
| all_w4 | w4(bxd product)을 macro/mid/micro 모두 사용 | 0.081000 | 0.161300 |
| all_w5 | w5(exd product)을 macro/mid/micro 모두 사용 | 0.081900 | 0.160200 |
| all_w6 | w6(bxd+a residual)을 macro/mid/micro 모두 사용 | 0.080900 | 0.159900 |
| mixed_1 | macro/mid=w4, micro=w1 | 0.081400 | 0.158700 |
| mixed_2 | macro=w4, mid=w6, micro=w1 | 0.081300 | 0.160400 |
| mixed_3 | macro=w6, mid/micro=w1 | 0.080300 | 0.161700 |

### Stage B: `bias_mode`
| name | 설명 | mean valid MRR@20 | mean test MRR@20 |
|---|---|---:|---:|
| bias_off | bias 비활성 | 0.080725 | 0.161225 |
| bias_feat | feature_group_bias만 사용 | 0.080950 | 0.161150 |
| bias_group_feat | group feature prior bias 명시 모드 | 0.081125 | 0.160400 |
| bias_rule | rule target bias만 사용 | 0.081500 | 0.160075 |
| bias_both | feature/group + rule bias 결합 | 0.081225 | 0.160475 |
| bias_group_feat_rule | group feature + rule bias 결합 | 0.081225 | 0.160575 |

### Stage C: `source_profile`
| name | 설명 | mean valid MRR@20 | mean test MRR@20 |
|---|---|---:|---:|
| src_base | 기본 source 맵(a/b/c=both, d/e=feature) | 0.080675 | 0.161000 |
| src_all_both | 모든 primitive를 both로 통일 | 0.080825 | 0.161150 |
| src_a_hidden_b_d_feature | a=hidden, b/d=feature 중심 | 0.080950 | 0.160650 |
| src_abc_feature | a/b/c를 feature로 통일 | 0.081300 | 0.160425 |

### Stage D: `topk_profile`
| name | 설명 | mean valid MRR@20 | mean test MRR@20 |
|---|---|---:|---:|
| tk_dense | primitive/final top-k 모두 dense | 0.080867 | 0.161167 |
| tk_d1 | d_cond top-k=1, final dense | 0.080033 | 0.160600 |
| tk_d1_final4 | d_cond top-k=1 + final top-k=4 | 0.080067 | 0.160667 |
| tk_a3_d1_final4 | a=3, d=1, final=4 | 0.079633 | 0.160500 |

## 2) Base settings 4개 (검증용)

### A: `B1_phasewise_combo` (rule 1개 슬롯)
- wrapper: `all_w5`
- bias: `bias_rule`
- source: `src_abc_feature`
- top-k: `tk_dense`
- 근거: 단계별 평균 winner 조합의 결합

### B: `B2_best_learned_mixed2`
- wrapper: `mixed_2`
- bias: `bias_group_feat`
- source: `src_base`
- top-k: `tk_dense`
- 근거: non-rule 상위 단일 run 반영

### C: `B3_clean_all_w5`
- wrapper: `all_w5`
- bias: `bias_off`
- source: `src_base`
- top-k: `tk_dense`
- 근거: clean baseline 강자

### D: `B4_clean_all_w2`
- wrapper: `all_w2`
- bias: `bias_off`
- source: `src_base`
- top-k: `tk_dense`
- 근거: clean baseline 강자 + wrapper 다양성

## 3) Hparam variants 4개
- 변경 축: `embedding / d_ff / d_expert_hidden / d_router_hidden / wd / hidden_dropout`

### H1_baseline
- `embedding=128, d_ff=256, d_expert_hidden=128, d_router_hidden=64, wd=1e-6, hidden_dropout=0.15`

### H2_capacity_up_light_reg
- `embedding=160, d_ff=320, d_expert_hidden=160, d_router_hidden=80, wd=5e-7, hidden_dropout=0.12`

### H3_capacity_up_bal_reg
- `embedding=160, d_ff=320, d_expert_hidden=160, d_router_hidden=80, wd=2e-6, hidden_dropout=0.18`

### H4_compact_strong_reg
- `embedding=112, d_ff=224, d_expert_hidden=112, d_router_hidden=56, wd=3e-6, hidden_dropout=0.20`

## 4) 실행 매트릭스 / run naming / 재시작 정책
- seeds: `s1,s2,s3,s4`
- 총 실험: `4 x 4 x 4 = 64`
- phase 폴더: `P8_2`
- run id: `A~D_H1~H4_S<seed>`
- run_phase 표기: `P8_2_A_H1_S1` 형태
- 우선순위:
  - 1순위: `A/B`의 모든 variant x seed
  - 2순위: `C/D`의 모든 variant x seed

### 중단/재시작 규칙
- 같은 이름 로그(`A_H1_S1.log`)를 기준으로 판단
- 파일 안에 `[RUN_STATUS]` 문자열이 있으면 **이미 종료 처리된 run**으로 보고 시작하지 않음
- `[RUN_STATUS]`가 없으면 미완료 로그로 간주하고 **해당 파일을 덮어써서 재실행**

## 5) 평가 기준
- 1차: seed 평균 `valid MRR@20`
- 2차: seed 표준편차(낮을수록 우선), seed 최솟값, `test MRR@20`
- 최종 2개 추천: `mean(valid)` 상위 후보 중 `std/min` 안정성이 높은 조합 우선

## Test checklist
- `64`개 조합이 정확히 생성되는지 확인
- run name이 `base/hvar/seed` 역파싱 가능한지 확인
- 본 문서 평균표 숫자가 `summary.csv` 집계와 일치하는지 확인
- rule-bias 포함 base가 정확히 1개인지 확인

## Assumptions
- `phase8` D가 아직 진행 중이므로 본 문서는 스냅샷 기반 verification v1.
- `source=default`는 `src_base` 의미.
- 이번 phase8_2는 구조 확장이 아니라 재현/안정성 검증 목적.
