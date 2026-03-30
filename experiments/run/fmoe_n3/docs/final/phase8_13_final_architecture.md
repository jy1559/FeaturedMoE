# Phase8~13 Final Architecture Recommendation (KuaiRecLargeStrictPosV2_0.2)

## 목적
- 후보를 과하게 늘리지 않고, **구조적으로 바꿀 수 있는 축**만 남겨서 추천.
- 각 축에서 추천은 **1~3개**로 제한.
- 근거는 `valid / test / special(cold<=5, short1-2) / diag`를 함께 사용.

## 해석 원칙
- 아래 추천은 P8~13 wide+verification 완료 run 집계 기반이다.
- 축별 최고 setting은 존재하지만, 축 간 조합은 일부가 **교차 실험 미완료**이므로 “강한 추천 recipe”로 해석.
- 신뢰도: `n>=5` 높음, `n=2~4` 중간, `n=1` 낮음.

## 1) 구조별로 실제 바꿀 수 있는 변수와 추천 (1~3개)

### A. Router / Wrapper Core (P8)
| 변수(축) | 추천 setting (1~3) | 왜 이 setting인가 (근거) | 신뢰도 |
| --- | --- | --- | --- |
| wrapper 구조 | `ALL_W4`, `ALL_W5` | `ALL_W4`: test=0.1613, cold=0.1213로 균형. `ALL_W5`: valid=0.0819, short=0.0294로 valid/short 강점. | 낮음 (n=1) |
| bias 주입 | `ALL_W5_BIAS_GROUP_FEAT_RULE`, `MIXED_1_BIAS_GROUP_FEAT` | `GROUP_FEAT_RULE`: test=0.1617. `MIXED_1_GROUP_FEAT`: test=0.1618, cold=0.1211. `BIAS_OFF`는 short=0.0588 강점 있지만 cold/test 변동 큼. | 낮음 (n=1) |
| router source | `MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH` | test=0.1617, cold=0.1219, diag도 n_eff=11.84 / ent=2.302 / top1=0.213으로 과집중 억제형. | 낮음 (n=1) |
| top-k profile | `...TK_DENSE` (dense) | test=0.1619, short=0.0294로 가장 강함. D1_FINAL4 대비 n_eff/ent도 높음. | 낮음 (n=1) |

### B. Feature Router/Input (P10)
| 변수(축) | 추천 setting (1~3) | 왜 이 setting인가 (근거) | 신뢰도 |
| --- | --- | --- | --- |
| feature subset | `FULL`, `Tempo_Focus_Exposure` | `FULL`: valid=0.0811, cold=0.1216. `Tempo_Focus_Exposure`: test=0.1618, cold=0.1220(단 n=1). | 중간 (`FULL` n=9) |
| availability | `NO_TIMESTAMP`, `NO_CATEGORY` | `NO_TIMESTAMP`: test=0.1617, cold=0.1212. `NO_CATEGORY`: valid 상대우위(0.0809)이나 cold/test는 낮아짐. | 높음 (n=9/9) |
| compactness | `TOP2_PER_GROUP`, `COMMON_TEMPLATE` | `TOP2_PER_GROUP`: test=0.1618, short=0.0044. `COMMON_TEMPLATE`: 성능은 약간 낮지만 안정 대안. | 높음 (n=9/9) |
| stochastic | `FEATURE_DROPOUT`, `FAMILY_DROPOUT` | 두 setting 모두 test 상단(0.1619대). `FEATURE_DROPOUT`이 valid/intra에서 소폭 유리. | 높음 (n=9/9) |

### C. Stage / Layout (P11~P12)
| 변수(축) | 추천 setting (1~3) | 왜 이 setting인가 (근거) | 신뢰도 |
| --- | --- | --- | --- |
| stage 구성 | `MID_MICRO`, `MACRO_MID_MICRO` | `MID_MICRO`: test=0.1617, cold=0.1218. `MACRO_MID_MICRO`: valid=0.0816로 valid 우위. | 중간~높음 (n=1/5) |
| routing granularity | `TOKEN_TOKEN_TOKEN`, `TOKEN_SESSION_TOKEN` | `TOKEN_TOKEN_TOKEN`: test=0.1614, cold=0.1215로 가장 안정. `TOKEN_SESSION_TOKEN`은 n_eff/ent 높지만 n=1. | 중간 |
| stage order | `MICRO_MACRO_MID`, `MID_MICRO_MACRO` | `MICRO_MACRO_MID`: valid=0.0822 최고. `MID_MICRO_MACRO`: test=0.1621, cold=0.1224(단 n=1). | 중간~낮음 |
| layout variant | `MACRO_REPEATED`, `ATTN_MICRO_BEFORE` | `MACRO_REPEATED`: valid=0.0815. `ATTN_MICRO_BEFORE`: test=0.1621에 가까운 상단, cold도 유지. | 중간 |
| bundle strategy | `BUNDLE_MACROMICRO_LEARNED` | test=0.1616 + cold=0.1233 + short=0.0091로 가장 실전형. | 높음 (n=5) |

## 2) diag 관점에서 어떤 동작이 좋은가 (선택 기준)
- 전체/축-중심/통제 상관 모두에서 공통적으로:
  - `n_eff ↑`, `entropy ↑`, `intra_group_nn ↑` 방향이 test와 양(+) 경향.
  - `top1_max_frac ↓` 방향이 test와 양(+) 경향.
- 통제(동일 축/동일 setting, 주로 hparam/seed 차이)에서도 동일:
  - `rho_test`: n_eff=0.7719, ent=0.7607, intra_nn=0.7379, top1=-0.6337
- 따라서 구조 선택 시 “한 expert 과집중(top1)”보다 “분산/혼합 라우팅(n_eff, ent)”이 유리한 쪽으로 선택.

## 3) 최종 3개 제안 (컨셉별, 바로 가져다 쓰는 recipe)

## FINAL-1: Valid 우선
- 컨셉: 논문 본문에서 valid best를 강조할 때.
- recipe:
`P8(wrapper=ALL_W5, source=MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH, topk=TK_DENSE)`
`P10(FULL + NO_CATEGORY + COMMON_TEMPLATE + FEATURE_DROPOUT)`
`P11(MACRO_MID_MICRO + MICRO_MACRO_MID + TOKEN_TOKEN_TOKEN)`
`P12(MACRO_REPEATED + BUNDLE_MACROMICRO_LEARNED)`
- 근거:
  - valid 강한 축 채택: `MICRO_MACRO_MID(valid=0.0822)`, `MACRO_MID_MICRO(valid=0.0816)`, `MACRO_REPEATED(valid=0.0815)`, `FULL(valid=0.0811)`
  - test/cold가 완전히 무너지지 않도록 `TOKEN_TOKEN_TOKEN`, `BUNDLE_MACROMICRO_LEARNED`로 하단 방어.

## FINAL-2: Test + Special(cold/short) 우선
- 컨셉: 실제 서비스 일반화/콜드 대응 최우선.
- recipe:
`P8(wrapper=ALL_W4, bias=ALL_W5_BIAS_GROUP_FEAT_RULE, source=MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH, topk=TK_DENSE)`
`P10(FULL + NO_TIMESTAMP + TOP2_PER_GROUP + FEATURE_DROPOUT)`
`P11(MID_MICRO + TOKEN_TOKEN_TOKEN)`
`P12(ATTN_MICRO_BEFORE + BUNDLE_MACROMICRO_LEARNED)`
- 근거:
  - test/cold 우위 축 결합: `NO_TIMESTAMP(test=0.1617,cold=0.1212)`, `TOP2_PER_GROUP(test=0.1618)`, `FEATURE_DROPOUT(test=0.1619)`, `MID_MICRO(cold=0.1218)`, `BUNDLE_MACROMICRO_LEARNED(cold=0.1233)`
  - diag 선택 기준에도 부합: 분산 라우팅(n_eff/ent) 우위, 과집중(top1) 억제.

## FINAL-3: 균형형 (권장 기본)
- 컨셉: valid/test/special을 동시에 크게 해치지 않는 default.
- recipe:
`P8(wrapper=ALL_W4, source=MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH, topk=TK_DENSE)`
`P10(FULL + NO_TIMESTAMP + TOP2_PER_GROUP + FEATURE_DROPOUT)`
`P11(MID_MICRO + TOKEN_TOKEN_TOKEN)`
`P12(MACRO_REPEATED + BUNDLE_MACROMICRO_LEARNED)`
- 근거:
  - valid 극단/short 극단을 피하고, test/cold의 안정 구간을 우선 채택.
  - 여러 축에서 `n>=5`인 setting을 최대한 포함해서 재현성 위험을 낮춤.

## 4) 실제 실행 추천 순서
1. `FINAL-3`를 메인으로 4-seed 재검증.
2. `FINAL-2`를 cold/short 중심 대조군으로 병행.
3. 논문 valid 강조가 필요하면 `FINAL-1` 추가.

