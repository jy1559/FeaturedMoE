# Phase 4 Results

대상:
- dataset: `KuaiRecLargeStrictPosV2_0.2`
- axis: `phase4_residual_topk_v2`
- phase: `P4`
- model: `FMoEN3`

주요 확인 경로:
- sidecar: `experiments/run/artifacts/results/fmoe_n3/sidecar/phase4_residual_topk_v2/P4/KuaiRecLargeStrictPosV2_0.2/FMoEN3`
- normal: `experiments/run/artifacts/results/fmoe_n3/normal/phase4_residual_topk_v2/{R,K,F}/KuaiRecLargeStrictPosV2_0.2/FMoEN3`
- special: `experiments/run/artifacts/results/fmoe_n3/special/phase4_residual_topk_v2/{R,K,F}/KuaiRecLargeStrictPosV2_0.2/FMoEN3`
- diag 공통 산출물: `experiments/run/artifacts/results/fmoe_n3/diag/phase4_residual_topk_v2/{R,K,F}/KuaiRecLargeStrictPosV2_0.2/FMoEN3`

표기 규칙:
- 초록: 해당 표에서 상대적으로 가장 좋은 값
- 빨강: 해당 표에서 상대적으로 가장 약한 값
- 표의 `best/avg/worst`는 각각 4개 combo 또는 7개 residual variation 안에서의 최고/평균/최저

## TL;DR

- Residual 축에서는 여전히 `R_base`가 가장 강했다. KuaiRec 기준 최고 `test MRR@20`은 `R_base C1/C2 = 0.1622`였다.
- `shared_only`는 명확한 열화다. MoE residual을 완전히 빼면 모든 지표에서 가장 약했고, 특히 `valid MRR@20` 평균이 `0.0785`, `test MRR@20` 평균이 `0.1589`까지 떨어졌다.
- residual 재설계 중에서는 `shared_moe_warmup`과 `shared_moe_global`, `shared_moe_stage`가 가장 낫다. 이 셋은 서로 매우 비슷하지만, 최고 단일 run 기준으로는 `shared_moe_warmup C4 = test MRR@20 0.1616`가 가장 좋았다.
- combo 쪽에서는 `C1/C2`가 base에는 가장 잘 맞고, 수정 residual에는 `C3/C4`가 상대적으로 더 잘 버틴다. 즉, KuaiRec에서는 `base + standard lane`이 여전히 기준선이고, residual을 바꾸면 factored lane이 손실을 조금 줄여주는 구조다.
- routing은 `무조건 더 균등할수록 좋다`는 그림이 아니었다. `test MRR@20`과의 상관을 보면 `cv_usage`는 오히려 약한 양의 상관, `top1_max_frac`는 약한 음의 상관, `micro route jitter`는 비교적 뚜렷한 음의 상관이었다. 즉, 완전 평탄 routing보다 적당한 specialization은 괜찮지만, 세션 단위로 routing이 흔들리는 것은 손해다.
- K는 초기 sidecar snapshot만 보면 `12expert_dense`, `group_dense`가 안정권이었고 `R_base 0.1622`를 넘지 못했다. 다만 이후 rerun까지 포함하면 `group_dense C3 = 0.1623`까지 올라와서 test 최고점만 보면 base와 사실상 동급 이상이다. 대신 valid 쪽 우위는 아니라서, `강한 일반적 승리`라기보다 `K도 충분히 경쟁권`이라고 보는 편이 맞다.
- F는 sidecar 메타만으로는 빈약했지만, 별도 `F` normal/special/diag 결과를 직접 보면 `feat_full`과 `feat_feature_only`가 가장 안정적이고, `feat_hidden_only`는 명확히 약하다. 즉, feature 정보는 실제로 유효하고, 특히 `hidden-only`보다 `injection/feature path`가 더 중요한 기여를 하는 그림에 가깝다.

## Residual Variation 요약

### 1. best valid MRR@20 / avg / worst

| Variation | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `base` | `C1 0.0812` | <span style="color:#0a7f2e"><strong>0.0809</strong></span> | `C3 0.0805` |
| `shared_only` | `C1 0.0786` | <span style="color:#b42318"><strong>0.0785</strong></span> | <span style="color:#b42318"><strong>C3 0.0784</strong></span> |
| `shared_moe_fixed03` | `C1 0.0807` | `0.0804` | `C4 0.0801` |
| `shared_moe_fixed05` | `C1 0.0809` | `0.0806` | `C3 0.0803` |
| `shared_moe_global` | `C2 0.0807` | `0.0806` | `C3 0.0805` |
| `shared_moe_stage` | `C1 0.0809` | `0.0807` | `C3 0.0804` |
| `shared_moe_warmup` | `C4 0.0807` | `0.0806` | `C3 0.0803` |

### 2. best valid HR@10 / avg / worst

| Variation | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `base` | <span style="color:#0a7f2e"><strong>C2 0.1156</strong></span> | <span style="color:#0a7f2e"><strong>0.1150</strong></span> | `C4 0.1142` |
| `shared_only` | `C2 0.1097` | <span style="color:#b42318"><strong>0.1092</strong></span> | <span style="color:#b42318"><strong>C4 0.1087</strong></span> |
| `shared_moe_fixed03` | `C1 0.1146` | `0.1135` | `C4 0.1123` |
| `shared_moe_fixed05` | `C1 0.1153` | `0.1141` | `C3 0.1133` |
| `shared_moe_global` | `C2 0.1154` | `0.1137` | `C4 0.1124` |
| `shared_moe_stage` | `C1 0.1149` | `0.1141` | `C3 0.1132` |
| `shared_moe_warmup` | `C4 0.1147` | `0.1139` | `C3 0.1132` |

### 3. best test MRR@20 / avg / worst

| Variation | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `base` | <span style="color:#0a7f2e"><strong>C2 0.1622</strong></span> | <span style="color:#0a7f2e"><strong>0.1618</strong></span> | `C3 0.1612` |
| `shared_only` | `C4 0.1594` | <span style="color:#b42318"><strong>0.1589</strong></span> | <span style="color:#b42318"><strong>C1 0.1581</strong></span> |
| `shared_moe_fixed03` | `C3 0.1613` | `0.1609` | `C1 0.1605` |
| `shared_moe_fixed05` | `C3 0.1614` | `0.1611` | `C2 0.1609` |
| `shared_moe_global` | `C3 0.1615` | `0.1610` | `C2 0.1605` |
| `shared_moe_stage` | `C1 0.1613` | `0.1609` | `C2 0.1604` |
| `shared_moe_warmup` | `C4 0.1616` | `0.1610` | `C3 0.1604` |

### 4. best test HR@10 / avg / worst

| Variation | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `base` | `C1 0.1897` | <span style="color:#0a7f2e"><strong>0.1890</strong></span> | `C3 0.1884` |
| `shared_only` | `C1 0.1866` | <span style="color:#b42318"><strong>0.1860</strong></span> | <span style="color:#b42318"><strong>C4 0.1851</strong></span> |
| `shared_moe_fixed03` | `C2 0.1892` | `0.1887` | `C1 0.1883` |
| `shared_moe_fixed05` | <span style="color:#0a7f2e"><strong>C1 0.1898</strong></span> | `0.1888` | `C2 0.1871` |
| `shared_moe_global` | `C3 0.1885` | `0.1879` | `C1 0.1865` |
| `shared_moe_stage` | `C3 0.1893` | `0.1883` | `C4 0.1874` |
| `shared_moe_warmup` | `C4 0.1895` | `0.1887` | `C1 0.1881` |

## Combo 기준 요약

### 1. combo별 best valid MRR@20 / avg / worst

| Combo | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `C1` | <span style="color:#0a7f2e"><strong>base 0.0812</strong></span> | <span style="color:#0a7f2e"><strong>0.0805</strong></span> | `shared_only 0.0786` |
| `C2` | `base 0.0811` | `0.0804` | `shared_only 0.0785` |
| `C3` | `shared_moe_global 0.0805` | <span style="color:#b42318"><strong>0.0801</strong></span> | <span style="color:#b42318"><strong>shared_only 0.0784</strong></span> |
| `C4` | `shared_moe_warmup 0.0807` | `0.0802` | <span style="color:#b42318"><strong>shared_only 0.0784</strong></span> |

### 2. combo별 best valid HR@10 / avg / worst

| Combo | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `C1` | `base 0.1155` | <span style="color:#0a7f2e"><strong>0.1141</strong></span> | `shared_only 0.1095` |
| `C2` | <span style="color:#0a7f2e"><strong>base 0.1156</strong></span> | `0.1138` | `shared_only 0.1097` |
| `C3` | `base 0.1147` | <span style="color:#b42318"><strong>0.1127</strong></span> | `shared_only 0.1088` |
| `C4` | `shared_moe_warmup 0.1147` | `0.1129` | <span style="color:#b42318"><strong>shared_only 0.1087</strong></span> |

### 3. combo별 best test MRR@20 / avg / worst

| Combo | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `C1` | <span style="color:#0a7f2e"><strong>base 0.1622</strong></span> | `0.1608` | <span style="color:#b42318"><strong>shared_only 0.1581</strong></span> |
| `C2` | <span style="color:#0a7f2e"><strong>base 0.1622</strong></span> | `0.1608` | `shared_only 0.1591` |
| `C3` | `shared_moe_global 0.1615` | <span style="color:#0a7f2e"><strong>0.1609</strong></span> | `shared_only 0.1592` |
| `C4` | `shared_moe_warmup 0.1616` | <span style="color:#0a7f2e"><strong>0.1609</strong></span> | `shared_only 0.1594` |

### 4. combo별 best test HR@10 / avg / worst

| Combo | Best | Avg | Worst |
| --- | ---: | ---: | ---: |
| `C1` | <span style="color:#0a7f2e"><strong>shared_moe_fixed05 0.1898</strong></span> | `0.1882` | `shared_moe_global 0.1865` |
| `C2` | `shared_moe_fixed03 0.1892` | <span style="color:#b42318"><strong>0.1880</strong></span> | `shared_only 0.1862` |
| `C3` | `shared_moe_stage 0.1893` | <span style="color:#0a7f2e"><strong>0.1883</strong></span> | `shared_only 0.1862` |
| `C4` | `shared_moe_warmup 0.1895` | `0.1882` | <span style="color:#b42318"><strong>shared_only 0.1851</strong></span> |

## Residual 해석

### 1. 어떤 residual이 좋은가

- KuaiRec에서는 아직 `R_base (h + MoE)`가 최종 승자다. validation, test 모두 평균과 최고점에서 가장 강하다.
- `shared_only`는 분명히 좋지 않다. MoE residual을 완전히 끄면 모든 표에서 최하위권이고, `test MRR@20` 평균 기준으로 base 대비 약 `-0.0029` 손해다.
- 고정 계수는 `0.5 > 0.3`로 보인다. `shared_moe_fixed05`가 `shared_moe_fixed03`보다 valid/test 모두 소폭 우세했다.
- learnable alpha 계열에서는 `warmup`, `global`, `stage`가 거의 비슷하고, 단일 최고점은 `shared_moe_warmup C4 = 0.1616`이었다. 다만 이 값도 `base 0.1622`에는 아직 못 미친다.
- 즉, 이번 phase에서 얻은 결론은 `base를 이긴 residual 재설계는 아직 없음`, 다만 `shared_only는 버리고`, 그 다음 후보는 `warmup/global/stage` 순으로 좁혀볼 만하다는 것이다.

### 2. combo는 어떤 그림인가

- `C1/C2`는 base에 가장 잘 맞는다. 최고 `test MRR@20`이 둘 다 `0.1622`다.
- 반대로 수정 residual에서는 `C3/C4`가 상대적으로 더 강하다. `shared_moe_global` 최고점이 `C3`, `shared_moe_warmup` 최고점이 `C4`였다.
- 따라서 KuaiRec에서 combo 해석은 이렇게 보는 게 맞다.
  - `원래 base를 가장 세게 밀어주는 lane`: `C1/C2`
  - `residual을 바꿨을 때 손실을 제일 적게 줄여주는 lane`: `C3/C4`
- 즉 `factored lane`이 절대 우세한 건 아니고, residual 변경과 함께 볼 때만 의미가 커진다.

## Valid/Test 차이

- `test MRR@20 - valid MRR@20` 범위는 `0.0795 ~ 0.0811`, 평균은 `0.0805`였다.
- `test HR@10 - valid HR@10` 범위는 `0.0720 ~ 0.0774`, 평균은 `0.0748`이었다.
- gap 최저는 `shared_only C1`, 최고는 `base C2`였다. 차이가 크긴 하지만 residual 사이 편차는 아주 크지 않다.
- 따라서 이번 phase에서는 `특정 residual이 유독 overfit`이라기보다, 아예 dataset split 자체가 `test가 valid보다 쉬운 구조`에 가깝다고 보는 편이 맞다.
- special 로그를 보면 실제로 valid/test slice 절대값 차이가 크다. 예를 들어 `R_base C2`의 `<=5 target popularity`는 valid `0.0343`인데 test는 `0.1212`였다.
- 가능한 해석은 다음 정도다.
  - strict positive split에서 valid가 더 hard-negative 성격을 띠고 있을 가능성
  - valid/test에서 인기 bin 구성과 긴 세션 비중이 꽤 달라서 절대값이 달라질 가능성
  - 현재 special bin 정의가 valid/test difficulty 차이를 크게 반영하고 있을 가능성
- 따라서 이 phase 결과는 `절대 valid/test 차이 크기`보다 `residual 간 상대 순위`를 더 신뢰하는 게 안전하다.

## Special Logging 해석

대표 run 원본을 직접 확인했다.
- `R_base C2`
- `R_shared_only C1`
- `R_shared_moe_warmup C1`
- `R_shared_moe_global C3`

대표 관찰:
- `R_base C2` test slice
  - `<=5 popularity mrr@20 = 0.1212`
  - `6-20 popularity mrr@20 = 0.1487`
  - `21-100 popularity mrr@20 = 0.1769`
  - `3-5 session_len mrr@20 = 0.1523`
  - `6-10 session_len mrr@20 = 0.1516`
  - `11+ session_len mrr@20 = 0.1710`
- `R_shared_moe_warmup C1` test slice
  - `<=5 = 0.1183`
  - `6-20 = 0.1493`
  - `21-100 = 0.1745`
  - `3-5 = 0.1501`
  - `6-10 = 0.1505`
  - `11+ = 0.1707`
- `R_shared_moe_global C3` test slice
  - `<=5 = 0.1178`
  - `6-20 = 0.1484`
  - `21-100 = 0.1770`
  - `3-5 = 0.1505`
  - `6-10 = 0.1515`
  - `11+ = 0.1706`
- `R_shared_only C1` test slice
  - `<=5 = 0.1022`
  - `6-20 = 0.1480`
  - `21-100 = 0.1758`
  - `3-5 = 0.1459`
  - `6-10 = 0.1479`
  - `11+ = 0.1678`

해석:
- `shared_only`의 가장 큰 손해는 가장 차가운 tail(`<=5`)과 짧은 세션(`3-5`, `6-10`)에서 나온다. 즉 MoE residual을 완전히 빼면 어려운 케이스에서 가장 먼저 무너진다.
- 반대로 `6-20` 구간은 `warmup C1`이 가장 좋았고, `21-100` 구간은 `global C3`가 가장 좋았다. 즉 learnable residual은 중간 인기대에서는 의미가 있다.
- 하지만 모든 slice를 동시에 제일 안정적으로 가져가는 건 여전히 `base C2` 쪽이다. 특별히 어떤 residual이 특정 slice를 크게 뒤집었다기보다, `warmup/global`이 일부 구간에서 base에 근접하거나 살짝 앞서는 그림이다.

주의:
- 일부 trial 내부의 `valid_special_metrics/test_special_metrics` 필드가 비어 있는 경우가 있었다. 그런데 파일 최상단 집계값은 정상이다.
- 그래서 per-trial 세부 special parsing은 노이즈가 있었고, 본 문서는 파일 최상단 집계와 대표 원본 파일 직접 확인을 우선했다.

## Routing / Diag 해석

sidecar의 `diag` 본문은 비어 있었지만, run result JSON 자체에 trial별 routing 통계가 들어 있어 그것으로 비교했다. representative best run 기준으로 보면:

| Variation | Combo | test MRR@20 | alpha `(macro, mid, micro)` | n_eff `(macro, mid, micro)` | top1 max frac `(macro, mid, micro)` | micro jitter |
| --- | --- | ---: | --- | --- | --- | ---: |
| `base` | `C1` | `0.1622` | `(1.0, 1.0, 1.0)` | `(11.34, 8.97, 11.32)` | `(0.174, 0.296, 0.137)` | `0.323` |
| `shared_only` | `C4` | `0.1594` | `(0.0, 0.0, 0.0)` | `(11.99, 11.99, 11.99)` | `(0.135, 0.209, 0.141)` | `0.565` |
| `shared_moe_fixed03` | `C3` | `0.1613` | `(0.300, 0.300, 0.300)` | `(10.99, 9.40, 9.77)` | `(0.224, 0.380, 0.192)` | `0.330` |
| `shared_moe_fixed05` | `C3` | `0.1614` | `(0.500, 0.500, 0.500)` | `(11.41, 9.50, 9.36)` | `(0.199, 0.387, 0.180)` | `0.307` |
| `shared_moe_global` | `C3` | `0.1615` | `(0.568, 0.568, 0.568)` | `(11.44, 9.02, 9.21)` | `(0.158, 0.378, 0.264)` | `0.343` |
| `shared_moe_stage` | `C1` | `0.1613` | `(0.548, 0.574, 0.557)` | `(10.90, 10.34, 10.91)` | `(0.266, 0.183, 0.154)` | `0.303` |
| `shared_moe_warmup` | `C4` | `0.1616` | `(0.577, 0.595, 0.596)` | `(11.55, 10.43, 10.06)` | `(0.191, 0.289, 0.214)` | `0.336` |

이 표에서 보이는 핵심은 다음이다.

- `shared_only`는 expert usage 자체는 가장 평평하다. `n_eff`가 거의 12에 붙고 `top1 max frac`도 낮다. 그런데 성능은 최악이고 `micro route jitter = 0.565`로 압도적으로 높다.
- 즉 KuaiRec에서는 `균등한 routing` 그 자체가 목표가 아니다. MoE residual을 빼버리면 routing은 넓게 퍼져도 실제 유용한 specialization이 사라지고, 세션 단위 안정성도 나빠진다.
- `fixed03 -> fixed05 -> global/stage/warmup`으로 갈수록 alpha가 `0.3 -> 0.5 -> 약 0.55~0.60`까지 올라간다. 결국 이 데이터에서는 residual 내 MoE 비중을 너무 낮게 두는 것보다 `중간 이상`으로 두는 편이 낫다.
- `stage`는 특히 mid stage를 매끈하게 만든다. representative best run에서 `mid n_eff = 10.34`, `mid top1 max frac = 0.183`으로 가장 균형이 좋다.
- `warmup`은 alpha를 최종적으로 `0.58~0.60` 수준까지 끌고 가면서도 최고 수정 residual 점수를 냈다. 즉 `처음엔 shared 위주로 가다가 이후 MoE residual을 올리는` 전략이 KuaiRec에서는 가장 자연스러운 절충으로 보인다.

상관 계수도 비슷한 그림이다.

### test MRR@20 와 routing 지표의 상관

- `macro_1.n_eff = -0.1037`
- `mid_1.n_eff = -0.2800`
- `micro_1.n_eff = -0.2945`
- `macro_1.cv_usage = 0.2940`
- `mid_1.cv_usage = 0.3373`
- `micro_1.cv_usage = 0.4522`
- `macro_1.top1_max_frac = -0.2817`
- `mid_1.top1_max_frac = -0.0952`
- `micro_1.top1_max_frac = -0.0884`
- `micro_1.route_jitter_session = -0.3656`

해석:
- `n_eff`가 높다고 성능이 오르지 않았다. 오히려 약한 음의 상관이라서, 지나치게 flat한 expert usage는 이 데이터에서 이점이 크지 않았다.
- `cv_usage`는 약한 양의 상관이었다. 약간의 imbalance, 즉 expert specialization이 있는 편이 오히려 유리할 가능성이 있다.
- 하지만 `top1_max_frac`는 다시 음의 상관이라서, 한 expert로 몰리는 수준의 concentration까지 가면 안 좋다.
- 가장 일관된 악재는 `micro route jitter`였다. 세션 단위로 micro routing이 흔들릴수록 성능이 떨어졌다.
- 즉 좋은 residual은 `완전 균등`도 아니고 `collapse`도 아니며, `적당히 specialized + 세션 단위로 안정적`인 routing을 만드는 쪽이다.

### valid-test gap 과 routing 지표의 상관

- `gap_mrr20` 최저: `0.0795` at `shared_only C1`
- `gap_mrr20` 최고: `0.0811` at `base C2`
- `gap_hr10` 최저: `0.0720` at `shared_moe_global C1`
- `gap_hr10` 최고: `0.0774` at `shared_only C3`

`gap_mrr20`과의 상관은 다음 경향을 보였다.
- `macro_1.n_eff = 0.5761`
- `macro_1.top1_max_frac = -0.7022`
- `micro_1.route_jitter_session = 0.4803`

이건 test split이 더 쉬운 구조라는 점과 함께 보면, routing이 더 넓게 퍼지는 모델이 test에서 더 큰 상승폭을 얻는 경향이 있다는 정도로 해석하는 게 안전하다. 이 값을 `일반화가 더 좋다`라고 단정하는 것은 과하다.

## Feature / K / F 종합 메모

- `feature_ablation` 메타 자체는 sidecar 안에서 여전히 빈약하다. 즉, residual 파트처럼 `번들 내부 feature delta`만으로는 결론을 내릴 수 없다.
- 대신 이번 문서에서는 `F 축 전용 run`을 사실상 feature ablation proxy로 보고 직접 읽었다.
- 또한 K/F는 초기 sidecar snapshot 이후에 일부 rerun이 더 들어와 있어서, 이 파트는 `초기 sidecar`와 `최신 normal/special/diag`를 함께 봐야 정확하다.

## K 축 분석

### 1. 간단 요약

- K 축의 중심 결론은 `더 많은 expert/group 구조가 실제로 경쟁력이 있다`는 것이다.
- 최신 결과까지 포함하면 최고 `test MRR@20`은 `group_dense C3 = 0.1623`으로, `R_base` 최고점 `0.1622`와 사실상 동급 이상이다.
- 다만 `valid MRR@20` 기준으로는 K가 뚜렷하게 앞서지 않는다. 즉, `test에서 잘 맞은 top run`은 있지만 `robust baseline이 완전히 바뀌었다`고 말할 정도는 아니다.
- 전체적으로는 `4expert < 12expert ~= group` 구도가 명확했고, 특히 `group_dense`, `group_top2`, `12e_dense`, `12e_top6`가 안정적이었다.

### 2. K variation별 test MRR@20 추세

| K Variation | Avg | Best | Worst |
| --- | ---: | ---: | ---: |
| `group_dense` | <span style="color:#0a7f2e"><strong>0.1619</strong></span> | <span style="color:#0a7f2e"><strong>C3 0.1623</strong></span> | `C2 0.1616` |
| `group_top2` | `0.1618` | `C4 0.1621` | `C3 0.1614` |
| `12e_dense` | `0.1618` | `C3 0.1621` | `C1 0.1615` |
| `12e_top3` | `0.1617` | `C4 0.1621` | `C2 0.1610` |
| `12e_top6` | `0.1616` | `C1 0.1621` | `C4 0.1609` |
| `4e_top2` | `0.1616` | `C3 0.1619` | `C2 0.1611` |
| `groupignore_global6` | `0.1615` | `C1 0.1620` | `C3 0.1607` |
| `4e_dense` | `0.1613` | `C2 0.1619` | `C3 0.1607` |
| `group_top1` | `0.1613` | `C4 0.1617` | `C1 0.1605` |
| `4e_top1` | <span style="color:#b42318"><strong>0.1610</strong></span> | `C4 0.1618` | <span style="color:#b42318"><strong>C1 0.1604</strong></span> |

해석:
- `4expert`는 capacity 한계가 보인다. 특히 `top1`은 sparse routing 자체는 가능하지만 평균 성능이 제일 약하다.
- `12expert`에서는 `dense/top3/top6` 차이가 작아진다. 즉, expert 수가 충분해지면 top-k sparsity 자체보다 expert pool 크기가 더 중요해진다.
- `group` 계열은 잘 맞을 때 최고점이 가장 높다. `group_dense C3`가 대표적이다.
- 대신 `group_top1`은 이득이 줄고 분산이 커졌다. group 구조 위에 top1까지 겹치면 specialization이 좋아지기보다 정보 손실이 더 커지는 것으로 보인다.

### 3. valid/test 관계

- K 축에서 test 최고점은 `group_dense C3 = 0.1623`이지만, 같은 run의 `valid MRR@20 = 0.0805`로 validation 최고권은 아니다.
- 반대로 validation 상위권은 `12e_top3 C1`, `12e_top6 C2`, `12e_dense C4`처럼 12expert family가 가져갔다.
- 즉 K에서는 `validation strong`과 `test top`이 완전히 같은 family로 수렴하지 않았다.
- 이건 `K가 특정 test regime에는 잘 맞지만 split 전반에서 매우 안정적인지는 아직 애매하다`는 신호다. 따라서 Phase 5에서 K를 다시 볼 때는 평균/seed 안정성을 같이 보는 게 중요하다.

### 4. K special slice 분석

대표 포인트:
- cold item(`target_popularity_abs <= 5`) 최고: `group_top2 C4 = 0.1226`
- short session(`session_len 3-5`) 최고: `group_dense C3 = 0.1521`
- cold item 최저: `4e_top1 C2 = 0.1144`

해석:
- `group_top2`와 `group_dense`가 cold/short regime에서 가장 낫다. 이건 더 큰 expert pool과 적당한 specialization이 tail regime에서 실제로 도움이 된다는 뜻이다.
- 반대로 `4e_top1`은 sparse routing이 너무 공격적이라 cold item에서 크게 무너진다.
- `12e_top6 C1`도 cold slice(`0.1220`)와 중간 popularity(`6-20`, `0.1489`)에서 꽤 좋았고, `12e_dense C2`는 head 쪽보다 cold/mid-tail에서 고르게 버티는 쪽에 가깝다.
- 정리하면 K의 진짜 장점은 `전체 평균을 아주 크게 올리는 것`보다 `cold/short slice를 덜 잃고, 때로는 더 올려 주는 것`에 있다.

### 5. K routing / diag 해석

diag가 잘 남은 대표 run을 보면 다음 경향이 보였다.

- `group_dense C3` (`test MRR@20 = 0.1623`)
  - `n_eff = (11.09, 10.52, 9.49)`
  - `cv_usage = (0.287, 0.375, 0.514)`
  - `top1_max_frac = (0.260, 0.268, 0.195)`
  - `micro jitter = 0.283`
- `4e_top2 C2` (`test MRR@20 = 0.1622`)
  - `n_eff = (3.93, 3.38, 3.57)`
  - `cv_usage = (0.137, 0.428, 0.348)`
  - `top1_max_frac = (0.307, 0.479, 0.415)`
  - `micro jitter = 0.162`

이 둘을 같이 보면:
- `4e_top2 C2`는 jitter는 더 낮지만 expert 풀이 너무 작고 mid/micro 집중이 더 세다.
- `group_dense C3`는 jitter가 아주 낮지는 않아도, 더 넓은 expert pool과 적당한 imbalance를 유지하면서 최고 test 성능을 냈다.
- 즉 K에서 중요한 건 `무조건 jitter 최소`가 아니라, `충분한 capacity + 과도하지 않은 concentration + 감당 가능한 jitter`의 조합이다.
- 특히 `group_dense`는 micro에서만 적당히 specialization이 생기고 macro/mid는 비교적 정돈된 형태라, Phase 4 전체 결론과도 잘 맞는다.

## F 축 분석

### 1. 간단 요약

- F 축은 사실상 feature usage ablation을 간접 비교한 파트로 볼 수 있다.
- 최신 결과 기준으로는 `feat_full`과 `feat_feature_only`가 가장 강했고, `feat_hidden_only`는 명확히 약했다.
- 최고 단일 run만 놓고 보면 `feat_full C1`이 가장 안정적인 기준선이고, `feat_feature_only C4`가 바로 뒤를 붙는다.
- `feat_injection_only C3`도 생각보다 잘 버텼다. 즉, feature 정보는 실제로 유효하며, 그 효과는 `hidden router만`보다 `feature injection / explicit feature path` 쪽에서 더 크게 살아난다.

### 2. F variation별 성능

아래 값은 최신 `normal/special/diag` 기준으로 읽은 것이다.

| F Variation | Valid MRR@20 | Test MRR@20 | Test HR@10 | 해석 |
| --- | ---: | ---: | ---: | --- |
| `feat_full C1` | <span style="color:#0a7f2e"><strong>0.0810</strong></span> | <span style="color:#0a7f2e"><strong>0.1622</strong></span> | <span style="color:#0a7f2e"><strong>0.1890</strong></span> | 가장 균형이 좋음 |
| `feat_feature_only C4` | `0.0801` | `0.1620` | `0.1880` | 성능 거의 유지, 안정성은 다소 손해 |
| `feat_injection_only C3` | `0.0802` | `0.1617` | `0.1891` | feature path만으로도 상당 부분 유지 |
| `feat_hidden_only C2` | <span style="color:#b42318"><strong>0.0791</strong></span> | <span style="color:#b42318"><strong>0.1602</strong></span> | `0.1890` | ranking quality가 가장 약함 |

해석:
- `hidden_only`가 가장 약하다는 건, router가 hidden representation만 보고는 feature regime를 충분히 분리하지 못한다는 뜻에 가깝다.
- 반면 `feature_only`와 `injection_only`가 둘 다 `0.1617~0.1620`까지 나온 건 feature 신호가 실제로 강하고, explicit conditioning이 가치 있다는 뜻이다.
- 그래도 최종 기준선은 `feat_full`이다. 즉, feature path가 중요하지만 hidden을 완전히 버리는 것보다는 `둘 다 쓰는 편`이 가장 안정적이다.

### 3. F special slice 분석

특히 눈에 띄는 차이는 cold/short regime였다.

- `feat_full`
  - `pop <= 5 = 0.1215`
  - `pop 21-100 = 0.1762`
  - `len 6-10 = 0.1521`
  - `len 11+ = 0.1712`
- `feat_feature_only`
  - `pop <= 5 = 0.1217` (cold 쪽은 오히려 가장 좋음)
  - `len 3-5 = 0.1521` (short-mid session도 우수)
- `feat_hidden_only`
  - `pop <= 5 = 0.1119`
  - `len 3-5 = 0.1491`
  - `len 6-10 = 0.1496`
  - `len 11+ = 0.1696`

이걸 보면:
- `hidden_only`는 cold item에서 가장 크게 무너진다. head item(`>100`)에서는 오히려 `0.9415`로 잘 버티지만, 이건 쉬운 구간이어서 강점으로 보긴 어렵다.
- `feature_only`는 cold/short regime를 꽤 잘 지킨다. 즉, feature 자체가 tail 대응에 직접 기여하고 있다는 뜻이다.
- `full`은 cold 최고는 아니어도 mid-popularity, 중간 길이 세션, 긴 세션까지 전체적으로 가장 고르게 좋다. 그래서 실제 운용 기준선으로는 `feat_full`이 가장 설득력 있다.

### 4. F routing / diag 해석

diag가 비교적 잘 남은 두 run은 다음과 같다.

- `feat_full C1` (`test MRR@20 = 0.1622`)
  - `n_eff = (11.49, 9.66, 11.20)`
  - `cv_usage = (0.210, 0.492, 0.267)`
  - `top1_max_frac = (0.212, 0.317, 0.172)`
  - `micro jitter = 0.311`
- `feat_feature_only C4` (`test MRR@20 = 0.1620`)
  - `n_eff = (11.82, 9.22, 7.73)`
  - `cv_usage = (0.124, 0.549, 0.743)`
  - `top1_max_frac = (0.364, 0.382, 0.355)`
  - `micro jitter = 0.453`

해석:
- `feature_only`는 성능을 거의 유지하지만, micro 단계 concentration과 jitter가 더 크다.
- `full`은 비슷한 성능을 더 낮은 concentration, 더 낮은 micro jitter로 낸다.
- 즉 feature만으로도 분류는 가능하지만, hidden까지 같이 보는 `full`이 더 안정적인 specialization을 만든다고 볼 수 있다.
- 이 결과는 Phase 5 방향과도 잘 맞는다. 앞으로는 `feature를 없앨지 말지`보다 `feature를 어떻게 더 안정적으로 routing에 반영할지`가 핵심이다.

## 이번 데이터셋에서의 실무적 결론

- KuaiRec에서는 residual을 굳이 바꿔야 한다면 `shared_moe_warmup`이 1순위, `shared_moe_global`과 `shared_moe_stage`가 그 다음이다.
- 다만 Phase 4 전체를 다 보면 `R_base`만 남는 그림은 아니다. K에서는 `group_dense C3 = 0.1623`까지 올라와 test 최고점은 사실상 동급 이상이었고, cold/short slice에서도 강점이 있었다.
- 그래도 `배포/기준선` 관점에서는 여전히 `R_base + C1/C2` 혹은 `feat_full`이 가장 안전하다. K 최상위 run은 좋지만 validation 우위와 재현성 증거가 더 필요하다.
- `shared_only`는 제외해도 된다. 성능도 낮고, routing도 너무 flat하면서 jitter가 커서 해석상 이점도 없다.
- K 쪽 후속 우선순위는 `group_dense`, `group_top2`, `12e_dense`, `12e_top6` 정도로 좁히는 게 맞다. `4e_top1`은 공격적인 sparse routing 대비 얻는 게 적다.
- F 쪽에서는 `feat_hidden_only`를 사실상 탈락시켜도 된다. 반대로 `feat_full`과 `feat_feature_only`, `feat_injection_only` 비교는 꽤 의미가 있었다. 여기서는 `feature가 필요하냐`보다 `feature를 안정적으로 주입하느냐`가 더 중요한 질문이다.
- `더 balance 맞을수록 무조건 성능 증가`는 아니었다. 적당한 imbalance는 괜찮지만, session-level jitter는 줄이는 쪽이 확실히 중요했다.
- 다음 residual 후속은 `warmup/global/stage`만 남기고, `alpha 범위/초기화/warmup 스케줄`을 더 좁혀 보는 것이 맞다. 다만 Phase 5에서는 residual 자체보다 `specialization을 살리면서 jitter와 monopoly를 제어하는 logging/aux-reg` 쪽이 더 우선순위가 높다.
