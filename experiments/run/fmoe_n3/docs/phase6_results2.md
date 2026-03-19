# Phase 6 Results v2 (Beginner-Friendly + 1-Page Presentation Ready)

작성일: 2026-03-18  
데이터셋: KuaiRecLargeStrictPosV2_0.2  
메인 지표: **best val MRR@20**  
보조 지표: test MRR@20, special slice(cold/short), diag(routing)

---

## 1) 먼저 용어부터 (처음 보는 사람용)

### 1.1 Phase6에서 자주 나온 이름 뜻
| 이름 | 의미 | 실제로 바뀌는 것 |
|---|---|---|
| `X1` | Router x Injection 실험의 Context 1 | base residual + topk=1 + global_flat |
| `X2` | Router x Injection 실험의 Context 2 | warmup residual + topk=2 + group_top2_pergroup |
| `M0` | 정규화 ablation 모드 0 | specialization 유도 정규화를 거의 끈 baseline(하드 off) |
| `M1` | 정규화 ablation 모드 1 | smoothness만 강하게 |
| `M2` | 정규화 ablation 모드 2 | consistency만 강하게 |
| `M3` | 정규화 ablation 모드 3 | sharp + monopoly 강화 |
| `M4` | 정규화 ablation 모드 4 | prior + consistency + smoothness 혼합 강정규화 |
| `B0` | baseline bridge의 SASRec-equivalent | 사실상 SASRec 쪽 기준선 |
| `B2` | hidden-only MoE | hidden source만 MoE routing |
| `B4` | feature-only MoE | feature source만 MoE routing |

### 1.2 special / diag 지표 뜻
| 지표 | 해석 |
|---|---|
| `cold_item_mrr@20` | test에서 저빈도 아이템(<=5) 구간 MRR@20 |
| `sess_3_5_mrr@20`, `sess_11p_mrr@20` | 짧은/긴 세션 구간 성능 비교 |
| `diag_micro_cons` | micro stage routing consistency (높을수록 라우팅 일관) |
| `diag_micro_feat_cons` | feature group 기준 routing consistency |
| `diag_micro_jitter` | 인접 시점 라우팅 변동성 (낮을수록 안정) |

주의: `new_user_enabled=false`라 cold user slice는 이번 로그에 없음. 이번 문서는 cold item/short session으로 해석.

---

## 2) 전체 요약 (메인=best val)

- 전체 run 디렉터리: 79개 (재시도 포함)
- combo dedup: 55개 (계획 combo 모두 관측)
- suite별 평균(best val):
  - cand3x: **0.0809**
  - baseline_bridge: **0.0798**
  - router2x2: **0.0804**
  - spec_ablation: **0.0808**
  - feature_ablation: **0.0797**

해석 요약:
- best val 기준으론 `cand/spec/router` 상위 조합이 비슷한 레벨에서 경쟁.
- test/cold 관점까지 같이 보면 `spec`과 일부 `router(X1)`가 설득력 있음.
- feature ablation은 절대 val 고점은 낮지만, special/diag(특히 feature-consistency) 증거가 강함.

---

## 3) Baseline 3종 비교 (논문 설득용 핵심)

비교 대상:
- SASRec-equivalent: `B0`
- hidden-only MoE: `B2`
- feature-only MoE: `B4`

### 3.1 성능 + special 비교
| 모델 | best val MRR@20 | test MRR@20 | cold item MRR@20 | sess_3_5 MRR@20 | sess_11+ MRR@20 |
|---|---:|---:|---:|---:|---:|
| B0 (SASRec-eq) | 0.0782 | 0.1588 | 0.1087 | 0.1470 | 0.1684 |
| B2 (hidden-only MoE) | 0.0793 | 0.1601 | 0.1110 | 0.1489 | 0.1693 |
| B4 (feature-only MoE) | 0.0807 | 0.1617 | 0.1213 | 0.1515 | 0.1705 |

### 3.2 B0 대비 개선량
| 모델 | Δ best val | Δ test | Δ cold item | Δ sess_3_5 | Δ sess_11+ |
|---|---:|---:|---:|---:|---:|
| B2 vs B0 | +0.0011 | +0.0013 | +0.0023 | +0.0019 | +0.0009 |
| B4 vs B0 | **+0.0025** | **+0.0029** | **+0.0126** | **+0.0045** | **+0.0021** |

### 3.3 diag 비교
| 모델 | diag_micro_cons | diag_micro_feat_cons | diag_micro_jitter | 해석 |
|---|---:|---:|---:|---|
| B0 | NA | NA | NA | SASRec-equivalent run에 해당 diag 산출물 없음 |
| B2 | 1.0000 | 1.0000 | 0.3595 | 매우 안정/일관하지만 cold 개선은 제한적 |
| B4 | 0.9946 | 0.9853 | 0.6280 | 라우팅 변동성은 크지만 cold 성능 이득이 큼 |

핵심 메시지:
- **SASRec -> hidden-only -> feature-only**로 갈수록 성능/특히 cold item이 커짐.
- feature-only는 jitter 증가를 감수하고 어려운 slice(cold)에서 이득을 가져가는 형태.
- 논문 포인트로는 "feature 기반 specialization이 어려운 샘플에서 유의미한 이득"이 가장 설득력 있음.

---

## 4) 축(변경 단위)별로 보면 무엇이 좋았나

## 4.1 Router 축 (X1 vs X2가 핵심)
| 그룹 | best val avg | test avg | cold item avg | diag_micro_cons avg | diag_feat_cons avg | jitter avg |
|---|---:|---:|---:|---:|---:|---:|
| `router_context=x1` | **0.08075** | **0.16175** | **0.12088** | **0.95133** | **0.91324** | **0.48529** |
| `router_context=x2` | 0.08008 | 0.15820 | 0.11171 | 0.93805 | 0.90696 | 0.57077 |

해석:
- X1이 X2보다 성능/콜드/diag 안정성 모두 우세.
- router type(`sta` vs `fac`)보다 context(X1/X2) 영향이 훨씬 큼.

## 4.2 Specialization 정규화 축 (M0~M4)
| 모드 | best val avg | test avg | cold item avg | diag_micro_cons avg | diag_feat_cons avg | jitter avg |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.08095 | **0.16210** | 0.12033 | 0.96838 | 0.93999 | 0.48875 |
| M1 | 0.08085 | 0.16185 | 0.12058 | 0.97255 | 0.94639 | 0.46599 |
| M2 | 0.08080 | 0.16190 | 0.12016 | **0.98258** | **0.96874** | 0.58622 |
| M3 | 0.08080 | 0.16180 | **0.12137** | 0.95359 | 0.91503 | **0.35233** |
| M4 | 0.08060 | 0.16080 | 0.11535 | 0.97124 | 0.95007 | 0.52096 |

해석:
- M0: 전체 test 고점 쪽.
- M3: cold item, jitter 안정 쪽.
- M4: 성능/콜드 모두 하락 -> 과정규화 가능성 높음.

## 4.3 Feature 축
| 그룹 | best val avg | test avg | cold item avg | diag_feat_cons avg |
|---|---:|---:|---:|---:|
| window W5 | **0.07975** | **0.16135** | 0.11942 | 0.98463(8개중) |
| window W10 | 0.07967 | 0.16128 | 0.11954 | 0.98499(10개중) |
| mask size 1 | 0.07939 | 0.16116 | 0.11781 | 0.98458 |
| mask size 2 | **0.07993** | **0.16142** | **0.12055** | **0.98499** |
| family 포함=tempo | 0.07976 | 0.16139 | 0.11999 | 0.98398 |
| family 포함=memory | **0.07985** | **0.16156** | **0.12053** | **0.98679** |

해석:
- 단일 family보다 2-family 조합이 일관 우세.
- tempo/memory 포함 조합이 가장 안정적으로 좋음.

---

## 5) Special + Diag + 성능의 관계 (상관분석)

기준: combo dedup 55개, 목표변수 = best val MRR@20

### 5.1 전체 상관
| 변수 | Pearson r | Spearman rho | 해석 |
|---|---:|---:|---|
| test MRR@20 | +0.374 | +0.501 | val과 test는 중간 이상 양의 연관 |
| cold item MRR@20 | +0.546 | +0.423 | val이 높은 조합은 cold도 대체로 좋음 |
| diag_micro_cons | -0.291 | -0.424 | 일관성만 높다고 val이 높아지진 않음 |
| diag_micro_feat_cons | -0.535 | -0.578 | feature-consistency 고점이 성능 고점과 분리되는 경향 |
| diag_micro_jitter | -0.087 | -0.097 | 전체에선 약한 음의 관계 |

### 5.2 왜 음의 상관이 나왔나 (중요 해석)
- suite 간 confounding이 큼.
- 예: feature_ablation은 diag_feat_cons가 매우 높은데 val 고점은 상대적으로 낮음.
- 즉, diag 지표는 "성능 그 자체"보다 "specialization 스타일"을 보여주는 증거로 보는 게 타당.

### 5.3 suite 내부 상관 (요약)
| suite | val vs cold | val vs diag_cons | val vs diag_feat_cons | val vs jitter |
|---|---:|---:|---:|---:|
| spec_ablation | +0.553 | -0.182 | -0.183 | -0.085 |
| feature_ablation | **+0.694** | **-0.665** | -0.373 | -0.406 |
| router2x2 | +0.775 | +0.240 | -0.175 | **-0.587** |
| baseline_bridge | +0.929 | -0.580 | -0.651 | +0.186 |

핵심:
- 대부분 축에서 val과 cold는 강한 양의 관계.
- diag는 "높을수록 무조건 좋다"가 아니라, 성능과 trade-off를 가진다.

---

## 6) Specialization 관점 결론

이번 phase6에서 보이는 specialization 패턴:

1. **feature source를 쓰면 cold slice 이득이 커짐**
- B4가 B0 대비 cold item +0.0126로 가장 큼.

2. **정규화는 목적별로 모드가 다름**
- M0는 전체 점수, M3는 cold/jitter 안정성에 강점.

3. **feature ablation은 2-family 조합이 유리**
- mask size=2가 val/test/cold를 모두 끌어올림.

4. **diag는 설명 변수이지 단일 목적함수 아님**
- feature-consistency를 최대로 만들면 오히려 val이 떨어질 수 있음.

실전적으로는:
- "X1 context + (A-anchor) M0/M3 절충 + tempo/memory 2-family"가 다음 실험의 가장 유망한 출발점.

---

## 7) 한 장짜리 발표자료(슬라이드 1장) 설계안

목표: "FMoE specialization이 단순 점수 상승이 아니라, 어려운 slice(cold)와 routing 특성 변화를 동반한다"를 한 장에서 설득.

### 7.1 넣을 그래프/표 (딱 1장 구성)
| 영역 | 시각화 | 데이터 | 메시지 | 구현 팁 |
|---|---|---|---|---|
| 좌상단(핵심 성능) | 막대그래프 1 | B0/B2/B4의 best val, test | SASRec -> hidden-only -> feature-only로 성능 상승 | bar 2개씩 묶음(best val 크게, test 얇게) |
| 우상단(어려운 slice) | 막대그래프 2 | B0/B2/B4의 cold item MRR@20 | feature-only에서 cold 개선이 크게 발생 | delta 라벨(+0.0126) 직접 표기 |
| 좌하단(정규화 trade-off) | 산점도 1 | spec(M0~M4, A/B)에서 x=diag_feat_cons, y=best val, 색=cold | 정규화 모드별 specialization trade-off | 점 라벨 M0/M3/M4 |
| 우하단(축 요약 표) | 표 1 | X1 vs X2, mask1 vs mask2, M0 vs M3 | 다음 실험에 가져갈 변화 제안 | up/down 화살표로 요약 |
| 하단 한 줄 | conclusion strip | 2~3 문장 | "feature 기반 specialization이 cold 개선을 견인" | bold로 결론 고정 |

### 7.2 슬라이드용 표(바로 복붙용)
| 비교축 | 더 좋은 선택 | 근거 수치(메인=best val) | 같이 보여줄 보조 지표 |
|---|---|---|---|
| Baseline bridge | B4(feature-only) | best val 0.0807 (vs B0 0.0782) | cold 0.1213 (vs 0.1087), test 0.1617 |
| Router context | X1 | val avg 0.08075 (vs X2 0.08008) | test 0.16175 (vs 0.15820), jitter 0.485 (vs 0.571) |
| Spec mode | M0 or M3 | M0 test 최상(0.1621), M3 cold 최상(0.1214) | M3 jitter 최저(0.352) |
| Feature mask | 2-family | val avg 0.07993 (vs 1-family 0.07939) | cold 0.12055 (vs 0.11781) |

### 7.3 발표 멘트 예시 (30초)
- "메인 지표(best val) 기준으로도 feature source를 넣은 MoE가 기준선보다 낫고, 특히 cold item에서 개선폭이 큽니다."
- "다만 diag는 단조 관계가 아니라 trade-off를 보여서, M0는 전체 성능, M3는 cold/안정성에 강점이 있습니다."
- "그래서 다음 실험은 X1 context를 고정하고, A-anchor에서 M0/M3 절충 + tempo/memory 2-family를 교차하는 방향이 가장 효율적입니다."

---

## 8) 다음 실험 제안 (짧게)

1. **X1 + A-anchor 고정 실험**
- M0/M3 중간 강도 3~4개 포인트만 sweep

2. **tempo+memory 2-family를 기본값으로 고정**
- spec 모드만 바꿔 cold/test 균형 확인

3. **B0/B2/B4를 동일 seed로 재검증 1회**
- baseline 3종 비교를 논문 그림용으로 더 깔끔히 고정
