# 260319 Presentation Plan (FMoE_N3 on KuaiRec 중심)

## 0. 발표 목표와 한 줄 메시지
- 목표: KuaiRec 기준으로 FMoE_N3의 개선 경로를 축별로 명확히 보여주고, 왜 현재 best가 나왔는지 구조적 근거를 제시한다.
- 한 줄 메시지: Feature schema(v3) + stage/group-aware MoE 설계 + residual/aux 조합이 결합되면서, baseline 대비 일관된 MRR@20 개선과 해석 가능한 routing 특성을 확보했다.
- 필수 비교 맥락: SASRec baseline 대비 성능/안정성/복잡도.

## 1. 권장 발표 순서 (총 25~35분)

### Part A. Problem Setup and Data (5~7분)
1. 왜 KuaiRecLargeStrictPosV2_0.2 + lastfm0.03인가
- KuaiRec: 스케일과 분포 편향이 커서 구조적 차이를 보기 좋음.
- lastfm0.03: 상대적으로 다른 도메인/밀도로 일반화 경향 체크.
- 슬라이드 1장: 두 데이터셋의 핵심 스탯 표(세션 수, 아이템 수, 평균 세션 길이, sparsity).

2. dataset/feature 파이프라인 재구성 소개
- feature_added_v3 스키마(4 stage x 4 family x 4 features = 64) 설계 철학.
- sampling 비율/방법, KuaiRec aggressive drop의 이유와 현재 상태.
- 슬라이드 1장: 전처리 파이프라인 다이어그램(raw -> sampling/drop -> split -> feature v3).

3. split 정책과 validation-test gap 해석
- 현재 split: session 내부 70/15/15 (LOO 근사 의도였으나 현재 구현은 intra-session split).
- 관찰된 현상: Kuai에서 test가 valid보다 높게 나오는 구조적 이유(유효 prefix 길이 차이).
- 슬라이드 1장: valid/test prefix 길이 분포 비교 히스토그램 + 간단한 사례 타임라인.

### Part B. Experimental Snapshot First (3~4분)
4. 결과 선요약 (먼저 결론)
- 단일 표: SASRec vs 초기 FMoE vs 현재 best (Kuai 중심, lastfm 보조).
- 지표 우선순위: MRR@20, 보조로 HR@10.
- 슬라이드 1장: scoreboard table + gain(절대/상대) 막대.

### Part C. Model Axes and Design Space (6~8분)
5. 모델 축 소개 (탐색 공간을 청중이 따라오게)
- 공통 backbone(레이어, dim, max_len, 대략적 파라미터/연산량).
- feature injection 축: FFN(no feature, FiLM, gated bias), MoE(hidden/feature/both, stage/group).
- router 축: standard vs factored, top-k scope/global/group.
- residual 축: base/shared_only/shared_moe_fixed/shared_moe_learned/shared_moe_global/shared_moe_learned_warmup.
- aux/reg 축: balance/z/entropy/rule/group prior/factored group balance + specialization 계열.
- 슬라이드 2장:
  - (1) 축 맵(의사결정 트리)
  - (2) ablation taxonomy 표(축별 후보군)

### Part D. Ablation Results by Axis (7~10분)
6. 축별 ablation 결과 (best 중심이 아니라 축 중심)
- 원칙: "최종 best와만 비교"가 아니라 "축 안에서의 경향"을 먼저 보여준다.
- 각 축에서 2~4개 대표 비교를 선택해 noisy detail은 appendix로.
- Kuai는 valid/test를 같이 보여 일반화 성향까지 제시.
- 슬라이드 3~4장:
  - router/injection/residual/aux 각각 소형 패널 또는 facet chart.
  - 각 패널에 MRR@20(valid/test), seed 범위(가능하면 error bar).

7. 현재 best 조합 해설
- 어떤 축 선택의 조합이 best를 만들었는지 "조합 논리"를 언어화.
- 실패 조합 1~2개도 함께 제시해 설득력 강화.
- 슬라이드 1장: best recipe 카드 + counter-example 카드.

### Part E. Special/Diag Insights (4~6분)
8. special/diag 기반 직관과 주장 연결
- special slice: short session, low-pop 등 난조건에서의 변화.
- diag: entropy, n_eff, dead expert, top1 concentration, group consistency.
- "성능이 왜 올랐는지"를 routing quality로 연결.
- 슬라이드 2장:
  - (1) special slice delta heatmap
  - (2) diag radar/parallel coordinates (SASRec, 초기 FMoE, best 비교)

### Part F. Limits and Next Steps (2~3분)
9. 한계와 TODO
- 대조군 실험 보강 필요(특히 baseline breadth).
- lastfm 튜닝 확장 + 추가 데이터셋 검증.
- 후속 아이디어: hetero expert, collaborative expert, router distillation 확장.
- 슬라이드 1장: "What is done / What is missing / Next 3 runs".

## 2. 시각화 패키지 제안 (실제 제작 체크리스트)

### Must-have (본 발표 본문)
- Dataset profile table: Kuai vs lastfm 핵심 규모/분포.
- Prefix-length distribution: train/valid/test 비교.
- Scoreboard: SASRec vs FMoE line.
- Axis-wise ablation facets: router, injection, residual, aux.
- Special slice heatmap: slice x model delta.
- Diag summary panel: entropy/n_eff/top1/dead expert.

### Nice-to-have (appendix)
- Seed variance violin plot.
- Pareto plot: MRR@20 vs inference cost proxy.
- Family ablation spider plot (Tempo/Focus/Memory/Exposure).
- Split 방법 비교(현재 intra-session vs future session-level) mini pilot.

## 3. 데이터/지표 표기 규칙 (청중 혼란 방지)
- 기본 메트릭: MRR@20 (모든 핵심 그래프 동일).
- 보조 메트릭: HR@10.
- 각 그래프 caption에 반드시 표기:
  - dataset
  - split(valid/test)
  - seeds
  - best checkpoint 기준 여부
- baseline 표기 고정: "SASRec (same split, same search budget policy)".

## 4. 스토리텔링 가이드 (발표 톤)
- 순서 원칙: "결론 먼저 -> 근거 구조화 -> 세부 실험".
- 주장 템플릿:
  - Claim: 무엇이 좋아졌는가
  - Evidence: 어떤 축/그래프가 뒷받침하는가
  - Mechanism: diag/special로 왜 그런가
  - Boundary: 어디까지 일반화 가능한가
- 피해야 할 것:
  - 축이 다른 실험을 한 그래프에서 섞어 해석
  - best 하나만 강조하고 실패 사례를 숨김
  - valid/test를 분리 제시해 gap 원인을 놓침

## 5. 실제 슬라이드 구성안 (12장 내외)
1. Title + one-line message
2. Why these datasets
3. Data pipeline + v3 feature schema
4. Split policy and valid-test gap intuition
5. Snapshot results (SASRec vs FMoE timeline)
6. Model axis map (design space)
7. Router/injection ablation
8. Residual/aux ablation
9. Best config and counter-example
10. Special slice improvements
11. Diag routing behavior interpretation
12. Limitations + next experiments

## 6. 발표 전 점검 체크리스트
- Kuai main numbers 재검증(MRR@20, HR@10, seed count).
- SASRec 대비 표에서 split/search budget 동등성 명시.
- feature_added_v3 사용 여부와 cache provenance 확인 결과 한 줄로 명시.
- 그래프별 축/범례/단위 통일.
- Appendix에 raw table(phase별 상위 run) 준비.

## 7. 발표에서 바로 쓸 수 있는 핵심 멘트 예시
- "이번 실험의 핵심은 성능 수치 자체보다, 어떤 축 조합이 안정적으로 이득을 만들었는지 구조를 확인한 것입니다."
- "Kuai에서 test가 valid보다 높은 현상은 모델 우연이 아니라 split에서 생기는 prefix 길이 차이의 영향으로 해석됩니다."
- "best 조합은 factored routing, group-aware injection, 그리고 residual/aux의 균형이 동시에 맞아떨어졌을 때 나왔습니다."
- "special/diag를 같이 보면, 단순 점수 개선이 아니라 expert 사용과 분산이 더 건강해졌다는 증거가 있습니다."

## 8. 후속 실험 후보 (발표 마지막 1장)
1. Kuai + lastfm 공통 축 재현 (same seeds, same budget)으로 주장의 안정성 강화.
2. baseline bridge 보강(SASRec 및 FFN-only 변형 추가)으로 기여 분해.
3. hetero expert 또는 collaborative routing 파일럿(소규모 max-evals) 진행.
