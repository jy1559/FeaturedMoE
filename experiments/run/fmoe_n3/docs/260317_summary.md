# FMoE_N3 전체 실험 정리 (260317)

작성일: 2026-03-17  
대상: FeaturedMoE_N3 (core, P1, P2, P3, P4, P5)  
기본 지표: MRR@20 (보조: test MRR@20, HR@10)

---

## 요약 결론
- KuaiRec 기준 FMoE_N3는 SASRec(`0.0785`) 대비 phase 진행과 함께 `0.0801 -> 0.0811 -> 0.0813~0.0814`로 상승했다.
- 후반 phase의 핵심 메시지는 "balance 강제"보다 "specialization 유지 + micro jitter 제어"가 성능 설명력이 높다는 점이다.
- 오늘 최종 결론용 A/B/C는 모두 specialization 관점으로 재정의하고, seed=3 + 넓은 LR space로 결정한다.

---

## 1) Phase별 목적-과정-결과 정리 (발표 스토리라인)

## 1.1 기준선 확립 (Baseline)
- 왜: FMoE 개선폭의 절대 기준 필요.
- 무엇: SASRec best 확보.
- 결과:
  - KuaiRecLargeStrictPosV2_0.2: best MRR@20 `0.0785`, test MRR@20 `0.1597`, test HR@10 `0.1859`
  - lastfm0.03: best MRR@20 `0.4020`, test MRR@20 `0.3783`

## 1.2 Core (`core_ablation_v2`): "뭐가 기본적으로 먹히는가"
- 왜: stage/MoE/router/injection의 기본 지형 파악.
- 무엇:
  - dense_plain vs moe
  - router source(hidden/feature/both)
  - feature injection on/off
  - top-k, length, encoder 변형
- 결과:
  - KuaiRec 최고: `C70` best MRR@20 `0.0801`, test `0.1614`
  - lastfm 최고: `P00` best MRR@20 `0.4074`, test `0.3858`
  - 핵심 교훈: KuaiRec에서 `feature-only`(R32)는 collapse, `both`가 안전한 상위권.

## 1.3 P1 (`phase1_upgrade_v1`): "상위권 anchor 재현 + 실전축 확장"
- 왜: core 신호를 재현성 있는 anchor로 고정하고, 길이/스케줄/레이아웃 확장.
- 무엇: A/F/L/N/S/X 계열 28개.
- 결과:
  - KuaiRec 최고: `A05` best `0.0811`, test `0.1620~0.1622`
  - 평균 관점에서 A/F/S 계열이 안정적으로 높음.

## 1.4 P2 (`phase2_router_v1`): "router/feature/aux를 구조적으로 분해"
- 왜: P1에서 섞여 있던 효과를 block 단위로 분리.
- 무엇: 40개 (A/B/C/D block).
- 결과:
  - 최고: `PA05` best `0.0811`, test `0.1620`
  - `PD04(factored + group_gated_bias)` 상위권
  - `PD02(pure feature factored)` 실패/미기록
  - 강한 group balance는 평균적으로 불리
- 해석: 구조 자체 이득보다 "구조 x 하이퍼" 상호작용이 큼.

## 1.5 P3 (`phase3_focus_v1`): "유망 구조 4개 집중"
- 왜: P2 광역 탐색에서 얻은 유망 구조를 고정해 촘촘히 탐색.
- 무엇: S1(standard+gated), S2(factored+group_gated), S3(feature-source), S4(deep prefix).
- 결과:
  - 집계 최고: `P3S1_05` best `0.0813`
  - S2(factored) 계열이 성능/해석 모두에서 일관된 경쟁력.

## 1.6 P4 (`phase4_residual_topk_v2`): "residual/K/F 분해 실험"
- 왜: specialization 기제를 residual, expert 규모/top-k, feature 경로로 분리 검증.
- 무엇:
  - R: residual mode 비교
  - K: 4e/12e/group + top-k variants
  - F: full/hidden_only/feature_only/injection_only
- 결과(근거: `outputs/phase4_kf_report.txt`):
  - valid 최고: `K_12e_top6_C1` best `0.0814`
  - test 최고: `group_dense C3` test `0.1623`, HR@10 `0.1905`
  - K 평균 test: `group_dense(0.1619)`가 최상위
  - F 비교: `hidden_only`가 최약 (`0.1602`), `injection_only`가 상대 우수 (`0.1617`)
  - R 비교: `shared_only` 명확 열화, `base` 안정 우위

## 1.7 P5 (`phase5_specialization_v1`): "specialization 정규화 방향성"
- 왜: 성능을 유지하면서 specialization quality를 높일 보조손실 탐색.
- 무엇: Method(M0~M4) x Combo(C0~C7) = 40.
- 결과(근거: `phase5_results.md`):
  - 방법별 최고 test MRR@20: `M1=0.1623`
  - `micro_1.route_jitter_adjacent`와 test MRR@20 상관: `-0.6002`
  - 방법 간 절대 차이는 작고, combo+라우팅 안정성의 영향이 큼
- 주의: phase5는 결과 집계 파일 불일치/ablation file 공백이 있어 "경향 증거"로 사용.

### 1.7-a phase5_plan 대비 "추가 로깅" 구현 점검

| 계획 항목 (`phase5_plan.md`) | 구현 상태 | 근거/비고 |
|---|---|---|
| usage/top1/CV/n_eff/top1_max | 구현됨 | `N3DiagnosticCollector`에서 stage별 usage/top1 집계 및 n_eff/cv/top1_max 계산 |
| routing entropy(stage별) | 구현됨 | stage entropy + group entropy/factored group entropy 계산 경로 존재 |
| session jitter/smoothness | 구현됨(핵심) | `route_jitter_adjacent`, `route_jitter_session` 산출, P5 해석에도 사용 |
| feature-route consistency(kNN JS) | 부분 구현 | 학습 loss(`route_consistency_lambda`)는 있으나, 계획서의 별도 "추가 로깅 지표" 형태 저장은 약함 |
| feature bucket -> expert heatmap | 부분 구현 | family-expert 축적은 있으나, 계획서 수준의 bucket heatmap 산출물 표준화는 미흡 |
| expert purity summary | 부분 구현 | expert output similarity/usage 계열은 있음, purity 요약 리포트는 제한적 |
| slice 성능 + routing 동시 로그 | 부분 구현 | special metrics와 diag가 각각 저장되나, slice별 routing 동반 테이블은 부족 |
| run 단일 폴더 통합(`run_meta`, `links`) | 미구현/전환중 | 기존 normal/special/diag 중심 구조 유지, 완전한 run-centric 인덱싱은 미완 |

정리하면, P5에서 "jitter 중심 진단"은 실제로 들어갔지만, 계획서에 적은 고비용 해석 로그(특히 consistency/bucket/purity/slice-동반표)는 아직 부분 구현 상태다.

### 1.7-b feature ablation 파일 공백 원인 (확정)

- 직접 원인: phase5 런처(`run_phase5_specialization.py`)의 `build_command`에 `fmoe_feature_ablation_logging=true` 전달이 없다.
- 구조 원인: `hyperopt_tune.py`는 `fmoe_feature_ablation_logging`이 true일 때만 `feature_ablation_metrics`를 수집하고, 이 값이 있을 때만 `feature_ablation.json`을 기록한다.
- 결과: P5는 기본값(false) 경로를 타서 `feature_ablation_file`이 비거나 미생성으로 남는다.
- 참고: phase1/2/3/4 런처는 동일 플래그를 CLI 인자로 전달하는 경로가 존재한다.

## 1.8 발전 경로 요약 (발표용 1장)
- Baseline 고정 -> Core에서 위험구간 제거 -> P1/P2로 anchor 고도화 -> P3 구조 수렴 -> P4에서 specialization 메커니즘 분해 -> P5에서 jitter 중심 정규화 방향 확인.

---

## 2) 축별 정리 + 최종 후보 A/B/C (논문 설득 포인트 포함)

## 2.1 6개 핵심 축 (model_arch_sum_v3 기준)

| 축 | 현재 결론 | 핵심 근거 |
|---|---|---|
| 구조 축 (`layer_layout`, size/len) | `[macro,mid,micro]`가 안정 기준, deep prefix는 보조 후보 | P1/P2/P3에서 anchor 다수 상위권, S4는 서프라이즈지만 일관 1위는 아님 |
| compute 축 (`stage_compute_mode`) | `moe` 고정 | core에서 dense/plain 대비 moe 우위 |
| router 축 (`source/type/granularity/top-k`) | `source=both` 기본, `standard/factored` 둘 다 유효 | R32 collapse, PD04/S2 경쟁력, K축에서 group/factored 계열 상위 |
| feature 축 (`encoder/injection/family/macro_window`) | gated_bias 계열 유효, hidden_only 약함 | P2/P4 F축: injection_only/full이 hidden_only 대비 우위 |
| residual 축 (`stage_residual_mode`) | `base` 안정, warmup은 잠재력 | P4 R축에서 base 평균 우위, shared_only 열화 |
| aux/reg 축 (route 규제 포함) | imbalance 강제보다 jitter 제어형이 유망 | P2 strong balance 열세, P5에서 micro jitter 음상관 |

## 2.5 Feature 축 상세: macro window(5/10) + 4개 group 사용 범위

Feature family 기본 그룹은 `Tempo / Memory / Focus / Exposure` 4개다.

### (a) macro_history_window 5/10 사용 이력
- core/P1/P2/P3/P4 런처에서는 combo별로 `macro_history_window`를 명시(기본 5, 일부 10 비교).
- P5 런처(`run_phase5_specialization.py`)는 `macro_history_window` 오버라이드를 두지 않아, 이 축은 P5에서 사실상 고정 축이다.
- 따라서 "P5 성능 차이"를 macro window 효과로 해석하면 안 되고, method/combo(aux-reg + router/residual) 효과로 제한 해석해야 한다.

### (b) stage_feature_family_mask(4개 group 선택) 사용 이력
- core/P2/P3에는 family mask 탐색 흔적이 존재(예: `Tempo+Focus`, `Memory+Exposure`, `Focus`).
- P5 런처는 `stage_feature_family_mask` 오버라이드를 두지 않아, 4개 group 전체 사용(full_v3) 고정으로 보는 것이 맞다.
- 결론적으로 P5는 "specialization regularization 실험"이지, "feature family 선택 실험"은 아니다.

### (c) 발표용 문장 권고
- "feature 축은 P2/P4에서 유효성(특히 injection 경로)을 확인했고, P5에서는 feature 정의를 고정한 상태에서 specialization 안정화 정규화만 검증했다"로 서술하는 것이 데이터-일치적이다.

## 2.2 router-injection 짝(standard<->gated, factored<->group_gated) 근거

질문: "router가 standard면 gated_bias, factored면 group_gated_bias를 맞춰 쓰는 게 근거가 있나?"

결론: **강한 실무 근거는 있고, 완전 분리된 인과 증명은 아직 부족**.

- 근거 1: P2 상위 조합
  - `PA05(standard + gated_bias)` 최고권
  - `PD04(factored + group_gated_bias)` 상위권
- 근거 2: P3 구조 설계 자체가 이 짝으로 선정되어 재검증됨(S1 vs S2).
- 근거 3: P4 K/F에서 factored/group 계열이 cold/short 및 test 상위권을 반복적으로 만듦.

왜 논문에서 설득 가능한가:
- standard router는 expert별 직접 logits 구조라 global gated_bias가 자연스럽다.
- factored router는 group-level 의사결정이 있으므로 group_gated_bias가 구조적으로 일관된다.

남은 한계:
- 동일 조건에서 `standard + group_gated`와 `factored + gated` 교차 실험이 충분치 않다.
- 따라서 논문 서술은 "구조적 priors + 실험적 일관성"으로 쓰고, "엄밀 인과"는 보강 실험으로 제시하는 것이 안전.

## 2.3 최종 후보 A/B/C (specialization 중심으로 재정의)

아래 3개는 모두 "imbalance 강제"를 주축으로 쓰지 않고, specialization quality를 올리는 설정으로 정의한다.

## 후보 A: Standard-Spec (안정 specialization 기준선)
- 목적: 가장 재현성 높은 specialization 기준선.
- 설정:
  - layout `[macro,mid,micro]`
  - router `standard`, source `both`
  - injection `gated_bias`
  - residual `base`
  - aux/reg: `balance` 약하게(혹은 0에 가깝게), `z_loss` 약하게, `route_smoothness` 소량
- 논문 설득 포인트:
  - "최소한의 안정화만 둔 specialization baseline"으로서 해석 가능.

## 후보 B: Factored-Spec (group specialization 기준선)
- 목적: group-aware specialization을 직접 보여주는 주력 후보.
- 설정:
  - router `factored`
  - injection `group_gated_bias`
  - top-k scope `group_dense` 또는 `group_top2`
  - residual `base`
  - aux/reg: `group_prior_align` 약하게 + `route_smoothness` 소량
- 논문 설득 포인트:
  - "group-level routing + group-conditioned injection"의 구조 일관성.
  - P4 K축 최고 test(`group_dense C3=0.1623`) 근거.

## 후보 C: Factored-Warmup-Spec (안정-특화 절충)
- 목적: 초기 불안정은 줄이고 후반 specialization을 키우는 최종형.
- 설정:
  - 후보 B 기반
  - residual `shared_moe_learned_warmup`
  - `residual_alpha_init<0`, `alpha_warmup_until` 짧게
  - aux/reg: `route_consistency` 또는 `route_prior` 소량
- 논문 설득 포인트:
  - "training dynamics 관점에서 specialization을 점진적으로 활성화"했다는 서사가 가능.

## 2.4 오늘 최종 결론용 실행 가이드 (A/B/C 공정 비교)
- seed: 각 후보 3개
- LR space: 기존보다 약간 확대
  - 1차: `1.5e-4 ~ 3e-3`
  - 2차(상위 재탐색): `2.5e-4 ~ 1.2e-3`
- max-evals/tune-epochs는 충분히 크게 설정(현재까지의 최종 결론용)
- 선정 기준:
  - 1순위: best MRR@20
  - 2순위: test MRR@20
  - 3순위: micro jitter 분산(낮을수록 우선)

---

## 3) 축별 비교표 + 부족 구간 보강 실험 설계

## 3.1 축별 "근거 존재 여부" 한눈표

| 축 | 비교 데이터 존재 | 현재 강한 결론 | 아직 빈 구간 |
|---|---|---|---|
| 구조 축 | 높음 (core~P4) | `[macro,mid,micro]` 안정, 일부 deep prefix 경쟁력 | 구조 단독 효과(다른 축 고정) 분리 부족 |
| compute 축 | 중간 | `moe` 우위 | `none/dense_plain` 최신 조건 재검증 부족 |
| router 축 | 높음 (P2~P5) | both 유리, factored 생존, group_dense 강함 | standard/factored 교차 조합 공정 비교 부족 |
| feature 축 | 중간~높음 (P2/P4) | injection 경로 유효, hidden_only 약함 | family mask, macro_window 체계 비교 부족 (P5는 해당 축 고정) |
| residual 축 | 중간 (P4 중심) | base 안정, warmup 잠재력 | warmup/global/stage를 동일 조건으로 장기 비교 부족 |
| aux/reg 축 | 중간 (P2/P5) | 과한 imbalance 불리, jitter 제어형 유망 | 정규화 단독 기여도(one-factor ablation) 부족 |

## 3.2 부족한 부분 위주 보강 실험 (축 비교용)

### E1. Router x Injection 교차 2x2 (가장 중요)
- 목적: 짝 사용의 인과 근거 강화.
- 실험:
  - standard + gated_bias
  - standard + group_gated_bias
  - factored + gated_bias
  - factored + group_gated_bias
- 고정: layout, residual, LR space, seed.

### E2. Residual 3종 장기 비교
- 목적: C 후보(warmup) 타당성 검증.
- 실험:
  - base
  - shared_moe_global
  - shared_moe_learned_warmup
- 평가: best/test MRR + micro jitter + cold slice.

### E3. Aux/Reg 단독 기여 분해
- 목적: "specialization 유지 + 안정화" 정당화.
- 실험:
  - baseline
  - +route_smoothness
  - +route_consistency
  - +route_prior
  - +smoothness+consistency (소량)

### E4. Feature family/macro window 보강
- 목적: feature 축의 빈칸 채우기.
- 실험:
  - family mask 3~4개
  - macro window {5,8,10,12}

### E5. Phase5 로깅 누락 항목 보강(해석력 강화)
- 목적: phase5_plan의 "부분 구현" 항목을 논문용 증거로 완성.
- 실험/개발:
- consistency를 loss 값이 아니라 eval 지표(JS/kNN)로 별도 저장
- bucket->expert heatmap 및 expert purity summary를 `router_diag.json` 표준 키로 저장
- slice 성능표에 routing 지표(jitter/cv/top1_max)를 조인한 통합 csv 1종 추가
- feature ablation은 phase5 런처 플래그 ON + best checkpoint 1회 수집으로 고정

## 3.3 병렬 아이디어(축 비교 외)
- 로그 품질 보강:
  - phase5 결과 집계 경로/키 불일치 점검
  - feature ablation 파일 강제 생성 검증
- 데이터셋 전이:
  - KuaiRec winner 2개를 lastfm 전이 테스트
  - 단, 본 문서의 우선순위는 "축 비교 빈칸 보강"으로 둔다.

## 3.4 실험 국룰 세팅(유연 버전)

고정이 아니라 "범위 가이드"로 사용:
- LR:
  - 기본: `2e-4 ~ 2e-3`
  - 대형 모델/작은 batch: 상단 축소(`<=1.2e-3`) 권장
  - 작은 모델/큰 batch: 상단 확대(`<=3e-3`) 가능
- WD: `[1e-7, 1e-6, 1e-5, 5e-5]`
- dropout: `[0.10, 0.15, 0.20]`
- scheduler: `warmup_cosine` 기본, 불안정 시 warmup ratio 확대
- 모델 크기:
  - baseline: `embedding_size=128`, `expert_scale=3`
  - 확장 실험: `expert_scale=4` 또는 `embedding_size=256`는 별도 축으로 분리

---

## 발표용 마무리 문장
- "본 연구의 핵심은 MoE를 단순 도입한 것이 아니라, specialization을 구조적으로 설계하고(standard/factored + injection), training dynamics(residual warmup, jitter regularization)로 안정화한 점이다."
- "최종 모델은 A/B/C 3축 공정 비교로 결정하며, 이후 보강 실험은 축별 빈칸(E1~E4)을 채워 논문 인과성을 강화한다."
