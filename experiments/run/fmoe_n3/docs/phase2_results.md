## P2 결과 요약 (phase2_router_v1)

대상: KuaiRecLargeStrictPosV2_0.2, FeaturedMoE_N3
집계 기준: normal/phase2_router_v1 폴더의 run_phase별 최신 결과 JSON
기본 지표: best_mrr@20 (valid), 보조: test_mrr@20

### 1) 핵심 결론
- 이번 P2에서 best_mrr@20 최고값은 0.0811로 확인됨.
- 상위권이 0.0805~0.0811에 촘촘히 몰려 있어 "구조 대이득"보다 "구조-하이퍼 상호작용"이 크다.
- D(block 구조 변경)는 최고점(0.0809)은 만들었지만 평균은 A/B/C보다 낮아 아직 안정화 단계.
- B(block wd 고정) 테스트 지표는 강함(test 평균 약 0.1620) -> 최종 튜닝에서 wd/dropout 미세 탐색 가치가 높음.

### 2) 블록별 요약
- A: n=16, mean(best)=0.0805, max(best)=0.0811, mean(test)=0.1605
- B: n=4, mean(best)=0.0805, max(best)=0.0805, mean(test)=0.1620
- C: n=8, mean(best)=0.0805, max(best)=0.0807, mean(test)=0.1613
- D: n=11, mean(best)=0.0802, max(best)=0.0809, mean(test)=0.1609

참고:
- PD02(factored_router_pure_feature)는 결과 수치가 비어 있음(nan) -> 실패/미기록 케이스로 취급.

### 3) 상위 조합 (best_mrr@20 기준)
- PA05 = 0.0811 (all_gated_bias)
- PA_FS = 0.0809 (feature_source_only)
- PD04 = 0.0809 (factored_router_plus_group_gated_bias)
- PA_MX1 = 0.0809 (macro_complex + gated_bias + aggressive)
- PA_F03 / PD03 / PD10 / PC05 = 0.0807

### 4) 요소별 해석
- feature injection:
  - all_gated_bias(PA05)가 최고점 -> feature-conditioned FFN 경로는 여전히 강함.
  - group_gated_bias 단독(PD03)은 상위권이나 최고점은 아님 -> 그룹화 자체는 유효, 안정화/정규화 추가 필요.
- routing 구조:
  - factored+group_gated_bias(PD04)가 상위 동률 -> 계층형(group->expert) 아이디어는 실효성 있음.
  - pure feature factored(PD02) 실패 -> hidden 신호를 완전히 버리는 구성은 위험.
- aux/reg:
  - 강한 balance+z(PC05)와 group_prior_align 포함(PC04, PD10)이 상위권 -> routing 규제는 "약하게" 도움.
  - factored_group_balance를 강하게 준 PD11/PD12는 중하위 -> 과도 규제는 유연성 감소 가능.
- 구조/복잡도:
  - deep layer prefix(PA_L08)는 크게 망하지 않음(0.0805) -> surprise 후보로 유지 가치.
  - embed256(PA_X65)은 오히려 하락 -> 현재 데이터/탐색 예산에서는 과대모델 이득 제한.

---

## P3 설계 (4 구조 x 5 콤보 = 20)

의도:
- P2에서 구조와 하이퍼를 동시에 넓게 봐서 신호가 섞였으므로,
- P3는 구조를 4개로 고정하고, 각 구조 내에서 5개만 촘촘히 탐색.
- 각 구조당 3개는 보수형, 2개는 특이형.

선정된 4 구조:
1. S1: standard router + all gated_bias (PA05 계열, 성적 1위)
2. S2: factored router + group_gated_bias (PD04 계열, 이론/성능 균형)
3. S3: feature-source routing (PA_FS 계열, feature 활용 강한 축)
4. S4: deep layer-prefix (PA_L08 계열, 의외로 버틴 surprise 축)

P3 combo 구성:
- S1: P3S1_01~05
- S2: P3S2_01~05
- S3: P3S3_01~05
- S4: P3S4_01~05

실행 파일:
- 러너: experiments/run/fmoe_n3/run_phase3_20.py
- 쉘: experiments/run/fmoe_n3/phase_3_20.sh

요약 저장:
- run_phase3_20.py 종료 시 build_fmoe_n3_summaries + build_fmoe_n3_axis_summary(axis=phase3_focus_v1) 호출
- 따라서 artifacts/results/summary.csv 및 축 요약 CSV 갱신 경로가 기존과 동일하게 유지됨.

### 권장 실행 순서
1. dry-run으로 manifest 확인
2. KuaiRec 먼저 20개 전량 실행
3. 필요 시 상위 6~8개만 lastfm0.03 재평가
