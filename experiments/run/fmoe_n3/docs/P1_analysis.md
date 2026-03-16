# FMoE_N3 KuaiRec P1 분석 및 Phase2 제안

작성일: 2026-03-14  
대상: `KuaiRecLargeStrictPosV2_0.2` (`phase1_upgrade_v1/P1`)

## 1) 분석 목적
- KuaiRec P1 결과를 한 번에 정리한다.
- 어떤 설정이 실제로 좋은지 수치로 비교한다.
- 비슷한 유형끼리 묶어서 평균 성능을 낸다.
- 이 근거로 KuaiRec phase2(장기 탐색)와 그 다음 마무리 러닝 방향을 제안한다.

## 2) 데이터 소스
- `experiments/run/artifacts/logs/fmoe_n3/phase1_upgrade_v1/phase1_upgrade_v1_summary.csv`
- `experiments/run/artifacts/logs/fmoe_n3/phase1_upgrade_v1/P1/KuaiRecLargeStrictPosV2_0.2/P1_summary.csv`
- SASRec baseline 참고:
	- `experiments/run/artifacts/results/baseline/KuaiRecLargeStrictPosV2_0.2_SASRec_p0_full_pair01_c4_20260312_203258_569533_pid120708.json`

## 3) 핵심 결과 요약

### 3.1 단일 최고치
- P1에서 관측된 최고 `best_mrr@20`: **0.0811** (`A05`, both + gated_bias)
- P1에서 관측된 최고 `test_mrr@20`: **0.1629** (`X65`, 현재 `running` 상태)
- 완료(`success`) 기준 최고 `test_mrr@20`: **0.1622** (`A04`)

### 3.2 SASRec 대비
- SASRec 기준 (`C4`):
	- `best_mrr@20 = 0.0785`
	- `test_mrr@20 = 0.1597`
- P1 상위권은 SASRec 대비 대략:
	- `best_mrr@20`: +0.0020 ~ +0.0026
	- `test_mrr@20`: +0.0023 ~ +0.0032 (완료 기준)

해석: KuaiRec에서는 FMoE_N3가 baseline 대비 유의미한 우위를 유지하고 있고, 특히 `A/F/S` 계열에서 재현성 있는 개선 구간이 보인다.

## 4) 상위 조합 비교 (P1)

`test_mrr@20` 상위(요약):

1. `X65`: best 0.0799 / test 0.1629 (`running`, 4/10)
2. `A04`: best 0.0801 / test 0.1622
3. `L02`: best 0.0803 / test 0.1621
4. `S03`: best 0.0804 / test 0.1621
5. `N50A`: best 0.0802 / test 0.1621
6. `F01`: best 0.0806 / test 0.1620
7. `F02`: best 0.0802 / test 0.1620
8. `S02`: best 0.0805 / test 0.1620
9. `A05`: best 0.0811 / test 0.1620

참고: `X65`는 아직 러닝 중이라 최종 안정 조합 판단은 보류.

## 5) 유형별 평균 비교 (비슷한 것끼리 묶음)

분류 규칙:
- `A*`: Anchor
- `F*`: Feature
- `L*`: Layout
- `N*`: Length
- `S*`: Scheduler
- `X*`: Embed/Aux

유형별 평균 (`best_mrr@20`, `test_mrr@20`):

- `A` (n=6): avg best **0.0804**, avg test **0.1618**
- `F` (n=4): avg best **0.0804**, avg test **0.1619**
- `L` (n=8): avg best **0.0803**, avg test **0.1615**
- `N` (n=5): avg best **0.0805**, avg test **0.1615**
- `S` (n=3): avg best **0.0805**, avg test **0.1620**
- `X` (n=2): avg best **0.0799**, avg test **0.1616**

핵심 해석:
- 평균 관점에서는 `S`, `F`, `A` 축이 가장 안정적으로 높다.
- `L`, `N`은 잘 뜨는 조합이 있지만 평균은 약간 낮고 분산이 큰 편.
- `X`는 상위치(`X65`)는 강하지만 표본 수가 작고 진행 중이라 신중하게 봐야 한다.

## 6) 어떤 세팅이 좋은가 (실전 기준)

현재까지 KuaiRec P1 기준으로 "좋은 세팅"은 다음 성격이 겹친 조합:

- 라우팅/피처 결합:
	- `both` 라우팅 + `gated_bias` 계열 (`A05`, `F04` 계열)
- 스케줄러:
	- `S02/S03` 계열처럼 학습률 스케줄을 명시적으로 둔 경우
- 길이/레이아웃:
	- `N50A`, `L02`처럼 길이나 구조를 바꾸되 학습률을 중저 구간으로 안정화한 조합
- LR 대역(관측 기준):
	- 최상위권에서 대체로 `2e-4 ~ 8e-4` 대역이 안정적
	- `1e-3` 이상도 일부 성공은 있으나 분산 증가

## 7) KuaiRec Phase2 어떻게 갈지 (추천)

요청사항 반영:
- 더 오래 돌리는 실험 허용
- combo 수를 크게 늘리거나(max 40 근처) `max_eval` 확장
- 한 번 더 돌린 뒤, 마지막 한 번으로 마무리

### 7.1 Phase2-1 (이번 바로 실행)
- 목적: P1에서 확인된 강한 축(`A/F/S/N50/L02`)만 집중 고예산 검증
- 권장 예산:
	- `max-evals=40`
	- `tune-epochs=120`
	- `tune-patience=12`
- 기대효과:
	- 상위 조합의 "재현성 + 최적점"을 동시에 확보

### 7.2 Phase2-2 (다음, 거의 마무리 러닝)
- 목적: Phase2-1 상위 6~8개만 좁은 LR/WD 밴드로 재탐색
- 권장 예산:
	- `max-evals=50`
	- LR 밴드 축소 (`1.8e-4 ~ 7e-4` 중심)
	- seed 다변화(+2~3 seed)
- 기대효과:
	- 단일 최고점이 아닌 "최종 제출용 안정 조합" 고정

## 8) 후보 조합 전략 (유형별 best 조합 + 조합 전략)

### 8.1 유형별 대표 후보
- Anchor: `A04`, `A05`
- Feature: `F01`, `F02`, `F04`
- Scheduler: `S02`, `S03`
- Length/Layout: `N50A`, `L02`
- Embed/Aux: `X65` (완료 후 재평가)

### 8.2 조합 방식 추천
- 방식 1: "상위축 교차"
	- 예: `A05`의 피처 주입 성격 + `S03` 스케줄 + `N50A` 길이
- 방식 2: "레이아웃 고정 + 학습 안정화"
	- `L02` 고정 후 `S02/S03`, dropout/wd만 세밀 조정
- 방식 3: "안 만진 축 추가"
	- `X65` 완료 결과를 보고 embed/aux 강도 축을 2~3포인트만 추가

## 9) 실행 우선순위

1. KuaiRec Phase2-1: P1 고예산 재탐색 (`max-evals=40`)
2. lastfm0.03 core_ablation_v2: 동일하게 고예산 러닝
3. 결과 취합 후 KuaiRec Phase2-2(최종 수렴 러닝)

## 10) 주의사항
- `X65`는 현재 진행 중 상태라, 완료 전까지는 최종 best로 확정하지 않는다.
- 성능 주장 시 항상 SASRec baseline 동반 기재.
- 최종 판단 지표 기본은 `MRR@20`.

## 11) 실제로 바꿔본 파라미터 축 점검

코드/로그 기준으로 확인한 "이미 실험된 축"은 아래와 같다.

### 11.1 이미 충분히 본 축
- 레이아웃 축 (`L01~L08`)
- 길이 축 (`N30*`, `N50A`)
- feature encoder/injection 축 (`F01~F04`, `A04`, `A05`)
- scheduler 축 (`S01~S03`)
- 기본 최적화 축 (lr, dropout, wd)

### 11.2 일부만 본 축 (신뢰도 낮음)
- `macro_history_window`:
	- core에서 `M22(window=5)` vs `X60(window=10)` 비교는 1쌍 수준
	- 결과: `M22`가 근소 우위 (`best/test` 모두 약간 높음)
	- 결론: 지금 데이터만으로 window 우열 확정은 이르다.
- aux/reg:
	- P1에서 사실상 `X64/X65`로 일부만 건드림
	- 나머지 조합에서는 고정값 성격이 강함
	- 결론: aux/reg는 체계적인 스윕이 아직 부족.
- router family mask:
	- core `X61(Tempo+Memory only)` 1회성 수준
	- 결론: family 단위 효과 검증이 부족.

### 11.3 거의 안 본 축
- 고정 lr/drop에서 wd만 미세 조정하는 정밀 sweep
- `balance_loss_lambda`, `z_loss_lambda`, `gate_entropy_lambda`의 연속값 탐색
- `moe_top_k`의 스케줄링(고정값 0/2 외)
- router source를 family/group 관점으로 세분화한 설계

## 12) macro5 vs macro10 확인

현재 확인 가능한 가장 직접 비교는 core의:
- `M22` (기본 `macro_history_window=5`)
- `X60` (`macro_history_window=10`)

관측상 KuaiRec에서는 `M22`가 근소 우위였고, `X60`은 비슷하거나 약간 낮았다.

다만 신뢰성 이슈:
- 각 설정 표본 수가 작다.
- window 외 요인이 완벽히 고정된 반복(seed 다변화 포함)이 부족.

따라서 다음 phase2에서 반드시 추가:
- 동일 상위 anchor(예: A05/S03/L02 계열)에서
- `window = 5, 8, 10, 12`를 동일 budget/seed 체계로 반복 비교.

## 13) aux_loss/reg 결과 해석과 추가 방향

현재까지 해석:
- 강한 성능권은 주로 feature/router/layout/scheduler 조합에서 나옴.
- aux/reg 자체의 독립 기여를 분리하기 어려운 상태.
- `X65`처럼 용량+aux 강한 조합이 test 고점을 보일 가능성은 있으나, 아직 running.

phase2에서 해야 할 것:
- 국룰 세팅을 고정한 뒤 aux/reg만 바꿔서 효과 분리.
- 특히 아래 3개를 연속값으로 탐색:
	- `balance_loss_lambda`
	- `z_loss_lambda`
	- `gate_entropy_lambda`

권장 탐색 범위:
- `balance_loss_lambda`: 0.0015, 0.0025, 0.0040, 0.0060
- `z_loss_lambda`: 0.0, 1e-4, 2e-4, 4e-4
- `gate_entropy_lambda`: 0.0, 2e-4, 5e-4

## 14) router/group-family 아이디어 반영 가능성

요청하신 방향 요약:
- stage 전체 신호로 group(tempo/memory/focus/exposure 등) 가중치 산출
- group 내부에서 family feature 기반으로 expert 선택/전문화
- 계층형(Hierarchical)까지는 아니어도 group-score + family-score를 합성

현재 상태:
- config 수준에서는 `stage_feature_family_mask`로 일부 family 제한만 가능.
- 진짜 "group score + family score 합성 router"는 모델 코드 수정이 필요.

코드 수정 전, 지금 당장 가능한 근사 실험:
- family mask 조합을 늘려서 group별 민감도 측정
	- tempo/memory
	- focus/exposure
	- tempo/focus
	- memory/exposure
- router_source(`both/hidden/feature`)와 결합해 group별 안정성 확인

코드 수정 후(다음 단계) 추가 가능한 설계:
- GroupAdd router: 
	- stage 전체 state로 group logits
	- group별 family projection으로 expert logits
	- 두 logits 합산 후 top-k
- GroupParallel router:
	- group별 MoE 출력을 먼저 계산
	- weighted sum 후 FFN(선택)

## 15) 다음 phase2 콤보 설계 (최대 40개)

아래는 "국룰 세팅 고정 + 필요한 축만 변형" 원칙으로 만든 40개 구성안.

### 15.1 국룰 세팅 (Base Canonical, B0)
- 구조: `macro, mid, micro`
- feature: `gated_bias` 계열(A05 성격)
- router: learned + both
- scheduler: plateau 또는 warmup_cosine 안정형(S03/S02 상위값)
- `macro_history_window=5`
- `moe_top_k=0`
- 초기 고정값(튜닝 블록용):
	- `learning_rate=3.0e-4`
	- `dropout=0.20`
	- `weight_decay=1e-6`

### 15.2 40개 배분 (최종)
- Block A. 상위권 풀 섞기/변형: 16개
- Block B. 고정 lr/drop + wd/연속값 정밀: 4개
- Block C. aux/reg(+신규 reg) 스윕: 8개
- Block D. router family/group 구조 변경: 12개

합계 관점:
- Block A+B = 20개 (16/4): 기존 성능 상향 + 일반 튜닝
- Block C+D = 20개 (8/12): routing 성능/구조 집중 탐색

총합 40개.

### 15.3 Block A (16) 상위권 풀 섞기/변형
- "딱 2축 고정"이 아니라 상위권 후보 풀에서 다중 섞기 방식으로 간다.
- seed pool(상위권): `A04`, `A05`, `L02`, `S02`, `S03`, `N50A`, `F01`, `F02`, `X65(완료시)`
- 조합 원칙:
	- 구조 1개(layout/len), 라우팅 1개(router_source or granularity), feature 1개(encoder/injection), 학습안정 1개(scheduler/aux)를 동시에 선택
	- 같은 축의 미세변형(예: macro complex on/off, mid token on/off)을 붙여 16개 구성
	- 상위 조합 "복제"보다 상위 조합 "변형" 비율을 높인다 (권장 7:3)

### 15.4 Block B (4) lr/drop 고정 + wd/연속값
- 고정:
	- lr=3.0e-4
	- dropout=0.20

- sweep(4점):
	1) wd=0
	2) wd=1e-6
	3) wd=1e-5
	4) wd=1e-6 + `group_prior_align_lambda=5e-4`

의도:
- "lr/drop 요인 제거" 상태에서 wd 민감도만 분리.
- 마지막 1점은 wd + 연속값(reg) 결합으로, 단일 wd sweep보다 정보량을 높인다.

### 15.5 Block C (8) aux/reg 전용
- 기준 구조는 B0 고정.
- 기존 + 신규 파라미터를 함께 탐색:
	- `balance_loss_lambda`: 0.002 / 0.004 / 0.006
	- `z_loss_lambda`: 0 / 1e-4 / 3e-4
	- `gate_entropy_lambda`: 0 / 2e-4
	- `group_prior_align_lambda`: 0 / 5e-4 / 1e-3
- 포인트 구성은 "1개 파라미터만 단독 변경" 3개 + "2개 결합" 3개 + "3개 결합" 2개로 설계.

### 15.6 Block D (12) router-family/group 구조 변경 [업데이트됨]
- `stage_router_type` 및 `group_gated_bias` 신규 구현 반영:
  - 구현 추가된 변형:
    - **`factored` router**: `group_router_net(feature → n_groups)` + `intra_router_net(hidden → n_experts)` 합산 → feature가 group 선택을 직접 드라이브
    - **`group_gated_bias` injection**: group마다 해당 group 소속 raw feature만으로 개별 gated_bias 계산 → expert별 conditioned hidden
	- 12개 구성 축:
		1) 구조 변형 4개 (`factored`, `pure feature factored`, `group_gated_bias`, 결합)
		2) feature group ablation 3개 (2-group/1-group 축소)
		3) lambda 탐색 3개 (`feature_group_bias_lambda`, `group_prior_align_lambda` 조합)
		4) group routing 안정화 reg 2개 (`factored_group_balance_lambda`, `group_coverage_lambda` 강화)

### 15.7 run_phase2_40 실제 combo 매핑
- Block A (16): `PA01`, `PA04`, `PA05`, `PA_L04`, `PA_L08`, `PA_X65`, `PA_X64`, `PA_N50`, `PA_FS`, `PA_F03`, `PA_F04`, `PA_T3`, `PA_S02`, `PA_S03`, `PA_MX1`, `PA_MX2`
- Block B (4): `PB01`, `PB02`, `PB03`, `PB04`
- Block C (8): `PC01`, `PC02`, `PC03`, `PC04`, `PC05`, `PC06`, `PC07`, `PC08`
- Block D (12): `PD01`, `PD02`, `PD03`, `PD04`, `PD05`, `PD06`, `PD07`, `PD08`, `PD09`, `PD10`, `PD11`, `PD12`

Block D 세부 의도:
- `PD01~PD04`: router/injection 구조 자체 변경
- `PD05~PD07`: group family ablation (2개만 유지하거나 1개만 유지)
- `PD08~PD10`: `feature_group_bias_lambda=0`, `group_prior_align_lambda=0` 기준에서 벗어난 lambda 탐색
- `PD11~PD12`: routing 붕괴 방지용 aux/reg 강화 (`factored_group_balance_lambda`, `group_coverage_lambda`)

## 16) phase2 이후(마지막 1회) 마무리 방식

phase2(40개) 완료 후:
- 상위 6~8개만 선별
- 그 안에서 seed 2~3개 재검증
- lr 범위를 좁혀 최종 수렴
- 최종 선정 기준:
	- 평균 `MRR@20`
	- 표준편차(안정성)
	- special/diag에서 붕괴 신호(top1 쏠림, route change) 여부

즉, 마지막 1회는 "최고점"보다 "재현되는 강한 조합"을 고르는 단계로 간다.

