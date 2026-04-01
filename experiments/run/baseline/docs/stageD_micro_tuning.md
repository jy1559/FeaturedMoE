# Stage D (Micro Knobs) - baseline

## 목표
- Stage C에서 고정한 구조/중요 knob를 유지하고, 미세 regularization(`dropout`, `weight_decay`) 중심으로 정밀 탐색한다.
- early stop 패턴으로 잠재력 없는 조합을 빠르게 줄이고, 잠재력이 있는 조합만 Stage E로 올린다.

## A~C 관찰 요약 (핵심)
- `lastfm0.03`
  - 성장폭 큼: `DIFSR +0.0102`, `GRU4Rec +0.0135`, `SASRec +0.0074`, `DuoRec +0.0085` (A->C valid MRR@20).
  - Stage C 기준 SASRec 대비: `DIFSR +0.0048`로 우위, `DuoRec -0.0072`로 추격권.
  - selected config에서 `DIFSR/SASRec/GRU4Rec`는 best trial이 후반 epoch까지 가는 경우가 많아 Stage D 확장 가치가 있음.
- `amazon_beauty`
  - Stage C 기준 SASRec(0.1264)가 여전히 기준선.
  - `DuoRec(0.1226)`만 근접, `DIFSR/GRU4Rec/FAME`는 격차가 크고 early stop이 매우 이르게 발생.
  - 따라서 Amazon에서는 모델별 예산을 차등 배정한다.

## Stage C 고정 입력 (Stage D 시작점)

### lastfm0.03
- `SASRec`: `B6>C1`, `B6>C6`
- `GRU4Rec`: `B6>C3`, `B6>C6`
- `DuoRec`: `B6>C1`, `B3>C1`
- `DIFSR`: `B6>C6`, `B6>C3`
- `FAME`: `B6>C2`, `B4>C2`

### amazon_beauty
- `SASRec`: `B2>C6`, `B3>C4`
- `GRU4Rec`: `B4>C2`, `B6>C6`
- `DuoRec`: `B3>C1`, `B6>C1`
- `DIFSR`: `B1>C5`, `B6>C6`
- `FAME`: `B2>C5`, `B1>C6`

## 탐색 축 (Stage D)
- 공통 미세축
  - 기본: `dropout +-0.03`, `weight_decay x[0.7, 1.4]`
  - 공격 모드: `dropout +-0.06`, `weight_decay x[0.45, 2.2]`
- 모델별 추가 미세축
  - `DuoRec`: 기본 `tau +-0.05`, 공격 모드 `tau +-0.12`, `lmd in {0.0,0.01,0.02,0.03,0.05}`
  - `DIFSR`: 기본 `lambda_attr +-0.05`, 공격 모드 `lambda_attr +-0.12` (clamp `[0.0, 0.35]`), `fusion_type`는 Stage C best 고정
  - `FAME`: `num_experts`는 Stage C best 고정, regularization만 조정
  - `SASRec/GRU4Rec`: regularization만 조정

## Stage D 2-pass 운영 (권장)
- `Pass-1 (Aggressive Breadth)`
  - 폭 넓은 범위로 국소 최적 탈출 시도.
  - `max_evals`를 기본 대비 `+30%`.
- `Pass-2 (Local Refine)`
  - Pass-1 상위 2개 근방만 재탐색.
  - `dropout +-0.02`, `weight_decay x[0.8,1.25]`.

## 예산 (모델/데이터셋별)

### lastfm0.03
- `SASRec`: `max_evals=10`, `tune_epochs=60`, `patience=8`
- `DIFSR`: `max_evals=10`, `tune_epochs=60`, `patience=8`
- `GRU4Rec`: `max_evals=8`, `tune_epochs=56`, `patience=8`
- `DuoRec`: `max_evals=8`, `tune_epochs=42`, `patience=6`
- `FAME`: `max_evals=8`, `tune_epochs=48`, `patience=7`

### amazon_beauty
- `SASRec`: `max_evals=10`, `tune_epochs=56`, `patience=8`
- `DuoRec`: `max_evals=8`, `tune_epochs=40`, `patience=6`
- `DIFSR`: `max_evals=4`, `tune_epochs=24`, `patience=3`
- `GRU4Rec`: `max_evals=4`, `tune_epochs=20`, `patience=3`
- `FAME`: `max_evals=4`, `tune_epochs=24`, `patience=3`

## early stop 기반 게이팅 규칙
- 모델/데이터셋 조합별 Stage D trial 집합에서 아래를 계산:
  - `median(best_epoch / tune_epochs)`
  - `best_valid_mrr20 - stageC_best_valid_mrr20`
- 중단(저잠재력) 조건:
  - `median(best_epoch / tune_epochs) <= 0.35` 이고
  - 개선폭 `< +0.0010`
- 확장(고잠재력) 조건:
  - `median(best_epoch / tune_epochs) >= 0.80` 또는
  - 개선폭 `>= +0.0015`

## Stage E 승급 규칙 (D -> E)
- 기본: 모델/데이터셋별 `Top-1 + Stability`.
- 예외: `SASRec`, `DIFSR(lastfm0.03)`, `DuoRec(amazon_beauty)`는 `Top-2`까지 승급 허용.
- 안정성 점수:
  - `score = valid_mrr20 - 0.5 * std(top3_trial_valid_mrr20) - 0.01 * (1-completion_ratio)`

## B/C 재실행 여부를 결정하는 최소 기준
- 아래 2개를 동시에 만족하면 B/C 재실행 우선:
  - Stage C `gap(top1-top2) <= 0.0015`
  - Stage C early-stop ratio `>= 0.70`
- 위 기준이면 D만 넓히기 전에 B/C 분해력을 먼저 보강하는 편이 신뢰성에 유리.
