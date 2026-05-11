# P0 실험 결과 기반 P1 설계안 (baseline)

작성일: 2026-03-14
대상 로그:
- experiments/run/artifacts/logs/baseline/KuaiRecLargeStrictPosV2_0.2/P0
- experiments/run/artifacts/logs/baseline/lastfm0.03/P0

기준 지표: MRR@20 (valid best)

---

## P0 실험 결과

### 1) 완료 상태

- KuaiRecLargeStrictPosV2_0.2: 32개 중 31개 완료, 1개 중단
  - 중단: srgnn_041_c4_wide.log
- lastfm0.03: 19개 중 17개 완료, 2개 중단
  - 중단: sigma_060_c3_long.log, fame_056_c4_wide.log

해석:
- lastfm0.03 쪽은 타겟 모델(SOTA군) 커버리지가 불균형함.
- 그래서 "SASRec이 제일 높다"는 결론은 맞지만, 타 모델이 불리한 탐색 예산/완료율을 받은 면이 분명히 있음.

### 2) KuaiRecLargeStrictPosV2_0.2 성능 요약 (타겟 5모델 + SASRec)

| 모델 | P0 최고 combo | Best MRR@20 | SASRec 대비 |
|---|---|---:|---:|
| SASRec | C4 | 0.0785 | 0.0000 |
| SIGMA | C3 | 0.0796 | +0.0011 |
| FENRec | C1 | 0.0777 | -0.0008 |
| PAtt | C1 | 0.0776 | -0.0009 |
| BSARec | C3 | 0.0774 | -0.0011 |
| FAME | C1 | 0.0759 | -0.0026 |

시간(10 eval 기준, P0 로그):
- BSARec: 11.1~48.2분
- PAtt: 13.0~13.2분 (C1/C2만 완료)
- FAME: 18.0~65.9분
- FENRec: 19.5~95.8분
- SIGMA: 20.5~123.3분

### 3) lastfm0.03 성능 요약 (타겟 5모델 + SASRec)

| 모델 | P0 최고 combo | Best MRR@20 | SASRec 대비 |
|---|---|---:|---:|
| SASRec | C3 | 0.4020 | 0.0000 |
| SIGMA | C1 | 0.4013 | -0.0007 |
| PAtt | C2 | 0.3989 | -0.0031 |
| BSARec | C2 | 0.3978 | -0.0042 |
| FENRec | C3 | 0.3968 | -0.0052 |
| FAME | C2 | 0.3936 | -0.0084 |

시간(10 eval 기준, P0 로그):
- BSARec: 93.0~212.6분
- PAtt: 81.7~235.1분
- FAME: 251.7~390.1분 (C4 중단)
- FENRec: 267.4분 (C3만 완료)
- SIGMA: 482.2분 (C1 완료, C3 중단)

### 4) lr/정규화 경향 요약

- Kuai는 고 lr에서 잘 나오는 경우가 많음.
  - 타겟 모델 기준 유효 구간이 대체로 1e-3~5e-3에 몰림.
  - P1에서는 상한 1e-2까지 열어보는 것이 합리적.
- lastfm은 Kuai 대비 저 lr 쪽이 안정적.
  - 대체로 1e-4~1e-3 구간이 유리.
- weight_decay는 모델 의존적이지만, Kuai는 1e-4 근처가 많이 선택됨.

### 5) wd/dropout 탐색 방식 결정 (1/2/3)

검토안:
- 1안: `loguniform_zero` 등 연속형 탐색
- 2안: 소수의 choice 후보(2~3개)
- 3안: wd/dropout 고정, lr만 탐색

최종 선택: 3안

선정 이유:
- 현재 예산(max_eval)이 크지 않아, lr + wd + dropout을 동시에 탐색하면 trial당 정보 밀도가 낮아짐.
- P0 로그에서 성능 분산의 주요 축은 lr이었고, wd/dropout은 모델별로 상대적으로 안정적인 구간이 관찰됨.
- 타겟 5모델 x 2데이터셋 x 4콤보를 운영할 때는 차라리 wd/dropout을 모델/데이터셋별로 고정하고 lr에 예산을 집중하는 편이 성공 확률이 높음.

고정 wd/dropout 값(이번 P1):
- Kuai: BSARec(0.15,1e-4), PAtt(0.10,1e-4), FAME(0.10,1e-4), FENRec(0.10,1e-4), SIGMA(0.20,1e-4)
- lastfm: BSARec(0.10,5e-4), PAtt(0.15,0), FAME(0.10,1e-4), FENRec(0.20,0), SIGMA(0.05,1e-4)

---

## 다음 실험 모델/데이터셋별 combo 구성 상태

요청 반영:
- 모델당 combo 8개로 확대 (wave1: C1~C4, wave2: C5~C8)
- GPU 4개에 combo 1개씩 동시 배치 (wave별 4개)
- 모델 처리 순서: BSARec -> PAtt -> FAME -> FENRec -> SIGMA
- 각 모델 내 8개 combo는 구조 다양성(차원/레이어/전용 파라미터) + outlier 성격 조합 포함

### A) KuaiRecLargeStrictPosV2_0.2 (고 lr 허용)

#### BSARec (K_BS_C1~C4)
- C1: hidden=128, layers=1, heads=4, max_len=10
- C2: hidden=128, layers=2, heads=4, max_len=10
- C3: hidden=128, layers=2, heads=8, max_len=10
- C4: hidden=160, layers=3, heads=4, max_len=20
- lr search: [3e-4, 1e-2]

#### PAtt (K_PA_C1~C4)
- C1: hidden=128, layers=1, heads=2, gamma=0.05, max_len=10
- C2: hidden=128, layers=2, heads=2, gamma=0.10, max_len=10
- C3: hidden=128, layers=2, heads=4, gamma=0.10, max_len=10
- C4: hidden=160, layers=3, heads=4, gamma=0.20, max_len=20
- lr search: [5e-4, 1e-2]

#### FAME (K_FA_C1~C4)
- C1: hidden=128, layers=1, heads=2, experts=2, max_len=10
- C2: hidden=128, layers=2, heads=4, experts=4, max_len=10
- C3: hidden=128, layers=2, heads=4, experts=8, max_len=10
- C4: hidden=160, layers=3, heads=4, experts=4, max_len=20
- lr search: [2e-4, 6e-3]

#### FENRec (K_FE_C1~C4)
- C1: hidden=128, layers=1, heads=2, cl_w=0.05, cl_t=0.20, max_len=10
- C2: hidden=128, layers=2, heads=2, cl_w=0.10, cl_t=0.20, max_len=10
- C3: hidden=128, layers=2, heads=4, cl_w=0.10, cl_t=0.10, max_len=10
- C4: hidden=160, layers=3, heads=4, cl_w=0.20, cl_t=0.20, max_len=20
- lr search: [3e-4, 8e-3]

#### SIGMA (K_SI_C1~C4)
- C1: hidden=128, layers=1, state=16, kernel=8, remain=0.5, max_len=10
- C2: hidden=128, layers=2, state=16, kernel=4, remain=0.5, max_len=10
- C3: hidden=128, layers=2, state=32, kernel=4, remain=0.7, max_len=10
- C4: hidden=160, layers=2, state=32, kernel=8, remain=0.5, max_len=20
- lr search: [1e-4, 4e-3]

### B) lastfm0.03 (시간/안정성 고려)

원칙:
- combo는 4개 유지
- 다만 lastfm에서는 극단적으로 무거운 설정 비중을 줄여서 탐색량을 확보

#### BSARec (L_BS_C1~C4)
- C1: hidden=128, layers=1, heads=4, max_len=10
- C2: hidden=128, layers=2, heads=4, max_len=10
- C3: hidden=128, layers=2, heads=8, max_len=10
- C4: hidden=160, layers=2, heads=4, max_len=20
- lr search: [8e-5, 1.5e-3]

#### PAtt (L_PA_C1~C4)
- C1: hidden=128, layers=1, heads=2, gamma=0.05, max_len=10
- C2: hidden=128, layers=2, heads=2, gamma=0.10, max_len=10
- C3: hidden=128, layers=2, heads=4, gamma=0.10, max_len=10
- C4: hidden=160, layers=2, heads=4, gamma=0.15, max_len=20
- lr search: [1e-4, 1.2e-3]

#### FAME (L_FA_C1~C4)
- C1: hidden=128, layers=1, heads=2, experts=2, max_len=10
- C2: hidden=128, layers=2, heads=4, experts=4, max_len=10
- C3: hidden=128, layers=2, heads=4, experts=6, max_len=10
- C4: hidden=160, layers=2, heads=4, experts=4, max_len=20
- lr search: [6e-5, 8e-4]

#### FENRec (L_FE_C1~C4)
- C1: hidden=128, layers=1, heads=2, cl_w=0.05, cl_t=0.20, max_len=10
- C2: hidden=128, layers=2, heads=2, cl_w=0.10, cl_t=0.20, max_len=10
- C3: hidden=128, layers=2, heads=4, cl_w=0.10, cl_t=0.10, max_len=10
- C4: hidden=160, layers=2, heads=4, cl_w=0.15, cl_t=0.15, max_len=20
- lr search: [8e-5, 1.0e-3]

#### SIGMA (L_SI_C1~C4)
- C1: hidden=128, layers=1, state=16, kernel=4, remain=0.5, max_len=10
- C2: hidden=128, layers=2, state=16, kernel=4, remain=0.5, max_len=10
- C3: hidden=128, layers=2, state=32, kernel=4, remain=0.7, max_len=10
- C4: hidden=160, layers=2, state=32, kernel=8, remain=0.5, max_len=20
- lr search: [5e-5, 8e-4]

실행 방식(요청 반영):
- 한 모델 실행 시: GPU0~GPU3에 wave별 4개 combo를 각각 1개씩 고정 배치
  - wave1: C1->GPU0, C2->GPU1, C3->GPU2, C4->GPU3
  - wave2: C5->GPU0, C6->GPU1, C7->GPU2, C8->GPU3
- 전체 순서: wave1에서 dataset 순서대로(예: Kuai -> lastfm) 완료 후, wave2에서 다시 같은 dataset 순서 반복
- GPU 큐는 독립 실행: 다른 GPU/phase/dataset 종료를 기다리지 않고, 자기 큐 다음 task를 즉시 실행
- summary.csv 자동 갱신: 각 task 완료 후 + 전체 종료 후 dataset별 재갱신
- special logging 지원: `--special-logging` 옵션으로 활성화

combo 성격 배치(요청 반영):
- C1~C4: 일반형 3개 + 특이형 1개
- C5~C8: 일반형 1개 + 특이형 3개
- 일반형은 가급적 P0에서 덜 본 축(예: 특정 layers/heads 조합) 위주로 구성

실행 스크립트:
- experiments/run/baseline/p1_target5_fixed_lr.sh

예시:
- dry-run
  - `bash experiments/run/baseline/p1_target5_fixed_lr.sh --datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03 --gpus 0,1,2,3 --base-max-evals 20 --special-logging --dry-run`
- 실제 실행
  - `bash experiments/run/baseline/p1_target5_fixed_lr.sh --datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03 --gpus 0,1,2,3 --base-max-evals 20 --special-logging`

---

## 모델/데이터셋별 가중치, lr 범위, max_eval=20 기준 러프 시간

### 1) eval 가중치 규칙

기준 입력값: base_max_eval (예: 20)

계산식:
- eval(model, dataset) = max(3, floor(base_max_eval * dataset_weight * model_weight))

dataset_weight:
- KuaiRecLargeStrictPosV2_0.2: 1.00
- lastfm0.03 (dataset 두 개 함께 돌릴 때): 0.75
- lastfm0.03만 단독 실행 시: 1.00

model_weight (가벼운 BSARec=100% 기준):
- BSARec: 1.00 (100%)
- PAtt: 0.90 (90%)
- FAME: 0.75 (75%)
- FENRec: 0.65 (65%)
- SIGMA: 0.50 (50%)

비고:
- 최소 eval은 3으로 강제
- 과도한 축소(예: 10분의1 수준)는 방지됨

### 2) base_max_eval=20일 때 실제 적용 eval

KuaiRecLargeStrictPosV2_0.2:
- BSARec: 20
- PAtt: 18
- FAME: 15
- FENRec: 13
- SIGMA: 10

lastfm0.03 (Kuai와 함께 실행 시, dataset_weight=0.75):
- BSARec: 15
- PAtt: 13
- FAME: 11
- FENRec: 9
- SIGMA: 7

사용자 예시 검증:
- lastfm SIGMA = 20 * 0.75 * 0.50 = 7.5 -> floor 7

### 3) 모델/데이터셋별 lr 탐색 범위 요약

KuaiRecLargeStrictPosV2_0.2:
- BSARec: [3e-4, 1e-2]
- PAtt: [5e-4, 1e-2]
- FAME: [2e-4, 6e-3]
- FENRec: [3e-4, 8e-3]
- SIGMA: [1e-4, 4e-3]

lastfm0.03:
- BSARec: [8e-5, 1.5e-3]
- PAtt: [1e-4, 1.2e-3]
- FAME: [6e-5, 8e-4]
- FENRec: [8e-5, 1.0e-3]
- SIGMA: [5e-5, 8e-4]

### 4) max_eval=20 기준 러프 총 시간 (4 GPU, 모델 순차, 모델당 combo 4개 병렬)

가정:
- 모델별 소요시간 = 해당 모델의 4개 combo 중 가장 오래 걸리는 combo 시간
- runtime은 P0 로그 + 누락 combo에 대한 보수적 추정으로 계산
- 오차 범위는 큼(대략 +-30%)

#### A. Kuai만 실행

예상 모델별 벽시계 시간:
- BSARec: 약 1.6h
- PAtt: 약 1.4h
- FAME: 약 1.6h
- FENRec: 약 2.2h
- SIGMA: 약 2.1h

합계: 약 8.9h

#### B. Kuai -> lastfm 연속 실행

lastfm 예상 모델별 벽시계 시간(가중 eval 적용):
- BSARec: 약 4.5h
- PAtt: 약 4.3h
- FAME: 약 5.0h
- FENRec: 약 3.5h
- SIGMA: 약 3.9h

합계:
- Kuai 약 8.9h + lastfm 약 21.2h = 약 30.1h

해석:
- 두 데이터셋 연속 실행 시 lastfm 구간이 전체 시간 대부분을 차지함.
- 그래도 가중치 방식 덕분에 heavy 모델도 최소 7~11 eval 수준은 확보되어 "탐색이 안 되는" 상태는 피할 수 있음.
