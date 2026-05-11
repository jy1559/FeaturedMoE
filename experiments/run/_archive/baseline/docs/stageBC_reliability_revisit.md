# Stage B/C Reliability Revisit Plan

## 왜 재검증이 필요한가
- 일부 모델/데이터셋은 Stage C에서 상위 프로파일 격차가 매우 작고, early stop 비율이 높다.
- 이 경우 현재 Top-1이 진짜 최선인지, 아니면 빠른 종료/예산 영향인지 구분이 어렵다.

## 현재 분해력 진단 (핵심)
- 재검증 우선 (gap 작고 early-stop 높음)
  - `amazon_beauty-SASRec`: Stage C `gap12=0.0003`, `early_stop_ratio=0.816`
  - `amazon_beauty-DIFSR`: Stage C `gap12=0.0009`, `early_stop_ratio=1.000`
  - `lastfm0.03-DIFSR`: Stage C `gap12=0.0011`, `early_stop_ratio=0.514` (성능은 높아 확인 가치 큼)
- 재검증 필요 낮음 (gap 충분)
  - `lastfm0.03-DuoRec`: `gap12=0.0091`
  - `amazon_beauty-DuoRec`: `gap12=0.0065`

## 권장 전략: Hybrid
- `Track-1 (Selective B/C Revisit)`
  - 분해력 부족 조합만 B/C를 재실행해서 ranking 신뢰성 확보.
- `Track-2 (Aggressive D)`
  - 분해력이 이미 충분한 조합은 D를 공격적으로 확장해 peak 탐색.

## Track-1 상세 (선택 재실행)

### 대상 조합
- `amazon_beauty`: `SASRec`, `DIFSR`
- `lastfm0.03`: `DIFSR` (필수), `SASRec` (선택)

### 실행 원칙
- B 재실행: 구조 분해력 재확인 목적
  - `tune_epochs`/`patience` 증가 + seed 2개
- C 재실행: focus knob 재확인 목적
  - `b-topk 3`로 parent 다양성 확대
  - seed 2개, patience 증가

### 권장 커맨드
```bash
# B revisit (selective)
bash experiments/run/baseline/stageB_structure.sh \
  --datasets lastfm0.03,amazon_beauty \
  --models sasrec,difsr \
  --gpus 0,1 \
  --seeds 1,2 \
  --tune-epochs-default 56 \
  --tune-patience-default 9 \
  --max-evals-default 8

# C revisit (selective, parent 다양성 확대)
bash experiments/run/baseline/stageC_focus.sh \
  --datasets lastfm0.03,amazon_beauty \
  --models sasrec,difsr \
  --b-topk 3 \
  --gpus 0,1 \
  --seeds 1,2 \
  --tune-epochs-default 60 \
  --tune-patience-default 9 \
  --max-evals-default 8
```

## Track-2 상세 (Aggressive D 우선)
- 대상: `lastfm0.03`의 `SASRec/DIFSR/DuoRec`, `amazon_beauty`의 `SASRec/DuoRec`.
- Stage D에서 aggressive 범위(`dropout +-0.06`, `wd x[0.45,2.2]`)를 적용.
- 저잠재력군(`amazon_beauty`의 `GRU4Rec/DIFSR/FAME`)은 저예산 유지.

## 신뢰성 판정 기준 (최종)
- ranking 안정성:
  - 재실행 전후 Top-2 순위가 seed 평균 기준으로 일치하거나,
  - 순위가 바뀌어도 delta가 `<= 0.001`이면 동급 처리
- 실전 승급:
  - seed mean valid MRR@20 기준으로 Stage E/F 후보 확정
  - seed std `<= 0.0025`면 안정 후보
