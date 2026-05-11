# Baseline WideSearch (Anchor-2, Core-5)

## 목적
`Final_all_datasets`에서 확인된 모델별 편차를 줄이기 위해, 모델 수를 줄이고 프로파일 다양성을 크게 늘린 탐색 트랙.

- Track: `baseline`
- Axis: `WideSearch_anchor2_core5`
- Phase: `P15`
- Eval mode: `session_fixed`
- Feature mode: `full_v3`

## 왜 이 2개 데이터셋/5개 모델인가
### 데이터셋
- `lastfm0.03`: 상대적으로 dense한 구간 (모델 상위권 분해가 잘 보임)
- `amazon_beauty`: sparse/long-tail 특성이 강한 구간 (regularization/profile robustness 확인)

### 모델
- `SASRec`: strong baseline anchor
- `GRU4Rec`: conventional 축 대표
- `DuoRec`: SOTA(contrastive) 축 대표
- `DIFSR`: feature injection 축 대표
- `FAME`: MoE 축 대표

실행 모드:
- `hparam`: 기본 `AUTO4` (모델별 4 profiles), 필요 시 `AUTO16` 가능
- `lr`: 모델별 고정(무난) hparam + LR space 4개만 탐색 (`LR1..LR4`)

## 16 Profile 설계
프로파일 ID: `C{1..4}D{1..4}`

### Concept (C: 큰 방향성)
- `C1 Compact-Short`: 작은 폭/강한 규제/짧은 시퀀스 쪽
- `C2 Balanced-Short`: 기준형(중간 폭/중간 규제)
- `C3 Capacity-Short`: 큰 폭/깊이 증가/완화된 규제
- `C4 Offbeat-Mixed`: 비정형 조합(일부 비표준 dim/규제 조합)

### Detail (D: 디테일 클래스, C 위에 얹히는 핵심 분류)
- `D1 TinyCtx-Stable`: `max_len` 매우 짧음(10 중심), 작은 폭, 보수 LR
- `D2 ShortStd`: 짧은 표준형(20 중심), 기본 폭/규제
- `D3 ShortWide`: 짧은 문맥 + 넓은 폭/큰 FFN(`inner_ratio` 상향), 공격적 LR
- `D4 UnusualMix`: 짧은 문맥 + 비정형 비율(`embedding`, `inner_size` 비표준), 강한 규제

설계 원칙:
- `C`는 “컨셉”만 정의하고, 실제 run의 디테일(폭, LR band, `max_len`, FFN 비율, 일부 model-specific)은 `D`가 주도.
- 최종 16개는 `C x D` 조합으로 생성.
- `max_len`은 대부분 `8~20` 범위(일부 offbeat만 예외).

### LR 적용 규칙
- dataset multiplier:
  - `amazon_beauty x1.00`
  - `lastfm0.03 x1.00`
- clamp:
  - LR `[2e-4, 1e-2]`
  - dropout `[0.05, 0.45]`
- run 내부 탐색은 `learning_rate`만 (`loguniform`)
- 나머지 파라미터는 singleton search로 고정
- `D`/`C` 조합에 따라 LR band 자체도 달라지므로, LR-only여도 profile 간 분리가 큼

모델별 LR band(좁은 구간, D별 분할):
- `SASRec`: D1 `[2e-4,6e-4]`, D2 `[6e-4,2e-3]`, D3 `[2e-3,6e-3]`, D4 `[3e-3,1e-2]`
- `GRU4Rec`: D1 `[2.5e-4,8e-4]`, D2 `[8e-4,2.5e-3]`, D3 `[2.5e-3,7e-3]`, D4 `[4e-3,1e-2]`
- `DuoRec`: D1 `[2e-4,5e-4]`, D2 `[5e-4,1.5e-3]`, D3 `[1.2e-3,4e-3]`, D4 `[3e-3,1e-2]`
- `DIFSR`: D1 `[2e-4,6e-4]`, D2 `[6e-4,2e-3]`, D3 `[2e-3,6e-3]`, D4 `[3e-3,1e-2]`
- `FAME`: D1 `[2e-4,5e-4]`, D2 `[5e-4,1.5e-3]`, D3 `[1.2e-3,4e-3]`, D4 `[3e-3,1e-2]`

`lr` 모드의 4개 기본 space:
- `LR1_2e4_6e4`: `[2e-4, 6e-4]`
- `LR2_6e4_2e3`: `[6e-4, 2e-3]`
- `LR3_2e3_6e3`: `[2e-3, 6e-3]`
- `LR4_3e3_1e2`: `[3e-3, 1e-2]`

`hparam` 모드 기본 `AUTO4`(모델별 추천 4개):
- `SASRec`: `C1D3, C3D3, C2D2, C2D3`
- `GRU4Rec`: `C3D3, C2D3, C2D2, C4D3`
- `DuoRec`: `C1D4, C2D1, C1D3, C2D3`
- `DIFSR`: `C1D3, C2D3, C2D2, C3D3`
- `FAME`: `C1D4, C2D3, C2D2, C3D3`

## 모델별 기준값과 상세 규칙
### 기준(base)
- SASRec: hidden128, layers2, heads4, max_len30, dropout0.15, wd2e-4
- GRU4Rec: hidden160, layers2, max_len30, dropout0.20, wd2e-4
- DuoRec: hidden128, layers2, heads2, max_len24, dropout0.12, wd1.5e-4
- DIFSR: hidden160, layers2, heads4, max_len30, dropout0.15, wd2e-4
- FAME: hidden144, layers2, heads4, max_len30, dropout0.15, wd2e-4

### 공통 계산
- `hidden_size`: base x `C(hidden_mult)` x `D(width_mult)` 기반
- `inner_size`: `hidden_size x inner_ratio` (`D` 중심, 일부 `C` 보정)
- `embedding_size`: 모델/프로파일별로 분리 조정(일부 offbeat 비율 포함)
- hidden/head 정합: hidden을 heads 배수로 정렬
- DIFSR은 `attribute_hidden_size`도 detail별 비율로 분리

### 모델별 detail 규칙
- SASRec:
  - `n_layers/n_heads/inner_size/MAX_ITEM_LIST_LENGTH/dropout_ratio`
  - 기본 heads=4, 단 `D4` + hidden<128이면 heads=2
- GRU4Rec:
  - `num_layers/dropout_prob/MAX_ITEM_LIST_LENGTH`
  - D1: layers 한 단계 감소
  - D3: layers 한 단계 증가(cap3)
  - D4: layers=1 고정
- DuoRec:
  - D1: `contrast=un,tau=0.2,lmd=0.04,lmd_sem=0.0`
  - D2: `contrast=su,tau=0.45,lmd=0.0,lmd_sem=0.06`
  - D3: `contrast=us_x,tau=0.8,lmd=0.1,lmd_sem=0.08`
  - D4: `contrast=un,tau=0.3,lmd=0.06,lmd_sem=0.0`
- DIFSR:
  - D1: `fusion_type=sum,use_attribute_predictor=true,lambda_attr=0.05`
  - D2: `fusion_type=gate,use_attribute_predictor=true,lambda_attr=0.10`
  - D3: `fusion_type=gate,use_attribute_predictor=true,lambda_attr=0.15`
  - D4: `fusion_type=concat,use_attribute_predictor=false,lambda_attr=0.0`
- FAME:
  - `num_experts`: D1=2, D2=3, D3=4, D4=6 (cap6)

## 효율화 정책 (이번 수정)
- wrapper 기본 `--tune-epochs 80` (필요 시 CLI로 조정 가능).
- 실제 run은 모델/프로파일별 동적 cap을 적용:
  - 대략 `16~55 epochs` 중심으로 자동 설정(입력 상한에 따라 변동)
  - `tune_patience`도 run별로 함께 축소
- 짧은 `max_len` 프로파일에서 batch size를 상향해 GPU 처리량을 늘림.
- 결과적으로 full 100epoch를 모든 run이 소모하지 않도록 설계.

## 실행 스크립트
- Python runner:
  - `experiments/run/baseline/run_widesearch_anchor_core.py`
- Shell wrapper:
  - `experiments/run/baseline/phase_15_baseline_widesearch.sh`

기본값:
- `datasets=lastfm0.03,amazon_beauty`
- `models=sasrec,gru4rec,duorec,difsr,fame`
- `search_mode=lr` (wrapper 기본)
- `profiles=AUTO4`
- `lr_spaces=AUTO4`
- `max_evals=10`, `tune_epochs=80`, `tune_patience=12`
- `seeds=1`

## 실행 예시
### Dry-run
```bash
bash experiments/run/baseline/phase_15_baseline_widesearch.sh --dry-run
```

### LR mode (DuoRec 우선)
```bash
bash experiments/run/baseline/phase_15_baseline_widesearch.sh \
  --mode lr \
  --models duorec \
  --lr-spaces AUTO4 \
  --gpus 0,1,2,3
```

### Hparam mode (모델별 4개만)
```bash
bash experiments/run/baseline/phase_15_baseline_widesearch.sh \
  --mode hparam \
  --models duorec,difsr,fame \
  --profiles AUTO4 \
  --gpus 0,1,2,3
```

### Smoke
```bash
bash experiments/run/baseline/phase_15_baseline_widesearch.sh \
  --gpus 0 \
  --smoke-test \
  --smoke-max-runs 4
```

### Full
```bash
bash experiments/run/baseline/phase_15_baseline_widesearch.sh --gpus 0,1,2,3
```

## 로깅/재개 규칙
- shared GPU queue(work-stealing)
- strict 완료 마커: `[RUN_STATUS] END status=normal`
- `--verify-logging` 활성 시 `special_result_file`/`special_log_file` 존재해야 skip
- 로그/결과 경로:
  - logs: `artifacts/logs/baseline/WideSearch_anchor2_core5/<dataset>/<model>/<profile>/...`
  - results: `artifacts/results/baseline/...`
  - logging mirror: `artifacts/logging/baseline/WideSearch_anchor2_core5/<dataset>/<model>/<profile>/...`
- summary 컬럼 추가:
  - `profile_id`, `concept_id`, `detail_id`
  - 기존 valid/test best 컬럼 유지

## 나머지 4개 모델 전이 레시피 (문서용)
- `TiSASRec` -> SASRec 상위 profile 2개 전이
- `FEARec` -> DuoRec 상위 profile 2개 전이
- `BSARec` -> SASRec + DIFSR 상위 profile 교차 전이
- `SIGMA` -> DIFSR profile 중 regularized detail(`D1/D4`) 우선 전이

전이 단계에서는 LR-only `max_evals 6~8`로 빠르게 검증 후 확장 권장.
