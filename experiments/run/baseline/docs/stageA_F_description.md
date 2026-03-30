# Baseline Tuning Pipeline v3 (Stage A~F)

## 목적
기존의 넓고 불안정한 탐색을 단계형 파이프라인으로 쪼개서, 속도와 일관성을 확보한다.

- Track: `baseline`
- Anchor datasets: `lastfm0.03`, `amazon_beauty`
- Core models: `SASRec`, `GRU4Rec`, `DuoRec`, `DIFSR`, `FAME`
- 공통 평가: `eval_mode=session_fixed`, `feature_mode=full_v3`

## 왜 이 앵커/모델 조합인가
- `lastfm0.03`: 모델 간 분해력이 높아 튜닝 신호가 잘 드러남.
- `amazon_beauty`: sparse/long-tail 성향으로 regularization 견고성 확인에 유리.
- Core5:
  - `SASRec`: transformer baseline anchor
  - `GRU4Rec`: conventional anchor
  - `DuoRec`: contrastive SOTA anchor
  - `DIFSR`: feature injection anchor
  - `FAME`: MoE anchor

## Stage 구성
### Stage A: LR 신뢰구간 스캔
- LR-only 탐색(6개 narrow band)
- 기본 `max_len=10`, LR clamp `[8e-5, 1e-2]`
- 모델별 고정 hparam + 모델별 경량 예산
- 출력: 모델/데이터셋별 LR 후보 선별 (`Top-2 + Stability`)

### Stage B: 큰 구조 튜닝
- 대상: `hidden/dim`, `layers`, `heads`, `ffn ratio`
- `max_len=10` 고정
- Stage A 상위 LR 후보에서만 수행

### Stage C: 중간 중요 knob 튜닝
- 공통: `max_len`은 `{10, 15, 20}`만 허용
- 모델별 핵심 1개 축 중심:
  - FAME: `num_experts`
  - DIFSR: `fusion_type/lambda_attr`
  - DuoRec: `contrast/tau/lmd` 계열
  - SASRec/GRU4Rec: 구조 보정축 1개

### Stage D: 미세 knob 튜닝
- 대상: `dropout`, `weight_decay`, 기타 미세 regularization
- 구조/중요 knob 고정 후 좁은 범위 정밀화

### Stage E: 재-LR + 소수 seed 검증
- B~D에서 바뀐 구조 기준으로 local LR 재탐색(보통 3 bands)
- 소수 seed(기본 2)로 안정성 1차 확인

### Stage F: Full 실행
- 최종 비교용 실행 단계
- 다중 seed(기본 3+) + full matrix(공식 비교 대상 전체)
- 이 단계는 튜닝이 아니라 최종 재현/비교 목적

## Stage 전이 규칙
모든 Stage에서 동일 규칙 적용.

1. 기본 승급: `Top-2 + Stability`
2. 안정성 패널티:
   - Stage A 기준 점수:  
     `score = valid_mrr20 - 0.5 * std(top3_trial_valid_mrr20) - 0.01 * (1 - completion_ratio)`
3. 경계 보정:
   - best가 LR 최외곽 band이거나,
   - best LR가 band 경계 10% 이내면,
   - 인접 band 1개를 추가 승급
4. 산출물:
   - 각 stage 종료 시 dataset별 `stageX_candidates.json` 저장
   - 다음 stage는 이 파일만 입력으로 사용

## 운영 원칙
- 초기 stage는 속도 우선(짧은 epoch/낮은 patience)으로 분포를 먼저 본다.
- 후반 stage(E/F)에서만 seed와 full을 늘린다.
- Stage A에서 DuoRec은 경량 예산으로 운영하고, 최종 비교(Stage F)에서 동일 공정으로 재평가한다.
