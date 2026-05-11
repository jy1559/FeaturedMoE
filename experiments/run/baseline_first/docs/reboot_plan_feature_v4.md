# Reboot Plan (feature_added_v4 + seen/cold 분리 평가)

## 목적
- 기존 `feature_added_v3` 기반 session_fixed 실험을 v4 프로토콜로 재시작.
- main 성능은 seen-target 기준으로 평가하고, unseen-target(cold)은 별도 지표로 함께 기록.

## v4 데이터 생성
- 스크립트: `experiments/tools/build_feature_v4_from_v3.py`
- 입력: `Datasets/processed/feature_added_v3/<dataset>/<dataset>.train|valid|test.inter`
- 출력: `Datasets/processed/feature_added_v4/<dataset>/<dataset>.train|valid|test.inter`
- 규칙:
  - train은 v3 그대로 유지
  - valid+test 세션 병합 후 session start time strict 정렬, 50/50 재분할
  - valid/test에서 non-target unseen interaction drop
  - target unseen은 유지

실행 예시:
```bash
python experiments/tools/build_feature_v4_from_v3.py --overwrite
```

## 평가/로깅
- 설정:
  - `feature_mode=full_v4`
  - `exclude_unseen_target_from_main_eval=true`
  - `log_unseen_target_metrics=true`
- 동작:
  - main eval: unseen target 제외
  - special eval: seen/unseen 모두 집계
  - cold 지표는 `target_popularity_abs.cold_0`에서 추출

## baseline_2 파이프라인
- 러너: `experiments/run/baseline_2/run_staged_tuning.py`
- Stage: A/B/C/D
- 자동 승급: A top8 → B top4 → C(top4 변이8) top2 → D(top2 변이8) top1
- stage별 산출:
  - `summary.csv`: 실행 결과 전체
  - `promotion.csv`: 다음 stage 승급 후보
  - `leaderboard.csv`: dataset/model best
  - `manifest.json`: 예산/설정/카운트 메타

## 실행
```bash
bash experiments/run/baseline_2/run_all_stages.sh
```

필요 시 stage 단독 실행:
```bash
bash experiments/run/baseline_2/stageA_baseline2.sh
bash experiments/run/baseline_2/stageB_baseline2.sh
bash experiments/run/baseline_2/stageC_baseline2.sh
bash experiments/run/baseline_2/stageD_baseline2.sh
```
