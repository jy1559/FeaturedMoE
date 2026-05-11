# 결과/시각화 운영 플레이북(Playbook)

## 1. 문서 구조 규약(Convention)
- 데이터 산출물(Data): `docs/data/phase8_9/*.csv`
- 결과 문서(Result): `docs/results/phaseX_[주제].md`
- 시각화 노트북(Visualization): `docs/visualization/phaseX_[주제].ipynb`
- 계획 문서(Plan): `docs/plans/*.md`

## 2. 지표/필터 표준
- 메인 지표(Main metric): `best valid MRR@20`
- 보조 지표(Sub metric): `test MRR@20`
- 본통계(Main stats): `n_completed >= 20`
- partial/pending은 본통계에서 분리해 별도 표로 저장.

## 3. 중복 제거(Dedup) 규칙
1. key: `run_phase`
2. 최신 `timestamp` 우선
3. 동률이면 `n_completed` 큰 값 우선
4. 그래도 동률이면 `best_mrr@20` 큰 값 우선

## 4. 결과 문서 필수 섹션
1. 실험 목적(왜 했는지)
2. 집계 정책(필터/중복 제거)
3. ID 설명(candidate/concept/combo/hparam 의미)
4. 표 + 표 해석(숫자 요약만 금지)
5. 주장 가능 포인트(논문 연결 문장)

## 5. 노트북 필수 그래프
1. 성능 히트맵(Performance heatmap)
2. 시드 분포(Box/Violin)
3. 평균-표준편차 산점도(Mean-Std stability)
4. 특수 슬라이스 막대그래프(Special slices)
5. 진단 대 성능 산점도(Diag vs metric scatter)
6. 상관 히트맵(Correlation heatmap)
7. PCA 산점도 + 로딩 막대그래프(PCA scatter + loading)

## 6. 그래프 표현 규칙
- Figure 내부 텍스트는 영어만 사용.
- 설명은 `print()`로 한국어 출력 가능.
- 범례는 가능한 그래프 외부(`bbox_to_anchor`)로 배치해 데이터 가림 최소화.

## 7. 운영 루프(Loop)
1. 탐색 phase: 축별 설정 비교(보통 seed 1)
2. 후보 압축: 축별 winner와 위험 신호 정리
3. 검증 phase(X_2): `base x hparam x seed`
4. 문서화: 결과 md + 시각화 ipynb + handoff 갱신

## 8. 체크리스트
1. 집계 스크립트 재실행
2. main/pending row count 검증
3. ID 설명 표 최신화
4. 표별 해석 문장 포함 여부 확인
5. nbconvert 실행 검증
