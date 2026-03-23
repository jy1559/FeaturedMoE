# Phase 12 — Layout Composition Wide Search (P12)

## 1) 목표
- 동일 stage set(`macro/mid/micro`)에서 layout/composition 방식이 성능에 미치는 영향을 검증한다.
- attention 배치, stage 반복, bundle 병렬 조합, aggregation 방식을 비교한다.

## 2) 실행 스펙 (고정)
- 실행 타입: `wide`
- setting 수: `32`
- hparam 수: `1 (H1)`
- seed 수: `1 (S1)`
- 총 run 수: `32 x 1 x 1 = 32`
- 실행 엔트리:
  - `experiments/run/fmoe_n3/run_phase12_layout_composition.py`
  - `experiments/run/fmoe_n3/phase_12_layout_composition.sh`

## 3) 로그/요약 규칙
- 로그 루트: `experiments/run/artifacts/logs/fmoe_n3/phase12_layout_composition_v1/<dataset>/`
- 로그 파일명: `P12_<axis_id>_<axis_desc>_<setting_id>_<setting_desc>.log`
- summary: 동일 경로의 `summary.csv`
- summary append 이벤트(강제):
  - `trigger=trial_new_best`
  - `trigger=run_complete`
- resume 정책:
  - 완료 로그(`"[RUN_STATUS] END status=normal"`)는 skip
  - 미완료 로그는 preamble overwrite 후 재실행

## 4) Bundle 코어 확장 (필수)
- `layer_layout`에 bundle 토큰 문법 허용:
  - `bundle_<stage1>_<stage2>[_<stage3>]_<agg>`
  - `agg in {sum, mean, learned, router}`
- executor 확장:
  - bundle branch를 병렬 계산 후 집계
  - `learned`: 정적 학습 가중치 집계
  - `router`: hidden + feature conditioned 집계

## 5) 세팅 매트릭스 (32)

### 5.1 Attention / layout variants (10)
- `P12-00_ATTN_ONESHOT`
- `P12-01_ATTN_MACRO_ONLY`
- `P12-02_ATTN_MICRO_BEFORE`
- `P12-03_NO_ATTN_ONLY_MOEFFN`
- `P12-04_LAYER_PLUS_MOEFFN`
- `P12-05_MACRO_REPEATED`
- `P12-06_MID_REPEATED`
- `P12-07_MICRO_REPEATED`
- `P12-08_MACRO_NOLOCALATTN`
- `P12-09_MID_NOLOCALATTN`

### 5.2 Bundle base sets (15)
- `P12-10_BUNDLE_MACROMID_SUM`
- `P12-11_BUNDLE_MACROMID_MEAN`
- `P12-12_BUNDLE_MACROMID_LEARNED`
- `P12-13_BUNDLE_MIDMICRO_SUM`
- `P12-14_BUNDLE_MIDMICRO_MEAN`
- `P12-15_BUNDLE_MIDMICRO_LEARNED`
- `P12-16_BUNDLE_MACROMICRO_SUM`
- `P12-17_BUNDLE_MACROMICRO_MEAN`
- `P12-18_BUNDLE_MACROMICRO_LEARNED`
- `P12-19_BUNDLE_ALL_SUM`
- `P12-20_BUNDLE_ALL_MEAN`
- `P12-21_BUNDLE_ALL_LEARNED`
- `P12-22_BUNDLE_MACROMID_THEN_MIDMICRO_LEARNED`
- `P12-23_BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED`
- `P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED`

### 5.3 8배수 보강 (+7)
- `P12-25_BUNDLE_ALL_ROUTER_CONDITIONED`
- `P12-26_BUNDLE_MACROMID_THEN_MIDMICRO_SUM`
- `P12-27_BUNDLE_MACROMID_THEN_MIDMICRO_MEAN`
- `P12-28_BUNDLE_MACROMID_THEN_MIDMICRO_ROUTER_CONDITIONED`
- `P12-29_BUNDLE_MACROMICRO_THEN_MIDMICRO_SUM`
- `P12-30_BUNDLE_MACROMICRO_THEN_MIDMICRO_MEAN`
- `P12-31_BUNDLE_MACROMICRO_THEN_MIDMICRO_ROUTER_CONDITIONED`

## 6) 구현 메모
- phase runner는 `layer_layout`만 세팅별로 주입하고 공통 실행/저장/summary는 wide 공용 유틸을 사용한다.
- bundle 설정에서도 stage별 gate/aux logging이 유지되도록 executor path를 통합한다.
