# Stage E (Re-LR + Seed Check) - baseline

## 목표
- Stage D에서 고정된 구조/미세 knob를 유지한 채 local LR 재탐색을 수행한다.
- 소수 seed로 1차 안정성을 확인하고 Stage F(full) 후보를 압축한다.

## 입력
- 입력은 Stage D 승급 후보(`Top-1 + Stability`, 일부 모델은 `Top-2`)만 사용.
- Stage C에서 이미 저잠재력으로 확인된 조합은 Stage D에서 개선이 없으면 Stage E 제외.

## Local LR 재탐색 규칙
- Stage D best LR를 `lr*`로 두고 3-band 재탐색:
  - `L`: `[0.75*lr*, 0.90*lr*]`
  - `M`: `[0.90*lr*, 1.10*lr*]`
  - `H`: `[1.10*lr*, 1.35*lr*]`
- clamp: `[8e-5, 1e-2]`.
- Stage D에서 early stop이 매우 빨랐던 저잠재력 조합은 `M` band만 사용(재확인용).

## 공격 모드 (선택)
- Stage D에서 개선폭이 `>= +0.0020`인 조합은 5-band로 확장:
  - `VL`: `[0.60*lr*, 0.75*lr*]`
  - `L`: `[0.75*lr*, 0.90*lr*]`
  - `M`: `[0.90*lr*, 1.10*lr*]`
  - `H`: `[1.10*lr*, 1.35*lr*]`
  - `VH`: `[1.35*lr*, 1.60*lr*]`
- 단, `amazon_beauty`의 저잠재력군(`GRU4Rec/DIFSR/FAME`)에는 적용하지 않는다.

## Seed 정책
- 기본 seed: `2개` (`seed=1,2`).
- 예외(저잠재력 조합):
  - Stage D 개선폭 `< +0.001` 이고 early-stop median `<= 0.35`면 `seed=1`만 수행.
- Stage F 승급 직전 추가 조건:
  - `std(valid_mrr20 across seeds) <= 0.0025`면 안정 후보.

## 예산

### lastfm0.03
- `SASRec`: `max_evals=3`, `tune_epochs=66`, `patience=9`, `seeds=2`
- `DIFSR`: `max_evals=3`, `tune_epochs=66`, `patience=9`, `seeds=2`
- `DuoRec`: `max_evals=3`, `tune_epochs=45`, `patience=7`, `seeds=2`
- `GRU4Rec`: `max_evals=3`, `tune_epochs=60`, `patience=8`, `seeds=2`
- `FAME`: `max_evals=3`, `tune_epochs=52`, `patience=8`, `seeds=2`

### amazon_beauty
- `SASRec`: `max_evals=3`, `tune_epochs=60`, `patience=8`, `seeds=2`
- `DuoRec`: `max_evals=3`, `tune_epochs=42`, `patience=6`, `seeds=2`
- `DIFSR`: `max_evals=1`, `tune_epochs=24`, `patience=3`, `seeds=1`
- `GRU4Rec`: `max_evals=1`, `tune_epochs=20`, `patience=3`, `seeds=1`
- `FAME`: `max_evals=1`, `tune_epochs=24`, `patience=3`, `seeds=1`

## Stage F 승급 기준
- 기본 점수(Seed-aware):
  - `seed_mean_valid_mrr20 - 0.5 * seed_std_valid_mrr20`
- SASRec 대비 유지 기준(모델 유지/중단 판단):
  - `lastfm0.03`: SASRec 대비 `-0.015` 이내면 Stage F 유지
  - `amazon_beauty`: SASRec 대비 `-0.008` 이내면 Stage F 유지
- 현재 A~C 추세 기준 권장 유지군:
  - `lastfm0.03`: `DIFSR`, `SASRec`, `DuoRec` 우선
  - `amazon_beauty`: `SASRec`, `DuoRec` 우선

## 운영 메모
- Stage E는 탐색 단계가 아니라 안정성 확인 단계다.
- 개선폭이 작고 early stop이 빠른 조합에 자원을 재투입하지 않는다.
- Stage F는 비교/재현 목적이므로 Stage E 통과 조합만 full seed로 확장한다.
