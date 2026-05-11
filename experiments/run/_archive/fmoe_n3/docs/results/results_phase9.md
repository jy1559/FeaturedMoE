# Phase9 AuxLoss 결과 요약 및 Phase9_2 후보 선정

작성일: 2026-03-22  
대상: `FeaturedMoE_N3` / `KuaiRecLargeStrictPosV2_0.2`

---

## 1) 집계 기준

- source axis/phase: `phase9_auxloss_v1 / P9`
- 집계 소스: `experiments/run/artifacts/results/fmoe_n3/*.json`
- 후보 선별 필터:
1. `run_phase`가 `P9_*` 형식
2. `n_completed >= 20`
3. `interrupted == false`
4. concept 내 우선순위: `best_valid_mrr20` 내림차순, 동률 시 `test_mrr20` 내림차순

참고: `P9_B4_C3_F3_S1`는 최신 성공 기록이 `n_completed=1`인 partial run이므로 최종 후보 선별에서는 제외했다.

---

## 2) Concept별 상위 성능 (요약)

| concept | run_phase | best_valid_mrr20 | test_mrr20 | 비고 |
|---|---|---:|---:|---|
| `C0` Natural | `P9_B4_C0_N4_S1` | 0.0827 | 0.1591 | z 단독 안정화 |
| `C1` CanonicalBalance | `P9_B3_C1_B1_S1` | 0.0826 | 0.1595 | balance+z 표준형 |
| `C2` Specialization | `P9_B1_C2_S3_S1` | 0.0818 | 0.1596 | sharpness + monopoly |
| `C3` FeatureAlignment | `P9_B2_C3_F2_S1` | 0.0820 | 0.1593 | route_prior + z |

---

## 3) Phase9_2 최종 후보 4개

Phase9_2는 아래 4개를 고정 후보로 사용한다.

| candidate_id | source_run_phase | base | concept/combo | main/support aux |
|---|---|---|---|---|
| `K1` | `P9_B4_C0_N4_S1` | `B4` | `C0/N4` | `z` / `none` |
| `K2` | `P9_B3_C1_B1_S1` | `B3` | `C1/B1` | `balance` / `z` |
| `K3` | `P9_B1_C2_S3_S1` | `B1` | `C2/S3` | `route_sharpness` / `route_monopoly` |
| `K4` | `P9_B2_C3_F2_S1` | `B2` | `C3/F2` | `route_prior_strong` / `z` |

---

## 4) Phase9_2 매트릭스 사양

- axis/phase: `phase9_2_verification_v2 / P9_2`
- matrix: `4(candidate) x 4(hparam) x 4(seed) = 64 runs`
- seed: `S1,S2,S3,S4`
- run_phase naming:
  - `P9_2_<CANDIDATE_ID>_<BASE>_<CONCEPT>_<COMBO>_<HVAR>_S<seed>`
  - 예: `P9_2_K4_B2_C3_F2_H3_S2`

### hparam 조합 (고정 4종)

| hvar | embedding | d_ff | d_expert_hidden | d_router_hidden | weight_decay | hidden_dropout |
|---|---:|---:|---:|---:|---:|---:|
| `H1` | 128 | 256 | 128 | 64 | 1e-6 | 0.15 |
| `H2` | 160 | 320 | 160 | 80 | 5e-7 | 0.12 |
| `H3` | 160 | 320 | 160 | 80 | 2e-6 | 0.18 |
| `H4` | 112 | 224 | 112 | 56 | 3e-6 | 0.20 |

---

## 5) 실행/산출 경로

### 실행

```bash
bash experiments/run/fmoe_n3/phase_9_2_verification.sh \
  --datasets KuaiRecLargeStrictPosV2_0.2 \
  --gpus 4,5,6,7 \
  --seeds 1,2,3,4
```

### 로그/매니페스트/요약

- 로그 루트: `experiments/run/artifacts/logs/fmoe_n3/phase9_2_verification_v2/P9_2/KuaiRecLargeStrictPosV2_0.2/FMoEN3`
- 매트릭스: `experiments/run/artifacts/logs/fmoe_n3/phase9_2_verification_v2/P9_2/KuaiRecLargeStrictPosV2_0.2/verification_matrix.json`
- 요약 CSV: `experiments/run/artifacts/logs/fmoe_n3/phase9_2_verification_v2/P9_2/KuaiRecLargeStrictPosV2_0.2/summary.csv`

resume/skip는 아래 둘 중 하나를 만족하면 완료 런으로 판단한다.
- result index에서 `best_mrr` 존재 + `n_completed > 0`
- log 파일에 `[RUN_STATUS] END status=normal` 존재

