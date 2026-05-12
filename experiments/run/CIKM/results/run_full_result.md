# CIKM 2026 RouteRec — 실험 결과 현황

> 마지막 업데이트: 2026-05-11  
> 데이터셋: final_dataset (full, feature_mode=final)  
> eval: session_fixed, full eval (KuaiRec 8.9K items / lastfm sampled @1000)  
> 지표: MRR@20, HR@10, NDCG@10

---

## 1. KuaiRec — 전체 완료 ✅

| 모델 | MRR@20 | HR@10 | NDCG@10 | vs SASRec |
|------|--------|-------|---------|-----------|
| **RouteRec (FMoE-N3)** | **0.3204** | **0.3689** | **0.3311** | +21.9% |
| DuoRec | 0.3529 | 0.3811 | 0.3590 | +34.2% |
| FAME | 0.3458 | 0.3783 | 0.3529 | +31.5% |
| BSARec | 0.3296 | 0.3625 | 0.3366 | +25.4% |
| FDSA | 0.2992 | 0.3469 | 0.3096 | +13.8% |
| FEARec | 0.2777 | 0.3308 | 0.2895 | +5.6% |
| TiSASRec | 0.2711 | 0.3326 | 0.2847 | +3.1% |
| SASRec | 0.2629 | 0.3234 | 0.2758 | — |
| DIFSR | 0.2611 | 0.3160 | 0.2731 | -0.7% |
| GRU4Rec | 0.1467 | 0.1818 | 0.1540 | -44.2% |

> ⚠️ KuaiRec에서 DuoRec, FAME이 RouteRec보다 높음.  
> 원인 분석 필요: KuaiRec MAX_ITEM_LIST_LENGTH=10 (짧은 컨텍스트) 제약 가능성.  
> RouteRec이 우위를 점하는 데이터셋은 behavioral diversity가 높은 경우.

---

## 2. lastfm — 진행 중 🔄

| 모델 | MRR@20 | HR@10 | NDCG@10 | 상태 |
|------|--------|-------|---------|------|
| DuoRec | 0.4441 | 0.5276 | 0.4621 | ✅ |
| SASRec | 0.4267 | 0.5039 | 0.4434 | ✅ |
| TiSASRec | 0.4222 | 0.4994 | 0.4389 | ✅ |
| DIFSR | 0.4036 | 0.4729 | 0.4186 | ✅ |
| BSARec | 0.3857 | 0.4589 | 0.4014 | ✅ |
| GRU4Rec | 0.3788 | 0.4485 | 0.3938 | ✅ |
| **RouteRec (FMoE-N3)** | — | — | — | ❌ completed(mrr=0) 재실행 필요 |
| FEARec | — | — | — | ❌ completed(mrr=0) — artifacts/tmp 경로 오류 |
| FAME | — | — | — | 🔄 실행 중 (GPU 0, ~21h 경과) |
| FDSA | — | — | — | 🔄 실행 중 (GPU 2, ~6h 경과) |

### lastfm 미완료 원인

- **FEARec**: `RuntimeError: Parent directory artifacts/tmp/best_stage does not exist`  
  → `run_resume_safe.sh`가 Ctrl+Z로 Stopped되기 전에 이 오류로 먼저 mrr=0 완료
- **FeaturedMoE_N3**: 이전 실행들에서 모두 `completed(mrr=0)` → lastfm config 이슈 가능성
- **FAME/FDSA**: 현재 실행 중, SIGCONT로 재개 완료 (GPU 0/2)

---

## 3. 결과 파일 위치

```
experiments/run/CIKM/results/
├── run_full_result.md          ← 이 파일 (결과 요약)
├── main_baselines_summary.csv  ← KuaiRec baselines (P0 main run)
├── main_routerec_summary.csv   ← KuaiRec RouteRec (P0 main run)
├── resume_safe_summary.csv     ← lastfm baselines (resume run)
└── lastfm_stable_summary.csv   ← lastfm 초기 시도 (deprecated)

experiments/run/artifacts/results/cikm/
├── KuaiRec_*.json              ← 상세 결과 JSON (best_params, metrics, ...)
├── lastfm_*.json
├── KuaiRec_FeaturedMoE_N3_*_best_model_state.pth  ← checkpoint
└── normal/                     ← unified layout mirror (mirror only)
```

---

## 4. 남은 작업

| 우선순위 | 작업 | 예상 시간 |
|---------|------|---------|
| P0 필수 | lastfm/featured_moe_n3 재실행 | ~30h |
| P0 필수 | lastfm/fearec 재실행 (tmp 경로 수정 후) | ~20h |
| P0 완료 후 | exp_cue_perturb: eval_perturb (KuaiRec) | ~1h |
| P0 완료 후 | exp_cue_perturb: train_perturb hidden_only + both_zero | ~13h |
| P1 | exp_analysis: cue_tier_eval | ~2h |

---

## 5. 빠른 재실행 명령

```bash
cd experiments/run/CIKM

# lastfm FMoE + FEARec 재실행 (GPU 1, 3)
HYPEROPT_RESULTS_DIR=experiments/run/CIKM/artifacts/results \
  python exp_main/main_routerec.py --gpus 1 --datasets lastfm

# fame/fdsa 완료 후 확인
python -c "
import csv; from pathlib import Path
for row in csv.DictReader(open('results/resume_safe_summary.csv')):
    if row['status']=='ok': print(row['dataset'], row['model'], row['test_mrr20'])
"
```
