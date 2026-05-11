# CIKM 2026 – RouteRec Experiments

**Paper**: RouteRec: Behavior-Guided Expert Routing for Sequential Recommendation  
**Submission deadline**: Full paper May 23 2026 AoE  
**Datasets**: KuaiRec (full, 3.2M train, 8,966 items) · lastfm (full, 11M train, 547K items)  
**Feature mode**: `final` → `Datasets/processed/final_dataset/`  
**Eval mode**: `session_fixed` (pre-split `.train/.valid/.test.inter`)

---

## Folder Structure

```
CIKM/
├── common.py            ← shared infrastructure (search spaces, job runner)
├── results/             ← CSV summaries written here automatically
│   ├── main_baselines_summary.csv
│   ├── main_routerec_summary.csv
│   └── ...
├── logs/                ← per-job logs (created on first run)
│
├── exp_main/            ← P0: main comparison table
│   ├── main_baselines.py / .sh   (9 baselines × KuaiRec + lastfm)
│   └── main_routerec.py / .sh    (RouteRec × KuaiRec + lastfm)
│
├── exp_cue_perturb/     ← P1-A: cue-perturbation ablation (shuffle / zero)
├── exp_capacity/        ← P1-B: capacity-matched SASRec-wide baseline
├── exp_seed/            ← P2-A: multi-seed robustness (seeds 42/123/456)
└── exp_analysis/        ← P2-B: BRD analysis & routing visualisation
```

---

## Quick Start

### Run all baselines (single GPU)
```bash
cd experiments/run/CIKM/exp_main
bash main_baselines.sh 0
```

### Run RouteRec (single GPU)
```bash
bash main_routerec.sh 0
```

### Run both in parallel on two GPUs
```bash
bash main_baselines.sh 0 &
bash main_routerec.sh 1 &
wait
```

### Subset run (e.g. only KuaiRec, only SASRec + GRU4Rec)
```bash
python main_baselines.py --gpus 0 --datasets KuaiRec --models sasrec gru4rec
```

---

## Configuration Details

| Setting | Value |
|---------|-------|
| Hyperopt algo | TPE |
| Max evals per job | 5 |
| Tune epochs | 100 |
| Early stopping patience | 10 |
| Valid metric | MRR@20 |
| KuaiRec eval | full (8,966 items) |
| lastfm eval | sampled (1,000 negatives, pop strategy) |
| Seed | 42 |

### Tuned params
Only `learning_rate` and `weight_decay` are tuned, in a narrow range around best configs from sampled-dataset experiments (`KuaiRecLargeStrictPosV2_0.2` and `lastfm0.03`).  
All structural hyperparams (dropout, seq length, model-specific) are **fixed** to their best known values from `experiments/run/final_experiment/`.

---

## Adding a New Experiment

1. Add your runner script to the relevant `exp_*/` subfolder.
2. Import from `../common.py` (already on `sys.path` when the script is in an `exp_*` subfolder).
3. Call `run_jobs_queued(jobs, gpus=..., summary_path=..., run_axis=..., run_phase=...)`.
4. Results land in `CIKM/results/` automatically.

---

## Results

Results are written as CSV rows to `CIKM/results/` as each job completes.  
Fields: `dataset, model, status, valid_mrr20, test_mrr20, valid_hr10, test_hr10, valid_ndcg10, test_ndcg10, elapsed_sec, ...`

See `CIKM_roadmap.md` at the repo root for the full experimental plan and paper narrative.
