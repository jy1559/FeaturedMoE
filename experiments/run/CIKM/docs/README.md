# exp_overall

This directory contains the queue runner for the full 10-model x 6-dataset sweep, plus the evidence files used to choose safe starting anchors.

## Scope

- Datasets: `beauty`, `foursquare`, `retail_rocket`, `movielens1m`, `KuaiRec`, `lastfm`
- Models: 9 baselines plus `featured_moe_n3`
- Queue policy: one global queue across all dataset/model jobs, with no dataset barrier between GPUs and cost-balanced ordering so long jobs start earlier
- Metrics: the runner forces `topk=[1,5,10,20]` so `HR/NDCG/MRR @1/@5/@10/@20` are emitted consistently

## Yes, Real JSON And CSV Inputs Were Used

This doc pass is backed by actual experiment artifacts, not only earlier notes.

Directly consumed sources:

- Raw baseline result JSON: `experiments/run/artifacts/results/final_experiment/*.json`
- Raw RouteRec result JSON: `experiments/run/artifacts/results/results_final_experiment_fmoe/*.json`
- Curated baseline digest: `experiments/run/final_experiment/selected_configs.csv`
- Curated route digest: `experiments/run/final_experiment/selected_configs_final_topk.json`

Generated evidence file:

- `docs/artifact_anchor_catalog.csv`

The catalog is built by:

```bash
cd /workspace/FeaturedMoE/experiments/run/CIKM_8/exp_overall
/venv/FMoE/bin/python tools/build_artifact_anchor_catalog.py
```

## What `artifact_anchor_catalog.csv` Contains

The catalog is the non-CIKM artifact summary requested for anchor archaeology.

- One flattened row per anchor candidate
- Up to 4 candidates per dataset/model pair
- Dataset normalization so sampled sources remain visible
	- `KuaiRecLargeStrictPosV2_0.2 -> KuaiRec`
	- `lastfm0.03 -> lastfm`
- Config reconstruction from `context_fixed + fixed_search + best_params`
- Batch-size backfill from the selected best trial when the config block did not record it directly
- Aggregate metrics such as `mean_test_mrr20`, `mean_valid_mrr20`, `mean_test_hr10`, `mean_valid_hr10`
- Runtime hints such as `mean_avg_epoch_time_sec`, `mean_epochs_run`, `n_completed_max`, `n_recorded_trials_max`
- A flattened hyperparameter surface so the CSV can be sorted or pivoted without reopening raw JSON

The generator now drops empty rows where the recorded result is effectively a shell entry with zero metrics and zero epochs. This keeps the catalog focused on usable anchors instead of failed or incomplete attempts.

## How Candidate Ranking Works

`candidate_rank` is confidence-aware, not pure score ordering.

Ranking order:

1. `stage_priority` where `stage3_seed_confirm > stage2_focus_search > stage1_broad_search`
2. `mean_test_mrr20`
3. `mean_valid_mrr20`
4. `source_file_count`

Interpretation:

- `candidate_rank=1` is the best default anchor for the next run
- `candidate_rank=2..4` are nearby alternatives worth trying when the first anchor is unstable or underperforms
- If you only care about raw peak score, sort the CSV by `mean_test_mrr20` yourself instead of using `candidate_rank`

This ranking rule is deliberate because the user asked for trend-based anchors rather than a single raw-score winner.

## Source Gaps And How They Were Handled

The workspace is not perfectly symmetric across baseline and route artifacts, so the docs explicitly track the gaps.

- `selected_configs.csv` only covers 5 baseline models: `duorec`, `fame`, `fdsa`, `fearec`, `gru4rec`
- The missing 4 baseline models are `sasrec`, `tisasrec`, `bsarec`, and `difsr`
- `selected_configs_final_topk.json` references raw `artifacts/results/final_topk/*.json`, but those raw files are not present in this workspace snapshot
- Raw route coverage under `results_final_experiment_fmoe` is good for `beauty`, `foursquare`, `KuaiRec`, `lastfm`, and `movielens1m`
- Raw route coverage for `retail_rocket` is missing in the workspace snapshot, so that pair must be structurally cross-checked against `selected_configs_final_topk.json`
- `movielens1m` route raw rows exist, but the recorded stage1 broad-search rows are less complete on optimizer fields than the curated route digest, so `selected_configs_final_topk.json` is also used there as a structural cross-check

Important nuance: the curated route digest uses `mean_valid_score` and `mean_test_score`, not the exact same `mrr@20` fields as the raw artifact catalog. It is used here for structure and fallback anchors, not for direct metric-to-metric comparison with `artifact_anchor_catalog.csv`.

## What The Runner Uses Today

The runner and the docs do not serve the same role.

- Runtime anchor selection in `common.py` now prefers `artifact_anchor_catalog.csv`, then falls back to the curated `selected_configs.csv` and `selected_configs_final_topk.json` digests where coverage is incomplete
- Route anchors stay curated-first for `movielens1m` and `retail_rocket`, because the curated digest is the safer optimizer source for `movielens1m` and the raw route artifacts are missing for `retail_rocket`
- The runtime applies pair-aware search budgets so lighter pairs get more hyperopt tries while full `KuaiRec` heavy baselines and full `lastfm` stay narrow or single-eval
- The runtime also applies explicit full-data safety overrides for the known hard cases, especially full `lastfm` and full `KuaiRec`

In other words, the CSV is now both the archaeology table and the primary launch-time anchor source, while the curated digests remain the fallback and route-optimizer correction layer.

## Dataset Root Policy

- Baselines use `Datasets/processed/final_dataset_light`
- `featured_moe_n3` uses `Datasets/processed/final_dataset`

The baseline path also injects `load_col.item=[]` so the light-dataset policy is explicit at launch time.

## Full-Data Safety Overrides

Historical sampled winners are not allowed to override full-data safety evidence.

- Full `lastfm`: stability-first execution, fixed-config bias, `MAX_ITEM_LIST_LENGTH=10` bias, and smaller batches
- Full `KuaiRec`: explicit overrides already wired for `fdsa` and `featured_moe_n3`

This is why `hparams.md` separates sampled historical quality from full-data launch safety.

## Practical Commands

Regenerate the artifact catalog:

```bash
cd /workspace/FeaturedMoE/experiments/run/CIKM_8/exp_overall
/venv/FMoE/bin/python tools/build_artifact_anchor_catalog.py
```

Dry-run the full table:

```bash
cd /workspace/FeaturedMoE/experiments/run/CIKM_8/exp_overall
/venv/FMoE/bin/python main.py --dry-run
```

Run selected datasets on selected GPUs:

```bash
cd /workspace/FeaturedMoE/experiments/run/CIKM_8/exp_overall
/venv/FMoE/bin/python main.py --gpus 0 1 --datasets beauty foursquare retail_rocket
```

Resume while skipping already recorded rows:

```bash
cd /workspace/FeaturedMoE/experiments/run/CIKM_8/exp_overall
/venv/FMoE/bin/python main.py --skip-existing
```

## What To Read Next

- `artifact_anchor_catalog.csv` tells you what the old non-CIKM artifacts actually did
- `hparams.md` tells you what anchor to start from now, dataset by dataset and model by model