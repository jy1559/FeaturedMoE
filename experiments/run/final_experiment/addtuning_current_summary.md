# Current Final Experiment Summary

Stage3 seed means are used when available. Otherwise the table uses the best test score found across existing stage1/2 runs, including the current final_experiment_addtuning track.

| Dataset | BSARec | DIFSR | DuoRec | FAME | FDSA | FEARec | GRU4Rec | SASRec | TiSASRec |
|---|---|---|---|---|---|---|---|---|---|
| KuaiRecLargeStrictPosV2_0.2 | 0.334478 | 0.316649 | 0.328149 | 0.333094 | 0.321813 | 0.333297 | 0.181881 | 0.304770 | 0.327544 |
| beauty | 0.012378 | 0.016783 | 0.092094 | 0.025137 | 0.100179 | 0.011564 | 0.036771 | 0.088335 | 0.098615 |
| foursquare | 0.187551 | 0.190444 | 0.221636 | 0.170003 | 0.215994 | 0.212750 | 0.156307 | 0.223197 | 0.216056 |
| lastfm0.03 | 0.308223 | 0.329173 | 0.333256 | 0.301096 | 0.337511 | 0.337322 | 0.270780 | 0.337594 | 0.332790 |
| movielens1m | 0.099389 | 0.100382 | 0.092339 | 0.106167 | 0.102622 | 0.067133 | 0.092089* | 0.095221 | 0.103908 |
| retail_rocket | 0.421437 | 0.421044 | 0.437900 | 0.408567 | 0.434767 | 0.442422 | 0.375178* | 0.438067 | 0.425844 |

`*` means stage3 was unavailable and the cell uses the best stage1/2 test score.

## Focused Addtuning Tiers

| Dataset | Model | Current | Best | Ratio | Tier | Source |
|---|---|---:|---:|---:|---|---|
| KuaiRecLargeStrictPosV2_0.2 | BSARec | 0.334478 | 0.334478 | 1.000 | light | stage3/ADD_STAGE3_KUAIRECLARGESTRICTPOSV2_0_2_BSAREC_C2 |
| KuaiRecLargeStrictPosV2_0.2 | DIFSR | 0.316649 | 0.334478 | 0.947 | medium | stage3/S3_KUAIRECLARGESTRICTPOSV2_0_2_DIFSR_C1 |
| KuaiRecLargeStrictPosV2_0.2 | DuoRec | 0.328149 | 0.334478 | 0.981 | light | stage3/ADD_STAGE3_KUAIRECLARGESTRICTPOSV2_0_2_DUOREC_C1 |
| KuaiRecLargeStrictPosV2_0.2 | FAME | 0.333094 | 0.334478 | 0.996 | light | stage3/ADD_STAGE3_KUAIRECLARGESTRICTPOSV2_0_2_FAME_C1 |
| KuaiRecLargeStrictPosV2_0.2 | FDSA | 0.321813 | 0.334478 | 0.962 | medium | stage3/ADD_STAGE3_KUAIRECLARGESTRICTPOSV2_0_2_FDSA_C1 |
| KuaiRecLargeStrictPosV2_0.2 | FEARec | 0.333297 | 0.334478 | 0.996 | light | stage3/S3_KUAIRECLARGESTRICTPOSV2_0_2_FEAREC_C2 |
| beauty | BSARec | 0.012378 | 0.100179 | 0.124 | heavy | stage3/ADD_STAGE3_BEAUTY_BSAREC_C2 |
| beauty | DIFSR | 0.016783 | 0.100179 | 0.168 | heavy | stage3/ADD_STAGE3_BEAUTY_DIFSR_C1 |
| beauty | DuoRec | 0.092094 | 0.100179 | 0.919 | heavy | stage3/S3_BEAUTY_DUOREC_C1 |
| beauty | FAME | 0.025137 | 0.100179 | 0.251 | heavy | stage3/S3_BEAUTY_FAME_C1 |
| beauty | FDSA | 0.100179 | 0.100179 | 1.000 | light | stage3/S3_BEAUTY_FDSA_C1 |
| beauty | FEARec | 0.011564 | 0.100179 | 0.115 | heavy | stage3/S3_BEAUTY_FEAREC_C2 |
| foursquare | BSARec | 0.187551 | 0.223197 | 0.840 | heavy | stage3/ADD_STAGE3_FOURSQUARE_BSAREC_C1 |
| foursquare | DIFSR | 0.190444 | 0.223197 | 0.853 | heavy | stage3/ADD_STAGE3_FOURSQUARE_DIFSR_C2 |
| foursquare | DuoRec | 0.221636 | 0.223197 | 0.993 | light | stage3/ADD_STAGE3_FOURSQUARE_DUOREC_C1 |
| foursquare | FAME | 0.170003 | 0.223197 | 0.762 | heavy | stage3/S3_FOURSQUARE_FAME_C1 |
| foursquare | FDSA | 0.215994 | 0.223197 | 0.968 | medium | stage3/S3_FOURSQUARE_FDSA_C1 |
| foursquare | FEARec | 0.212750 | 0.223197 | 0.953 | medium | stage3/S3_FOURSQUARE_FEAREC_C1 |
| lastfm0.03 | BSARec | 0.308223 | 0.337594 | 0.913 | heavy | stage3/ADD_STAGE3_LASTFM0_03_BSAREC_C1 |
| lastfm0.03 | DIFSR | 0.329173 | 0.337594 | 0.975 | light | stage3/S3_LASTFM0_03_DIFSR_C2 |
| lastfm0.03 | DuoRec | 0.333256 | 0.337594 | 0.987 | light | stage3/ADD_STAGE3_LASTFM0_03_DUOREC_C1 |
| lastfm0.03 | FAME | 0.301096 | 0.337594 | 0.892 | heavy | stage3/ADD_STAGE3_LASTFM0_03_FAME_C2 |
| lastfm0.03 | FDSA | 0.337511 | 0.337594 | 1.000 | light | stage3/ADD_STAGE3_LASTFM0_03_FDSA_C1 |
| lastfm0.03 | FEARec | 0.337322 | 0.337594 | 0.999 | light | stage3/S3_LASTFM0_03_FEAREC_C1 |
| movielens1m | BSARec | 0.099389 | 0.106167 | 0.936 | medium | stage3/ADD_STAGE3_MOVIELENS1M_BSAREC_C2 |
| movielens1m | DIFSR | 0.100382 | 0.106167 | 0.946 | medium | stage3/ADD_STAGE3_MOVIELENS1M_DIFSR_C2 |
| movielens1m | DuoRec | 0.092339 | 0.106167 | 0.870 | heavy | stage3/ADD_STAGE3_MOVIELENS1M_DUOREC_C1 |
| movielens1m | FAME | 0.106167 | 0.106167 | 1.000 | medium | stage3/ADD_STAGE3_MOVIELENS1M_FAME_C1 |
| movielens1m | FDSA | 0.102622 | 0.106167 | 0.967 | medium | stage3/ADD_STAGE3_MOVIELENS1M_FDSA_C2 |
| movielens1m | FEARec | 0.067133 | 0.106167 | 0.632 | heavy | stage3/ADD_STAGE3_MOVIELENS1M_FEAREC_C1 |
| retail_rocket | BSARec | 0.421437 | 0.442422 | 0.953 | medium | stage3/S3_RETAIL_ROCKET_BSAREC_C1 |
| retail_rocket | DIFSR | 0.421044 | 0.442422 | 0.952 | medium | stage3/ADD_STAGE3_RETAIL_ROCKET_DIFSR_C2 |
| retail_rocket | DuoRec | 0.437900 | 0.442422 | 0.990 | medium | stage3/ADD_STAGE3_RETAIL_ROCKET_DUOREC_C2 |
| retail_rocket | FAME | 0.408567 | 0.442422 | 0.923 | medium | stage3/ADD_STAGE3_RETAIL_ROCKET_FAME_C2 |
| retail_rocket | FDSA | 0.434767 | 0.442422 | 0.983 | medium | stage3/ADD_STAGE3_RETAIL_ROCKET_FDSA_C2 |
| retail_rocket | FEARec | 0.442422 | 0.442422 | 1.000 | medium | stage3/ADD_STAGE3_RETAIL_ROCKET_FEAREC_C1 |

## Stage And Tier Policy

Epoch/patience increases by stage and final confirmation uses 100/10. Max-evals are front-loaded in stage1 and reduced in later stages.
Stage1 now spends real budget on a few targeted local sweeps around the incumbent. Stage3 is a cheap 1-seed reranking gate over only the top 1--2 configs, and multi-seed confirmation is deferred to the final confirmation stage. Stage4 remains gap-driven, but now gets a meaningfully larger refinement budget when a model is still behind the dataset frontier.
Candidate counts below are per-axis option counts for local search refinement. They are not cartesian combo jobs; actual trials are capped by `max_evals` and the per-job time limit.

| Stage | Tier | Jobs | Epochs | Patience | Max Evals | LR Points | Other Points | Random Adds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 | light | 40 | 40 | 5 | 4-10 | 3-5 | 2-3 | 2 |
| stage1 | medium | 56 | 40 | 5 | 4-9 | 3-5 | 2-3 | 3 |
| stage1 | heavy | 48 | 40 | 5 | 5-18 | 4-7 | 2-5 | 4 |
| stage2 | light | 10 | 70 | 7 | 6-15 | 3-4 | 2-3 | 0 |
| stage2 | medium | 14 | 70 | 7 | 6-12 | 3-4 | 2-3 | 0 |
| stage2 | heavy | 12 | 70 | 7 | 7-21 | 3-5 | 2-4 | 0 |
| stage3 | light | 10 | 100 | 10 | 1 | - | - | 0 |
| stage3 | medium | 28 | 100 | 10 | 1 | - | - | 0 |
| stage3 | heavy | 24 | 100 | 10 | 1 | - | - | 0 |
| stage4 | light | 4 | 85 | 8 | 3-5 | 5 | 4 | 2 |
| stage4 | medium | 9 | 85 | 8 | 4-8 | 6 | 5 | 2 |
| stage4 | heavy | 7 | 85 | 8 | 5-10 | 7 | 6 | 3 |
| stage5 | light | 7 | 100 | 10 | 1 | - | - | 0 |
| stage5 | medium | 16 | 100 | 10 | 1 | - | - | 0 |
| stage5 | heavy | 11 | 100 | 10 | 1 | - | - | 0 |

## Estimated Runtime On 8 GPUs

Wall time is estimated as `max(total GPU-hours / 8, longest single job)` per stage, using historical per-epoch runtimes from existing stage1/2/3 runs of the same dataset/model when available.
All addtuning jobs inherit the existing safeguards: OOM retries halve train/eval batch size, the time budget stops launching new trials after the current trial finishes and runs final evaluation, and stage workers pull from one global GPU queue rather than dataset-partitioned queues.

| Stage | Jobs | Light | Medium | Heavy | GPU-hours | Est. Wall-hours | Longest Job |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 144 | 40 | 56 | 48 | 70.91 | 8.86 | 2.00 |
| stage2 | 36 | 10 | 14 | 12 | 31.90 | 3.99 | 2.00 |
| stage3 | 62 | 10 | 28 | 24 | 15.67 | 1.96 | 1.20 |
| stage4 | 20 | 4 | 9 | 7 | 11.15 | 2.00 | 2.00 |
| stage5 | 34 | 7 | 16 | 11 | 3.87 | 0.62 | 0.62 |

Total estimated GPU-hours: 133.51
Total estimated wall-hours on 8 GPUs: 17.43
