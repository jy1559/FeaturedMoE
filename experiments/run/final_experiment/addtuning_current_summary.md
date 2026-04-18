# Current Final Experiment Summary

Stage3 seed means are used when available. Missing stage3 cells fall back to the best stage1/2 test score.

| Dataset | BSARec | DIFSR | DuoRec | FAME | FDSA | FEARec | GRU4Rec | SASRec | TiSASRec |
|---|---|---|---|---|---|---|---|---|---|
| KuaiRecLargeStrictPosV2_0.2 | 0.306105* | 0.316649 | 0.325742 | 0.293660 | 0.307885 | 0.333297 | 0.181881 | 0.304770 | 0.327544 |
| beauty | 0.020223* | 0.017489* | 0.092094 | 0.025137 | 0.100179 | 0.011564 | 0.036771 | 0.088335 | 0.098615 |
| foursquare | 0.181700* | 0.126411* | 0.214987 | 0.170003 | 0.215994 | 0.212750 | 0.156307 | 0.223197 | 0.216056 |
| lastfm0.03 | 0.307678* | 0.329173 | 0.322000 | 0.291003 | 0.334728 | 0.337322 | 0.270780 | 0.337594 | 0.332790 |
| movielens1m | 0.097932 | 0.091256* | 0.092889* | 0.088966* | 0.100100* | 0.056714* | 0.092089* | 0.095221 | 0.103908 |
| retail_rocket | 0.421437 | 0.409422 | 0.428628* | 0.405544* | 0.412211* | 0.429225* | 0.375178* | 0.438067 | 0.425844 |

`*` means stage3 was unavailable and the cell uses the best stage1/2 test score.

## Focused Addtuning Tiers

| Dataset | Model | Current | Best | Ratio | Tier | Source |
|---|---|---:|---:|---:|---|---|
| KuaiRecLargeStrictPosV2_0.2 | BSARec | 0.306105 | 0.333297 | 0.918 | medium | stage1/S1_KUAIRECLARGESTRICTPOSV2_0_2_BSAREC |
| KuaiRecLargeStrictPosV2_0.2 | DIFSR | 0.316649 | 0.333297 | 0.950 | medium | stage3/S3_KUAIRECLARGESTRICTPOSV2_0_2_DIFSR_C1 |
| KuaiRecLargeStrictPosV2_0.2 | DuoRec | 0.325742 | 0.333297 | 0.977 | light | stage3/S3_KUAIRECLARGESTRICTPOSV2_0_2_DUOREC_C1 |
| KuaiRecLargeStrictPosV2_0.2 | FAME | 0.293660 | 0.333297 | 0.881 | heavy | stage3/S3_KUAIRECLARGESTRICTPOSV2_0_2_FAME_C1 |
| KuaiRecLargeStrictPosV2_0.2 | FDSA | 0.307885 | 0.333297 | 0.924 | medium | stage3/S3_KUAIRECLARGESTRICTPOSV2_0_2_FDSA_C1 |
| KuaiRecLargeStrictPosV2_0.2 | FEARec | 0.333297 | 0.333297 | 1.000 | light | stage3/S3_KUAIRECLARGESTRICTPOSV2_0_2_FEAREC_C2 |
| beauty | BSARec | 0.020223 | 0.100179 | 0.202 | heavy | stage2/S2_BEAUTY_BSAREC |
| beauty | DIFSR | 0.017489 | 0.100179 | 0.175 | heavy | stage2/S2_BEAUTY_DIFSR |
| beauty | DuoRec | 0.092094 | 0.100179 | 0.919 | heavy | stage3/S3_BEAUTY_DUOREC_C1 |
| beauty | FAME | 0.025137 | 0.100179 | 0.251 | heavy | stage3/S3_BEAUTY_FAME_C1 |
| beauty | FDSA | 0.100179 | 0.100179 | 1.000 | light | stage3/S3_BEAUTY_FDSA_C1 |
| beauty | FEARec | 0.011564 | 0.100179 | 0.115 | heavy | stage3/S3_BEAUTY_FEAREC_C2 |
| foursquare | BSARec | 0.181700 | 0.223197 | 0.814 | heavy | stage2/S2_FOURSQUARE_BSAREC |
| foursquare | DIFSR | 0.126411 | 0.223197 | 0.566 | heavy | stage2/S2_FOURSQUARE_DIFSR |
| foursquare | DuoRec | 0.214987 | 0.223197 | 0.963 | medium | stage3/S3_FOURSQUARE_DUOREC_C2 |
| foursquare | FAME | 0.170003 | 0.223197 | 0.762 | heavy | stage3/S3_FOURSQUARE_FAME_C1 |
| foursquare | FDSA | 0.215994 | 0.223197 | 0.968 | medium | stage3/S3_FOURSQUARE_FDSA_C1 |
| foursquare | FEARec | 0.212750 | 0.223197 | 0.953 | medium | stage3/S3_FOURSQUARE_FEAREC_C1 |
| lastfm0.03 | BSARec | 0.307678 | 0.337594 | 0.911 | medium | stage1/S1_LASTFM0_03_BSAREC |
| lastfm0.03 | DIFSR | 0.329173 | 0.337594 | 0.975 | light | stage3/S3_LASTFM0_03_DIFSR_C2 |
| lastfm0.03 | DuoRec | 0.322000 | 0.337594 | 0.954 | medium | stage3/S3_LASTFM0_03_DUOREC_C1 |
| lastfm0.03 | FAME | 0.291003 | 0.337594 | 0.862 | heavy | stage3/S3_LASTFM0_03_FAME_C2 |
| lastfm0.03 | FDSA | 0.334728 | 0.337594 | 0.992 | light | stage3/S3_LASTFM0_03_FDSA_C1 |
| lastfm0.03 | FEARec | 0.337322 | 0.337594 | 0.999 | light | stage3/S3_LASTFM0_03_FEAREC_C1 |
| movielens1m | BSARec | 0.097932 | 0.103908 | 0.942 | medium | stage3/S3_MOVIELENS1M_BSAREC_C1 |
| movielens1m | DIFSR | 0.091256 | 0.103908 | 0.878 | heavy | stage2/S2_MOVIELENS1M_DIFSR |
| movielens1m | DuoRec | 0.092889 | 0.103908 | 0.894 | heavy | stage2/S2_MOVIELENS1M_DUOREC |
| movielens1m | FAME | 0.088966 | 0.103908 | 0.856 | heavy | stage2/S2_MOVIELENS1M_FAME |
| movielens1m | FDSA | 0.100100 | 0.103908 | 0.963 | medium | stage2/S2_MOVIELENS1M_FDSA |
| movielens1m | FEARec | 0.056714 | 0.103908 | 0.546 | heavy | stage2/S2_MOVIELENS1M_FEAREC |
| retail_rocket | BSARec | 0.421437 | 0.438067 | 0.962 | medium | stage3/S3_RETAIL_ROCKET_BSAREC_C1 |
| retail_rocket | DIFSR | 0.409422 | 0.438067 | 0.935 | medium | stage3/S3_RETAIL_ROCKET_DIFSR_C1 |
| retail_rocket | DuoRec | 0.428628 | 0.438067 | 0.978 | medium | stage2/S2_RETAIL_ROCKET_DUOREC |
| retail_rocket | FAME | 0.405544 | 0.438067 | 0.926 | medium | stage2/S2_RETAIL_ROCKET_FAME |
| retail_rocket | FDSA | 0.412211 | 0.438067 | 0.941 | medium | stage2/S2_RETAIL_ROCKET_FDSA |
| retail_rocket | FEARec | 0.429225 | 0.438067 | 0.980 | medium | stage2/S2_RETAIL_ROCKET_FEAREC |

## Stage And Tier Policy

Epoch/patience increases by stage and final confirmation uses 100/10. Max-evals are front-loaded in stage1 and reduced in later stages.
Stage1 now spends real budget on a few targeted local sweeps around the incumbent. Stage3 is a cheap 1-seed reranking gate over only the top 1--2 configs, and multi-seed confirmation is deferred to the final confirmation stage. Stage4 remains gap-driven, but now gets a meaningfully larger refinement budget when a model is still behind the dataset frontier.
Candidate counts below are per-axis option counts for local search refinement. They are not cartesian combo jobs; actual trials are capped by `max_evals` and the per-job time limit.

| Stage | Tier | Jobs | Epochs | Patience | Max Evals | LR Points | Other Points | Random Adds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 | light | 24 | 40 | 5 | 4-10 | 3-5 | 2-3 | 2 |
| stage1 | medium | 64 | 40 | 5 | 4-9 | 3-5 | 2-3 | 3 |
| stage1 | heavy | 56 | 40 | 5 | 5-18 | 4-7 | 2-5 | 4 |
| stage2 | light | 6 | 70 | 7 | 6-15 | 3-4 | 2-3 | 0 |
| stage2 | medium | 16 | 70 | 7 | 6-12 | 3-4 | 2-3 | 0 |
| stage2 | heavy | 14 | 70 | 7 | 7-21 | 3-5 | 2-4 | 0 |
| stage3 | light | 6 | 100 | 10 | 1 | - | - | 0 |
| stage3 | medium | 32 | 100 | 10 | 1 | - | - | 0 |
| stage3 | heavy | 28 | 100 | 10 | 1 | - | - | 0 |
| stage4 | light | 4 | 85 | 8 | 3-5 | 5 | 4 | 2 |
| stage4 | medium | 4 | 85 | 8 | 6-8 | 6 | 5 | 2 |
| stage4 | heavy | 15 | 85 | 8 | 5-10 | 7 | 6 | 3 |
| stage5 | light | 6 | 100 | 10 | 1 | - | - | 0 |
| stage5 | medium | 8 | 100 | 10 | 1 | - | - | 0 |
| stage5 | heavy | 24 | 100 | 10 | 1 | - | - | 0 |

## Estimated Runtime On 8 GPUs

Wall time is estimated as `max(total GPU-hours / 8, longest single job)` per stage, using historical per-epoch runtimes from existing stage1/2/3 runs of the same dataset/model when available.
All addtuning jobs inherit the existing safeguards: OOM retries halve train/eval batch size, the time budget stops launching new trials after the current trial finishes and runs final evaluation, and stage workers pull from one global GPU queue rather than dataset-partitioned queues.

| Stage | Jobs | Light | Medium | Heavy | GPU-hours | Est. Wall-hours | Longest Job |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 144 | 24 | 64 | 56 | 73.34 | 9.17 | 2.00 |
| stage2 | 36 | 6 | 16 | 14 | 32.39 | 4.05 | 2.00 |
| stage3 | 66 | 6 | 32 | 28 | 16.65 | 2.08 | 1.20 |
| stage4 | 23 | 4 | 4 | 15 | 18.82 | 2.35 | 2.00 |
| stage5 | 38 | 6 | 8 | 24 | 7.13 | 0.89 | 0.85 |

Total estimated GPU-hours: 148.32
Total estimated wall-hours on 8 GPUs: 18.54
