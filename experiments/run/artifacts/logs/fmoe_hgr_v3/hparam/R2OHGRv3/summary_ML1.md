# HGRv3 R2OHGRv3 Summary

- Updated: `2026-03-10 08:32:27Z`
- Dataset: `movielens1m`
- Phase bucket: `R2OHGRv3`
- Completed result files: `4`
- Focus: hidden-only outer, weak inner distill probe, compare expert_top_k under fixed layout 15
- Current best: `0.0933` with `A1 / feat_off / k1 / 160/16/256/112`

## By Mode
| mode | runs | best | avg |
| --- | --- | --- | --- |
| feat_off | 1 | 0.0933 | 0.0933 |
| feat_weak | 1 | 0.0933 | 0.0933 |
| honly_strong | 1 | 0.0926 | 0.0926 |
| feat_strong | 1 | 0.0925 | 0.0925 |

## By Expert Top-K
| k | runs | best | avg |
| --- | --- | --- | --- |
| k1 | 4 | 0.0933 | 0.0929 |

## By Anchor
| anchor | runs | best | avg |
| --- | --- | --- | --- |
| A1 | 4 | 0.0933 | 0.0929 |

## Combo Table
| rank | combo | anchor | mode | k | dims | best | lr | wd | epoch | stop | setup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C00 | A1 | feat_off | k1 | 160/16/256/112 | 0.0933 | 3.50e-04 | - | 30 | N | L15 serial stage_wide |
| 2 | C01 | A1 | feat_weak | k1 | 160/16/256/112 | 0.0933 | 4.00e-04 | - | 30 | Y | L15 serial stage_wide |
| 3 | C03 | A1 | honly_strong | k1 | 160/16/256/112 | 0.0926 | 5.00e-04 | - | 30 | N | L15 serial stage_wide |
| 4 | C02 | A1 | feat_strong | k1 | 160/16/256/112 | 0.0925 | 1.80e-03 | - | 30 | N | L15 serial stage_wide |

