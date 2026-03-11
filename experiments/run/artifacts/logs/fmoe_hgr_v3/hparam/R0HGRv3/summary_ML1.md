# HGRv3 R0HGRv3 Summary

- Updated: `2026-03-10 05:13:08Z`
- Dataset: `movielens1m`
- Phase bucket: `R0HGRv3`
- Completed result files: `10`
- Focus: hidden-only outer, weak inner distill probe, compare expert_top_k under fixed layout 15
- Current best: `0.0933` with `A1 / off / k1 / 160/16/256/112`

## By Mode
| mode | runs | best | avg |
| --- | --- | --- | --- |
| off | 4 | 0.0933 | 0.0928 |
| distill | 6 | 0.0922 | 0.0916 |

## By Expert Top-K
| k | runs | best | avg |
| --- | --- | --- | --- |
| k1 | 4 | 0.0933 | 0.0928 |
| k2 | 6 | 0.0922 | 0.0916 |

## By Anchor
| anchor | runs | best | avg |
| --- | --- | --- | --- |
| A1 | 5 | 0.0933 | 0.0921 |
| A0 | 5 | 0.0930 | 0.0921 |

## Combo Table
| rank | combo | anchor | mode | k | dims | best | lr | wd | epoch | stop | setup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C04 | A1 | off | k1 | 160/16/256/112 | 0.0933 | 6.32e-04 | 6.47e-05 | 40 | N | L15 serial stage_wide |
| 2 | C00 | A0 | off | k1 | 128/16/160/64 | 0.0930 | 1.29e-03 | 1.52e-05 | 40 | N | L15 serial stage_wide |
| 3 | C00 | A0 | off | k1 | 128/16/160/64 | 0.0928 | 2.25e-03 | 1.76e-06 | 40 | N | L15 serial stage_wide |
| 4 | C06 | A1 | distill | k2 | 160/16/256/112 | 0.0922 | 4.27e-04 | 0.00e+00 | 27 | Y | L15 serial stage_wide |
| 5 | C04 | A1 | off | k1 | 160/16/256/112 | 0.0920 | 4.02e-04 | 0.00e+00 | 36 | Y | L15 serial stage_wide |
| 6 | C02 | A0 | distill | k2 | 128/16/160/64 | 0.0919 | 1.55e-03 | 3.95e-05 | 27 | Y | L15 serial stage_wide |
| 7 | C02 | A0 | distill | k2 | 128/16/160/64 | 0.0916 | 2.43e-03 | 1.00e-04 | 32 | Y | L15 serial stage_wide |
| 8 | C06 | A1 | distill | k2 | 160/16/256/112 | 0.0915 | 3.92e-04 | 1.00e-06 | 28 | Y | L15 serial stage_wide |
| 9 | C02 | A0 | distill | k2 | 128/16/160/64 | 0.0914 | 1.99e-03 | 8.33e-05 | 40 | N | L15 serial stage_wide |
| 10 | C06 | A1 | distill | k2 | 160/16/256/112 | 0.0913 | 3.92e-04 | 0.00e+00 | 27 | Y | L15 serial stage_wide |

