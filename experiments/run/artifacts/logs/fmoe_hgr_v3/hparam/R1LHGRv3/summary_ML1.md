# HGRv3 R1LHGRv3 Summary

- Updated: `2026-03-10 06:54:50Z`
- Dataset: `movielens1m`
- Phase bucket: `R1LHGRv3`
- Completed result files: `2`
- Focus: hidden-only outer, weak inner distill probe, compare expert_top_k under fixed layout 15
- Current best: `0.0931` with `- / off / k1 / 160/16/256/112`

## By Mode
| mode | runs | best | avg |
| --- | --- | --- | --- |
| off | 2 | 0.0931 | 0.0920 |

## By Expert Top-K
| k | runs | best | avg |
| --- | --- | --- | --- |
| k1 | 2 | 0.0931 | 0.0920 |

## By Anchor
| anchor | runs | best | avg |
| --- | --- | --- | --- |
| - | 2 | 0.0931 | 0.0920 |

## Combo Table
| rank | combo | anchor | mode | k | dims | best | lr | wd | epoch | stop | setup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C04 | - | off | k1 | 160/16/256/112 | 0.0931 | 5.49e-04 | 1.00e-06 | 25 | N | L15 serial stage_wide |
| 2 | C00 | - | off | k1 | 128/16/160/64 | 0.0909 | 8.68e-04 | 1.00e-06 | 25 | N | L15 serial stage_wide |

