# HGRv3 R1DHGRv3 Summary

- Updated: `2026-03-10 06:54:50Z`
- Dataset: `movielens1m`
- Phase bucket: `R1DHGRv3`
- Completed result files: `2`
- Focus: hidden-only outer, weak inner distill probe, compare expert_top_k under fixed layout 15
- Current best: `0.0919` with `A0 / weak / k1 / 128/16/160/64`

## By Mode
| mode | runs | best | avg |
| --- | --- | --- | --- |
| weak | 2 | 0.0919 | 0.0919 |

## By Expert Top-K
| k | runs | best | avg |
| --- | --- | --- | --- |
| k1 | 2 | 0.0919 | 0.0919 |

## By Anchor
| anchor | runs | best | avg |
| --- | --- | --- | --- |
| A0 | 1 | 0.0919 | 0.0919 |
| A1 | 1 | 0.0919 | 0.0919 |

## Combo Table
| rank | combo | anchor | mode | k | dims | best | lr | wd | epoch | stop | setup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C00 | A0 | weak | k1 | 128/16/160/64 | 0.0919 | 2.55e-03 | 1.00e-06 | 25 | N | L15 serial stage_wide |
| 2 | C04 | A1 | weak | k1 | 160/16/256/112 | 0.0919 | 5.72e-04 | 1.00e-06 | 25 | N | L15 serial stage_wide |

