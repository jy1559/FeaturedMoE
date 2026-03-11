# HGRv4 R0HGRv4 Summary

- Updated: `2026-03-10 16:31:45Z`
- Dataset: `movielens1m`
- Phase bucket: `R0HGRv4`
- Completed result files: `8`
- Focus: feature-aware outer restored, group-stat inner teacher, 4-level distill comparison
- Current best: `0.0956` with `- / distill / k1 / 128/16/160/64`

## By Mode
| mode | runs | best | avg |
| --- | --- | --- | --- |
| distill | 6 | 0.0956 | 0.0947 |
| off | 2 | 0.0942 | 0.0940 |

## By Expert Top-K
| k | runs | best | avg |
| --- | --- | --- | --- |
| k1 | 8 | 0.0956 | 0.0945 |

## By Anchor
| anchor | runs | best | avg |
| --- | --- | --- | --- |
| - | 8 | 0.0956 | 0.0945 |

## Combo Table
| rank | combo | anchor | mode | k | dims | best | lr | wd | epoch | stop | setup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C01 | - | distill | k1 | 128/16/160/64 | 0.0956 | 1.10e-03 | - | 34 | Y | L15 serial hybrid |
| 2 | C03 | - | distill | k1 | 128/16/160/64 | 0.0949 | 2.80e-03 | - | 38 | Y | L15 serial hybrid |
| 3 | C01 | - | distill | k1 | 128/16/160/64 | 0.0946 | 5.00e-04 | - | 40 | Y | L15 serial hybrid |
| 4 | C02 | - | distill | k1 | 128/16/160/64 | 0.0945 | 1.20e-03 | - | 28 | Y | L15 serial hybrid |
| 5 | C00 | - | off | k1 | 128/16/160/64 | 0.0942 | 7.00e-04 | - | 38 | Y | L15 serial hybrid |
| 6 | C03 | - | distill | k1 | 128/16/160/64 | 0.0942 | 1.70e-03 | - | 28 | Y | L15 serial hybrid |
| 7 | C02 | - | distill | k1 | 128/16/160/64 | 0.0941 | 2.20e-03 | - | 34 | Y | L15 serial hybrid |
| 8 | C00 | - | off | k1 | 128/16/160/64 | 0.0939 | 7.00e-04 | - | 40 | N | L15 serial hybrid |

