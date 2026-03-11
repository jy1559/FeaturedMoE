# FeaturedMoE_Individual R1INDv1 Summary

- Updated: `2026-03-10 09:20:29Z`
- Dataset: `movielens1m`
- Phase bucket: `R1INDv1`
- Completed result files: `4`
- Focus: feature-individual outer top-k=4 with dense inner router
- Current best: `0.0624` with `- / - / k- / 128/16/160/64`

## By Mode
| mode | runs | best | avg |
| --- | --- | --- | --- |
| - | 4 | 0.0624 | 0.0614 |

## By Expert Top-K
| k | runs | best | avg |
| --- | --- | --- | --- |
| k- | 4 | 0.0624 | 0.0614 |

## By Anchor
| anchor | runs | best | avg |
| --- | --- | --- | --- |
| - | 4 | 0.0624 | 0.0614 |

## Combo Table
| rank | combo | anchor | mode | k | dims | best | lr | wd | epoch | stop | setup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C02 | - | - | k- | 128/16/160/64 | 0.0624 | 2.17e-04 | - | 35 | Y | L2 serial - |
| 2 | C03 | - | - | k- | 128/16/160/64 | 0.0615 | 1.55e-03 | - | 19 | Y | L3 serial - |
| 3 | C01 | - | - | k- | 128/16/160/64 | 0.0614 | 5.59e-04 | - | 22 | Y | L1 serial - |
| 4 | C00 | - | - | k- | 128/16/160/64 | 0.0605 | 4.93e-04 | - | 24 | Y | L0 serial - |

