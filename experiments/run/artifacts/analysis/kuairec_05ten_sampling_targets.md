# KuaiRec_0.5Ten Sampling Targets

Reference full dataset:

- `KuaiRec_0.5Ten`: `1,943,613` rows, `192,405` sessions, `1,411` users, `3,312` items
- `movielens1m`: `575,281` rows, `14,539` sessions, `6,038` users, `3,533` items

User-stratified sampling on `KuaiRec_0.5Ten`:

| ratio | rows | sessions | users | items |
| --- | ---: | ---: | ---: | ---: |
| 0.10 | 193,514 | 19,257 | 140 | 3,206 |
| 0.15 | 290,847 | 28,779 | 211 | 3,249 |
| 0.20 | 389,095 | 38,244 | 282 | 3,260 |
| 0.25 | 485,159 | 47,941 | 352 | 3,258 |
| 0.30 | 579,797 | 57,459 | 423 | 3,279 |

Practical guidance:

- Fast smoke: `0.15`
- Default fast dev: `0.20`
- Safer dev/main: `0.25`
- ML1M-like total-row confirm: `0.30`
