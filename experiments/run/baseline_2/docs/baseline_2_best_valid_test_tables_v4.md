# baseline_2 best-valid/test tables refresh

Selection rule: within usable experiment results, rank candidates first by `test_special_metrics.overall_seen_target.mrr@20`, then break ties by higher `best_valid_result.mrr@20` and higher overall test MRR@20. This keeps selection centered on seen-test quality while still checking validation quality.

Source scope: conventional baselines are scanned from `artifacts/results/baseline_2`. RouteRec is selected from `artifacts/results/fmoe_n4` result JSONs, using the linked special-metrics JSON when needed, and falls back to `baseline_2` only if `fmoe_n4` has no usable candidate for that dataset.

Formatting rule: 1st place is bold, 2nd place has `*`, and the last place or any value at or below 75% of the best value is underlined.

## beauty

### Seen

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | **0.1101** | <u>0.0602</u> | 0.0917 | *0.1060 | 0.0946 | <u>0.0716</u> | <u>0.0372</u> | 0.0974 | *0.1060 | 0.1054 |
| hit@10 | 0.1288 | <u>0.0946</u> | 0.1318 | *0.1461 | 0.1318 | <u>0.0774</u> | <u>0.0458</u> | <u>0.1203</u> | 0.1347 | **0.1616** |
| hit@20 | 0.1616 | <u>0.1232</u> | *0.1977 | 0.1748 | 0.1920 | <u>0.1032</u> | <u>0.0602</u> | <u>0.1461</u> | 0.1633 | **0.2014** |
| ndcg@5 | 0.0786 | <u>0.0412</u> | 0.0785 | 0.0732 | 0.0684 | <u>0.0550</u> | <u>0.0312</u> | 0.0771 | **0.0852** | *0.0803 |
| ndcg@10 | 0.0851 | <u>0.0520</u> | 0.0912 | 0.0858 | 0.0802 | <u>0.0567</u> | <u>0.0339</u> | 0.0844 | *0.0943 | **0.0982** |
| ndcg@20 | 0.0933 | <u>0.0589</u> | *0.1081 | 0.0930 | 0.0955 | <u>0.0634</u> | <u>0.0375</u> | 0.0911 | 0.1017 | **0.1083** |
| mrr@5 | 0.0684 | <u>0.0350</u> | *0.0741 | 0.0622 | 0.0596 | <u>0.0494</u> | <u>0.0292</u> | 0.0703 | **0.0784** | 0.0721 |
| mrr@10 | 0.0712 | <u>0.0393</u> | 0.0792 | 0.0672 | 0.0643 | <u>0.0500</u> | <u>0.0303</u> | 0.0733 | **0.0820** | *0.0794 |
| mrr@20 | 0.0735 | <u>0.0411</u> | *0.0839 | 0.0692 | 0.0686 | <u>0.0519</u> | <u>0.0313</u> | 0.0752 | **0.0841** | 0.0822 |

<details>
<summary>Overall</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | **0.0738** | <u>0.0352</u> | <u>0.0536</u> | 0.0620 | <u>0.0553</u> | <u>0.0419</u> | <u>0.0218</u> | 0.0570 | 0.0620 | *0.0706 |
| hit@10 | *0.0863 | <u>0.0553</u> | <u>0.0771</u> | 0.0854 | <u>0.0771</u> | <u>0.0452</u> | <u>0.0268</u> | <u>0.0704</u> | <u>0.0787</u> | **0.1083** |
| hit@20 | 0.1083 | <u>0.0720</u> | *0.1156 | 0.1022 | 0.1122 | <u>0.0603</u> | <u>0.0352</u> | <u>0.0854</u> | <u>0.0955</u> | **0.1350** |
| ndcg@5 | *0.0527 | <u>0.0241</u> | 0.0459 | 0.0428 | <u>0.0400</u> | <u>0.0321</u> | <u>0.0182</u> | 0.0450 | 0.0498 | **0.0538** |
| ndcg@10 | *0.0570 | <u>0.0304</u> | 0.0533 | 0.0502 | <u>0.0469</u> | <u>0.0331</u> | <u>0.0198</u> | <u>0.0493</u> | 0.0551 | **0.0659** |
| ndcg@20 | 0.0625 | <u>0.0345</u> | *0.0632 | <u>0.0544</u> | 0.0558 | <u>0.0371</u> | <u>0.0219</u> | <u>0.0532</u> | 0.0595 | **0.0726** |
| mrr@5 | *0.0458 | <u>0.0205</u> | 0.0433 | 0.0364 | <u>0.0349</u> | <u>0.0289</u> | <u>0.0171</u> | 0.0411 | 0.0458 | **0.0483** |
| mrr@10 | 0.0477 | <u>0.0230</u> | 0.0463 | <u>0.0393</u> | <u>0.0376</u> | <u>0.0292</u> | <u>0.0177</u> | 0.0429 | *0.0479 | **0.0532** |
| mrr@20 | *0.0492 | <u>0.0240</u> | 0.0491 | <u>0.0404</u> | <u>0.0401</u> | <u>0.0304</u> | <u>0.0183</u> | 0.0440 | 0.0492 | **0.0551** |

</details>

<details>
<summary>Unseen</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hit@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hit@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

</details>

## retail_rocket

### Seen

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | *0.4842 | 0.4475 | 0.4700 | 0.4647 | 0.4611 | 0.4449 | 0.4371 | <u>0.4305</u> | 0.4344 | **0.4883** |
| hit@10 | *0.5605 | 0.5211 | 0.5435 | 0.5441 | 0.5377 | 0.4658 | 0.4507 | <u>0.4365</u> | 0.5133 | **0.5612** |
| hit@20 | **0.6325** | 0.5863 | 0.6135 | 0.6200 | 0.6134 | 0.4847 | <u>0.4629</u> | <u>0.4409</u> | 0.5886 | *0.6313 |
| ndcg@5 | 0.3867 | 0.3620 | 0.3724 | 0.3678 | 0.3616 | **0.3926** | 0.3892 | 0.3881 | <u>0.3396</u> | *0.3914 |
| ndcg@10 | *0.4116 | 0.3858 | 0.3963 | 0.3935 | 0.3864 | 0.3994 | 0.3935 | 0.3900 | <u>0.3651</u> | **0.4150** |
| ndcg@20 | *0.4298 | 0.4023 | 0.4139 | 0.4127 | 0.4057 | 0.4042 | 0.3967 | 0.3911 | <u>0.3842</u> | **0.4327** |
| mrr@5 | 0.3542 | 0.3335 | 0.3399 | 0.3355 | 0.3284 | **0.3748** | 0.3726 | *0.3733 | <u>0.3082</u> | 0.3591 |
| mrr@10 | 0.3645 | 0.3434 | 0.3498 | 0.3462 | 0.3388 | **0.3776** | *0.3745 | 0.3741 | <u>0.3187</u> | 0.3688 |
| mrr@20 | 0.3696 | 0.3479 | 0.3547 | 0.3515 | 0.3441 | **0.3789** | *0.3753 | 0.3744 | <u>0.3239</u> | 0.3737 |

<details>
<summary>Overall</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | *0.4098 | 0.3816 | 0.3969 | 0.3915 | 0.3884 | 0.3823 | 0.3755 | 0.3704 | <u>0.3659</u> | **0.4135** |
| hit@10 | *0.4754 | 0.4445 | 0.4596 | 0.4583 | 0.4529 | 0.4000 | 0.3870 | <u>0.3755</u> | 0.4324 | **0.4763** |
| hit@20 | **0.5371** | 0.5001 | 0.5198 | 0.5223 | 0.5167 | 0.4161 | <u>0.3974</u> | <u>0.3794</u> | 0.4958 | *0.5363 |
| ndcg@5 | 0.3267 | 0.3081 | 0.3141 | 0.3098 | 0.3046 | **0.3376** | *0.3346 | 0.3341 | <u>0.2861</u> | 0.3307 |
| ndcg@10 | *0.3480 | 0.3285 | 0.3345 | 0.3315 | 0.3255 | 0.3433 | 0.3383 | 0.3357 | <u>0.3075</u> | **0.3510** |
| ndcg@20 | *0.3636 | 0.3426 | 0.3497 | 0.3477 | 0.3417 | 0.3474 | 0.3410 | 0.3367 | <u>0.3236</u> | **0.3662** |
| mrr@5 | 0.2989 | 0.2837 | 0.2866 | 0.2826 | 0.2767 | **0.3223** | 0.3205 | *0.3214 | <u>0.2596</u> | 0.3031 |
| mrr@10 | 0.3078 | 0.2921 | 0.2951 | 0.2916 | 0.2853 | **0.3247** | 0.3220 | *0.3221 | <u>0.2684</u> | 0.3115 |
| mrr@20 | 0.3121 | 0.2960 | 0.2992 | 0.2961 | 0.2898 | **0.3259** | *0.3228 | 0.3223 | <u>0.2729</u> | 0.3157 |

</details>

<details>
<summary>Unseen</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | <u>0.0124</u> | <u>0.0296</u> | <u>0.0060</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0475 | 0.0466 | **0.0499** | <u>0.0000</u> | <u>0.0136</u> |
| hit@10 | <u>0.0203</u> | <u>0.0351</u> | <u>0.0115</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0484 | 0.0472 | **0.0502** | <u>0.0000</u> | <u>0.0227</u> |
| hit@20 | <u>0.0275</u> | 0.0396 | <u>0.0193</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0496 | 0.0478 | **0.0508** | <u>0.0000</u> | <u>0.0290</u> |
| ndcg@5 | <u>0.0059</u> | <u>0.0206</u> | <u>0.0028</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0435 | 0.0431 | **0.0455** | <u>0.0000</u> | <u>0.0064</u> |
| ndcg@10 | <u>0.0084</u> | <u>0.0224</u> | <u>0.0046</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0438 | 0.0433 | **0.0456** | <u>0.0000</u> | <u>0.0094</u> |
| ndcg@20 | <u>0.0102</u> | <u>0.0235</u> | <u>0.0066</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0441 | 0.0434 | **0.0457** | <u>0.0000</u> | <u>0.0109</u> |
| mrr@5 | <u>0.0038</u> | <u>0.0175</u> | <u>0.0018</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0421 | 0.0419 | **0.0439** | <u>0.0000</u> | <u>0.0041</u> |
| mrr@10 | <u>0.0048</u> | <u>0.0183</u> | <u>0.0025</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0422 | 0.0420 | **0.0440** | <u>0.0000</u> | <u>0.0053</u> |
| mrr@20 | <u>0.0053</u> | <u>0.0186</u> | <u>0.0030</u> | <u>0.0000</u> | <u>0.0000</u> | *0.0423 | 0.0420 | **0.0440** | <u>0.0000</u> | <u>0.0057</u> |

</details>

## foursquare

### Seen

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.2446 | <u>0.1771</u> | *0.2558 | 0.2398 | 0.2378 | 0.1996 | <u>0.1731</u> | 0.2152 | 0.2478 | **0.2562** |
| hit@10 | *0.3197 | <u>0.2218</u> | 0.3106 | 0.3070 | 0.3122 | <u>0.2341</u> | <u>0.2046</u> | 0.2575 | 0.2998 | **0.3225** |
| hit@20 | **0.3837** | <u>0.2566</u> | 0.3669 | 0.3653 | 0.3605 | <u>0.2742</u> | <u>0.2370</u> | 0.2962 | 0.3477 | *0.3777 |
| ndcg@5 | 0.1791 | <u>0.1352</u> | **0.1858** | 0.1719 | 0.1708 | 0.1480 | <u>0.1293</u> | 0.1645 | *0.1834 | 0.1827 |
| ndcg@10 | 0.2033 | <u>0.1497</u> | *0.2037 | 0.1935 | 0.1950 | 0.1594 | <u>0.1396</u> | 0.1784 | 0.2001 | **0.2041** |
| ndcg@20 | **0.2195** | <u>0.1585</u> | 0.2180 | 0.2083 | 0.2075 | 0.1695 | <u>0.1478</u> | 0.1882 | 0.2123 | *0.2181 |
| mrr@5 | 0.1575 | <u>0.1213</u> | **0.1628** | 0.1494 | 0.1487 | 0.1309 | <u>0.1148</u> | 0.1478 | *0.1620 | 0.1584 |
| mrr@10 | 0.1675 | <u>0.1273</u> | **0.1703** | 0.1583 | 0.1588 | 0.1357 | <u>0.1192</u> | 0.1535 | *0.1689 | 0.1673 |
| mrr@20 | 0.1719 | <u>0.1297</u> | **0.1742** | 0.1624 | 0.1623 | 0.1385 | <u>0.1214</u> | 0.1563 | *0.1723 | 0.1712 |

<details>
<summary>Overall</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.1719 | <u>0.1244</u> | *0.1797 | 0.1685 | 0.1671 | 0.1411 | <u>0.1216</u> | 0.1527 | 0.1741 | **0.1800** |
| hit@10 | *0.2247 | <u>0.1559</u> | 0.2182 | 0.2157 | 0.2193 | <u>0.1655</u> | <u>0.1438</u> | 0.1826 | 0.2106 | **0.2266** |
| hit@20 | **0.2696** | <u>0.1803</u> | 0.2578 | 0.2567 | 0.2533 | <u>0.1939</u> | <u>0.1665</u> | 0.2099 | 0.2443 | *0.2654 |
| ndcg@5 | 0.1258 | <u>0.0950</u> | **0.1305** | 0.1208 | 0.1200 | 0.1046 | <u>0.0908</u> | 0.1165 | *0.1288 | 0.1283 |
| ndcg@10 | 0.1429 | <u>0.1052</u> | *0.1431 | 0.1360 | 0.1370 | 0.1127 | <u>0.0981</u> | 0.1263 | 0.1406 | **0.1434** |
| ndcg@20 | **0.1542** | <u>0.1114</u> | 0.1532 | 0.1464 | 0.1458 | 0.1198 | <u>0.1039</u> | 0.1333 | 0.1492 | *0.1533 |
| mrr@5 | 0.1106 | <u>0.0852</u> | **0.1144** | 0.1050 | 0.1045 | 0.0925 | <u>0.0807</u> | 0.1046 | *0.1138 | 0.1113 |
| mrr@10 | 0.1177 | <u>0.0894</u> | **0.1196** | 0.1112 | 0.1116 | 0.0960 | <u>0.0837</u> | 0.1087 | *0.1186 | 0.1175 |
| mrr@20 | 0.1208 | <u>0.0911</u> | **0.1224** | 0.1141 | 0.1141 | 0.0979 | <u>0.0853</u> | 0.1106 | *0.1210 | 0.1203 |

</details>

<details>
<summary>Unseen</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0018** | *<u>0.0000</u> | *<u>0.0000</u> |
| hit@10 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0018** | *<u>0.0000</u> | *<u>0.0000</u> |
| hit@20 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0018** | *<u>0.0000</u> | *<u>0.0000</u> |
| ndcg@5 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0007** | *<u>0.0000</u> | *<u>0.0000</u> |
| ndcg@10 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0007** | *<u>0.0000</u> | *<u>0.0000</u> |
| ndcg@20 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0007** | *<u>0.0000</u> | *<u>0.0000</u> |
| mrr@5 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0004** | *<u>0.0000</u> | *<u>0.0000</u> |
| mrr@10 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0004** | *<u>0.0000</u> | *<u>0.0000</u> |
| mrr@20 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0004** | *<u>0.0000</u> | *<u>0.0000</u> |

</details>

## movielens1m

### Seen

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.0957 | 0.0920 | 0.1018 | <u>0.0860</u> | 0.0915 | **0.1050** | 0.0976 | 0.0888 | *0.1046 | 0.0953 |
| hit@10 | 0.1478 | 0.1408 | 0.1538 | 0.1422 | <u>0.1399</u> | *0.1589 | 0.1557 | 0.1501 | **0.1594** | 0.1515 |
| hit@20 | 0.2254 | <u>0.2091</u> | *0.2375 | 0.2226 | 0.2147 | 0.2282 | 0.2212 | 0.2300 | **0.2430** | 0.2249 |
| ndcg@5 | 0.0605 | 0.0590 | *0.0677 | <u>0.0552</u> | 0.0573 | **0.0680** | 0.0667 | 0.0566 | 0.0656 | 0.0588 |
| ndcg@10 | 0.0770 | 0.0745 | 0.0844 | 0.0731 | <u>0.0728</u> | **0.0854** | *0.0853 | 0.0764 | 0.0835 | 0.0769 |
| ndcg@20 | 0.0965 | 0.0916 | **0.1054** | 0.0932 | <u>0.0916</u> | 0.1028 | 0.1018 | 0.0963 | *0.1043 | 0.0954 |
| mrr@5 | 0.0489 | 0.0482 | *0.0565 | <u>0.0451</u> | 0.0461 | 0.0559 | **0.0566** | 0.0460 | 0.0529 | 0.0469 |
| mrr@10 | 0.0556 | 0.0545 | *0.0633 | <u>0.0523</u> | 0.0525 | 0.0631 | **0.0642** | 0.0542 | 0.0603 | 0.0544 |
| mrr@20 | 0.0608 | 0.0591 | **0.0690** | 0.0577 | <u>0.0576</u> | 0.0678 | *0.0687 | 0.0595 | 0.0658 | 0.0594 |

<details>
<summary>Overall</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.0954 | 0.0917 | 0.1014 | <u>0.0857</u> | 0.0912 | **0.1047** | 0.0973 | 0.0885 | *0.1042 | 0.0950 |
| hit@10 | 0.1473 | 0.1403 | 0.1533 | 0.1417 | <u>0.1394</u> | *0.1584 | 0.1552 | 0.1496 | **0.1589** | 0.1510 |
| hit@20 | 0.2246 | <u>0.2084</u> | *0.2367 | 0.2219 | 0.2140 | 0.2274 | 0.2205 | 0.2293 | **0.2422** | 0.2242 |
| ndcg@5 | 0.0603 | 0.0588 | *0.0675 | <u>0.0550</u> | 0.0571 | **0.0678** | 0.0664 | 0.0564 | 0.0654 | 0.0586 |
| ndcg@10 | 0.0768 | 0.0743 | 0.0841 | 0.0729 | <u>0.0726</u> | **0.0851** | *0.0850 | 0.0762 | 0.0832 | 0.0767 |
| ndcg@20 | 0.0962 | 0.0913 | **0.1051** | 0.0929 | <u>0.0913</u> | 0.1025 | 0.1014 | 0.0960 | *0.1039 | 0.0951 |
| mrr@5 | 0.0487 | 0.0481 | *0.0563 | <u>0.0449</u> | 0.0460 | 0.0557 | **0.0564** | 0.0459 | 0.0527 | 0.0467 |
| mrr@10 | 0.0554 | 0.0543 | *0.0631 | <u>0.0521</u> | 0.0523 | 0.0629 | **0.0640** | 0.0540 | 0.0602 | 0.0542 |
| mrr@20 | 0.0606 | 0.0589 | **0.0688** | 0.0575 | <u>0.0574</u> | 0.0676 | *0.0684 | 0.0593 | 0.0656 | 0.0592 |

</details>

<details>
<summary>Unseen</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hit@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hit@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

</details>

## lastfm0.03

### Seen

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | **0.3649** | <u>0.2829</u> | 0.3479 | 0.3573 | 0.3526 | 0.3294 | 0.3168 | 0.3406 | 0.3430 | *0.3584 |
| hit@10 | *0.3882 | <u>0.3090</u> | 0.3794 | 0.3845 | 0.3831 | 0.3467 | 0.3355 | 0.3586 | 0.3657 | **0.3893** |
| hit@20 | *0.4081 | <u>0.3341</u> | 0.4031 | 0.4060 | **0.4089** | 0.3603 | 0.3525 | 0.3766 | 0.3909 | 0.4060 |
| ndcg@5 | *0.3184 | <u>0.2589</u> | 0.3110 | 0.3038 | 0.2984 | 0.2998 | 0.2951 | 0.3137 | 0.3097 | **0.3196** |
| ndcg@10 | *0.3260 | <u>0.2673</u> | 0.3211 | 0.3127 | 0.3083 | 0.3053 | 0.3012 | 0.3195 | 0.3171 | **0.3297** |
| ndcg@20 | *0.3311 | <u>0.2736</u> | 0.3271 | 0.3182 | 0.3149 | 0.3088 | 0.3055 | 0.3240 | 0.3234 | **0.3340** |
| mrr@5 | 0.3029 | <u>0.2509</u> | 0.2987 | 0.2859 | 0.2803 | 0.2898 | 0.2877 | *0.3047 | 0.2986 | **0.3067** |
| mrr@10 | 0.3060 | <u>0.2543</u> | 0.3028 | 0.2896 | 0.2844 | 0.2921 | 0.2902 | *0.3071 | 0.3016 | **0.3109** |
| mrr@20 | 0.3075 | <u>0.2561</u> | 0.3045 | 0.2911 | 0.2863 | 0.2930 | 0.2914 | *0.3083 | 0.3033 | **0.3121** |

<details>
<summary>Overall</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | **0.2922** | <u>0.2265</u> | 0.2786 | 0.2861 | 0.2823 | 0.2574 | 0.2476 | 0.2662 | 0.2681 | *0.2870 |
| hit@10 | *0.3108 | <u>0.2475</u> | 0.3039 | 0.3079 | 0.3068 | 0.2710 | 0.2622 | 0.2803 | 0.2859 | **0.3117** |
| hit@20 | *0.3268 | <u>0.2675</u> | 0.3228 | 0.3251 | **0.3274** | 0.2816 | 0.2755 | 0.2944 | 0.3055 | 0.3251 |
| ndcg@5 | *0.2550 | <u>0.2073</u> | 0.2490 | 0.2433 | 0.2390 | 0.2343 | 0.2306 | 0.2452 | 0.2421 | **0.2560** |
| ndcg@10 | *0.2610 | <u>0.2141</u> | 0.2571 | 0.2504 | 0.2469 | 0.2387 | 0.2354 | 0.2497 | 0.2478 | **0.2640** |
| ndcg@20 | *0.2651 | <u>0.2191</u> | 0.2620 | 0.2548 | 0.2522 | 0.2413 | 0.2388 | 0.2533 | 0.2528 | **0.2674** |
| mrr@5 | *0.2425 | <u>0.2009</u> | 0.2392 | 0.2289 | 0.2244 | 0.2265 | 0.2249 | 0.2382 | 0.2334 | **0.2456** |
| mrr@10 | *0.2451 | <u>0.2037</u> | 0.2425 | 0.2319 | 0.2277 | 0.2283 | 0.2268 | 0.2400 | 0.2358 | **0.2489** |
| mrr@20 | *0.2462 | <u>0.2051</u> | 0.2439 | 0.2331 | 0.2292 | 0.2290 | 0.2278 | 0.2410 | 0.2371 | **0.2499** |

</details>

<details>
<summary>Unseen</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hit@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hit@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ndcg@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrr@20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

</details>

## KuaiRecLargeStrictPosV2_0.2

### Seen

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.3359 | <u>0.2710</u> | 0.3225 | *0.3382 | 0.3259 | 0.3337 | 0.3303 | 0.3046 | 0.3281 | **0.3471** |
| hit@10 | 0.3427 | <u>0.3024</u> | 0.3427 | 0.3438 | 0.3415 | 0.3359 | 0.3315 | 0.3113 | *0.3471 | **0.3617** |
| hit@20 | 0.3460 | 0.3337 | 0.3628 | 0.3639 | *0.3684 | 0.3371 | 0.3359 | <u>0.3169</u> | 0.3673 | **0.3975** |
| ndcg@5 | *0.3344 | <u>0.2480</u> | 0.3009 | 0.3218 | 0.3151 | 0.3307 | 0.3297 | 0.2930 | 0.3179 | **0.3407** |
| ndcg@10 | *0.3366 | <u>0.2585</u> | 0.3076 | 0.3235 | 0.3203 | 0.3314 | 0.3300 | 0.2953 | 0.3239 | **0.3454** |
| ndcg@20 | *0.3374 | <u>0.2664</u> | 0.3126 | 0.3286 | 0.3271 | 0.3317 | 0.3312 | 0.2967 | 0.3289 | **0.3545** |
| mrr@5 | *0.3339 | <u>0.2403</u> | 0.2937 | 0.3163 | 0.3115 | 0.3296 | 0.3295 | 0.2892 | 0.3145 | **0.3386** |
| mrr@10 | *0.3348 | <u>0.2448</u> | 0.2966 | 0.3170 | 0.3136 | 0.3299 | 0.3296 | 0.2902 | 0.3168 | **0.3405** |
| mrr@20 | *0.3350 | <u>0.2470</u> | 0.2979 | 0.3184 | 0.3155 | 0.3300 | 0.3300 | 0.2906 | 0.3182 | **0.3430** |

<details>
<summary>Overall</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | 0.1109 | <u>0.0894</u> | 0.1064 | *0.1116 | 0.1075 | 0.1109 | 0.1090 | 0.1005 | 0.1083 | **0.1146** |
| hit@10 | 0.1131 | <u>0.0998</u> | 0.1131 | 0.1135 | 0.1127 | 0.1116 | 0.1094 | 0.1027 | *0.1146 | **0.1194** |
| hit@20 | 0.1142 | 0.1101 | 0.1197 | 0.1201 | *0.1216 | 0.1131 | 0.1116 | <u>0.1046</u> | 0.1212 | **0.1312** |
| ndcg@5 | *0.1104 | <u>0.0819</u> | 0.0993 | 0.1062 | 0.1040 | 0.1094 | 0.1088 | 0.0967 | 0.1049 | **0.1124** |
| ndcg@10 | *0.1111 | <u>0.0853</u> | 0.1015 | 0.1068 | 0.1057 | 0.1097 | 0.1089 | 0.0975 | 0.1069 | **0.1140** |
| ndcg@20 | *0.1113 | <u>0.0879</u> | 0.1032 | 0.1084 | 0.1079 | 0.1100 | 0.1095 | 0.0979 | 0.1085 | **0.1170** |
| mrr@5 | *0.1102 | <u>0.0793</u> | 0.0969 | 0.1044 | 0.1028 | 0.1089 | 0.1087 | 0.0954 | 0.1038 | **0.1117** |
| mrr@10 | *0.1105 | <u>0.0808</u> | 0.0979 | 0.1046 | 0.1035 | 0.1090 | 0.1088 | 0.0958 | 0.1046 | **0.1124** |
| mrr@20 | *0.1106 | <u>0.0815</u> | 0.0983 | 0.1051 | 0.1041 | 0.1091 | 0.1089 | 0.0959 | 0.1050 | **0.1132** |

</details>

<details>
<summary>Unseen</summary>

| metric | SASRec | GRU4Rec | TiSASRec | FEARec | DuoRec | BSARec | FAME | DIFSR | FDSA | RouteRec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit@5 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0011** | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> |
| hit@10 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0011** | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> |
| hit@20 | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | **0.0028** | *<u>0.0011</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> |
| ndcg@5 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0005** | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> |
| ndcg@10 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0005** | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> |
| ndcg@20 | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | **0.0009** | *<u>0.0003</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> |
| mrr@5 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0002** | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> |
| mrr@10 | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | **0.0002** | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> | *<u>0.0000</u> |
| mrr@20 | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> | **0.0004** | *<u>0.0001</u> | <u>0.0000</u> | <u>0.0000</u> | <u>0.0000</u> |

</details>

## RouteRec coverage

| dataset | selected source | selected axis | run phase | best valid mrr@20 | seen test mrr@20 | result json |
|---|---|---|---|---:|---:|---|
| beauty | fmoe_n4 | crossdataset_a12_portfolio | P4XD_XD_BEAUTY_B25_LR_H8_SEEN_ANCHOR_S1 | 0.0865 | 0.0822 | /workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/beauty_FeaturedMoE_N3_p4xd_xd_beauty_b25_lr_h8_seen_anchor_s1_20260416_030201_046233_pid244350.json |
| retail_rocket | fmoe_n4 | crossdataset_a12_portfolio | P4XD_XD_RETAIL_ROCKET_R10_H13_WIDTH_REFINE_S1 | 0.3726 | 0.3737 | /workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/retail_rocket_FeaturedMoE_N3_p4xd_xd_retail_rocket_r10_h13_width_refine_s1_20260416_044940_836611_pid262468.json |
| foursquare | fmoe_n4 | crossdataset_a12_portfolio | P4XD_XD_FOURSQUARE_F26_H11_FAST_ATTACK_S1 | 0.2045 | 0.1712 | /workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/foursquare_FeaturedMoE_N3_p4xd_xd_foursquare_f26_h11_fast_attack_s1_20260416_065445_664405_pid287135.json |
| movielens1m | fmoe_n4 | crossdataset_a12_portfolio | P4XD_XD_MOVIELENS1M_M05_H6_E2_COMPACT_S1 | 0.0916 | 0.0594 | /workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/movielens1m_FeaturedMoE_N3_p4xd_xd_movielens1m_m05_h6_e2_compact_s1_20260415_122804_986294_pid160086.json |
| lastfm0.03 | fmoe_n4 | crossdataset_a12_portfolio | P4XD_XD_LASTFM0_03_L01_H5_ANCHOR_S1 | 0.3105 | 0.3121 | /workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/lastfm0.03_FeaturedMoE_N3_p4xd_xd_lastfm0_03_l01_h5_anchor_s1_20260415_144246_970770_pid179137.json |
| KuaiRecLargeStrictPosV2_0.2 | fmoe_n4 | stage1_a12_broadtemplates | P4S1_S1_KUAIRECLARGESTRICTPOSV2_0_2_T05_CAPACITY_H14_LO_S1 | 0.0137 | 0.3430 | /workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N3_p4s1_s1_kuaireclargestrictposv2_0_2_t05_capacity_h14_lo_s1_20260415_043811_989578_pid92982.json |

## Selected runs

<details>
<summary>Expand selected run metadata</summary>

| dataset | model | source | selected axis | best valid mrr@20 | seen test mrr@20 | selection score | run phase |
|---|---|---|---|---:|---:|---:|---|
| beauty | SASRec | baseline_2 | pair60_v3_lr10 | 0.0698 | 0.0735 | 0.0735 | PAIR60_V3_LR10_DBEAUTY_MSASREC_P021_C1_S1 |
| beauty | GRU4Rec | baseline_2 | pair60_addtuning3_2 | 0.0714 | 0.0411 | 0.0411 | BASELINE2_ADDTUNE3_2_BEAUTY_GRU4REC_K11 |
| beauty | TiSASRec | baseline_2 | pair60_addtuning3 | 0.0958 | 0.0839 | 0.0839 | BASELINE2_ADDTUNE3_BEAUTY_TISASREC_K1 |
| beauty | FEARec | baseline_2 | pair60_v4 | 0.0620 | 0.0692 | 0.0692 | PAIR60_V4_DBEAUTY_MFEAREC_P006_C2_S1 |
| beauty | DuoRec | baseline_2 | pair60_addtuning3 | 0.0653 | 0.0686 | 0.0686 | BASELINE2_ADDTUNE3_BEAUTY_DUOREC_K6 |
| beauty | BSARec | baseline_2 | pair60_addtuning3_2 | 0.0432 | 0.0519 | 0.0519 | BASELINE2_ADDTUNE3_2_BEAUTY_BSAREC_K27 |
| beauty | FAME | baseline_2 | pair60_addtuning3_2 | 0.0242 | 0.0313 | 0.0313 | BASELINE2_ADDTUNE3_2_BEAUTY_FAME_K41 |
| beauty | DIFSR | baseline_2 | pair60_addtuning3_2 | 0.0834 | 0.0752 | 0.0752 | BASELINE2_ADDTUNE3_2_BEAUTY_DIFSR_K35 |
| beauty | FDSA | baseline_2 | pair60_addtuning3 | 0.1001 | 0.0841 | 0.0841 | BASELINE2_ADDTUNE3_BEAUTY_FDSA_K8 |
| beauty | RouteRec | fmoe_n4 | crossdataset_a12_portfolio | 0.0865 | 0.0822 | 0.0822 | P4XD_XD_BEAUTY_B25_LR_H8_SEEN_ANCHOR_S1 |
| retail_rocket | SASRec | baseline_2 | abcd_v1 | 0.3699 | 0.3696 | 0.3696 | ABCD_v1_A_DRETAIL_ROCKET_MSASREC_A013_L05_S1 |
| retail_rocket | GRU4Rec | baseline_2 | abcd_v2_lean | 0.3467 | 0.3479 | 0.3479 | ABCD_v2_lean_B_DRETAIL_ROCKET_MGRU4REC_A009_L04_S1 |
| retail_rocket | TiSASRec | baseline_2 | pair60_v4_revised_long12h | 0.3548 | 0.3547 | 0.3547 | PAIR60_V4_REVISED_LONG12H_DRETAIL_ROCKET_MTISASREC_P013_C2_S1_REV1 |
| retail_rocket | FEARec | baseline_2 | pair60_v4 | 0.3569 | 0.3515 | 0.3515 | PAIR60_V4_DRETAIL_ROCKET_MFEAREC_P016_C2_S1 |
| retail_rocket | DuoRec | baseline_2 | pair60_v4 | 0.3483 | 0.3441 | 0.3441 | PAIR60_V4_DRETAIL_ROCKET_MDUOREC_P014_C1_S1 |
| retail_rocket | BSARec | baseline_2 | pair60_addtuning | 0.3774 | 0.3789 | 0.3789 | BASELINE2_ADDTUNE_RETAIL_ROCKET_BSAREC_K3 |
| retail_rocket | FAME | baseline_2 | pair60_addtuning | 0.3736 | 0.3753 | 0.3753 | BASELINE2_ADDTUNE_RETAIL_ROCKET_FAME_K1 |
| retail_rocket | DIFSR | baseline_2 | pair60_v4_revised_long12h | 0.3705 | 0.3744 | 0.3744 | PAIR60_V4_REVISED_LONG12H_DRETAIL_ROCKET_MDIFSR_P017_C1_S1_REV1 |
| retail_rocket | FDSA | baseline_2 | pair60_v4_revised_long12h | 0.3294 | 0.3239 | 0.3239 | PAIR60_V4_REVISED_LONG12H_DRETAIL_ROCKET_MFDSA_P019_C2_S1_REV1 |
| retail_rocket | RouteRec | fmoe_n4 | crossdataset_a12_portfolio | 0.3726 | 0.3737 | 0.3737 | P4XD_XD_RETAIL_ROCKET_R10_H13_WIDTH_REFINE_S1 |
| foursquare | SASRec | baseline_2 | abcd_v2_lean | 0.2105 | 0.1719 | 0.1719 | ABCD_v2_lean_B_DFOURSQUARE_MSASREC_A007_L03_S1 |
| foursquare | GRU4Rec | baseline_2 | abcd_v2_lean | 0.1635 | 0.1297 | 0.1297 | ABCD_v2_lean_B_DFOURSQUARE_MGRU4REC_A009_L04_S1 |
| foursquare | TiSASRec | baseline_2 | abcd_v2_lean | 0.2090 | 0.1742 | 0.1742 | ABCD_v2_lean_B_DFOURSQUARE_MTISASREC_A007_L02_S1 |
| foursquare | FEARec | baseline_2 | pair60_addtuning | 0.1859 | 0.1624 | 0.1624 | BASELINE2_ADDTUNE_FOURSQUARE_FEAREC_K1 |
| foursquare | DuoRec | baseline_2 | pair60_v4 | 0.1889 | 0.1623 | 0.1623 | PAIR60_V4_DFOURSQUARE_MDUOREC_P034_C1_S1 |
| foursquare | BSARec | baseline_2 | pair60_v3_lr10 | 0.1569 | 0.1385 | 0.1385 | PAIR60_V3_LR10_DFOURSQUARE_MBSAREC_P035_C3_S1 |
| foursquare | FAME | baseline_2 | pair60_addtuning | 0.1509 | 0.1214 | 0.1214 | BASELINE2_ADDTUNE_FOURSQUARE_FAME_K3 |
| foursquare | DIFSR | baseline_2 | pair60_v3_lr10 | 0.1603 | 0.1563 | 0.1563 | PAIR60_V3_LR10_DFOURSQUARE_MDIFSR_P037_C3_S1 |
| foursquare | FDSA | baseline_2 | pair60_addtuning | 0.2125 | 0.1723 | 0.1723 | BASELINE2_ADDTUNE_FOURSQUARE_FDSA_K1 |
| foursquare | RouteRec | fmoe_n4 | crossdataset_a12_portfolio | 0.2045 | 0.1712 | 0.1712 | P4XD_XD_FOURSQUARE_F26_H11_FAST_ATTACK_S1 |
| movielens1m | SASRec | baseline_2 | pair60_v4_revised_long12h | 0.0904 | 0.0608 | 0.0608 | PAIR60_V4_REVISED_LONG12H_DMOVIELENS1M_MSASREC_P041_C2_S1_REV1 |
| movielens1m | GRU4Rec | baseline_2 | abcd_v2_lean | 0.0865 | 0.0591 | 0.0591 | ABCD_v2_lean_A_DMOVIELENS1M_MGRU4REC_A009_L04_S1 |
| movielens1m | TiSASRec | baseline_2 | pair60_v4_revised_long12h | 0.1010 | 0.0690 | 0.0690 | PAIR60_V4_REVISED_LONG12H_DMOVIELENS1M_MTISASREC_P043_C2_S1_REV1 |
| movielens1m | FEARec | baseline_2 | pair60_v4 | 0.0789 | 0.0577 | 0.0577 | PAIR60_V4_DMOVIELENS1M_MFEAREC_P046_C1_S1 |
| movielens1m | DuoRec | baseline_2 | pair60_addtuning | 0.0840 | 0.0576 | 0.0576 | BASELINE2_ADDTUNE_MOVIELENS1M_DUOREC_K1 |
| movielens1m | BSARec | baseline_2 | pair60_v4 | 0.0984 | 0.0678 | 0.0678 | PAIR60_V4_DMOVIELENS1M_MBSAREC_P045_C2_S1 |
| movielens1m | FAME | baseline_2 | pair60_v4 | 0.0914 | 0.0687 | 0.0687 | PAIR60_V4_DMOVIELENS1M_MFAME_P048_C2_S1 |
| movielens1m | DIFSR | baseline_2 | pair60_v4 | 0.0900 | 0.0595 | 0.0595 | PAIR60_V4_DMOVIELENS1M_MDIFSR_P047_C1_S1 |
| movielens1m | FDSA | baseline_2 | pair60_v4 | 0.1036 | 0.0658 | 0.0658 | PAIR60_V4_DMOVIELENS1M_MFDSA_P049_C1_S1 |
| movielens1m | RouteRec | fmoe_n4 | crossdataset_a12_portfolio | 0.0916 | 0.0594 | 0.0594 | P4XD_XD_MOVIELENS1M_M05_H6_E2_COMPACT_S1 |
| lastfm0.03 | SASRec | baseline_2 | pair60_v4 | 0.3057 | 0.3075 | 0.3075 | PAIR60_V4_DLASTFM0_03_MSASREC_P051_C1_S1 |
| lastfm0.03 | GRU4Rec | baseline_2 | abcd_v2_lean | 0.2655 | 0.2561 | 0.2561 | ABCD_v2_lean_A_DLASTFM0.03_MGRU4REC_A006_L04_S1 |
| lastfm0.03 | TiSASRec | baseline_2 | abcd_v2_lean | 0.3027 | 0.3045 | 0.3045 | ABCD_v2_lean_A_DLASTFM0.03_MTISASREC_A001_L02_S1 |
| lastfm0.03 | FEARec | baseline_2 | pair60_v4 | 0.3001 | 0.2911 | 0.2911 | PAIR60_V4_DLASTFM0_03_MFEAREC_P056_C1_S1 |
| lastfm0.03 | DuoRec | baseline_2 | pair60_v4_revised_long12h | 0.2859 | 0.2863 | 0.2863 | PAIR60_V4_REVISED_LONG12H_DLASTFM0_03_MDUOREC_P054_C2_S1_REV1 |
| lastfm0.03 | BSARec | baseline_2 | pair60_v3_lr10 | 0.2907 | 0.2930 | 0.2930 | PAIR60_V3_LR10_DLASTFM0_03_MBSAREC_P015_C2_S1 |
| lastfm0.03 | FAME | baseline_2 | pair60_v3_lr10 | 0.2857 | 0.2914 | 0.2914 | PAIR60_V3_LR10_DLASTFM0_03_MFAME_P018_C1_S1 |
| lastfm0.03 | DIFSR | baseline_2 | pair60_v3_lr10 | 0.3006 | 0.3083 | 0.3083 | PAIR60_V3_LR10_DLASTFM0_03_MDIFSR_P017_C3_S1 |
| lastfm0.03 | FDSA | baseline_2 | pair60_v3_lr10 | 0.2987 | 0.3033 | 0.3033 | PAIR60_V3_LR10_DLASTFM0_03_MFDSA_P019_C2_S1 |
| lastfm0.03 | RouteRec | fmoe_n4 | crossdataset_a12_portfolio | 0.3105 | 0.3121 | 0.3121 | P4XD_XD_LASTFM0_03_L01_H5_ANCHOR_S1 |
| KuaiRecLargeStrictPosV2_0.2 | SASRec | baseline_2 | abcd_v1 | 0.0144 | 0.3350 | 0.3350 | ABCD_v1_A_DKUAIRECLARGESTRICTPOSV2_0.2_MSASREC_A013_L01_S1 |
| KuaiRecLargeStrictPosV2_0.2 | GRU4Rec | baseline_2 | abcd_v2_lean | 0.0159 | 0.2470 | 0.2470 | ABCD_v2_lean_A_DKUAIRECLARGESTRICTPOSV2_0.2_MGRU4REC_A009_L04_S1 |
| KuaiRecLargeStrictPosV2_0.2 | TiSASRec | baseline_2 | abcd_v2_lean | 0.0188 | 0.2979 | 0.2979 | ABCD_v2_lean_A_DKUAIRECLARGESTRICTPOSV2_0.2_MTISASREC_A017_L03_S1 |
| KuaiRecLargeStrictPosV2_0.2 | FEARec | baseline_2 | pair60_v4 | 0.0187 | 0.3184 | 0.3184 | PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MFEAREC_P026_C1_S1 |
| KuaiRecLargeStrictPosV2_0.2 | DuoRec | baseline_2 | pair60_v4 | 0.0189 | 0.3155 | 0.3155 | PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MDUOREC_P024_C2_S1 |
| KuaiRecLargeStrictPosV2_0.2 | BSARec | baseline_2 | pair60_v4 | 0.0133 | 0.3300 | 0.3300 | PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MBSAREC_P025_C1_S1 |
| KuaiRecLargeStrictPosV2_0.2 | FAME | baseline_2 | pair60_addtuning | 0.0132 | 0.3300 | 0.3300 | BASELINE2_ADDTUNE_KUAIRECLARGESTRICTPOSV2_0.2_FAME_K3 |
| KuaiRecLargeStrictPosV2_0.2 | DIFSR | baseline_2 | pair60_v4 | 0.0138 | 0.2906 | 0.2906 | PAIR60_V4_DKUAIRECLARGESTRICTPOSV2_0_2_MDIFSR_P027_C1_S1 |
| KuaiRecLargeStrictPosV2_0.2 | FDSA | baseline_2 | pair60_addtuning | 0.0175 | 0.3182 | 0.3182 | BASELINE2_ADDTUNE_KUAIRECLARGESTRICTPOSV2_0.2_FDSA_K4 |
| KuaiRecLargeStrictPosV2_0.2 | RouteRec | fmoe_n4 | stage1_a12_broadtemplates | 0.0137 | 0.3430 | 0.3430 | P4S1_S1_KUAIRECLARGESTRICTPOSV2_0_2_T05_CAPACITY_H14_LO_S1 |

</details>

