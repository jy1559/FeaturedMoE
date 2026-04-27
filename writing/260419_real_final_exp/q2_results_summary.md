# Q2 Results Summary - Seen Target Test Metrics

## KuaiRecLargeStrictPosV2_0.2

### Average Test MRR@20 (2 seeds averaged)

| Variant | MRR@20 |
|---------|--------|
| Shared FFN | 0.2981 |
| Hidden only | 0.3380 |
| Fusion bias | 0.3096 |
| Mixed | <u>0.3384</u> |
| Behavior-guided | **0.3385** |

### Average Test NDCG@20 (2 seeds averaged)

| Variant | NDCG@20 |
|---------|---------|
| Shared FFN | 0.3109 |
| Hidden only | 0.3468 |
| Fusion bias | 0.3214 |
| Mixed | **0.3474** |
| Behavior-guided | <u>0.3472</u> |

### Average Test HR@10 (2 seeds averaged)

| Variant | HR@10 |
|---------|-------|
| Shared FFN | 0.3286 |
| Hidden only | <u>0.3589</u> |
| Fusion bias | 0.3393 |
| Mixed | 0.3578 |
| Behavior-guided | **0.3606** |

---

## beauty

### Average Test MRR@20 (2 seeds averaged)

| Variant | MRR@20 |
|---------|--------|
| Shared FFN | <u>0.0753</u> |
| Hidden only | 0.0677 |
| Fusion bias | **0.0779** |
| Mixed | 0.0685 |
| Behavior-guided | 0.0677 |

### Average Test NDCG@20 (2 seeds averaged)

| Variant | NDCG@20 |
|---------|---------|
| Shared FFN | <u>0.0979</u> |
| Hidden only | 0.0920 |
| Fusion bias | **0.1036** |
| Mixed | 0.0930 |
| Behavior-guided | 0.0917 |

### Average Test HR@10 (2 seeds averaged)

| Variant | HR@10 |
|---------|-------|
| Shared FFN | **0.1390** |
| Hidden only | <u>0.1390</u> |
| Fusion bias | 0.1361 |
| Mixed | 0.1361 |
| Behavior-guided | 0.1332 |

---

## foursquare

### Average Test MRR@20 (2 seeds averaged)

| Variant | MRR@20 |
|---------|--------|
| Shared FFN | **0.1729** |
| Hidden only | <u>0.1709</u> |
| Fusion bias | 0.1678 |
| Mixed | 0.1709 |
| Behavior-guided | 0.1697 |

### Average Test NDCG@20 (2 seeds averaged)

| Variant | NDCG@20 |
|---------|---------|
| Shared FFN | **0.2184** |
| Hidden only | 0.2178 |
| Fusion bias | 0.2150 |
| Mixed | <u>0.2182</u> |
| Behavior-guided | 0.2171 |

### Average Test HR@10 (2 seeds averaged)

| Variant | HR@10 |
|---------|-------|
| Shared FFN | 0.3173 |
| Hidden only | 0.3207 |
| Fusion bias | **0.3225** |
| Mixed | 0.3201 |
| Behavior-guided | <u>0.3213</u> |

---

## Summary

**KuaiRecLargeStrictPosV2_0.2**: Behavior-guided variant performs best overall (MRR@20: **0.3385**, HR@10: **0.3606**), with Mixed variant close second on NDCG@20 (**0.3474**)

**beauty**: Fusion bias variant leads (MRR@20: **0.0779**, NDCG@20: **0.1036**), with Shared FFN strong on HR@10 (**0.1390**)

**foursquare**: Shared FFN dominant on MRR@20 (**0.1729**) and NDCG@20 (**0.2184**), but Fusion bias leads on HR@10 (**0.3225**)
