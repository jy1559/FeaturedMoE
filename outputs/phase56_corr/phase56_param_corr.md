# Param vs MRR (from phase5/phase6 artifact logs)

- parsed runs: 106

|param|n|unique_values|pearson_r(valid_mrr)|
|---|---:|---:|---:|
|group_prior_align_lambda|98|3|-0.1205|
|factored_group_balance_lambda|98|3|-0.1042|
|route_consistency_pairs|12|2|0.0910|
|route_consistency_lambda|106|4|0.0683|
|route_prior_lambda|106|3|0.0564|
|route_smoothness_lambda|106|4|0.0482|
|route_sharpness_lambda|53|2|0.0481|
|route_monopoly_lambda|53|2|0.0481|
|balance_loss_lambda|106|2|0.0379|
|z_loss_lambda|106|2|0.0349|

SVG files: param_scatter_*_vs_valid.svg