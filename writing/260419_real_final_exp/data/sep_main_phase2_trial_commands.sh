#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FeaturedMoE && /venv/FMoE/bin/python experiments/run/final_experiment/separation_main_trial_phase2.py --datasets KuaiRecLargeStrictPosV2_0.2,foursquare --gpus 0,1,2,3,4,5,6,7 --resume

# Selected trial candidates
# KuaiRecLargeStrictPosV2_0.2 highsep#1 trial=26 sep=0.02978 test_mrr20=0.3396
# KuaiRecLargeStrictPosV2_0.2 highsep#2 trial=15 sep=0.0274 test_mrr20=0.3382
# KuaiRecLargeStrictPosV2_0.2 highsep#3 trial=13 sep=0.02688 test_mrr20=0.3380
# KuaiRecLargeStrictPosV2_0.2 highsep#4 trial=10 sep=0.02076 test_mrr20=0.3369
# KuaiRecLargeStrictPosV2_0.2 perf#1 trial=3 sep=0.004595 test_mrr20=0.3412
# KuaiRecLargeStrictPosV2_0.2 perf#2 trial=25 sep=0.001655 test_mrr20=0.3410
# KuaiRecLargeStrictPosV2_0.2 perf#3 trial=4 sep=0.01782 test_mrr20=0.3408
# KuaiRecLargeStrictPosV2_0.2 perf#4 trial=7 sep=0.0091 test_mrr20=0.3407
# foursquare highsep#1 trial=23 sep=0.02883 test_mrr20=0.1751
# foursquare highsep#2 trial=12 sep=0.02882 test_mrr20=0.1733
# foursquare highsep#3 trial=16 sep=0.02754 test_mrr20=0.1720
# foursquare highsep#4 trial=29 sep=0.02636 test_mrr20=0.1731
# foursquare perf#1 trial=3 sep=0.001493 test_mrr20=0.1755
# foursquare perf#2 trial=21 sep=0.007362 test_mrr20=0.1758
# foursquare perf#3 trial=9 sep=0.007544 test_mrr20=0.1756
# foursquare perf#4 trial=13 sep=0.001599 test_mrr20=0.1759
