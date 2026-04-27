import json
f = '/workspace/FeaturedMoE/experiments/run/artifacts/results/final_experiment_ablation/beauty_FeaturedMoE_N3_q2_beauty_shared_ffn_r01_s3_20260419_013843_627362_pid692152.json'
d = json.load(open(f))
trials = d['trials']
print('n_trials:', len(trials))
t = trials[-1]
print('trial keys:', list(t.keys())[:20])
er = t.get('eval_result', {})
print('eval_result keys:', list(er.keys()))
ost = er.get('overall_seen_target', {})
print('overall_seen_target keys:', list(ost.keys())[:20])
if ost:
    print('sample:', {k: v for k,v in list(ost.items())[:10]})
