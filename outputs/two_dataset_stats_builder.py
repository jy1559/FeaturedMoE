import csv
import json
import math
from collections import Counter
from pathlib import Path

base = Path('Datasets/processed/feature_added_v2')
out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)


def read_header(path: Path):
    with path.open('r', encoding='utf-8') as f:
        cols = f.readline().rstrip('\n').split('\t')
    norm = [c.split(':', 1)[0] for c in cols]
    return cols, norm


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def compute_dataset(ds: str):
    p = base / ds / f'{ds}.inter'
    raw_cols, cols = read_header(p)
    rename = dict(zip(raw_cols, cols))
    feature_cols = [c for c in cols if c.startswith(('mac_', 'mid_', 'mic_'))]

    idx_session = cols.index('session_id')
    idx_item = cols.index('item_id')
    idx_ts = cols.index('timestamp')
    idx_features = [cols.index(c) for c in feature_cols]

    sess_stats = {}
    sess_events = {}
    item_counts = Counter()
    feat_sum = {c: 0.0 for c in feature_cols}
    feat_sq = {c: 0.0 for c in feature_cols}
    feat_n = {c: 0 for c in feature_cols}

    total_rows = 0
    with p.open('r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        _ = next(reader)
        for row in reader:
            total_rows += 1
            sid = row[idx_session]
            iid = row[idx_item]
            try:
                ts = float(row[idx_ts])
            except Exception:
                ts = 0.0

            if sid in sess_stats:
                sess_stats[sid][0] += 1
                if ts > sess_stats[sid][1]:
                    sess_stats[sid][1] = ts
            else:
                sess_stats[sid] = [1, ts]

            sess_events.setdefault(sid, []).append((ts, iid))
            item_counts[iid] += 1

            for c, idx in zip(feature_cols, idx_features):
                try:
                    val = float(row[idx])
                except Exception:
                    continue
                if not math.isfinite(val):
                    continue
                feat_sum[c] += val
                feat_sq[c] += val * val
                feat_n[c] += 1

    sess_records = [(sid, v[0], v[1]) for sid, v in sess_stats.items()]
    sess_records.sort(key=lambda x: x[2])
    n_sess = len(sess_records)

    n_train = int(n_sess * 0.7)
    n_valid = int(n_sess * 0.15)
    train_ids = set(sid for sid, _, _ in sess_records[:n_train])
    valid_ids = set(sid for sid, _, _ in sess_records[n_train:n_train + n_valid])

    sess_split = {}
    for sid, _, _ in sess_records:
        if sid in train_ids:
            sess_split[sid] = 'train'
        elif sid in valid_ids:
            sess_split[sid] = 'valid'
        else:
            sess_split[sid] = 'test'

    train_item_pop = Counter()
    for sid in train_ids:
        for _, iid in sess_events.get(sid, []):
            train_item_pop[iid] += 1
    train_items = set(train_item_pop.keys())
    total_train = float(sum(train_item_pop.values())) if train_item_pop else 1.0
    pop_values = sorted(train_item_pop.values())
    q99 = quantile(pop_values, 0.99) if pop_values else 0.0

    split_rows = []
    for sp in ['train', 'valid', 'test']:
        sids = [sid for sid, _, _ in sess_records if sess_split[sid] == sp]
        lens = [sess_stats[sid][0] for sid in sids]
        split_rows.append({
            'split': sp,
            'sessions': int(len(sids)),
            'interactions': int(sum(lens)),
            'avg_session_len': float(sum(lens) / len(lens)) if lens else 0.0,
            'median_session_len': float(quantile(sorted(lens), 0.5)) if lens else 0.0,
            'p90_session_len': float(quantile(sorted(lens), 0.9)) if lens else 0.0,
        })

    def collect_target_rows(split_name):
        rows = []
        for sid, events in sess_events.items():
            if sess_split[sid] != split_name:
                continue
            ordered = sorted(events, key=lambda x: x[0])
            if not ordered:
                continue
            target_item = ordered[-1][1]
            sess_len = len(ordered)
            rows.append((target_item, sess_len - 1))
        return rows

    valid_target = collect_target_rows('valid')
    test_target = collect_target_rows('test')

    def summarize_targets(sub):
        if not sub:
            return {
                'n_sessions': 0,
                'avg_prefix_len': 0.0,
                'median_prefix_len': 0.0,
                'oov_target_rate_vs_train': 0.0,
                'target_pop_mean': 0.0,
                'target_pop_median': 0.0,
                'target_pop_share_mean': 0.0,
                'top1pct_pop_target_rate': 0.0,
            }
        target_pops = [float(train_item_pop.get(item, 0)) for item, _ in sub]
        prefix_lens = sorted([float(pref) for _, pref in sub])
        oov_rate = sum(1 for item, _ in sub if item not in train_items) / len(sub)
        top1pct_rate = sum(1 for p in target_pops if p >= q99) / len(sub)
        return {
            'n_sessions': int(len(sub)),
            'avg_prefix_len': float(sum(prefix_lens) / len(prefix_lens)),
            'median_prefix_len': float(quantile(prefix_lens, 0.5)),
            'oov_target_rate_vs_train': float(oov_rate),
            'target_pop_mean': float(sum(target_pops) / len(target_pops)),
            'target_pop_median': float(quantile(sorted(target_pops), 0.5)),
            'target_pop_share_mean': float(sum(target_pops) / len(target_pops) / total_train),
            'top1pct_pop_target_rate': float(top1pct_rate),
        }

    feat_stats = []
    for c in feature_cols:
        n = feat_n[c]
        if n <= 1:
            continue
        mean = feat_sum[c] / n
        var = max(feat_sq[c] / n - mean * mean, 0.0)
        std = var ** 0.5
        feat_stats.append((c, std))
    feat_stats.sort(key=lambda x: x[1])
    low_var = [c for c, s in feat_stats if s < 1e-3]

    return {
        'dataset': ds,
        'n_interactions': int(total_rows),
        'n_sessions': int(n_sess),
        'n_items': int(len(item_counts)),
        'session_len_mean': float(sum(v[0] for v in sess_stats.values()) / n_sess) if n_sess else 0.0,
        'session_len_median': float(quantile(sorted([v[0] for v in sess_stats.values()]), 0.5)) if n_sess else 0.0,
        'session_len_p90': float(quantile(sorted([v[0] for v in sess_stats.values()]), 0.9)) if n_sess else 0.0,
        'split_stats': split_rows,
        'valid_target_stats': summarize_targets(valid_target),
        'test_target_stats': summarize_targets(test_target),
        'feature': {
            'n_feature_cols': int(len(feature_cols)),
            'low_variance_cols_lt1e-3': int(len(low_var)),
            'low_variance_examples': low_var[:15],
            'least_variable_cols': feat_stats[:15],
            'most_variable_cols': feat_stats[-15:],
        },
    }


def main():
    all_stats = {ds: compute_dataset(ds) for ds in ['KuaiRecLargeStrictPosV2_0.2', 'lastfm0.03']}
    out_path = out_dir / 'two_dataset_stats.json'
    out_path.write_text(json.dumps(all_stats, indent=2), encoding='utf-8')
    compact = {
        k: {
            'n_interactions': v['n_interactions'],
            'n_sessions': v['n_sessions'],
            'n_items': v['n_items'],
            'valid_pop_mean': v['valid_target_stats']['target_pop_mean'],
            'test_pop_mean': v['test_target_stats']['target_pop_mean'],
            'valid_oov': v['valid_target_stats']['oov_target_rate_vs_train'],
            'test_oov': v['test_target_stats']['oov_target_rate_vs_train'],
        }
        for k, v in all_stats.items()
    }
    print(json.dumps(compact, indent=2))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
