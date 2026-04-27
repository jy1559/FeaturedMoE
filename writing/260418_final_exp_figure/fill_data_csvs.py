#!/usr/bin/env python3
"""
fill_data_csvs.py
Populate data/*.csv files with real experiment values from ablation logs.
Handles Q2 / Q3 / Q4.  Preserves draft_fill rows for missing datasets.

Run from any directory:
  /workspace/FeaturedMoE/.venv/bin/python3 \
      /workspace/FeaturedMoE/writing/260418_final_exp_figure/fill_data_csvs.py
"""

import json, glob, re
from pathlib import Path
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ABLATION  = Path('/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment_ablation')
SPECIAL   = ABLATION / 'special'
DATA_DIR  = Path('/workspace/FeaturedMoE/writing/260418_final_exp_figure/data')

# ── dataset name normalisation ──────────────────────────────────────────────
DATASET_NORM = {
    'beauty':                    'Beauty',
    'kuaireclargestrictposv2_0.2': 'KuaiRec',
    'kuaireclargestrictposv2_0_2': 'KuaiRec',  # slug form
}

def norm_dataset(raw: str) -> str:
    return DATASET_NORM.get(raw.lower().replace('.', '_'), raw)


# ── load summary CSVs ───────────────────────────────────────────────────────
def load_summaries(paths):
    dfs = []
    for p in paths:
        p = Path(p)
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['status'] == 'ok'].copy()
    return df


def best_seed_per_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the row with highest valid_score per (dataset, setting_key)."""
    df = df.sort_values(['dataset','setting_key','valid_score','best_valid_mrr20'],
                        ascending=[True,True,False,False])
    return df.drop_duplicates(['dataset','setting_key'], keep='first').reset_index(drop=True)


# ── load special_metrics JSON index ─────────────────────────────────────────
def build_special_index(glob_pattern: str) -> dict:
    """
    Returns two dicts:
      - by_run_phase: {run_phase -> best_entry_with_data}
      - by_variant_key: {(dataset_lower, variant_lower) -> best_entry_with_data}
    Prefers entries where test_special_metrics.overall_seen_target.ndcg@20 is not None.
    """
    by_rp = {}   # run_phase → entry
    by_vk = {}   # (dataset, setting_key substring) → entry
    for fp in sorted(glob.glob(glob_pattern, recursive=True)):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        rp = d.get('run_phase', Path(fp).stem)
        ts = d.get('test_special_metrics', {})
        has_data = bool(ts.get('overall_seen_target', {}).get('ndcg@20'))

        # run_phase index: prefer entry with real data
        if rp not in by_rp or has_data:
            by_rp[rp] = d

        # variant-level fallback: derive key from run_phase
        # e.g. Q2_BEAUTY_ROUTER_HIDDEN_ONLY_R02_S3 → dataset=beauty, variant_stub=router_hidden_only
        m = re.match(r'Q\d+_([^_]+(?:_\d+_\d+)?)_(.+?)_R\d+', rp, re.IGNORECASE)
        if m:
            ds = m.group(1).lower()
            var = m.group(2).lower()
            vk = (ds, var)
            if vk not in by_vk or has_data:
                by_vk[vk] = d

    return by_rp, by_vk


def get_seen_metrics(special_idx_pair, run_phase: str,
                     dataset: str = None, setting_key: str = None,
                     split: str = 'test') -> dict:
    """Return {ndcg20, hr10, mrr20, ndcg20_seen, hr10_seen, ndcg20_unseen} for a run_phase.
    Falls back to variant-level match if run_phase has no data."""
    by_rp, by_vk = special_idx_pair

    entry = by_rp.get(run_phase, {})
    key = 'test_special_metrics' if split == 'test' else 'best_valid_special_metrics'
    sm = entry.get(key, {})
    ost = sm.get('overall_seen_target', {})

    # fallback to variant-level if no seen target data
    if not ost.get('ndcg@20') and dataset and setting_key:
        ds_slug = re.sub(r'[^a-z0-9]', '_', dataset.lower())
        var_slug = setting_key.lower()
        for vk in [(ds_slug, var_slug),
                   (ds_slug.replace('kuaireclargestrictposv2_0_2','kuaireclargestrictposv2'),
                    var_slug)]:
            fallback = by_vk.get(vk, {})
            sm2 = fallback.get(key, {})
            if sm2.get('overall_seen_target', {}).get('ndcg@20'):
                sm = sm2
                ost = sm.get('overall_seen_target', {})
                break

    oun = sm.get('overall_unseen_target', {})
    ov  = sm.get('overall', {})
    slices = sm.get('slices', {})
    return {
        'ndcg20':         ov.get('ndcg@20'),
        'hr10':           ov.get('hit@10'),
        'mrr20':          ov.get('mrr@20'),
        'ndcg20_seen':    ost.get('ndcg@20'),
        'hr10_seen':      ost.get('hit@10'),
        'mrr20_seen':     ost.get('mrr@20'),
        'ndcg20_unseen':  oun.get('ndcg@20'),
        'hr10_unseen':    oun.get('hit@10'),
        'slices':         slices,
    }


# ── row builder helpers ──────────────────────────────────────────────────────
def make_row(paper_section, panel, extra_id_cols: dict,
             variant_or_model, metric, cutoff, value,
             run_id, source_path, note='real_export'):
    base = dict(
        paper_section=paper_section, panel=panel,
        metric=metric, cutoff=cutoff, value=round(value, 6) if value is not None else None,
        split='test', selection_rule='best_valid_seen_target',
        run_id=run_id, source_path=str(source_path),
        notes=note, status='real',
    )
    base.update(extra_id_cols)
    base['variant_or_model'] = variant_or_model
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Q2  — routing control quality
# ─────────────────────────────────────────────────────────────────────────────
Q2_VARIANT_MAP = {
    'shared_ffn':             'shared_ffn',
    'router_hidden_only':     'hidden_only',
    'router_hidden_plus_feature': 'mixed_hidden_behavior',
    'router_feature_only':    'behavior_guided',
    # route_rec_full is omitted from the panel
}

def fill_q2_quality():
    print('== Q2 routing quality ==')
    summary_paths = [
        ABLATION / 'q2' / 'summary.csv',
        ABLATION / 'q2' / 'summary.tmp.693745.csv',
        ABLATION / 'q2' / 'summary.tmp.695584.csv',
    ]
    df = load_summaries(summary_paths)
    df = df[df['setting_key'].isin(Q2_VARIANT_MAP)]
    best = best_seed_per_variant(df)

    special_idx = build_special_index(
        str(SPECIAL / 'q2_routing_control' / '**' / '*_special_metrics.json')
    )

    existing = pd.read_csv(DATA_DIR / '02_routing_quality.csv')
    rows_new = []

    for _, r in best.iterrows():
        ds_norm   = norm_dataset(r['dataset'])
        variant   = Q2_VARIANT_MAP[r['setting_key']]
        run_phase = r['job_id']       # e.g. Q2_BEAUTY_ROUTER_FEATURE_ONLY_R01_S3
        sm        = get_seen_metrics(special_idx, run_phase,
                                     dataset=r['dataset'], setting_key=r['setting_key'])

        ndcg = sm.get('ndcg20_seen')
        hr10 = sm.get('hr10_seen')
        if ndcg is None or hr10 is None:
            print(f'  WARN: no special_metrics for {run_phase}')
            continue

        extra = {'dataset': ds_norm}
        rows_new.append(make_row('Q2','Q2_quality', extra, variant, 'NDCG', 20, ndcg, run_phase,
                                 f'special/q2_routing_control/{ds_norm}/{variant}'))
        rows_new.append(make_row('Q2','Q2_quality', extra, variant, 'HR',   10, hr10, run_phase,
                                 f'special/q2_routing_control/{ds_norm}/{variant}'))
        print(f'  {ds_norm:10s}  {variant:30s}  NDCG@20={ndcg:.4f}  HR@10={hr10:.4f}')

    if not rows_new:
        print('  No rows extracted — keeping existing file.')
        return

    new_df = pd.DataFrame(rows_new)
    REF_COLS = ['paper_section','panel','dataset','variant_or_model','metric','cutoff','value',
                'split','selection_rule','run_id','source_path','notes','status']
    # keep draft_fill for datasets not covered
    covered = set(new_df['dataset'].unique())
    fallback = existing[
        ~existing['dataset'].isin(covered) |
        existing['status'].str.startswith('placeholder')
    ]
    out = pd.concat([new_df, fallback], ignore_index=True)
    # sort: dataset, variant order, metric
    variant_order = list(Q2_VARIANT_MAP.values())
    out['_vo'] = pd.Categorical(out.get('variant_or_model', out.get('variant_or_model','')),
                                categories=variant_order, ordered=True)
    out = out.sort_values(['dataset', '_vo', 'metric', 'cutoff']).drop(columns='_vo')
    for c in REF_COLS:
        if c not in out.columns: out[c] = None
    out = out[[c for c in REF_COLS if c in out.columns]]
    out.to_csv(DATA_DIR / '02_routing_quality.csv', index=False)
    print(f'  Wrote {len(out)} rows to 02_routing_quality.csv')


# ─────────────────────────────────────────────────────────────────────────────
# Q3  — stage structure (stage_removal  +  dense_vs_staged)
# ─────────────────────────────────────────────────────────────────────────────
Q3_VARIANTS_ALL = [
    'full_three_stage',
    'remove_macro', 'remove_mid', 'remove_micro',
    'single_stage_macro', 'single_stage_mid', 'single_stage_micro',
    'dense_full_only',
]

def fill_q3():
    print('== Q3 stage structure ==')
    df = load_summaries([ABLATION / 'q3' / 'summary.csv'])
    best = best_seed_per_variant(df)

    special_idx = build_special_index(
        str(SPECIAL / 'q3_stage_structure' / '**' / '*_special_metrics.json')
    )

    # ── 03_stage_removal.csv ─────────────────────────────────────────────────
    # single dataset (Beauty primary), ablation_group = setting_key
    REMOVAL_VARIANTS = ['full_three_stage', 'remove_macro', 'remove_mid', 'remove_micro']

    existing_rem = pd.read_csv(DATA_DIR / '03_stage_removal.csv')
    rows_rem = []
    for ds_grp, grp in best.groupby('dataset'):
        ds_norm = norm_dataset(ds_grp)
        for _, r in grp[grp['setting_key'].isin(REMOVAL_VARIANTS)].iterrows():
            sm = get_seen_metrics(special_idx, r['job_id'],
                                  dataset=r['dataset'], setting_key=r['setting_key'])
            ndcg = sm.get('ndcg20_seen')
            hr10 = sm.get('hr10_seen')
            if ndcg is None:
                print(f'  WARN Q3-rem: no special for {r["job_id"]}')
                continue
            abbrev = r['setting_key']
            extra = {'ablation_group': abbrev, 'dataset': ds_norm}
            rows_rem.append(make_row('Q3','Q3_stage_removal', extra, 'RouteRec','NDCG',20,ndcg,
                                     r['job_id'], f'q3/summary.csv'))
            rows_rem.append(make_row('Q3','Q3_stage_removal', extra, 'RouteRec','HR',10,hr10,
                                     r['job_id'], f'q3/summary.csv'))
            print(f'  stage_rem  {ds_norm:8s}  {abbrev:20s}  NDCG@20={ndcg:.4f}')

    if rows_rem:
        new_rem = pd.DataFrame(rows_rem)
        # drop columns not in original if any
        ref_cols = existing_rem.columns.tolist()
        for c in new_rem.columns:
            if c not in ref_cols:
                new_rem = new_rem.drop(columns=c)
        for c in ref_cols:
            if c not in new_rem.columns:
                new_rem[c] = None
        new_rem = new_rem[ref_cols]
        covered_ds = set(new_rem.get('dataset', pd.Series()).unique()) if 'dataset' in new_rem.columns else set()
        # drop existing rows that we now have real values for
        if 'dataset' in existing_rem.columns:
            fallback_rem = existing_rem[~existing_rem['dataset'].isin(covered_ds)]
        else:
            fallback_rem = pd.DataFrame(columns=ref_cols)
        out_rem = pd.concat([new_rem, fallback_rem], ignore_index=True)
        out_rem.to_csv(DATA_DIR / '03_stage_removal.csv', index=False)
        print(f'  Wrote {len(out_rem)} rows to 03_stage_removal.csv')

    # ── 03_dense_vs_staged.csv ───────────────────────────────────────────────
    # layout_variant: dense_ffn, best_single_stage, best_two_stage, three_stage
    existing_dvs = pd.read_csv(DATA_DIR / '03_dense_vs_staged.csv')
    rows_dvs = []
    for ds_grp, grp in best.groupby('dataset'):
        ds_norm = norm_dataset(ds_grp)

        def get_sm_row(setting_key):
            r = grp[grp['setting_key'] == setting_key]
            if r.empty: return {}
            return get_seen_metrics(special_idx, r.iloc[0]['job_id'],
                                    dataset=ds_grp, setting_key=setting_key)

        dense = get_sm_row('dense_full_only')
        full3 = get_sm_row('full_three_stage')
        # best_single = best among single_stage_*
        single_keys = ['single_stage_macro','single_stage_mid','single_stage_micro']
        single_cands = []
        for sk in single_keys:
            sm = get_sm_row(sk)
            if sm.get('ndcg20_seen') is not None:
                single_cands.append(sm)
        best_single = max(single_cands, key=lambda x: x.get('ndcg20_seen', 0)) if single_cands else {}
        # best_two_stage = best among remove_* (one stage removed = two-stage)
        two_keys = ['remove_macro','remove_mid','remove_micro']
        two_cands = []
        for sk in two_keys:
            sm = get_sm_row(sk)
            if sm.get('ndcg20_seen') is not None:
                two_cands.append(sm)
        best_two = max(two_cands, key=lambda x: x.get('ndcg20_seen', 0)) if two_cands else {}

        layout_map = {
            'dense_ffn':         dense,
            'best_single_stage': best_single,
            'best_two_stage':    best_two,
            'three_stage':       full3,
        }
        for layout, sm in layout_map.items():
            ndcg = sm.get('ndcg20_seen')
            hr10 = sm.get('hr10_seen')
            if ndcg is None: continue
            extra = {'layout_variant': layout, 'dataset': ds_norm}
            rows_dvs.append(make_row('Q3','Q3_dense_vs_staged',extra,'RouteRec','NDCG',20,ndcg,'derived',
                                     'q3/summary.csv'))
            rows_dvs.append(make_row('Q3','Q3_dense_vs_staged',extra,'RouteRec','HR',10,hr10,'derived',
                                     'q3/summary.csv'))
            print(f'  stage_dvs  {ds_norm:8s}  {layout:20s}  NDCG@20={ndcg:.4f}')

    if rows_dvs:
        ref_cols = existing_dvs.columns.tolist()
        new_dvs = pd.DataFrame(rows_dvs)
        for c in ref_cols:
            if c not in new_dvs.columns: new_dvs[c] = None
        new_dvs = new_dvs[[c for c in ref_cols if c in new_dvs.columns]]
        covered_ds = set(new_dvs.get('dataset', pd.Series()).unique()) if 'dataset' in new_dvs.columns else set()
        if 'dataset' in existing_dvs.columns:
            fallback_dvs = existing_dvs[~existing_dvs['dataset'].isin(covered_ds)]
        else:
            fallback_dvs = pd.DataFrame(columns=ref_cols)
        out_dvs = pd.concat([new_dvs, fallback_dvs], ignore_index=True)
        out_dvs.to_csv(DATA_DIR / '03_dense_vs_staged.csv', index=False)
        print(f'  Wrote {len(out_dvs)} rows to 03_dense_vs_staged.csv')


# ─────────────────────────────────────────────────────────────────────────────
# Q4  — lightweight cues
# ─────────────────────────────────────────────────────────────────────────────
Q4_VARIANT_MAP = {
    'full':            'full',
    'remove_category': 'remove_category',
    'remove_time':     'remove_time',
    'sequence_only':   'sequence_only_portable',
}

def fill_q4():
    print('== Q4 cue ablation ==')
    df = load_summaries([ABLATION / 'q4' / 'summary.csv'])
    df = df[df['setting_key'].isin(Q4_VARIANT_MAP)]
    best = best_seed_per_variant(df)

    special_idx = build_special_index(
        str(SPECIAL / 'q4_cue_ablation' / '**' / '*_special_metrics.json')
    )

    existing = pd.read_csv(DATA_DIR / '04_cue_scores.csv')
    rows_new = []

    for _, r in best.iterrows():
        ds_norm = norm_dataset(r['dataset'])
        variant = Q4_VARIANT_MAP[r['setting_key']]
        sm = get_seen_metrics(special_idx, r['job_id'],
                              dataset=r['dataset'], setting_key=r['setting_key'])
        ndcg = sm.get('ndcg20_seen')
        hr10 = sm.get('hr10_seen')
        if ndcg is None:
            print(f'  WARN Q4: no special for {r["job_id"]}')
            continue
        extra = {'dataset': ds_norm}
        rows_new.append(make_row('Q4','Q4_cue_scores',extra,variant,'NDCG',20,ndcg,
                                  r['job_id'],'q4/summary.csv'))
        rows_new.append(make_row('Q4','Q4_cue_scores',extra,variant,'HR',10,hr10,
                                  r['job_id'],'q4/summary.csv'))
        print(f'  {ds_norm:10s}  {variant:30s}  NDCG@20={ndcg:.4f}  HR@10={hr10:.4f}')

    if rows_new:
        ref_cols = existing.columns.tolist()
        new_df = pd.DataFrame(rows_new)
        for c in ref_cols:
            if c not in new_df.columns: new_df[c] = None
        new_df = new_df[[c for c in ref_cols if c in new_df.columns]]
        covered = set(new_df['dataset'].unique())
        fallback = existing[~existing['dataset'].isin(covered)]
        out = pd.concat([new_df, fallback], ignore_index=True)
        out.to_csv(DATA_DIR / '04_cue_scores.csv', index=False)
        print(f'  Wrote {len(out)} rows to 04_cue_scores.csv')


# ─────────────────────────────────────────────────────────────────────────────
# Q2  — routing_consistency  (seen vs unseen per variant)
# ─────────────────────────────────────────────────────────────────────────────
def fill_q2_consistency():
    print('== Q2 routing consistency ==')
    summary_paths = [
        ABLATION / 'q2' / 'summary.csv',
        ABLATION / 'q2' / 'summary.tmp.693745.csv',
        ABLATION / 'q2' / 'summary.tmp.695584.csv',
    ]
    df = load_summaries(summary_paths)
    df = df[df['setting_key'].isin(Q2_VARIANT_MAP)]
    best = best_seed_per_variant(df)

    special_idx = build_special_index(
        str(SPECIAL / 'q2_routing_control' / '**' / '*_special_metrics.json')
    )

    existing = pd.read_csv(DATA_DIR / '02_routing_consistency.csv')
    print('  02_routing_consistency.csv columns:', existing.columns.tolist())
    rows_new = []

    for _, r in best.iterrows():
        ds_norm = norm_dataset(r['dataset'])
        variant = Q2_VARIANT_MAP[r['setting_key']]
        sm = get_seen_metrics(special_idx, r['job_id'],
                              dataset=r['dataset'], setting_key=r['setting_key'])
        ndcg_s = sm.get('ndcg20_seen')
        ndcg_u = sm.get('ndcg20_unseen')
        ndcg_o = sm.get('ndcg20')
        if ndcg_s is None:
            continue
        extra = {'dataset': ds_norm}
        for split_name, val in [('seen', ndcg_s), ('unseen', ndcg_u), ('overall', ndcg_o)]:
            if val is None: continue
            rows_new.append(make_row('Q2','Q2_consistency',extra,variant,'NDCG',20,val,
                                      r['job_id'],'q2/special',
                                      note=f'seen_target_split={split_name}'))
            rows_new[-1]['target_split'] = split_name

    if rows_new:
        out = pd.DataFrame(rows_new)
        # merge with existing schema
        ref_cols = existing.columns.tolist()
        for c in ref_cols:
            if c not in out.columns: out[c] = None
        extra_cols = [c for c in out.columns if c not in ref_cols]
        out_final = out[[c for c in ref_cols if c in out.columns]].copy()
        out_final.to_csv(DATA_DIR / '02_routing_consistency.csv', index=False)
        print(f'  Wrote {len(out_final)} rows to 02_routing_consistency.csv')


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fill_q2_quality()
    print()
    fill_q3()
    print()
    fill_q4()
    print()
    fill_q2_consistency()
    print()
    print('Done.')
