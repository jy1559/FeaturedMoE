#!/usr/bin/env python3
"""Compare dataset sampling strategies for session-based recommendation datasets.

This script is intentionally stdlib + numpy only so it can run in the current
environment without extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DATASET_FILES = {
    "KuaiRec": Path("/workspace/jy1559/FMoE/Datasets/processed/basic/KuaiRec/KuaiRec.inter"),
    "lastfm": Path("/workspace/jy1559/FMoE/Datasets/processed/basic/lastfm/lastfm.inter"),
    "KuaiRec0.3": Path("/workspace/jy1559/FMoE/Datasets/processed/basic/KuaiRec0.3/KuaiRec0.3.inter"),
    "lastfm0.3": Path("/workspace/jy1559/FMoE/Datasets/processed/basic/lastfm0.3/lastfm0.3.inter"),
}


SESSION_LEN_BINS = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 100, np.inf], dtype=float)
TIME_BINS = 10


@dataclass
class DatasetMeta:
    dataset: str
    path: Path
    total_interactions: int
    session_ids: List[str]
    session_lens: np.ndarray
    session_users: np.ndarray
    session_starts: np.ndarray
    session_targets: np.ndarray
    session_target_pops: np.ndarray
    all_user_ids: np.ndarray
    full_user_interactions: np.ndarray
    item_counts: Dict[int, int]
    min_ts: int
    max_ts: int
    time_gap: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="KuaiRec,lastfm",
        help="Comma-separated dataset names",
    )
    parser.add_argument(
        "--ratios",
        default="0.03,0.05",
        help="Comma-separated sampling ratios",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/jy1559/FMoE/experiments/run/artifacts/analysis/sampling_study",
        help="Directory for JSON/markdown outputs",
    )
    parser.add_argument(
        "--item-pass-methods",
        default="user_stratified,user_floor_session,session_stratified",
        help="Comma-separated methods to run exact item counting for",
    )
    return parser.parse_args()


def infer_time_gap(min_ts: int, max_ts: int) -> int:
    # KuaiRec uses seconds, lastfm uses milliseconds.
    return 30 * 60 * (1000 if max_ts > 10**12 else 1)


def read_dataset_meta(dataset: str, path: Path) -> DatasetMeta:
    session_len: Dict[str, int] = defaultdict(int)
    session_user: Dict[str, int] = {}
    session_start: Dict[str, int] = {}
    session_last_ts: Dict[str, int] = {}
    session_target: Dict[str, int] = {}
    user_inter: Dict[int, int] = defaultdict(int)
    item_counts: Dict[int, int] = defaultdict(int)
    min_ts = None
    max_ts = None
    total_interactions = 0

    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name.split(":")[0]: pos for pos, name in enumerate(header)}
        required = ("session_id", "item_id", "timestamp", "user_id")
        missing = [name for name in required if name not in idx]
        if missing:
            raise ValueError(f"{dataset}: missing columns {missing} in {path}")

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            session_id = parts[idx["session_id"]]
            item_id = int(parts[idx["item_id"]])
            timestamp = int(float(parts[idx["timestamp"]]))
            user_id = int(parts[idx["user_id"]])

            total_interactions += 1
            session_len[session_id] += 1
            user_inter[user_id] += 1
            item_counts[item_id] += 1

            if session_id not in session_user:
                session_user[session_id] = user_id
                session_start[session_id] = timestamp
                session_last_ts[session_id] = timestamp
                session_target[session_id] = item_id
            else:
                if timestamp < session_start[session_id]:
                    session_start[session_id] = timestamp
                if timestamp >= session_last_ts[session_id]:
                    session_last_ts[session_id] = timestamp
                    session_target[session_id] = item_id

            if min_ts is None or timestamp < min_ts:
                min_ts = timestamp
            if max_ts is None or timestamp > max_ts:
                max_ts = timestamp

    if min_ts is None or max_ts is None:
        raise ValueError(f"{dataset}: no rows in {path}")

    session_ids = list(session_len.keys())
    session_lens = np.fromiter((session_len[s] for s in session_ids), dtype=np.int32)
    session_users = np.fromiter((session_user[s] for s in session_ids), dtype=np.int64)
    session_starts = np.fromiter((session_start[s] for s in session_ids), dtype=np.int64)
    session_targets = np.fromiter((session_target[s] for s in session_ids), dtype=np.int64)
    session_target_pops = np.fromiter((item_counts[session_target[s]] for s in session_ids), dtype=np.int32)

    all_user_ids = np.fromiter(user_inter.keys(), dtype=np.int64)
    full_user_interactions = np.fromiter(user_inter.values(), dtype=np.int32)

    return DatasetMeta(
        dataset=dataset,
        path=path,
        total_interactions=total_interactions,
        session_ids=session_ids,
        session_lens=session_lens,
        session_users=session_users,
        session_starts=session_starts,
        session_targets=session_targets,
        session_target_pops=session_target_pops,
        all_user_ids=all_user_ids,
        full_user_interactions=full_user_interactions,
        item_counts=dict(item_counts),
        min_ts=min_ts,
        max_ts=max_ts,
        time_gap=infer_time_gap(min_ts, max_ts),
    )


def safe_quantile_edges(values: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    edges = np.quantile(values, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([values.min(), values.max()], dtype=float)
    return edges


def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int32)
    return np.digitize(values, edges[1:-1], right=False).astype(np.int32)


def entropy(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jsd_from_hist(full_hist: np.ndarray, sample_hist: np.ndarray) -> float:
    p = full_hist.astype(float)
    q = sample_hist.astype(float)
    p /= p.sum() if p.sum() else 1.0
    q /= q.sum() if q.sum() else 1.0
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def hist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=bins)[0]


def session_len_hist(values: np.ndarray) -> np.ndarray:
    bins = np.concatenate(([0.0], SESSION_LEN_BINS))
    return hist(values.astype(float), bins)


def user_inter_bins(full_user_interactions: np.ndarray) -> np.ndarray:
    edges = safe_quantile_edges(
        np.log1p(full_user_interactions.astype(float)),
        qs=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    if edges[0] == edges[-1]:
        edges = np.array([edges[0], edges[0] + 1.0], dtype=float)
    return edges


def target_pop_bins(session_target_pops: np.ndarray) -> np.ndarray:
    edges = safe_quantile_edges(
        np.log1p(session_target_pops.astype(float)),
        qs=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    if edges[0] == edges[-1]:
        edges = np.array([edges[0], edges[0] + 1.0], dtype=float)
    return edges


def time_bins(meta: DatasetMeta) -> np.ndarray:
    return np.linspace(meta.min_ts, meta.max_ts, TIME_BINS + 1, dtype=float)


def take_exact_or_one(indices: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    n = len(indices)
    if n == 0:
        return indices
    take = int(round(n * ratio))
    if ratio > 0 and take == 0:
        take = 1
    if take >= n:
        return indices.copy()
    return np.sort(rng.choice(indices, size=take, replace=False))


def sample_session_stratified(meta: DatasetMeta, ratio: float, rng: np.random.Generator) -> np.ndarray:
    len_edges = safe_quantile_edges(meta.session_lens.astype(float), qs=(0.0, 0.25, 0.5, 0.75, 1.0))
    pop_edges = target_pop_bins(meta.session_target_pops)
    len_bins = digitize_with_edges(meta.session_lens.astype(float), len_edges)
    pop_bins = digitize_with_edges(np.log1p(meta.session_target_pops.astype(float)), pop_edges)
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, key in enumerate(zip(len_bins, pop_bins)):
        groups[key].append(idx)
    selected = []
    for indices in groups.values():
        arr = np.asarray(indices, dtype=np.int32)
        chosen = take_exact_or_one(arr, ratio, rng)
        selected.append(chosen)
    if not selected:
        return np.array([], dtype=np.int32)
    return np.sort(np.concatenate(selected))


def sample_user_stratified(meta: DatasetMeta, ratio: float, rng: np.random.Generator) -> np.ndarray:
    user_edges = user_inter_bins(meta.full_user_interactions)
    user_bins = digitize_with_edges(np.log1p(meta.full_user_interactions.astype(float)), user_edges)
    user_groups: Dict[int, List[int]] = defaultdict(list)
    for user_idx, bin_id in enumerate(user_bins):
        user_groups[int(bin_id)].append(user_idx)
    chosen_users = []
    for user_indices in user_groups.values():
        arr = np.asarray(user_indices, dtype=np.int32)
        chosen_users.append(take_exact_or_one(arr, ratio, rng))
    chosen_user_ids = set(meta.all_user_ids[np.sort(np.concatenate(chosen_users))].tolist())
    selected_sessions = np.flatnonzero(np.isin(meta.session_users, list(chosen_user_ids)))
    return selected_sessions.astype(np.int32)


def sample_user_floor_session(
    meta: DatasetMeta,
    ratio: float,
    rng: np.random.Generator,
    min_users: int = 100,
) -> np.ndarray:
    total_users = int(meta.all_user_ids.size)
    target_user_count = min(total_users, max(int(round(total_users * ratio)), min_users))

    user_edges = user_inter_bins(meta.full_user_interactions)
    user_bins = digitize_with_edges(np.log1p(meta.full_user_interactions.astype(float)), user_edges)
    user_groups: Dict[int, List[int]] = defaultdict(list)
    for user_idx, bin_id in enumerate(user_bins):
        user_groups[int(bin_id)].append(user_idx)

    chosen_user_indices = []
    remaining = target_user_count
    group_items = list(user_groups.items())
    for group_id, user_indices in group_items:
        arr = np.asarray(user_indices, dtype=np.int32)
        expected = int(round(len(arr) * target_user_count / total_users))
        expected = min(len(arr), max(1 if remaining > 0 else 0, expected))
        chosen = arr if expected >= len(arr) else np.sort(rng.choice(arr, size=expected, replace=False))
        chosen_user_indices.append(chosen)
        remaining -= len(chosen)

    if remaining > 0:
        already = set(np.concatenate(chosen_user_indices).tolist()) if chosen_user_indices else set()
        candidates = np.asarray([idx for idx in range(total_users) if idx not in already], dtype=np.int32)
        if candidates.size:
            extra = candidates if remaining >= candidates.size else rng.choice(candidates, size=remaining, replace=False)
            chosen_user_indices.append(np.sort(extra))

    chosen_user_ids = set(meta.all_user_ids[np.sort(np.concatenate(chosen_user_indices))].tolist())
    selected_users_mask = np.isin(meta.session_users, list(chosen_user_ids))
    candidate_sessions = np.flatnonzero(selected_users_mask)
    if candidate_sessions.size == 0:
        return candidate_sessions.astype(np.int32)

    candidate_interactions = int(meta.session_lens[candidate_sessions].sum())
    target_interactions = meta.total_interactions * ratio
    keep_ratio = min(1.0, target_interactions / max(candidate_interactions, 1))

    user_to_sessions: Dict[int, List[int]] = defaultdict(list)
    for session_idx in candidate_sessions:
        user_to_sessions[int(meta.session_users[session_idx])].append(int(session_idx))

    selected_sessions = []
    for session_indices in user_to_sessions.values():
        arr = np.asarray(session_indices, dtype=np.int32)
        take = int(round(arr.size * keep_ratio))
        if take == 0:
            take = 1
        if take >= arr.size:
            chosen = arr
        else:
            chosen = np.sort(rng.choice(arr, size=take, replace=False))
        selected_sessions.append(chosen)

    return np.sort(np.concatenate(selected_sessions)).astype(np.int32)


def sample_time_window(meta: DatasetMeta, ratio: float, mode: str) -> np.ndarray:
    order = np.argsort(meta.session_starts)
    if mode == "tail":
        order = order[::-1]
    target_interactions = meta.total_interactions * ratio
    cumsum = np.cumsum(meta.session_lens[order], dtype=np.int64)
    cutoff = int(np.searchsorted(cumsum, target_interactions, side="left")) + 1
    return np.sort(order[:cutoff].astype(np.int32))


def build_selected_summary(meta: DatasetMeta, selected_sessions: np.ndarray, method: str, ratio: float) -> Dict[str, float]:
    if selected_sessions.size == 0:
        raise ValueError(f"{meta.dataset} {method} {ratio}: selected 0 sessions")

    session_lens = meta.session_lens[selected_sessions]
    session_users = meta.session_users[selected_sessions]
    session_starts = meta.session_starts[selected_sessions]
    session_target_pops = meta.session_target_pops[selected_sessions]

    unique_users, inverse = np.unique(session_users, return_inverse=True)
    user_interactions = np.bincount(inverse, weights=session_lens.astype(np.int64)).astype(np.int64)
    user_sessions = np.bincount(inverse).astype(np.int64)

    full_user_hist = hist(np.log1p(meta.full_user_interactions.astype(float)), user_inter_bins(meta.full_user_interactions))
    sample_user_hist = hist(np.log1p(user_interactions.astype(float)), user_inter_bins(meta.full_user_interactions))
    full_target_hist = hist(np.log1p(meta.session_target_pops.astype(float)), target_pop_bins(meta.session_target_pops))
    sample_target_hist = hist(np.log1p(session_target_pops.astype(float)), target_pop_bins(meta.session_target_pops))
    full_time_hist = hist(meta.session_starts.astype(float), time_bins(meta))
    sample_time_hist = hist(session_starts.astype(float), time_bins(meta))

    sample_interactions = int(session_lens.sum())
    sample_sessions = int(selected_sessions.size)
    sample_users_n = int(unique_users.size)

    avg_session_len = float(session_lens.mean())
    avg_sessions_per_user = float(user_sessions.mean())
    avg_inter_per_user = float(user_interactions.mean())
    item_reuse_proxy = float(sample_interactions / max(sample_sessions, 1))

    preserve_score = (
        jsd_from_hist(session_len_hist(meta.session_lens), session_len_hist(session_lens))
        + jsd_from_hist(full_user_hist, sample_user_hist)
        + jsd_from_hist(full_target_hist, sample_target_hist)
        + jsd_from_hist(full_time_hist, sample_time_hist)
        + abs(avg_session_len / float(meta.session_lens.mean()) - 1.0)
        + abs(avg_sessions_per_user / float(np.mean(np.bincount(np.unique(meta.session_users, return_inverse=True)[1]))) - 1.0)
        + abs(avg_inter_per_user / float(meta.full_user_interactions.mean()) - 1.0)
    )

    return {
        "method": method,
        "ratio": ratio,
        "interactions": sample_interactions,
        "sessions": sample_sessions,
        "users": sample_users_n,
        "items": None,
        "avg_session_len": avg_session_len,
        "median_session_len": float(np.median(session_lens)),
        "avg_sessions_per_user": avg_sessions_per_user,
        "avg_interactions_per_user": avg_inter_per_user,
        "session_len_jsd": jsd_from_hist(session_len_hist(meta.session_lens), session_len_hist(session_lens)),
        "user_inter_jsd": jsd_from_hist(full_user_hist, sample_user_hist),
        "target_pop_jsd": jsd_from_hist(full_target_hist, sample_target_hist),
        "time_jsd": jsd_from_hist(full_time_hist, sample_time_hist),
        "time_coverage": float((session_starts.max() - session_starts.min()) / max(meta.max_ts - meta.min_ts, 1)),
        "preserve_score": float(preserve_score),
        "_selected_sessions": selected_sessions,
        "_selected_users": unique_users,
        "_item_counter_mode": "sessions",
    }


def build_full_baseline(meta: DatasetMeta) -> Dict[str, float]:
    full_user_sessions = np.bincount(np.unique(meta.session_users, return_inverse=True)[1]).astype(np.int64)
    return {
        "method": "full",
        "ratio": 1.0,
        "interactions": int(meta.total_interactions),
        "sessions": int(meta.session_lens.size),
        "users": int(meta.all_user_ids.size),
        "items": int(len(meta.item_counts)),
        "avg_session_len": float(meta.session_lens.mean()),
        "median_session_len": float(np.median(meta.session_lens)),
        "avg_sessions_per_user": float(full_user_sessions.mean()),
        "avg_interactions_per_user": float(meta.full_user_interactions.mean()),
        "session_len_jsd": 0.0,
        "user_inter_jsd": 0.0,
        "target_pop_jsd": 0.0,
        "time_jsd": 0.0,
        "time_coverage": 1.0,
        "preserve_score": 0.0,
    }


def analyze_interaction_resession(meta: DatasetMeta, ratio: float, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    user_rows: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    with meta.path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name.split(":")[0]: pos for pos, name in enumerate(header)}
        for line in fh:
            if rng.random() > ratio:
                continue
            parts = line.rstrip("\n").split("\t")
            user_id = int(parts[idx["user_id"]])
            item_id = int(parts[idx["item_id"]])
            timestamp = int(float(parts[idx["timestamp"]]))
            user_rows[user_id].append((timestamp, item_id))

    new_session_lens: List[int] = []
    new_session_users: List[int] = []
    new_session_starts: List[int] = []
    new_session_target_pops: List[int] = []
    item_counter: Dict[int, int] = defaultdict(int)
    total_interactions = 0

    for user_id, rows in user_rows.items():
        rows.sort(key=lambda x: x[0])
        current_items: List[int] = []
        current_start = None
        prev_ts = None
        for ts, item_id in rows:
            if prev_ts is None or ts - prev_ts <= meta.time_gap:
                if current_start is None:
                    current_start = ts
                current_items.append(item_id)
            else:
                if len(current_items) >= 5:
                    new_session_lens.append(len(current_items))
                    new_session_users.append(user_id)
                    new_session_starts.append(current_start)
                    new_session_target_pops.append(meta.item_counts.get(current_items[-1], 0))
                    total_interactions += len(current_items)
                    for item in current_items:
                        item_counter[item] += 1
                current_items = [item_id]
                current_start = ts
            prev_ts = ts
        if len(current_items) >= 5:
            new_session_lens.append(len(current_items))
            new_session_users.append(user_id)
            new_session_starts.append(current_start if current_start is not None else 0)
            new_session_target_pops.append(meta.item_counts.get(current_items[-1], 0))
            total_interactions += len(current_items)
            for item in current_items:
                item_counter[item] += 1

    if not new_session_lens:
        raise ValueError(f"{meta.dataset} interaction_resession ratio={ratio} produced no valid sessions")

    session_lens = np.asarray(new_session_lens, dtype=np.int32)
    session_users = np.asarray(new_session_users, dtype=np.int64)
    session_starts = np.asarray(new_session_starts, dtype=np.int64)
    session_target_pops = np.asarray(new_session_target_pops, dtype=np.int32)
    unique_users, inverse = np.unique(session_users, return_inverse=True)
    user_interactions = np.bincount(inverse, weights=session_lens.astype(np.int64)).astype(np.int64)
    user_sessions = np.bincount(inverse).astype(np.int64)

    full_user_hist = hist(np.log1p(meta.full_user_interactions.astype(float)), user_inter_bins(meta.full_user_interactions))
    sample_user_hist = hist(np.log1p(user_interactions.astype(float)), user_inter_bins(meta.full_user_interactions))
    full_target_hist = hist(np.log1p(meta.session_target_pops.astype(float)), target_pop_bins(meta.session_target_pops))
    sample_target_hist = hist(np.log1p(session_target_pops.astype(float)), target_pop_bins(meta.session_target_pops))
    full_time_hist = hist(meta.session_starts.astype(float), time_bins(meta))
    sample_time_hist = hist(session_starts.astype(float), time_bins(meta))

    preserve_score = (
        jsd_from_hist(session_len_hist(meta.session_lens), session_len_hist(session_lens))
        + jsd_from_hist(full_user_hist, sample_user_hist)
        + jsd_from_hist(full_target_hist, sample_target_hist)
        + jsd_from_hist(full_time_hist, sample_time_hist)
        + abs(float(session_lens.mean()) / float(meta.session_lens.mean()) - 1.0)
        + abs(float(user_sessions.mean()) / float(np.mean(np.bincount(np.unique(meta.session_users, return_inverse=True)[1]))) - 1.0)
        + abs(float(user_interactions.mean()) / float(meta.full_user_interactions.mean()) - 1.0)
    )

    return {
        "method": "interaction_resession",
        "ratio": ratio,
        "interactions": int(total_interactions),
        "sessions": int(session_lens.size),
        "users": int(unique_users.size),
        "items": int(len(item_counter)),
        "avg_session_len": float(session_lens.mean()),
        "median_session_len": float(np.median(session_lens)),
        "avg_sessions_per_user": float(user_sessions.mean()),
        "avg_interactions_per_user": float(user_interactions.mean()),
        "session_len_jsd": jsd_from_hist(session_len_hist(meta.session_lens), session_len_hist(session_lens)),
        "user_inter_jsd": jsd_from_hist(full_user_hist, sample_user_hist),
        "target_pop_jsd": jsd_from_hist(full_target_hist, sample_target_hist),
        "time_jsd": jsd_from_hist(full_time_hist, sample_time_hist),
        "time_coverage": float((session_starts.max() - session_starts.min()) / max(meta.max_ts - meta.min_ts, 1)),
        "preserve_score": float(preserve_score),
        "_item_counter_mode": "done",
    }


def count_exact_items_for_sessions(meta: DatasetMeta, selected_sessions: Iterable[int]) -> int:
    session_ids = {meta.session_ids[int(i)] for i in selected_sessions}
    items = set()
    with meta.path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name.split(":")[0]: pos for pos, name in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[idx["session_id"]] not in session_ids:
                continue
            items.add(int(parts[idx["item_id"]]))
    return len(items)


def render_markdown(dataset: str, rows: List[Dict[str, float]]) -> str:
    lines = [f"# {dataset} sampling study", ""]
    lines.append("| method | ratio | interactions | sessions | users | items | avg_session_len | avg_sessions_per_user | avg_interactions_per_user | preserve_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        items = "-" if row["items"] is None else f"{int(row['items']):,}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    f"{row['ratio']:.2f}",
                    f"{int(row['interactions']):,}",
                    f"{int(row['sessions']):,}",
                    f"{int(row['users']):,}",
                    items,
                    f"{row['avg_session_len']:.2f}",
                    f"{row['avg_sessions_per_user']:.2f}",
                    f"{row['avg_interactions_per_user']:.2f}",
                    f"{row['preserve_score']:.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Lower `preserve_score` is better.")
    return "\n".join(lines)


def clean_row(row: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> None:
    args = parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    ratios = [float(value) for value in args.ratios.split(",") if value.strip()]
    item_pass_methods = {name.strip() for name in args.item_pass_methods.split(",") if name.strip()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in datasets:
        if dataset not in DATASET_FILES:
            raise KeyError(f"Unknown dataset: {dataset}")

        print(f"[load] {dataset}")
        meta = read_dataset_meta(dataset, DATASET_FILES[dataset])
        rows: List[Dict[str, float]] = [build_full_baseline(meta)]
        base_rng = np.random.default_rng(args.seed)

        for ratio in ratios:
            session_selected = sample_session_stratified(meta, ratio, np.random.default_rng(base_rng.integers(1 << 31)))
            rows.append(build_selected_summary(meta, session_selected, "session_stratified", ratio))

            user_selected = sample_user_stratified(meta, ratio, np.random.default_rng(base_rng.integers(1 << 31)))
            rows.append(build_selected_summary(meta, user_selected, "user_stratified", ratio))

            user_floor_selected = sample_user_floor_session(meta, ratio, np.random.default_rng(base_rng.integers(1 << 31)))
            rows.append(build_selected_summary(meta, user_floor_selected, "user_floor_session", ratio))

            head_selected = sample_time_window(meta, ratio, "head")
            rows.append(build_selected_summary(meta, head_selected, "chrono_head", ratio))

            tail_selected = sample_time_window(meta, ratio, "tail")
            rows.append(build_selected_summary(meta, tail_selected, "chrono_tail", ratio))

            rows.append(analyze_interaction_resession(meta, ratio, int(base_rng.integers(1 << 31))))

        for row in rows:
            if row["method"] in item_pass_methods and row.get("_item_counter_mode") == "sessions":
                print(f"[items] {dataset} {row['method']} ratio={row['ratio']:.2f}")
                row["items"] = count_exact_items_for_sessions(meta, row["_selected_sessions"])

        cleaned = [clean_row(row) for row in rows]
        cleaned_sorted = [cleaned[0]] + sorted(cleaned[1:], key=lambda x: (x["ratio"], x["preserve_score"], x["method"]))
        all_results[dataset] = cleaned_sorted

        markdown = render_markdown(dataset, cleaned_sorted)
        (output_dir / f"{dataset}_sampling_study.json").write_text(json.dumps(cleaned_sorted, indent=2))
        (output_dir / f"{dataset}_sampling_study.md").write_text(markdown)
        print(markdown)

    (output_dir / "sampling_study_all.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
