#!/usr/bin/env python3
"""Create fixed session-level train/valid/test split files by session start time.

Input:
  {processed_root}/{dataset}/{dataset}.inter
Output (in-place):
  {dataset}.train.inter
  {dataset}.valid.inter
  {dataset}.test.inter
  {dataset}.session_split_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict


DEFAULT_PROCESSED_ROOT = Path("/workspace/jy1559/FMoE/Datasets/processed/feature_added_v3")


def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def discover_datasets(processed_root: Path) -> list[str]:
    out: list[str] = []
    if not processed_root.exists():
        return out
    for p in sorted(processed_root.iterdir()):
        if not p.is_dir():
            continue
        ds = p.name
        if (p / f"{ds}.inter").exists() and (p / f"{ds}.item").exists():
            out.append(ds)
    return out


def parse_ratios(raw: str) -> tuple[float, float, float]:
    toks = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(toks) != 3:
        raise ValueError("--ratios must be 3 comma-separated floats, e.g. 0.7,0.15,0.15")
    vals = tuple(float(x) for x in toks)
    if any(v <= 0.0 for v in vals):
        raise ValueError("--ratios values must be > 0")
    s = sum(vals)
    if not math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"--ratios must sum to 1.0, got sum={s}")
    return vals  # type: ignore[return-value]


def parse_dataset_list(raw: str, processed_root: Path) -> list[str]:
    if str(raw).strip():
        return [x.strip() for x in str(raw).split(",") if x.strip()]
    return discover_datasets(processed_root)


@dataclass
class SessionInfo:
    start_ts: float
    first_row_idx: int


@dataclass
class SessionProfile:
    start_ts: float
    first_row_idx: int
    length: int
    target_item: str
    target_in_prefix: bool
    items: list[str]


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_header(inter_path: Path) -> list[str]:
    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        row = next(reader, None)
    if not row:
        raise RuntimeError(f"Empty inter file: {inter_path}")
    return [str(x) for x in row]


def pick_column(header: list[str], plain_name: str) -> str:
    for col in header:
        if strip_type(col) == plain_name:
            return col
    raise KeyError(f"Missing required column `{plain_name}` in header: {header}")


def gather_sessions(inter_path: Path, session_col: str, ts_col: str) -> Dict[str, SessionInfo]:
    sessions: Dict[str, SessionInfo] = {}
    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row_idx, row in enumerate(reader):
            sid = str(row[session_col])
            ts = _to_float(str(row[ts_col]))
            prev = sessions.get(sid)
            if prev is None:
                sessions[sid] = SessionInfo(start_ts=ts, first_row_idx=row_idx)
                continue
            prev_ts = prev.start_ts
            if (math.isnan(prev_ts) and not math.isnan(ts)) or (not math.isnan(ts) and ts < prev_ts):
                sessions[sid] = SessionInfo(start_ts=ts, first_row_idx=prev.first_row_idx)
    return sessions


def gather_session_profiles(
    inter_path: Path,
    session_col: str,
    ts_col: str,
    item_col: str,
) -> Dict[str, SessionProfile]:
    rows_by_session: Dict[str, list[tuple[float, int, str]]] = {}
    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row_idx, row in enumerate(reader):
            sid = str(row[session_col])
            ts = _to_float(str(row[ts_col]))
            item = str(row[item_col])
            rows_by_session.setdefault(sid, []).append((ts, row_idx, item))

    out: Dict[str, SessionProfile] = {}
    for sid, events in rows_by_session.items():
        ordered = sorted(
            events,
            key=lambda x: (
                math.inf if math.isnan(x[0]) else x[0],
                x[1],
            ),
        )
        items = [x[2] for x in ordered]
        target_item = items[-1]
        target_in_prefix = target_item in items[:-1]
        finite_ts = [x[0] for x in ordered if not math.isnan(x[0])]
        start_ts = min(finite_ts) if finite_ts else float("nan")
        first_row_idx = min(x[1] for x in ordered)
        out[sid] = SessionProfile(
            start_ts=start_ts,
            first_row_idx=first_row_idx,
            length=len(items),
            target_item=target_item,
            target_in_prefix=target_in_prefix,
            items=items,
        )
    return out


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    train = int(math.floor(total * ratios[0]))
    valid = int(math.floor(total * ratios[1]))
    test = total - train - valid
    if total >= 3:
        # Keep non-empty splits when possible.
        if train <= 0:
            train = 1
            test = max(0, test - 1)
        if valid <= 0:
            valid = 1
            test = max(0, test - 1)
        if test <= 0:
            test = 1
            if train > valid and train > 1:
                train -= 1
            elif valid > 1:
                valid -= 1
            elif train > 1:
                train -= 1
    # final normalization
    if train + valid + test != total:
        test = total - train - valid
    return train, valid, test


def assign_sessions(
    sessions: Dict[str, SessionInfo],
    ratios: tuple[float, float, float],
) -> tuple[Dict[str, str], dict]:
    order = sorted(
        sessions.items(),
        key=lambda kv: (
            math.inf if math.isnan(kv[1].start_ts) else kv[1].start_ts,
            kv[1].first_row_idx,
            kv[0],
        ),
    )
    total = len(order)
    train_n, valid_n, test_n = _split_counts(total, ratios)

    assignment: Dict[str, str] = {}
    for i, (sid, _) in enumerate(order):
        if i < train_n:
            assignment[sid] = "train"
        elif i < train_n + valid_n:
            assignment[sid] = "valid"
        else:
            assignment[sid] = "test"

    return assignment, {
        "total_sessions": total,
        "train_sessions": train_n,
        "valid_sessions": valid_n,
        "test_sessions": test_n,
    }


def _session_len_bin(length: int) -> str:
    if length <= 7:
        return "len_le_7"
    if length <= 12:
        return "len_8_12"
    return "len_ge_13"


def _pop_bin(train_count: int) -> str:
    if train_count <= 0:
        return "cold_0"
    if train_count <= 5:
        return "rare_1_5"
    return "seen_6p"


def assign_sessions_tail_stratified(
    session_profiles: Dict[str, SessionProfile],
    ratios: tuple[float, float, float],
) -> tuple[Dict[str, str], dict]:
    order = sorted(
        session_profiles.items(),
        key=lambda kv: (
            math.inf if math.isnan(kv[1].start_ts) else kv[1].start_ts,
            kv[1].first_row_idx,
            kv[0],
        ),
    )
    total = len(order)
    train_n, valid_n, test_n = _split_counts(total, ratios)
    tail_n = total - train_n
    valid_ratio_in_tail = (valid_n / tail_n) if tail_n > 0 else 0.0

    assignment: Dict[str, str] = {}
    train_sids: list[str] = []
    tail_sids: list[str] = []
    for idx, (sid, _) in enumerate(order):
        if idx < train_n:
            assignment[sid] = "train"
            train_sids.append(sid)
        else:
            tail_sids.append(sid)

    train_item_counts: Dict[str, int] = {}
    for sid in train_sids:
        prof = session_profiles[sid]
        for item in prof.items:
            train_item_counts[item] = train_item_counts.get(item, 0) + 1

    groups: Dict[tuple[str, str, str], list[str]] = {}
    for sid in tail_sids:
        prof = session_profiles[sid]
        repeat_bin = "repeat_yes" if prof.target_in_prefix else "repeat_no"
        pop_bin = _pop_bin(int(train_item_counts.get(prof.target_item, 0)))
        len_bin = _session_len_bin(int(prof.length))
        key = (repeat_bin, pop_bin, len_bin)
        groups.setdefault(key, []).append(sid)

    for sids in groups.values():
        sids.sort(
            key=lambda sid: (
                math.inf if math.isnan(session_profiles[sid].start_ts) else session_profiles[sid].start_ts,
                session_profiles[sid].first_row_idx,
                sid,
            )
        )

    desired_valid_by_group: Dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    base_sum = 0
    for k, sids in groups.items():
        exact = len(sids) * valid_ratio_in_tail
        base = int(math.floor(exact))
        desired_valid_by_group[k] = base
        base_sum += base
        remainders.append((exact - base, k))
    need_extra = max(0, valid_n - base_sum)
    for _rem, k in sorted(remainders, key=lambda x: x[0], reverse=True):
        if need_extra <= 0:
            break
        if desired_valid_by_group[k] < len(groups[k]):
            desired_valid_by_group[k] += 1
            need_extra -= 1

    for k, sids in groups.items():
        m = len(sids)
        q = max(0, min(desired_valid_by_group.get(k, 0), m))
        valid_assigned = 0
        for i, sid in enumerate(sids):
            should_be_valid = ((i + 1) * q // m) > (i * q // m)
            if should_be_valid and valid_assigned < q:
                assignment[sid] = "valid"
                valid_assigned += 1
            else:
                assignment[sid] = "test"

    valid_actual = sum(1 for sid in tail_sids if assignment[sid] == "valid")
    if valid_actual != valid_n and tail_sids:
        target_side = "valid" if valid_actual < valid_n else "test"
        source_side = "test" if target_side == "valid" else "valid"
        delta = abs(valid_n - valid_actual)
        tail_order = sorted(
            tail_sids,
            key=lambda sid: (
                math.inf if math.isnan(session_profiles[sid].start_ts) else session_profiles[sid].start_ts,
                session_profiles[sid].first_row_idx,
                sid,
            ),
        )
        for sid in tail_order:
            if delta <= 0:
                break
            if assignment[sid] != source_side:
                continue
            assignment[sid] = target_side
            delta -= 1

    repeat_stats = {"train": {"yes": 0, "no": 0}, "valid": {"yes": 0, "no": 0}, "test": {"yes": 0, "no": 0}}
    pop_stats = {
        "valid": {"cold_0": 0, "rare_1_5": 0, "seen_6p": 0},
        "test": {"cold_0": 0, "rare_1_5": 0, "seen_6p": 0},
    }
    for sid, split in assignment.items():
        prof = session_profiles[sid]
        rep_key = "yes" if prof.target_in_prefix else "no"
        repeat_stats[split][rep_key] += 1
        if split in {"valid", "test"}:
            pop_stats[split][_pop_bin(int(train_item_counts.get(prof.target_item, 0)))] += 1

    return assignment, {
        "total_sessions": total,
        "train_sessions": train_n,
        "valid_sessions": valid_n,
        "test_sessions": test_n,
        "split_strategy": "tail_stratified",
        "tail_sessions": tail_n,
        "tail_valid_ratio": valid_ratio_in_tail,
        "repeat_target_stats": repeat_stats,
        "target_pop_stats_valid_test": pop_stats,
    }


def write_split_files(
    *,
    inter_path: Path,
    header: list[str],
    session_col: str,
    assignment: Dict[str, str],
    out_train: Path,
    out_valid: Path,
    out_test: Path,
    overwrite: bool,
) -> dict:
    for p in (out_train, out_valid, out_test):
        if p.exists() and not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {p}")

    tmp_train = NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(out_train.parent), suffix=".tmp")
    tmp_valid = NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(out_valid.parent), suffix=".tmp")
    tmp_test = NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(out_test.parent), suffix=".tmp")

    try:
        writers = {
            "train": csv.writer(tmp_train, delimiter="\t"),
            "valid": csv.writer(tmp_valid, delimiter="\t"),
            "test": csv.writer(tmp_test, delimiter="\t"),
        }
        for w in writers.values():
            w.writerow(header)

        rows = {"train": 0, "valid": 0, "test": 0}
        sessions = {"train": set(), "valid": set(), "test": set()}
        ts_bounds = {
            "train": {"min": math.inf, "max": -math.inf},
            "valid": {"min": math.inf, "max": -math.inf},
            "test": {"min": math.inf, "max": -math.inf},
        }
        ts_col = next(col for col in header if strip_type(col) == "timestamp")

        with inter_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                sid = str(row[session_col])
                split = assignment[sid]
                writers[split].writerow([row[col] for col in header])
                rows[split] += 1
                sessions[split].add(sid)
                ts = _to_float(str(row[ts_col]))
                if not math.isnan(ts):
                    ts_bounds[split]["min"] = min(ts_bounds[split]["min"], ts)
                    ts_bounds[split]["max"] = max(ts_bounds[split]["max"], ts)

        for p in (tmp_train, tmp_valid, tmp_test):
            p.flush()
            os.fsync(p.fileno())
            p.close()
        os.replace(tmp_train.name, out_train)
        os.replace(tmp_valid.name, out_valid)
        os.replace(tmp_test.name, out_test)

        overlap_train_valid = len(sessions["train"] & sessions["valid"])
        overlap_train_test = len(sessions["train"] & sessions["test"])
        overlap_valid_test = len(sessions["valid"] & sessions["test"])

        def _fmt_bound(v: float) -> float | None:
            if v in (math.inf, -math.inf) or math.isnan(v):
                return None
            return float(v)

        return {
            "rows": {k: int(v) for k, v in rows.items()},
            "sessions": {k: int(len(v)) for k, v in sessions.items()},
            "session_overlap": {
                "train_valid": overlap_train_valid,
                "train_test": overlap_train_test,
                "valid_test": overlap_valid_test,
            },
            "timestamp_bounds": {
                k: {"min": _fmt_bound(v["min"]), "max": _fmt_bound(v["max"])}
                for k, v in ts_bounds.items()
            },
        }
    except Exception:
        for p in (tmp_train, tmp_valid, tmp_test):
            try:
                p.close()
            except Exception:
                pass
            try:
                os.unlink(p.name)
            except Exception:
                pass
        raise


def verify_temporal_session_order(sessions: Dict[str, SessionInfo], assignment: Dict[str, str]) -> dict:
    split_starts = {"train": [], "valid": [], "test": []}
    for sid, info in sessions.items():
        split_starts[assignment[sid]].append(info.start_ts)

    def _finite(vals: list[float]) -> list[float]:
        return [v for v in vals if not math.isnan(v)]

    train_vals = _finite(split_starts["train"])
    valid_vals = _finite(split_starts["valid"])
    test_vals = _finite(split_starts["test"])
    train_max = max(train_vals) if train_vals else None
    valid_min = min(valid_vals) if valid_vals else None
    valid_max = max(valid_vals) if valid_vals else None
    test_min = min(test_vals) if test_vals else None

    train_before_valid = True if (train_max is None or valid_min is None) else bool(train_max <= valid_min)
    valid_before_test = True if (valid_max is None or test_min is None) else bool(valid_max <= test_min)

    return {
        "train_max_session_start": train_max,
        "valid_min_session_start": valid_min,
        "valid_max_session_start": valid_max,
        "test_min_session_start": test_min,
        "train_before_valid": train_before_valid,
        "valid_before_test": valid_before_test,
    }


def process_dataset(
    *,
    processed_root: Path,
    dataset: str,
    ratios: tuple[float, float, float],
    strategy: str,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    ds_dir = processed_root / dataset
    inter_path = ds_dir / f"{dataset}.inter"
    item_path = ds_dir / f"{dataset}.item"
    if not inter_path.exists():
        raise FileNotFoundError(f"Missing inter file: {inter_path}")
    if not item_path.exists():
        raise FileNotFoundError(f"Missing item file: {item_path}")

    header = read_header(inter_path)
    session_col = pick_column(header, "session_id")
    ts_col = pick_column(header, "timestamp")
    item_col = pick_column(header, "item_id")

    sessions = gather_sessions(inter_path, session_col, ts_col)
    if strategy == "tail_stratified":
        profiles = gather_session_profiles(inter_path, session_col, ts_col, item_col)
        assignment, session_stats = assign_sessions_tail_stratified(profiles, ratios)
    else:
        assignment, session_stats = assign_sessions(sessions, ratios)
    temporal_check = verify_temporal_session_order(sessions, assignment)

    out_train = ds_dir / f"{dataset}.train.inter"
    out_valid = ds_dir / f"{dataset}.valid.inter"
    out_test = ds_dir / f"{dataset}.test.inter"

    if dry_run:
        write_stats = {
            "rows": {},
            "sessions": {
                "train": session_stats["train_sessions"],
                "valid": session_stats["valid_sessions"],
                "test": session_stats["test_sessions"],
            },
            "session_overlap": {"train_valid": 0, "train_test": 0, "valid_test": 0},
            "timestamp_bounds": {},
        }
    else:
        write_stats = write_split_files(
            inter_path=inter_path,
            header=header,
            session_col=session_col,
            assignment=assignment,
            out_train=out_train,
            out_valid=out_valid,
            out_test=out_test,
            overwrite=overwrite,
        )

    summary = {
        "dataset": dataset,
        "source_inter": str(inter_path),
        "source_item": str(item_path),
        "output": {
            "train_inter": str(out_train),
            "valid_inter": str(out_valid),
            "test_inter": str(out_test),
        },
        "ratios": {"train": ratios[0], "valid": ratios[1], "test": ratios[2]},
        "split_strategy": strategy,
        "session_allocation": session_stats,
        "write_stats": write_stats,
        "temporal_check": temporal_check,
        "dry_run": dry_run,
    }

    if not dry_run:
        summary_path = ds_dir / f"{dataset}.session_split_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. Empty means auto-discover from processed-root.",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.7,0.15,0.15",
        help="Session split ratios for train,valid,test.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["contiguous", "tail_stratified"],
        default="tail_stratified",
        help="contiguous: strict time-contiguous train/valid/test; tail_stratified: train is earliest 70%%, tail 30%% is stratified into valid/test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    ratios = parse_ratios(args.ratios)
    datasets = parse_dataset_list(args.datasets, processed_root)

    if not datasets:
        raise SystemExit(f"No datasets found under {processed_root}")

    print(f"processed_root={processed_root}")
    print(f"datasets={datasets}")
    print(f"ratios={ratios}")
    print(f"strategy={args.strategy}")
    print(f"overwrite={bool(args.overwrite)} dry_run={bool(args.dry_run)}")

    all_ok = True
    summaries = []
    for ds in datasets:
        print(f"\n[dataset] {ds}")
        summary = process_dataset(
            processed_root=processed_root,
            dataset=ds,
            ratios=ratios,
            strategy=str(args.strategy),
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        summaries.append(summary)

        alloc = summary["session_allocation"]
        ws = summary["write_stats"]
        temporal = summary["temporal_check"]
        overlap = ws["session_overlap"]
        ok_overlap = (
            int(overlap["train_valid"]) == 0
            and int(overlap["train_test"]) == 0
            and int(overlap["valid_test"]) == 0
        )
        if str(args.strategy) == "contiguous":
            ok_temporal = bool(temporal["train_before_valid"]) and bool(temporal["valid_before_test"])
        else:
            valid_min = temporal.get("valid_min_session_start")
            test_min = temporal.get("test_min_session_start")
            train_max = temporal.get("train_max_session_start")
            tail_min = None
            if valid_min is None:
                tail_min = test_min
            elif test_min is None:
                tail_min = valid_min
            else:
                tail_min = min(valid_min, test_min)
            ok_temporal = True if (train_max is None or tail_min is None) else bool(train_max <= tail_min)
        all_ok = all_ok and ok_overlap and ok_temporal

        print(
            "sessions(total/train/valid/test)="
            f"{alloc['total_sessions']}/{alloc['train_sessions']}/{alloc['valid_sessions']}/{alloc['test_sessions']}"
        )
        if ws["rows"]:
            print(
                "rows(train/valid/test)="
                f"{ws['rows']['train']}/{ws['rows']['valid']}/{ws['rows']['test']}"
            )
        print(
            f"overlap(tv/tt/vt)={overlap['train_valid']}/{overlap['train_test']}/{overlap['valid_test']} "
            f"temporal_ok={ok_temporal}"
        )
        if "summary_path" in summary:
            print(f"summary={summary['summary_path']}")

    print(f"\n[done] all_ok={all_ok}")
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
