#!/usr/bin/env python3
"""Small Slack webhook helper for experiment progress notifications."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, Iterable, List


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _fmt_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "--"
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _short_list(items: Iterable[str], *, limit: int = 4) -> str:
    uniq: List[str] = []
    seen = set()
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    if not uniq:
        return "-"
    if len(uniq) <= limit:
        return ", ".join(uniq)
    return ", ".join(uniq[:limit]) + f", +{len(uniq) - limit}"


def _dataset_counts(rows: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        dataset = str(row.get("dataset", "")).strip() or "unknown"
        counts[dataset] += 1
    return counts


class SlackProgressNotifier:
    """Emit compact plan/progress messages if SLACK_WEBHOOK_URL is configured."""

    def __init__(
        self,
        *,
        phase_label: str,
        rows: Iterable[Dict[str, Any]],
        progress_step: int | None = None,
    ) -> None:
        self.webhook_url = str(os.environ.get("SLACK_WEBHOOK_URL", "")).strip()
        self.enabled = bool(self.webhook_url) and _env_flag("SLACK_NOTIFY", True)
        self.phase_label = str(phase_label).strip() or "Experiment"
        self.note = str(os.environ.get("SLACK_NOTIFY_NOTE", "")).strip()
        self.scope_label = str(os.environ.get("SLACK_NOTIFY_SCOPE_LABEL", "")).strip()
        self.progress_step = max(1, min(progress_step or _env_int("SLACK_NOTIFY_PROGRESS_STEP", 20), 100))
        self.rows_snapshot = list(rows)
        self.total_rows_local = len(self.rows_snapshot)
        self.dataset_totals = _dataset_counts(self.rows_snapshot)
        self.dataset_done: Counter[str] = Counter()
        self.precompleted_local = 0
        self.global_total = max(_env_int("SLACK_NOTIFY_TOTAL_RUNS", self.total_rows_local), self.total_rows_local, 1)
        self.global_done_base = max(_env_int("SLACK_NOTIFY_GLOBAL_DONE_BASE", 0), 0)
        self._next_threshold = self.progress_step
        self._started_at = time.time()

    def _post(self, text: str) -> None:
        if not self.enabled:
            return
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except urllib.error.URLError as exc:
            print(f"[slack] notify failed: {exc}")
        except Exception as exc:
            print(f"[slack] notify failed: {exc}")

    def _done_local(self) -> int:
        return int(sum(self.dataset_done.values()))

    def _done_global(self) -> int:
        return int(self.global_done_base + self._done_local())

    def _percent(self) -> int:
        return min(100, int((self._done_global() * 100) / max(self.global_total, 1)))

    def _eta_seconds(self) -> float | None:
        runtime_done = max(self._done_local() - self.precompleted_local, 0)
        if runtime_done <= 0:
            return None
        elapsed = max(time.time() - self._started_at, 1.0)
        rate = elapsed / float(runtime_done)
        remaining = max(self.global_total - self._done_global(), 0)
        return rate * float(remaining)

    def _dataset_brief(self) -> tuple[str, str, str]:
        done = []
        active = []
        remaining = []
        for dataset, total in self.dataset_totals.items():
            count = int(self.dataset_done.get(dataset, 0))
            if count >= int(total):
                done.append(dataset)
            elif count > 0:
                active.append(f"{dataset}({count}/{int(total)})")
            else:
                remaining.append(dataset)
        return _short_list(done), _short_list(active), _short_list(remaining)

    def _compose(self, header: str, *, include_resume: bool = False) -> str:
        done_ds, active_ds, remaining_ds = self._dataset_brief()
        lines = [
            header,
            f"runs {self._done_global()}/{self.global_total} ({self._percent()}%)",
            f"done={done_ds}",
            f"active={active_ds}",
            f"remaining={remaining_ds}",
            f"elapsed={_fmt_seconds(time.time() - self._started_at)} | ETA={_fmt_seconds(self._eta_seconds())}",
        ]
        if include_resume and self.precompleted_local > 0:
            lines.insert(2, f"resume_skip={self.precompleted_local}")
        if self.scope_label:
            lines.insert(1, f"scope={self.scope_label}")
        if self.note:
            lines.append(f"note={self.note}")
        return "\n".join(lines)

    def notify_plan(self, *, precompleted_rows: Iterable[Dict[str, Any]] | None = None) -> None:
        if precompleted_rows:
            for row in precompleted_rows:
                dataset = str(row.get("dataset", "")).strip() or "unknown"
                self.dataset_done[dataset] += 1
        self.precompleted_local = self._done_local()
        header = f":clipboard: [{self.phase_label}] plan"
        self._post(self._compose(header, include_resume=True))
        while self._next_threshold <= self._percent():
            self._next_threshold += self.progress_step

    def mark_complete(self, row: Dict[str, Any]) -> None:
        dataset = str(row.get("dataset", "")).strip() or "unknown"
        self.dataset_done[dataset] += 1
        while self._next_threshold <= self._percent():
            header = f":hourglass_flowing_sand: [{self.phase_label}] {self._next_threshold}%"
            self._post(self._compose(header))
            self._next_threshold += self.progress_step
