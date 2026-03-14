#!/usr/bin/env python3
"""Generate next 3 experiment ideas from summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_DATASET_ORDER = [
    "movielens1m",
    "retail_rocket",
    "amazon_beauty",
    "foursquare",
    "kuairec0.3",
    "lastfm0.3",
]


def canonical_dataset(name: str) -> str:
    raw = (name or "").strip().lower()
    aliases = {
        "kuairec": "kuairec0.3",
        "kuairec0.3": "kuairec0.3",
        "retailrocket": "retail_rocket",
        "ml1m": "movielens1m",
        "movielens_1m": "movielens1m",
        "movielens-1m": "movielens1m",
        "lastfm": "lastfm0.3",
    }
    return aliases.get(raw, raw)


@dataclass
class Row:
    dataset: str
    model: str
    metric_value: float
    run_group: str
    run_axis: str
    run_phase: str


@dataclass
class Suggestion:
    priority: int
    title: str
    dataset: str
    track: str
    hypothesis: str
    reason: str
    command: str


def classify_track(row: Row) -> str:
    model = row.model.lower()
    group = row.run_group.lower()
    if "hir" in model or group == "fmoe_hir":
        return "hir"
    if "featuredmoe" in model or group == "fmoe":
        return "fmoe"
    if group == "baseline":
        return "baseline"
    return "unknown"


def load_rows(path: Path) -> list[Row]:
    out: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                metric_value = float(row.get("metric_value", "nan"))
            except ValueError:
                continue
            out.append(
                Row(
                    dataset=canonical_dataset(row.get("dataset", "")),
                    model=(row.get("model", "") or "").strip(),
                    metric_value=metric_value,
                    run_group=(row.get("run_group", "") or "").strip(),
                    run_axis=(row.get("run_axis", "") or "").strip(),
                    run_phase=(row.get("run_phase", "") or "").strip(),
                )
            )
    return out


def best_by_track(rows: list[Row], dataset: str, track: str) -> Row | None:
    cand = [r for r in rows if r.dataset == dataset and classify_track(r) == track]
    if not cand:
        return None
    return max(cand, key=lambda r: r.metric_value)


def infer_repo_root(summary: Path) -> Path:
    resolved = summary.resolve()
    cur = resolved.parent
    # Prefer explicit marker discovery.
    while True:
        marker = cur / ".codex" / "skills" / "fmoe" / "scripts"
        if marker.exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # Fallback for known layouts:
    # - <repo>/experiments/run/artifacts/results/summary.csv
    # - <repo>/experiments/run/hyperopt_results/summary.csv
    parts = resolved.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        if idx > 0:
            return Path(*parts[:idx])
    return Path.cwd()


def build_suggestions(rows: list[Row], repo_root: Path) -> list[Suggestion]:
    skill_launcher = repo_root / ".codex" / "skills" / "fmoe" / "scripts" / "launch_track.sh"
    fmoe_tune = repo_root / "experiments" / "run" / "fmoe" / "tune_hparam.sh"
    suggestions: list[Suggestion] = []
    pri = 1

    primary = ["movielens1m", "retail_rocket"]
    ml_fmoe = best_by_track(rows, "movielens1m", "fmoe")
    rr_fmoe = best_by_track(rows, "retail_rocket", "fmoe")

    if ml_fmoe is None or rr_fmoe is None:
        missing = []
        if ml_fmoe is None:
            missing.append("movielens1m")
        if rr_fmoe is None:
            missing.append("retail_rocket")
        ds_csv = ",".join(missing if missing else primary)
        suggestions.append(
            Suggestion(
                priority=pri,
                title="1차 데이터셋 FMoE 메인트랙 보강",
                dataset=ds_csv,
                track="fmoe-main",
                hypothesis="ML1M/RetailRocket anchor가 채워지면 이후 전이 튜닝의 분산을 줄일 수 있다.",
                reason="핵심 데이터셋의 FMoE 결과가 비어 있거나 부족하다.",
                command=f"bash {skill_launcher} --track fmoe-main --datasets {ds_csv} --gpus 0,1 --seed-base 42",
            )
        )
        pri += 1

    for ds in primary:
        fmoe_best = best_by_track(rows, ds, "fmoe")
        hir_best = best_by_track(rows, ds, "hir")
        if fmoe_best and hir_best is None:
            suggestions.append(
                Suggestion(
                    priority=pri,
                    title=f"{ds} HiR 비교 트랙 실행",
                    dataset=ds,
                    track="hir-compare",
                    hypothesis="계층 라우팅이 해당 데이터셋에서 routing collapse를 줄일 수 있다.",
                    reason="FMoE 기준선은 있으나 HiR 비교 결과가 없다.",
                    command=f"bash {skill_launcher} --track hir-compare --datasets {ds} --gpus 0 --seed-base 42 --dry-run",
                )
            )
            pri += 1
        elif fmoe_best and hir_best:
            gap = fmoe_best.metric_value - hir_best.metric_value
            if gap <= 0.005:
                suggestions.append(
                    Suggestion(
                        priority=pri,
                        title=f"{ds} HiR 온도/병합축 재탐색",
                        dataset=ds,
                        track="hir-compare",
                        hypothesis="serial/parallel + temp 축 재조합으로 FMoE 격차를 좁히거나 역전할 수 있다.",
                        reason=f"HiR가 FMoE 대비 근접(gap={gap:.4f})해 추가 탐색 가치가 높다.",
                        command=f"bash {skill_launcher} --track hir-compare --datasets {ds} --gpus 0 --seed-base 42 --max-evals 20 --search-profile wide",
                    )
                )
                pri += 1

    for ds in DEFAULT_DATASET_ORDER[2:]:
        if best_by_track(rows, ds, "fmoe") is None:
            suggestions.append(
                Suggestion(
                    priority=pri,
                    title=f"{ds}로 FMoE 확장",
                    dataset=ds,
                    track="fmoe-hparam-bootstrap",
                    hypothesis="ML1M/RetailRocket에서 얻은 layout/schedule priors가 중간 규모 데이터셋으로 전이될 수 있다.",
                    reason="확장 순서에 있는 데이터셋인데 FMoE 결과가 아직 없다.",
                    command=(
                        f"bash {fmoe_tune} --dataset {ds} --gpu 0 --layout-id 0 "
                        "--schedule-preset off --max-evals 20 --phase P1EXT --dry-run"
                    ),
                )
            )
            pri += 1

    if not suggestions:
        suggestions.append(
            Suggestion(
                priority=1,
                title="아키텍처 탐색 프로브",
                dataset="movielens1m",
                track="arch-probe",
                hypothesis="신규 라우팅/전문가 구조가 현재 로컬 optimum을 벗어날 수 있다.",
                reason="핵심 트랙 결과가 이미 충분히 채워져 있어 구조 가설 검증이 유효하다.",
                command=f"bash {skill_launcher} --track arch-probe",
            )
        )

    suggestions.sort(key=lambda x: x.priority)
    return suggestions


def write_outputs(
    out_dir: Path,
    mode: str,
    topn: int,
    suggestions: list[Suggestion],
) -> tuple[Path, Path]:
    picked = suggestions[:topn]
    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "topn": topn,
        "suggestions": [
            {
                "priority": s.priority,
                "title": s.title,
                "dataset": s.dataset,
                "track": s.track,
                "hypothesis": s.hypothesis,
                "reason": s.reason,
                "command": s.command,
            }
            for s in picked
        ],
    }

    json_path = out_dir / "next_plan.json"
    md_path = out_dir / "next_plan.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# Next Experiment Plan")
    lines.append("")
    lines.append(f"- generated_at: {payload['generated_at']}")
    lines.append(f"- mode: {mode}")
    lines.append(f"- topn: {topn}")
    lines.append("")
    for s in picked:
        lines.append(f"## Priority {s.priority}: {s.title}")
        lines.append("")
        lines.append(f"- dataset: {s.dataset}")
        lines.append(f"- track: {s.track}")
        lines.append(f"- hypothesis: {s.hypothesis}")
        lines.append(f"- reason: {s.reason}")
        lines.append("- command:")
        lines.append("```bash")
        lines.append(s.command)
        lines.append("```")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend next FMoE experiments.")
    parser.add_argument("--summary", required=True, help="Path to summary.csv from collect_results.py")
    parser.add_argument("--mode", default="fmoe-first")
    parser.add_argument("--topn", type=int, default=3)
    args = parser.parse_args()

    if args.mode != "fmoe-first":
        raise SystemExit("Only --mode fmoe-first is supported")
    if args.topn < 1:
        raise SystemExit("--topn must be >= 1")

    summary = Path(args.summary).resolve()
    if not summary.exists():
        raise SystemExit(f"summary file not found: {summary}")

    rows = load_rows(summary)
    repo_root = infer_repo_root(summary)
    suggestions = build_suggestions(rows, repo_root)
    md_path, json_path = write_outputs(summary.parent, args.mode, args.topn, suggestions)

    print(f"[OK] next_plan.md: {md_path}")
    print(f"[OK] next_plan.json: {json_path}")
    print(f"[INFO] suggestions_generated={len(suggestions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
