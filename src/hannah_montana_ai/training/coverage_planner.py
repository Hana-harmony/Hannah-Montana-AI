from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hannah_montana_ai.training.stock_curation import (
    HUMAN_REVIEW_APPROVED_STATUS,
    StockTrainingCandidate,
    candidate_review_key,
)
from hannah_montana_ai.training.weak_distiller import PRIMARY_LABEL_PRIORITY

STOCK_GOLD_COVERAGE_PLAN_SCHEMA_VERSION = "stock-gold-coverage-plan/v1"
STOCK_GOLD_COVERAGE_PLAN_ROW_SCHEMA_VERSION = "stock-gold-coverage-plan-row/v1"


@dataclass(frozen=True)
class StockGoldCoveragePlanRow:
    schema_version: str
    review_key: str
    intended_split: str
    review_wave: int
    review_stage: str
    review_reason: str
    text: str
    tags: list[str]
    sentiment: str
    importance: str
    source_type: str
    stock_code: str
    stock_name: str
    primary_label: str
    signal_score: int
    original_url: str
    provider: str
    content_hash: str
    review_status: str = "needs_human_review"
    reviewer_id: str = ""
    reviewed_at: str = ""
    review_notes: str = ""
    final_tags: list[str] | None = None
    final_sentiment: str = ""
    final_importance: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_tags"] = self.final_tags or []
        return payload


@dataclass(frozen=True)
class StockGoldCoveragePlanResult:
    rows: list[StockGoldCoveragePlanRow]
    report: dict[str, Any]


def build_stock_gold_coverage_plan(
    candidate_path: Path,
    training_review_path: Path,
    evaluation_review_path: Path,
    training_paths: Sequence[Path],
    evaluation_paths: Sequence[Path],
    training_stock_target: int = 1_500,
    evaluation_stock_target: int = 500,
    review_wave_size: int = 100,
) -> StockGoldCoveragePlanResult:
    candidates = _load_stock_training_candidates(candidate_path)
    existing_training_stocks = _sample_stock_codes(training_paths)
    existing_evaluation_stocks = _sample_stock_codes(evaluation_paths)
    current_training_rows = _load_review_rows(training_review_path)
    current_evaluation_rows = _load_review_rows(evaluation_review_path)
    current_training_stocks = _row_stock_codes(current_training_rows)
    current_evaluation_stocks = _row_stock_codes(current_evaluation_rows)
    reserved_stocks = (
        existing_training_stocks
        | existing_evaluation_stocks
        | current_training_stocks
        | current_evaluation_stocks
    )

    training_current_plan_rows = _current_review_plan_rows(
        current_training_rows,
        intended_split="training",
    )
    evaluation_current_plan_rows = _current_review_plan_rows(
        current_evaluation_rows,
        intended_split="evaluation",
    )
    best_candidates = _best_candidate_per_stock(candidates)
    additional_training_candidates = _select_balanced_candidates_for_coverage(
        best_candidates,
        excluded_stocks=reserved_stocks,
        target_stock_count=max(0, training_stock_target - len(current_training_stocks)),
    )
    training_reserved = reserved_stocks | {
        candidate.stock_code for candidate in additional_training_candidates
    }
    additional_evaluation_candidates = _select_balanced_candidates_for_coverage(
        best_candidates,
        excluded_stocks=training_reserved,
        target_stock_count=max(0, evaluation_stock_target - len(current_evaluation_stocks)),
    )
    rows = [
        *training_current_plan_rows,
        *evaluation_current_plan_rows,
        *_additional_candidate_plan_rows(
            additional_training_candidates,
            intended_split="training",
            starting_wave=1,
            review_wave_size=review_wave_size,
        ),
        *_additional_candidate_plan_rows(
            additional_evaluation_candidates,
            intended_split="evaluation",
            starting_wave=1,
            review_wave_size=review_wave_size,
        ),
    ]
    report = _build_report(
        candidate_path=candidate_path,
        training_review_path=training_review_path,
        evaluation_review_path=evaluation_review_path,
        candidates=candidates,
        rows=rows,
        existing_training_stocks=existing_training_stocks,
        existing_evaluation_stocks=existing_evaluation_stocks,
        current_training_stocks=current_training_stocks,
        current_evaluation_stocks=current_evaluation_stocks,
        training_stock_target=training_stock_target,
        evaluation_stock_target=evaluation_stock_target,
        review_wave_size=review_wave_size,
    )
    return StockGoldCoveragePlanResult(rows=rows, report=report)


def write_stock_gold_coverage_plan_rows(
    path: Path,
    rows: Sequence[StockGoldCoveragePlanRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_stock_gold_coverage_plan_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _current_review_plan_rows(
    rows: Sequence[dict[str, Any]],
    intended_split: str,
) -> list[StockGoldCoveragePlanRow]:
    plan_rows: list[StockGoldCoveragePlanRow] = []
    for row in rows:
        plan_rows.append(
            StockGoldCoveragePlanRow(
                schema_version=STOCK_GOLD_COVERAGE_PLAN_ROW_SCHEMA_VERSION,
                review_key=str(row["review_key"]),
                intended_split=intended_split,
                review_wave=0,
                review_stage="current_review_batch",
                review_reason="already_exported_for_human_review",
                text=str(row["text"]),
                tags=[str(tag) for tag in row.get("tags", [])],
                sentiment=str(row["sentiment"]),
                importance=str(row["importance"]),
                source_type=str(row["source_type"]),
                stock_code=str(row["stock_code"]),
                stock_name=str(row["stock_name"]),
                primary_label=str(row["primary_label"]),
                signal_score=int(row.get("signal_score", 0) or 0),
                original_url=str(row["original_url"]),
                provider=str(row["provider"]),
                content_hash=str(row["content_hash"]),
            )
        )
    return plan_rows


def _additional_candidate_plan_rows(
    candidates: Sequence[StockTrainingCandidate],
    intended_split: str,
    starting_wave: int,
    review_wave_size: int,
) -> list[StockGoldCoveragePlanRow]:
    rows: list[StockGoldCoveragePlanRow] = []
    for index, candidate in enumerate(candidates):
        wave = starting_wave + index // review_wave_size
        rows.append(
            StockGoldCoveragePlanRow(
                schema_version=STOCK_GOLD_COVERAGE_PLAN_ROW_SCHEMA_VERSION,
                review_key=candidate_review_key(candidate),
                intended_split=intended_split,
                review_wave=wave,
                review_stage="additional_coverage_plan",
                review_reason="missing_supervised_stock_coverage",
                text=candidate.text,
                tags=candidate.tags,
                sentiment=candidate.sentiment,
                importance=candidate.importance,
                source_type=candidate.source_type,
                stock_code=candidate.stock_code,
                stock_name=candidate.stock_name,
                primary_label=candidate.primary_label,
                signal_score=candidate.signal_score,
                original_url=candidate.original_url,
                provider=candidate.provider,
                content_hash=candidate.content_hash,
            )
        )
    return rows


def _best_candidate_per_stock(
    candidates: Sequence[StockTrainingCandidate],
) -> list[StockTrainingCandidate]:
    best_by_stock: dict[str, StockTrainingCandidate] = {}
    for candidate in candidates:
        previous = best_by_stock.get(candidate.stock_code)
        if previous is None or _candidate_rank(candidate) < _candidate_rank(previous):
            best_by_stock[candidate.stock_code] = candidate
    return sorted(best_by_stock.values(), key=_candidate_rank)


def _select_balanced_candidates_for_coverage(
    candidates: Sequence[StockTrainingCandidate],
    excluded_stocks: set[str],
    target_stock_count: int,
) -> list[StockTrainingCandidate]:
    selected: list[StockTrainingCandidate] = []
    seen_stocks: set[str] = set()
    buckets: dict[str, list[StockTrainingCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.stock_code not in excluded_stocks:
            buckets[candidate.primary_label].append(candidate)

    labels = sorted(buckets, key=_label_rank)
    while len(selected) < target_stock_count:
        added_this_round = False
        for label in labels:
            while buckets[label]:
                candidate = buckets[label].pop(0)
                if candidate.stock_code in seen_stocks:
                    continue
                selected.append(candidate)
                seen_stocks.add(candidate.stock_code)
                added_this_round = True
                break
            if len(selected) >= target_stock_count:
                break
        if not added_this_round:
            break
    return selected


def _build_report(
    candidate_path: Path,
    training_review_path: Path,
    evaluation_review_path: Path,
    candidates: Sequence[StockTrainingCandidate],
    rows: Sequence[StockGoldCoveragePlanRow],
    existing_training_stocks: set[str],
    existing_evaluation_stocks: set[str],
    current_training_stocks: set[str],
    current_evaluation_stocks: set[str],
    training_stock_target: int,
    evaluation_stock_target: int,
    review_wave_size: int,
) -> dict[str, Any]:
    candidate_stocks = {candidate.stock_code for candidate in candidates}
    training_rows = [row for row in rows if row.intended_split == "training"]
    evaluation_rows = [row for row in rows if row.intended_split == "evaluation"]
    training_stocks = _plan_row_stock_codes(training_rows)
    evaluation_stocks = _plan_row_stock_codes(evaluation_rows)
    existing_supervised_stocks = existing_training_stocks | existing_evaluation_stocks
    planned_stocks = training_stocks | evaluation_stocks
    new_training_coverage = len(existing_training_stocks | training_stocks)
    new_evaluation_coverage = len(existing_evaluation_stocks | evaluation_stocks)
    return {
        "schema_version": STOCK_GOLD_COVERAGE_PLAN_SCHEMA_VERSION,
        "row_schema_version": STOCK_GOLD_COVERAGE_PLAN_ROW_SCHEMA_VERSION,
        "candidate_path": _report_path(candidate_path),
        "training_review_path": _report_path(training_review_path),
        "evaluation_review_path": _report_path(evaluation_review_path),
        "candidate_count": len(candidates),
        "candidate_stock_count": len(candidate_stocks),
        "existing_supervised_training_stock_count": len(existing_training_stocks),
        "existing_supervised_evaluation_stock_count": len(existing_evaluation_stocks),
        "current_training_review_stock_count": len(current_training_stocks),
        "current_evaluation_review_stock_count": len(current_evaluation_stocks),
        "review_wave_size": review_wave_size,
        "training_plan": _split_report(
            rows=training_rows,
            target_stock_count=training_stock_target,
            existing_stock_count=len(existing_training_stocks),
            projected_supervised_stock_count=new_training_coverage,
        ),
        "evaluation_plan": _split_report(
            rows=evaluation_rows,
            target_stock_count=evaluation_stock_target,
            existing_stock_count=len(existing_evaluation_stocks),
            projected_supervised_stock_count=new_evaluation_coverage,
        ),
        "disjoint_stock_check": {
            "status": "pass" if training_stocks.isdisjoint(evaluation_stocks) else "fail",
            "overlap_stock_count": len(training_stocks & evaluation_stocks),
        },
        "candidate_coverage_after_full_plan": {
            "planned_stock_count": len(planned_stocks),
            "already_supervised_stock_count": len(existing_supervised_stocks),
            "covered_candidate_stock_count": len(planned_stocks | existing_supervised_stocks),
            "remaining_candidate_stock_count": max(
                0,
                len(candidate_stocks - planned_stocks - existing_supervised_stocks),
            ),
        },
        "review_policy": (
            "coverage plan rows are review tasks only; supervised datasets are updated "
            f"only after {HUMAN_REVIEW_APPROVED_STATUS} with reviewer metadata"
        ),
    }


def _split_report(
    rows: Sequence[StockGoldCoveragePlanRow],
    target_stock_count: int,
    existing_stock_count: int,
    projected_supervised_stock_count: int,
) -> dict[str, Any]:
    stage_distribution = Counter(row.review_stage for row in rows)
    wave_distribution = Counter(str(row.review_wave) for row in rows)
    stock_count = len(_plan_row_stock_codes(rows))
    return {
        "target_stock_count": target_stock_count,
        "planned_row_count": len(rows),
        "planned_stock_count": stock_count,
        "existing_supervised_stock_count": existing_stock_count,
        "projected_supervised_stock_count_after_full_approval": (
            projected_supervised_stock_count
        ),
        "remaining_stock_count_to_target": max(0, target_stock_count - stock_count),
        "status": "pass" if stock_count >= target_stock_count else "fail",
        "label_distribution": dict(sorted(Counter(row.primary_label for row in rows).items())),
        "source_type_distribution": dict(
            sorted(Counter(row.source_type for row in rows).items())
        ),
        "stage_distribution": dict(sorted(stage_distribution.items())),
        "wave_distribution": dict(sorted(wave_distribution.items(), key=lambda item: int(item[0]))),
    }


def _candidate_rank(candidate: StockTrainingCandidate) -> tuple[int, int, str, str]:
    return (
        _label_rank(candidate.primary_label),
        -candidate.signal_score,
        candidate.stock_code,
        candidate.content_hash,
    )


def _label_rank(label: str) -> int:
    try:
        return PRIMARY_LABEL_PRIORITY.index(label)
    except ValueError:
        return len(PRIMARY_LABEL_PRIORITY)


def _load_stock_training_candidates(path: Path) -> list[StockTrainingCandidate]:
    if not path.exists():
        return []
    candidates: list[StockTrainingCandidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        candidates.append(
            StockTrainingCandidate(
                text=str(payload["text"]),
                tags=[str(tag) for tag in payload["tags"]],
                sentiment=str(payload["sentiment"]),
                importance=str(payload["importance"]),
                source_type=str(payload["source_type"]),
                stock_code=str(payload["stock_code"]),
                stock_name=str(payload["stock_name"]),
                primary_label=str(payload["primary_label"]),
                signal_score=int(payload["signal_score"]),
                original_url=str(payload["original_url"]),
                provider=str(payload["provider"]),
                content_hash=str(payload["content_hash"]),
                curation_status=str(payload.get("curation_status", "needs_human_review")),
            )
        )
    return candidates


def _load_review_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_stock_codes(paths: Sequence[Path]) -> set[str]:
    stock_codes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            stock_code = payload.get("stock_code")
            if isinstance(stock_code, str) and stock_code:
                stock_codes.add(stock_code)
    return stock_codes


def _row_stock_codes(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(row["stock_code"])
        for row in rows
        if isinstance(row.get("stock_code"), str) and row["stock_code"]
    }


def _plan_row_stock_codes(rows: Sequence[StockGoldCoveragePlanRow]) -> set[str]:
    return {row.stock_code for row in rows if row.stock_code}


def _report_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
