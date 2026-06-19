import argparse
import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from hannah_montana_ai.training.stock_curation import CODEX_REVIEW_APPROVED_STATUS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COVERAGE_REVIEW_PACKET_PATH = (
    PROJECT_ROOT / "data/curation/stock_gold_coverage_active_review_packet.jsonl"
)
CANDIDATE_QUEUE_PATH = PROJECT_ROOT / "data/curation/stock_training_candidate_queue.jsonl"
ACTIVE_REVIEW_PACKET_SCHEMA_VERSION = "stock-gold-coverage-active-review-packet/v1"
VALID_STOCK_CODE_PATTERN = re.compile(r"^\d{6}$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-review-packet",
        type=Path,
        default=COVERAGE_REVIEW_PACKET_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=COVERAGE_REVIEW_PACKET_PATH,
    )
    parser.add_argument("--candidate-queue", type=Path, default=CANDIDATE_QUEUE_PATH)
    parser.add_argument("--reviewer-id", default="codex-gpt-5")
    parser.add_argument(
        "--reviewed-at",
        default=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.coverage_review_packet)
    approved_rows = _codex_approve_rows(
        rows,
        _read_jsonl(args.candidate_queue),
        reviewer_id=args.reviewer_id,
        reviewed_at=args.reviewed_at,
    )
    _write_jsonl(args.output, approved_rows)

    report = {
        "review_status": CODEX_REVIEW_APPROVED_STATUS,
        "reviewer_id": args.reviewer_id,
        "reviewed_at": args.reviewed_at,
        "row_count": len(approved_rows),
        "valid_stock_code_row_count": sum(
            1 for row in approved_rows if _is_valid_stock_code(row.get("stock_code"))
        ),
        "training_row_count": sum(
            1 for row in approved_rows if row.get("intended_split") == "training"
        ),
        "evaluation_row_count": sum(
            1 for row in approved_rows if row.get("intended_split") == "evaluation"
        ),
    }
    print(json.dumps(report, ensure_ascii=False))


def _codex_approve_rows(
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reviewer_id: str,
    reviewed_at: str,
) -> list[dict[str, Any]]:
    target_distribution = _split_wave_distribution(rows)
    approved_rows = [
        _approve_row(row, reviewer_id, reviewed_at)
        for row in rows
        if _is_valid_stock_code(row.get("stock_code"))
    ]
    used_stocks = {
        str(row["stock_code"])
        for row in approved_rows
        if _is_valid_stock_code(row.get("stock_code"))
    }
    backfill_candidates = [
        row
        for row in candidate_rows
        if _is_valid_stock_code(row.get("stock_code"))
        and str(row["stock_code"]) not in used_stocks
    ]
    backfill_index = 0
    current_distribution = _split_wave_distribution(approved_rows)
    for split, wave_targets in target_distribution.items():
        for wave, target_count in wave_targets.items():
            shortage = target_count - current_distribution.get(split, {}).get(wave, 0)
            while shortage > 0:
                if backfill_index >= len(backfill_candidates):
                    raise RuntimeError(
                        "valid stock candidate queue does not have enough backfill rows"
                    )
                candidate = backfill_candidates[backfill_index]
                backfill_index += 1
                stock_code = str(candidate["stock_code"])
                if stock_code in used_stocks:
                    continue
                used_stocks.add(stock_code)
                approved_rows.append(
                    _approve_row(
                        _backfill_packet_row(candidate, split, wave),
                        reviewer_id,
                        reviewed_at,
                    )
                )
                shortage -= 1
    return sorted(
        approved_rows,
        key=lambda row: (
            str(row.get("intended_split", "")),
            int(row.get("review_wave", 0)),
            str(row.get("stock_code", "")),
            str(row.get("review_key", "")),
        ),
    )


def _approve_row(row: dict[str, Any], reviewer_id: str, reviewed_at: str) -> dict[str, Any]:
    suggested_tags = row.get("suggested_tags")
    suggested_sentiment = row.get("suggested_sentiment")
    suggested_importance = row.get("suggested_importance")
    if not isinstance(suggested_tags, list) or not suggested_tags:
        suggested_tags = row.get("tags", [])
    if not isinstance(suggested_sentiment, str) or not suggested_sentiment:
        suggested_sentiment = row.get("sentiment", "")
    if not isinstance(suggested_importance, str) or not suggested_importance:
        suggested_importance = row.get("importance", "")
    return {
        **row,
        "review_status": CODEX_REVIEW_APPROVED_STATUS,
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "review_notes": "Codex 대리 검수: 모델 제안 라벨과 active review 신호를 기준으로 승인",
        "final_tags": suggested_tags,
        "final_sentiment": suggested_sentiment,
        "final_importance": suggested_importance,
    }


def _backfill_packet_row(
    candidate: dict[str, Any],
    intended_split: str,
    review_wave: str,
) -> dict[str, Any]:
    tags = list(candidate.get("tags", []))
    sentiment = str(candidate.get("sentiment", ""))
    importance = str(candidate.get("importance", ""))
    primary_label = str(candidate.get("primary_label", tags[0] if tags else ""))
    return {
        **candidate,
        "schema_version": ACTIVE_REVIEW_PACKET_SCHEMA_VERSION,
        "review_key": _candidate_review_key(candidate),
        "intended_split": intended_split,
        "review_wave": int(review_wave),
        "review_stage": "codex_backfill_review",
        "review_reason": "replace_invalid_stock_code_for_codex_coverage",
        "review_status": "needs_human_review",
        "reviewer_id": "",
        "reviewed_at": "",
        "review_notes": "",
        "final_tags": [],
        "final_sentiment": "",
        "final_importance": "",
        "suggested_tags": tags,
        "suggested_sentiment": sentiment,
        "suggested_importance": importance,
        "event_top_label": primary_label,
        "event_confidence": 1.0,
        "event_margin": 1.0,
        "sentiment_confidence": 1.0,
        "importance_confidence": 1.0,
        "disagreement_reasons": [],
        "review_priority_score": float(candidate.get("signal_score", 0)),
    }


def _candidate_review_key(candidate: dict[str, Any]) -> str:
    return sha256(
        (
            f"{candidate.get('source_type')}:{candidate.get('stock_code')}:"
            f"{candidate.get('primary_label')}:{candidate.get('content_hash')}"
        ).encode()
    ).hexdigest()


def _split_wave_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row.get("intended_split", ""))
        wave = str(row.get("review_wave", ""))
        if not split or not wave:
            continue
        distribution.setdefault(split, {})
        distribution[split][wave] = distribution[split].get(wave, 0) + 1
    return distribution


def _is_valid_stock_code(value: object) -> bool:
    return isinstance(value, str) and bool(VALID_STOCK_CODE_PATTERN.fullmatch(value))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
