from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from hannah_montana_ai.training.dataset import LabeledAlert, load_labeled_alerts

PRIMARY_LABEL_PRIORITY = (
    "RISK",
    "CONTRACT",
    "CAPITAL_ACTION",
    "CORPORATE_ACTION",
    "EARNINGS",
    "MACRO",
    "DISCLOSURE",
    "GENERAL_MARKET",
)

LABEL_QUOTAS = {
    "RISK": 900,
    "CONTRACT": 800,
    "CAPITAL_ACTION": 800,
    "CORPORATE_ACTION": 800,
    "EARNINGS": 750,
    "MACRO": 750,
    "DISCLOSURE": 650,
    "GENERAL_MARKET": 450,
}

DISCLOSURE_NOISE_PATTERNS = (
    "집합투자증권",
    "투자설명서",
    "증권발행실적보고서",
    "일괄신고",
    "소액공모",
    "투자회사",
    "자산운용",
    "ETF",
    "ETN",
)

HIGH_SIGNAL_PATTERNS = {
    "RISK": (
        "상장폐지",
        "거래정지",
        "횡령",
        "배임",
        "감사의견",
        "불성실공시",
        "소송",
    ),
    "CONTRACT": ("공급계약", "수주", "납품", "양산", "계약체결"),
    "CAPITAL_ACTION": ("유상증자", "무상증자", "감자", "자사주", "배당", "전환사채"),
    "CORPORATE_ACTION": ("합병", "분할", "인수", "매각", "최대주주", "지분취득", "지분처분"),
    "EARNINGS": ("실적", "영업이익", "매출", "순이익", "흑자", "적자", "잠정"),
    "MACRO": ("환율", "금리", "물가", "수출", "코스피", "코스닥", "외국인", "반도체"),
    "DISCLOSURE": ("공시", "보고서", "주요사항", "결정", "제출"),
    "GENERAL_MARKET": ("코스피", "코스닥", "증시", "시장", "ETF", "외국인"),
}

_SPACE_PATTERN = re.compile(r"\s+")
_BRACKET_PATTERN = re.compile(r"\[[^\]]{1,20}\]|\([^)]{1,20}\)")


@dataclass(frozen=True)
class WeakDistillationResult:
    samples: list[LabeledAlert]
    report: dict[str, Any]


def distill_weak_labeled_alerts(path: Path) -> WeakDistillationResult:
    if not path.exists():
        return WeakDistillationResult(
            samples=[],
            report={
                "source_path": str(path),
                "status": "missing",
                "candidate_count": 0,
                "accepted_count": 0,
            },
        )

    candidates = load_labeled_alerts(path)
    accepted_by_label: dict[str, list[tuple[int, str, LabeledAlert]]] = defaultdict(list)
    rejected_reasons: Counter[str] = Counter()
    seen_texts: set[str] = set()

    for sample in candidates:
        normalized_text = _canonical_text(sample.text)
        reason = _reject_reason(sample, normalized_text)
        if reason:
            rejected_reasons[reason] += 1
            continue
        if normalized_text in seen_texts:
            rejected_reasons["duplicate_text"] += 1
            continue

        primary_label = _primary_label(sample.tags)
        score = _confidence_score(sample, normalized_text, primary_label)
        if score < 2:
            rejected_reasons["low_signal"] += 1
            continue

        seen_texts.add(normalized_text)
        tie_breaker = sha256(f"{primary_label}:{normalized_text}".encode()).hexdigest()
        accepted_by_label[primary_label].append((score, tie_breaker, sample))

    distilled_samples: list[LabeledAlert] = []
    accepted_count_by_primary_label: dict[str, int] = {}
    for label in PRIMARY_LABEL_PRIORITY:
        rows = sorted(accepted_by_label[label], key=lambda item: (-item[0], item[1]))
        quota = LABEL_QUOTAS[label]
        selected = [sample for _, _, sample in rows[:quota]]
        distilled_samples.extend(selected)
        accepted_count_by_primary_label[label] = len(selected)

    report = _build_report(
        path=path,
        candidates=candidates,
        samples=distilled_samples,
        rejected_reasons=rejected_reasons,
        accepted_count_by_primary_label=accepted_count_by_primary_label,
    )
    return WeakDistillationResult(samples=distilled_samples, report=report)


def _reject_reason(sample: LabeledAlert, normalized_text: str) -> str | None:
    if len(normalized_text) < 12:
        return "too_short"
    if len(normalized_text) > 360:
        return "too_long"
    if sample.source_type == "DISCLOSURE" and any(
        pattern in normalized_text for pattern in DISCLOSURE_NOISE_PATTERNS
    ):
        return "disclosure_noise"
    if not sample.tags:
        return "missing_event_tag"
    if sample.tags == ["DISCLOSURE"]:
        return "disclosure_only"
    return None


def _confidence_score(
    sample: LabeledAlert,
    normalized_text: str,
    primary_label: str,
) -> int:
    score = sum(1 for pattern in HIGH_SIGNAL_PATTERNS[primary_label] if pattern in normalized_text)
    for tag in sample.tags:
        score += sum(
            1
            for pattern in HIGH_SIGNAL_PATTERNS.get(tag, ())
            if pattern in normalized_text
        )
    if sample.source_type == "DISCLOSURE" and "DISCLOSURE" in sample.tags:
        score += 1
    if sample.importance in {"HIGH", "CRITICAL"}:
        score += 1
    if sample.sentiment != "NEUTRAL":
        score += 1
    return score


def _primary_label(tags: list[str]) -> str:
    tag_set = set(tags)
    for label in PRIMARY_LABEL_PRIORITY:
        if label in tag_set:
            return label
    return "GENERAL_MARKET"


def _canonical_text(text: str) -> str:
    stripped = _BRACKET_PATTERN.sub(" ", text)
    return _SPACE_PATTERN.sub(" ", stripped).strip()


def _build_report(
    path: Path,
    candidates: list[LabeledAlert],
    samples: list[LabeledAlert],
    rejected_reasons: Counter[str],
    accepted_count_by_primary_label: dict[str, int],
) -> dict[str, Any]:
    event_distribution: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    sentiment_distribution: Counter[str] = Counter()
    importance_distribution: Counter[str] = Counter()
    for sample in samples:
        event_distribution.update(sample.tags)
        source_distribution.update([sample.source_type])
        sentiment_distribution.update([sample.sentiment])
        importance_distribution.update([sample.importance])

    return {
        "source_path": _report_path(path),
        "status": "distilled",
        "candidate_count": len(candidates),
        "accepted_count": len(samples),
        "rejected_count": len(candidates) - len(samples),
        "accepted_count_by_primary_label": accepted_count_by_primary_label,
        "rejected_count_by_reason": dict(sorted(rejected_reasons.items())),
        "event_label_distribution": dict(sorted(event_distribution.items())),
        "source_type_distribution": dict(sorted(source_distribution.items())),
        "sentiment_label_distribution": dict(sorted(sentiment_distribution.items())),
        "importance_label_distribution": dict(sorted(importance_distribution.items())),
        "label_quotas": LABEL_QUOTAS,
    }


def _report_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def write_weak_distillation_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
