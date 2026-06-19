import json
from dataclasses import dataclass, field
from pathlib import Path

from hannah_montana_ai.domain.schemas import Importance, Sentiment, SourceType


@dataclass(frozen=True)
class LabeledAlert:
    text: str
    tags: list[str]
    sentiment: Sentiment
    importance: Importance
    source_type: SourceType = "NEWS"
    stock_code: str | None = None
    stock_name: str | None = None
    stock_aliases: list[str] = field(default_factory=list)
    source_review_status: str = ""
    reviewer_id: str = ""


def load_labeled_alerts(path: Path) -> list[LabeledAlert]:
    samples: list[LabeledAlert] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        samples.append(
            LabeledAlert(
                text=payload["text"],
                tags=payload["tags"],
                sentiment=payload["sentiment"],
                importance=payload["importance"],
                source_type=payload.get("source_type", "NEWS"),
                stock_code=payload.get("stock_code"),
                stock_name=payload.get("stock_name"),
                stock_aliases=payload.get("stock_aliases", []),
                source_review_status=payload.get("source_review_status", ""),
                reviewer_id=payload.get("reviewer_id", ""),
            )
        )
    return samples
