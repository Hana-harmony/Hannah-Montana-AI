import json
from pathlib import Path


class KeywordFinancialNlpModel:
    def __init__(self, model_path: Path) -> None:
        with model_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        self.version: str = payload["version"]
        self.event_keywords: dict[str, list[str]] = payload["event_keywords"]

    def predict_event_tags(self, text: str) -> list[str]:
        tags = [
            tag
            for tag, keywords in self.event_keywords.items()
            if any(keyword in text for keyword in keywords)
        ]
        return tags or ["GENERAL_MARKET"]
