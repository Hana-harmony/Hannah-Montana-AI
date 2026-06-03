import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = PROJECT_ROOT / "data/training/financial_alert_corpus.jsonl"
MODEL_PATH = PROJECT_ROOT / "src/hannah_montana_ai/model_store/financial_nlp_baseline.json"


SEED_KEYWORDS = {
    "EARNINGS": ["실적", "매출", "영업이익", "순이익", "흑자", "적자"],
    "DISCLOSURE": ["공시", "보고서", "제출", "정정"],
    "CAPITAL_ACTION": ["유상증자", "무상증자", "감자", "배당", "자사주"],
    "CORPORATE_ACTION": ["합병", "분할", "인수", "매각", "최대주주"],
    "CONTRACT": ["공급계약", "수주", "계약", "납품"],
    "RISK": ["거래정지", "상장폐지", "횡령", "배임", "소송", "제재", "과징금"],
}


def main() -> None:
    tag_counts: dict[str, set[str]] = defaultdict(set)
    for line in TRAINING_DATA.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        for tag in sample["tags"]:
            tag_counts[tag].update(SEED_KEYWORDS.get(tag, []))

    payload = {
        "version": "financial-keyword-baseline-2026-06-03",
        "trained_at": "2026-06-03T00:00:00+09:00",
        "training_data": "data/training/financial_alert_corpus.jsonl",
        "event_keywords": {tag: sorted(keywords) for tag, keywords in tag_counts.items()},
    }
    MODEL_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
