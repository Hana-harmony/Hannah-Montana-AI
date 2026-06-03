import argparse
import json
from datetime import date
from pathlib import Path

from hannah_montana_ai.training.collector import (
    collect_naver_news,
    collect_open_dart,
    collection_status_to_dict,
    load_local_env,
    merge_raw_alerts,
    raw_alert_to_dict,
    read_raw_alerts,
    should_write_raw_alerts,
    write_jsonl,
)
from hannah_montana_ai.training.weak_labeler import weak_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ALERTS_PATH = PROJECT_ROOT / "data/raw/collected_alerts.jsonl"
WEAK_LABELED_PATH = PROJECT_ROOT / "data/processed/weak_labeled_alerts.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports/dataset-collection.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-news-per-query", type=int, default=200)
    parser.add_argument("--news-sleep-seconds", type=float, default=0.4)
    parser.add_argument("--news-max-retries", type=int, default=3)
    parser.add_argument("--dart-days", type=int, default=30)
    parser.add_argument("--dart-pages", type=int, default=3)
    parser.add_argument("--dart-end-date", type=date.fromisoformat)
    parser.add_argument("--dart-sleep-seconds", type=float, default=0.1)
    parser.add_argument("--reuse-existing-raw", action="store_true")
    parser.add_argument("--force-overwrite-raw", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-dart", action="store_true")
    args = parser.parse_args()

    load_local_env(PROJECT_ROOT / "secrets.local.env")
    existing_raw_alerts = read_raw_alerts(RAW_ALERTS_PATH)
    raw_alerts = []
    collection_statuses = []
    if args.reuse_existing_raw:
        raw_alerts.extend(existing_raw_alerts)
    if not args.skip_news:
        result = collect_naver_news(
            max_per_query=args.max_news_per_query,
            sleep_seconds=args.news_sleep_seconds,
            max_retries=args.news_max_retries,
        )
        raw_alerts.extend(result.alerts)
        collection_statuses.append(result.status)
    if not args.skip_dart:
        result = collect_open_dart(
            days=args.dart_days,
            pages=args.dart_pages,
            end_date=args.dart_end_date,
            sleep_seconds=args.dart_sleep_seconds,
        )
        raw_alerts.extend(result.alerts)
        collection_statuses.append(result.status)
    raw_alerts = merge_raw_alerts(raw_alerts)
    labeled_alerts = [label for alert in raw_alerts if (label := weak_label(alert))]
    should_write = should_write_raw_alerts(
        existing_count=len(existing_raw_alerts),
        next_count=len(raw_alerts),
        force=args.force_overwrite_raw,
    )

    if should_write:
        write_jsonl(RAW_ALERTS_PATH, [raw_alert_to_dict(alert) for alert in raw_alerts])
        write_jsonl(
            WEAK_LABELED_PATH,
            [
                {
                    "text": alert.text,
                    "tags": alert.tags,
                    "sentiment": alert.sentiment,
                    "importance": alert.importance,
                    "source_type": alert.source_type,
                }
                for alert in labeled_alerts
            ],
        )

    report = {
        "raw_write_status": "written" if should_write else "skipped_existing_dataset_is_larger",
        "existing_raw_count": len(existing_raw_alerts),
        "raw_count": len(raw_alerts),
        "weak_labeled_count": len(labeled_alerts),
        "providers": {
            "naver-news": sum(1 for alert in raw_alerts if alert.provider == "naver-news"),
            "open-dart": sum(1 for alert in raw_alerts if alert.provider == "open-dart"),
        },
        "provider_statuses": collection_status_to_dict(collection_statuses),
    }
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
