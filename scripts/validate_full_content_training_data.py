import json
from pathlib import Path

from hannah_montana_ai.training.full_content_dataset import (
    build_full_content_dataset_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data/training/financial_alert_full_content_gold.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports/full-content-training-dataset-report.json"


def main() -> None:
    report = build_full_content_dataset_report(DATASET_PATH)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
