import argparse
import json
from pathlib import Path

from hannah_montana_ai.training.stock_curation import (
    build_stock_training_candidates,
    write_stock_curation_report,
    write_stock_training_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ALERTS_PATH = PROJECT_ROOT / "data/raw/collected_alerts.jsonl"
STOCK_UNIVERSE_PATH = PROJECT_ROOT / "data/reference/korea_stock_universe.csv"
CANDIDATE_QUEUE_PATH = PROJECT_ROOT / "data/curation/stock_training_candidate_queue.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports/stock-training-candidate-report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-alerts", type=Path, default=RAW_ALERTS_PATH)
    parser.add_argument("--stock-universe", type=Path, default=STOCK_UNIVERSE_PATH)
    parser.add_argument("--output", type=Path, default=CANDIDATE_QUEUE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--per-stock-label-quota", type=int, default=2)
    parser.add_argument("--minimum-signal-score", type=int, default=3)
    parser.add_argument("--minimum-stock-count", type=int, default=300)
    args = parser.parse_args()

    result = build_stock_training_candidates(
        raw_alert_path=args.raw_alerts,
        stock_universe_path=args.stock_universe,
        per_stock_label_quota=args.per_stock_label_quota,
        minimum_signal_score=args.minimum_signal_score,
        minimum_stock_count=args.minimum_stock_count,
    )
    write_stock_training_candidates(args.output, result.candidates)
    write_stock_curation_report(args.report, result.report)
    print(json.dumps(result.report, ensure_ascii=False))


if __name__ == "__main__":
    main()
