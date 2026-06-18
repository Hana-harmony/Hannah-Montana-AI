import argparse
import json
from pathlib import Path

from hannah_montana_ai.training.stock_curation import (
    validate_stock_gold_coverage_review_packet,
    write_stock_gold_review_validation_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COVERAGE_REVIEW_PACKET_PATH = (
    PROJECT_ROOT / "data/curation/stock_gold_coverage_active_review_packet.jsonl"
)
REPORT_PATH = PROJECT_ROOT / "reports/stock-gold-coverage-validation-report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-review-packet",
        type=Path,
        default=COVERAGE_REVIEW_PACKET_PATH,
    )
    parser.add_argument("--training-stock-target", type=int, default=1_500)
    parser.add_argument("--evaluation-stock-target", type=int, default=500)
    parser.add_argument("--minimum-wave-approved-stocks", type=int, default=100)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    result = validate_stock_gold_coverage_review_packet(
        coverage_review_packet_path=args.coverage_review_packet,
        training_stock_target=args.training_stock_target,
        evaluation_stock_target=args.evaluation_stock_target,
        minimum_wave_approved_stocks=args.minimum_wave_approved_stocks,
    )
    write_stock_gold_review_validation_report(args.report, result.report)
    print(json.dumps(result.report, ensure_ascii=False))


if __name__ == "__main__":
    main()
