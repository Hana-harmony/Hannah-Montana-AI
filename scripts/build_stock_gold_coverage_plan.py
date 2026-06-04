import argparse
import json
from pathlib import Path

from hannah_montana_ai.training.coverage_planner import (
    build_stock_gold_coverage_plan,
    write_stock_gold_coverage_plan_report,
    write_stock_gold_coverage_plan_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_QUEUE_PATH = PROJECT_ROOT / "data/curation/stock_training_candidate_queue.jsonl"
TRAINING_REVIEW_PATH = PROJECT_ROOT / "data/curation/stock_gold_training_review_batch.jsonl"
EVALUATION_REVIEW_PATH = (
    PROJECT_ROOT / "data/curation/stock_gold_evaluation_review_batch.jsonl"
)
PLAN_PATH = PROJECT_ROOT / "data/curation/stock_gold_coverage_review_plan.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports/stock-gold-coverage-plan-report.json"
TRAINING_PATHS = [
    PROJECT_ROOT / "data/training/financial_alert_corpus.jsonl",
    PROJECT_ROOT / "data/training/financial_alert_augmented.jsonl",
    PROJECT_ROOT / "data/training/financial_alert_news_style_augmented.jsonl",
    PROJECT_ROOT / "data/training/financial_alert_real_news_gold.jsonl",
    PROJECT_ROOT / "data/training/financial_alert_stock_review_gold.jsonl",
]
EVALUATION_PATHS = [
    PROJECT_ROOT / "data/evaluation/financial_alert_eval.jsonl",
    PROJECT_ROOT / "data/evaluation/financial_alert_real_disclosure_gold.jsonl",
    PROJECT_ROOT / "data/evaluation/financial_alert_real_news_gold.jsonl",
    PROJECT_ROOT / "data/evaluation/financial_alert_stock_review_gold.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-queue", type=Path, default=CANDIDATE_QUEUE_PATH)
    parser.add_argument("--training-review", type=Path, default=TRAINING_REVIEW_PATH)
    parser.add_argument("--evaluation-review", type=Path, default=EVALUATION_REVIEW_PATH)
    parser.add_argument("--plan-output", type=Path, default=PLAN_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--training-stock-target", type=int, default=1_500)
    parser.add_argument("--evaluation-stock-target", type=int, default=500)
    parser.add_argument("--review-wave-size", type=int, default=100)
    args = parser.parse_args()

    result = build_stock_gold_coverage_plan(
        candidate_path=args.candidate_queue,
        training_review_path=args.training_review,
        evaluation_review_path=args.evaluation_review,
        training_paths=TRAINING_PATHS,
        evaluation_paths=EVALUATION_PATHS,
        training_stock_target=args.training_stock_target,
        evaluation_stock_target=args.evaluation_stock_target,
        review_wave_size=args.review_wave_size,
    )
    write_stock_gold_coverage_plan_rows(args.plan_output, result.rows)
    write_stock_gold_coverage_plan_report(args.report, result.report)
    print(json.dumps(result.report, ensure_ascii=False))


if __name__ == "__main__":
    main()
