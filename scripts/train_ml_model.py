import json
from pathlib import Path

from hannah_montana_ai.training.ml_trainer import train_ml_model
from hannah_montana_ai.training.weak_distiller import (
    distill_weak_labeled_alerts,
    write_weak_distillation_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "src/hannah_montana_ai/model_store/financial_nlp_ml.joblib"
REPORT_PATH = PROJECT_ROOT / "reports/ml-training-report.json"
WEAK_LABEL_PATH = PROJECT_ROOT / "data/processed/weak_labeled_alerts.jsonl"
WEAK_DISTILLATION_REPORT_PATH = PROJECT_ROOT / "reports/weak-distillation-report.json"
TRAINING_PATHS = [
    PROJECT_ROOT / "data/training/financial_alert_corpus.jsonl",
    PROJECT_ROOT / "data/training/financial_alert_augmented.jsonl",
    PROJECT_ROOT / "data/training/financial_alert_news_style_augmented.jsonl",
    PROJECT_ROOT / "data/training/financial_alert_real_news_gold.jsonl",
]


def main() -> None:
    distillation = distill_weak_labeled_alerts(WEAK_LABEL_PATH)
    distillation_report = {
        **distillation.report,
        "promotion_status": "not_promoted_to_supervised_loss",
        "promotion_reason": (
            "distilled weak labels are tracked as candidates until gold quality gates "
            "confirm they do not reduce benchmark or real-news performance"
        ),
    }
    write_weak_distillation_report(
        distillation_report,
        WEAK_DISTILLATION_REPORT_PATH,
    )

    report = train_ml_model(TRAINING_PATHS, MODEL_PATH)
    REPORT_PATH.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
