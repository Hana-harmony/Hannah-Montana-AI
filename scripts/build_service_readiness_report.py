import argparse
import json
from pathlib import Path
from typing import Any

from hannah_montana_ai.training.service_readiness import build_service_readiness_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_RELEASE_REPORT_PATH = PROJECT_ROOT / "reports/model-release-report.json"
DEFAULT_LIVE_NEWS_MONITORING_STATUS_PATH = (
    PROJECT_ROOT / "reports/live-news-monitoring-status.json"
)
DEFAULT_FULL_UNIVERSE_COVERAGE_REPORT_PATH = (
    PROJECT_ROOT / "reports/full-universe-codex-coverage-report.json"
)
DEFAULT_STOCK_COVERAGE_REPORT_PATH = PROJECT_ROOT / "reports/stock-coverage-report.json"
DEFAULT_STOCK_LINKER_TRAINING_REPORT_PATH = (
    PROJECT_ROOT / "reports/stock-linker-training-report.json"
)
DEFAULT_PSEUDO_LABEL_MONITORING_REPORT_PATH = (
    PROJECT_ROOT / "reports/pseudo-label-promotion-monitoring.json"
)
DEFAULT_CONFIDENCE_CALIBRATION_REPORT_PATH = (
    PROJECT_ROOT / "reports/model-confidence-calibration.json"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports/service-readiness-report.json"


def main() -> None:
    args = _parse_args()
    report = build_service_readiness_report(
        model_release_report=_read_json(_project_path(args.model_release_report)),
        live_news_monitoring_status=_read_json(
            _project_path(args.live_news_monitoring_status)
        ),
        full_universe_coverage_report=_read_json(
            _project_path(args.full_universe_coverage_report)
        ),
        stock_coverage_report=_read_json(_project_path(args.stock_coverage_report)),
        stock_linker_training_report=_read_json(
            _project_path(args.stock_linker_training_report)
        ),
        pseudo_label_monitoring_report=_read_json(
            _project_path(args.pseudo_label_monitoring_report)
        ),
        confidence_calibration_report=_read_json(
            _project_path(args.confidence_calibration_report)
        ),
    )
    output_path = _project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final Hannah AI service readiness report."
    )
    parser.add_argument(
        "--model-release-report",
        type=Path,
        default=DEFAULT_MODEL_RELEASE_REPORT_PATH,
    )
    parser.add_argument(
        "--live-news-monitoring-status",
        type=Path,
        default=DEFAULT_LIVE_NEWS_MONITORING_STATUS_PATH,
    )
    parser.add_argument(
        "--full-universe-coverage-report",
        type=Path,
        default=DEFAULT_FULL_UNIVERSE_COVERAGE_REPORT_PATH,
    )
    parser.add_argument(
        "--stock-coverage-report",
        type=Path,
        default=DEFAULT_STOCK_COVERAGE_REPORT_PATH,
    )
    parser.add_argument(
        "--stock-linker-training-report",
        type=Path,
        default=DEFAULT_STOCK_LINKER_TRAINING_REPORT_PATH,
    )
    parser.add_argument(
        "--pseudo-label-monitoring-report",
        type=Path,
        default=DEFAULT_PSEUDO_LABEL_MONITORING_REPORT_PATH,
    )
    parser.add_argument(
        "--confidence-calibration-report",
        type=Path,
        default=DEFAULT_CONFIDENCE_CALIBRATION_REPORT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
