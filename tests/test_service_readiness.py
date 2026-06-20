import json
from datetime import UTC, datetime
from pathlib import Path

from hannah_montana_ai.training.service_readiness import (
    SERVICE_READINESS_REPORT_SCHEMA_VERSION,
    build_service_readiness_report,
)


def test_service_readiness_report_passes_current_committed_reports() -> None:
    report = build_service_readiness_report(
        model_release_report=_read_json(Path("reports/model-release-report.json")),
        live_news_monitoring_status=_read_json(Path("reports/live-news-monitoring-status.json")),
        full_universe_coverage_report=_read_json(
            Path("reports/full-universe-codex-coverage-report.json")
        ),
        stock_coverage_report=_read_json(Path("reports/stock-coverage-report.json")),
        stock_linker_training_report=_read_json(Path("reports/stock-linker-training-report.json")),
        pseudo_label_monitoring_report=_read_json(
            Path("reports/pseudo-label-promotion-monitoring.json")
        ),
        confidence_calibration_report=_read_json(Path("reports/model-confidence-calibration.json")),
        generated_at=datetime(2026, 6, 20, tzinfo=UTC),
    )

    assert report["schema_version"] == SERVICE_READINESS_REPORT_SCHEMA_VERSION
    assert report["overall_status"] == "pass"
    assert all(check["status"] == "pass" for check in report["checks"].values())
    assert report["policy"]["confidence_usage"] == "observe_only"
    assert "자동 차단 결정을 만들지 않는다" in report["policy"]["description"]


def test_service_readiness_report_fails_when_live_news_is_stale() -> None:
    live_status = _read_json(Path("reports/live-news-monitoring-status.json"))
    live_status["overall_status"] = "stale"

    report = build_service_readiness_report(
        model_release_report=_read_json(Path("reports/model-release-report.json")),
        live_news_monitoring_status=live_status,
        full_universe_coverage_report=_read_json(
            Path("reports/full-universe-codex-coverage-report.json")
        ),
        stock_coverage_report=_read_json(Path("reports/stock-coverage-report.json")),
        stock_linker_training_report=_read_json(Path("reports/stock-linker-training-report.json")),
        pseudo_label_monitoring_report=_read_json(
            Path("reports/pseudo-label-promotion-monitoring.json")
        ),
        confidence_calibration_report=_read_json(Path("reports/model-confidence-calibration.json")),
        generated_at=datetime(2026, 6, 20, tzinfo=UTC),
    )

    assert report["overall_status"] == "fail"
    assert report["checks"]["live_news_monitoring"]["status"] == "fail"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
