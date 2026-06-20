from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

SERVICE_READINESS_REPORT_SCHEMA_VERSION = "service-readiness-report/v1"
MINIMUM_TRAINING_REFERENCE_STOCK_COUNT = 3_000
MINIMUM_EVALUATION_REFERENCE_STOCK_COUNT = 500
MINIMUM_TRAINABLE_STOCK_NAME_ACCURACY = 0.99


def build_service_readiness_report(
    *,
    model_release_report: dict[str, Any],
    live_news_monitoring_status: dict[str, Any],
    full_universe_coverage_report: dict[str, Any],
    stock_coverage_report: dict[str, Any],
    stock_linker_training_report: dict[str, Any],
    pseudo_label_monitoring_report: dict[str, Any],
    confidence_calibration_report: dict[str, Any],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    model_version = str(model_release_report.get("model_version") or "")
    checks = {
        "model_release_gate": _equals_check(
            actual=model_release_report.get("overall_status"),
            expected="pass",
        ),
        "bootstrap_service_readiness": _equals_check(
            actual=model_release_report.get("service_readiness", {}).get("overall_status"),
            expected="pass",
        ),
        "audited_gold_readiness": _equals_check(
            actual=model_release_report.get("audited_gold_readiness", {}).get("overall_status"),
            expected="pass",
        ),
        "live_news_monitoring": _equals_check(
            actual=live_news_monitoring_status.get("overall_status"),
            expected="pass",
        ),
        "confidence_policy_observe_only": _equals_check(
            actual=live_news_monitoring_status.get("policy", {}).get("confidence_usage"),
            expected="observe_only",
        ),
        "full_universe_reference_coverage": _full_universe_coverage_check(
            full_universe_coverage_report
        ),
        "stock_reference_coverage": _stock_reference_coverage_check(stock_coverage_report),
        "stock_linker_coverage": _stock_linker_coverage_check(stock_linker_training_report),
        "pseudo_label_monitoring": _equals_check(
            actual=pseudo_label_monitoring_report.get("overall_status"),
            expected="pass",
        ),
        "confidence_calibration_report": _confidence_calibration_check(
            confidence_calibration_report,
            model_version,
        ),
    }
    overall_status = (
        "pass" if all(check["status"] == "pass" for check in checks.values()) else "fail"
    )
    return {
        "schema_version": SERVICE_READINESS_REPORT_SCHEMA_VERSION,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "overall_status": overall_status,
        "model_version": model_version,
        "checks": checks,
        "policy": {
            "confidence_usage": "observe_only",
            "description": (
                "confidence 값은 품질 관측과 UI 표시용 메타데이터로만 제공한다. "
                "Hannah는 신뢰도 기반 자동 차단 결정을 만들지 않는다."
            ),
        },
        "continuous_operations": {
            "human_gold_increment": (
                "사람 검수 gold label은 운영 로그와 월별 증분 수집으로 계속 확대한다."
            ),
            "drift_monitoring": (
                "live-news smoke/drift 리포트가 stale 또는 attention이면 운영 credential 환경에서 "
                "배치를 재생성하고 release 전 원인을 확인한다."
            ),
            "rollback": (
                "model-release-report와 service-readiness-report가 pass였던 "
                "직전 artifact로 되돌린다."
            ),
        },
        "required_action": _required_action(overall_status),
    }


def _equals_check(*, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "actual": actual,
        "expected": expected,
        "status": "pass" if actual == expected else "fail",
    }


def _full_universe_coverage_check(report: dict[str, Any]) -> dict[str, Any]:
    valid_universe_count = int(report.get("valid_numeric_universe_count") or 0)
    full_coverage_count = int(report.get("full_coverage_stock_count") or 0)
    missing_count = int(report.get("missing_stock_count_after_generation") or 0)
    status = (
        "pass"
        if missing_count == 0 and full_coverage_count >= valid_universe_count > 0
        else "fail"
    )
    return {
        "valid_numeric_universe_count": valid_universe_count,
        "full_coverage_stock_count": full_coverage_count,
        "missing_stock_count_after_generation": missing_count,
        "review_status": report.get("review_status"),
        "status": status,
    }


def _stock_reference_coverage_check(report: dict[str, Any]) -> dict[str, Any]:
    gate_status = report.get("coverage_gates", {}).get("overall_status")
    training_stock_count = int(report.get("training_stock_count") or 0)
    evaluation_stock_count = int(report.get("evaluation_stock_count") or 0)
    status = (
        "pass"
        if gate_status == "pass"
        and training_stock_count >= MINIMUM_TRAINING_REFERENCE_STOCK_COUNT
        and evaluation_stock_count >= MINIMUM_EVALUATION_REFERENCE_STOCK_COUNT
        else "fail"
    )
    return {
        "coverage_gate_status": gate_status,
        "training_stock_count": training_stock_count,
        "minimum_training_stock_count": MINIMUM_TRAINING_REFERENCE_STOCK_COUNT,
        "evaluation_stock_count": evaluation_stock_count,
        "minimum_evaluation_stock_count": MINIMUM_EVALUATION_REFERENCE_STOCK_COUNT,
        "status": status,
    }


def _stock_linker_coverage_check(report: dict[str, Any]) -> dict[str, Any]:
    evaluation = report.get("evaluation", {})
    all_code_accuracy = float(evaluation.get("all_stock_code_template_accuracy") or 0.0)
    trainable_name_accuracy = float(
        evaluation.get("trainable_stock_name_template_accuracy") or 0.0
    )
    coverage_gate_status = report.get("coverage_gate", {}).get("status")
    status = (
        "pass"
        if coverage_gate_status == "pass"
        and all_code_accuracy == 1.0
        and trainable_name_accuracy >= MINIMUM_TRAINABLE_STOCK_NAME_ACCURACY
        else "fail"
    )
    return {
        "coverage_gate_status": coverage_gate_status,
        "all_stock_code_template_accuracy": all_code_accuracy,
        "required_all_stock_code_template_accuracy": 1.0,
        "trainable_stock_name_template_accuracy": trainable_name_accuracy,
        "minimum_trainable_stock_name_template_accuracy": MINIMUM_TRAINABLE_STOCK_NAME_ACCURACY,
        "status": status,
    }


def _confidence_calibration_check(report: dict[str, Any], model_version: str) -> dict[str, Any]:
    datasets = report.get("datasets", {})
    required_datasets = {"benchmark", "real_news_gold"}
    missing_datasets = sorted(required_datasets - set(datasets))
    status = (
        "pass"
        if report.get("schema_version") == "model-confidence-calibration/v1"
        and report.get("model_version") == model_version
        and not missing_datasets
        else "fail"
    )
    return {
        "schema_version": report.get("schema_version"),
        "model_version": report.get("model_version"),
        "expected_model_version": model_version,
        "required_datasets": sorted(required_datasets),
        "missing_datasets": missing_datasets,
        "status": status,
    }


def _required_action(overall_status: str) -> str:
    if overall_status == "pass":
        return (
            "추가 조치 없음. 현재 release, coverage, live-news, confidence 정책이 "
            "서비스 readiness gate를 통과했다."
        )
    return (
        "fail 상태 check를 수정한 뒤 관련 리포트를 재생성하고 "
        "service readiness gate를 다시 실행한다."
    )
