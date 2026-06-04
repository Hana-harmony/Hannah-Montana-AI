from functools import lru_cache
from time import perf_counter

from fastapi import APIRouter, HTTPException, status

from hannah_montana_ai.domain.schemas import AlertAnalysisRequest, AlertAnalysisResponse
from hannah_montana_ai.services.analyzer import AlertAnalyzer
from hannah_montana_ai.services.audit import AnalysisAuditLogger
from hannah_montana_ai.services.model import ModelArtifactError

router = APIRouter(tags=["analysis"])


@lru_cache
def get_analyzer() -> AlertAnalyzer:
    return AlertAnalyzer()


@lru_cache
def get_audit_logger() -> AnalysisAuditLogger:
    return AnalysisAuditLogger()


@router.post("/alerts/analyze", response_model=AlertAnalysisResponse)
def analyze_alert(request: AlertAnalysisRequest) -> AlertAnalysisResponse:
    started_at = perf_counter()
    audit_logger = get_audit_logger()
    try:
        analyzer = get_analyzer()
    except ModelArtifactError as exception:
        audit_logger.record_failure(
            request=request,
            latency_ms=_elapsed_ms(started_at),
            failure_reason="model_artifact_unavailable",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model artifact is unavailable",
        ) from exception
    try:
        response = analyzer.analyze(request)
    except Exception:
        audit_logger.record_failure(
            request=request,
            latency_ms=_elapsed_ms(started_at),
            failure_reason="analysis_error",
        )
        raise

    audit_logger.record_success(
        request=request,
        response=response,
        latency_ms=_elapsed_ms(started_at),
    )
    return response


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
