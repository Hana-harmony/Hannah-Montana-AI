from functools import lru_cache

from fastapi import APIRouter

from hannah_montana_ai.domain.schemas import AlertAnalysisRequest, AlertAnalysisResponse
from hannah_montana_ai.services.analyzer import AlertAnalyzer

router = APIRouter(tags=["analysis"])


@lru_cache
def get_analyzer() -> AlertAnalyzer:
    return AlertAnalyzer()


@router.post("/alerts/analyze", response_model=AlertAnalysisResponse)
def analyze_alert(request: AlertAnalysisRequest) -> AlertAnalysisResponse:
    return get_analyzer().analyze(request)
