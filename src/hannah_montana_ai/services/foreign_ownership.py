from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from hannah_montana_ai.domain.schemas import (
    ForeignOwnershipHistoryPoint,
    ForeignOwnershipTimeseriesPredictionRequest,
    ForeignOwnershipTimeseriesPredictionResponse,
)

MODEL_VERSION = "hannah-foreign-ownership-timeseries-v1"
SNAPSHOT_ONLY_UNCERTAINTY_RATE = 0.05
MAX_INTRADAY_UNCERTAINTY_RATE = 0.5
MAX_HISTORY_UNCERTAINTY_RATE = 1.5
INTRADAY_VOLUME_WEIGHT = 0.05


class ForeignOwnershipTimeseriesPredictionService:
    def predict(
        self,
        request: ForeignOwnershipTimeseriesPredictionRequest,
    ) -> ForeignOwnershipTimeseriesPredictionResponse:
        history = _sorted_history(request.history)
        trend = _trend_stats(history)
        order_impact_rate = _order_impact_rate(
            request.side,
            request.quantity,
            request.foreign_limit_quantity,
        )
        base_rate = _round_rate(
            request.foreign_limit_exhaustion_rate + order_impact_rate + trend.daily_change_rate
        )
        intraday_uncertainty_rate = max(
            _intraday_uncertainty_rate(
                request.foreign_limit_quantity,
                request.observed_intraday_volume,
            ),
            trend.uncertainty_rate,
        )
        min_rate = _round_rate(max(0.0, base_rate - intraday_uncertainty_rate))
        max_rate = _round_rate(base_rate + intraday_uncertainty_rate)
        confidence = _confidence(request.observed_intraday_volume, trend.observation_count)

        return ForeignOwnershipTimeseriesPredictionResponse(
            stock_code=request.stock_code,
            min_foreign_limit_exhaustion_rate=min_rate,
            base_foreign_limit_exhaustion_rate=base_rate,
            max_foreign_limit_exhaustion_rate=max_rate,
            order_impact_rate=order_impact_rate,
            intraday_uncertainty_rate=_round_rate(intraday_uncertainty_rate),
            observed_intraday_volume=request.observed_intraday_volume,
            trend_daily_change_rate=trend.daily_change_rate,
            history_observation_count=trend.observation_count,
            history_window_days=trend.window_days,
            base_date=request.base_date,
            calculated_at=datetime.now(UTC),
            confidence_level=confidence.level,
            confidence_score=confidence.score,
            model_version=MODEL_VERSION,
            source=_source(request.observed_intraday_volume, trend.observation_count),
        )


@dataclass(frozen=True)
class _TrendStats:
    daily_change_rate: float
    uncertainty_rate: float
    observation_count: int
    window_days: int


@dataclass(frozen=True)
class _Confidence:
    level: str
    score: float


def _sorted_history(
    history: list[ForeignOwnershipHistoryPoint],
) -> list[ForeignOwnershipHistoryPoint]:
    return sorted(history, key=lambda point: point.base_date)


def _trend_stats(history: list[ForeignOwnershipHistoryPoint]) -> _TrendStats:
    if len(history) < 2:
        return _TrendStats(0.0, 0.0, len(history), 0)

    first = history[0]
    last = history[-1]
    window_days = max(1, (last.base_date - first.base_date).days)
    daily_change_rate = _round_rate(
        (last.foreign_limit_exhaustion_rate - first.foreign_limit_exhaustion_rate) / window_days
    )
    uncertainty_rate = min(
        MAX_HISTORY_UNCERTAINTY_RATE,
        _average_absolute_daily_change(history),
    )
    return _TrendStats(
        daily_change_rate=daily_change_rate,
        uncertainty_rate=_round_rate(uncertainty_rate),
        observation_count=len(history),
        window_days=window_days,
    )


def _average_absolute_daily_change(history: list[ForeignOwnershipHistoryPoint]) -> float:
    total = 0.0
    intervals = 0
    for previous, current in zip(history, history[1:], strict=False):
        days = max(1, (current.base_date - previous.base_date).days)
        daily_change = (
            current.foreign_limit_exhaustion_rate
            - previous.foreign_limit_exhaustion_rate
        )
        total += abs(daily_change) / days
        intervals += 1
    return 0.0 if intervals == 0 else total / intervals


def _order_impact_rate(side: str, quantity: int, foreign_limit_quantity: int) -> float:
    if side != "BUY" or quantity <= 0 or foreign_limit_quantity <= 0:
        return 0.0
    return _round_rate(quantity * 100 / foreign_limit_quantity)


def _intraday_uncertainty_rate(
    foreign_limit_quantity: int,
    observed_intraday_volume: int,
) -> float:
    if foreign_limit_quantity <= 0:
        return 0.0
    if observed_intraday_volume <= 0:
        return SNAPSHOT_ONLY_UNCERTAINTY_RATE
    volume_rate = observed_intraday_volume * 100 / foreign_limit_quantity * INTRADAY_VOLUME_WEIGHT
    return min(MAX_INTRADAY_UNCERTAINTY_RATE, _round_rate(volume_rate))


def _confidence(observed_intraday_volume: int, observation_count: int) -> _Confidence:
    has_realtime_volume = observed_intraday_volume > 0
    if observation_count >= 5 and has_realtime_volume:
        return _Confidence("AI_TIME_SERIES_REALTIME_ADJUSTED", 0.88)
    if observation_count >= 5:
        return _Confidence("AI_TIME_SERIES_ADJUSTED", 0.8)
    if observation_count >= 2:
        return _Confidence("AI_LIMITED_TIME_SERIES", 0.62)
    if has_realtime_volume:
        return _Confidence("AI_REALTIME_VOLUME_ADJUSTED", 0.57)
    return _Confidence("AI_SNAPSHOT_ONLY", 0.46)


def _source(observed_intraday_volume: int, observation_count: int) -> str:
    source = "HANNAH_MONTANA_AI_FOREIGN_OWNERSHIP"
    if observation_count >= 2:
        source += "+DAILY_TIMESERIES"
    if observed_intraday_volume > 0:
        source += "+KIS_WEBSOCKET_TRADE_VOLUME"
    return source


def _round_rate(value: float) -> float:
    return round(value, 6)
