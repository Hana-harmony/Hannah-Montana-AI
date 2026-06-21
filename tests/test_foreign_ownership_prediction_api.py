from fastapi.testclient import TestClient

from hannah_montana_ai.main import app


def test_foreign_ownership_prediction_uses_daily_timeseries() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/market/foreign-ownership/predict",
        json={
            "stock_code": "005930",
            "side": "BUY",
            "quantity": 1,
            "foreign_owned_quantity": 995,
            "foreign_ownership_rate": 49.75,
            "foreign_limit_quantity": 1000,
            "foreign_limit_exhaustion_rate": 99.5,
            "base_date": "2025-06-04",
            "observed_intraday_volume": 0,
            "history": [
                {
                    "base_date": "2025-05-31",
                    "foreign_owned_quantity": 980,
                    "foreign_ownership_rate": 49.0,
                    "foreign_limit_quantity": 1000,
                    "foreign_limit_exhaustion_rate": 98.0,
                },
                {
                    "base_date": "2025-06-01",
                    "foreign_owned_quantity": 985,
                    "foreign_ownership_rate": 49.25,
                    "foreign_limit_quantity": 1000,
                    "foreign_limit_exhaustion_rate": 98.5,
                },
                {
                    "base_date": "2025-06-02",
                    "foreign_owned_quantity": 989,
                    "foreign_ownership_rate": 49.45,
                    "foreign_limit_quantity": 1000,
                    "foreign_limit_exhaustion_rate": 98.9,
                },
                {
                    "base_date": "2025-06-03",
                    "foreign_owned_quantity": 992,
                    "foreign_ownership_rate": 49.6,
                    "foreign_limit_quantity": 1000,
                    "foreign_limit_exhaustion_rate": 99.2,
                },
                {
                    "base_date": "2025-06-04",
                    "foreign_owned_quantity": 995,
                    "foreign_ownership_rate": 49.75,
                    "foreign_limit_quantity": 1000,
                    "foreign_limit_exhaustion_rate": 99.5,
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["stock_code"] == "005930"
    assert data["base_foreign_limit_exhaustion_rate"] == 99.975
    assert data["max_foreign_limit_exhaustion_rate"] > 100.0
    assert data["history_observation_count"] == 5
    assert data["history_window_days"] == 4
    assert data["confidence_level"] == "AI_TIME_SERIES_ADJUSTED"
    assert data["confidence_score"] == 0.8
    assert data["model_version"] == "hannah-foreign-ownership-timeseries-v1"
    assert data["source"] == "HANNAH_MONTANA_AI_FOREIGN_OWNERSHIP+DAILY_TIMESERIES"


def test_foreign_ownership_prediction_keeps_confidence_observe_only() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/market/foreign-ownership/predict",
        json={
            "stock_code": "005930",
            "side": "BUY",
            "quantity": 1,
            "foreign_owned_quantity": 995,
            "foreign_ownership_rate": 49.75,
            "foreign_limit_quantity": 1000,
            "foreign_limit_exhaustion_rate": 99.5,
            "base_date": "2025-06-04",
            "observed_intraday_volume": 1_000_000,
            "history": [],
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["confidence_level"] == "AI_REALTIME_VOLUME_ADJUSTED"
    assert data["confidence_score"] == 0.57
    assert "blocking" not in data
    assert "orderable" not in data
