from pathlib import Path

from hannah_montana_ai.domain.schemas import GlobalPeerMatchRequest
from hannah_montana_ai.services.global_peer_matcher import GlobalPeerMatcher
from hannah_montana_ai.training.global_peer_trainer import load_us_stock_universe


def test_us_stock_universe_covers_full_listed_symbol_directory() -> None:
    entries = load_us_stock_universe(Path("data/reference/us_stock_universe.csv"))

    assert len(entries) >= 5_000
    assert any(entry.ticker == "HALO" for entry in entries)


def test_global_peer_model_matches_alteogen_to_halozyme() -> None:
    matcher = GlobalPeerMatcher(Path("src/hannah_montana_ai/model_store/global_peer_ml.joblib"))

    response = matcher.match(
        GlobalPeerMatchRequest(
            stock_code="196170",
            stock_name="알테오젠",
            stock_name_en="Alteogen",
            market="KOSDAQ",
        )
    )

    assert response.primary_peer.ticker == "HALO"
    assert response.primary_peer.company_name == "Halozyme Therapeutics"
    assert response.primary_peer.sector == "Health Care"
    assert response.primary_peer.industry == "Biotechnology"
    assert response.primary_peer.business_model == "Biotech platform licensing"
    assert response.primary_peer.scale_bucket == "MID_CAP"
    assert response.primary_peer.revenue_usd is not None
    assert response.primary_peer.operating_income_usd is not None
    assert "SEC_COMPANYFACTS" in response.primary_peer.financial_data_source
    assert "NASDAQ_SUMMARY_MARKET_CAP" in response.primary_peer.financial_data_source
    assert response.primary_peer.financial_similarity_score is not None
    assert any("Sector" in factor for factor in response.primary_peer.matched_factors)
    assert any("Scale" in factor for factor in response.primary_peer.matched_factors)
    assert any("Financial similarity" in factor for factor in response.primary_peer.matched_factors)
    assert response.confidence_level in {"MEDIUM", "HIGH"}
    assert "Alteogen Is The 'Halozyme Therapeutics'" in response.headline
    assert "drug-delivery technology" in response.summary
    assert response.model_version.startswith("global-peer-tfidf-")


def test_global_peer_model_quality_smoke_matches_core_korean_stocks() -> None:
    matcher = GlobalPeerMatcher(Path("src/hannah_montana_ai/model_store/global_peer_ml.joblib"))
    cases = [
        ("000660", "SK하이닉스", "SK hynix", "KOSPI", "MU"),
        ("035420", "NAVER", "NAVER", "KOSPI", "GOOGL"),
        ("017670", "SK텔레콤", "SK Telecom", "KOSPI", "VZ"),
        ("066570", "LG전자", "LG Electronics", "KOSPI", "WHR"),
        ("373220", "LG에너지솔루션", "LG Energy Solution", "KOSPI", "TSLA"),
    ]

    for stock_code, stock_name, stock_name_en, market, expected_ticker in cases:
        response = matcher.match(
            GlobalPeerMatchRequest(
                stock_code=stock_code,
                stock_name=stock_name,
                stock_name_en=stock_name_en,
                market=market,
            )
        )

        assert response.primary_peer.ticker == expected_ticker
        assert response.primary_peer.ticker not in {stock_name_en, stock_name}
        assert response.primary_peer.sector != "Unclassified"
        assert response.primary_peer.industry != "Unclassified"
        assert response.primary_peer.matched_factors
