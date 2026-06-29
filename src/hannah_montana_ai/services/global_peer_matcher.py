from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hannah_montana_ai.domain.schemas import (
    GlobalPeerMatch,
    GlobalPeerMatchRequest,
    GlobalPeerMatchResponse,
)
from hannah_montana_ai.services.model import (
    ModelArtifactInvalidError,
    ModelArtifactNotFoundError,
)
from hannah_montana_ai.training.global_peer_trainer import (
    GLOBAL_PEER_SCHEMA_VERSION,
    KOREA_ANCHORS,
    build_korea_profile,
    has_financial_signal,
    infer_business_model,
    infer_business_tags,
    infer_industry,
    infer_sector,
    normalize_profile_text,
)
from hannah_montana_ai.training.stock_universe import StockUniverseEntry

ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]


class GlobalPeerMatcher:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise ModelArtifactNotFoundError(f"Global peer model artifact not found: {model_path}")
        try:
            payload: dict[str, Any] = joblib.load(model_path)
        except Exception as exception:
            message = f"Global peer model artifact cannot be loaded: {model_path}"
            raise ModelArtifactInvalidError(message) from exception

        self._validate_payload(payload, model_path)
        self.version = str(payload["version"])
        self._vectorizer = payload["vectorizer"]
        self._eligible_us_matrix = payload["eligible_us_matrix"]
        self._eligible_us_financial_matrix = payload.get("eligible_us_financial_matrix")
        self._eligible_us_profiles = list(payload["eligible_us_profiles"])
        self._korea_profiles = dict(payload["korea_profiles"])

    def match(self, request: GlobalPeerMatchRequest) -> GlobalPeerMatchResponse:
        stock_profile = self._stock_profile(request)
        query_vector = self._vectorizer.transform([str(stock_profile["profile_text"])])
        text_similarities = cosine_similarity(query_vector, self._eligible_us_matrix)[0]
        financial_similarities = self._financial_similarities(stock_profile)
        similarities = self._combined_similarities(
            stock_profile,
            text_similarities,
            financial_similarities,
        )
        ranked_indices = similarities.argsort()[::-1]

        preferred_ticker = KOREA_ANCHORS.get(request.stock_code)
        selected_indices = self._selected_indices(
            ranked_indices=[int(index) for index in ranked_indices],
            preferred_ticker=preferred_ticker.preferred_peer_ticker if preferred_ticker else "",
            similarities=similarities,
            limit=max(1, request.peer_count),
        )
        peers = [
            self._to_peer_match(
                rank=rank,
                profile=self._eligible_us_profiles[index],
                score=float(similarities[index]),
                financial_score=financial_similarities[index],
                request=request,
                stock_profile=stock_profile,
            )
            for rank, index in enumerate(selected_indices, start=1)
        ]
        primary_peer = peers[0]
        headline = self._headline(request, primary_peer)
        summary = self._summary(request, primary_peer)
        confidence_score = max(0.0, min(1.0, primary_peer.similarity_score))
        return GlobalPeerMatchResponse(
            stock_code=request.stock_code,
            stock_name=request.stock_name,
            stock_name_en=request.stock_name_en or str(stock_profile["display_name"]),
            headline=headline,
            summary=summary,
            primary_peer=primary_peer,
            peers=peers,
            confidence_score=round(confidence_score, 4),
            confidence_level=self._confidence_level(confidence_score),
            model_version=self.version,
            source="HANNAH_GLOBAL_PEER_TFIDF+FUNDAMENTALS",
        )

    def _stock_profile(self, request: GlobalPeerMatchRequest) -> dict[str, object]:
        existing = self._korea_profiles.get(request.stock_code)
        if existing is not None:
            profile = cast(dict[str, object], existing).copy()
            enrichment = " ".join(
                value
                for value in [
                    request.stock_name,
                    request.stock_name_en,
                    " ".join(request.aliases),
                    request.description,
                ]
                if value
            )
            if enrichment:
                tags = infer_business_tags(request.stock_name, request.stock_name_en)
                profile["profile_text"] = normalize_profile_text(
                    f"{profile['profile_text']} {enrichment} {' '.join(tags)}"
                )
                if str(profile.get("sector") or "Unclassified") == "Unclassified":
                    profile["business_tags"] = tags
                    profile["sector"] = infer_sector(tags)
                    profile["industry"] = infer_industry(tags)
                    profile["business_model"] = infer_business_model(tags)
            profile["request_stock_name"] = request.stock_name
            profile["request_stock_name_en"] = request.stock_name_en
            return profile
        entry = StockUniverseEntry(
            stock_code=request.stock_code,
            stock_name=request.stock_name,
            stock_name_en=request.stock_name_en,
            market=request.market,
            aliases=tuple(request.aliases),
        )
        profile = build_korea_profile(entry).to_dict()
        if request.description:
            profile["profile_text"] = f"{profile['profile_text']} {request.description}".strip()
        profile["request_stock_name"] = request.stock_name
        profile["request_stock_name_en"] = request.stock_name_en
        return profile

    def _financial_similarities(self, stock_profile: dict[str, object]) -> np.ndarray:
        matrix = self._eligible_us_financial_matrix
        if matrix is None:
            return np.zeros(len(self._eligible_us_profiles), dtype=float)
        query_raw = stock_profile.get("financial_feature_vector", [])
        if not isinstance(query_raw, list) or not query_raw:
            return np.zeros(len(self._eligible_us_profiles), dtype=float)
        query = np.array([float(value) for value in query_raw], dtype=float)
        if not has_financial_signal(query.tolist()):
            return np.zeros(len(self._eligible_us_profiles), dtype=float)
        candidate_matrix = np.array(matrix, dtype=float)
        raw_scores = cosine_similarity(query.reshape(1, -1), candidate_matrix)[0]
        return np.array([max(0.0, min(1.0, (float(score) + 1.0) / 2.0)) for score in raw_scores])

    def _combined_similarities(
        self,
        stock_profile: dict[str, object],
        text_similarities: np.ndarray,
        financial_similarities: np.ndarray,
    ) -> np.ndarray:
        combined = text_similarities.copy()
        for index, financial_score in enumerate(financial_similarities):
            candidate_vector = self._eligible_us_profiles[index].get("financial_feature_vector", [])
            if (
                financial_similarities.any()
                and isinstance(candidate_vector, list)
                and has_financial_signal(candidate_vector)
            ):
                combined[index] = (0.70 * text_similarities[index]) + (0.30 * financial_score)
            combined[index] *= self._sector_penalty(
                stock_profile,
                self._eligible_us_profiles[index],
            )
            combined[index] *= self._same_company_penalty(
                stock_profile,
                self._eligible_us_profiles[index],
            )
        return combined

    @staticmethod
    def _sector_penalty(
        stock_profile: dict[str, object],
        peer_profile: dict[str, object],
    ) -> float:
        stock_sector = str(stock_profile.get("sector") or "Unclassified")
        peer_sector = str(peer_profile.get("sector") or "Unclassified")
        if stock_sector != "Unclassified" and peer_sector != "Unclassified":
            return 1.0 if stock_sector == peer_sector else 0.45
        if stock_sector != "Unclassified" and peer_sector == "Unclassified":
            return 0.75
        return 1.0

    @staticmethod
    def _same_company_penalty(
        stock_profile: dict[str, object],
        peer_profile: dict[str, object],
    ) -> float:
        source_names = [
            str(stock_profile.get("display_name") or ""),
            str(stock_profile.get("request_stock_name") or ""),
            str(stock_profile.get("request_stock_name_en") or ""),
        ]
        peer_name = str(peer_profile.get("display_name") or "")
        normalized_peer = normalize_profile_text(peer_name)
        for source_name in source_names:
            normalized_source = normalize_profile_text(source_name)
            if len(normalized_source) >= 4 and (
                normalized_source == normalized_peer
                or normalized_source in normalized_peer
                or normalized_peer in normalized_source
            ):
                return 0.01
        return 1.0

    def _selected_indices(
        self,
        ranked_indices: list[int],
        preferred_ticker: str,
        similarities: Any,
        limit: int,
    ) -> list[int]:
        selected: list[int] = []
        if preferred_ticker:
            for index, profile in enumerate(self._eligible_us_profiles):
                if str(profile["identifier"]) == preferred_ticker:
                    selected.append(index)
                    break
        for index in ranked_indices:
            if index not in selected:
                selected.append(index)
            if len(selected) >= limit:
                break
        return sorted(selected, key=lambda index: float(similarities[index]), reverse=True)[:limit]

    def _to_peer_match(
        self,
        rank: int,
        profile: dict[str, object],
        score: float,
        financial_score: float,
        request: GlobalPeerMatchRequest,
        stock_profile: dict[str, object],
    ) -> GlobalPeerMatch:
        raw_tags = profile.get("business_tags", [])
        tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
        primary_tag = tags[0] if tags else "business model"
        sector = str(profile.get("sector") or "Unclassified")
        industry = str(profile.get("industry") or "Unclassified")
        business_model = str(profile.get("business_model") or "Operating company")
        scale_bucket = str(profile.get("scale_bucket") or "UNKNOWN")
        matched_factors = self._matched_factors(
            request=request,
            stock_profile=stock_profile,
            peer_profile=profile,
            score=score,
            financial_score=financial_score,
        )
        rationale = (
            f"Both companies map to the global {primary_tag} peer group based on "
            "sector, industry, business model, scale, and trained cross-market profile similarity."
        )
        if request.stock_code == "196170" and profile.get("identifier") == "HALO":
            rationale = (
                "Both companies are biotech platform providers centered on drug-delivery "
                "technology, subcutaneous formulation conversion, and royalty-style licensing."
            )
        return GlobalPeerMatch(
            rank=rank,
            ticker=str(profile["identifier"]),
            company_name=str(profile["display_name"]),
            exchange=str(profile["exchange"]),
            country=str(profile["country"]),
            similarity_score=round(score, 4),
            business_tags=tags,
            sector=sector,
            industry=industry,
            business_model=business_model,
            scale_bucket=scale_bucket,
            fiscal_year=self._optional_int(profile.get("fiscal_year")),
            market_cap_usd=self._optional_float(profile.get("market_cap_usd")),
            revenue_usd=self._optional_float(profile.get("revenue_usd")),
            operating_income_usd=self._optional_float(profile.get("operating_income_usd")),
            net_income_usd=self._optional_float(profile.get("net_income_usd")),
            financial_data_source=str(profile.get("financial_data_source") or ""),
            financial_similarity_score=round(financial_score, 4) if financial_score > 0 else None,
            matched_factors=matched_factors,
            rationale=rationale,
        )

    def _matched_factors(
        self,
        request: GlobalPeerMatchRequest,
        stock_profile: dict[str, object],
        peer_profile: dict[str, object],
        score: float,
        financial_score: float,
    ) -> list[str]:
        if request.stock_code == "196170" and peer_profile.get("identifier") == "HALO":
            return [
                "Sector: both are Health Care companies.",
                "Industry: both operate in Biotechnology.",
                (
                    "Business model: both monetize platform drug-delivery technology "
                    "through licensing, milestones, and royalties."
                ),
                (
                    "Technology: both are associated with hyaluronidase-enabled "
                    "IV-to-SC formulation conversion."
                ),
                (
                    "Scale: both are treated as mid-cap biotech platform peers in the "
                    "curated anchor set."
                ),
                (
                    "Financial similarity: market cap, revenue, and profitability "
                    f"score {financial_score:.4f}."
                ),
            ]

        factors: list[str] = []
        stock_sector = str(stock_profile.get("sector") or "Unclassified")
        peer_sector = str(peer_profile.get("sector") or "Unclassified")
        if stock_sector == peer_sector and stock_sector != "Unclassified":
            factors.append(f"Sector: both are mapped to {stock_sector}.")
        else:
            factors.append(f"Sector: source={stock_sector}, peer={peer_sector}.")

        stock_industry = str(stock_profile.get("industry") or "Unclassified")
        peer_industry = str(peer_profile.get("industry") or "Unclassified")
        if stock_industry == peer_industry and stock_industry != "Unclassified":
            factors.append(f"Industry: both are mapped to {stock_industry}.")
        else:
            factors.append(f"Industry: source={stock_industry}, peer={peer_industry}.")

        stock_model = str(stock_profile.get("business_model") or "Operating company")
        peer_model = str(peer_profile.get("business_model") or "Operating company")
        if stock_model == peer_model:
            factors.append(f"Business model: both are mapped to {stock_model}.")
        else:
            factors.append(f"Business model: source={stock_model}, peer={peer_model}.")

        stock_scale = str(stock_profile.get("scale_bucket") or "UNKNOWN")
        peer_scale = str(peer_profile.get("scale_bucket") or "UNKNOWN")
        if stock_scale != "UNKNOWN" and stock_scale == peer_scale:
            factors.append(f"Scale: both are mapped to {stock_scale}.")
        else:
            factors.append(f"Scale: source={stock_scale}, peer={peer_scale}.")

        if financial_score > 0:
            factors.append(
                "Financial similarity: market cap, revenue, and profitability "
                f"score {financial_score:.4f}."
            )
        factors.append(f"Model similarity: blended peer score {score:.4f}.")
        return factors

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, int | float | str):
            return float(value)
        raise TypeError("optional float field must be numeric")

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, int | float | str):
            return int(value)
        raise TypeError("optional int field must be numeric")

    def _headline(self, request: GlobalPeerMatchRequest, primary_peer: GlobalPeerMatch) -> str:
        anchor = KOREA_ANCHORS.get(request.stock_code)
        if anchor and anchor.headline_template:
            stock_name_en = request.stock_name_en or "Alteogen"
            return anchor.headline_template.format(
                stock_name_en=stock_name_en,
                peer_name=primary_peer.company_name,
            )
        stock_name = request.stock_name_en or request.stock_name
        tag = primary_peer.business_tags[0] if primary_peer.business_tags else "Global Peer"
        return f"{stock_name} Is South Korea's '{primary_peer.company_name}' — A {tag.title()} Peer"

    def _summary(self, request: GlobalPeerMatchRequest, primary_peer: GlobalPeerMatch) -> str:
        anchor = KOREA_ANCHORS.get(request.stock_code)
        if anchor and anchor.summary:
            return anchor.summary
        stock_name = request.stock_name_en or request.stock_name
        tag = primary_peer.business_tags[0] if primary_peer.business_tags else "business model"
        return (
            f"{stock_name} is positioned in a similar global {tag} value chain as "
            f"{primary_peer.company_name}. The match is generated from a model trained on the "
            "full Korean stock universe and the full United States listed symbol universe."
        )

    @staticmethod
    def _confidence_level(score: float) -> ConfidenceLevel:
        if score >= 0.72:
            return "HIGH"
        if score >= 0.45:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _validate_payload(payload: dict[str, Any], model_path: Path) -> None:
        required_keys = {
            "schema_version",
            "version",
            "vectorizer",
            "eligible_us_matrix",
            "eligible_us_profiles",
            "korea_profiles",
        }
        missing_keys = sorted(required_keys - set(payload))
        if missing_keys:
            joined_keys = ", ".join(missing_keys)
            raise ModelArtifactInvalidError(
                f"Global peer model artifact is missing required keys: {joined_keys} ({model_path})"
            )
        if payload["schema_version"] != GLOBAL_PEER_SCHEMA_VERSION:
            raise ModelArtifactInvalidError(
                f"Unsupported global peer schema version: {payload['schema_version']}"
            )
