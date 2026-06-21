# API Standard

모든 비즈니스 REST API는 공통 응답 envelope를 사용한다. `/health`처럼 인프라가 직접 소비하는 운영 endpoint는 예외로 둔다.

## Success Response

```json
{
  "success": true,
  "status": 200,
  "code": "COMMON_000",
  "message": "OK",
  "data": {},
  "timestamp": "2026-06-18T00:00:00Z"
}
```

## Error Response

```json
{
  "success": false,
  "status": 422,
  "code": "COMMON_002",
  "message": "Request validation failed",
  "errors": [
    {
      "field": "body.title",
      "reason": "String should have at least 1 character"
    }
  ],
  "timestamp": "2026-06-18T00:00:00Z"
}
```

## Error Codes

| Code | HTTP | Meaning |
| --- | --- | --- |
| `COMMON_000` | 200 | Success |
| `COMMON_001` | 400 | Invalid request |
| `COMMON_002` | 422 | Validation failed |
| `COMMON_999` | 500 | Internal server error |
| `AI_001` | 503 | AI model unavailable |

## Swagger

- Swagger UI: `/docs`
- OpenAPI JSON: `/openapi.json`

## Foreign Ownership Timeseries Prediction

- `POST /api/v1/market/foreign-ownership/predict`
- Hana-OmniLens-API가 수집한 KIS 외국인 보유 snapshot, 일별 시계열, KIS WebSocket 장중 누적 거래량을 받아 외국인 한도소진율 예측 boundary를 반환한다.
- 요청 `data`에는 `stock_code`, `side`, `quantity`, `foreign_owned_quantity`, `foreign_ownership_rate`, `foreign_limit_quantity`, `foreign_limit_exhaustion_rate`, `base_date`, `observed_intraday_volume`, `history[]`를 포함한다.
- 응답 `data`에는 `min_foreign_limit_exhaustion_rate`, `base_foreign_limit_exhaustion_rate`, `max_foreign_limit_exhaustion_rate`, `order_impact_rate`, `trend_daily_change_rate`, `history_observation_count`, `confidence_level`, `confidence_score`, `model_version`, `source`를 포함한다.
- 이 endpoint는 confidence와 boundary를 관측/표시용으로 제공하며 주문 차단 결정을 반환하지 않는다.

## Tax Document Verification

- `POST /api/v1/tax/documents/verify`
- 외부 OCR/MTS upload gateway가 추출한 텍스트와 위변조 signal을 받아 공통 envelope로 서류 검증 결과를 반환한다.
- 응답 `data.verification_status`는 `VERIFIED`, `PENDING`, `REJECTED` 중 하나이며, `fraud_risk_score`, `risk_level`, `manual_review_required`, `missing_required_fields`, `rejection_reasons`, `document_model_version`을 포함한다.
- 이 endpoint는 원본 파일을 저장하거나 국세청 제출을 수행하지 않는다.
