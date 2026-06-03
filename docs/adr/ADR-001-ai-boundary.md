# ADR-001 AI 서비스 경계

## 결정
Hannah-Montana-AI는 뉴스·공시 분석 전용 FastAPI 서비스로 분리한다.

## 이유
- Hana-OmniLens-API는 협력사 연동 REST/WebSocket API를 담당한다.
- AI 모델 학습, 평가, 배포는 API 게이트웨이성 서비스와 변경 주기가 다르다.

## 영향
- 이 레포는 주문, 모의 투자, 사용자 알림 저장을 구현하지 않는다.
- 분석 결과는 Hana-OmniLens-API가 알림 payload에 포함해 협력사로 전송한다.
