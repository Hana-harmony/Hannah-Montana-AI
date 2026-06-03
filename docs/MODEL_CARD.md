# 금융 NLP ML 모델 카드

## 모델명
`financial-ml-tfidf-logreg-20260603184019`

## 목적
- 한국 주식 뉴스·공시의 이벤트 태그, 감성, 중요도를 자체 ML 모델로 분류한다.
- Hana-OmniLens-API의 Watchlist News & Disclosure Alert API payload 생성에 사용한다.
- ChatGPT API나 외부 LLM에 의존하지 않는다.

## 입력
- source type: `NEWS` 또는 `DISCLOSURE`
- 제목
- snippet
- 후보 종목 목록

## 출력
- 대표 종목
- 이벤트 태그
- 감성: `POSITIVE`, `NEUTRAL`, `NEGATIVE`
- 중요도: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- 중복 제거 키
- 모델 버전

## 학습 데이터
- 수집 raw 위치: `data/raw/collected_alerts.jsonl` (gitignore)
- 약지도 라벨 위치: `data/processed/weak_labeled_alerts.jsonl` (gitignore)
- 수동 curated corpus: `data/training/financial_alert_corpus.jsonl`
- 합성 증강 corpus: `data/training/financial_alert_augmented.jsonl`
- curated gold benchmark: `data/evaluation/financial_alert_eval.jsonl`
- 이번 학습 샘플 수: 12,372
- 실제 수집 raw 총량: 14,101건
- 실제 수집 원천: OpenDART 공시검색 12,967건
- 실제 수집 원천: Naver News Search 1,134건
- 합성 증강 샘플 수: 960건
- gold benchmark 샘플 수: 768건
- 수집기는 429 rate limit과 5xx 장애에 대해 재시도와 지수 백오프를 수행한다.
- 수집 실패로 새 결과가 기존 raw 수보다 줄어들면 기본값으로 기존 코퍼스를 덮어쓰지 않는다.

## 학습 방식
- `scripts/collect_training_data.py`가 Naver News Search와 OpenDART에서 원문 제목·snippet·링크를 수집한다.
- `weak_labeler.py`가 수집 원문에 약지도 라벨을 부여해 학습 후보를 만든다.
- `scripts/build_augmented_training_data.py`가 저작권 문제가 없는 금융 문장 증강 corpus를 생성한다.
- `scripts/build_gold_evaluation_data.py`가 훈련셋과 별도 문장 패턴의 768건 benchmark를 생성한다.
- `scripts/train_ml_model.py`가 TF-IDF char n-gram feature와 Logistic Regression 기반 supervised ML 모델을 학습한다.
- 이벤트 태그는 One-vs-Rest multilabel classifier로 학습한다.
- 감성과 중요도는 다중 클래스 Logistic Regression으로 학습한다.
- 이벤트 태그 probability threshold는 평가셋 기준 과잉 태그를 줄이기 위해 0.35로 튜닝했다.
- 학습 시 전체 코퍼스를 80:20 holdout으로 나눠 검증한 뒤 전체 코퍼스로 최종 artifact를 재학습한다.
- 생성 artifact는 `src/hannah_montana_ai/model_store/financial_nlp_ml.joblib`이다.

## Holdout 검증 결과
- 위치: `reports/ml-training-report.json`
- 학습 split: 9,897건
- 검증 split: 2,475건
- 이벤트 subset recall: 0.9858585858585859
- 이벤트 macro F1: 0.9024190469377846
- 감성 accuracy: 0.9632323232323232
- 중요도 accuracy: 0.9608080808080808
- 라벨별 F1: `DISCLOSURE` 0.9906, `RISK` 0.9560, `GENERAL_MARKET` 0.9462, `CAPITAL_ACTION` 0.9250, `EARNINGS` 0.9091, `CORPORATE_ACTION` 0.8652, `CONTRACT` 0.8327, `MACRO` 0.7945
- 감성·중요도 confusion matrix를 함께 기록한다.

## Gold 평가 결과
- 위치: `reports/ml-model-evaluation.json`
- 평가 샘플 수: 768
- 이벤트 태그 recall: 0.8385416666666666
- 이벤트 태그 macro F1: 0.8592101121358015
- 감성 accuracy: 0.8854166666666666
- 중요도 accuracy: 0.8268229166666666
- 종목 매핑 accuracy: 1.0
- 라벨별 precision, recall, F1과 감성·중요도 confusion matrix를 함께 기록한다.

## 한계
- OpenDART 비중이 여전히 높아 뉴스 도메인 일반화는 추가 수집 후 재학습이 필요하다.
- 약지도 라벨은 대규모 bootstrapping 용도이며, 운영 전 사람이 검수한 실데이터 gold label set을 추가 확장해야 한다.
- 본문 전문을 저장·재배포하지 않고 제목·snippet 중심으로 학습한다.
- 실제 투자 판단을 위한 추천 모델이 아니다.

## 운영 전 필수 보강
- Naver 뉴스 수집 쿼리 확대와 일 단위 증분 수집
- 사람이 검수한 gold label과 약지도 label의 품질 비교
- 모델 drift 감시와 재학습 기준 정의
