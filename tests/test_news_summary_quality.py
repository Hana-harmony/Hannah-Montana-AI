from hannah_montana_ai.domain.schemas import AlertAnalysisRequest, StockCandidate
from hannah_montana_ai.services.analyzer import AlertAnalyzer
from hannah_montana_ai.services.rule_engine import FinancialRuleEngine


def test_summary_ignores_news_site_navigation_noise() -> None:
    engine = FinancialRuleEngine()
    content = (
        "SK하이닉스, 삼성전자 시총 맹추격 로그인 회원가입 전체 메뉴 열기 검색 열기 "
        "머니 증권 은행 보험 카드 부동산 경제일반 산업 재계 자동차 전기전자. "
        "22일 한국거래소에 따르면 이날 오전 11시 기준 SK하이닉스의 주가는 "
        "전 거래일 대비 4.59% 상승한 289만1000원에 거래되고 있다. "
        "이는 연초 대비 삼성전자는 195.25% 오른 반면 SK하이닉스는 324.58% "
        "급등한 영향이다. "
        "시장에서는 HBM 수요와 반도체 업황 개선이 시가총액 격차를 좁히는 "
        "핵심 배경으로 거론된다. "
        "오늘의 NEWS STAND 30대 교사 부부 대박 주식 이야기와 소름 돋는 폭탄 발언."
    )

    summary = engine.summarize_what_why_impact(
        "SK하이닉스, 삼성전자 시총 맹추격",
        "",
        content,
        "HIGH",
        "POSITIVE",
    )

    assert "로그인" not in summary.what
    assert "전체 메뉴" not in summary.what
    assert "NEWS STAND" not in summary.impact
    assert "SK하이닉스" in summary.what
    assert "급등" in summary.why or "HBM" in summary.why
    assert "사용자" in summary.impact or "시장" in summary.impact


def test_clean_article_text_keeps_financial_sentences_in_original_order() -> None:
    engine = FinancialRuleEngine()
    content = (
        "본문 바로가기 로그인 회원가입 전체 메뉴 열기 검색 열기. "
        "삼성전자는 AI 서버 투자 확대로 반도체 실적 개선 기대가 커졌다. "
        "메모리 가격 반등과 HBM 공급 확대가 주요 배경이다. "
        "투자자는 영업이익 회복 속도와 수요 지속성을 확인해야 한다. "
        "이용약관 개인정보 처리방침 저작권 안내."
    )

    cleaned = engine.clean_article_text(content, "삼성전자 실적 개선")

    assert "본문 바로가기" not in cleaned
    assert "이용약관" not in cleaned
    assert cleaned.index("삼성전자는") < cleaned.index("메모리 가격")
    assert "영업이익" in cleaned


def test_analyzer_prefers_first_internal_stock_over_limited_request_universe() -> None:
    analyzer = AlertAnalyzer()
    response = analyzer.analyze(
        AlertAnalysisRequest(
            source_type="NEWS",
            title="NH-Amundi운용, 반도체 ETF 리밸런싱...SK스퀘어 신규 편입",
            snippet="SK하이닉스와 삼성전자 등 반도체 종목을 담는 ETF가 정기 리밸런싱을 마쳤다.",
            content=(
                "NH-Amundi자산운용은 국내 반도체 산업을 대표하는 ETF에 SK스퀘어를 "
                "신규 편입했다고 밝혔다. SK하이닉스 주가 상승과 HBM 수요 확대가 "
                "반도체 투자 심리를 끌어올리고 있다."
            ),
            original_url="https://news.example.com/article",
            stock_universe=[
                StockCandidate(
                    stock_code="000660",
                    stock_name="SK하이닉스",
                    stock_name_en="SK hynix",
                )
            ],
        )
    )

    assert response.stock_code == "402340"
    assert response.stock_name == "SK스퀘어"
