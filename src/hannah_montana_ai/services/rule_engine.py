import re

from hannah_montana_ai.domain.schemas import Importance, Sentiment


class FinancialRuleEngine:
    critical_keywords = ("상장폐지", "거래정지", "횡령", "배임", "감사의견 거절")
    high_keywords = ("유상증자", "합병", "분할", "실적", "공급계약", "소송", "자사주")
    negative_keywords = ("하락", "손실", "적자", "감소", "리콜", "제재", "과징금")
    positive_keywords = ("상승", "흑자", "증가", "수주", "계약", "배당", "호실적")

    def classify_sentiment(self, text: str) -> Sentiment:
        negative_score = self._count_keywords(text, self.negative_keywords)
        positive_score = self._count_keywords(text, self.positive_keywords)
        if negative_score > positive_score:
            return "NEGATIVE"
        if positive_score > negative_score:
            return "POSITIVE"
        return "NEUTRAL"

    def classify_importance(self, text: str, source_type: str) -> Importance:
        if self._contains_any(text, self.critical_keywords):
            return "CRITICAL"
        if source_type == "DISCLOSURE" or self._contains_any(text, self.high_keywords):
            return "HIGH"
        if len(text) > 80:
            return "MEDIUM"
        return "LOW"

    def summarize(self, title: str, snippet: str) -> str:
        normalized = re.sub(r"\s+", " ", f"{title}. {snippet}").strip()
        return normalized[:220]

    def holder_target(self, importance: Importance) -> bool:
        return importance in {"HIGH", "CRITICAL"}

    def watchlist_target(self, importance: Importance) -> bool:
        return importance in {"MEDIUM", "HIGH", "CRITICAL"}

    def _contains_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _count_keywords(self, text: str, keywords: tuple[str, ...]) -> int:
        return sum(1 for keyword in keywords if keyword in text)
