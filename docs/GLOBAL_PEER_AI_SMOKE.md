# 글로벌 피어 AI Smoke 테스트

## 목적
- API 계약 테스트가 아니라 실제 한국 종목을 넣었을 때 AI 매칭 결과가 투자자가 이해할 수 있는 peer인지 점검한다.
- 동일 회사의 미국 ADR이 peer로 잡히는 중복 상장 노이즈를 배제하고, 섹터/산업/사업모델/규모/재무 유사도 근거가 함께 나오는지 확인한다.

## 최신 실행
- 리포트: `reports/global-peer-ai-smoke-report.json`
- 모델 버전: `global-peer-tfidf-20260629201516`
- 샘플 수: 15개 한국 대형/대표 종목

| 한국 종목 | AI primary peer | 핵심 근거 |
| --- | --- | --- |
| Samsung Electronics | Micron Technology | 반도체, 메모리/전자 제조, 메가캡 |
| SK hynix | Micron Technology | 메모리 반도체, DRAM/NAND/HBM |
| NAVER | Alphabet | 검색/광고/플랫폼, 인터넷 플랫폼 |
| Hyundai Motor | General Motors | 자동차 제조, 대형 완성차 |
| LG Energy Solution | Tesla | EV 배터리/에너지 저장 proxy, 대형 배터리 생태계 |
| Samsung Biologics | Thermo Fisher Scientific | 바이오 CDMO/생명과학 제조 서비스 |
| Celltrion | Biogen | 바이오의약품/바이오시밀러 |
| KB Financial Group | American Financial Group | 금융지주/금융서비스 |
| Shinhan Financial Group | American Financial Group | 금융지주/금융서비스 |
| Hana Financial Group | American Financial Group | 금융지주/금융서비스 |
| LG Chem | Dow | 화학/첨단소재 |
| Samsung SDI | American Battery Technology | 배터리/에너지 저장 |
| LG Electronics | Whirlpool | 가전/소비자 전자 |
| SK Telecom | Verizon Communications | 통신 네트워크/무선 가입 서비스 |
| Alteogen | Halozyme Therapeutics | 약물전달 플랫폼/로열티 라이선싱 |

## 품질 기준
- `tests/test_global_peer_matcher.py`의 core smoke regression은 SK hynix, NAVER, SK Telecom, LG Electronics, LG Energy Solution의 primary peer를 고정해 품질 퇴행을 막는다.
- 알테오젠은 Halozyme top1 anchor를 별도 release gate로 유지한다.
- 각 응답은 `matched_factors`에 섹터, 산업, 사업모델, 규모, 재무 유사도, 모델 유사도를 포함한다.

## 남은 한계
- 현재 peer universe는 미국 상장 종목 중심이라 CATL, Panasonic, Lonza처럼 더 직관적인 비미국 peer는 후보에 없다.
- 배터리 및 복합 대기업은 사업부별 매출 비중 feature가 없어서 segment-level peer보다 company-level proxy에 가깝다.
- 금융지주는 은행/보험/증권 segment 비중을 더 넣으면 peer 설명력이 개선될 수 있다.
