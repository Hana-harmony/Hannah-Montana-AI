from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from hannah_montana_ai.core.config import get_settings
from hannah_montana_ai.training.global_peer_trainer import (
    GlobalPeerFundamentals,
    fetch_krx_market_caps_usd,
    fetch_open_dart_annual_fundamentals,
    fetch_sec_annual_fundamentals,
    fetch_sec_ticker_cik_map,
    load_us_stock_universe,
    write_global_peer_fundamentals,
)
from hannah_montana_ai.training.stock_universe import load_env_file, load_stock_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports/global-peer-fundamentals-sync-report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=get_settings().global_peer_fundamentals_path)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "secrets.local.env")
    parser.add_argument("--fiscal-year", type=int, default=date.today().year - 1)
    parser.add_argument("--krw-usd-rate", type=float, default=0.00072)
    parser.add_argument("--krx-base-date", default=_previous_weekday().strftime("%Y%m%d"))
    parser.add_argument("--korea-stock-limit", type=int, default=0)
    parser.add_argument("--us-stock-limit", type=int, default=0)
    parser.add_argument("--korea-stock-codes", default="")
    parser.add_argument("--us-tickers", default="")
    args = parser.parse_args()

    load_env_file(args.env_file)
    rows: list[GlobalPeerFundamentals] = []
    failures: list[dict[str, str]] = []

    korea_rows, korea_failures = sync_korea_fundamentals(
        fiscal_year=args.fiscal_year,
        krw_usd_rate=args.krw_usd_rate,
        krx_base_date=args.krx_base_date,
        stock_limit=args.korea_stock_limit,
        stock_codes=_csv_set(args.korea_stock_codes),
    )
    rows.extend(korea_rows)
    failures.extend(korea_failures)

    us_rows, us_failures = sync_us_fundamentals(
        fiscal_year=args.fiscal_year,
        stock_limit=args.us_stock_limit,
        tickers=_csv_set(args.us_tickers),
    )
    rows.extend(us_rows)
    failures.extend(us_failures)

    write_global_peer_fundamentals(args.output, rows)
    report = {
        "schema_version": "global-peer-fundamentals-sync/v1",
        "fiscal_year": args.fiscal_year,
        "output_path": str(args.output),
        "row_count": len(rows),
        "korea_row_count": len(korea_rows),
        "us_row_count": len(us_rows),
        "failure_count": len(failures),
        "failures_sample": failures[:20],
        "credential_policy": "credentials are loaded from gitignored local env only",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


def sync_korea_fundamentals(
    fiscal_year: int,
    krw_usd_rate: float,
    krx_base_date: str,
    stock_limit: int,
    stock_codes: set[str],
) -> tuple[list[GlobalPeerFundamentals], list[dict[str, str]]]:
    api_key = os.environ.get("OPEN_DART_API_KEY", "")
    if not api_key:
        return [], [{"market": "KR", "reason": "OPEN_DART_API_KEY is not configured"}]
    stocks = load_stock_universe(get_settings().stock_universe_path)
    if stock_codes:
        stocks = [stock for stock in stocks if stock.stock_code in stock_codes]
    selected = stocks[:stock_limit] if stock_limit else stocks
    market_caps = _optional_krx_market_caps(krx_base_date, krw_usd_rate)
    rows: list[GlobalPeerFundamentals] = []
    failures: list[dict[str, str]] = []
    for stock in selected:
        if not stock.dart_corp_code:
            continue
        try:
            row = fetch_open_dart_annual_fundamentals(
                api_key=api_key,
                stock_code=stock.stock_code,
                corp_code=stock.dart_corp_code,
                fiscal_year=fiscal_year,
                krw_usd_rate=krw_usd_rate,
            )
            if row:
                row = replace(row, market_cap_usd=market_caps.get(stock.stock_code))
                rows.append(row)
        except Exception as exception:
            failures.append(
                {
                    "market": "KR",
                    "identifier": stock.stock_code,
                    "reason": str(exception),
                }
            )
    return rows, failures


def sync_us_fundamentals(
    fiscal_year: int,
    stock_limit: int,
    tickers: set[str],
) -> tuple[list[GlobalPeerFundamentals], list[dict[str, str]]]:
    ticker_cik = fetch_sec_ticker_cik_map()
    stocks = load_us_stock_universe(get_settings().us_stock_universe_path)
    if tickers:
        stocks = [stock for stock in stocks if stock.ticker.upper() in tickers]
    selected = stocks[:stock_limit] if stock_limit else stocks
    rows: list[GlobalPeerFundamentals] = []
    failures: list[dict[str, str]] = []
    for stock in selected:
        cik = ticker_cik.get(stock.ticker.upper())
        if not cik:
            continue
        try:
            row = fetch_sec_annual_fundamentals(
                ticker=stock.ticker,
                cik=cik,
                fiscal_year=fiscal_year,
            )
            if row:
                rows.append(row)
        except Exception as exception:
            failures.append(
                {
                    "market": "US",
                    "identifier": stock.ticker,
                    "reason": str(exception),
                }
            )
    return rows, failures


def _csv_set(value: str) -> set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def _optional_krx_market_caps(base_date: str, krw_usd_rate: float) -> dict[str, float]:
    auth_key = os.environ.get("KRX_OPEN_API_AUTH_KEY", "")
    if not auth_key:
        return {}
    return fetch_krx_market_caps_usd(
        auth_key=auth_key,
        base_date=base_date,
        krw_usd_rate=krw_usd_rate,
    )


def _previous_weekday() -> date:
    candidate = date.today() - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


if __name__ == "__main__":
    main()
