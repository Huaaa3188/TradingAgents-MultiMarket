import queue
import sys
import threading
import time
from datetime import datetime

import pandas as pd
from stockstats import wrap

from .cache import disk_cache, get_disk_cache, invalidate_disk_cache_where
from .config import get_config
from .contracts import (
    ContractGateResult,
    DataNotice,
    DataResult,
    SourceMeta,
    data_notice,
    render_contract_gate,
    render_notices,
    validate_data_result,
)
from .instruments import (
    InstrumentType,
    MarketType,
    detect_instrument_type,
    detect_market_type,
    normalize_ticker_symbol,
    to_akshare_symbol,
)
from .tiantian_fund import get_fund_nav_history, get_fund_profile_tables

_MACRO_NEWS_TOTAL_BUDGET_SECONDS = 15
_MACRO_NEWS_SOURCE_TIMEOUT_SECONDS = 12
_MACRO_NEWS_SOURCE_COOLDOWN_SECONDS = 60 * 60
_macro_news_source_health: dict[str, float] = {}


def _prewarm_mini_racer() -> None:
    """Pre-warm V8 engine (mini_racer / py_mini_racer) in a single-threaded context.

    Prevents multi-threaded address_pool_manager.cc initialization race condition
    C++ FATAL crashes when multiple agents execute concurrently. Runs once at
    module import (Python's import lock makes this single-threaded by design);
    fails open when neither library is installed.
    """
    try:
        from mini_racer import MiniRacer
        mr = MiniRacer()
        mr.eval("1+1")
    except Exception:
        try:
            from py_mini_racer import MiniRacer
            mr = MiniRacer()
            mr.eval("1+1")
        except Exception:
            pass


_prewarm_mini_racer()
_OHLCV_RENAME_MAP = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
    "成交额": "Amount",
    "振幅": "Amplitude",
    "涨跌幅": "Pct Change",
    "涨跌额": "Change",
    "换手率": "Turnover",
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "amount": "Amount",
}
_OHLCV_REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def _get_cache():
    return get_disk_cache("akshare")


def akshare_disk_cache(expire=14400):  # 默认缓存 4 小时 (14400 秒)
    """AkShare namespace wrapper over the shared dataflow disk cache."""
    return disk_cache("akshare", expire=expire)


class AkShareDataError(RuntimeError):
    """Recoverable AkShare data-source error that allows configured fallback."""


def _ak():
    try:
        import akshare as ak
    except ImportError as exc:
        raise AkShareDataError(
            "AkShare is required for the 'akshare' data vendor. Install the project with the akshare dependency."
        ) from exc
    return ak


@akshare_disk_cache(expire=14400)
def get_stock(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve A-share equity or listed fund OHLCV data using AkShare."""
    try:
        return _render_stock_result(_get_stock_result(symbol, start_date, end_date))
    except AkShareDataError:
        raise
    except Exception as e:
        raise AkShareDataError(f"Error retrieving AkShare data for {symbol}: {str(e)}") from e


def get_stock_result(symbol: str, start_date: str, end_date: str) -> DataResult:
    """Return structured China price/NAV data without changing tool string output."""
    return _get_stock_result(symbol, start_date, end_date)


def _get_stock_result(symbol: str, start_date: str, end_date: str) -> DataResult:
    normalized = normalize_ticker_symbol(symbol)
    semantic = "nav" if detect_market_type(normalized) == MarketType.CN_FUND else "ohlcv"
    result = _load_ohlcv_result(symbol, start_date, end_date)
    data = result.payload if isinstance(result.payload, pd.DataFrame) else pd.DataFrame()
    if data.empty:
        detail = (
            "The requested range may contain only non-trading days, holidays, "
            "a suspension, or unavailable vendor coverage."
        )
        pure_code = to_akshare_symbol(normalized)
        if pure_code.startswith("4"):
            detail += (
                " Codes starting with 4 are NEEQ listings, which the Beijing "
                "exchange data loader may not cover."
            )
        notice = data_notice(
            "no_rows",
            (
                f"No {semantic.upper()} rows found for '{normalized}' between "
                f"{start_date} and {end_date}."
            ),
            source=result.meta.source,
            detail=detail,
        )
        return DataResult(
            meta=result.meta,
            payload=data,
            notices=result.notices + (notice,),
            ok=False,
            missing_reason="no_rows",
            error_type=result.error_type,
            metadata={**result.metadata, "start_date": start_date, "end_date": end_date},
        )

    as_of = _frame_as_of(data)
    meta = SourceMeta(
        vendor=result.meta.vendor,
        source=result.meta.source,
        symbol=normalized,
        semantic=semantic,
        as_of=as_of,
        retrieved_at=_now_timestamp(),
    )
    return DataResult(
        meta=meta,
        payload=data,
        notices=result.notices,
        ok=True,
        error_type=result.error_type,
        metadata={**result.metadata, "start_date": start_date, "end_date": end_date},
    )


def _render_stock_result(result: DataResult) -> str:
    normalized = result.meta.symbol
    start_date = result.metadata.get("start_date", "")
    end_date = result.metadata.get("end_date", "")
    if not result.ok or not isinstance(result.payload, pd.DataFrame) or result.payload.empty:
        notice_text = (
            f"[Data Availability Notice] No {result.meta.source} data found for symbol "
            f"'{normalized}' between {start_date} and {end_date}. "
            "This might be because the requested date range consists entirely of non-trading days "
            "(weekends or public holidays), or the market was closed, or the ticker is temporarily "
            "suspended. If trading data is missing, check alternative vendors or rely on the latest "
            "available history."
        )
        text = notice_text + render_notices(result.notices)
        return _append_contract_gate(
            text,
            result,
            analysis_date=end_date,
            max_staleness_days=_date_window_days(start_date, end_date),
        )

    csv_string = result.payload.to_csv(index=False)
    if result.meta.semantic == "nav":
        header = f"# Tiantian Fund NAV data for {normalized} from {start_date} to {end_date}\n"
    else:
        header = f"# AkShare data for {normalized} from {start_date} to {end_date}\n"
    header += f"# Source: {result.meta.source}\n"
    header += f"# Semantic: {result.meta.semantic}\n"
    if result.meta.as_of:
        header += f"# As of: {result.meta.as_of}\n"
    header += f"# Total records: {len(result.payload)}\n"
    header += f"# Data retrieved on: {result.meta.retrieved_at or _now_timestamp()}\n\n"
    text = header + csv_string + render_notices(result.notices)
    return _append_contract_gate(
        text,
        result,
        analysis_date=end_date,
        max_staleness_days=_date_window_days(start_date, end_date),
    )


def _append_contract_gate(
    text: str,
    result: DataResult,
    *,
    analysis_date: str | None = None,
    max_staleness_days: int | None = None,
) -> str:
    gate = validate_data_result(
        result,
        analysis_date=analysis_date,
        expected_semantic=result.meta.semantic,
        max_staleness_days=max_staleness_days,
    )
    # Surface payload-level diagnostics (e.g. ``window_fallback``, fallback
    # ``source_error``) on the contract gate as well, not just in the text-only
    # Data Notices block, so they reach the parsed ``data_contract_status`` and
    # the Data Reliability render (``render_data_contract_status``).
    gate = _merge_result_notices_into_gate(gate)
    return text + "\n\n" + render_contract_gate(gate, "AkShare Data Contract Gate")


def _merge_result_notices_into_gate(gate: ContractGateResult) -> ContractGateResult:
    """Fold warning/error notices carried on the data result into the gate.

    ``validate_data_result`` only emits diagnostics it infers itself (staleness,
    schema drift, semantic mismatch). Payload-level notices a vendor attached to
    a result (e.g. ``window_fallback``) are otherwise invisible to the parsed
    ``data_contract_status``. Merge them in, de-duplicating by code.
    """
    warnings = list(gate.warnings)
    failures = list(gate.failures)
    known_codes = {notice.code for notice in (*warnings, *failures)}
    for notice in gate.result.notices:
        if notice.code in known_codes or notice.severity not in ("warning", "error"):
            continue
        (failures if notice.severity == "error" else warnings).append(notice)
        known_codes.add(notice.code)
    return ContractGateResult(
        ok=not failures,
        result=gate.result,
        failures=tuple(failures),
        warnings=tuple(warnings),
        expected_semantic=gate.expected_semantic,
    )


def _date_window_days(start_date: str | None, end_date: str | None) -> int | None:
    if not start_date or not end_date:
        return None
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return max(0, int((end - start).days))


@akshare_disk_cache(expire=14400)
def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    supported = _indicator_descriptions()
    indicator = indicator.lower()
    if indicator not in supported:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(supported.keys())}"
        )

    try:
        end_dt = pd.to_datetime(curr_date)
        start_dt = end_dt - pd.DateOffset(days=max(look_back_days, 250))
        ohlcv_result = _load_ohlcv_result(
            symbol,
            start_dt.strftime("%Y-%m-%d"),
            curr_date,
        )
        data = ohlcv_result.payload if isinstance(ohlcv_result.payload, pd.DataFrame) else pd.DataFrame()
        if data.empty:
            text = f"No AkShare OHLCV data found for {symbol} up to {curr_date}"
            return _append_contract_gate(
                text,
                ohlcv_result,
                analysis_date=curr_date,
                max_staleness_days=look_back_days,
            )

        df = data.rename(columns=str.lower)
        df["date"] = pd.to_datetime(df["date"])
        df = wrap(df)
        df[indicator]

        window_start = (end_dt - pd.DateOffset(days=look_back_days)).strftime("%Y-%m-%d")
        rows = pd.DataFrame(df).reset_index()
        rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
        rows = rows[rows["date"] >= window_start][["date", indicator]]
        lines = []
        for _, row in rows.iterrows():
            value = row[indicator]
            rendered = "N/A" if pd.isna(value) else str(value)
            lines.append(f"{row['date']}: {rendered}")

        result = (
            f"## {indicator} values from {window_start} to {curr_date}:\n\n"
            + "\n".join(lines)
            + "\n\n"
            + supported[indicator]
        )
        if detect_market_type(symbol) == MarketType.CN_FUND:
            result += (
                "\n\nFund NAV note: this indicator is computed from daily fund net asset value, "
                "not exchange-traded OHLCV. Volume-derived indicators are not meaningful for OTC mutual funds."
            )
        return _append_contract_gate(
            result,
            ohlcv_result,
            analysis_date=curr_date,
            max_staleness_days=look_back_days,
        )
    except AkShareDataError:
        raise
    except Exception as e:
        raise AkShareDataError(f"Error getting AkShare indicator data for {symbol}: {str(e)}") from e


@akshare_disk_cache(expire=14400)
def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    result = _get_fundamentals_result(ticker, curr_date)
    return _append_contract_gate(result.text or "", result, analysis_date=curr_date)


def get_fundamentals_result(ticker: str, curr_date: str | None = None) -> DataResult:
    """Return structured China fundamentals/profile data."""
    return _get_fundamentals_result(ticker, curr_date)


def _get_fundamentals_result(ticker: str, curr_date: str | None = None) -> DataResult:
    normalized = normalize_ticker_symbol(ticker)
    if detect_instrument_type(normalized) == InstrumentType.FUND:
        return _get_fund_profile_result(normalized, curr_date)
    return _get_equity_profile_result(normalized, curr_date)


@akshare_disk_cache(expire=14400)
def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    if detect_instrument_type(ticker) == InstrumentType.FUND:
        return _fund_statement_not_applicable(ticker, "balance sheet")
    return _financial_statement(ticker, "balance_sheet", freq, curr_date)


@akshare_disk_cache(expire=14400)
def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    if detect_instrument_type(ticker) == InstrumentType.FUND:
        return _fund_statement_not_applicable(ticker, "cash flow statement")
    return _financial_statement(ticker, "cashflow", freq, curr_date)


@akshare_disk_cache(expire=14400)
def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    if detect_instrument_type(ticker) == InstrumentType.FUND:
        return _fund_statement_not_applicable(ticker, "income statement")
    return _financial_statement(ticker, "income_statement", freq, curr_date)


@akshare_disk_cache(expire=14400)
def get_news(ticker: str, start_date: str, end_date: str) -> str:
    result = _get_news_result(ticker, start_date, end_date)
    has_window_fallback = any(n.code == "window_fallback" for n in result.notices)
    return _append_contract_gate(
        result.text or "",
        result,
        analysis_date=end_date,
        # Window-fallback results are, by definition, older than the requested
        # window: staleness would be a false alarm layered on top of the
        # explicit window_fallback warning, so skip the staleness threshold
        # and let the warning speak for itself.
        max_staleness_days=None if has_window_fallback else _date_window_days(start_date, end_date),
    )


def get_news_result(ticker: str, start_date: str, end_date: str) -> DataResult:
    """Return structured China news/announcement data."""
    return _get_news_result(ticker, start_date, end_date)


def _get_news_result(ticker: str, start_date: str, end_date: str) -> DataResult:
    ak = _ak()
    normalized = normalize_ticker_symbol(ticker)
    symbol = to_akshare_symbol(ticker)
    retrieved_at = _now_timestamp()
    notices: list[DataNotice] = []
    try:
        if detect_instrument_type(normalized) == InstrumentType.FUND:
            fund_label = _fund_label(normalized)
            report_announcements = _safe_call(lambda: ak.fund_announcement_report_em(symbol=symbol))
            dividend_announcements = _safe_call(lambda: ak.fund_announcement_dividend_em(symbol=symbol))
            personnel_announcements = _safe_call(lambda: ak.fund_announcement_personnel_em(symbol=symbol))
            data = pd.concat(
                [
                    report_announcements.assign(_source="fund report announcement"),
                    dividend_announcements.assign(_source="fund dividend announcement"),
                    personnel_announcements.assign(_source="fund personnel announcement"),
                ],
                ignore_index=True,
            )
            if data.empty:
                text = (
                    f"No AkShare {fund_label} announcements found for {normalized} between "
                    f"{start_date} and {end_date}. Use {_fund_context_hint(normalized)} instead."
                )
                notices.append(
                    data_notice(
                        "no_news",
                        f"No AkShare {fund_label} announcements were available in the requested window.",
                        source="akshare_fund_announcements",
                    )
                )
                return DataResult(
                    meta=SourceMeta(
                        vendor="akshare",
                        source="akshare_fund_announcements",
                        symbol=normalized,
                        semantic="news",
                        retrieved_at=retrieved_at,
                    ),
                    payload=data,
                    notices=tuple(notices),
                    ok=False,
                    missing_reason="no_news",
                    text=text + render_notices(tuple(notices)),
                    metadata={"start_date": start_date, "end_date": end_date},
                )
            data, used_fallback = _filter_fund_announcements(data, start_date, end_date)
            if data.empty:
                text = f"No AkShare {fund_label} announcements found for {normalized} on or before {end_date}"
                notices.append(
                    data_notice(
                        "no_news",
                        f"No AkShare {fund_label} announcements were available on or before {end_date}.",
                        source="akshare_fund_announcements",
                    )
                )
                return DataResult(
                    meta=SourceMeta(
                        vendor="akshare",
                        source="akshare_fund_announcements",
                        symbol=normalized,
                        semantic="news",
                        retrieved_at=retrieved_at,
                    ),
                    payload=data,
                    notices=tuple(notices),
                    ok=False,
                    missing_reason="no_news",
                    text=text + render_notices(tuple(notices)),
                    metadata={"start_date": start_date, "end_date": end_date},
                )
            rows = _render_news_rows(data)
            if used_fallback:
                heading = (
                    f"## {normalized} Recent {fund_label.title()} Announcements before {end_date} "
                    f"(none found from {start_date} to {end_date}):\n\n"
                )
                notices.append(
                    data_notice(
                        "window_fallback",
                        (
                            "No fund announcements were found inside the requested window; "
                            "using recent announcements before the end date."
                        ),
                        source="akshare_fund_announcements",
                        severity="warning",
                    )
                )
            else:
                heading = f"## {normalized} {fund_label.title()} Announcements, from {start_date} to {end_date}:\n\n"
            text = heading + "\n".join(rows) + render_notices(tuple(notices))
            return DataResult(
                meta=SourceMeta(
                    vendor="akshare",
                    source="akshare_fund_announcements",
                    symbol=normalized,
                    semantic="news",
                    as_of=_news_as_of(data),
                    retrieved_at=retrieved_at,
                ),
                payload=data,
                notices=tuple(notices),
                ok=True,
                text=text,
                metadata={
                    "start_date": start_date,
                    "end_date": end_date,
                    "used_fallback": used_fallback,
                },
            )

        data = ak.stock_news_em(symbol=symbol)
        if data is None or data.empty:
            text = f"No AkShare news found for {ticker}"
            notices.append(
                data_notice(
                    "no_news",
                    f"No AkShare stock news rows were available for {normalized}.",
                    source="akshare_stock_news",
                )
            )
            return DataResult(
                meta=SourceMeta(
                    vendor="akshare",
                    source="akshare_stock_news",
                    symbol=normalized,
                    semantic="news",
                    retrieved_at=retrieved_at,
                ),
                payload=pd.DataFrame() if data is None else data,
                notices=tuple(notices),
                ok=False,
                missing_reason="no_news",
                text=text + render_notices(tuple(notices)),
                metadata={"start_date": start_date, "end_date": end_date},
            )
        data = _filter_date_rows(data, start_date, end_date)
        if data.empty:
            text = f"No AkShare news found for {ticker} between {start_date} and {end_date}"
            notices.append(
                data_notice(
                    "no_news",
                    f"No AkShare stock news rows survived date filtering for {normalized}.",
                    source="akshare_stock_news",
                )
            )
            return DataResult(
                meta=SourceMeta(
                    vendor="akshare",
                    source="akshare_stock_news",
                    symbol=normalized,
                    semantic="news",
                    retrieved_at=retrieved_at,
                ),
                payload=data,
                notices=tuple(notices),
                ok=False,
                missing_reason="no_news",
                text=text + render_notices(tuple(notices)),
                metadata={"start_date": start_date, "end_date": end_date},
            )
        rows = _render_news_rows(data)
        text = (
            f"## {normalize_ticker_symbol(ticker)} News, from {start_date} to {end_date}:\n\n"
            + "\n".join(rows)
        )
        return DataResult(
            meta=SourceMeta(
                vendor="akshare",
                source="akshare_stock_news",
                symbol=normalized,
                semantic="news",
                as_of=_news_as_of(data),
                retrieved_at=retrieved_at,
            ),
            payload=data,
            notices=tuple(notices),
            ok=True,
            text=text,
            metadata={"start_date": start_date, "end_date": end_date},
        )
    except AkShareDataError:
        raise
    except Exception as e:
        raise AkShareDataError(f"Error fetching AkShare news for {ticker}: {str(e)}") from e


def get_global_news(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    start_date = (pd.to_datetime(curr_date) - pd.DateOffset(days=look_back_days)).strftime("%Y-%m-%d")

    try:
        ak = _ak()
        data = _collect_china_macro_news(ak, start_date, curr_date, limit)
    except Exception as exc:
        return (
            "Error fetching AkShare China macro/policy news: "
            f"{exc}. Report that local macro news is unavailable instead of inventing policy context."
        )

    if data.empty:
        return (
            f"No AkShare China macro/policy news found between {start_date} and {curr_date}. "
            "Do not infer policy catalysts without tool evidence."
        )

    rows = _render_news_rows(data.head(limit))
    return f"## China Macro and Policy News, from {start_date} to {curr_date}:\n\n" + "\n".join(rows)


def get_insider_transactions(ticker: str) -> str:
    """Return China insider/ownership transaction rows for A-share tickers.

    Uses AkShare's management-holdings-detail endpoint. The shared tool
    signature carries no date, so the latest available rows are returned
    without window filtering. Fails open with an explicit no-data result —
    never a hardcoded stub — so the news analyst treats absent insider
    activity as a lookup outcome rather than "not implemented".
    """
    normalized = normalize_ticker_symbol(ticker)
    if detect_instrument_type(normalized) == InstrumentType.FUND:
        return f"Insider transaction data is not applicable to {_fund_label(normalized)} {normalized}."
    if detect_market_type(normalized) != MarketType.CN_A:
        return (
            f"AkShare insider transaction data is not available for {normalized}; "
            "only mainland China A-share tickers are covered by this vendor."
        )

    ak = _ak()
    symbol = to_akshare_symbol(normalized)
    data = _safe_call(lambda: ak.stock_hold_management_detail_em(symbol=symbol))
    if data is None or data.empty:
        return (
            f"No AkShare insider transaction rows were available for {normalized}. "
            "Do not fabricate insider-activity figures."
        )

    header = (
        f"# AkShare Insider Transactions (management holdings detail) for {normalized}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + data.to_csv(index=False)


def _load_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    result = _load_ohlcv_result(symbol, start_date, end_date)
    return result.payload if isinstance(result.payload, pd.DataFrame) else pd.DataFrame()


def _load_ohlcv_result(symbol: str, start_date: str, end_date: str) -> DataResult:
    normalized = normalize_ticker_symbol(symbol)
    if detect_market_type(normalized) == MarketType.CN_FUND:
        data = _normalize_ohlcv(
            get_fund_nav_history(normalized, start_date, end_date),
            start_date,
            end_date,
        )
        return DataResult(
            meta=SourceMeta(
                vendor="akshare",
                source="tiantian_fund_nav",
                symbol=normalized,
                semantic="nav",
                as_of=_frame_as_of(data),
                retrieved_at=_now_timestamp(),
            ),
            payload=data,
            ok=not data.empty,
            missing_reason=None if not data.empty else "no_rows",
            metadata={"start_date": start_date, "end_date": end_date},
        )

    ak = _ak()
    pure_symbol = to_akshare_symbol(normalized)
    start = _compact_date(start_date)
    end = _compact_date(end_date)

    errors = []
    empty_sources = []
    notices: list[DataNotice] = []
    for source_name, loader in _ohlcv_source_loaders(ak, normalized, pure_symbol, start, end):
        try:
            raw = loader()
            data = _normalize_ohlcv(raw, start_date, end_date)
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            notices.append(
                data_notice(
                    "source_error",
                    f"{source_name} failed while loading OHLCV data.",
                    source=source_name,
                    detail=str(exc),
                    severity="warning",
                )
            )
            continue
        if data.empty:
            empty_sources.append(source_name)
            drift_notice = _ohlcv_empty_notice(source_name, raw, start_date, end_date)
            if drift_notice is not None:
                notices.append(drift_notice)
            continue
        return DataResult(
            meta=SourceMeta(
                vendor="akshare",
                source=source_name,
                symbol=normalized,
                semantic="ohlcv",
                as_of=_frame_as_of(data),
                retrieved_at=_now_timestamp(),
            ),
            payload=data,
            notices=tuple(notices),
            ok=True,
            error_type="schema_drift" if any(n.code == "schema_drift" for n in notices) else None,
            metadata={"start_date": start_date, "end_date": end_date},
        )

    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if errors:
        details = " | ".join(errors)
        if empty_sources:
            details += " | empty sources: " + ", ".join(empty_sources)
        raise AkShareDataError(f"AkShare OHLCV sources failed for {normalized}: {details}")
    return DataResult(
        meta=SourceMeta(
            vendor="akshare",
            source=",".join(empty_sources) if empty_sources else "akshare_ohlcv",
            symbol=normalized,
            semantic="ohlcv",
            retrieved_at=_now_timestamp(),
        ),
        payload=pd.DataFrame(columns=columns),
        notices=tuple(notices),
        ok=False,
        missing_reason="no_rows",
        error_type="schema_drift" if any(n.code == "schema_drift" for n in notices) else None,
        metadata={"start_date": start_date, "end_date": end_date},
    )


@akshare_disk_cache(expire=14400)
def load_index_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Retrieve mainland China index OHLCV rows for local alpha baselines."""
    return _load_index_ohlcv(symbol, start_date, end_date)


def _load_index_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _ak()
    ak_symbol = _to_akshare_index_symbol(symbol)
    start = _compact_date(start_date)
    end = _compact_date(end_date)

    errors = []
    empty_sources = []
    for source_name, loader in _index_source_loaders(ak, ak_symbol, start, end):
        try:
            data = _normalize_ohlcv(loader(), start_date, end_date)
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            continue
        if data.empty:
            empty_sources.append(source_name)
            continue
        return data

    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if errors:
        details = " | ".join(errors)
        if empty_sources:
            details += " | empty sources: " + ", ".join(empty_sources)
        raise AkShareDataError(f"AkShare index OHLCV sources failed for {symbol}: {details}")
    return pd.DataFrame(columns=columns)


def _index_source_loaders(ak, ak_symbol: str, start: str, end: str):
    yield (
        "tencent_stock_zh_index_daily_tx",
        lambda: ak.stock_zh_index_daily_tx(symbol=ak_symbol, start_date=start, end_date=end),
    )
    yield (
        "sina_stock_zh_index_daily",
        lambda: ak.stock_zh_index_daily(symbol=ak_symbol),
    )
    yield (
        "eastmoney_index_zh_a_hist",
        lambda: ak.index_zh_a_hist(
            symbol=ak_symbol[-6:],
            period="daily",
            start_date=start,
            end_date=end,
        ),
    )


def _ohlcv_source_loaders(ak, normalized: str, pure_symbol: str, start: str, end: str):
    if detect_instrument_type(normalized) == InstrumentType.FUND:
        if pure_symbol.startswith(("159", "5")):
            yield (
                "eastmoney_fund_etf_hist_em",
                lambda: ak.fund_etf_hist_em(
                    symbol=pure_symbol,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust="",
                ),
            )
        else:
            yield (
                "eastmoney_fund_lof_hist_em",
                lambda: ak.fund_lof_hist_em(
                    symbol=pure_symbol,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust="",
                ),
            )
        yield (
            "sina_fund_etf_hist_sina",
            lambda: ak.fund_etf_hist_sina(symbol=_prefixed_cn_symbol(normalized)),
        )
        return

    yield (
        "eastmoney_stock_zh_a_hist",
        lambda: ak.stock_zh_a_hist(
            symbol=pure_symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="",
        ),
    )
    yield (
        "sina_stock_zh_a_daily",
        lambda: ak.stock_zh_a_daily(
            symbol=_prefixed_cn_symbol(normalized),
            start_date=start,
            end_date=end,
            adjust="",
        ),
    )
    yield (
        "tencent_stock_zh_a_hist_tx",
        lambda: ak.stock_zh_a_hist_tx(
            symbol=_prefixed_cn_symbol(normalized),
            start_date=start,
            end_date=end,
            adjust="",
        ),
    )


def _normalize_ohlcv(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    normalized = data.rename(columns={k: v for k, v in _OHLCV_RENAME_MAP.items() if k in data.columns}).copy()
    for column in _OHLCV_REQUIRED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized = normalized.dropna(subset=["Date"])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    normalized = normalized[(normalized["Date"] >= start_dt) & (normalized["Date"] <= end_dt)]
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["Close"])
    normalized["Date"] = normalized["Date"].dt.strftime("%Y-%m-%d")
    columns = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "Amount", "Pct Change", "Turnover"] if c in normalized.columns]
    return normalized[columns]


def _ohlcv_empty_notice(
    source_name: str,
    raw: pd.DataFrame | None,
    start_date: str,
    end_date: str,
) -> DataNotice | None:
    if raw is None or raw.empty:
        return data_notice(
            "no_rows",
            f"{source_name} returned no raw rows.",
            source=source_name,
            detail=f"requested={start_date}..{end_date}",
            severity="info",
        )

    renamed = raw.rename(columns={k: v for k, v in _OHLCV_RENAME_MAP.items() if k in raw.columns})
    missing_required = [column for column in ("Date", "Close") if column not in renamed.columns]
    if missing_required:
        return data_notice(
            "schema_drift",
            f"{source_name} returned rows but missed required column(s): {', '.join(missing_required)}.",
            source=source_name,
            detail=f"columns={list(raw.columns)}",
            severity="warning",
        )

    return data_notice(
        "filtered_empty",
        f"{source_name} returned rows but none survived date/price validation.",
        source=source_name,
        detail=f"requested={start_date}..{end_date}",
        severity="info",
    )


def _frame_as_of(data: pd.DataFrame, date_column: str = "Date") -> str | None:
    if data is None or data.empty or date_column not in data.columns:
        return None
    parsed = pd.to_datetime(data[date_column], errors="coerce").dropna()
    if parsed.empty:
        return None
    return parsed.max().strftime("%Y-%m-%d")


def _tables_as_of(tables) -> str | None:
    dates = []
    for table in tables or []:
        data = getattr(table, "data", None)
        if data is None or data.empty:
            continue
        for column in data.columns:
            if "日期" not in str(column) and "date" not in str(column).lower():
                continue
            parsed = pd.to_datetime(data[column], errors="coerce").dropna()
            if not parsed.empty:
                dates.append(parsed.max())
    if not dates:
        return None
    return max(dates).strftime("%Y-%m-%d")


def _news_as_of(data: pd.DataFrame) -> str | None:
    if data is None or data.empty:
        return None
    for column in data.columns:
        if "时间" in str(column) or "日期" in str(column) or "date" in str(column).lower():
            parsed = _parse_datetime_series(data[column]).dropna()
            if not parsed.empty:
                return parsed.max().strftime("%Y-%m-%d")
    return None


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_fund_profile(ticker: str, curr_date: str | None) -> str:
    return _get_fund_profile_result(ticker, curr_date).text or ""


def _get_fund_profile_result(ticker: str, curr_date: str | None) -> DataResult:
    pure_symbol = to_akshare_symbol(ticker)
    is_otc_fund = detect_market_type(ticker) == MarketType.CN_FUND
    title = "China OTC Fund Profile" if is_otc_fund else "Listed Fund Profile"
    retrieved_at = _now_timestamp()
    lines = [f"# {title} for {ticker}", f"# Data retrieved on: {retrieved_at}", ""]
    notices: list[DataNotice] = []

    tiantian_tables = _safe_tiantian_fund_tables(pure_symbol, curr_date)
    for table in tiantian_tables:
        lines.extend(_render_table_like(table.title, table.data, max_rows=table.max_rows))

    if is_otc_fund:
        if not tiantian_tables:
            lines.append(f"No Tiantian Fund profile data found for {pure_symbol}.")
            notices.append(
                data_notice(
                    "no_profile_data",
                    f"No Tiantian Fund profile tables were available for {pure_symbol}.",
                    source="tiantian_fund_profile",
                )
            )
        lines.append(
            "Fund analysis focus: NAV trend, fund category/share class, fund manager, assets under management, fees, asset allocation, holdings concentration, QDII/FX risk where applicable, and redemption/subscription constraints. Do not use listed-company financial statement semantics."
        )
        text = "\n".join(lines) + render_notices(tuple(notices))
        return DataResult(
            meta=SourceMeta(
                vendor="akshare",
                source="tiantian_fund_profile",
                symbol=ticker,
                semantic="fund_profile",
                as_of=_tables_as_of(tiantian_tables),
                retrieved_at=retrieved_at,
            ),
            payload=tiantian_tables,
            notices=tuple(notices),
            ok=bool(tiantian_tables),
            missing_reason=None if tiantian_tables else "no_profile_data",
            text=text,
        )

    ak = _ak()
    overview = _safe_call(lambda: ak.fund_overview_em(symbol=pure_symbol))
    fee = _safe_call(lambda: ak.fund_fee_em(symbol=pure_symbol, indicator="运作费用"))
    holdings = _safe_call(
        lambda: ak.fund_portfolio_hold_em(symbol=pure_symbol, date=_fund_holdings_year(curr_date))
    )

    lines.extend(_render_table_like("AkShare Fund Overview", overview))
    lines.extend(_render_table_like("AkShare Fund Fees", fee))
    lines.extend(_render_table_like("AkShare Fund Top Holdings", holdings, max_rows=10))
    lines.append(
        "Fund analysis focus: benchmark/theme exposure, premium or discount, liquidity, fund size, fees, holdings concentration, and market risk."
    )
    payload_tables = [
        ("Tiantian Fund", table.data) for table in tiantian_tables
    ] + [
        ("AkShare Fund Overview", overview),
        ("AkShare Fund Fees", fee),
        ("AkShare Fund Top Holdings", holdings),
    ]
    has_profile_rows = any(isinstance(data, pd.DataFrame) and not data.empty for _, data in payload_tables)
    if not has_profile_rows:
        notices.append(
            data_notice(
                "no_profile_data",
                f"No listed fund profile tables were available for {pure_symbol}.",
                source="akshare_fund_profile",
            )
        )
    text = "\n".join(lines) + render_notices(tuple(notices))
    return DataResult(
        meta=SourceMeta(
            vendor="akshare",
            source="akshare_fund_profile",
            symbol=ticker,
            semantic="fund_profile",
            as_of=_tables_as_of(tiantian_tables),
            retrieved_at=retrieved_at,
        ),
        payload=payload_tables,
        notices=tuple(notices),
        ok=has_profile_rows,
        missing_reason=None if has_profile_rows else "no_profile_data",
        text=text,
    )


def _get_equity_profile(ticker: str, curr_date: str | None) -> str:
    return _get_equity_profile_result(ticker, curr_date).text or ""


def _get_equity_profile_result(ticker: str, curr_date: str | None) -> DataResult:
    ak = _ak()
    symbol = to_akshare_symbol(ticker)
    retrieved_at = _now_timestamp()
    lines = [f"# A-share Company Fundamentals for {ticker}", f"# Data retrieved on: {retrieved_at}", ""]
    spot = _safe_call(lambda: ak.stock_individual_info_em(symbol=symbol))
    financial = _safe_call(lambda: ak.stock_financial_abstract(symbol=symbol))
    lines.extend(_render_table_like("Company Profile", spot))
    lines.extend(_render_table_like("Financial Abstract", _filter_statement_by_date(financial, curr_date), max_rows=20))
    has_profile_rows = (spot is not None and not spot.empty) or (financial is not None and not financial.empty)
    notices: tuple[DataNotice, ...] = ()
    if not has_profile_rows:
        notices = (
            data_notice(
                "no_profile_data",
                f"No AkShare company profile or financial abstract rows were available for {ticker}.",
                source="akshare_equity_profile",
            ),
        )
    text = "\n".join(lines) + render_notices(notices)
    return DataResult(
        meta=SourceMeta(
            vendor="akshare",
            source="akshare_equity_profile",
            symbol=ticker,
            semantic="company_profile",
            retrieved_at=retrieved_at,
        ),
        payload=[("Company Profile", spot), ("Financial Abstract", financial)],
        notices=notices,
        ok=has_profile_rows,
        missing_reason=None if has_profile_rows else "no_profile_data",
        text=text,
    )


def _financial_statement(ticker: str, statement: str, freq: str, curr_date: str | None) -> str:
    ak = _ak()
    symbol = _to_akshare_statement_symbol(ticker)
    fn_name = {
        "balance_sheet": "stock_balance_sheet_by_report_em",
        "cashflow": "stock_cash_flow_sheet_by_report_em",
        "income_statement": "stock_profit_sheet_by_report_em",
    }[statement]
    try:
        data = getattr(ak, fn_name)(symbol=symbol)
        data = _filter_statement_by_date(data, curr_date)
        if data is None or data.empty:
            return f"No AkShare {statement.replace('_', ' ')} data found for {ticker}"
        header = f"# AkShare {statement.replace('_', ' ').title()} for {normalize_ticker_symbol(ticker)} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv(index=False)
    except AkShareDataError:
        raise
    except Exception as e:
        raise AkShareDataError(
            f"Error retrieving AkShare {statement.replace('_', ' ')} for {ticker}: {str(e)}"
        ) from e


def _fund_statement_not_applicable(ticker: str, statement_name: str) -> str:
    normalized = normalize_ticker_symbol(ticker)
    return (
        f"{statement_name.title()} is not applicable to {_fund_label(normalized)} {normalized}. "
        "Use fund overview, fees, holdings, NAV trend, asset allocation, liquidity or redemption terms, and benchmark/theme exposure instead."
    )


def _fund_label(ticker: str) -> str:
    return "OTC fund" if detect_market_type(ticker) == MarketType.CN_FUND else "listed fund"


def _fund_context_hint(ticker: str) -> str:
    if detect_market_type(ticker) == MarketType.CN_FUND:
        return "NAV trend, fund category, fees, asset allocation, holdings, fund manager, QDII/FX exposure, and subscription/redemption context"
    return "China market, benchmark/theme, liquidity, premium/discount, fees, and holdings context"


def _filter_statement_by_date(data: pd.DataFrame | None, curr_date: str | None) -> pd.DataFrame:
    if data is None or data.empty or not curr_date:
        return pd.DataFrame() if data is None else data
    result = data.copy()
    date_columns = [c for c in result.columns if "日期" in str(c) or "date" in str(c).lower() or "报告期" in str(c)]
    if not date_columns:
        return result
    parsed = _parse_datetime_series(result[date_columns[0]])
    return result[parsed <= pd.to_datetime(curr_date)]


def _filter_date_rows(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    result = data.copy()
    date_columns = [c for c in result.columns if "时间" in str(c) or "日期" in str(c) or "date" in str(c).lower()]
    if not date_columns:
        return result
    parsed = _parse_datetime_series(result[date_columns[0]])
    end_exclusive = pd.to_datetime(end_date) + pd.DateOffset(days=1)
    return result[(parsed >= pd.to_datetime(start_date)) & (parsed < end_exclusive)]


def _filter_fund_announcements(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    fallback_limit: int = 5,
) -> tuple[pd.DataFrame, bool]:
    result = data.copy()
    date_columns = [c for c in result.columns if "时间" in str(c) or "日期" in str(c) or "date" in str(c).lower()]
    if not date_columns:
        return result.head(fallback_limit), False

    parsed = _parse_datetime_series(result[date_columns[0]])
    result = result.assign(_parsed_date=parsed).dropna(subset=["_parsed_date"])
    if result.empty:
        return result, False

    start_dt = pd.to_datetime(start_date)
    end_exclusive = pd.to_datetime(end_date) + pd.DateOffset(days=1)
    in_window = result[(result["_parsed_date"] >= start_dt) & (result["_parsed_date"] < end_exclusive)]
    if not in_window.empty:
        return in_window.drop(columns=["_parsed_date"]), False

    fallback = result[result["_parsed_date"] < end_exclusive]
    fallback = fallback.sort_values("_parsed_date", ascending=False).head(fallback_limit)
    return fallback.drop(columns=["_parsed_date"]), True


def _collect_china_macro_news(ak, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
    frames = []
    errors = []
    deadline = time.monotonic() + _MACRO_NEWS_TOTAL_BUDGET_SECONDS

    for source_name, loader in _china_macro_news_loaders(ak, start_date, end_date):
        remaining = _macro_news_budget_remaining(deadline)
        if remaining <= 0:
            errors.append(
                f"macro news budget exhausted after {_MACRO_NEWS_TOTAL_BUDGET_SECONDS} seconds"
            )
            break

        if _is_macro_news_source_in_cooldown(source_name):
            errors.append(f"{source_name}: skipped due to recent timeout")
            continue

        try:
            data = _call_macro_news_loader(
                source_name,
                loader,
                timeout_seconds=min(_MACRO_NEWS_SOURCE_TIMEOUT_SECONDS, remaining),
            )
        except TimeoutError as exc:
            _mark_macro_news_source_timeout(source_name)
            errors.append(f"{source_name}: {exc}")
            continue
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            continue
        _mark_macro_news_source_success(source_name)
        normalized = _normalize_news_frame(data, source_name)
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        if errors:
            raise AkShareDataError(" | ".join(errors))
        return pd.DataFrame(columns=["标题", "来源", "发布时间", "摘要", "链接"])

    combined = pd.concat(frames, ignore_index=True)
    combined = _filter_date_rows(combined, start_date, end_date)
    if combined.empty:
        return combined
    combined["_dedupe_title"] = combined["标题"].astype(str).str.strip()
    combined = combined.drop_duplicates(subset=["_dedupe_title"])
    combined["_parsed_date"] = _parse_datetime_series(combined["发布时间"])
    combined = combined.sort_values("_parsed_date", ascending=False, na_position="last")
    return combined.drop(columns=["_dedupe_title", "_parsed_date"]).head(limit)


def _china_macro_news_loaders(ak, start_date: str, end_date: str):
    if not _is_historical_macro_news_date(end_date):
        yield ("Eastmoney 7x24", lambda: ak.stock_info_global_em())
        yield ("Sina finance 7x24", lambda: ak.stock_info_global_sina())
        yield ("THS finance live", lambda: ak.stock_info_global_ths())
        yield ("CLS telegraph", lambda: ak.stock_info_global_cls(symbol="全部"))

    current = pd.to_datetime(end_date)
    start = pd.to_datetime(start_date)
    cctv_dates = []
    while current >= start and len(cctv_dates) < 3:
        cctv_dates.append(current.strftime("%Y%m%d"))
        current -= pd.DateOffset(days=1)
    for date_value in cctv_dates:
        yield (f"CCTV News {date_value}", lambda date_value=date_value: ak.news_cctv(date=date_value))


def _call_macro_news_loader(source_name: str, loader, timeout_seconds: float):
    result_queue = queue.Queue(maxsize=1)

    def run_loader():
        try:
            result_queue.put(("ok", loader()))
        except Exception as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(
        target=run_loader,
        name=f"akshare-macro-news-{source_name}",
        daemon=True,
    )
    thread.start()
    thread.join(max(timeout_seconds, 0))
    if thread.is_alive():
        raise TimeoutError(
            f"{source_name} did not return within {timeout_seconds:.2f} seconds"
        )

    status, value = result_queue.get_nowait()
    if status == "error":
        raise value
    return value


def _macro_news_budget_remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _is_macro_news_source_in_cooldown(source_name: str) -> bool:
    now = time.time()
    disabled_until = _get_macro_news_disabled_until(source_name)
    if disabled_until is None:
        return False
    if disabled_until <= now:
        _clear_macro_news_source_health(source_name)
        return False
    return True


def _get_macro_news_disabled_until(source_name: str) -> float | None:
    active_cache = _get_cache()
    key = _macro_news_source_health_key(source_name)
    if active_cache is not None:
        try:
            value = active_cache.get(key)
            if value is not None:
                return float(value)
        except Exception as exc:
            print(f"[Warning] DiskCache read failure for {key}: {exc}", file=sys.stderr)
    return _macro_news_source_health.get(source_name)


def _clear_macro_news_source_health(source_name: str) -> None:
    active_cache = _get_cache()
    key = _macro_news_source_health_key(source_name)
    if active_cache is not None:
        try:
            active_cache.delete(key)
        except Exception as exc:
            print(f"[Warning] DiskCache delete failure for {key}: {exc}", file=sys.stderr)
    _macro_news_source_health.pop(source_name, None)


def _mark_macro_news_source_timeout(source_name: str) -> None:
    disabled_until = time.time() + _MACRO_NEWS_SOURCE_COOLDOWN_SECONDS
    _macro_news_source_health[source_name] = disabled_until
    active_cache = _get_cache()
    if active_cache is None:
        return
    key = _macro_news_source_health_key(source_name)
    try:
        active_cache.set(
            key,
            disabled_until,
            expire=_MACRO_NEWS_SOURCE_COOLDOWN_SECONDS,
        )
    except Exception as exc:
        print(f"[Warning] DiskCache write failure for {key}: {exc}", file=sys.stderr)


def _mark_macro_news_source_success(source_name: str) -> None:
    _clear_macro_news_source_health(source_name)


def _macro_news_source_health_key(source_name: str) -> str:
    return f"macro_news_source_disabled:{source_name}"


def _is_historical_macro_news_date(end_date: str) -> bool:
    try:
        requested = pd.to_datetime(end_date).normalize()
    except Exception:
        return False
    return requested < pd.Timestamp.now().normalize()


def _normalize_news_frame(data: pd.DataFrame | None, source_name: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["标题", "来源", "发布时间", "摘要", "链接"])

    rows = []
    for _, row in data.iterrows():
        title = _first_present(
            row,
            ["新闻标题", "标题", "公告标题", "title", "Title", "内容", "摘要", "资讯标题"],
            default="",
        )
        summary = _first_present(
            row,
            ["摘要", "简介", "summary", "Summary", "新闻内容", "内容"],
            default="",
        )
        if not title and summary:
            title, summary = str(summary)[:80], str(summary)
        if not title:
            continue

        rows.append(
            {
                "标题": str(title).strip(),
                "来源": _first_present(
                    row,
                    ["文章来源", "来源", "source", "Source", "媒体", "信息来源"],
                    default=source_name,
                ),
                "发布时间": _first_present(
                    row,
                    ["发布时间", "日期", "公告日期", "时间", "publish_time", "datetime", "update_time"],
                    default="",
                ),
                "摘要": str(summary).strip() if summary else "",
                "链接": _first_present(
                    row,
                    ["新闻链接", "链接", "公告链接", "url", "URL", "网址"],
                    default="",
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_table_like(title: str, data: pd.DataFrame | None, max_rows: int = 12) -> list[str]:
    if data is None or data.empty:
        return [f"## {title}", "No data available.", ""]
    return [f"## {title}", data.head(max_rows).to_csv(index=False), ""]


def _parse_datetime_series(values) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def _render_news_rows(data: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in data.head(get_config()["news_article_limit"]).iterrows():
        title = _first_present(row, ["新闻标题", "标题", "公告标题", "title"], default="No title")
        source = _first_present(row, ["文章来源", "来源", "_source", "source"], default="Unknown")
        date = _first_present(row, ["发布时间", "日期", "公告日期", "date"], default="")
        url = _first_present(row, ["新闻链接", "链接", "公告链接", "url"], default="")
        summary = _first_present(row, ["摘要", "summary", "简介"], default="")
        body = f"### {title} (source: {source})\nDate: {date}\n"
        if summary:
            body += f"{summary}\n"
        if url:
            body += f"Link: {url}\n"
        rows.append(body)
    return rows


def _safe_call(func):
    try:
        return func()
    except Exception:
        return pd.DataFrame()


def _safe_tiantian_fund_tables(symbol: str, curr_date: str | None):
    try:
        return get_fund_profile_tables(symbol, curr_date)
    except Exception:
        return []


def _first_present(row: pd.Series, columns: list[str], default: str = ""):
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return default


def _compact_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d")


def _to_akshare_statement_symbol(ticker: str) -> str:
    normalized = normalize_ticker_symbol(ticker)
    code, _, exchange = normalized.partition(".")
    if exchange in ("SH", "SZ", "BJ"):
        return f"{exchange}{code}"
    return to_akshare_symbol(ticker)


def _prefixed_cn_symbol(ticker: str) -> str:
    normalized = normalize_ticker_symbol(ticker)
    code, _, exchange = normalized.partition(".")
    if exchange == "SH":
        return f"sh{code}"
    if exchange == "SZ":
        return f"sz{code}"
    if exchange == "BJ":
        return f"bj{code}"
    return normalized.lower()


def _to_akshare_index_symbol(symbol: str) -> str:
    normalized = symbol.strip().lower()
    if normalized.startswith(("sh", "sz", "bj")) and len(normalized) == 8:
        return normalized
    upper = symbol.strip().upper()
    code, _, exchange = upper.partition(".")
    if exchange in ("SS", "SH"):
        return f"sh{code}"
    if exchange == "SZ":
        return f"sz{code}"
    if exchange == "BJ":
        return f"bj{code}"
    if code.isdigit() and len(code) == 6:
        if code.startswith(("3", "399")):
            return f"sz{code}"
        return f"sh{code}"
    return normalized


def _fund_holdings_year(curr_date: str | None) -> str:
    if not curr_date:
        return str(datetime.now().year)
    return str(pd.to_datetime(curr_date).year)


def _indicator_descriptions() -> dict[str, str]:
    return {
        "close_50_sma": "50 SMA: medium-term trend indicator for support, resistance, and trend direction.",
        "close_200_sma": "200 SMA: long-term trend benchmark for strategic trend confirmation.",
        "close_10_ema": "10 EMA: short-term responsive trend indicator.",
        "macd": "MACD: momentum indicator based on moving-average convergence and divergence.",
        "macds": "MACD Signal: signal line used to identify crossovers.",
        "macdh": "MACD Histogram: visualizes the gap between MACD and signal line.",
        "rsi": "RSI: momentum indicator for overbought or oversold conditions.",
        "boll": "Bollinger Middle: moving-average basis for Bollinger Bands.",
        "boll_ub": "Bollinger Upper Band: upper volatility band.",
        "boll_lb": "Bollinger Lower Band: lower volatility band.",
        "atr": "ATR: average true range volatility indicator.",
        "vwma": "VWMA: volume-weighted moving average.",
        "mfi": "MFI: money flow index using price and volume.",
    }


def get_ticker_display_name(ticker: str, force_refresh: bool = False) -> str:
    """Get verified Chinese display name (abbreviation or full name) for China instruments.

    ``force_refresh=True`` bypasses the on-disk cache so a rename is picked up
    immediately instead of waiting out the TTL (7 days).
    """
    if force_refresh:
        _invalidate_display_name_cache(ticker)
    normalized = normalize_ticker_symbol(ticker)
    market_type = detect_market_type(normalized)
    if market_type == MarketType.CN_FUND:
        return _get_cn_fund_display_name(ticker, normalized)
    if market_type != MarketType.CN_A:
        return ticker
    return _get_cn_a_ticker_display_name(ticker, normalized)


def _invalidate_display_name_cache(ticker: str) -> None:
    normalized = normalize_ticker_symbol(ticker)

    def match(key: str) -> bool:
        return (
            key.startswith("disk:_get_cn_a_ticker_display_name:")
            or key.startswith("disk:_get_cn_fund_display_name:")
        ) and normalized in key

    invalidate_disk_cache_where("akshare", match)


@akshare_disk_cache(expire=86400 * 7)  # 7 天磁盘缓存；force_refresh 可立即刷新
def _get_cn_a_ticker_display_name(ticker: str, normalized: str) -> str:
    code = to_akshare_symbol(normalized)

    if detect_instrument_type(normalized) == InstrumentType.FUND:
        try:
            ak = _ak()
            df = ak.fund_overview_em(symbol=code)
            if not df.empty:
                for _, row in df.iterrows():
                    if row.get("项目") in ("基金简称", "基金简称：", "基金全称"):
                        return str(row.get("内容"))
        except Exception:
            pass
    else:
        try:
            ak = _ak()
            df = ak.stock_individual_info_em(symbol=code)
            if not df.empty:
                for _, row in df.iterrows():
                    if row.get("item") in ("股票简称", "股票全称"):
                        return str(row.get("value"))
        except Exception:
            pass
    return ticker


@akshare_disk_cache(expire=86400 * 7)  # 7 天磁盘缓存；force_refresh 可立即刷新
def _get_cn_fund_display_name(ticker: str, normalized: str) -> str:
    code = to_akshare_symbol(normalized)
    for table in _safe_tiantian_fund_tables(code, None):
        if table.title != "Tiantian Fund Overview":
            continue
        for _, row in table.data.iterrows():
            if row.get("项目") == "基金简称" and row.get("内容"):
                return str(row.get("内容"))
    return ticker
