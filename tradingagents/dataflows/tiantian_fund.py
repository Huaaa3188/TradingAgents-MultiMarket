from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from parsel import Selector

from .cache import disk_cache
from .contracts import DataResult, SourceMeta, data_notice

DETAIL_URL = "https://fund.eastmoney.com/pingzhongdata/{symbol}.js"
HOLDINGS_URL = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
REQUEST_TIMEOUT = 10


class TiantianFundDataError(RuntimeError):
    """Recoverable Tiantian Fund / Eastmoney data-source error."""


@dataclass(frozen=True)
class TiantianTable:
    title: str
    data: pd.DataFrame
    max_rows: int = 12


def get_fund_profile_tables_result(
    symbol: str,
    curr_date: str | None = None,
    holdings_limit: int = 10,
) -> DataResult:
    code = _pure_fund_code(symbol)
    try:
        tables = get_fund_profile_tables(code, curr_date, holdings_limit)
    except TiantianFundDataError as exc:
        drifted = _detect_page_drift(code)
        notice = (
            data_notice(
                "js_schema_drift",
                "The Tiantian pingzhongdata page no longer exposes the expected JS variables "
                "(fS_name/fS_code/Data_netWorthTrend) — the site structure appears to have changed.",
                source="eastmoney_pingzhongdata",
                severity="error",
            )
            if drifted
            else data_notice(
                "source_error",
                "Tiantian Fund profile data could not be loaded.",
                source="eastmoney_pingzhongdata",
                detail=str(exc),
                severity="error",
            )
        )
        return DataResult(
            meta=SourceMeta(
                vendor="tiantian_fund",
                source="eastmoney_pingzhongdata",
                symbol=code,
                semantic="fund_profile",
                retrieved_at=_now_timestamp(),
            ),
            payload=[],
            notices=(notice,),
            ok=False,
            missing_reason="no_profile_data",
            error_type="js_schema_drift" if drifted else "source_error",
        )

    return DataResult(
        meta=SourceMeta(
            vendor="tiantian_fund",
            source="eastmoney_pingzhongdata",
            symbol=code,
            semantic="fund_profile",
            as_of=_tables_as_of(tables),
            retrieved_at=_now_timestamp(),
        ),
        payload=tables,
        ok=bool(tables),
        missing_reason=None if tables else "no_profile_data",
    )


@disk_cache("tiantian_fund", expire=86400)
def get_fund_profile_tables(
    symbol: str,
    curr_date: str | None = None,
    holdings_limit: int = 10,
) -> list[TiantianTable]:
    code = _pure_fund_code(symbol)
    script = _http_get_text(DETAIL_URL.format(symbol=code))
    tables = _parse_detail_script(script, code, curr_date)

    holdings = _safe_fetch_top_holdings(code, curr_date, holdings_limit)
    if not holdings.empty:
        tables.append(TiantianTable("Tiantian Fund Top Holdings", holdings, holdings_limit))

    non_empty_tables = [table for table in tables if table.data is not None and not table.data.empty]
    if not non_empty_tables:
        raise TiantianFundDataError(f"No Tiantian Fund profile data returned for {code}")
    return non_empty_tables


def get_fund_nav_history_result(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> DataResult:
    code = _pure_fund_code(symbol)
    try:
        data = get_fund_nav_history(code, start_date, end_date)
    except TiantianFundDataError as exc:
        drifted = _detect_page_drift(code)
        notice = (
            data_notice(
                "js_schema_drift",
                "The Tiantian nav-trend page no longer exposes the expected JS variables "
                "(fS_name/fS_code/Data_netWorthTrend) — the site structure appears to have changed.",
                source="eastmoney_nav_trend",
                severity="error",
            )
            if drifted
            else data_notice(
                "source_error",
                "Tiantian Fund NAV history could not be loaded.",
                source="eastmoney_nav_trend",
                detail=str(exc),
                severity="error",
            )
        )
        return DataResult(
            meta=SourceMeta(
                vendor="tiantian_fund",
                source="eastmoney_nav_trend",
                symbol=code,
                semantic="nav",
                retrieved_at=_now_timestamp(),
            ),
            payload=pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"]),
            notices=(notice,),
            ok=False,
            missing_reason="no_nav_data",
            error_type="js_schema_drift" if drifted else "source_error",
        )

    return DataResult(
        meta=SourceMeta(
            vendor="tiantian_fund",
            source="eastmoney_nav_trend",
            symbol=code,
            semantic="nav",
            as_of=_frame_as_of(data),
            retrieved_at=_now_timestamp(),
        ),
        payload=data,
        ok=not data.empty,
        missing_reason=None if not data.empty else "no_nav_data",
    )


@disk_cache("tiantian_fund", expire=14400)
def get_fund_nav_history(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return daily fund NAV history as OHLCV-shaped rows for tool compatibility."""
    code = _pure_fund_code(symbol)
    script = _http_get_text(DETAIL_URL.format(symbol=code))
    trend = _extract_js_var(script, "Data_netWorthTrend")
    data = _nav_history_frame(trend, start_date, end_date)
    if data.empty:
        raise TiantianFundDataError(f"No Tiantian Fund NAV data returned for {code}")
    return data


_NAV_REQUIRED_JS_VARS = ("fS_name", "fS_code", "Data_netWorthTrend")


def _page_structure_drifted(script: str, required_vars: tuple[str, ...] = _NAV_REQUIRED_JS_VARS) -> bool:
    """Heuristic for a redesign that renamed/removed expected JS variables.

    Several of the variables the parser relies on being simultaneously absent
    (not merely empty) is cheap evidence that the pingzhongdata page structure
    changed, as opposed to an ordinary gap in one dataset. An unreadable page
    counts as drifted too.
    """
    if not script or not script.strip():
        return True
    present = sum(
        1 for var in required_vars if re.search(rf"\bvar\s+{re.escape(var)}\s*=", script)
    )
    return present < len(required_vars) - 1


def _detect_page_drift(symbol: str) -> bool:
    """Read the raw pingzhongdata page and report whether it is structurally drifted."""
    try:
        script = _http_get_text(DETAIL_URL.format(symbol=symbol))
    except TiantianFundDataError:
        return True
    return _page_structure_drifted(script)


def _http_get_text(url: str, params: dict[str, Any] | None = None) -> str:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "Mozilla/5.0 TradingAgents/0.2.5",
                "Referer": "https://fund.eastmoney.com/",
            },
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        return response.text
    except requests.RequestException as exc:
        raise TiantianFundDataError(f"Tiantian Fund request failed: {exc}") from exc


def _parse_detail_script(
    script: str,
    symbol: str,
    curr_date: str | None,
) -> list[TiantianTable]:
    values = {
        name: _extract_js_var(script, name)
        for name in (
            "fS_name",
            "fS_code",
            "fund_sourceRate",
            "fund_Rate",
            "fund_minsg",
            "syl_1y",
            "syl_3y",
            "syl_6y",
            "syl_1n",
            "Data_netWorthTrend",
            "Data_fluctuationScale",
            "Data_assetAllocation",
            "Data_holderStructure",
            "Data_currentFundManager",
        )
    }

    tables = [
        TiantianTable("Tiantian Fund Overview", _overview_frame(values, symbol)),
        TiantianTable("Tiantian Fund Returns", _return_frame(values)),
        TiantianTable("Tiantian Fund Latest NAV", _latest_nav_frame(values, curr_date), 5),
        TiantianTable("Tiantian Fund Scale", _scale_frame(values, curr_date), 5),
        TiantianTable("Tiantian Fund Asset Allocation", _series_frame(values.get("Data_assetAllocation"), curr_date), 5),
        TiantianTable("Tiantian Fund Holder Structure", _series_frame(values.get("Data_holderStructure"), curr_date), 5),
        TiantianTable("Tiantian Fund Managers", _manager_frame(values, curr_date), 5),
    ]
    return [table for table in tables if not table.data.empty]


def _safe_fetch_top_holdings(symbol: str, curr_date: str | None, holdings_limit: int) -> pd.DataFrame:
    years = _candidate_holding_years(curr_date)
    for year in years:
        try:
            text = _http_get_text(
                HOLDINGS_URL,
                params={
                    "type": "jjcc",
                    "code": symbol,
                    "topline": str(holdings_limit),
                    "year": str(year),
                    "month": "",
                },
            )
        except TiantianFundDataError:
            continue
        holdings = _parse_holdings_response(text, curr_date)
        if not holdings.empty:
            return holdings.head(holdings_limit)
    return pd.DataFrame()


def _parse_holdings_response(text: str, curr_date: str | None) -> pd.DataFrame:
    content = _extract_js_object_string_property(text, "content")
    if not content:
        return pd.DataFrame()

    cutoff = _extract_cutoff_date(content)
    if cutoff and curr_date and pd.to_datetime(cutoff) > pd.to_datetime(curr_date):
        return pd.DataFrame()

    selector = Selector(text=content)
    rows = []
    for tr in selector.css("tbody tr"):
        cells = [_clean_text(td.xpath("string()").get()) for td in tr.css("td")]
        if len(cells) < 9:
            continue
        rows.append(
            {
                "序号": cells[0],
                "股票代码": cells[1],
                "股票名称": cells[2],
                "占净值比例": cells[6],
                "持股数（万股）": cells[7],
                "持仓市值（万元）": cells[8],
                "截止日期": cutoff or "",
                "来源": "天天基金/东方财富",
            }
        )
    return pd.DataFrame(rows)


def _overview_frame(values: dict[str, Any], symbol: str) -> pd.DataFrame:
    has_source_value = any(value not in (None, "", [], {}) for value in values.values())
    rows = []
    if values.get("fS_code") or has_source_value:
        rows.append({"项目": "基金代码", "内容": values.get("fS_code") or symbol, "来源": "天天基金/东方财富"})
    if values.get("fS_name"):
        rows.append({"项目": "基金简称", "内容": values.get("fS_name"), "来源": "天天基金/东方财富"})
    for key, label in (
        ("fund_sourceRate", "原费率"),
        ("fund_Rate", "现费率"),
        ("fund_minsg", "最小申购金额"),
    ):
        value = values.get(key)
        if value not in (None, ""):
            rows.append({"项目": label, "内容": value, "来源": "天天基金/东方财富"})
    return pd.DataFrame([row for row in rows if row["内容"] not in (None, "")])


def _return_frame(values: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, label in (
        ("syl_1y", "近1月收益率(%)"),
        ("syl_3y", "近3月收益率(%)"),
        ("syl_6y", "近6月收益率(%)"),
        ("syl_1n", "近1年收益率(%)"),
    ):
        value = values.get(key)
        if value not in (None, ""):
            rows.append({"指标": label, "数值": value, "来源": "天天基金/东方财富"})
    return pd.DataFrame(rows)


def _latest_nav_frame(values: dict[str, Any], curr_date: str | None) -> pd.DataFrame:
    rows = []
    for item in values.get("Data_netWorthTrend") or []:
        if not isinstance(item, dict) or "x" not in item:
            continue
        date = pd.to_datetime(item.get("x"), unit="ms", errors="coerce")
        if pd.isna(date):
            continue
        rows.append(
            {
                "日期": date.strftime("%Y-%m-%d"),
                "单位净值": item.get("y"),
                "净值回报(%)": item.get("equityReturn"),
                "分红/拆分": item.get("unitMoney") or "",
                "来源": "天天基金/东方财富",
            }
        )
    return _latest_by_date(pd.DataFrame(rows), "日期", curr_date, limit=5)


def _nav_history_frame(
    trend: Any,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    rows = []
    for item in trend or []:
        if not isinstance(item, dict) or "x" not in item:
            continue
        date = pd.to_datetime(item.get("x"), unit="ms", errors="coerce")
        nav = pd.to_numeric(item.get("y"), errors="coerce")
        if pd.isna(date) or pd.isna(nav):
            continue
        rows.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Open": nav,
                "High": nav,
                "Low": nav,
                "Close": nav,
                "Volume": 0,
                "Pct Change": pd.to_numeric(item.get("equityReturn"), errors="coerce"),
                "Source": "天天基金/东方财富",
            }
        )

    data = pd.DataFrame(rows)
    if data.empty:
        return data

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    if start_date:
        data = data[data["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data["Date"] <= pd.to_datetime(end_date)]
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    return data


def _scale_frame(values: dict[str, Any], curr_date: str | None) -> pd.DataFrame:
    data = values.get("Data_fluctuationScale")
    if not isinstance(data, dict):
        return pd.DataFrame()
    categories = data.get("categories") or []
    series = data.get("series") or []
    if not categories or not series:
        return pd.DataFrame()

    rows = []
    for idx, date in enumerate(categories):
        period = series[idx] if idx < len(series) and isinstance(series[idx], dict) else {}
        rows.append(
            {
                "日期": date,
                "净资产规模(亿元)": period.get("y", ""),
                "较上期环比": period.get("mom", ""),
                "来源": "天天基金/东方财富",
            }
        )
    return _latest_by_date(pd.DataFrame(rows), "日期", curr_date, limit=5)


def _series_frame(data: Any, curr_date: str | None) -> pd.DataFrame:
    if not isinstance(data, dict):
        return pd.DataFrame()
    categories = data.get("categories") or []
    series = data.get("series") or []
    rows = []
    for idx, date in enumerate(categories):
        row = {"日期": date, "来源": "天天基金/东方财富"}
        for item in series:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            row[name] = _series_value(item.get("data"), idx)
        rows.append(row)
    return _latest_by_date(pd.DataFrame(rows), "日期", curr_date, limit=5)


def _manager_frame(values: dict[str, Any], curr_date: str | None) -> pd.DataFrame:
    rows = []
    for manager in values.get("Data_currentFundManager") or []:
        if not isinstance(manager, dict):
            continue
        as_of = _manager_as_of_date(manager)
        if as_of and curr_date and pd.to_datetime(as_of) > pd.to_datetime(curr_date):
            continue
        rows.append(
            {
                "基金经理": manager.get("name") or "",
                "评级": manager.get("star") or "",
                "从业年限": manager.get("workTime") or "",
                "管理规模": manager.get("fundSize") or "",
                "数据日期": as_of or "",
                "来源": "天天基金/东方财富",
            }
        )
    return pd.DataFrame(rows)


def _extract_js_var(script: str, name: str) -> Any:
    marker = re.search(rf"\bvar\s+{re.escape(name)}\s*=", script)
    if not marker:
        return None
    start = _skip_ws(script, marker.end())
    try:
        literal, _ = _scan_js_value(script, start)
        return _parse_js_literal(literal)
    except (ValueError, json.JSONDecodeError):
        return None


def _extract_js_object_string_property(text: str, property_name: str) -> str:
    marker = re.search(rf"\b{re.escape(property_name)}\s*:\s*", text)
    if not marker:
        return ""
    start = _skip_ws(text, marker.end())
    try:
        literal, _ = _scan_js_value(text, start)
        value = _parse_js_literal(literal)
    except (ValueError, json.JSONDecodeError):
        return ""
    return value if isinstance(value, str) else ""


def _scan_js_value(text: str, start: int) -> tuple[str, int]:
    if start >= len(text):
        raise ValueError("empty JavaScript value")
    first = text[start]
    if first in ("'", '"'):
        return _scan_js_string(text, start)
    if first in ("[", "{"):
        return _scan_balanced_js_value(text, start)

    end = text.find(";", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip(), end


def _scan_js_string(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    idx = start + 1
    escaped = False
    while idx < len(text):
        char = text[idx]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return text[start : idx + 1], idx + 1
        idx += 1
    raise ValueError("unterminated JavaScript string")


def _scan_balanced_js_value(text: str, start: int) -> tuple[str, int]:
    pairs = {"[": "]", "{": "}"}
    stack = [pairs[text[start]]]
    idx = start + 1
    in_string = ""
    escaped = False
    while idx < len(text):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = ""
        elif char in ("'", '"'):
            in_string = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[start : idx + 1], idx + 1
        idx += 1
    raise ValueError("unterminated JavaScript object")


def _parse_js_literal(literal: str) -> Any:
    literal = literal.strip()
    if literal.startswith("'") and literal.endswith("'"):
        literal = '"' + literal[1:-1].replace('"', '\\"') + '"'
    return json.loads(literal)


def _candidate_holding_years(curr_date: str | None) -> list[int]:
    base_year = pd.to_datetime(curr_date).year if curr_date else datetime.now().year
    return [base_year, base_year - 1]


def _extract_cutoff_date(content: str) -> str:
    match = re.search(r"截止至：\s*(?:<[^>]+>)*\s*(\d{4}-\d{2}-\d{2})", content)
    return match.group(1) if match else ""


def _latest_by_date(
    data: pd.DataFrame,
    date_column: str,
    curr_date: str | None,
    limit: int,
) -> pd.DataFrame:
    if data.empty or date_column not in data.columns:
        return data
    result = data.copy()
    parsed = pd.to_datetime(result[date_column], errors="coerce")
    result = result.assign(_parsed_date=parsed).dropna(subset=["_parsed_date"])
    if curr_date:
        result = result[result["_parsed_date"] <= pd.to_datetime(curr_date)]
    if result.empty:
        return pd.DataFrame(columns=list(data.columns))
    result = result.sort_values("_parsed_date", ascending=False).head(limit)
    return result.drop(columns=["_parsed_date"])


def _manager_as_of_date(manager: dict[str, Any]) -> str:
    for section in ("power", "profit"):
        value = manager.get(section)
        if isinstance(value, dict) and value.get("jzrq"):
            return str(value["jzrq"])
    return ""


def _series_value(values: Any, idx: int) -> Any:
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return ""


def _tables_as_of(tables: list[TiantianTable]) -> str | None:
    dates = []
    for table in tables:
        if table.data is None or table.data.empty:
            continue
        for column in table.data.columns:
            if "日期" not in str(column) and "date" not in str(column).lower():
                continue
            parsed = pd.to_datetime(table.data[column], errors="coerce").dropna()
            if not parsed.empty:
                dates.append(parsed.max())
    if not dates:
        return None
    return max(dates).strftime("%Y-%m-%d")


def _frame_as_of(data: pd.DataFrame) -> str | None:
    if data is None or data.empty or "Date" not in data.columns:
        return None
    parsed = pd.to_datetime(data["Date"], errors="coerce").dropna()
    if parsed.empty:
        return None
    return parsed.max().strftime("%Y-%m-%d")


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pure_fund_code(symbol: str) -> str:
    return symbol.split(".", 1)[0].strip()


def _skip_ws(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", value)
