import time

import pandas as pd
import pytest
from diskcache import Cache

import tradingagents.default_config as default_config
from tradingagents.dataflows import akshare, cache as dataflow_cache, tiantian_fund
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.contracts import parse_contract_gate_status
from tradingagents.dataflows.interface import route_to_vendor


@pytest.fixture(autouse=True)
def reset_dataflow_config():
    set_config(default_config.DEFAULT_CONFIG.copy())
    yield
    set_config(default_config.DEFAULT_CONFIG.copy())


@pytest.fixture(autouse=True)
def isolate_akshare_cache(monkeypatch, tmp_path):
    dataflow_cache.reset_cache_stats()
    temp_cache = Cache(str(tmp_path / "test_akshare_cache"))
    dataflow_cache.set_disk_cache("akshare", temp_cache)
    dataflow_cache.set_disk_cache("tiantian_fund", Cache(str(tmp_path / "test_tiantian_cache")))
    monkeypatch.setattr(akshare, "get_fund_profile_tables", lambda symbol, curr_date: [])
    akshare._macro_news_source_health.clear()
    yield
    dataflow_cache.clear_disk_cache("akshare")
    dataflow_cache.clear_disk_cache("tiantian_fund")
    dataflow_cache.reset_cache_stats()
    akshare._macro_news_source_health.clear()


class FakeAkShare:
    def __init__(self):
        self.calls = []

    def fund_etf_hist_em(self, **kwargs):
        self.calls.append(("fund_etf_hist_em", kwargs))
        return _ohlcv_frame()

    def fund_lof_hist_em(self, **kwargs):
        self.calls.append(("fund_lof_hist_em", kwargs))
        return _ohlcv_frame()

    def stock_zh_a_hist(self, **kwargs):
        self.calls.append(("stock_zh_a_hist", kwargs))
        return _ohlcv_frame()

    def stock_zh_a_daily(self, **kwargs):
        self.calls.append(("stock_zh_a_daily", kwargs))
        return _lowercase_ohlcv_frame()

    def stock_zh_a_hist_tx(self, **kwargs):
        self.calls.append(("stock_zh_a_hist_tx", kwargs))
        return _lowercase_ohlcv_frame()

    def fund_etf_hist_sina(self, **kwargs):
        self.calls.append(("fund_etf_hist_sina", kwargs))
        return _lowercase_ohlcv_frame()

    def fund_overview_em(self, **kwargs):
        self.calls.append(("fund_overview_em", kwargs))
        return pd.DataFrame(
            [
                {"项目": "基金简称", "内容": "沪深300ETF"},
                {"项目": "基金类型", "内容": "ETF"},
                {"项目": "基金规模", "内容": "100亿元"},
                {"项目": "基金管理人", "内容": "测试基金公司"},
            ]
        )

    def fund_fee_em(self, **kwargs):
        self.calls.append(("fund_fee_em", kwargs))
        return pd.DataFrame([{"费用类型": "管理费率", "费率": "0.50%"}])

    def fund_portfolio_hold_em(self, **kwargs):
        self.calls.append(("fund_portfolio_hold_em", kwargs))
        return pd.DataFrame([{"股票名称": "贵州茅台", "占净值比例": "5.00%"}])

    def stock_balance_sheet_by_report_em(self, **kwargs):
        self.calls.append(("stock_balance_sheet_by_report_em", kwargs))
        return _statement_frame()

    def stock_cash_flow_sheet_by_report_em(self, **kwargs):
        self.calls.append(("stock_cash_flow_sheet_by_report_em", kwargs))
        return _statement_frame()

    def stock_profit_sheet_by_report_em(self, **kwargs):
        self.calls.append(("stock_profit_sheet_by_report_em", kwargs))
        return _statement_frame()

    def stock_news_em(self, **kwargs):
        self.calls.append(("stock_news_em", kwargs))
        return pd.DataFrame(
            [
                {
                    "新闻标题": "公司新闻",
                    "文章来源": "测试来源",
                    "发布时间": "2026-01-02",
                    "新闻链接": "https://example.invalid/stock",
                }
            ]
        )

    def fund_announcement_report_em(self, **kwargs):
        self.calls.append(("fund_announcement_report_em", kwargs))
        return pd.DataFrame(
            [
                {
                    "公告标题": "基金定期报告",
                    "公告日期": "2026-01-02",
                    "公告链接": "https://example.invalid/fund-report",
                }
            ]
        )

    def fund_announcement_dividend_em(self, **kwargs):
        self.calls.append(("fund_announcement_dividend_em", kwargs))
        return pd.DataFrame()

    def fund_announcement_personnel_em(self, **kwargs):
        self.calls.append(("fund_announcement_personnel_em", kwargs))
        return pd.DataFrame()


def test_get_stock_routes_etf_history(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_stock("510300", "2026-01-01", "2026-01-03")

    assert "AkShare data for 510300.SH" in result
    assert "Date,Open,High,Low,Close,Volume" in result
    assert fake.calls[0][0] == "fund_etf_hist_em"
    assert fake.calls[0][1]["symbol"] == "510300"


def test_get_stock_routes_lof_history(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    akshare.get_stock("161725", "2026-01-01", "2026-01-03")

    assert fake.calls[0][0] == "fund_lof_hist_em"
    assert fake.calls[0][1]["symbol"] == "161725"


def test_get_stock_routes_equity_history(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    assert fake.calls[0][0] == "stock_zh_a_hist"
    assert fake.calls[0][1]["symbol"] == "600519"


def test_get_stock_falls_back_to_sina_equity_history(monkeypatch):
    class FakeAkShareWithFailedEastmoney(FakeAkShare):
        def stock_zh_a_hist(self, **kwargs):
            self.calls.append(("stock_zh_a_hist", kwargs))
            raise RuntimeError("eastmoney unavailable")

    fake = FakeAkShareWithFailedEastmoney()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    assert "AkShare data for 600519.SH" in result
    assert "2026-01-03" in result
    assert fake.calls[1][0] == "stock_zh_a_daily"
    assert fake.calls[1][1]["symbol"] == "sh600519"


def test_get_stock_falls_back_when_primary_ohlcv_shape_drifts(monkeypatch):
    class FakeAkShareWithDriftedEastmoney(FakeAkShare):
        def stock_zh_a_hist(self, **kwargs):
            self.calls.append(("stock_zh_a_hist", kwargs))
            return pd.DataFrame(
                [
                    {"日期": "2026-01-01", "开盘": 10, "最高": 11, "最低": 9, "成交量": 1000},
                ]
            )

    fake = FakeAkShareWithDriftedEastmoney()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    assert "AkShare data for 600519.SH" in result
    assert "2026-01-03" in result
    assert fake.calls[0][0] == "stock_zh_a_hist"
    assert fake.calls[1][0] == "stock_zh_a_daily"


def test_get_stock_falls_back_to_sina_fund_history(monkeypatch):
    class FakeAkShareWithFailedEastmoney(FakeAkShare):
        def fund_etf_hist_em(self, **kwargs):
            self.calls.append(("fund_etf_hist_em", kwargs))
            raise RuntimeError("eastmoney unavailable")

    fake = FakeAkShareWithFailedEastmoney()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_stock("510300", "2026-01-01", "2026-01-03")

    assert "AkShare data for 510300.SH" in result
    assert "2026-01-03" in result
    assert fake.calls[1][0] == "fund_etf_hist_sina"
    assert fake.calls[1][1]["symbol"] == "sh510300"


def test_get_indicator_uses_normalized_ohlcv(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_indicator("510300", "rsi", "2026-01-03", 2)

    assert "## rsi values" in result
    assert "2026-01-02" in result
    assert "RSI:" in result


def test_get_stock_routes_cn_otc_fund_nav(monkeypatch):
    monkeypatch.setattr(akshare, "_ak", lambda: (_ for _ in ()).throw(AssertionError("_ak should not be called")))
    monkeypatch.setattr(akshare, "get_fund_nav_history", lambda symbol, start, end: _nav_frame())

    result = akshare.get_stock("012920", "2026-01-01", "2026-01-03")

    assert "Tiantian Fund NAV data for 012920" in result
    assert "Date,Open,High,Low,Close,Volume" in result
    assert "2026-01-03" in result
    checks = parse_contract_gate_status(result)
    assert checks[0]["semantic"] == "nav"
    assert checks[0]["warnings"] == ["nav_semantic"]


def test_get_stock_contract_marks_cn_otc_fund_as_nav(monkeypatch):
    monkeypatch.setattr(akshare, "_ak", lambda: (_ for _ in ()).throw(AssertionError("_ak should not be called")))
    monkeypatch.setattr(akshare, "get_fund_nav_history", lambda symbol, start, end: _nav_frame())

    result = akshare.get_stock_result("012920", "2026-01-01", "2026-01-03")

    assert result.ok is True
    assert result.meta.semantic == "nav"
    assert result.meta.source == "tiantian_fund_nav"
    assert result.meta.as_of == "2026-01-03"
    assert result.rows == 3


def test_newly_recognized_otc_fund_codes_route_to_nav(monkeypatch):
    """R1 — 005827 / 110011 are now OTC funds and must go through the Tiantian
    NAV path (semantic='nav'), never the equity/ETF OHLCV sources."""
    monkeypatch.setattr(akshare, "_ak", lambda: (_ for _ in ()).throw(AssertionError("_ak should not be called")))
    monkeypatch.setattr(akshare, "get_fund_nav_history", lambda symbol, start, end: _nav_frame())

    for code in ("005827", "110011"):
        result = akshare.get_stock_result(code, "2026-01-01", "2026-01-03")
        assert result.ok is True, f"{code} should resolve via Tiantian NAV"
        assert result.meta.semantic == "nav"
        assert result.meta.source == "tiantian_fund_nav"
        assert result.meta.symbol == code

        rendered = akshare.get_stock(code, "2026-01-01", "2026-01-03")
        checks = parse_contract_gate_status(rendered)
        assert checks[0]["semantic"] == "nav"
        assert checks[0]["warnings"] == ["nav_semantic"]



def test_get_stock_contract_records_schema_drift_before_fallback(monkeypatch):
    class FakeAkShareWithDriftedEastmoney(FakeAkShare):
        def stock_zh_a_hist(self, **kwargs):
            self.calls.append(("stock_zh_a_hist", kwargs))
            return pd.DataFrame(
                [
                    {"日期": "2026-01-01", "开盘": 10, "最高": 11, "最低": 9, "成交量": 1000},
                ]
            )

    fake = FakeAkShareWithDriftedEastmoney()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_stock_result("600519", "2026-01-01", "2026-01-03")

    assert result.ok is True
    assert result.error_type == "schema_drift"
    assert any(notice.code == "schema_drift" for notice in result.notices)
    assert result.meta.source == "sina_stock_zh_a_daily"


def test_get_stock_output_marks_schema_drift_gate_failed(monkeypatch):
    class FakeAkShareWithDriftedEastmoney(FakeAkShare):
        def stock_zh_a_hist(self, **kwargs):
            self.calls.append(("stock_zh_a_hist", kwargs))
            return pd.DataFrame(
                [
                    {"日期": "2026-01-01", "开盘": 10, "最高": 11, "最低": 9, "成交量": 1000},
                ]
            )

    fake = FakeAkShareWithDriftedEastmoney()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    checks = parse_contract_gate_status(result)
    assert checks[0]["status"] == "fail"
    assert checks[0]["failures"] == ["schema_drift"]


def test_get_stock_contract_marks_empty_data_with_missing_reason(monkeypatch):
    monkeypatch.setattr(
        akshare,
        "_load_ohlcv_result",
        lambda symbol, start, end: akshare.DataResult(
            meta=akshare.SourceMeta(
                vendor="akshare",
                source="akshare_ohlcv",
                symbol=akshare.normalize_ticker_symbol(symbol),
                semantic="ohlcv",
            ),
            payload=pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"]),
            ok=False,
            missing_reason="no_rows",
        ),
    )

    result = akshare.get_stock_result("600519", "2026-01-01", "2026-01-03")

    assert result.ok is False
    assert result.missing_reason == "no_rows"
    assert any(notice.code == "no_rows" for notice in result.notices)


def test_get_indicator_uses_cn_otc_fund_nav(monkeypatch):
    monkeypatch.setattr(akshare, "get_fund_nav_history", lambda symbol, start, end: _nav_frame())

    result = akshare.get_indicator("012920", "rsi", "2026-01-03", 2)

    assert "## rsi values" in result
    assert "Fund NAV note" in result
    assert "RSI:" in result
    checks = parse_contract_gate_status(result)
    assert checks[0]["semantic"] == "nav"
    assert checks[0]["warnings"] == ["nav_semantic"]


def test_get_fundamentals_returns_fund_profile(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_fundamentals("510300", "2026-01-03")

    assert "Listed Fund Profile for 510300.SH" in result
    assert "沪深300ETF" in result
    assert "管理费率" in result
    assert "贵州茅台" in result
    assert ("fund_portfolio_hold_em", {"symbol": "510300", "date": "2026"}) in fake.calls
    checks = parse_contract_gate_status(result)
    assert checks[0]["semantic"] == "fund_profile"


def test_get_fundamentals_returns_cn_otc_fund_profile(monkeypatch):
    monkeypatch.setattr(akshare, "_ak", lambda: (_ for _ in ()).throw(AssertionError("_ak should not be called")))
    monkeypatch.setattr(
        akshare,
        "get_fund_profile_tables",
        lambda symbol, curr_date: [
            tiantian_fund.TiantianTable(
                "Tiantian Fund Overview",
                pd.DataFrame(
                    [{"项目": "基金简称", "内容": "易方达全球成长精选混合(QDII)人民币A", "来源": "天天基金/东方财富"}]
                ),
            )
        ],
    )

    result = akshare.get_fundamentals("012920", "2026-06-04")

    assert "China OTC Fund Profile for 012920" in result
    assert "易方达全球成长精选混合(QDII)人民币A" in result
    assert "NAV trend" in result
    assert "AkShare Fund Overview" not in result
    checks = parse_contract_gate_status(result)
    assert checks[0]["semantic"] == "fund_profile"


def test_get_fundamentals_contract_reports_missing_otc_profile():
    result = akshare.get_fundamentals_result("012920", "2026-06-04")

    assert result.ok is False
    assert result.meta.semantic == "fund_profile"
    assert result.missing_reason == "no_profile_data"
    assert any(notice.code == "no_profile_data" for notice in result.notices)


def test_fund_financial_statements_are_not_applicable():
    assert "not applicable to listed fund 510300.SH" in akshare.get_balance_sheet("510300")
    assert "not applicable to listed fund 510300.SH" in akshare.get_cashflow("510300")
    assert "not applicable to listed fund 510300.SH" in akshare.get_income_statement("510300")
    assert "not applicable to OTC fund 012920" in akshare.get_balance_sheet("012920")


def test_equity_financial_statements_use_prefixed_akshare_symbol(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_balance_sheet("600519", curr_date="2026-01-03")

    assert "AkShare Balance Sheet for 600519.SH" in result
    assert ("stock_balance_sheet_by_report_em", {"symbol": "SH600519"}) in fake.calls


def test_sz_equity_financial_statements_use_prefixed_akshare_symbol(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    akshare.get_income_statement("000001", curr_date="2026-01-03")

    assert ("stock_profit_sheet_by_report_em", {"symbol": "SZ000001"}) in fake.calls


def test_fund_news_uses_fund_announcements_not_stock_news(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_news("510300", "2026-01-01", "2026-01-03")

    call_names = [name for name, _ in fake.calls]
    assert "510300.SH Listed Fund Announcements" in result
    assert "基金定期报告" in result
    assert "fund_announcement_report_em" in call_names
    assert "stock_news_em" not in call_names
    checks = parse_contract_gate_status(result)
    assert checks[0]["semantic"] == "news"
    assert checks[0]["status"] == "pass"


def test_news_contract_carries_fund_announcement_metadata(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_news_result("510300", "2026-01-01", "2026-01-03")

    assert result.ok is True
    assert result.meta.semantic == "news"
    assert result.meta.source == "akshare_fund_announcements"
    assert result.meta.as_of == "2026-01-02"
    assert result.rows == 1


def test_fund_news_falls_back_to_recent_announcements_before_end_date(monkeypatch):
    class FakeAkShareWithOldAnnouncements(FakeAkShare):
        def fund_announcement_report_em(self, **kwargs):
            self.calls.append(("fund_announcement_report_em", kwargs))
            return pd.DataFrame(
                [
                    {
                        "公告标题": "最近季度报告",
                        "公告日期": "2026-04-22",
                        "公告链接": "https://example.invalid/recent",
                    },
                    {
                        "公告标题": "更早年度报告",
                        "公告日期": "2026-03-31",
                        "公告链接": "https://example.invalid/older",
                    },
                ]
            )

    fake = FakeAkShareWithOldAnnouncements()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_news("510300", "2026-05-01", "2026-05-22")

    assert "Recent Listed Fund Announcements before 2026-05-22" in result
    assert "none found from 2026-05-01 to 2026-05-22" in result
    assert "最近季度报告" in result
    assert "更早年度报告" in result


def test_news_date_filter_excludes_next_day_date_only_rows():
    data = pd.DataFrame(
        [
            {"新闻标题": "窗口内", "发布时间": "2026-01-03"},
            {"新闻标题": "次日", "发布时间": "2026-01-04"},
        ]
    )

    result = akshare._filter_date_rows(data, "2026-01-01", "2026-01-03")

    assert result["新闻标题"].tolist() == ["窗口内"]


def test_fund_announcement_filter_excludes_next_day_date_only_rows():
    data = pd.DataFrame(
        [
            {"公告标题": "窗口内公告", "公告日期": "2026-01-03"},
            {"公告标题": "次日公告", "公告日期": "2026-01-04"},
        ]
    )

    result, used_fallback = akshare._filter_fund_announcements(
        data,
        "2026-01-01",
        "2026-01-03",
    )

    assert used_fallback is False
    assert result["公告标题"].tolist() == ["窗口内公告"]


def test_global_news_renders_china_macro_policy_and_filters_dates(monkeypatch):
    class FakeMacroAkShare:
        def stock_info_global_em(self):
            return pd.DataFrame(
                [
                    {
                        "标题": "央行强调保持流动性合理充裕",
                        "发布时间": "2026-05-22 10:00:00",
                        "来源": "东方财富",
                        "链接": "https://example.invalid/policy",
                    },
                    {
                        "标题": "未来消息不应进入回测",
                        "发布时间": "2026-05-23 09:00:00",
                        "来源": "东方财富",
                    },
                ]
            )

        def stock_info_global_cls(self, symbol="全部"):
            return pd.DataFrame(
                [
                    {
                        "标题": "央行强调保持流动性合理充裕",
                        "时间": "2026-05-22 10:01:00",
                        "来源": "财联社",
                    },
                    {
                        "标题": "证监会发布资本市场政策安排",
                        "时间": "2026-05-21 16:00:00",
                        "来源": "财联社",
                    },
                ]
            )

        def stock_info_global_sina(self):
            return pd.DataFrame()

        def stock_info_global_ths(self):
            return pd.DataFrame()

        def news_cctv(self, date):
            return pd.DataFrame(
                [
                    {
                        "标题": f"新闻联播政策摘要 {date}",
                        "日期": "2026-05-20",
                        "来源": "央视新闻",
                    }
                ]
            )

    monkeypatch.setattr(akshare, "_ak", lambda: FakeMacroAkShare())
    monkeypatch.setattr(akshare, "_is_historical_macro_news_date", lambda end_date: False)

    result = akshare.get_global_news("2026-05-22", look_back_days=2, limit=3)

    assert "China Macro and Policy News, from 2026-05-20 to 2026-05-22" in result
    assert "央行强调保持流动性合理充裕" in result
    assert "证监会发布资本市场政策安排" in result
    assert "新闻联播政策摘要" in result
    assert "未来消息不应进入回测" not in result
    assert result.count("央行强调保持流动性合理充裕") == 1


def test_global_news_empty_result_is_explicit(monkeypatch):
    class EmptyMacroAkShare:
        def stock_info_global_em(self):
            return pd.DataFrame()

        def stock_info_global_cls(self, symbol="全部"):
            return pd.DataFrame()

        def stock_info_global_sina(self):
            return pd.DataFrame()

        def stock_info_global_ths(self):
            return pd.DataFrame()

        def news_cctv(self, date):
            return pd.DataFrame()

    monkeypatch.setattr(akshare, "_ak", lambda: EmptyMacroAkShare())
    monkeypatch.setattr(akshare, "_is_historical_macro_news_date", lambda end_date: False)

    result = akshare.get_global_news("2026-05-22", look_back_days=1, limit=5)

    assert "No AkShare China macro/policy news found between 2026-05-21 and 2026-05-22" in result
    assert "Do not infer policy catalysts" in result


def test_global_news_source_errors_degrade_to_notice(monkeypatch):
    class FailingMacroAkShare:
        def stock_info_global_em(self):
            raise RuntimeError("eastmoney down")

        def stock_info_global_cls(self, symbol="全部"):
            raise RuntimeError("cls down")

        def stock_info_global_sina(self):
            raise RuntimeError("sina down")

        def stock_info_global_ths(self):
            raise RuntimeError("ths down")

        def news_cctv(self, date):
            raise RuntimeError("cctv down")

    monkeypatch.setattr(akshare, "_ak", lambda: FailingMacroAkShare())
    monkeypatch.setattr(akshare, "_is_historical_macro_news_date", lambda end_date: False)

    result = akshare.get_global_news("2026-05-22", look_back_days=1, limit=5)

    assert "Error fetching AkShare China macro/policy news" in result
    assert "local macro news is unavailable" in result


def test_global_news_historical_dates_skip_realtime_sources(monkeypatch):
    calls = []

    class HistoricalMacroAkShare:
        def stock_info_global_em(self):
            calls.append("eastmoney")
            return pd.DataFrame()

        def stock_info_global_cls(self, symbol="全部"):
            calls.append("cls")
            return pd.DataFrame()

        def stock_info_global_sina(self):
            calls.append("sina")
            return pd.DataFrame()

        def stock_info_global_ths(self):
            calls.append("ths")
            return pd.DataFrame()

        def news_cctv(self, date):
            calls.append(f"cctv:{date}")
            return pd.DataFrame(
                [
                    {
                        "标题": f"历史宏观政策 {date}",
                        "日期": "2026-05-20",
                        "来源": "央视新闻",
                    }
                ]
            )

    monkeypatch.setattr(akshare, "_ak", lambda: HistoricalMacroAkShare())
    monkeypatch.setattr(akshare, "_is_historical_macro_news_date", lambda end_date: True)

    result = akshare.get_global_news("2026-05-22", look_back_days=2, limit=5)

    assert "历史宏观政策" in result
    assert calls
    assert all(call.startswith("cctv:") for call in calls)


def test_global_news_source_timeout_degrades_to_available_sources(monkeypatch):
    class SlowThenFastMacroAkShare:
        def stock_info_global_em(self):
            time.sleep(0.05)
            return pd.DataFrame()

        def stock_info_global_sina(self):
            return pd.DataFrame(
                [
                    {
                        "标题": "快速宏观来源",
                        "发布时间": "2026-05-22 09:00:00",
                        "来源": "新浪财经",
                    }
                ]
            )

        def stock_info_global_ths(self):
            return pd.DataFrame()

        def stock_info_global_cls(self, symbol="全部"):
            return pd.DataFrame()

        def news_cctv(self, date):
            return pd.DataFrame()

    monkeypatch.setattr(akshare, "_ak", lambda: SlowThenFastMacroAkShare())
    monkeypatch.setattr(akshare, "_is_historical_macro_news_date", lambda end_date: False)
    monkeypatch.setattr(akshare, "_MACRO_NEWS_SOURCE_TIMEOUT_SECONDS", 0.01)

    result = akshare.get_global_news("2026-05-22", look_back_days=1, limit=5)

    assert "快速宏观来源" in result
    assert "China Macro and Policy News" in result


def test_global_news_total_budget_returns_available_sources(monkeypatch):
    calls = []

    class BudgetedMacroAkShare:
        def stock_info_global_em(self):
            calls.append("eastmoney")
            return pd.DataFrame(
                [
                    {
                        "标题": "预算内宏观来源",
                        "发布时间": "2026-05-22 09:00:00",
                        "来源": "东方财富",
                    }
                ]
            )

        def stock_info_global_sina(self):
            calls.append("sina")
            time.sleep(0.05)
            return pd.DataFrame()

        def stock_info_global_ths(self):
            calls.append("ths")
            return pd.DataFrame()

        def stock_info_global_cls(self, symbol="全部"):
            calls.append("cls")
            return pd.DataFrame()

        def news_cctv(self, date):
            calls.append(f"cctv:{date}")
            return pd.DataFrame()

    monkeypatch.setattr(akshare, "_ak", lambda: BudgetedMacroAkShare())
    monkeypatch.setattr(akshare, "_is_historical_macro_news_date", lambda end_date: False)
    monkeypatch.setattr(akshare, "_MACRO_NEWS_TOTAL_BUDGET_SECONDS", 0.02)
    monkeypatch.setattr(akshare, "_MACRO_NEWS_SOURCE_TIMEOUT_SECONDS", 0.05)

    result = akshare.get_global_news("2026-05-22", look_back_days=1, limit=5)

    assert "预算内宏观来源" in result
    assert "sina" in calls
    assert "ths" not in calls


def test_global_news_recent_timeout_skips_unhealthy_source(monkeypatch):
    calls = []

    class FlakyMacroAkShare:
        def stock_info_global_em(self):
            calls.append("eastmoney")
            time.sleep(0.05)
            return pd.DataFrame()

        def stock_info_global_sina(self):
            calls.append("sina")
            return pd.DataFrame(
                [
                    {
                        "标题": "健康宏观来源",
                        "发布时间": "2026-05-22 09:00:00",
                        "来源": "新浪财经",
                    }
                ]
            )

        def stock_info_global_ths(self):
            calls.append("ths")
            return pd.DataFrame()

        def stock_info_global_cls(self, symbol="全部"):
            calls.append("cls")
            return pd.DataFrame()

        def news_cctv(self, date):
            calls.append(f"cctv:{date}")
            return pd.DataFrame()

    monkeypatch.setattr(akshare, "_ak", lambda: FlakyMacroAkShare())
    monkeypatch.setattr(akshare, "_is_historical_macro_news_date", lambda end_date: False)
    monkeypatch.setattr(akshare, "_MACRO_NEWS_TOTAL_BUDGET_SECONDS", 1)
    monkeypatch.setattr(akshare, "_MACRO_NEWS_SOURCE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(akshare, "_MACRO_NEWS_SOURCE_COOLDOWN_SECONDS", 60)

    first = akshare.get_global_news("2026-05-22", look_back_days=1, limit=5)
    second = akshare.get_global_news("2026-05-22", look_back_days=1, limit=5)

    assert "健康宏观来源" in first
    assert "健康宏观来源" in second
    assert calls.count("eastmoney") == 1
    assert calls.count("sina") == 2


def test_load_index_ohlcv_uses_akshare_index_symbol(monkeypatch):
    class FakeIndexAkShare:
        def __init__(self):
            self.calls = []

        def stock_zh_index_daily_tx(self, **kwargs):
            self.calls.append(("stock_zh_index_daily_tx", kwargs))
            return _lowercase_ohlcv_frame()

        def stock_zh_index_daily(self, **kwargs):
            self.calls.append(("stock_zh_index_daily", kwargs))
            return pd.DataFrame()

        def index_zh_a_hist(self, **kwargs):
            self.calls.append(("index_zh_a_hist", kwargs))
            return pd.DataFrame()

    fake = FakeIndexAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    data = akshare.load_index_ohlcv("000001.SS", "2026-01-01", "2026-01-03")

    assert data["Close"].tolist() == [10.5, 11.5, 12.5]
    assert fake.calls[0] == (
        "stock_zh_index_daily_tx",
        {"symbol": "sh000001", "start_date": "20260101", "end_date": "20260103"},
    )


def test_route_to_vendor_can_select_akshare(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)
    set_config({"tool_vendors": {"get_stock_data": "akshare"}})

    result = route_to_vendor("get_stock_data", "600519", "2026-01-01", "2026-01-03")

    assert "AkShare data for 600519.SH" in result
    assert fake.calls[0][0] == "stock_zh_a_hist"


def test_route_to_vendor_requires_explicit_akshare_for_cn_a(monkeypatch):
    def fallback_stock(symbol, start_date, end_date):
        return f"fallback:{symbol}:{start_date}:{end_date}"

    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "yfinance", fallback_stock)

    with pytest.raises(RuntimeError, match="Supported vendor\\(s\\): akshare"):
        route_to_vendor("get_stock_data", "600519.SH", "2026-01-01", "2026-01-03")


def test_route_to_vendor_requires_explicit_akshare_for_cn_otc_fund(monkeypatch):
    def fallback_stock(symbol, start_date, end_date):
        return f"fallback:{symbol}:{start_date}:{end_date}"

    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "yfinance", fallback_stock)

    with pytest.raises(RuntimeError, match="cn_fund ticker '012920'. Supported vendor\\(s\\): akshare"):
        route_to_vendor("get_stock_data", "012920", "2026-01-01", "2026-01-03")


def test_route_to_vendor_does_not_fall_back_to_yfinance_for_cn_a(monkeypatch):
    def fail_stock(symbol, start_date, end_date):
        raise akshare.AkShareDataError("rate limited")

    def fallback_stock(symbol, start_date, end_date):
        return f"fallback:{symbol}:{start_date}:{end_date}"

    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "akshare", fail_stock)
    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "yfinance", fallback_stock)
    set_config({"tool_vendors": {"get_stock_data": "akshare,yfinance"}})

    with pytest.raises(RuntimeError, match="Last recoverable error: akshare: rate limited"):
        route_to_vendor("get_stock_data", "600519.SH", "2026-01-01", "2026-01-03")


def test_route_to_vendor_falls_back_on_akshare_data_error_for_us_ticker(monkeypatch):
    def fail_stock(symbol, start_date, end_date):
        raise akshare.AkShareDataError("rate limited")

    def fallback_stock(symbol, start_date, end_date):
        return f"fallback:{symbol}:{start_date}:{end_date}"

    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "akshare", fail_stock)
    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "yfinance", fallback_stock)
    set_config({"tool_vendors": {"get_stock_data": "akshare,yfinance"}})

    assert route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-03") == (
        "fallback:AAPL:2026-01-01:2026-01-03"
    )


def test_route_to_vendor_skips_akshare_for_us_ticker(monkeypatch):
    calls = []

    def akshare_stock(symbol, start_date, end_date):
        calls.append(("akshare", symbol))
        return "wrong-vendor"

    def yfinance_stock(symbol, start_date, end_date):
        calls.append(("yfinance", symbol))
        return f"yfinance:{symbol}:{start_date}:{end_date}"

    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "akshare", akshare_stock)
    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_stock_data"], "yfinance", yfinance_stock)
    set_config({"tool_vendors": {"get_stock_data": "akshare,yfinance"}})

    assert route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-03") == (
        "yfinance:AAPL:2026-01-01:2026-01-03"
    )
    assert calls == [("yfinance", "AAPL")]


def _ohlcv_frame():
    return pd.DataFrame(
        [
            {"日期": "2026-01-01", "开盘": 10, "最高": 11, "最低": 9, "收盘": 10.5, "成交量": 1000},
            {"日期": "2026-01-02", "开盘": 10.5, "最高": 12, "最低": 10, "收盘": 11.5, "成交量": 1200},
            {"日期": "2026-01-03", "开盘": 11.5, "最高": 13, "最低": 11, "收盘": 12.5, "成交量": 1300},
        ]
    )


def _nav_frame():
    return pd.DataFrame(
        [
            {"Date": "2026-01-01", "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 0},
            {"Date": "2026-01-02", "Open": 1.1, "High": 1.1, "Low": 1.1, "Close": 1.1, "Volume": 0},
            {"Date": "2026-01-03", "Open": 1.2, "High": 1.2, "Low": 1.2, "Close": 1.2, "Volume": 0},
        ]
    )


def _ohlcv_result(symbol="600519", source="test_loader"):
    normalized = akshare.normalize_ticker_symbol(symbol)
    return akshare.DataResult(
        meta=akshare.SourceMeta(
            vendor="akshare",
            source=source,
            symbol=normalized,
            semantic="ohlcv",
            as_of="2026-01-03",
            retrieved_at="2026-01-03 12:00:00",
        ),
        payload=_ohlcv_frame().rename(
            columns={
                "日期": "Date",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume",
            }
        ),
        ok=True,
        metadata={"start_date": "2026-01-01", "end_date": "2026-01-03"},
    )


def _lowercase_ohlcv_frame():
    return pd.DataFrame(
        [
            {"date": "2026-01-01", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
            {"date": "2026-01-02", "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 1200},
            {"date": "2026-01-03", "open": 11.5, "high": 13, "low": 11, "close": 12.5, "volume": 1300},
        ]
    )


def _statement_frame():
    return pd.DataFrame(
        [
            {"报告期": "2025-12-31", "资产总计": 100},
            {"报告期": "2026-06-30", "资产总计": 120},
        ]
    )


def test_akshare_disk_cache_behavior(tmp_path, monkeypatch):
    from tradingagents.dataflows import akshare

    # 1. 实例化一个临时隔离的 Cache，并替换全局全局对象
    temp_cache = Cache(str(tmp_path))
    dataflow_cache.set_disk_cache("akshare", temp_cache)

    # 2. Mock 真正的底层 _load_ohlcv 数据源
    load_count = 0
    def mock_load_ohlcv_result(symbol, start, end):
        nonlocal load_count
        load_count += 1
        return _ohlcv_result(symbol)
    monkeypatch.setattr(akshare, "_load_ohlcv_result", mock_load_ohlcv_result)

    # 3. 第一次调用：应该调用底层加载方法，并且回写磁盘
    res1 = akshare.get_stock("600519", "2026-01-01", "2026-01-03")
    assert load_count == 1

    # 4. 第二次调用：直接从缓存中获取，即使 _load_ohlcv 抛出异常也应该成功返回结果
    def fail_load_ohlcv_result(symbol, start, end):
        raise RuntimeError("Network API should not be called!")
    monkeypatch.setattr(akshare, "_load_ohlcv_result", fail_load_ohlcv_result)

    res2 = akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    # 5. 断言两次结果 100% 一致且缓存拦截成功
    assert res1 == res2
    assert "AkShare data for 600519.SH" in res2


def test_akshare_disk_cache_records_hit_miss_stats(tmp_path, monkeypatch):
    from tradingagents.dataflows import akshare

    dataflow_cache.reset_cache_stats("akshare")
    dataflow_cache.set_disk_cache("akshare", Cache(str(tmp_path / "stats_cache")))
    monkeypatch.setattr(akshare, "_load_ohlcv_result", lambda symbol, start, end: _ohlcv_result(symbol))

    akshare.get_stock("600519", "2026-01-01", "2026-01-03")
    akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    stats = dataflow_cache.get_cache_stats("akshare")["get_stock"]
    assert stats["misses"] == 1
    assert stats["writes"] == 1
    assert stats["hits"] == 1


def test_akshare_disk_cache_graceful_fallback(monkeypatch):
    from tradingagents.dataflows import akshare

    # 1. 模拟缓存不可用（如宕机或损坏为 None）
    dataflow_cache.set_disk_cache("akshare", None)

    # 2. Mock 正常的数据底层加载
    load_count = 0
    def mock_load_ohlcv_result(symbol, start, end):
        nonlocal load_count
        load_count += 1
        return _ohlcv_result(symbol)
    monkeypatch.setattr(akshare, "_load_ohlcv_result", mock_load_ohlcv_result)

    # 3. 两次调用应该两次都透传到底层（虽然无缓存，但核心业务不被阻断）
    res1 = akshare.get_stock("600519", "2026-01-01", "2026-01-03")
    res2 = akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    assert load_count == 2
    assert res1 == res2


def test_akshare_cache_lazy_loads_under_configured_cache_dir(monkeypatch, tmp_path):
    from tradingagents.dataflows import akshare

    created_paths = []

    class FakeCache:
        def __init__(self, path):
            created_paths.append(path)
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, expire=None):
            self.store[key] = value

    custom_cache_dir = tmp_path / "configured-cache"
    set_config({"data_cache_dir": str(custom_cache_dir)})
    dataflow_cache.clear_disk_cache("akshare")
    monkeypatch.setattr(dataflow_cache, "Cache", FakeCache)
    monkeypatch.setattr(akshare, "_load_ohlcv_result", lambda symbol, start, end: _ohlcv_result(symbol))

    result = akshare.get_stock("600519", "2026-01-01", "2026-01-03")

    assert "AkShare data for 600519.SH" in result
    assert created_paths == [f"{custom_cache_dir}/akshare"]


def test_bj_stock_routing_and_prefixes(monkeypatch):
    from tradingagents.dataflows import akshare
    from tradingagents.dataflows.instruments import (
        InstrumentType,
        MarketType,
        detect_instrument_type,
        detect_market_type,
        normalize_ticker_symbol,
        to_akshare_symbol,
    )

    # 1. 验证 830833 (北交所股票) 的识别与规范化
    assert normalize_ticker_symbol("830833") == "830833.BJ"
    assert detect_market_type("830833") == MarketType.CN_A
    assert detect_instrument_type("830833") == InstrumentType.EQUITY
    assert to_akshare_symbol("830833.BJ") == "830833"

    # 2. 验证财报接口前缀拼装
    assert akshare._to_akshare_statement_symbol("830833") == "BJ830833"

    # 3. 验证新浪备用数据源前缀拼装
    assert akshare._prefixed_cn_symbol("830833") == "bj830833"

    # 4. Mock 测试其能成功走向东财 A 股行情加载
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    akshare.get_stock("830833", "2026-01-01", "2026-01-03")
    assert fake.calls[0][0] == "stock_zh_a_hist"
    assert fake.calls[0][1]["symbol"] == "830833"


def test_akshare_get_display_name_cached(monkeypatch, tmp_path):
    from tradingagents.dataflows import akshare

    # 1. 实例化一个临时隔离的 Cache
    temp_cache = Cache(str(tmp_path / "test_display_cache"))
    dataflow_cache.set_disk_cache("akshare", temp_cache)

    # 2. 验证海外标的直接降级返回自身
    assert akshare.get_ticker_display_name("AAPL") == "AAPL"

    # 3. Mock 验证基金和股票简称拉取
    class FakeOverviewAkShare:
        def __init__(self):
            self.calls = []

        def fund_overview_em(self, symbol):
            self.calls.append(("fund_overview_em", symbol))
            if symbol == "510300":
                return pd.DataFrame(
                    [
                        {"项目": "基金简称", "内容": "沪深300ETF"},
                    ]
                )
            return pd.DataFrame()

        def stock_individual_info_em(self, symbol):
            self.calls.append(("stock_individual_info_em", symbol))
            if symbol == "600519":
                return pd.DataFrame(
                    [
                        {"item": "股票简称", "value": "贵州茅台"},
                    ]
                )
            return pd.DataFrame()

    fake = FakeOverviewAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    # 第一次调用：基金拉取
    assert akshare.get_ticker_display_name("510300") == "沪深300ETF"
    assert len(fake.calls) == 1
    assert fake.calls[0] == ("fund_overview_em", "510300")

    # 缓存测试：第二次调用应该直接从缓存返回，不需要再次调用 _ak
    assert akshare.get_ticker_display_name("510300") == "沪深300ETF"
    assert len(fake.calls) == 1

    # 第一次调用：股票拉取
    assert akshare.get_ticker_display_name("600519") == "贵州茅台"
    assert len(fake.calls) == 2
    assert fake.calls[1] == ("stock_individual_info_em", "600519")

    # 缓存测试：第二次调用股票应该直接从缓存返回
    assert akshare.get_ticker_display_name("600519") == "贵州茅台"
    assert len(fake.calls) == 2


def test_get_insider_transactions_returns_real_rows(monkeypatch):
    class FakeInsiderAkShare:
        def stock_hold_management_detail_em(self, symbol):
            return pd.DataFrame(
                [
                    {"变动人姓名": "张三", "职务": "董事", "变动日期": "2026-01-05",
                     "变动数量": 10000, "变动后持股数": 50000},
                ]
            )

    fake = FakeInsiderAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_insider_transactions("600519")

    assert "AkShare Insider Transactions" in result
    assert "张三" in result
    assert "not available in a stable MVP format" not in result


def test_get_insider_transactions_no_rows_is_explicit_no_data(monkeypatch):
    class FakeEmptyInsiderAkShare:
        def stock_hold_management_detail_em(self, symbol):
            return pd.DataFrame()

    fake = FakeEmptyInsiderAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_insider_transactions("600519")

    assert "No AkShare insider transaction rows" in result
    assert "Do not fabricate" in result


def test_get_insider_transactions_not_applicable_for_fund():
    result = akshare.get_insider_transactions("510300")
    assert "not applicable" in result


def test_ticker_display_name_force_refresh_bypasses_cache(monkeypatch, tmp_path):
    from tradingagents.dataflows import akshare

    temp_cache = Cache(str(tmp_path / "test_display_force_cache"))
    dataflow_cache.set_disk_cache("akshare", temp_cache)

    fetch_count = {"n": 0}

    class FakeOverviewAkShare:
        def fund_overview_em(self, symbol):
            fetch_count["n"] += 1
            return pd.DataFrame([{"项目": "基金简称", "内容": f"简称-{fetch_count['n']}"}])

    fake = FakeOverviewAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    # 首次调用抓取并缓存；第二次命中缓存不再抓取。
    assert akshare.get_ticker_display_name("510300") == "简称-1"
    assert akshare.get_ticker_display_name("510300") == "简称-1"
    assert fetch_count["n"] == 1

    # force_refresh 绕过磁盘缓存重新抓取。
    assert akshare.get_ticker_display_name("510300", force_refresh=True) == "简称-2"
    assert fetch_count["n"] == 2


@pytest.mark.parametrize("ticker", ["510300", "012920"])
def test_window_fallback_is_warning_and_renders_warn(monkeypatch, tmp_path, ticker):
    """R3c — window-fallback fund announcements surface as WARN in the Data
    Reliability render (not a silent info-level notice), for both listed-fund
    and OTC-fund announcement paths."""
    from tradingagents.dataflows.contracts import (
        build_data_contract_status,
        parse_contract_gate_status,
        render_data_contract_status,
    )

    temp_cache = Cache(str(tmp_path / "test_window_fallback_cache"))
    dataflow_cache.set_disk_cache("akshare", temp_cache)

    class FakeFallbackAkShare:
        def fund_announcement_report_em(self, symbol):
            # 只有窗口外的公告（2025-12-31，早于 start_date 2026-01-01）
            return pd.DataFrame(
                [{"公告标题": "窗口外公告", "公告日期": "2025-12-31", "公告链接": "https://example.invalid/x"}]
            )

        def fund_announcement_dividend_em(self, symbol):
            return pd.DataFrame()

        def fund_announcement_personnel_em(self, symbol):
            return pd.DataFrame()

    fake = FakeFallbackAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    result = akshare.get_news(ticker, "2026-01-01", "2026-01-03")

    assert "none found from 2026-01-01 to 2026-01-03" in result
    assert "WARNING window_fallback" in result

    checks = parse_contract_gate_status(result)
    assert checks and "window_fallback" in checks[0]["warnings"]
    rendered = render_data_contract_status(build_data_contract_status(checks))
    assert "WARN" in rendered
    assert "window_fallback" in rendered


def test_mini_racer_prewarm_fails_open_when_unavailable():
    """Neither mini_racer nor py_mini_racer installed → prewarm is a no-op."""
    import sys
    from unittest import mock

    from tradingagents.dataflows import akshare

    with mock.patch.dict(sys.modules, {"mini_racer": None, "py_mini_racer": None}):
        akshare._prewarm_mini_racer()  # must not raise


def test_mini_racer_prewarm_uses_mini_racer_first():
    import sys
    from unittest import mock

    from tradingagents.dataflows import akshare

    class FakeMiniRacer:
        instances = []

        def __init__(self):
            self.evals = []
            FakeMiniRacer.instances.append(self)

        def eval(self, code):
            self.evals.append(code)

    fake_module = mock.Mock()
    fake_module.MiniRacer = FakeMiniRacer
    with mock.patch.dict(sys.modules, {"mini_racer": fake_module, "py_mini_racer": None}):
        akshare._prewarm_mini_racer()

    assert FakeMiniRacer.instances, "mini_racer should be instantiated"
    assert FakeMiniRacer.instances[0].evals == ["1+1"]


def test_mini_racer_prewarm_falls_back_to_py_mini_racer():
    import sys
    from unittest import mock

    from tradingagents.dataflows import akshare

    class FakeMiniRacer:
        instances = []

        def __init__(self):
            self.evals = []
            FakeMiniRacer.instances.append(self)

        def eval(self, code):
            self.evals.append(code)

    fake_module = mock.Mock()
    fake_module.MiniRacer = FakeMiniRacer
    with mock.patch.dict(sys.modules, {"mini_racer": None, "py_mini_racer": fake_module}):
        akshare._prewarm_mini_racer()

    assert FakeMiniRacer.instances, "py_mini_racer should be instantiated as fallback"
    assert FakeMiniRacer.instances[0].evals == ["1+1"]


def test_mini_racer_prewarm_survives_mini_racer_constructor_failure():
    """If the primary library is importable but its engine fails to start, the
    fallback (and ultimately fail-open) paths must still be safe."""
    import sys
    from unittest import mock

    from tradingagents.dataflows import akshare

    def boom():
        raise RuntimeError("engine failed to start")

    broken_module = mock.Mock()
    broken_module.MiniRacer = mock.Mock(side_effect=boom)
    with mock.patch.dict(sys.modules, {"mini_racer": broken_module, "py_mini_racer": None}):
        akshare._prewarm_mini_racer()  # must not raise


def test_akshare_disk_cache_disabled_by_config(tmp_path, monkeypatch):
    from tradingagents.dataflows import akshare
    from tradingagents.dataflows.config import set_config

    # 1. 实例化一个临时隔离的 Cache 并绑定
    temp_cache = Cache(str(tmp_path / "test_disable_cache"))
    dataflow_cache.set_disk_cache("akshare", temp_cache)

    # 2. Mock 底层 _load_ohlcv
    load_count = 0
    def mock_load_ohlcv_result(symbol, start, end):
        nonlocal load_count
        load_count += 1
        return _ohlcv_result(symbol)
    monkeypatch.setattr(akshare, "_load_ohlcv_result", mock_load_ohlcv_result)

    # 3. 开启缓存（默认）：调用一次，填充缓存
    set_config({"enable_data_cache": True})
    akshare.get_stock("600519", "2026-01-01", "2026-01-03")
    assert load_count == 1

    # 4. 关闭缓存：调用第二次，由于禁用了缓存，即使缓存有值，也应当穿透调用底层
    set_config({"enable_data_cache": False})
    akshare.get_stock("600519", "2026-01-01", "2026-01-03")
    assert load_count == 2

    stats = dataflow_cache.get_cache_stats("akshare")["get_stock"]
    assert stats["disabled"] == 1
