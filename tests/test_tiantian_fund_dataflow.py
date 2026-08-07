import pandas as pd
import pytest
import requests
from diskcache import Cache

import tradingagents.default_config as default_config
from tradingagents.dataflows import akshare, cache as dataflow_cache, tiantian_fund
from tradingagents.dataflows.config import set_config


@pytest.fixture(autouse=True)
def isolate_dataflow_cache(tmp_path):
    set_config(default_config.DEFAULT_CONFIG.copy())
    dataflow_cache.reset_cache_stats()
    dataflow_cache.set_disk_cache("akshare", Cache(str(tmp_path / "akshare_cache")))
    dataflow_cache.set_disk_cache("tiantian_fund", Cache(str(tmp_path / "tiantian_cache")))
    yield
    dataflow_cache.clear_disk_cache("akshare")
    dataflow_cache.clear_disk_cache("tiantian_fund")
    dataflow_cache.reset_cache_stats()
    set_config(default_config.DEFAULT_CONFIG.copy())


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_tiantian_fund_profile_tables_parse_detail_and_holdings(monkeypatch):
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params))
        if "pingzhongdata" in url:
            return FakeResponse(_detail_script())
        return FakeResponse(_holdings_response("2026-03-31"))

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    tables = tiantian_fund.get_fund_profile_tables("510300.SH", "2026-05-22", holdings_limit=2)

    by_title = {table.title: table.data for table in tables}
    assert by_title["Tiantian Fund Overview"].loc[0, "内容"] == "510300"
    assert "沪深300ETF华泰柏瑞" in by_title["Tiantian Fund Overview"]["内容"].tolist()
    assert by_title["Tiantian Fund Returns"]["指标"].tolist() == [
        "近1月收益率(%)",
        "近3月收益率(%)",
        "近6月收益率(%)",
        "近1年收益率(%)",
    ]
    assert by_title["Tiantian Fund Latest NAV"]["日期"].tolist() == ["2026-05-22", "2026-05-21", "2025-12-31"]
    assert by_title["Tiantian Fund Scale"]["日期"].tolist() == ["2026-03-31", "2025-12-31"]
    assert by_title["Tiantian Fund Asset Allocation"]["股票占净比"].tolist() == [97.76, 98.65]
    assert by_title["Tiantian Fund Holder Structure"]["机构持有比例"].tolist() == [90.27, 87.1]
    assert by_title["Tiantian Fund Managers"]["基金经理"].tolist() == ["柳军"]
    assert by_title["Tiantian Fund Top Holdings"]["股票名称"].tolist() == ["宁德时代", "贵州茅台"]
    assert by_title["Tiantian Fund Top Holdings"]["截止日期"].tolist() == ["2026-03-31", "2026-03-31"]
    assert calls[0][0] == "https://fund.eastmoney.com/pingzhongdata/510300.js"
    assert calls[1][1]["type"] == "jjcc"


def test_tiantian_fund_profile_tables_exclude_future_rows(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        if "pingzhongdata" in url:
            return FakeResponse(_detail_script())
        return FakeResponse(_holdings_response("2026-03-31"))

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    tables = tiantian_fund.get_fund_profile_tables("510300", "2025-12-31", holdings_limit=2)

    by_title = {table.title: table.data for table in tables}
    assert by_title["Tiantian Fund Latest NAV"]["日期"].tolist() == ["2025-12-31"]
    assert by_title["Tiantian Fund Scale"]["日期"].tolist() == ["2025-12-31"]
    assert by_title["Tiantian Fund Asset Allocation"]["日期"].tolist() == ["2025-12-31"]
    assert by_title["Tiantian Fund Holder Structure"]["日期"].tolist() == ["2025-12-31"]
    assert "Tiantian Fund Managers" not in by_title
    assert "Tiantian Fund Top Holdings" not in by_title


def test_tiantian_fund_profile_keeps_detail_when_holdings_endpoint_fails(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        if "pingzhongdata" in url:
            return FakeResponse(_detail_script())
        raise requests.Timeout("holdings timeout")

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    tables = tiantian_fund.get_fund_profile_tables("510300", "2026-05-22", holdings_limit=2)

    by_title = {table.title: table.data for table in tables}
    assert "Tiantian Fund Overview" in by_title
    assert "Tiantian Fund Latest NAV" in by_title
    assert "Tiantian Fund Top Holdings" not in by_title


def test_tiantian_fund_nav_history_returns_ohlcv_shape(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        return FakeResponse(_detail_script())

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    data = tiantian_fund.get_fund_nav_history("012920", "2026-05-21", "2026-05-22")

    assert data["Date"].tolist() == ["2026-05-21", "2026-05-22"]
    assert data["Close"].tolist() == [3.22, 3.24]
    assert data["Open"].tolist() == [3.22, 3.24]
    assert data["Volume"].tolist() == [0, 0]
    assert data["Pct Change"].tolist() == [0.2, 0.3]


def test_tiantian_fund_nav_history_result_marks_nav_semantic(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        return FakeResponse(_detail_script())

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    result = tiantian_fund.get_fund_nav_history_result("012920", "2026-05-21", "2026-05-22")

    assert result.ok is True
    assert result.meta.vendor == "tiantian_fund"
    assert result.meta.semantic == "nav"
    assert result.meta.as_of == "2026-05-22"
    assert result.rows == 2


def test_tiantian_fund_profile_rejects_empty_detail_contract(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        if "pingzhongdata" in url:
            return FakeResponse("var unrelated_shape = {};")
        return FakeResponse("var apidata={content:'',arryear:[2026]};")

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    with pytest.raises(tiantian_fund.TiantianFundDataError, match="No Tiantian Fund profile data returned"):
        tiantian_fund.get_fund_profile_tables("510300", "2026-05-22", holdings_limit=2)


def test_tiantian_fund_profile_result_reports_js_schema_drift(monkeypatch):
    """R7 — a page missing the required JS variables reports js_schema_drift,
    not a generic source_error (site redesign is distinguishable from a data gap)."""
    def fake_get(url, params=None, **kwargs):
        if "pingzhongdata" in url:
            return FakeResponse("var unrelated_shape = {};")
        return FakeResponse("var apidata={content:'',arryear:[2026]};")

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    result = tiantian_fund.get_fund_profile_tables_result("510300", "2026-05-22", holdings_limit=2)

    assert result.ok is False
    assert result.meta.semantic == "fund_profile"
    assert result.error_type == "js_schema_drift"
    assert result.missing_reason == "no_profile_data"
    assert any(notice.code == "js_schema_drift" for notice in result.notices)


def test_tiantian_fund_nav_result_reports_js_schema_drift(monkeypatch):
    """R7 — NAV lookup on a structurally drifted page carries js_schema_drift."""
    def fake_get(url, params=None, **kwargs):
        return FakeResponse('var fS_name = "仅剩一个必需变量";')

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    result = tiantian_fund.get_fund_nav_history_result("012920", "2026-05-21", "2026-05-22")

    assert result.ok is False
    assert result.error_type == "js_schema_drift"
    assert result.missing_reason == "no_nav_data"
    assert any(notice.code == "js_schema_drift" for notice in result.notices)


def test_tiantian_fund_nav_result_intact_page_has_no_drift_notice(monkeypatch):
    """R7 — an intact page succeeds and carries no js_schema_drift notice."""
    def fake_get(url, params=None, **kwargs):
        return FakeResponse(_detail_script())

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    result = tiantian_fund.get_fund_nav_history_result("012920", "2026-05-21", "2026-05-22")

    assert result.ok is True
    assert result.rows == 2
    assert not any(notice.code == "js_schema_drift" for notice in result.notices)


def test_page_structure_drift_heuristic():
    assert tiantian_fund._page_structure_drifted(_detail_script()) is False
    assert tiantian_fund._page_structure_drifted('var fS_name = "x";') is True
    assert tiantian_fund._page_structure_drifted("") is True
    assert tiantian_fund._page_structure_drifted("   ") is True


def test_tiantian_fund_profile_ignores_drifted_holdings_shape(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        if "pingzhongdata" in url:
            return FakeResponse(_detail_script())
        return FakeResponse(_drifted_holdings_response("2026-03-31"))

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    tables = tiantian_fund.get_fund_profile_tables("510300", "2026-05-22", holdings_limit=2)

    by_title = {table.title: table.data for table in tables}
    assert "Tiantian Fund Overview" in by_title
    assert "Tiantian Fund Top Holdings" not in by_title


def test_tiantian_fund_nav_history_rejects_missing_nav_trend(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        return FakeResponse('var fS_name = "沪深300ETF华泰柏瑞"; var fS_code = "510300";')

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    with pytest.raises(tiantian_fund.TiantianFundDataError, match="No Tiantian Fund NAV data returned"):
        tiantian_fund.get_fund_nav_history("510300", "2026-05-21", "2026-05-22")


def test_tiantian_fund_profile_tables_use_disk_cache(monkeypatch):
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params))
        if "pingzhongdata" in url:
            return FakeResponse(_detail_script())
        return FakeResponse(_holdings_response("2026-03-31"))

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    first = tiantian_fund.get_fund_profile_tables("510300", "2026-05-22", holdings_limit=2)
    assert len(calls) == 2

    def fail_get(url, params=None, **kwargs):
        raise AssertionError("network should not be called after cache is warm")

    monkeypatch.setattr(tiantian_fund.requests, "get", fail_get)

    second = tiantian_fund.get_fund_profile_tables("510300", "2026-05-22", holdings_limit=2)

    assert [table.title for table in second] == [table.title for table in first]
    assert second[0].data.loc[0, "内容"] == "510300"
    assert len(calls) == 2
    stats = dataflow_cache.get_cache_stats("tiantian_fund")["get_fund_profile_tables"]
    assert stats["misses"] == 1
    assert stats["writes"] == 1
    assert stats["hits"] == 1


def test_tiantian_fund_nav_history_uses_disk_cache(monkeypatch):
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params))
        return FakeResponse(_detail_script())

    monkeypatch.setattr(tiantian_fund.requests, "get", fake_get)

    first = tiantian_fund.get_fund_nav_history("012920", "2026-05-21", "2026-05-22")
    assert len(calls) == 1

    def fail_get(url, params=None, **kwargs):
        raise AssertionError("network should not be called after cache is warm")

    monkeypatch.setattr(tiantian_fund.requests, "get", fail_get)

    second = tiantian_fund.get_fund_nav_history("012920", "2026-05-21", "2026-05-22")

    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1
    stats = dataflow_cache.get_cache_stats("tiantian_fund")["get_fund_nav_history"]
    assert stats["misses"] == 1
    assert stats["writes"] == 1
    assert stats["hits"] == 1


def test_akshare_fund_profile_includes_tiantian_enrichment(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)
    monkeypatch.setattr(
        akshare,
        "get_fund_profile_tables",
        lambda symbol, curr_date: [
            tiantian_fund.TiantianTable(
                "Tiantian Fund Overview",
                pd.DataFrame(
                    [{"项目": "基金简称", "内容": "沪深300ETF华泰柏瑞", "来源": "天天基金/东方财富"}]
                ),
            )
        ],
    )

    result = akshare.get_fundamentals("510300", "2026-05-22")

    assert "Tiantian Fund Overview" in result
    assert "沪深300ETF华泰柏瑞" in result
    assert "AkShare Fund Overview" in result
    assert "AkShare Fund Fees" in result
    assert "AkShare Fund Top Holdings" in result
    assert ("fund_overview_em", {"symbol": "510300"}) in fake.calls


def test_akshare_fund_profile_falls_back_when_tiantian_fails(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setattr(akshare, "_ak", lambda: fake)

    def fail_tiantian(symbol, curr_date):
        raise RuntimeError("unexpected parser shape")

    monkeypatch.setattr(akshare, "get_fund_profile_tables", fail_tiantian)

    result = akshare.get_fundamentals("510300", "2026-05-22")

    assert "Tiantian Fund Overview" not in result
    assert "AkShare Fund Overview" in result
    assert "沪深300ETF" in result
    assert "管理费率" in result
    assert "贵州茅台" in result


class FakeAkShare:
    def __init__(self):
        self.calls = []

    def fund_overview_em(self, **kwargs):
        self.calls.append(("fund_overview_em", kwargs))
        return pd.DataFrame(
            [
                {"项目": "基金简称", "内容": "沪深300ETF"},
                {"项目": "基金类型", "内容": "ETF"},
            ]
        )

    def fund_fee_em(self, **kwargs):
        self.calls.append(("fund_fee_em", kwargs))
        return pd.DataFrame([{"费用类型": "管理费率", "费率": "0.50%"}])

    def fund_portfolio_hold_em(self, **kwargs):
        self.calls.append(("fund_portfolio_hold_em", kwargs))
        return pd.DataFrame([{"股票名称": "贵州茅台", "占净值比例": "5.00%"}])


def _detail_script():
    return """
var fS_name = "沪深300ETF华泰柏瑞";
var fS_code = "510300";
var fund_sourceRate="1.20%";
var fund_Rate="0.12%";
var fund_minsg="10元";
var syl_1n="31.31";
var syl_6y="8.56";
var syl_3y="4.2";
var syl_1y="2.38";
var Data_netWorthTrend = [
  {"x":1767139200000,"y":3.12,"equityReturn":0.1,"unitMoney":""},
  {"x":1779321600000,"y":3.22,"equityReturn":0.2,"unitMoney":""},
  {"x":1779408000000,"y":3.24,"equityReturn":0.3,"unitMoney":""},
  {"x":1779494400000,"y":3.26,"equityReturn":0.4,"unitMoney":""}
];
var Data_fluctuationScale = {
  "categories":["2025-12-31","2026-03-31"],
  "series":[{"y":4222.58,"mom":"-0.78%"},{"y":1999.14,"mom":"-52.66%"}]
};
var Data_assetAllocation = {
  "series":[
    {"name":"股票占净比","type":null,"data":[98.65,97.76]},
    {"name":"债券占净比","type":null,"data":[0,0]},
    {"name":"现金占净比","type":null,"data":[1.25,2.03]}
  ],
  "categories":["2025-12-31","2026-03-31"]
};
var Data_holderStructure = {
  "series":[
    {"name":"机构持有比例","data":[87.1,90.27]},
    {"name":"个人持有比例","data":[12.2,8.98]}
  ],
  "categories":["2025-12-31","2026-03-31"]
};
var Data_currentFundManager = [
  {
    "name":"柳军",
    "star":5,
    "workTime":"17年",
    "fundSize":"3388.01亿",
    "power":{"jzrq":"2026-05-22"}
  }
];
"""


def _holdings_response(cutoff_date: str):
    content = (
        "<table><tbody>"
        "<tr><td>1</td><td>300750</td><td>宁德时代</td><td></td><td></td><td></td>"
        "<td>4.27%</td><td>2,126.97</td><td>854,404.57</td></tr>"
        "<tr><td>2</td><td>600519</td><td>贵州茅台</td><td></td><td></td><td></td>"
        "<td>3.65%</td><td>503.85</td><td>730,579.89</td></tr>"
        f"</tbody></table><label>来源：天天基金 截止至：<font>{cutoff_date}</font></label>"
    )
    return f'var apidata={{ content:"{content}",arryear:[2026]}};'


def _drifted_holdings_response(cutoff_date: str):
    content = (
        "<table><tbody>"
        "<tr><td>300750</td><td>宁德时代</td><td>4.27%</td></tr>"
        f"</tbody></table><label>来源：天天基金 截止至：<font>{cutoff_date}</font></label>"
    )
    return f'var apidata={{ content:"{content}",arryear:[2026]}};'
