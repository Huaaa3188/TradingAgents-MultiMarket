import copy

import pytest
from typer.testing import CliRunner

import cli.main as cli_main
from tradingagents.default_config import DEFAULT_CONFIG


def test_apply_data_vendor_override_updates_all_data_categories():
    config = copy.deepcopy(DEFAULT_CONFIG)

    cli_main._apply_data_vendor_override(config, " akshare, yfinance ")

    expected_data_vendors = copy.deepcopy(DEFAULT_CONFIG["data_vendors"])
    expected_data_vendors.update({
        "core_stock_apis": "akshare,yfinance",
        "technical_indicators": "akshare,yfinance",
        "fundamental_data": "akshare,yfinance",
        "news_data": "akshare,yfinance",
    })
    assert config["data_vendors"] == expected_data_vendors


def test_apply_data_vendor_override_rejects_empty_value():
    config = copy.deepcopy(DEFAULT_CONFIG)

    with pytest.raises(ValueError, match="No valid data vendor"):
        cli_main._apply_data_vendor_override(config, " , ")


def test_apply_data_vendor_override_rejects_unknown_vendor():
    config = copy.deepcopy(DEFAULT_CONFIG)

    with pytest.raises(ValueError, match="Unknown data vendor\\(s\\): akshre"):
        cli_main._apply_data_vendor_override(config, "akshre,yfinance")


def test_resolve_data_vendor_override_defaults_cn_a_to_akshare():
    selections = {"market_type": "cn_a"}

    assert cli_main._resolve_data_vendor_override(selections) == "akshare"


def test_resolve_data_vendor_override_defaults_cn_fund_to_akshare():
    selections = {"market_type": "cn_fund"}

    assert cli_main._resolve_data_vendor_override(selections) == "akshare"


def test_resolve_data_vendor_override_keeps_explicit_value():
    selections = {"market_type": "cn_a"}

    assert cli_main._resolve_data_vendor_override(selections, "akshare,yfinance") == "akshare,yfinance"


def test_resolve_data_vendor_override_leaves_non_cn_a_default_unset():
    selections = {"market_type": "us"}

    assert cli_main._resolve_data_vendor_override(selections) is None


def test_analyze_command_passes_data_vendor_override(monkeypatch):
    captured = {}

    def fake_run_analysis(
        *,
        checkpoint=False,
        data_vendors=None,
        save_report=None,
        display_report=None,
    ):
        captured["checkpoint"] = checkpoint
        captured["data_vendors"] = data_vendors
        captured["save_report"] = save_report
        captured["display_report"] = display_report

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = CliRunner().invoke(
        cli_main.app,
        [
            "--data-vendors",
            "akshare,yfinance",
            "--no-save-report",
            "--no-display-report",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "checkpoint": False,
        "data_vendors": "akshare,yfinance",
        "save_report": False,
        "display_report": False,
    }


def test_analyze_command_keeps_report_prompt_flags_unset_by_default(monkeypatch):
    captured = {}

    def fake_run_analysis(
        *,
        checkpoint=False,
        data_vendors=None,
        save_report=None,
        display_report=None,
    ):
        captured["save_report"] = save_report
        captured["display_report"] = display_report

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = CliRunner().invoke(cli_main.app, [])

    assert result.exit_code == 0
    assert captured == {"save_report": None, "display_report": None}


@pytest.mark.parametrize(
    ("ticker", "expected_market_type"),
    [
        ("510300", "cn_a"),
        ("012920", "cn_fund"),
    ],
)
def test_analyze_command_defaults_china_fund_tickers_to_akshare_without_mutating_defaults(
    monkeypatch,
    tmp_path,
    ticker,
    expected_market_type,
):
    captured = {}
    default_data_vendors = copy.deepcopy(DEFAULT_CONFIG["data_vendors"])

    class FakePropagator:
        def create_initial_state(
            self,
            ticker,
            analysis_date,
            *,
            asset_type,
            instrument_type,
            market_type,
            instrument_context,
        ):
            captured["initial_state"] = {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "asset_type": asset_type,
                "instrument_type": instrument_type,
                "market_type": market_type,
                "instrument_context": instrument_context,
            }
            return {"messages": []}

        def get_graph_args(self, callbacks=None):
            return {}

    class FakeCompiledGraph:
        def stream(self, init_agent_state, **args):
            yield {
                "market_report": "market report",
                "final_trade_decision": "HOLD",
            }

    class FakeTradingAgentsGraph:
        def __init__(self, selected_analysts, *, config, debug, callbacks):
            captured["config"] = copy.deepcopy(config)
            self.propagator = FakePropagator()
            self.graph = FakeCompiledGraph()

        def resolve_instrument_context(self, ticker, asset_type):
            return "instrument context"

        def process_signal(self, final_trade_decision):
            return "HOLD"

        def enter_checkpoint_stream(self, ticker, analysis_date, *, asset_type):
            return {}

        def exit_checkpoint_stream(self):
            return None

    class NoopLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cli_main, "fetch_announcements", lambda: None)
    monkeypatch.setattr(cli_main, "display_announcements", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main, "get_ticker", lambda: ticker)
    monkeypatch.setattr(cli_main, "get_analysis_date", lambda: "2026-05-29")
    monkeypatch.setattr(cli_main, "ask_output_language", lambda: "Chinese")
    monkeypatch.setattr(
        cli_main,
        "select_analysts",
        lambda asset_type: [cli_main.AnalystType.MARKET],
    )
    monkeypatch.setattr(cli_main, "select_research_depth", lambda: 1)
    monkeypatch.setattr(
        cli_main,
        "select_llm_provider",
        lambda: ("openai", "https://api.openai.com/v1"),
    )
    monkeypatch.setattr(cli_main, "ensure_api_key", lambda provider: None)
    monkeypatch.setattr(
        cli_main,
        "select_shallow_thinking_agent",
        lambda provider: DEFAULT_CONFIG["quick_think_llm"],
    )
    monkeypatch.setattr(
        cli_main,
        "select_deep_thinking_agent",
        lambda provider: DEFAULT_CONFIG["deep_think_llm"],
    )
    monkeypatch.setattr(cli_main, "ask_openai_reasoning_effort", lambda: None)
    monkeypatch.setattr(cli_main, "TradingAgentsGraph", FakeTradingAgentsGraph)
    monkeypatch.setattr(cli_main, "Live", NoopLive)
    monkeypatch.setattr(cli_main, "update_display", lambda *args, **kwargs: None)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["results_dir"] = str(tmp_path)
    monkeypatch.setattr(cli_main, "DEFAULT_CONFIG", config)

    result = CliRunner().invoke(
        cli_main.app,
        ["--no-save-report", "--no-display-report"],
    )

    assert result.exit_code == 0, result.output
    assert captured["initial_state"] == {
        "ticker": ticker,
        "analysis_date": "2026-05-29",
        "asset_type": "stock",
        "instrument_type": "fund",
        "market_type": expected_market_type,
        "instrument_context": "instrument context",
    }
    expected_runtime_data_vendors = copy.deepcopy(default_data_vendors)
    expected_runtime_data_vendors.update({
        "core_stock_apis": "akshare",
        "technical_indicators": "akshare",
        "fundamental_data": "akshare",
        "news_data": "akshare",
    })
    assert captured["config"]["data_vendors"] == expected_runtime_data_vendors
    assert config["data_vendors"] == default_data_vendors
    assert DEFAULT_CONFIG["data_vendors"] == default_data_vendors
