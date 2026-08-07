from __future__ import annotations

import copy

from langchain_core.messages import ToolMessage

import cli.main as cli_main
from tradingagents.agents.utils.agent_utils import create_msg_delete
from tradingagents.default_config import DEFAULT_CONFIG


def _status():
    return {
        "overall": "warning",
        "checks": [
            {
                "status": "pass",
                "source": "tiantian_fund_nav",
                "symbol": "012920",
                "semantic": "nav",
                "expected_semantic": "nav",
                "as_of": "2026-05-22",
                "rows": 3,
                "failures": [],
                "warnings": ["nav_semantic"],
            }
        ],
    }


def test_message_buffer_includes_data_reliability_in_current_and_final_report():
    buffer = cli_main.MessageBuffer()
    buffer.init_for_analysis(["market"])

    buffer.update_report_section("market_report", "Market report body.")
    buffer.update_data_contract_status(_status())

    assert "### Data Reliability" in buffer.current_report
    assert "nav_semantic" in buffer.current_report
    assert "## Data Reliability" in buffer.final_report
    assert "Market report body." in buffer.final_report


def test_save_report_to_disk_writes_data_reliability(tmp_path):
    final_state = {
        "data_contract_status": _status(),
        "market_report": "Market report body.",
    }

    report_path = cli_main.save_report_to_disk(final_state, "012920", tmp_path)

    data_reliability = tmp_path / "data_reliability.md"
    assert data_reliability.exists()
    assert "## Data Reliability" in data_reliability.read_text(encoding="utf-8")
    assert "nav_semantic" in report_path.read_text(encoding="utf-8")


def test_run_analysis_stream_saves_data_reliability_from_contract_gate(monkeypatch, tmp_path):
    save_path = tmp_path / "saved-report"
    gate_text = """## AkShare Data Contract Gate

- Status: PASS
- Source: tiantian_fund_nav
- Symbol: 012920
- Semantic: nav
- Expected semantic: nav
- As of: 2026-05-22
- Rows: 3

### Warnings
- WARNING nav_semantic: source=tiantian_fund_nav; This result is daily fund NAV.
"""

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
            return {
                "messages": [],
                "company_of_interest": ticker,
                "trade_date": analysis_date,
                "data_contract_status": {"overall": "not_checked", "checks": []},
            }

        def get_graph_args(self, callbacks=None):
            return {}

    class FakeCompiledGraph:
        def stream(self, init_agent_state, **args):
            tool_message = ToolMessage(
                content=gate_text,
                tool_call_id="call_1",
                id="tool_1",
            )
            yield {"messages": [tool_message]}
            yield create_msg_delete()(
                {
                    **init_agent_state,
                    "messages": [tool_message],
                }
            )
            yield {
                "market_report": "Market report body.",
                "final_trade_decision": "HOLD",
            }

    class FakeTradingAgentsGraph:
        def __init__(self, selected_analysts, *, config, debug, callbacks):
            self.propagator = FakePropagator()
            self.graph = FakeCompiledGraph()

        def resolve_instrument_context(self, ticker, asset_type):
            return "instrument context"

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

    monkeypatch.setattr(
        cli_main,
        "get_user_selections",
        lambda: {
            "ticker": "012920",
            "asset_type": "stock",
            "instrument_type": "fund",
            "market_type": "cn_fund",
            "analysis_date": "2026-05-22",
            "analysts": [cli_main.AnalystType.MARKET],
            "research_depth": 1,
            "llm_provider": "openai",
            "backend_url": "https://api.openai.com/v1",
            "shallow_thinker": DEFAULT_CONFIG["quick_think_llm"],
            "deep_thinker": DEFAULT_CONFIG["deep_think_llm"],
            "google_thinking_level": None,
            "openai_reasoning_effort": None,
            "anthropic_effort": None,
            "output_language": "Chinese",
        },
    )
    monkeypatch.setattr(cli_main, "TradingAgentsGraph", FakeTradingAgentsGraph)
    monkeypatch.setattr(cli_main, "Live", NoopLive)
    monkeypatch.setattr(cli_main, "update_display", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main.typer, "prompt", lambda *args, **kwargs: str(save_path))

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["results_dir"] = str(tmp_path / "results")
    monkeypatch.setattr(cli_main, "DEFAULT_CONFIG", config)

    cli_main.run_analysis(save_report=True, display_report=False)

    complete_report = (save_path / "complete_report.md").read_text(encoding="utf-8")
    data_reliability = (save_path / "data_reliability.md").read_text(encoding="utf-8")
    assert "## Data Reliability" in complete_report
    assert "nav_semantic" in complete_report
    assert "tiantian_fund_nav" in data_reliability
    assert "Market report body." in complete_report
