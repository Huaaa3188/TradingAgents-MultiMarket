"""Tests for deterministic instrument-identity resolution (#814) and the
context-anchored message placeholder (#888)."""

import unittest
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    create_msg_delete,
    get_instrument_context_from_state,
    resolve_instrument_identity,
)
from tradingagents.dataflows.instruments import (
    InstrumentType,
    MarketType,
    detect_instrument_type,
    detect_market_type,
    normalize_ticker_symbol,
)


@pytest.mark.unit
class ChinaCodeClassificationTests(unittest.TestCase):
    """R1 — six-digit China code classification is segment-accurate.

    Exchange-listed funds (``5``/``159``/``16``/``18`` prefixes) and equities
    (precise SH/SZ/BJ segments) get their exchange suffix; every remaining
    six-digit code is an OTC-fund candidate kept bare. Regression cases:
    ``005827`` (starts with ``0`` but outside the SZ equity segments) and
    ``110011`` (outside every listed segment) were previously misclassified as
    ``.SZ`` equity / ``OTHER`` respectively.
    """

    def assert_classified(self, raw, market, instrument, normalized):
        self.assertEqual(normalize_ticker_symbol(raw), normalized)
        self.assertEqual(detect_market_type(raw), market)
        self.assertEqual(detect_instrument_type(raw), instrument)

    def test_otc_fund_codes_that_collide_with_equity_prefixes(self):
        self.assert_classified("005827", MarketType.CN_FUND, InstrumentType.FUND, "005827")
        self.assert_classified("110011", MarketType.CN_FUND, InstrumentType.FUND, "110011")

    def test_recognized_otc_fund_codes(self):
        self.assert_classified("012920", MarketType.CN_FUND, InstrumentType.FUND, "012920")

    def test_listed_funds(self):
        self.assert_classified("510300", MarketType.CN_A, InstrumentType.FUND, "510300.SH")
        self.assert_classified("161725", MarketType.CN_A, InstrumentType.FUND, "161725.SZ")
        self.assert_classified("159915", MarketType.CN_A, InstrumentType.FUND, "159915.SZ")

    def test_a_share_equities(self):
        self.assert_classified("000001", MarketType.CN_A, InstrumentType.EQUITY, "000001.SZ")
        self.assert_classified("600519", MarketType.CN_A, InstrumentType.EQUITY, "600519.SH")
        self.assert_classified("300750", MarketType.CN_A, InstrumentType.EQUITY, "300750.SZ")
        self.assert_classified("830833", MarketType.CN_A, InstrumentType.EQUITY, "830833.BJ")
        self.assert_classified("920002", MarketType.CN_A, InstrumentType.EQUITY, "920002.BJ")
        self.assert_classified("430047", MarketType.CN_A, InstrumentType.EQUITY, "430047.BJ")

    def test_explicit_suffix_is_respected(self):
        self.assert_classified("600519.SH", MarketType.CN_A, InstrumentType.EQUITY, "600519.SH")

    def test_non_cn_symbols_unchanged(self):
        self.assert_classified("AAPL", MarketType.US, InstrumentType.EQUITY, "AAPL")
        self.assert_classified("0700.HK", MarketType.HK, InstrumentType.EQUITY, "0700.HK")
        self.assert_classified("BTC-USD", MarketType.CRYPTO, InstrumentType.CRYPTO, "BTC-USD")


@pytest.mark.unit
class ResolveInstrumentIdentityTests(unittest.TestCase):
    def setUp(self):
        resolve_instrument_identity.cache_clear()

    def test_resolves_company_metadata_from_yfinance(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {
                "longName": "TOTO LTD.",
                "shortName": "TOTO",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
                "quoteType": "EQUITY",
            }
            identity = resolve_instrument_identity("totdy")
        mock.assert_called_once_with("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO LTD.")
        self.assertEqual(identity["sector"], "Industrials")
        self.assertEqual(identity["industry"], "Building Products & Equipment")
        self.assertEqual(identity["exchange"], "PNK")

    def test_falls_back_to_short_name(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {"shortName": "TOTO", "sector": "Industrials"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO")

    def test_skips_placeholder_values(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "  ", "sector": "None", "industry": "n/a"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity, {})

    def test_fails_open_on_exception(self):
        with patch(
            "tradingagents.agents.utils.agent_utils.yf.Ticker",
            side_effect=RuntimeError("rate limited"),
        ):
            self.assertEqual(resolve_instrument_identity("TOTDY"), {})

    def test_result_is_cached(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "TOTO LTD."}
            first = resolve_instrument_identity("TOTDY")
            second = resolve_instrument_identity("TOTDY")
        mock.assert_called_once()  # second call served from cache
        self.assertEqual(first, second)

    def test_cn_tickers_skip_yfinance_identity_lookup(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            self.assertEqual(resolve_instrument_identity("600519"), {})
            self.assertEqual(resolve_instrument_identity("012920"), {})

        mock.assert_not_called()


@pytest.mark.unit
class BuildInstrumentContextTests(unittest.TestCase):
    def test_mentions_exact_symbol_without_identity(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)
        self.assertNotIn("Resolved identity", context)

    def test_injects_resolved_identity(self):
        context = build_instrument_context(
            "TOTDY", "stock",
            {
                "company_name": "TOTO LTD.",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
            },
        )
        self.assertIn("Company: TOTO LTD.", context)
        self.assertIn("Industrials / Building Products & Equipment", context)
        self.assertIn("Exchange: PNK", context)
        self.assertIn("Do not substitute a different company", context)

    def test_crypto_uses_name_label_and_keeps_hint(self):
        context = build_instrument_context(
            "BTC-USD", "crypto", {"company_name": "Bitcoin USD"}
        )
        self.assertIn("Name: Bitcoin USD", context)
        self.assertIn("crypto asset rather than a company", context)


@pytest.mark.unit
class GetInstrumentContextFromStateTests(unittest.TestCase):
    def test_prefers_precomputed_context(self):
        state = {"company_of_interest": "TOTDY", "instrument_context": "PRECOMPUTED"}
        self.assertEqual(get_instrument_context_from_state(state), "PRECOMPUTED")

    def test_fallback_is_network_free_ticker_only(self):
        # No instrument_context and no yfinance call — must not hit the network.
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            context = get_instrument_context_from_state(
                {"company_of_interest": "NVDA", "asset_type": "stock"}
            )
        mock.assert_not_called()
        self.assertIn("NVDA", context)

    def test_fallback_respects_asset_type(self):
        context = get_instrument_context_from_state(
            {"company_of_interest": "BTC-USD", "asset_type": "crypto"}
        )
        self.assertIn("crypto asset", context)


@pytest.mark.unit
class ContextAnchoredPlaceholderTests(unittest.TestCase):
    """#888 — the message-clear placeholder must not be a bare 'Continue'."""

    def _run(self, state_extra):
        state = {
            "messages": [
                HumanMessage(content="old", id="h1"),
                AIMessage(content="reply", id="a1"),
            ],
            **state_extra,
        }
        return create_msg_delete()(state)

    def test_placeholder_is_not_bare_continue(self):
        result = self._run(
            {"company_of_interest": "EC", "asset_type": "stock", "trade_date": "2026-05-28"}
        )
        placeholder = result["messages"][-1]
        self.assertIsInstance(placeholder, HumanMessage)
        self.assertNotEqual(placeholder.content.strip(), "Continue")

    def test_placeholder_carries_resolved_identity(self):
        result = self._run(
            {
                "company_of_interest": "EC",
                "instrument_context": "The instrument to analyze is `EC`. Resolved identity: Company: Ecopetrol.",
                "trade_date": "2026-05-28",
            }
        )
        content = result["messages"][-1].content
        self.assertIn("Ecopetrol", content)
        self.assertIn("2026-05-28", content)

    def test_old_messages_are_removed(self):
        result = self._run({"company_of_interest": "EC", "trade_date": "2026-05-28"})
        removals = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(len(removals), 2)
        self.assertEqual(len(humans), 1)

    def test_safe_defaults_when_state_minimal(self):
        result = create_msg_delete()({"messages": [], "company_of_interest": "EC"})
        placeholder = result["messages"][-1]
        self.assertNotEqual(placeholder.content.strip(), "Continue")
        self.assertIn("EC", placeholder.content)

    def test_collects_data_contract_status_before_deleting_messages(self):
        gate_text = """## Verified Market Data Contract Gate

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
        result = create_msg_delete()(
            {
                "messages": [
                    ToolMessage(content=gate_text, tool_call_id="call_1", id="tool_1"),
                ],
                "company_of_interest": "012920",
                "trade_date": "2026-05-22",
            }
        )

        self.assertEqual(result["data_contract_status"]["overall"], "warning")
        self.assertEqual(result["data_contract_status"]["checks"][0]["semantic"], "nav")
        self.assertEqual(result["data_contract_status"]["checks"][0]["warnings"], ["nav_semantic"])


if __name__ == "__main__":
    unittest.main()
