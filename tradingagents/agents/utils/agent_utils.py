import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.prediction_markets_tools import get_prediction_markets
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
from tradingagents.dataflows.contracts import collect_data_contract_status_from_messages
from tradingagents.dataflows.instruments import (
    InstrumentType,
    MarketType,
    detect_market_type,
)
from tradingagents.dataflows.symbol_utils import normalize_symbol

# Public surface: the data tools are imported here so agents and the graph
# import them from one place, plus the instrument/language helpers defined below.
__all__ = [
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
    "get_verified_market_snapshot",
    "build_instrument_context",
    "resolve_instrument_identity",
    "get_instrument_context_from_state",
    "get_language_instruction",
    "create_msg_delete",
]

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def is_chinese_output_language(lang: str = None) -> bool:
    if lang is None:
        from tradingagents.dataflows.config import get_config

        lang = get_config().get("output_language", "English")
    normalized = str(lang).strip().lower()
    return normalized in {"chinese", "zh", "zh-cn", "cn", "中文", "简体中文", "汉语"}


_FUND_COMPANY_SEMANTIC_TERMS = (
    "company revenue",
    "company earnings",
    "company profit",
    "profit margin",
    "corporate debt",
    "balance sheet",
    "balance-sheet",
    "cash flow statement",
    "cash-flow",
    "公司营收",
    "公司收入",
    "公司利润",
    "利润率",
    "资产负债表",
    "现金流量表",
)


def find_fund_company_semantic_violations(report: str) -> list[str]:
    """Return company-fundamental terms that should not drive fund analysis."""
    if not isinstance(report, str) or not report:
        return []
    lower_report = report.lower()
    return [
        term
        for term in _FUND_COMPANY_SEMANTIC_TERMS
        if term.lower() in lower_report
    ]


def append_fund_semantic_warning(state: Mapping[str, Any], report: str) -> str:
    """Append a non-blocking warning when a fund report uses company semantics."""
    if not report or state.get("instrument_type") != InstrumentType.FUND.value:
        return report
    if "Fund semantics warning:" in report:
        return report

    violations = find_fund_company_semantic_violations(report)
    if not violations:
        return report

    terms = ", ".join(sorted(set(violations), key=str.lower))
    return (
        report
        + "\n\n> Fund semantics warning: this fund report contains operating-company "
        + f"term(s): {terms}. Review these claims against fund/NAV/holdings context; "
        + "do not treat them as company fundamentals without explicit tool evidence."
    )


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per ticker per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).
    """
    if detect_market_type(ticker) in (MarketType.CN_A, MarketType.CN_FUND):
        return {}

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
    instrument_type: str | None = None,
    market_type: str | None = None,
    company_display_name: str | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    if identity is not None and not isinstance(identity, Mapping):
        instrument_type = str(identity)
        identity = None

    instrument_value = (instrument_type or "").lower()
    market_value = (market_type or "").lower()
    is_crypto = asset_type == "crypto" or instrument_value == InstrumentType.CRYPTO.value

    if is_crypto:
        instrument_label = "asset"
        extra_hint = (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    elif instrument_value == InstrumentType.FUND.value:
        if market_value == MarketType.CN_FUND.value:
            instrument_label = "fund"
            extra_hint = (
                " Treat it as an OTC mutual fund or QDII fund share class, not as an operating company"
                " and not as an exchange-traded listed fund. Focus on NAV trend, fund category,"
                " manager, assets under management, fees, asset allocation, holdings concentration,"
                " redemption/subscription constraints, QDII/FX exposure where applicable, and market risk;"
                " do not infer company revenue, earnings, balance-sheet, cash-flow fundamentals,"
                " exchange-traded volume, or premium/discount unless a tool explicitly provides it."
            )
        else:
            instrument_label = "listed fund"
            extra_hint = (
                " Treat it as an exchange-traded fund or listed fund, not as an operating company."
                " Focus on the tracked benchmark or theme, premium/discount, liquidity, fund size,"
                " fees, holdings, and market risk; do not infer company revenue, earnings,"
                " balance-sheet, or cash-flow fundamentals."
            )
    else:
        instrument_label = "instrument"
        extra_hint = ""

    extra_hint += build_market_rules_context(instrument_value, market_value)

    display_name_hint = ""
    if company_display_name and company_display_name != ticker:
        display_name_hint = (
            f" (Verified Name: **{company_display_name}**). You MUST strictly "
            "analyze this specific target and strictly refer to it by its verified "
            "name. Do NOT hallucinate any other names or indexes."
        )

    context = (
        f"The {instrument_label} to analyze is `{ticker}`{display_name_hint}. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.SH`, `.SZ`, `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    return context + extra_hint


def build_market_rules_context(instrument_type: str = "", market_type: str = "") -> str:
    """Return deterministic market-rule context for locally supported China markets."""
    instrument_value = (instrument_type or "").lower()
    market_value = (market_type or "").lower()

    if market_value == MarketType.CN_A.value:
        if instrument_value == InstrumentType.FUND.value:
            return (
                " Market rules context: this is a mainland China listed fund quoted in CNY. "
                "Use exchange-traded listed-fund mechanics: exchange trading calendar, midday break, "
                "T+1-style settlement constraints, exchange holidays, possible trading halts, "
                "secondary-market liquidity, tracking error, fees, and premium/discount when provided. "
                "Do not apply listed-company revenue or financial-statement semantics."
            )
        return (
            " Market rules context: this is a mainland China A-share equity quoted in CNY. "
            "Account for A-share trading sessions with a midday break, T+1-style settlement constraints, "
            "exchange holidays, board-specific daily price-limit rules, possible ST/suspension status, "
            "and China-specific disclosure and financial-reporting cadence. Do not assume US-style "
            "continuous trading, same-day round trips, or unlimited intraday price movement."
        )

    if market_value == MarketType.CN_FUND.value:
        return (
            " Market rules context: this is a mainland China OTC fund code quoted by daily NAV in CNY "
            "unless the share class states otherwise. Use fund NAV publication cadence, subscription "
            "and redemption status, fund disclosure cadence, holiday effects, and QDII overseas-market "
            "and FX timing where relevant. Do not infer exchange-traded volume, intraday liquidity, "
            "or premium/discount for this OTC mutual fund unless a tool explicitly provides it."
        )

    return ""


def build_verified_target_context(state) -> str:
    """Build the shared target identity context from graph state."""
    return get_instrument_context_from_state(state)


def get_instrument_target_label(state) -> str:
    instrument_type = state.get("instrument_type", "")
    if state.get("asset_type") == "crypto" or instrument_type == InstrumentType.CRYPTO.value:
        return "asset"
    if instrument_type == InstrumentType.FUND.value:
        if state.get("market_type") == MarketType.CN_FUND.value:
            return "fund"
        return "listed fund"
    return "stock"


def get_fundamentals_report_label(state) -> str:
    instrument_type = state.get("instrument_type", "")
    if state.get("asset_type") == "crypto" or instrument_type == InstrumentType.CRYPTO.value:
        return "Asset fundamentals report (may be unavailable for crypto)"
    if instrument_type == InstrumentType.FUND.value:
        return "Fund profile report"
    return "Company fundamentals report"


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
        instrument_type=state.get("instrument_type"),
        market_type=state.get("market_type"),
        company_display_name=state.get("company_display_name"),
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        updates = {"messages": removal_operations + [placeholder]}
        # Only the new checks collected from these tool messages; the state
        # channel reducer accumulates them across nodes.
        contract_status = collect_data_contract_status_from_messages(messages)
        if contract_status is not None:
            updates["data_contract_status"] = contract_status
        return updates

    return delete_messages
