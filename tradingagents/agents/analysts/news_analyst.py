from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    append_fund_semantic_warning,
    get_global_news,
    get_instrument_context_from_state,
    get_instrument_target_label,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.dataflows.instruments import MarketType


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_label = get_instrument_target_label(state)
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        instrument_type = state.get("instrument_type")
        if instrument_type == "fund":
            if state.get("market_type") == MarketType.CN_FUND.value:
                fund_label = "fund"
                fund_events = "NAV updates, fund manager changes, fee changes, asset-allocation or holdings disclosures, subscription/redemption status, and QDII overseas-market or FX developments"
                benchmark_phrase = "investment theme, asset class, or portfolio exposures"
            else:
                fund_label = "listed fund"
                fund_events = "dividends, fund manager changes, share conversions, or trading suspensions"
                benchmark_phrase = "tracked index or benchmark"
            system_message = (
                f"You are a fund news researcher tasked with analyzing recent announcements and macroeconomic trends over the past week for this {fund_label}. "
                f"Please write a comprehensive report covering key fund announcements (such as {fund_events}) as well as broader macroeconomic or sector policies affecting the {benchmark_phrase}. "
                f"Use the available tools: get_news(ticker, start_date, end_date) for {asset_label}-specific or targeted announcements/news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events when relevant. "
                "Provide specific, actionable insights with supporting evidence to help traders make informed decisions. Do not describe the fund as an operating company and do not infer company product, revenue, or business-moat news."
                + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
                + get_language_instruction()
            )
        else:
            system_message = (
                f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(ticker, start_date, end_date) for {asset_label}-specific news by ticker symbol, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
                + get_language_instruction()
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
            report = append_fund_semantic_warning(state, report)

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
