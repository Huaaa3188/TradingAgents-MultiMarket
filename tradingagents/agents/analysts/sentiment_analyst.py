"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    append_fund_semantic_warning,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.instruments import MarketType
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)
        market_type = state.get("market_type")
        is_china_market = market_type in (MarketType.CN_A.value, MarketType.CN_FUND.value)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)
        if is_china_market:
            stocktwits_block = (
                "<not_used> StockTwits is not treated as a representative source for "
                "mainland China tickers in this workflow. Infer sentiment from China-local "
                "news, fund/company announcements, policy context, and market data instead."
            )
            reddit_block = (
                "<not_used> Reddit is not treated as a core sentiment source for mainland "
                "China tickers. If local discussion data is unavailable, state that the "
                "retail/social sentiment read is limited instead of substituting US forum chatter."
            )
        else:
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            instrument_type=state.get("instrument_type"),
            market_type=market_type,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    # No tool-calling here: the data is pre-fetched into the
                    # prompt, so tool-range wording would only invite a
                    # hallucinated tool call (#1130).
                    " Today's date is {current_date}; treat it as 'now' for all analysis. {instrument_context}"
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )
        report_text = append_fund_semantic_warning(state, report_text)

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    instrument_type: str = "equity",
    market_type: str = "",
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    is_otc_fund = instrument_type == "fund" and market_type == MarketType.CN_FUND.value
    is_china_market = market_type in (MarketType.CN_A.value, MarketType.CN_FUND.value)
    instrument_label = "fund" if is_otc_fund else "listed fund" if instrument_type == "fund" else "instrument"

    if is_otc_fund:
        best_practices = """1. **Read retail and news sentiment as context for the fund's investment theme and asset class.** Focus on optimism/pessimism regarding the underlying portfolio exposures rather than the fund as an operating company.
2. **Look for divergences between fund NAV performance, fund flows/attention, and macro news.** Do not infer exchange-traded volume, premium/discount, or intraday liquidity unless a source explicitly provides it.
3. **Weight discussions by engagement and theme.** Pay special attention to popular threads discussing the target sector, global market exposure, QDII timing, FX risk, fees, manager changes, holdings, or redemption/subscription constraints.
4. **Identify recurring thematic narratives.** What driving themes or macro policies are retail and news talking about for this fund's holdings or asset class?
5. **Absolutely avoid analyzing company revenue, earnings, competitive business moat, or corporate management.** This is a fund share class. Keep all insights focused on NAV, assets it holds, scale, fees, allocation, manager, and thematic risk.
6. **Be honest about data limits.** If StockTwits or Reddit returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, flag this caveat explicitly."""
    elif instrument_type == "fund":
        best_practices = """1. **Read the StockTwits and Reddit sentiment as a leading indicator of thematic or index-level sentiment.** Focus on the overall optimism/pessimism regarding the tracked index, asset class, or sector rather than the fund as an operating company.
2. **Look for divergences in capital flows and pricing sentiment.** Note if there is retail excitement on forums but institutional discount/premium anomalies, or if there is panic while the underlying index fundamentals remain stable.
3. **Weight discussions by engagement and theme.** Pay special attention to popular threads discussing the target thematic sector, tracking quality, fees, or liquidity of the fund.
4. **Identify recurring thematic narratives.** What driving themes or macro policies are retail and news talking about for this tracking index/benchmark?
5. **Absolutely avoid analyzing company revenue, earnings, competitive business moat, or corporate management.** This is a listed fund/ETF. Keep all insights focused on the assets it holds, its tracked index, scale, liquidity, and thematic risk.
6. **Be honest about data limits.** If StockTwits or Reddit returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, flag this caveat explicitly."""
        if is_china_market:
            best_practices += "\n7. **For mainland China listed funds, treat China-local news, fund announcements, policy context, exchange trading constraints, liquidity, and premium/discount evidence as primary. Do not substitute US forum chatter for local investor sentiment.**"
    elif is_china_market:
        best_practices = """1. **Treat China-local news, announcements, and policy context as the primary sentiment evidence.** Do not substitute US forum chatter for A-share investor sentiment.
2. **Separate policy/event evidence from opinion.** Policy headlines, exchange announcements, regulatory actions, and company disclosures are evidence; market rumors are not.
3. **Look for divergences between price/NAV behavior, company or fund announcements, and local macro policy context.**
4. **Account for A-share market structure.** Daily price limits, trading halts, T+1-style constraints, and holiday effects can amplify or mute sentiment signals.
5. **Be honest about data limits.** If local social discussion is unavailable, explicitly lower confidence rather than inventing retail sentiment."""
    else:
        best_practices = """1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.
2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).
3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.
4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.
5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.
6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this caveat explicitly. If the sources are silent on a given subreddit, say so.
7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.
8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call."""

    if is_china_market:
        data_sources = f"""### China local news and announcements — AkShare / Eastmoney / Tiantian Fund
Primary evidence for mainland China sentiment. This block may include company news, fund announcements, NAV/fund context, and policy-sensitive local headlines.

<start_of_china_local_news>
{news_block}
<end_of_china_local_news>

### Mainland China retail/social sentiment availability
Use this as a data-quality caveat, not as evidence when it says unavailable.

<start_of_china_retail_context>
{stocktwits_block}
<end_of_china_retail_context>

### Offshore English-language forum caveat
Use this as a coverage caveat. Do not treat Reddit or StockTwits as core sentiment evidence for China-local tickers unless real, ticker-specific data is present.

<start_of_offshore_forum_context>
{reddit_block}
<end_of_offshore_forum_context>"""
    else:
        data_sources = f"""### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>"""

    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for the {instrument_label} {ticker} covering the period from {start_date} to {end_date}, drawing on the pre-fetched data blocks below.

## Data sources (pre-fetched, in this prompt)

{data_sources}

## How to analyze this data (best practices)

{best_practices}

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
