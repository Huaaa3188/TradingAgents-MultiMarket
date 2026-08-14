"""Background analysis manager for the web UI.

Each analysis runs ``TradingAgentsGraph.graph.stream`` in a daemon thread and
forwards the graph's per-node chunks to per-subscriber event queues consumed
by the SSE endpoint. The graph factory is injectable so tests can substitute a
fake graph without touching the real LLM/data path.

Run lifecycle: ``pending`` -> ``running`` -> ``done`` | ``error``.
Events pushed to the run's queue (replayed on SSE connect):

  - ``{"type": "meta", ...}``            ticker/market/instrument detection
  - ``{"type": "chunk", "data": {...}}``  report-section deltas from the graph
  - ``{"type": "signal", "signal": ...}`` final BUY/SELL/HOLD read
  - ``{"type": "done", "report_path": ...}``
  - ``{"type": "error", "message": ...}``
  - ``None``                             end-of-stream sentinel
"""

from __future__ import annotations

import copy
import itertools
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cli.utils import (
    AssetType,
    detect_asset_type,
    filter_analysts_for_asset_type,
    provider_default_url,
)
from tradingagents.dataflows.instruments import (
    MarketType,
    detect_instrument_type,
    detect_market_type,
    normalize_ticker_symbol,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.reporting import write_report_tree

# Fields forwarded to the browser per chunk; messages are intentionally
# excluded (huge LangChain objects the UI does not render).
_CHUNK_FIELDS = (
    "company_display_name",
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_debate_state",
    "risk_debate_state",
    "trader_investment_plan",
    "final_trade_decision",
    "data_contract_status",
    "instrument_type",
    "market_type",
)

_ANALYST_KEYS = ("market", "social", "news", "fundamentals")


def default_graph_factory(
    selected_analysts: tuple[str, ...], config: dict[str, Any], debug: bool = False
):
    """Build the real TradingAgentsGraph; tests substitute their own factory."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph(selected_analysts, config=config, debug=debug)


def build_graph_config(request: dict[str, Any]) -> dict[str, Any]:
    """Assemble a run config from the web form, honoring env precedence.

    Mirrors ``cli.main._build_run_config`` for the fields the browser exposes:
    provider, models, backend URL, output language, round counts, checkpoint.
    Explicit env overrides on ``DEFAULT_CONFIG`` are preserved.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    provider = str(request.get("provider") or config["llm_provider"]).lower()
    config["llm_provider"] = provider
    if request.get("quick_think_llm"):
        config["quick_think_llm"] = request["quick_think_llm"]
    if request.get("deep_think_llm"):
        config["deep_think_llm"] = request["deep_think_llm"]

    requested_url = request.get("backend_url")
    config["backend_url"] = (
        requested_url
        if requested_url
        else config.get("backend_url") or provider_default_url(provider)
    )

    if request.get("output_language"):
        config["output_language"] = request["output_language"]
    if request.get("max_debate_rounds") is not None:
        config["max_debate_rounds"] = int(request["max_debate_rounds"])
    if request.get("max_risk_discuss_rounds") is not None:
        config["max_risk_discuss_rounds"] = int(request["max_risk_discuss_rounds"])
    if request.get("checkpoint") is not None:
        config["checkpoint_enabled"] = bool(request["checkpoint"])

    from cli.main import _apply_data_vendor_override
    data_vendors = request.get("data_vendors")
    if not data_vendors:
        # Market-aware default: China tickers get the akshare chain; other
        # markets keep the yfinance defaults (no failing akshare first hop).
        market_type = detect_market_type(request.get("ticker") or "")
        if market_type in (MarketType.CN_A, MarketType.CN_FUND):
            data_vendors = "akshare"
    if data_vendors:
        _apply_data_vendor_override(config, data_vendors)

    return config


def select_analyst_keys(request: dict[str, Any], asset_type: AssetType) -> list[str]:
    """Normalize the analyst selection, filtering asset-inapplicable roles."""
    requested = request.get("analysts") or list(_ANALYST_KEYS)
    selected = [key for key in _ANALYST_KEYS if key in requested]
    if not selected:
        selected = list(_ANALYST_KEYS)

    from cli.models import AnalystType

    typed = [AnalystType(key) for key in selected]
    filtered = filter_analysts_for_asset_type(typed, asset_type)
    return [analyst.value for analyst in filtered]


def _chunk_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {key: chunk[key] for key in _CHUNK_FIELDS if chunk.get(key) is not None}


class AnalysisManager:
    """Registry of in-flight and finished analyses, each with an event queue."""

    def __init__(
        self,
        graph_factory: Callable | None = None,
        results_dir: str | None = None,
    ):
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._counter = itertools.count(1)
        self._graph_factory = graph_factory or default_graph_factory
        self._results_dir = results_dir or DEFAULT_CONFIG["results_dir"]

    def start(self, request: dict[str, Any]) -> str:
        """Validate the request and launch an analysis; return its run id."""
        ticker = normalize_ticker_symbol(str(request.get("ticker") or ""))
        if not ticker:
            raise ValueError("ticker is required")
        analysis_date = str(request.get("analysis_date") or "").strip()
        if not analysis_date:
            raise ValueError("analysis_date is required (YYYY-MM-DD)")

        run_id = f"run-{next(self._counter)}"
        run = {
            "id": run_id,
            "status": "pending",
            "ticker": ticker,
            "analysis_date": analysis_date,
            "events": [],            # replay buffer for late SSE connects
            "subscribers": [],       # per-SSE-client queues broadcast by _emit
            "signal": None,
            "report_path": None,
            "error": None,
            "started_at": time.time(),
        }
        with self._lock:
            self._runs[run_id] = run
        thread = threading.Thread(
            target=self._execute, args=(run_id, request, run), daemon=True
        )
        thread.start()
        return run_id

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return a snapshot of the run (without subscriber queues)."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return {key: value for key, value in run.items() if key != "subscribers"}

    def subscribe(self, run_id: str) -> tuple[queue.Queue, list[Any]] | None:
        """Register a new SSE client; return (queue, replay snapshot).

        Registration and snapshot happen under the same lock as ``_emit``, so
        an event is never both missed (emitted between snapshot and subscribe)
        and duplicated (emitted before the snapshot and re-pushed to the new
        queue).
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            subscriber: queue.Queue = queue.Queue()
            run["subscribers"].append(subscriber)
            snapshot = list(run["events"])
        return subscriber, snapshot

    def unsubscribe(self, run_id: str, subscriber: queue.Queue) -> None:
        """Remove a disconnected SSE client's queue."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None and subscriber in run["subscribers"]:
                run["subscribers"].remove(subscriber)

    def _emit(self, run: dict[str, Any], event: dict[str, Any] | None) -> None:
        with self._lock:
            run["events"].append(event)
            for subscriber in run["subscribers"]:
                subscriber.put(event)

    def _execute(
        self, run_id: str, request: dict[str, Any], run: dict[str, Any]
    ) -> None:
        graph = None
        try:
            ticker = run["ticker"]
            analysis_date = run["analysis_date"]
            asset_type = detect_asset_type(ticker)
            instrument_type = detect_instrument_type(ticker).value
            market_type = detect_market_type(ticker).value
            analyst_keys = select_analyst_keys(request, asset_type)
            config = build_graph_config(request)

            run["status"] = "running"
            self._emit(
                run,
                {
                    "type": "meta",
                    "ticker": ticker,
                    "analysis_date": analysis_date,
                    "asset_type": asset_type.value,
                    "instrument_type": instrument_type,
                    "market_type": market_type,
                    "analysts": analyst_keys,
                },
            )

            graph = self._graph_factory(tuple(analyst_keys), config=config, debug=False)
            instrument_context = graph.resolve_instrument_context(ticker, asset_type.value)
            init_state = graph.propagator.create_initial_state(
                ticker,
                analysis_date,
                asset_type=asset_type.value,
                instrument_type=instrument_type,
                market_type=market_type,
                instrument_context=instrument_context,
            )
            args = graph.propagator.get_graph_args()
            checkpoint_args = graph.enter_checkpoint_stream(ticker, analysis_date, asset_type=asset_type.value)
            if checkpoint_args:
                args.setdefault("config", {}).setdefault("configurable", {}).update(
                    checkpoint_args["config"]["configurable"]
                )

            final_state: dict[str, Any] = {}
            for chunk in graph.graph.stream(init_state, **args):
                final_state.update(chunk)
                payload = _chunk_payload(chunk)
                if payload:
                    self._emit(run, {"type": "chunk", "data": payload})

            signal = None
            final_decision = final_state.get("final_trade_decision")
            if final_decision:
                signal = graph.process_signal(final_decision)

            save_path = (
                Path(self._results_dir) / safe_component(ticker) / safe_component(analysis_date)
            )
            complete_report = write_report_tree(final_state, ticker, save_path)
            run["report_path"] = str(complete_report.relative_to(self._results_dir))
            run["signal"] = signal
            run["status"] = "done"
            self._emit(
                run,
                {
                    "type": "done",
                    "signal": signal,
                    "report_path": run["report_path"],
                    "ticker": ticker,
                },
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            run["status"] = "error"
            run["error"] = f"{type(exc).__name__}: {exc}"
            self._emit(run, {"type": "error", "message": run["error"]})
        finally:
            if graph is not None:
                graph.exit_checkpoint_stream()
            run["finished_at"] = time.time()
            self._emit(run, None)  # end-of-stream sentinel


def safe_component(value: str) -> str:
    """Keep user-provided path components confined to a single directory.

    Invalid values degrade to "" (which can never match a real report path)
    instead of raising, so hostile input yields 404 rather than a 500.
    """
    from tradingagents.dataflows.utils import safe_ticker_component

    try:
        return safe_ticker_component(value)
    except ValueError:
        return ""


__all__ = [
    "AnalysisManager",
    "build_graph_config",
    "default_graph_factory",
    "select_analyst_keys",
]
