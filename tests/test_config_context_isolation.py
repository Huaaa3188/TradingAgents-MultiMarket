"""Per-context (thread) config isolation regression tests.

Before the ContextVar refactor, ``TradingAgentsGraph.__init__`` (and the
propagate path) mutated a process-wide dataflow config, so two concurrent
web UI runs with different vendor chains could observe each other's config
mid-run. These tests encode the invariant: each worker thread only ever sees
its own config, and a worker's ``set_config`` never leaks into other threads.
"""

import threading
import time

import pytest

from tradingagents.dataflows.config import get_config, set_config


def test_set_config_is_isolated_per_thread():
    main_chain = get_config()["data_vendors"]["core_stock_apis"]
    barrier = threading.Barrier(3)
    results = {}

    def worker(name, vendor):
        set_config({"data_vendors": {"core_stock_apis": vendor}})
        barrier.wait(timeout=10)
        results[name] = get_config()["data_vendors"]["core_stock_apis"]

    threads = [
        threading.Thread(target=worker, args=("a", "akshare")),
        threading.Thread(target=worker, args=("b", "alpha_vantage")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=10)
    for thread in threads:
        thread.join(timeout=10)

    assert results == {"a": "akshare", "b": "alpha_vantage"}
    # The main thread never observed either worker's override.
    assert get_config()["data_vendors"]["core_stock_apis"] == main_chain


class _FakePropagator:
    def create_initial_state(self, *args, **kwargs):
        return {"messages": []}

    def get_graph_args(self, callbacks=None):
        return {"stream_mode": "values", "config": {"recursion_limit": 100}}


def _recording_graph_factory(barrier, observed):
    """Graph factory whose stream records the worker thread's dataflow config.

    The constructor mirrors ``TradingAgentsGraph.__init__``'s ``set_config``
    side effect so the test reproduces the real cross-run contamination
    scenario; ``observed`` maps the run's own configured vendor chain to what
    the thread actually read while streaming.
    """

    def factory(selected_analysts, config, debug=False):
        graph = _FakeGraph(selected_analysts, config, debug=debug)

        def stream(init_state, **args):
            barrier.wait(timeout=10)
            observed[graph.config["data_vendors"]["core_stock_apis"]] = (
                get_config()["data_vendors"]["core_stock_apis"]
            )
            yield {"market_report": "# Market\n\n**bold** view"}
            yield {"investment_debate_state": {"bull_history": "bull", "judge_decision": "judge"}}
            yield {"final_trade_decision": "## Decision\n\n**Rating**: BUY"}
            yield {"data_contract_status": {"overall": "pass", "checks": []}}

        graph.stream = stream
        return graph

    return factory


class _FakeGraph:
    def __init__(self, selected_analysts, config, debug=False):
        self.selected_analysts = selected_analysts
        self.config = config
        self.propagator = _FakePropagator()
        self.graph = self
        # Mirror TradingAgentsGraph.__init__'s global-config side effect.
        set_config(config)

    def resolve_instrument_context(self, ticker, asset_type):
        return "instrument context"

    def process_signal(self, final_trade_decision):
        return "BUY"

    def enter_checkpoint_stream(self, ticker, analysis_date, asset_type="stock"):
        return {}

    def exit_checkpoint_stream(self):
        return None

    def stream(self, init_state, **args):  # pragma: no cover - replaced in factory
        raise AssertionError("factory must install the recording stream")


def _wait_for_terminals(manager, run_ids, deadline=15.0):
    end = time.time() + deadline
    while time.time() < end:
        snapshots = [manager.get(run_id) for run_id in run_ids]
        if all(snapshot["status"] in ("done", "error") for snapshot in snapshots):
            return snapshots
        time.sleep(0.05)
    pytest.fail(f"runs did not reach a terminal state within {deadline}s")


def test_concurrent_web_runs_keep_vendor_chains_isolated(tmp_path):
    from webapp.tasks import AnalysisManager

    barrier = threading.Barrier(2)
    observed = {}
    manager = AnalysisManager(
        graph_factory=_recording_graph_factory(barrier, observed),
        results_dir=str(tmp_path),
    )

    run_a = manager.start(
        {"ticker": "SPY", "analysis_date": "2026-01-03", "data_vendors": "akshare"}
    )
    run_b = manager.start(
        {"ticker": "AAPL", "analysis_date": "2026-01-03", "data_vendors": "alpha_vantage"}
    )

    snapshot_a, snapshot_b = _wait_for_terminals(manager, (run_a, run_b))
    for snapshot in (snapshot_a, snapshot_b):
        assert snapshot["status"] == "done", snapshot
        assert snapshot["error"] is None, snapshot

    # Each worker thread read its own configured chain, never the other run's.
    assert observed.get("akshare") == "akshare"
    assert observed.get("alpha_vantage") == "alpha_vantage"
