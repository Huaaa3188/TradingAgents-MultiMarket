"""Test checkpoint resume: crash mid-analysis, re-run resumes from last node."""

import tempfile
import unittest
from typing import TypedDict

from langgraph.graph import END, StateGraph

from tradingagents.graph.checkpointer import (
    checkpoint_step,
    clear_checkpoint,
    get_checkpointer,
    has_checkpoint,
    thread_id,
)

# Mutable flag to simulate crash on first run
_should_crash = False


class _SimpleState(TypedDict):
    count: int


def _node_a(state: _SimpleState) -> dict:
    return {"count": state["count"] + 1}


def _node_b(state: _SimpleState) -> dict:
    if _should_crash:
        raise RuntimeError("simulated mid-analysis crash")
    return {"count": state["count"] + 10}


def _build_graph() -> StateGraph:
    builder = StateGraph(_SimpleState)
    builder.add_node("analyst", _node_a)
    builder.add_node("trader", _node_b)
    builder.set_entry_point("analyst")
    builder.add_edge("analyst", "trader")
    builder.add_edge("trader", END)
    return builder


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"
        self.date = "2026-04-20"

    def test_crash_and_resume(self):
        """Crash at 'trader' node, then resume from checkpoint."""
        global _should_crash
        builder = _build_graph()
        tid = thread_id(self.ticker, self.date)
        cfg = {"configurable": {"thread_id": tid}}

        # Run 1: crash at trader node
        _should_crash = True
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config=cfg)

        # Checkpoint should exist at step 1 (analyst completed)
        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))
        step = checkpoint_step(self.tmpdir, self.ticker, self.date)
        self.assertEqual(step, 1)

        # Run 2: resume — trader succeeds this time
        _should_crash = False
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke(None, config=cfg)

        # analyst added 1, trader added 10 → 11
        self.assertEqual(result["count"], 11)

    def test_clear_checkpoint_allows_fresh_start(self):
        """After clearing, the graph starts from scratch."""
        global _should_crash
        builder = _build_graph()
        tid = thread_id(self.ticker, self.date)
        cfg = {"configurable": {"thread_id": tid}}

        # Create a checkpoint by crashing
        _should_crash = True
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config=cfg)

        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))

        # Clear it
        clear_checkpoint(self.tmpdir, self.ticker, self.date)
        self.assertFalse(has_checkpoint(self.tmpdir, self.ticker, self.date))

        # Fresh run succeeds from scratch
        _should_crash = False
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config=cfg)

        self.assertEqual(result["count"], 11)


    def test_different_date_starts_fresh(self):
        """A different date must NOT resume from an existing checkpoint."""
        global _should_crash
        builder = _build_graph()
        date2 = "2026-04-21"

        # Run with date1 — crash to leave a checkpoint
        _should_crash = True
        tid1 = thread_id(self.ticker, self.date)
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid1}})

        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))

        # date2 should have no checkpoint
        self.assertFalse(has_checkpoint(self.tmpdir, self.ticker, date2))

        # Run with date2 — should start fresh and succeed
        _should_crash = False
        tid2 = thread_id(self.ticker, date2)
        self.assertNotEqual(tid1, tid2)

        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid2}})

        # Fresh run: analyst +1, trader +10 = 11
        self.assertEqual(result["count"], 11)

        # Original date checkpoint still exists (untouched)
        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))


class TestCheckpointSignature(unittest.TestCase):
    """A different graph shape (analyst selection / depth / asset mode) must not
    resume the previous run's checkpoint (#1089)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"
        self.date = "2026-04-20"

    def test_empty_signature_is_legacy_id(self):
        self.assertEqual(
            thread_id(self.ticker, self.date),
            thread_id(self.ticker, self.date, ""),
        )

    def test_signature_changes_thread_id(self):
        legacy = thread_id(self.ticker, self.date)
        sig_a = thread_id(self.ticker, self.date, "analysts=market,news|asset=stock")
        sig_b = thread_id(self.ticker, self.date, "analysts=market|asset=stock")
        self.assertNotEqual(sig_a, sig_b)          # different graph shapes differ
        self.assertNotEqual(legacy, sig_a)         # signature-keyed differs from legacy
        self.assertEqual(                          # same inputs are stable
            sig_a, thread_id(self.ticker, self.date, "analysts=market,news|asset=stock")
        )

    def test_different_signature_starts_fresh(self):
        global _should_crash
        builder = _build_graph()
        sig1 = "analysts=market,news,fundamentals|asset=stock"
        sig2 = "analysts=market|asset=stock"       # dropped analysts -> different graph

        _should_crash = True
        tid1 = thread_id(self.ticker, self.date, sig1)
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid1}})

        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date, sig1))
        # A different graph shape has no checkpoint to resume from.
        self.assertFalse(has_checkpoint(self.tmpdir, self.ticker, self.date, sig2))

        _should_crash = False
        tid2 = thread_id(self.ticker, self.date, sig2)
        self.assertNotEqual(tid1, tid2)
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid2}})
        self.assertEqual(result["count"], 11)
        # sig1's checkpoint remains untouched.
        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date, sig1))

    def test_run_signature_captures_graph_shape(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # Build a bare instance to exercise the pure helper without heavy __init__.
        g = object.__new__(TradingAgentsGraph)
        g.selected_analysts = ("market", "news")
        g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
        base = g._run_signature("stock")

        self.assertNotEqual(base, g._run_signature("crypto"))     # asset mode
        g.selected_analysts = ("market",)
        self.assertNotEqual(base, g._run_signature("stock"))      # analyst selection
        g.selected_analysts = ("market", "news")
        g.config = {"max_debate_rounds": 3, "max_risk_discuss_rounds": 1}
        self.assertNotEqual(base, g._run_signature("stock"))      # debate depth
        g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 5}
        self.assertNotEqual(base, g._run_signature("stock"))      # risk depth
        # Stable for identical inputs.
        g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
        self.assertEqual(base, g._run_signature("stock"))


class TestGraphCheckpointStream(unittest.TestCase):
    """R2 — TradingAgentsGraph.enter_checkpoint_stream() assembles the same
    per-ticker SqliteSaver + deterministic thread_id on the CLI stream path as
    propagate() does, so a crashed run resumes from the last completed node on
    the same ticker+date (and a different graph shape starts fresh)."""

    def setUp(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        self.tmpdir = tempfile.mkdtemp()
        self.graph = object.__new__(TradingAgentsGraph)
        self.graph.config = {
            "checkpoint_enabled": True,
            "data_cache_dir": self.tmpdir,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
        }
        self.graph.selected_analysts = ("market",)
        self.graph._checkpointer_ctx = None
        self.graph.workflow = _build_graph()

    def test_enter_checkpoint_stream_compiles_saver_and_resumes(self):
        global _should_crash
        ticker, date = "TEST", "2026-04-20"
        signature = self.graph._run_signature("stock")

        cfg = self.graph.enter_checkpoint_stream(ticker, date)
        self.assertIn("configurable", cfg["config"])
        tid = cfg["config"]["configurable"]["thread_id"]
        self.assertEqual(tid, thread_id(ticker, date, signature))

        # Run 1: crash at the trader node.
        _should_crash = True
        with self.assertRaises(RuntimeError):
            self.graph.graph.invoke({"count": 0}, **cfg)
        self.graph.exit_checkpoint_stream()

        self.assertEqual(checkpoint_step(self.tmpdir, ticker, date, signature), 1)

        # Run 2: resume — same deterministic thread id, continues from step 1.
        _should_crash = False
        cfg2 = self.graph.enter_checkpoint_stream(ticker, date)
        self.assertEqual(cfg2["config"]["configurable"]["thread_id"], tid)
        result = self.graph.graph.invoke(None, **cfg2)
        self.assertEqual(result["count"], 11)  # analyst +1 then trader +10
        self.graph.exit_checkpoint_stream()

    def test_crash_resume_sequence_is_repeatable(self):
        """The crash→resume cycle succeeds consistently on repeated runs."""
        global _should_crash
        ticker, date = "TEST", "2026-04-21"
        signature = self.graph._run_signature("stock")

        for _ in range(2):
            cfg = self.graph.enter_checkpoint_stream(ticker, date)
            _should_crash = True
            with self.assertRaises(RuntimeError):
                self.graph.graph.invoke({"count": 0}, **cfg)
            self.graph.exit_checkpoint_stream()
            self.assertEqual(checkpoint_step(self.tmpdir, ticker, date, signature), 1)

            _should_crash = False
            cfg2 = self.graph.enter_checkpoint_stream(ticker, date)
            result = self.graph.graph.invoke(None, **cfg2)
            self.assertEqual(result["count"], 11)
            self.graph.exit_checkpoint_stream()
            # propagate() clears the checkpoint on successful completion; mimic it.
            clear_checkpoint(self.tmpdir, ticker, date, signature)
            self.assertIsNone(checkpoint_step(self.tmpdir, ticker, date, signature))

    def test_different_graph_shape_starts_fresh(self):
        global _should_crash
        ticker, date = "TEST", "2026-04-22"

        cfg = self.graph.enter_checkpoint_stream(ticker, date)
        tid1 = cfg["config"]["configurable"]["thread_id"]
        _should_crash = True
        with self.assertRaises(RuntimeError):
            self.graph.graph.invoke({"count": 0}, **cfg)
        self.graph.exit_checkpoint_stream()
        self.assertIsNotNone(checkpoint_step(self.tmpdir, ticker, date, self.graph._run_signature("stock")))

        # Same ticker+date but a different analyst selection → different thread id.
        self.graph.selected_analysts = ("market", "news")
        cfg2 = self.graph.enter_checkpoint_stream(ticker, date)
        self.assertNotEqual(cfg2["config"]["configurable"]["thread_id"], tid1)
        _should_crash = False
        # Fresh thread (no checkpoint for this signature) needs full input.
        result = self.graph.graph.invoke({"count": 0}, **cfg2)
        self.assertEqual(result["count"], 11)  # fresh run, not a resume
        self.graph.exit_checkpoint_stream()

    def test_checkpoint_disabled_returns_empty_args(self):
        self.graph.config["checkpoint_enabled"] = False
        cfg = self.graph.enter_checkpoint_stream("TEST", "2026-04-20")
        self.assertEqual(cfg, {})
        self.assertIsNone(self.graph._checkpointer_ctx)
        self.graph.exit_checkpoint_stream()  # no-op


if __name__ == "__main__":
    unittest.main()
