"""End-to-end HTTP tests for the web UI, driven through the real server.

The graph is substituted with a fake factory (per the manager's injection
point) so the tests exercise the shipped HTTP + SSE + task layers without any
LLM/network call.
"""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from webapp.server import build_server


class _FakePropagator:
    def create_initial_state(self, *args, **kwargs):
        return {"messages": []}

    def get_graph_args(self, callbacks=None):
        return {"stream_mode": "values", "config": {"recursion_limit": 100}}


class _FakeCompiled:
    def stream(self, init_state, **args):
        yield {"market_report": "# Market\n\n**bold** view"}
        yield {"investment_debate_state": {"bull_history": "bull case", "judge_decision": "judge"}}
        yield {"final_trade_decision": "## Decision\n\n**Rating**: BUY"}
        yield {"data_contract_status": {"overall": "pass", "checks": []}}


class _FakeGraph:
    def __init__(self, selected_analysts, config, debug=False):
        self.selected_analysts = selected_analysts
        self.config = config
        self.propagator = _FakePropagator()
        self.graph = _FakeCompiled()

    def resolve_instrument_context(self, ticker, asset_type):
        return "instrument context"

    def process_signal(self, final_trade_decision):
        return "BUY"

    def enter_checkpoint_stream(self, ticker, analysis_date, asset_type="stock"):
        return {}

    def exit_checkpoint_stream(self):
        return None


def _fake_graph_factory(selected_analysts, config, debug=False):
    return _FakeGraph(selected_analysts, config, debug=debug)


@pytest.fixture()
def server(tmp_path):
    srv = build_server(port=0, graph_factory=_fake_graph_factory, results_dir=str(tmp_path))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    host, port = srv.server_address
    base = f"http://{host}:{port}"
    yield base
    srv.shutdown()
    srv.server_close()


def _request(base, path, payload=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"error": body}


def _get(base, path):
    return _request(base, path)


def _post(base, path, payload):
    return _request(base, path, payload)


def test_config_exposes_providers_models_and_defaults(server):
    status, data = _get(server, "/api/config")
    assert status == 200
    assert any(p["key"] == "openai" for p in data["providers"])
    assert "models" in data and "openai" in data["models"]
    assert data["defaults"]["llm_provider"]
    assert data["defaults"]["today"]


def test_detect_classifies_cn_ticker(server):
    status, data = _get(server, "/api/detect?ticker=600519")
    assert status == 200
    assert data["ticker"] == "600519.SH"
    assert data["market_type"] == "cn_a"
    assert data["instrument_type"] == "equity"
    assert data["asset_type"] == "stock"


def test_render_endpoint_escapes_html(server):
    status, data = _post(server, "/api/render", {"markdown": "<script>alert(1)</script> **bold**"})
    assert status == 200
    assert "<script>" not in data["html"]
    assert "<strong>bold</strong>" in data["html"]


def test_analyze_run_lifecycle_and_sse(server):
    status, started = _post(
        server,
        "/api/analyze",
        {
            "ticker": "600519",
            "analysis_date": "2026-01-03",
            "provider": "openai",
            "analysts": ["market"],
        },
    )
    assert status == 200
    run_id = started["id"]

    status, snapshot = _get(server, f"/api/analyze/{run_id}")
    assert status == 200
    assert snapshot["status"] in ("pending", "running", "done")

    # SSE stream: collect data: lines until end-of-stream.
    events = []
    with urllib.request.urlopen(f"{server}/api/analyze/{run_id}/events", timeout=20) as resp:
        assert resp.headers.get("Content-Type", "").startswith("text/event-stream")
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    types = [ev["type"] for ev in events]
    assert "meta" in types
    assert "chunk" in types
    assert "done" in types
    assert types[-1] == "end"

    done = next(ev for ev in events if ev["type"] == "done")
    assert done["signal"] == "BUY"
    assert done["report_path"].endswith("complete_report.md")

    # Chunks carry the report deltas.
    chunk = next(ev for ev in events if ev["type"] == "chunk")
    assert "market_report" in chunk["data"]


def test_analyze_rejects_missing_fields(server):
    status, data = _post(server, "/api/analyze", {"analysis_date": "2026-01-03"})
    assert status == 400
    assert "ticker" in data["error"]


def test_reports_listing_and_rendering(server):
    # Run one analysis so a report tree lands on disk.
    _post(server, "/api/analyze", {"ticker": "600519", "analysis_date": "2026-01-03"})
    # Give the background thread a moment to finish writing the report.
    deadline = 15
    while deadline > 0:
        status, data = _get(server, "/api/reports")
        if data["reports"]:
            break
        import time

        time.sleep(0.1)
        deadline -= 0.1
    assert data["reports"], "expected a saved report"

    entry = data["reports"][0]
    status, rendered = _get(
        server,
        "/api/reports/data?ticker={}&date={}&file=complete_report.md".format(
            urllib.parse.quote(entry["ticker"]), urllib.parse.quote(entry["date"])
        ),
    )
    assert status == 200
    assert "<h1>Market</h1>" in rendered["html"] or "Market" in rendered["html"]


def test_report_data_rejects_path_traversal(server):
    status, data = _get(
        server, "/api/reports/data?ticker=600519&date=2026-01-03&file=..%2F..%2Fetc%2Fpasswd"
    )
    assert status == 400
    assert "unsupported" in data["error"]

    # A hostile ticker must not escape the results dir either.
    status, data = _get(
        server, "/api/reports/data?ticker=..%2F..&date=x&file=complete_report.md"
    )
    assert status in (400, 404)
