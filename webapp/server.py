"""HTTP server + SSE endpoint for the TradingAgents web UI.

Standard-library only (``http.server`` + threads) so the web UI adds zero
dependencies. Routes:

  GET  /                            → index.html
  GET  /app.js  /style.css          → static assets
  GET  /api/config                  → providers / models / key status for the form
  GET  /api/detect?ticker=…         → market/instrument/asset detection
  POST /api/analyze                 → start an analysis; returns ``{"id": ...}``
  GET  /api/analyze/{id}/events     → SSE stream of graph chunks
  GET  /api/analyze/{id}            → run snapshot
  GET  /api/reports                 → list saved reports (ticker/date)
  GET  /api/reports/data            → render one report file to safe HTML

Run with ``python -m webapp.server --port 8000``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import urllib.parse
from contextlib import suppress
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from webapp.markdown_render import render as render_markdown
from webapp.tasks import AnalysisManager, safe_component

_STATIC_DIR = Path(__file__).parent / "static"
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}
# Report files allowed through /api/reports/data (defense-in-depth on top of
# the results_dir confinement).
_REPORT_FILE_RE = re.compile(
    r"^(?:complete_report|data_reliability)\.md$"
    r"|^(?:1_analysts|2_research|3_trading|4_risk|5_portfolio)/[A-Za-z0-9_.-]+\.md$"
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    manager: AnalysisManager = None  # set on the server instance
    server: ThreadingHTTPServer  # type: ignore[assignment]

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # -- helpers ----------------------------------------------------------

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: str) -> None:
        name = urllib.parse.unquote(path.lstrip("/"))
        target = (_STATIC_DIR / name).resolve()
        if not str(target).startswith(str(_STATIC_DIR.resolve())) or not target.is_file():
            self._send_text("not found", "text/plain; charset=utf-8", 404)
            return
        self._send_text(target.read_text(encoding="utf-8"), _MIME.get(target.suffix, "text/plain"))

    # -- GET --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/":
                self._send_static("index.html")
            elif path in ("/app.js", "/style.css"):
                self._send_static(path.lstrip("/"))
            elif path == "/api/config":
                self._send_json(self._api_config())
            elif path == "/api/detect":
                self._send_json(self._api_detect(query))
            elif path == "/api/reports":
                self._send_json(self._api_reports())
            elif path == "/api/reports/data":
                self._send_json(self._api_report_data(query))
            elif (match := re.fullmatch(r"/api/analyze/([A-Za-z0-9_-]+)/events", path)):
                self._stream_events(match.group(1))
            elif (match := re.fullmatch(r"/api/analyze/([A-Za-z0-9_-]+)", path)):
                run = self.manager.get(match.group(1))
                if run is None:
                    self._send_json({"error": "run not found"}, 404)
                else:
                    self._send_json(run)
            else:
                self._send_text("not found", "text/plain; charset=utf-8", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-stream

    # -- POST -------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        if parsed.path == "/api/analyze":
            try:
                run_id = self.manager.start(body)
                self._send_json({"id": run_id})
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
        elif parsed.path == "/api/render":
            markdown = body.get("markdown") or ""
            self._send_json({"html": render_markdown(markdown)})
        else:
            self._send_json({"error": "not found"}, 404)

    # -- API implementations ----------------------------------------------

    def _api_config(self) -> dict[str, Any]:
        from cli.utils import _llm_provider_table
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        from tradingagents.llm_clients.model_catalog import get_model_options

        providers = []
        for display, key, url in _llm_provider_table():
            env_var = get_api_key_env(key)
            has_key = env_var is None or bool(os.environ.get(env_var))
            providers.append(
                {"key": key, "name": display, "url": url, "env_var": env_var, "has_key": has_key}
            )

        models: dict[str, dict[str, list[dict[str, str]]]] = {}
        for _, key, _ in _llm_provider_table():
            entry: dict[str, list[dict[str, str]]] = {}
            for mode in ("quick", "deep"):
                try:
                    entry[mode] = [
                        {"display": display, "value": value}
                        for display, value in get_model_options(key, mode)
                    ]
                except KeyError:
                    entry[mode] = []
            models[key] = entry

        return {
            "providers": providers,
            "models": models,
            "defaults": {
                "llm_provider": DEFAULT_CONFIG["llm_provider"],
                "deep_think_llm": DEFAULT_CONFIG["deep_think_llm"],
                "quick_think_llm": DEFAULT_CONFIG["quick_think_llm"],
                "max_debate_rounds": DEFAULT_CONFIG["max_debate_rounds"],
                "max_risk_discuss_rounds": DEFAULT_CONFIG["max_risk_discuss_rounds"],
                "output_language": DEFAULT_CONFIG["output_language"],
                "checkpoint_enabled": DEFAULT_CONFIG["checkpoint_enabled"],
                "today": date.today().isoformat(),
            },
        }

    def _api_detect(self, query: dict[str, list[str]]) -> dict[str, Any]:
        from cli.utils import detect_asset_type
        from tradingagents.dataflows.instruments import (
            detect_instrument_type,
            detect_market_type,
            normalize_ticker_symbol,
        )

        ticker = normalize_ticker_symbol(query.get("ticker", [""])[0])
        return {
            "ticker": ticker,
            "asset_type": detect_asset_type(ticker).value,
            "instrument_type": detect_instrument_type(ticker).value,
            "market_type": detect_market_type(ticker).value,
        }

    def _api_reports(self) -> dict[str, Any]:
        root = Path(self.manager._results_dir)  # noqa: SLF001 - same package boundary
        entries = []
        if root.is_dir():
            for ticker_dir in sorted(root.iterdir()):
                if not ticker_dir.is_dir():
                    continue
                for date_dir in sorted(ticker_dir.iterdir(), reverse=True):
                    complete = date_dir / "complete_report.md"
                    if date_dir.is_dir() and complete.is_file():
                        entries.append(
                            {
                                "ticker": ticker_dir.name,
                                "date": date_dir.name,
                                "has_data_reliability": (date_dir / "data_reliability.md").is_file(),
                            }
                        )
        return {"reports": entries}

    def _api_report_data(self, query: dict[str, list[str]]) -> None:
        ticker = safe_component(query.get("ticker", [""])[0])
        date_dir = safe_component(query.get("date", [""])[0])
        file_name = query.get("file", ["complete_report.md"])[0]
        if not _REPORT_FILE_RE.fullmatch(file_name):
            self._send_json({"error": "unsupported file"}, 400)
            return

        root = (Path(self.manager._results_dir) / ticker / date_dir).resolve()  # noqa: SLF001
        target = (root / file_name).resolve()
        if not str(target).startswith(str(root)) or not target.is_file():
            self._send_json({"error": "report not found"}, 404)
            return

        markdown = target.read_text(encoding="utf-8")
        self._send_json(
            {
                "html": render_markdown(markdown),
                "markdown": markdown,
                "file": file_name,
                "ticker": ticker,
                "date": date_dir,
            }
        )

    # -- SSE ---------------------------------------------------------------

    def _stream_events(self, run_id: str) -> None:
        subscribed = self.manager.subscribe(run_id)
        if subscribed is None:
            self._send_json({"error": "run not found"}, 404)
            return
        subscriber, buffered = subscribed

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def _write(event: dict[str, Any] | None) -> bool:
            if event is None:
                self.wfile.write(b"data: {\"type\": \"end\"}\n\n")
                self.close_connection = True  # end-of-stream: close after flush
                return False
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            return True

        try:
            # Replay events buffered so far (a reconnecting EventSource catches
            # up). Each client consumes its own queue, so concurrent viewers
            # never steal events from one another.
            for event in buffered:
                if not _write(event):
                    return
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keep-alive comment
                    self.wfile.flush()
                    continue
                if not _write(event):
                    return
        finally:
            self.manager.unsubscribe(run_id, subscriber)
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.flush()


def build_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    graph_factory=None,
    results_dir: str | None = None,
) -> ThreadingHTTPServer:
    manager = AnalysisManager(graph_factory=graph_factory, results_dir=results_dir)
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    _Handler.manager = manager  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingAgents web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = build_server(host=args.host, port=args.port)
    print(f"TradingAgents web UI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
