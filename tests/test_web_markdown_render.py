"""Tests for the zero-dependency, XSS-safe Markdown renderer used by the web UI."""

import pytest

from webapp.markdown_render import render


def test_renders_headings_and_paragraphs():
    html = render("# Title\n\nSome *body* text.")
    assert "<h1>Title</h1>" in html
    assert "<p>Some <em>body</em> text.</p>" in html


def test_renders_tables():
    html = render("| A | B |\n|---|---|\n| 1 | 2 |\n")
    assert "<table>" in html
    assert "<th>A</th>" in html
    assert "<td>1</td>" in html


def test_renders_lists():
    html = render("- one\n- two\n\n1. first\n2. second")
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<ol><li>first</li><li>second</li></ol>" in html


def test_renders_fenced_code_block():
    html = render("```python\nprint('hi')\n```")
    assert "<pre><code>print(" in html
    assert "&lt;/code&gt;" not in html


def test_escapes_html_from_llm_output():
    html = render("Report <script>alert(1)</script> & more")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp; more" in html


def test_blocks_javascript_links():
    html = render("[click](javascript:alert(1))")
    assert "javascript:" not in html
    assert 'href="#"' in html


def test_allows_https_links():
    html = render("[repo](https://github.com/TauricResearch/TradingAgents)")
    assert 'href="https://github.com/TauricResearch/TradingAgents"' in html
    assert "rel=\"noopener\"" in html


def test_inline_code_not_reinterpreted_as_bold():
    html = render("Use `**not bold**` here.")
    assert "<code>**not bold**</code>" in html
    assert "<strong>" not in html


def test_renders_blockquote():
    html = render("> risk warning")
    assert "<blockquote>risk warning</blockquote>" in html


def test_renders_thematic_break():
    html = render("before\n\n---\n\nafter")
    assert "<hr>" in html


def test_empty_input_renders_empty():
    assert render(None) == ""
    assert render("") == ""


@pytest.mark.parametrize(
    "payload",
    [
        '<img src=x onerror=alert(1)>',
        "`<script>`",
        "[x](data:text/html;base64,PHNjcmlwdD4=)",
        "<b>bold tag</b>",
    ],
)
def test_hostile_inputs_never_emit_raw_html(payload):
    html = render(payload)
    assert "<script" not in html
    assert "<img" not in html
    assert "<b>" not in html
    assert "data:text" not in html
    assert "javascript:" not in html
