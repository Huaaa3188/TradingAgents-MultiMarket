"""Minimal, XSS-safe Markdown→HTML renderer for the web report view.

The analyst reports are Markdown produced by our own writers (ATX headings,
pipe tables, lists, inline emphasis, fenced code blocks, blockquotes). This
renderer covers exactly that surface — deliberately tiny, zero third-party
dependency — and HTML-escapes *before* applying markup, so LLM-produced text
can never inject tags or scripts.

Supported syntax:
  - ATX headings ``#`` .. ``######``
  - paragraphs (blank-line separated, single newlines become ``<br>``)
  - unordered (``-``/``*``) and ordered (``1.``) lists
  - pipe tables (header row + ``---`` separator row)
  - fenced code blocks (````` ``` ````)
  - blockquotes (``> ``)
  - thematic breaks (``---``)
  - inline: ``**bold**``, ``*italic*``, `` `code` ``, ``[text](url)``

Everything else renders as escaped plain text. URLs are restricted to
http(s) to keep ``javascript:``/``data:`` payloads inert.
"""

from __future__ import annotations

import re
from html import escape

_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_UNORDERED_RE = re.compile(r"^\s*([-*])\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_TABLE_SEPARATOR_RE = re.compile(r"^[\s|:\-]+$")


def _safe_url(url: str) -> str:
    """Allow only http(s) links; everything else degrades to '#'."""
    stripped = url.strip()
    if stripped.lower().startswith(("http://", "https://")):
        return escape(stripped, quote=True)
    return "#"


def _inline(text: str) -> str:
    """Apply inline markup on already-escaped text.

    Inline code is hoisted to placeholders first so its content is never
    re-interpreted by the emphasis/link rules.
    """
    placeholders: dict[str, str] = {}

    def _code_placeholder(match: re.Match) -> str:
        token = f"\x00{len(placeholders)}\x00"
        placeholders[token] = f"<code>{match.group(1)}</code>"
        return token

    text = _INLINE_CODE_RE.sub(_code_placeholder, text)
    text = _LINK_RE.sub(
        lambda m: f'<a href="{_safe_url(m.group(2))}" target="_blank" rel="noopener">{m.group(1)}</a>',
        text,
    )
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    for token, html in placeholders.items():
        text = text.replace(token, html)
    return text


def _render_table(header_cells: list[str], rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header_cells) + "</tr></thead>"
    body_rows = []
    for row in rows:
        cells = row + [""] * (len(header_cells) - len(row))  # pad short rows
        body_rows.append(
            "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells[: len(header_cells)]) + "</tr>"
        )
    return f"<table>{thead}<tbody>{''.join(body_rows)}</tbody></table>"


def render(markdown_text: str | None) -> str:
    """Convert a report's Markdown to safe HTML. Empty input renders empty."""
    if not markdown_text:
        return ""
    lines = markdown_text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fenced code block: content escaped verbatim, no inline interpretation.
        if _FENCE_RE.match(line):
            buf: list[str] = []
            i += 1
            while i < len(lines) and not _FENCE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1  # skip the closing fence (or EOF)
            out.append(f"<pre><code>{escape(chr(10).join(buf))}</code></pre>")
            continue

        # ATX heading.
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(escape(heading.group(2)))}</h{level}>")
            i += 1
            continue

        # Thematic break.
        if _RULE_RE.match(line):
            out.append("<hr>")
            i += 1
            continue

        # Blockquote (consecutive lines).
        if _QUOTE_RE.match(line):
            buf = []
            while i < len(lines):
                quote = _QUOTE_RE.match(lines[i])
                if not quote:
                    break
                buf.append(_inline(escape(quote.group(1))))
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue

        # Pipe table: header row followed by a separator row.
        if "|" in line and i + 1 < len(lines) and _is_table_separator(lines[i + 1]):
            header_cells = _split_row(line)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            out.append(_render_table(header_cells, rows))
            continue

        # Unordered list.
        unordered = _UNORDERED_RE.match(line)
        if unordered:
            buf = []
            while i < len(lines):
                item = _UNORDERED_RE.match(lines[i])
                if item:
                    buf.append(_inline(escape(item.group(2))))
                    i += 1
                    continue
                break
            out.append("<ul>" + "".join(f"<li>{item_html}</li>" for item_html in buf) + "</ul>")
            continue

        # Ordered list.
        ordered = _ORDERED_RE.match(line)
        if ordered:
            buf = []
            while i < len(lines):
                item = _ORDERED_RE.match(lines[i])
                if item:
                    buf.append(_inline(escape(item.group(2))))
                    i += 1
                    continue
                break
            out.append("<ol>" + "".join(f"<li>{item_html}</li>" for item_html in buf) + "</ol>")
            continue

        # Paragraph: consecutive non-blank lines that hit no special rule.
        if line.strip():
            buf = []
            while i < len(lines) and lines[i].strip():
                buf.append(_inline(escape(lines[i])))
                i += 1
            out.append("<p>" + "<br>".join(buf) + "</p>")
            continue

        # Blank line: skip.
        i += 1

    return "\n".join(out)


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and "-" in stripped and bool(_TABLE_SEPARATOR_RE.fullmatch(stripped))
