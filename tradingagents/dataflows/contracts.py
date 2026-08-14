from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import pandas as pd

DataSemantic = Literal[
    "ohlcv",
    "nav",
    "fund_profile",
    "company_profile",
    "news",
    "notice",
]

DataContractOverall = Literal["not_checked", "pass", "warning", "fail"]


@dataclass(frozen=True)
class SourceMeta:
    """Metadata needed to judge whether a vendor payload is safe to use."""

    vendor: str
    source: str
    symbol: str
    semantic: DataSemantic
    as_of: str | None = None
    retrieved_at: str | None = None


@dataclass(frozen=True)
class DataNotice:
    """Structured diagnostic for missing, stale, or schema-drifted data."""

    code: str
    message: str
    source: str | None = None
    detail: str | None = None
    severity: Literal["info", "warning", "error"] = "warning"


@dataclass(frozen=True)
class DataResult:
    """Structured dataflow result with explicit source, semantic, and notices."""

    meta: SourceMeta
    payload: pd.DataFrame | str | list[Any] | None = None
    notices: tuple[DataNotice, ...] = ()
    ok: bool = True
    stale: bool = False
    missing_reason: str | None = None
    error_type: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        if isinstance(self.payload, pd.DataFrame):
            return len(self.payload)
        if isinstance(self.payload, list):
            return len(self.payload)
        if self.payload is None:
            return 0
        return 1

    def with_text(self, text: str) -> DataResult:
        return DataResult(
            meta=self.meta,
            payload=self.payload,
            notices=self.notices,
            ok=self.ok,
            stale=self.stale,
            missing_reason=self.missing_reason,
            error_type=self.error_type,
            text=text,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ContractGateResult:
    """Validation outcome for a structured data contract."""

    ok: bool
    result: DataResult
    failures: tuple[DataNotice, ...] = ()
    warnings: tuple[DataNotice, ...] = ()
    expected_semantic: DataSemantic | None = None

    @property
    def notices(self) -> tuple[DataNotice, ...]:
        return self.failures + self.warnings


def data_notice(
    code: str,
    message: str,
    *,
    source: str | None = None,
    detail: str | None = None,
    severity: Literal["info", "warning", "error"] = "warning",
) -> DataNotice:
    return DataNotice(
        code=code,
        message=message,
        source=source,
        detail=detail,
        severity=severity,
    )


def render_notices(notices: tuple[DataNotice, ...] | list[DataNotice]) -> str:
    if not notices:
        return ""
    lines = ["", "## Data Notices"]
    for notice in notices:
        source = f" source={notice.source};" if notice.source else ""
        detail = f" detail={notice.detail}" if notice.detail else ""
        lines.append(
            f"- {notice.severity.upper()} {notice.code}:{source} {notice.message}{detail}".strip()
        )
    return "\n".join(lines)


def _gate_strictness() -> str:
    """Read the contract-gate strictness from the current dataflow config.

    ``strict`` (the default) is fail-closed: stale data is a hard failure.
    ``lenient`` is for operators who accept degraded freshness: ``stale``
    downgrades to a warning while look-ahead (``future_data``) and
    ``schema_drift`` stay hard failures. Read lazily at validation time so the
    module stays import-light.
    """
    from .config import get_config

    configured = get_config().get("data_contract_gate") or "strict"
    return "strict" if configured in ("strict", True) else "lenient"


def validate_data_result(
    result: DataResult,
    *,
    analysis_date: str | date | None = None,
    max_staleness_days: int | None = None,
    expected_semantic: DataSemantic | None = None,
    forbid_schema_drift: bool = True,
    allow_missing: bool = False,
    stale_fails: bool = True,
) -> ContractGateResult:
    """Fail closed when a structured data result is unsafe for factual claims.

    In ``lenient`` gate mode (config key ``data_contract_gate`` / env
    ``TRADINGAGENTS_DATA_CONTRACT_GATE``) ``stale_fails`` is overridden to
    ``False``: stale data then surfaces as a warning, while ``future_data``
    (look-ahead) and ``schema_drift`` remain hard failures.
    """
    if _gate_strictness() == "lenient" and not allow_missing:
        stale_fails = False
    failures: list[DataNotice] = []
    warnings: list[DataNotice] = []

    if not result.meta.source or not result.meta.symbol or not result.meta.semantic:
        failures.append(
            data_notice(
                "missing_contract_fields",
                "Data result is missing required source, symbol, or semantic contract fields.",
                severity="error",
            )
        )

    if expected_semantic is not None and result.meta.semantic != expected_semantic:
        failures.append(
            data_notice(
                "semantic_mismatch",
                f"Expected semantic={expected_semantic}, got semantic={result.meta.semantic}.",
                source=result.meta.source,
                severity="error",
            )
        )

    if result.error_type == "schema_drift" or any(n.code == "schema_drift" for n in result.notices):
        notice = data_notice(
            "schema_drift",
            "Vendor payload schema drift was detected before this result was produced.",
            source=result.meta.source,
            severity="error" if forbid_schema_drift else "warning",
        )
        if forbid_schema_drift:
            failures.append(notice)
        else:
            warnings.append(notice)

    if not result.ok and not allow_missing:
        failures.append(
            data_notice(
                result.missing_reason or result.error_type or "data_unavailable",
                "Data result is not OK and cannot be treated as reliable evidence.",
                source=result.meta.source,
                severity="error",
            )
        )

    if result.stale:
        notice = data_notice(
            "stale_data",
            "Data source marked this result as stale.",
            source=result.meta.source,
            severity="error" if stale_fails else "warning",
        )
        if stale_fails:
            failures.append(notice)
        else:
            warnings.append(notice)

    analysis_ts = _parse_date(analysis_date)
    as_of_ts = _parse_date(result.meta.as_of)
    if analysis_date is not None and result.meta.as_of and as_of_ts is None:
        failures.append(
            data_notice(
                "invalid_as_of",
                f"Could not parse as_of={result.meta.as_of!r} for contract validation.",
                source=result.meta.source,
                severity="error",
            )
        )
    if analysis_ts is not None and as_of_ts is not None:
        delta_days = (analysis_ts - as_of_ts).days
        if delta_days < 0:
            failures.append(
                data_notice(
                    "future_data",
                    f"Data as_of={result.meta.as_of} is after analysis_date={analysis_ts.date()}.",
                    source=result.meta.source,
                    severity="error",
                )
            )
        elif max_staleness_days is not None and delta_days > max_staleness_days:
            notice = data_notice(
                "stale_data",
                (
                    f"Data as_of={result.meta.as_of} is {delta_days} day(s) before "
                    f"analysis_date={analysis_ts.date()}, exceeding max_staleness_days={max_staleness_days}."
                ),
                source=result.meta.source,
                severity="error" if stale_fails else "warning",
            )
            if stale_fails:
                failures.append(notice)
            else:
                warnings.append(notice)

    if result.meta.semantic == "nav":
        warnings.append(
            data_notice(
                "nav_semantic",
                (
                    "This result is daily fund NAV, not exchange-traded OHLCV. Do not infer "
                    "intraday volume, exchange liquidity, or premium/discount unless another tool "
                    "explicitly provides those fields."
                ),
                source=result.meta.source,
                severity="warning",
            )
        )

    return ContractGateResult(
        ok=not failures,
        result=result,
        failures=tuple(failures),
        warnings=tuple(warnings),
        expected_semantic=expected_semantic,
    )


def render_contract_gate(gate: ContractGateResult, title: str = "Data Contract Gate") -> str:
    """Render a compact gate block for tool outputs and validation errors."""
    result = gate.result
    lines = [
        f"## {title}",
        "",
        f"- Status: {'PASS' if gate.ok else 'FAIL'}",
        f"- Source: {result.meta.source or 'n/a'}",
        f"- Symbol: {result.meta.symbol or 'n/a'}",
        f"- Semantic: {result.meta.semantic or 'n/a'}",
    ]
    if gate.expected_semantic:
        lines.append(f"- Expected semantic: {gate.expected_semantic}")
    lines.extend(
        [
            f"- As of: {result.meta.as_of or 'n/a'}",
            f"- Rows: {result.rows}",
        ]
    )
    if result.missing_reason:
        lines.append(f"- Missing reason: {result.missing_reason}")
    if result.error_type:
        lines.append(f"- Error type: {result.error_type}")

    if gate.failures:
        lines.extend(["", "### Failures"])
        lines.extend(_render_gate_notices(gate.failures))
    if gate.warnings:
        lines.extend(["", "### Warnings"])
        lines.extend(_render_gate_notices(gate.warnings))
    return "\n".join(lines)


def contract_gate_status(gate: ContractGateResult) -> dict[str, Any]:
    """Convert a gate result into a JSON-serializable status check."""
    return {
        "status": "pass" if gate.ok else "fail",
        "source": gate.result.meta.source,
        "symbol": gate.result.meta.symbol,
        "semantic": gate.result.meta.semantic,
        "expected_semantic": gate.expected_semantic,
        "as_of": gate.result.meta.as_of,
        "rows": gate.result.rows,
        "missing_reason": gate.result.missing_reason,
        "error_type": gate.result.error_type,
        "failures": [notice.code for notice in gate.failures],
        "warnings": [notice.code for notice in gate.warnings],
    }


def build_data_contract_status(checks: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build the shared state object for data-contract health."""
    normalized_checks = [_normalize_status_check(check) for check in (checks or ())]
    return {
        "overall": _overall_status(normalized_checks),
        "checks": normalized_checks,
    }


def merge_data_contract_status(
    existing: Mapping[str, Any] | None,
    checks: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge new gate checks into an existing state object, de-duplicating by content."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for check in list((existing or {}).get("checks") or []) + list(checks):
        normalized = _normalize_status_check(check)
        key = (
            normalized.get("source"),
            normalized.get("symbol"),
            normalized.get("semantic"),
            normalized.get("expected_semantic"),
            normalized.get("as_of"),
            normalized.get("status"),
            tuple(normalized.get("failures") or ()),
            tuple(normalized.get("warnings") or ()),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return build_data_contract_status(merged)


def merge_contract_status_channel(left: Any, right: Any) -> Any:
    """LangGraph channel reducer for the ``data_contract_status`` state field.

    Merges an incoming status update into the accumulated channel value and
    de-duplicates checks by content, so any node writing the field contributes
    checks instead of last-write-wins silently erasing earlier gate results.
    """
    if not left:
        return right
    if not right:
        return left
    if not isinstance(left, dict) or not isinstance(right, dict):
        return right
    return merge_data_contract_status(left, right.get("checks") or [])


def parse_contract_gate_status(text: str) -> list[dict[str, Any]]:
    """Parse rendered contract-gate Markdown back into compact status checks."""
    if not isinstance(text, str) or "Contract Gate" not in text:
        return []

    checks: list[dict[str, Any]] = []
    for block in _contract_gate_blocks(text):
        check: dict[str, Any] = {
            "status": "pass",
            "source": None,
            "symbol": None,
            "semantic": None,
            "expected_semantic": None,
            "as_of": None,
            "rows": None,
            "missing_reason": None,
            "error_type": None,
            "failures": [],
            "warnings": [],
        }
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("- Status:"):
                raw = stripped.split(":", 1)[1].strip().lower()
                check["status"] = "fail" if raw == "fail" else "pass"
            elif stripped.startswith("- Source:"):
                check["source"] = _clean_gate_value(stripped)
            elif stripped.startswith("- Symbol:"):
                check["symbol"] = _clean_gate_value(stripped)
            elif stripped.startswith("- Semantic:"):
                check["semantic"] = _clean_gate_value(stripped)
            elif stripped.startswith("- Expected semantic:"):
                check["expected_semantic"] = _clean_gate_value(stripped)
            elif stripped.startswith("- As of:"):
                check["as_of"] = _clean_gate_value(stripped)
            elif stripped.startswith("- Rows:"):
                rows = _clean_gate_value(stripped)
                check["rows"] = int(rows) if rows and rows.isdigit() else None
            elif stripped.startswith("- Missing reason:"):
                check["missing_reason"] = _clean_gate_value(stripped)
            elif stripped.startswith("- Error type:"):
                check["error_type"] = _clean_gate_value(stripped)
            elif stripped.startswith("- ERROR "):
                code = _diagnostic_code(stripped)
                if code:
                    check["failures"].append(code)
            elif stripped.startswith("- WARNING "):
                code = _diagnostic_code(stripped)
                if code:
                    check["warnings"].append(code)
        if check["failures"]:
            check["status"] = "fail"
        checks.append(_normalize_status_check(check))
    return checks


def collect_data_contract_status_from_messages(
    messages: Iterable[Any],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Collect rendered contract gates from LangChain messages into state."""
    checks: list[dict[str, Any]] = []
    for message in messages or ():
        content = _message_content_text(getattr(message, "content", None))
        checks.extend(parse_contract_gate_status(content))
    if not checks:
        return dict(existing) if existing else None
    return merge_data_contract_status(existing, checks)


def render_data_contract_status(
    status: Mapping[str, Any] | None,
    *,
    title: str = "## Data Reliability",
    compact: bool = False,
) -> str:
    """Render state-level data reliability for CLI panels and saved reports."""
    if not status:
        return ""
    checks = [_normalize_status_check(check) for check in (status.get("checks") or [])]
    if not checks:
        return ""

    overall = _overall_status(checks)
    label = {
        "pass": "PASS",
        "warning": "WARN",
        "fail": "FAIL",
        "not_checked": "NOT CHECKED",
    }[overall]
    failures = sum(len(check.get("failures") or ()) for check in checks)
    warnings = sum(len(check.get("warnings") or ()) for check in checks)
    lines = [
        title,
        "",
        f"- Overall: {label}",
        f"- Checks: {len(checks)}; failures={failures}; warnings={warnings}",
    ]
    if compact:
        for check in checks[-3:]:
            lines.append(f"- {_check_summary_line(check)}")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| Status | Source | Symbol | Semantic | As of | Rows | Diagnostics |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for check in checks:
        diagnostics = ", ".join((check.get("failures") or []) + (check.get("warnings") or [])) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    _check_display_status(check),
                    str(check.get("source") or "n/a"),
                    str(check.get("symbol") or "n/a"),
                    str(check.get("semantic") or "n/a"),
                    str(check.get("as_of") or "n/a"),
                    str(check.get("rows") if check.get("rows") is not None else "n/a"),
                    diagnostics,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_gate_notices(notices: tuple[DataNotice, ...]) -> list[str]:
    lines = []
    for notice in notices:
        source = f" source={notice.source};" if notice.source else ""
        detail = f" detail={notice.detail}" if notice.detail else ""
        lines.append(
            f"- {notice.severity.upper()} {notice.code}:{source} {notice.message}{detail}".strip()
        )
    return lines


def _normalize_status_check(check: Mapping[str, Any]) -> dict[str, Any]:
    failures = [str(code) for code in (check.get("failures") or []) if code]
    warnings = [str(code) for code in (check.get("warnings") or []) if code]
    status = str(check.get("status") or "pass").lower()
    if failures:
        status = "fail"
    elif status not in {"pass", "fail"}:
        status = "pass"
    rows = check.get("rows")
    return {
        "status": status,
        "source": _none_if_na(check.get("source")),
        "symbol": _none_if_na(check.get("symbol")),
        "semantic": _none_if_na(check.get("semantic")),
        "expected_semantic": _none_if_na(check.get("expected_semantic")),
        "as_of": _none_if_na(check.get("as_of")),
        "rows": int(rows) if isinstance(rows, str) and rows.isdigit() else rows,
        "missing_reason": _none_if_na(check.get("missing_reason")),
        "error_type": _none_if_na(check.get("error_type")),
        "failures": failures,
        "warnings": warnings,
    }


def _overall_status(checks: list[Mapping[str, Any]]) -> DataContractOverall:
    if not checks:
        return "not_checked"
    if any(check.get("status") == "fail" or check.get("failures") for check in checks):
        return "fail"
    if any(check.get("warnings") for check in checks):
        return "warning"
    return "pass"


def _contract_gate_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_block = False
    for line in lines:
        if re.match(r"^## .*\bContract Gate\b", line):
            if current:
                blocks.append("\n".join(current))
            current = [line]
            in_block = True
            continue
        if in_block and line.startswith("## ") and not line.startswith("### "):
            blocks.append("\n".join(current))
            current = []
            in_block = False
        if in_block:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _clean_gate_value(line: str) -> str | None:
    value = line.split(":", 1)[1].strip()
    return _none_if_na(value)


def _none_if_na(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "none", "null"}:
        return None
    return text


def _diagnostic_code(line: str) -> str | None:
    parts = line.split()
    if len(parts) < 3:
        return None
    return parts[2].rstrip(":")


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return str(content.get("text") or content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _check_display_status(check: Mapping[str, Any]) -> str:
    if check.get("status") == "fail" or check.get("failures"):
        return "FAIL"
    if check.get("warnings"):
        return "WARN"
    return "PASS"


def _check_summary_line(check: Mapping[str, Any]) -> str:
    diagnostics = ", ".join((check.get("failures") or []) + (check.get("warnings") or [])) or "none"
    return (
        f"{_check_display_status(check)} {check.get('semantic') or 'n/a'} "
        f"{check.get('source') or 'n/a'} as_of={check.get('as_of') or 'n/a'} "
        f"rows={check.get('rows') if check.get('rows') is not None else 'n/a'} "
        f"diagnostics={diagnostics}"
    )


def _parse_date(value: str | date | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()
