from contextvars import ContextVar
from copy import deepcopy

import tradingagents.default_config as default_config

# Per-context (thread) configuration. Each analysis thread (CLI main thread,
# web UI worker thread, embedder thread) carries its own copy, so concurrent
# runs never observe each other's vendor/language/contract-gate overrides.
# ``get_config``/``set_config`` keep the original merge semantics; only the
# storage moves from a process-wide global to a ContextVar.
_config: ContextVar[dict | None] = ContextVar(
    "tradingagents_dataflow_config", default=None
)


def initialize_config():
    """Initialize the current context's configuration with default values."""
    if _config.get() is None:
        _config.set(deepcopy(default_config.DEFAULT_CONFIG))


def set_config(config: dict):
    """Update the current context's configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.
    """
    initialize_config()
    incoming = deepcopy(config)
    current = _config.get()
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict) and value:
            current[key].update(value)
        else:
            current[key] = value


def get_config() -> dict:
    """Get the current context's configuration."""
    initialize_config()
    return deepcopy(_config.get())


def reset_config() -> None:
    """Replace the current context's configuration with a clean DEFAULT_CONFIG.

    ``set_config`` merges and never clears keys absent from the override, so
    tests (and embedders that rebuild a graph) use this for a hard reset.
    """
    _config.set(deepcopy(default_config.DEFAULT_CONFIG))
