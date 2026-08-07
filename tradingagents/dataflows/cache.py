from __future__ import annotations

import functools
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import suppress

from diskcache import Cache

from .config import get_config

_UNINITIALIZED_CACHE = object()
_CACHES: dict[str, object] = {}
_CACHE_STATS: dict[tuple[str, str], Counter] = {}
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def get_disk_cache(namespace: str):
    """Return the lazily initialized DiskCache for a dataflow namespace."""
    _validate_namespace(namespace)
    active_cache = _CACHES.get(namespace, _UNINITIALIZED_CACHE)
    if active_cache is _UNINITIALIZED_CACHE:
        cache_dir = os.path.join(get_config()["data_cache_dir"], namespace)
        try:
            active_cache = Cache(cache_dir)
        except Exception as exc:  # noqa: BLE001 - cache failures must not block data fetches
            print(f"[Warning] Failed to initialize DiskCache at {cache_dir}: {exc}", file=sys.stderr)
            active_cache = None
        _CACHES[namespace] = active_cache
    return active_cache


def set_disk_cache(namespace: str, cache_obj) -> None:
    """Inject a cache object for tests or controlled runtime overrides."""
    _validate_namespace(namespace)
    _CACHES[namespace] = cache_obj


def clear_disk_cache(namespace: str | None = None) -> None:
    """Forget cached Cache instances so future calls re-read current config."""
    if namespace is not None:
        _validate_namespace(namespace)
        cache_obj = _CACHES.pop(namespace, None)
        _close_cache(cache_obj)
        return

    cache_objects = list(_CACHES.values())
    _CACHES.clear()
    for cache_obj in cache_objects:
        _close_cache(cache_obj)


def invalidate_disk_cache_where(namespace: str, predicate) -> int:
    """Delete cached entries whose key satisfies ``predicate(key)``.

    Returns the number of entries removed. Used for targeted refresh (e.g.
    forcing a display-name re-fetch without dropping the whole namespace).
    Safe when the cache is disabled or unavailable: it is a no-op returning 0.
    """
    _validate_namespace(namespace)
    active_cache = get_disk_cache(namespace)
    if active_cache is None:
        return 0
    removed = 0
    for key in list(active_cache):
        if isinstance(key, str) and predicate(key):
            try:
                del active_cache[key]
                removed += 1
            except KeyError:
                pass
    return removed


def get_cache_stats(namespace: str | None = None) -> dict:
    """Return in-process cache counters for diagnostics and tests."""
    if namespace is not None:
        _validate_namespace(namespace)
        return {
            func_name: dict(counter)
            for (ns, func_name), counter in _CACHE_STATS.items()
            if ns == namespace
        }
    stats: dict[str, dict[str, dict[str, int]]] = {}
    for (ns, func_name), counter in _CACHE_STATS.items():
        stats.setdefault(ns, {})[func_name] = dict(counter)
    return stats


def reset_cache_stats(namespace: str | None = None) -> None:
    """Clear in-process cache counters."""
    if namespace is not None:
        _validate_namespace(namespace)
        for key in list(_CACHE_STATS):
            if key[0] == namespace:
                _CACHE_STATS.pop(key, None)
        return
    _CACHE_STATS.clear()


def disk_cache(namespace: str, expire: int = 14400) -> Callable:
    """Decorate dataflow fetchers with a shared fail-open DiskCache layer."""
    _validate_namespace(namespace)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not get_config().get("enable_data_cache", True):
                _record_cache_event(namespace, func.__name__, "disabled")
                return func(*args, **kwargs)

            active_cache = get_disk_cache(namespace)
            if active_cache is None:
                _record_cache_event(namespace, func.__name__, "unavailable")
                return func(*args, **kwargs)

            key = _cache_key(func, args, kwargs)
            try:
                cached_val = active_cache.get(key)
                if cached_val is not None:
                    _record_cache_event(namespace, func.__name__, "hits")
                    return cached_val
                _record_cache_event(namespace, func.__name__, "misses")
            except Exception as exc:  # noqa: BLE001 - fall through to live fetch
                _record_cache_event(namespace, func.__name__, "read_errors")
                print(f"[Warning] DiskCache read failure for {func.__name__}: {exc}", file=sys.stderr)

            val = func(*args, **kwargs)

            try:
                active_cache.set(key, val, expire=expire)
                _record_cache_event(namespace, func.__name__, "writes")
            except Exception as exc:  # noqa: BLE001 - fetched data is still usable
                _record_cache_event(namespace, func.__name__, "write_errors")
                print(f"[Warning] DiskCache write failure for {func.__name__}: {exc}", file=sys.stderr)
            return val

        return wrapper

    return decorator


def _cache_key(func, args, kwargs) -> str:
    return f"disk:{func.__name__}:{args}:{sorted(kwargs.items())}"


def _record_cache_event(namespace: str, func_name: str, event: str) -> None:
    _CACHE_STATS.setdefault((namespace, func_name), Counter())[event] += 1


def _validate_namespace(namespace: str) -> None:
    if not namespace or not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(f"Invalid dataflow cache namespace: {namespace!r}")


def _close_cache(cache_obj) -> None:
    close = getattr(cache_obj, "close", None)
    if callable(close):
        with suppress(Exception):
            close()
