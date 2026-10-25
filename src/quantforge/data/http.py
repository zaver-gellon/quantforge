"""Small JSON GET helper. Responses are untrusted; callers validate the shape."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from quantforge.data.base import DataError

_UA = "quantforge/0.1 (research; public-market-data)"


def get_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise DataError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise DataError(f"request failed for {url}: {exc.reason}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"response was not JSON for {url}") from exc
