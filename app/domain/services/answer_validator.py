"""Deterministic checks that figures in an answer come from the evidence it was built on."""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SOURCE_MARKER = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def _normalize(token: str) -> str | None:
    try:
        return format(Decimal(token.replace(",", "")).normalize(), "f")
    except InvalidOperation:
        return None


def _numbers(text: str) -> set[str]:
    return {value for token in _NUMBER.findall(text) if (value := _normalize(token))}


def unsupported_numbers(answer: str, evidence: Iterable[Any]) -> list[str]:
    """Return numbers stated in the answer that appear in none of the evidence values."""
    cleaned = _LIST_MARKER.sub("", _SOURCE_MARKER.sub("", answer))
    supported = _numbers(" ".join(str(item) for item in evidence))
    return sorted(_numbers(cleaned) - supported)
