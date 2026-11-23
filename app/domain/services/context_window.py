"""Merge neighbouring passages into one context window without duplicating their overlap."""

from typing import Optional, Sequence

# (content, source_offset_start, source_offset_end), ordered by passage order.
PassageRow = tuple[str, Optional[int], Optional[int]]


def merge_passage_window(rows: Sequence[PassageRow]) -> str:
    """Join passages; content[i] sits at source offset start + i, so overlap is trimmed exactly."""
    text = ""
    previous_end: Optional[int] = None
    for content, start, end in rows:
        if (
            start is not None
            and end is not None
            and previous_end is not None
            and start < previous_end
        ):
            content = content[previous_end - start :]
            text += content
        else:
            text = f"{text} {content}" if text else content
        if end is not None:
            previous_end = end if previous_end is None else max(previous_end, end)
    return text
