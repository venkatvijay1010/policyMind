"""Render a PDF page to PNG with the cited text highlighted."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import settings

# Translucent amber fill with a darker outline stays readable on both text and scans.
_OUTLINE = (0.85, 0.45, 0.0)
_FILL = (1.0, 0.85, 0.15)
_FILL_OPACITY = 0.35


class PageRenderError(ValueError):
    """The requested page could not be rendered."""


@dataclass(frozen=True)
class RenderedPage:
    png: bytes
    # Top of the first highlight as a 0-1 fraction of the page height, if any.
    highlight_top: float | None


def is_valid_box(value: object) -> bool:
    """Return whether a stored bounding box is four finite numbers."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    )


def render_page_png(
    pdf_bytes: bytes,
    page_number: int,
    highlight_boxes: Sequence[Sequence[float]] = (),
    scale: float = 1.6,
) -> RenderedPage:
    """Render one 1-based page.

    PyMuPDF reports extracted boxes in unrotated page space and draws in that same space,
    so boxes are drawn as recorded; the rotation matrix maps them to where they appear on
    the rendered page, which is what the returned scroll position refers to.
    """
    import pymupdf

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PageRenderError("The source file is not a readable PDF") from exc

    with document:
        if document.needs_pass:
            raise PageRenderError("Encrypted PDFs cannot be displayed")
        if not 1 <= page_number <= document.page_count:
            raise PageRenderError("Page not found")
        try:
            page = document.load_page(page_number - 1)
            if page.rect.width * page.rect.height * scale * scale > settings.pdf_max_render_pixels:
                raise PageRenderError("The page is too large to display")
            top: float | None = None
            for box in highlight_boxes:
                rect = pymupdf.Rect(box).normalize()
                page.draw_rect(
                    rect,
                    color=_OUTLINE,
                    fill=_FILL,
                    fill_opacity=_FILL_OPACITY,
                    width=1,
                    overlay=True,
                )
                visible_top = (rect * page.rotation_matrix).y0
                top = visible_top if top is None else min(top, visible_top)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            ratio = None if top is None else min(max(top / page.rect.height, 0.0), 1.0)
            return RenderedPage(png=pixmap.tobytes("png"), highlight_top=ratio)
        except PageRenderError:
            raise
        except Exception as exc:
            raise PageRenderError("The page could not be rendered") from exc
