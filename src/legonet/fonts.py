"""Portable font loading for generated LegoNet visualizations."""

from __future__ import annotations

from PIL import ImageFont


DEFAULT_VISUALIZATION_FONT = "DejaVuSans.ttf"


def load_visualization_font(
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load an open visualization font with Pillow's built-in fallback."""
    try:
        return ImageFont.truetype(DEFAULT_VISUALIZATION_FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)
