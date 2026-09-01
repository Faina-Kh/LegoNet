"""Tests for portable visualization font loading."""

from unittest import mock

from PIL import ImageFont

from legonet import fonts


def test_load_visualization_font_prefers_open_dejavu_font() -> None:
    """Normal installations use Pillow's open DejaVu Sans font."""
    expected = mock.Mock(spec=ImageFont.FreeTypeFont)
    with mock.patch.object(ImageFont, "truetype", return_value=expected) as truetype:
        result = fonts.load_visualization_font(18)

    assert result is expected
    truetype.assert_called_once_with(fonts.DEFAULT_VISUALIZATION_FONT, 18)


def test_load_visualization_font_falls_back_to_pillow_default() -> None:
    """A missing TrueType font cannot break visualization generation."""
    expected = mock.Mock(spec=ImageFont.ImageFont)
    with (
        mock.patch.object(ImageFont, "truetype", side_effect=OSError),
        mock.patch.object(ImageFont, "load_default", return_value=expected) as fallback,
    ):
        result = fonts.load_visualization_font(22)

    assert result is expected
    fallback.assert_called_once_with(size=22)
