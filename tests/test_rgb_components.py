"""Tests for per-channel RGB encoding/decoding (normal and delta)."""

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_bytes_to_codes, convert_to_bitstream
from lzw_compression.core.decoder import (
    codes_to_image_grayscale,
    image_file_decoder_grayscale_differences,
)
from lzw_compression.core.encoder import (
    image_file_encoder_grayscale,
    image_file_encoder_grayscale_differences,
)
from lzw_compression.core.io import (
    open_bitstream_file_with_dimensions,
    open_color_image_file,
    save_image_file,
    write_bitstream_with_dimensions,
)

_RGB_IMAGE = np.array(
    [
        [[255, 100, 50], [200, 120, 75], [150, 140, 100]],
        [[220, 110, 60], [180, 130, 85], [140, 150, 110]],
        [[190, 120, 70], [160, 140, 95], [130, 160, 120]],
    ],
    dtype=np.uint8,
)

_SMOOTH_RGB = np.array(
    [
        [[255, 240, 220], [254, 240, 222], [253, 240, 224]],
        [[248, 235, 218], [247, 235, 220], [246, 235, 222]],
        [[240, 230, 215], [239, 230, 217], [238, 230, 219]],
    ],
    dtype=np.uint8,
)


def _channels(rgb: np.ndarray, tag: str) -> list[tuple[str, np.ndarray]]:
    """Save *rgb*, decompose into R/G/B, return (name, array) pairs."""
    path = f"samples/{tag}.png"
    save_image_file(rgb, path)
    r, g, b = open_color_image_file(path)
    return [("red", r), ("green", g), ("blue", b)]


@test
def test_rgb_normal_cycle():
    """Per-channel normal LZW: encode -> file -> decode -> verify."""
    h, w = _RGB_IMAGE.shape[:2]

    for name, ch in _channels(_RGB_IMAGE, "sample_color"):
        ch_path = f"samples/{name}_ch.png"
        lzw_path = f"samples/{name}_normal.lzw"
        save_image_file(ch, ch_path)

        codes = image_file_encoder_grayscale(ch_path)
        bs = convert_to_bitstream(codes)
        write_bitstream_with_dimensions(bs, lzw_path, h, w)

        read_bs, _, _ = open_bitstream_file_with_dimensions(lzw_path)
        decoded = codes_to_image_grayscale(
            convert_bytes_to_codes(read_bs),
            (h, w),
        )

        assert decoded.shape == ch.shape, f"{name}: shape mismatch"
        assert np.array_equal(decoded, ch), f"{name}: pixel mismatch"


@test
def test_rgb_difference_cycle():
    """Per-channel delta LZW: encode -> file -> decode -> verify."""
    h, w = _SMOOTH_RGB.shape[:2]

    for name, ch in _channels(_SMOOTH_RGB, "sample_color_smooth"):
        ch_path = f"samples/{name}_ch_diff.png"
        lzw_path = f"samples/{name}_diff.lzw"
        save_image_file(ch, ch_path)

        codes = image_file_encoder_grayscale_differences(ch_path)
        bs = convert_to_bitstream(codes)
        write_bitstream_with_dimensions(bs, lzw_path, h, w)

        decoded = image_file_decoder_grayscale_differences(lzw_path)

        assert decoded.shape == ch.shape, f"{name}: shape mismatch"
        assert np.array_equal(decoded, ch), f"{name}: pixel mismatch"
