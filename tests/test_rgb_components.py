"""RGB flow tests: split channels, compress independently, store, decode, merge."""

from pathlib import Path

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_bytes_to_codes, convert_to_bitstream
from lzw_compression.core.decoder import (
    codes_to_image_grayscale,
    codes_to_image_grayscale_differences,
)
from lzw_compression.core.encoder import (
    encode_grayscale_array_lzw,
    encode_grayscale_array_lzw_with_differences,
)
from lzw_compression.core.io import (
    open_color_bitstreams_with_dimensions,
    open_color_image_file,
    save_image_file,
    write_color_bitstreams_with_dimensions,
)

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _sample(name: str) -> str:
    return str(_SAMPLES_DIR / name)


def _encode_decode_rgb_via_container(rgb: np.ndarray, use_differences: bool) -> np.ndarray:
    """Apply project RGB container flow and return reconstructed RGB image."""
    src = _sample("test_rgb_source.png")
    out = _sample("test_rgb_container.lzw")
    save_image_file(rgb, src)

    red, green, blue = open_color_image_file(src)
    if use_differences:
        red_codes = encode_grayscale_array_lzw_with_differences(red)
        green_codes = encode_grayscale_array_lzw_with_differences(green)
        blue_codes = encode_grayscale_array_lzw_with_differences(blue)
    else:
        red_codes = encode_grayscale_array_lzw(red)
        green_codes = encode_grayscale_array_lzw(green)
        blue_codes = encode_grayscale_array_lzw(blue)

    red_bs = convert_to_bitstream(red_codes)
    green_bs = convert_to_bitstream(green_codes)
    blue_bs = convert_to_bitstream(blue_codes)

    h, w = rgb.shape[:2]
    write_color_bitstreams_with_dimensions(red_bs, green_bs, blue_bs, out, h, w)

    loaded_red_bs, loaded_green_bs, loaded_blue_bs, loaded_h, loaded_w = (
        open_color_bitstreams_with_dimensions(out)
    )
    assert loaded_h == h
    assert loaded_w == w

    red_loaded_codes = convert_bytes_to_codes(loaded_red_bs)
    green_loaded_codes = convert_bytes_to_codes(loaded_green_bs)
    blue_loaded_codes = convert_bytes_to_codes(loaded_blue_bs)

    if use_differences:
        red_decoded = codes_to_image_grayscale_differences(red_loaded_codes, (h, w))
        green_decoded = codes_to_image_grayscale_differences(green_loaded_codes, (h, w))
        blue_decoded = codes_to_image_grayscale_differences(blue_loaded_codes, (h, w))
    else:
        red_decoded = codes_to_image_grayscale(red_loaded_codes, (h, w))
        green_decoded = codes_to_image_grayscale(green_loaded_codes, (h, w))
        blue_decoded = codes_to_image_grayscale(blue_loaded_codes, (h, w))

    return np.stack((red_decoded, green_decoded, blue_decoded), axis=2).astype(np.uint8)


@test
def test_rgb_independent_channel_flow_without_differences():
    """RGB container round-trip using plain LZW per channel."""
    rgb = np.array(
        [
            [[255, 100, 50], [200, 120, 75], [150, 140, 100]],
            [[220, 110, 60], [180, 130, 85], [140, 150, 110]],
            [[190, 120, 70], [160, 140, 95], [130, 160, 120]],
        ],
        dtype=np.uint8,
    )
    decoded = _encode_decode_rgb_via_container(rgb, use_differences=False)
    assert np.array_equal(decoded, rgb)


@test
def test_rgb_independent_channel_flow_with_differences():
    """RGB container round-trip using delta+LZW per channel."""
    rgb = np.array(
        [
            [[255, 240, 220], [254, 240, 222], [253, 240, 224]],
            [[248, 235, 218], [247, 235, 220], [246, 235, 222]],
            [[240, 230, 215], [239, 230, 217], [238, 230, 219]],
        ],
        dtype=np.uint8,
    )
    decoded = _encode_decode_rgb_via_container(rgb, use_differences=True)
    assert np.array_equal(decoded, rgb)
