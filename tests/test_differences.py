"""Tests for 2D delta (difference) encoding and decoding."""

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_to_bitstream
from lzw_compression.core.decoder import image_file_decoder_grayscale_differences
from lzw_compression.core.encoder import image_file_encoder_grayscale_differences
from lzw_compression.core.io import save_image_file, write_bitstream_with_dimensions


def _round_trip(image: np.ndarray, tag: str) -> None:
    """Encode *image* with 2D deltas, write to .lzw, decode, and assert equality."""
    img_path = f"samples/{tag}.png"
    lzw_path = f"samples/{tag}.lzw"
    save_image_file(image, img_path)

    codes = image_file_encoder_grayscale_differences(img_path)
    h, w = image.shape
    write_bitstream_with_dimensions(convert_to_bitstream(codes), lzw_path, h, w)

    decoded = image_file_decoder_grayscale_differences(lzw_path)
    assert decoded.shape == image.shape, f"{tag}: shape mismatch"
    assert np.array_equal(decoded, image), f"{tag}: pixel mismatch"


@test
def test_gradient_image():
    """Smooth gradient — ideal for delta encoding."""
    _round_trip(
        np.array(
            [
                [50, 75, 100, 125, 150],
                [75, 100, 125, 150, 175],
                [100, 125, 150, 175, 200],
                [125, 150, 175, 200, 225],
            ],
            dtype=np.uint8,
        ),
        "gradient_image",
    )


@test
def test_mixed_pattern():
    """Flat areas, descending columns, mixed gradients."""
    _round_trip(
        np.array(
            [
                [100, 105, 110, 115, 120],
                [50, 100, 150, 120, 90],
                [25, 80, 160, 140, 100],
                [10, 90, 170, 150, 110],
            ],
            dtype=np.uint8,
        ),
        "mixed_pattern",
    )
