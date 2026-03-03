"""Tests for compression metrics (CR, CF, SS, entropy, avg code length)."""

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_to_bitstream
from lzw_compression.core.encoder import image_file_encoder_grayscale, text_file_encoder
from lzw_compression.core.io import (
    save_image_file,
    write_bitstream_to_text_file,
    write_bitstream_with_dimensions,
)
from lzw_compression.core.metrics import (
    calculate_average_code_length,
    calculate_compression_factor,
    calculate_compression_ratio,
    calculate_entropy,
    calculate_image_compression_metrics,
    calculate_space_saving,
    calculate_text_compression_metrics,
)

_SRC = "samples/short_text.csv"


def _compress_text(suffix: str = "") -> tuple[str, list[int], bytes]:
    """Encode the standard CSV sample and return (lzw_path, codes, bitstream)."""
    codes = text_file_encoder(_SRC)
    bs = convert_to_bitstream(codes)
    lzw = f"samples/short_text{suffix}.lzw"
    write_bitstream_to_text_file(bs, lzw)
    return lzw, codes, bs


# ── Individual metric functions ──────────────────────────────────────────


@test
def test_compression_ratio():
    """CR is between 0 and 1 for short text."""
    lzw, *_ = _compress_text("_cr")
    assert 0 <= calculate_compression_ratio(_SRC, lzw) <= 1


@test
def test_compression_factor():
    """CF is positive."""
    lzw, *_ = _compress_text("_cf")
    assert calculate_compression_factor(_SRC, lzw) > 0


@test
def test_space_saving():
    """SS is between 0 and 1."""
    lzw, *_ = _compress_text("_ss")
    assert 0 <= calculate_space_saving(_SRC, lzw) <= 1


@test
def test_entropy():
    """Entropy of a simple distribution is positive and <= 8 bits."""
    pixels = np.array([100, 100, 100, 100, 200, 200], dtype=np.uint8)
    ent = calculate_entropy(pixels)
    max_entropy = 8
    assert 0 < ent <= max_entropy


@test
def test_average_code_length():
    """Average code length is positive for a short code sequence."""
    codes = [65, 66, 67, 68, 256, 257]
    assert calculate_average_code_length(convert_to_bitstream(codes), codes) > 0


# ── Aggregate metric helpers ─────────────────────────────────────────────


@test
def test_text_compression_metrics():
    """All expected keys present; values in valid ranges."""
    lzw, *_ = _compress_text("_metrics")
    m = calculate_text_compression_metrics(_SRC, lzw)

    for key in (
        "original_size",
        "compressed_size",
        "compression_ratio",
        "compression_factor",
        "space_saving_percent",
    ):
        assert key in m

    max_pct = 100
    assert m["compression_factor"] > 0
    assert 0 <= m["space_saving_percent"] <= max_pct


@test
def test_image_compression_metrics():
    """Image metrics contain all keys with sane value ranges."""
    img = np.array(
        [[100, 150, 200], [50, 100, 150], [25, 75, 125]],
        dtype=np.uint8,
    )
    img_path = "samples/test_image_metrics.png"
    lzw_path = "samples/test_image_metrics.lzw"
    save_image_file(img, img_path)

    codes = image_file_encoder_grayscale(img_path)
    bs = convert_to_bitstream(codes)
    h, w = img.shape
    write_bitstream_with_dimensions(bs, lzw_path, h, w)

    m = calculate_image_compression_metrics(
        img_path,
        lzw_path,
        img.flatten(),
        codes,
        bs,
    )

    for key in (
        "original_size",
        "compressed_size",
        "entropy",
        "average_code_length",
        "compression_ratio",
        "compression_factor",
        "space_saving_percent",
    ):
        assert key in m

    max_entropy = 8
    max_bits = 16
    assert 0 <= m["entropy"] <= max_entropy
    assert 0 < m["average_code_length"] <= max_bits
