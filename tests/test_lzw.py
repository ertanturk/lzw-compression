"""Tests for LZW encoder / decoder — text and image round-trips."""

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_bytes_to_codes, convert_to_bitstream
from lzw_compression.core.decoder import codes_to_text, image_file_decoder_grayscale
from lzw_compression.core.encoder import image_file_encoder_grayscale, text_file_encoder
from lzw_compression.core.io import (
    open_bitstream_file,
    open_text_file,
    save_image_file,
    write_bitstream_to_text_file,
    write_bitstream_with_dimensions,
)

# ── Text encoding correctness ───────────────────────────────────────────


@test
def test_single_character():
    """Single char produces its ASCII code."""
    assert text_file_encoder("samples/single_char.txt") == [65]


@test
def test_repeated_character():
    """Repeated chars build and reuse dictionary entries."""
    assert text_file_encoder("samples/repeated_char.txt") == [65, 256, 65]


@test
def test_pattern_repetition():
    """Repeating AB pattern reuses code 256."""
    assert text_file_encoder("samples/pattern_repeat.txt") == [65, 66, 256, 256]


@test
def test_no_repetition():
    """All unique chars — each maps to its ASCII code."""
    assert text_file_encoder("samples/no_repeat.txt") == [65, 66, 67, 68, 69, 70]


@test
def test_classic_abracadabra():
    """Classic LZW example (ABRACADABRA)."""
    assert text_file_encoder("samples/short_text.csv") == [
        65,
        66,
        82,
        65,
        67,
        65,
        68,
        256,
        258,
    ]


@test
def test_lorem_ipsum():
    """Short real-world text with spaces and varied patterns."""
    expected = [
        76,
        111,
        114,
        101,
        109,
        32,
        73,
        112,
        115,
        117,
        260,
        105,
        115,
        32,
        115,
        105,
        109,
        112,
        108,
        121,
        32,
        100,
        265,
        109,
        275,
        116,
        101,
        120,
        116,
    ]
    assert text_file_encoder("samples/short_text.txt") == expected


# ── Full round-trip cycles ───────────────────────────────────────────────


@test
def test_text_encode_decode_cycle():
    """Encode text -> bitstream -> file -> decode -> verify original."""
    src = "samples/short_text.csv"
    lzw = "samples/encoded.lzw"
    original = open_text_file(src)

    codes = text_file_encoder(src)
    write_bitstream_to_text_file(convert_to_bitstream(codes), lzw)

    decoded_codes = convert_bytes_to_codes(open_bitstream_file(lzw))
    assert decoded_codes == codes
    assert codes_to_text(decoded_codes) == original


@test
def test_image_encode_decode_cycle():
    """Encode grayscale image -> .lzw -> decode -> pixel-perfect match."""
    img = np.array(
        [[100, 150, 200, 100], [50, 100, 150, 50], [25, 75, 125, 25], [255, 200, 150, 100]],
        dtype=np.uint8,
    )
    img_path = "samples/test_cycle_image.png"
    lzw_path = "samples/test_cycle_image.lzw"
    save_image_file(img, img_path)

    codes = image_file_encoder_grayscale(img_path)
    h, w = img.shape
    write_bitstream_with_dimensions(convert_to_bitstream(codes), lzw_path, h, w)

    decoded = image_file_decoder_grayscale(lzw_path)
    assert decoded.shape == img.shape
    assert np.array_equal(decoded, img)
