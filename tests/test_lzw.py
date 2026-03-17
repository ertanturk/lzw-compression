"""Core LZW tests: symbol encoding and end-to-end text/image cycles."""

from pathlib import Path

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_bytes_to_codes, convert_to_bitstream
from lzw_compression.core.decoder import codes_to_text, image_file_decoder_grayscale
from lzw_compression.core.encoder import encode_grayscale_array_lzw, text_file_encoder
from lzw_compression.core.io import (
    open_bitstream_file,
    open_text_file,
    save_image_file,
    write_bitstream_to_text_file,
    write_bitstream_with_dimensions,
)

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _sample(name: str) -> str:
    return str(_SAMPLES_DIR / name)


def _write_text(path: str, content: str) -> None:
    with open(path, "w") as file:
        file.write(content)


@test
def test_text_encoder_known_output_abracadabra():
    """Known sequence check for classic ABRACADABRA."""
    expected = [65, 66, 82, 65, 67, 65, 68, 256, 258]
    src = _sample("test_abracadabra.txt")
    _write_text(src, "ABRACADABRA")
    assert text_file_encoder(src) == expected


@test
def test_text_encoder_known_output_repeated_char():
    """Known sequence check for repeated character input."""
    src = _sample("test_repeated_char.txt")
    _write_text(src, "AAAA")
    assert text_file_encoder(src) == [65, 256, 65]


@test
def test_text_roundtrip_via_bitstream_and_decoder():
    """text -> codes -> bitstream -> codes -> text remains unchanged."""
    src = _sample("test_text_roundtrip_input.txt")
    out = _sample("test_text_roundtrip.lzw")

    _write_text(src, "Lorem Ipsum is simply dummy text")

    original = open_text_file(src)
    codes = text_file_encoder(src)
    bitstream = convert_to_bitstream(codes)
    write_bitstream_to_text_file(bitstream, out)

    reloaded_codes = convert_bytes_to_codes(open_bitstream_file(out))
    decoded_text = codes_to_text(reloaded_codes)

    assert reloaded_codes == codes
    assert decoded_text == original


@test
def test_grayscale_array_roundtrip_via_file_decoder():
    """array -> LZW -> file-with-dims -> decode reproduces exact pixels."""
    image = np.array(
        [[0, 64, 128, 192], [1, 65, 129, 193], [2, 66, 130, 194], [3, 67, 131, 195]],
        dtype=np.uint8,
    )
    image_path = _sample("test_lzw_gray.png")
    lzw_path = _sample("test_lzw_gray.lzw")

    save_image_file(image, image_path)

    codes = encode_grayscale_array_lzw(image)
    bitstream = convert_to_bitstream(codes)
    h, w = image.shape
    write_bitstream_with_dimensions(bitstream, lzw_path, h, w)

    decoded = image_file_decoder_grayscale(lzw_path)
    assert decoded.shape == image.shape
    assert np.array_equal(decoded, image)
