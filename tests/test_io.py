"""Tests for lzw_compression.core.io — file read/write operations."""

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.io import (
    open_color_image_file,
    open_text_file,
    save_image_file,
    write_bitstream_to_text_file,
)


@test
def test_open_text_file():
    """Read .csv and .txt sample files and verify content."""
    assert open_text_file("samples/short_text.csv") == "ABRACADABRA"
    assert open_text_file("samples/short_text.txt") == "Lorem Ipsum is simply dummy text"


@test
def test_open_text_file_not_found():
    """Missing file triggers sys.exit(1)."""
    try:
        open_text_file("samples/non_existing_file.txt")
    except SystemExit as exc:
        assert exc.code == 1


@test
def test_write_and_read_bitstream():
    """Write a bitstream to disk and read it back unchanged."""
    data = b"\x00\x01\x02\x03\x04\x05"
    path = "samples/test_bitstream.lzw"
    write_bitstream_to_text_file(data, path)

    with open(path, "rb") as fh:
        assert fh.read() == data


@test
def test_color_image_decomposition():
    """Save an RGB image, decompose into channels, verify shapes."""
    img = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    path = "samples/test_rgb_io.png"
    save_image_file(img, path)

    r, g, b = open_color_image_file(path)
    assert r.shape == (2, 2)
    assert np.array_equal(r, img[:, :, 0])
    assert np.array_equal(g, img[:, :, 1])
    assert np.array_equal(b, img[:, :, 2])
