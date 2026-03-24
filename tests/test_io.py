"""I/O tests for text, grayscale, and color container helpers."""

from pathlib import Path

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.io import (
	open_bitstream_file,
	open_bitstream_file_with_dimensions,
	open_color_bitstreams_with_dimensions,
	open_color_image_file,
	open_image_file,
	open_text_file,
	save_image_file,
	write_bitstream_to_text_file,
	write_bitstream_with_dimensions,
	write_color_bitstreams_with_dimensions,
)

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _sample(name: str) -> str:
	return str(_SAMPLES_DIR / name)


def _write_text(path: str, content: str) -> None:
	with open(path, "w") as file:
		file.write(content)


@test
def test_open_text_file_reads_known_samples() -> None:
	"""Text loader reads expected sample contents."""
	csv_path = _sample("test_io_text_a.txt")
	txt_path = _sample("test_io_text_b.txt")
	_write_text(csv_path, "ABRACADABRA")
	_write_text(txt_path, "Lorem Ipsum is simply dummy text")

	assert open_text_file(csv_path) == "ABRACADABRA"
	assert open_text_file(txt_path) == "Lorem Ipsum is simply dummy text"


@test
def test_open_text_file_missing_raises_system_exit() -> None:
	"""Missing file should terminate with exit code 1."""
	try:
		open_text_file(_sample("does_not_exist.txt"))
	except SystemExit as exc:
		assert exc.code == 1


@test
def test_bitstream_raw_write_and_open_roundtrip() -> None:
	"""Raw bitstream write/open path preserves bytes exactly."""
	payload = b"\x07\x10\x20\x30\x40"
	out = _sample("test_io_raw_bitstream.lzw")
	write_bitstream_to_text_file(payload, out)
	assert open_bitstream_file(out) == payload


@test
def test_bitstream_with_dimensions_roundtrip_header_and_payload() -> None:
	"""Dimensioned bitstream helper should preserve payload and dimensions."""
	payload = b"\x01\x02\x03\x04"
	out = _sample("test_io_dims.lzw")
	write_bitstream_with_dimensions(payload, out, height=11, width=17)

	loaded_payload, h, w = open_bitstream_file_with_dimensions(out)
	assert loaded_payload == payload
	assert h == 11
	assert w == 17


@test
def test_open_image_file_returns_numpy_array() -> None:
	"""Saved grayscale image can be loaded back as array with same pixels."""
	image = np.array([[0, 127], [200, 255]], dtype=np.uint8)
	path = _sample("test_io_gray.png")
	save_image_file(image, path)

	loaded = open_image_file(path)
	assert loaded.shape == image.shape
	assert np.array_equal(loaded, image)


@test
def test_open_color_image_file_splits_rgb_channels() -> None:
	"""Color loader should return independent R/G/B channel arrays."""
	rgb = np.array(
		[[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
		dtype=np.uint8,
	)
	path = _sample("test_io_rgb.png")
	save_image_file(rgb, path)

	red, green, blue = open_color_image_file(path)
	assert np.array_equal(red, rgb[:, :, 0])
	assert np.array_equal(green, rgb[:, :, 1])
	assert np.array_equal(blue, rgb[:, :, 2])


@test
def test_open_color_image_file_grayscale_returns_triplicate_channels() -> None:
	"""Grayscale input should map to identical R/G/B outputs by design."""
	gray = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
	path = _sample("test_io_gray_as_color.png")
	save_image_file(gray, path)

	red, green, blue = open_color_image_file(path)
	assert np.array_equal(red, gray)
	assert np.array_equal(green, gray)
	assert np.array_equal(blue, gray)


@test
def test_color_bitstream_container_roundtrip() -> None:
	"""Color container should preserve three channel payloads and dimensions."""
	out = _sample("test_io_color_container.lzw")
	red = b"\x01\x11\x21"
	green = b"\x02\x12"
	blue = b"\x03\x13\x23\x33"

	write_color_bitstreams_with_dimensions(
		red, green, blue, out, height=9, width=13
	)
	loaded_red, loaded_green, loaded_blue, h, w = (
		open_color_bitstreams_with_dimensions(out)
	)

	assert loaded_red == red
	assert loaded_green == green
	assert loaded_blue == blue
	assert h == 9
	assert w == 13
