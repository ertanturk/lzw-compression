"""Tests for left/top 2D differences and delta-LZW round-trips."""

from pathlib import Path

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_to_bitstream
from lzw_compression.core.decoder import (
	codes_to_image_grayscale_differences,
	image_file_decoder_grayscale_differences,
)
from lzw_compression.core.encoder import (
	compute_left_top_differences_2d,
	encode_grayscale_array_lzw_with_differences,
)
from lzw_compression.core.io import save_image_file, write_bitstream_with_dimensions

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _sample(name: str) -> str:
	return str(_SAMPLES_DIR / name)


def _assert_delta_roundtrip(image: np.ndarray, tag: str) -> None:
	"""Encode with differences and decode back; assert exact equality."""
	lzw_path = _sample(f"{tag}.lzw")
	codes = encode_grayscale_array_lzw_with_differences(image)
	h, w = image.shape
	write_bitstream_with_dimensions(convert_to_bitstream(codes), lzw_path, h, w)

	decoded = image_file_decoder_grayscale_differences(lzw_path)
	assert decoded.shape == image.shape
	assert np.array_equal(decoded, image)


@test
def test_compute_left_top_differences_expected_values() -> None:
	"""Verify exact difference transform semantics on a small fixed matrix."""
	arr = np.array(
		[
			[10, 13, 15],
			[20, 24, 30],
			[25, 26, 40],
		],
		dtype=np.uint8,
	)

	got = compute_left_top_differences_2d(arr)
	expected = np.array(
		[
			[10, 3, 2],
			[10, 4, 6],
			[5, 1, 14],
		],
		dtype=int,
	)
	assert np.array_equal(got, expected)


@test
def test_codes_to_image_grayscale_differences_direct_cycle() -> None:
	"""Direct encode/decode cycle without filesystem helpers."""
	image = np.array(
		[
			[50, 75, 100, 125],
			[55, 80, 105, 130],
			[60, 85, 110, 135],
		],
		dtype=np.uint8,
	)
	codes = encode_grayscale_array_lzw_with_differences(image)
	decoded = codes_to_image_grayscale_differences(codes, image.shape)
	assert np.array_equal(decoded, image)


@test
def test_difference_lzw_file_roundtrip_gradient() -> None:
	"""Gradient image round-trip via .lzw file with dimensions."""
	image = np.array(
		[
			[50, 75, 100, 125, 150],
			[75, 100, 125, 150, 175],
			[100, 125, 150, 175, 200],
			[125, 150, 175, 200, 225],
		],
		dtype=np.uint8,
	)
	_assert_delta_roundtrip(image, "test_diff_gradient")


@test
def test_difference_lzw_file_roundtrip_mixed_pattern() -> None:
	"""Mixed local structure also reconstructs exactly."""
	image = np.array(
		[
			[100, 105, 110, 115, 120],
			[50, 100, 150, 120, 90],
			[25, 80, 160, 140, 100],
			[10, 90, 170, 150, 110],
		],
		dtype=np.uint8,
	)
	save_image_file(image, _sample("test_diff_source.png"))
	_assert_delta_roundtrip(image, "test_diff_mixed")
