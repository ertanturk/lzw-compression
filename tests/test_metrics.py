"""Tests for metric helpers (entropy, bit lengths, CR/CF/SS, aggregate reports)."""

from pathlib import Path

import numpy as np
from pyforge_test import test  # type: ignore

from lzw_compression.core.bitstream import convert_to_bitstream
from lzw_compression.core.encoder import encode_grayscale_array_lzw, text_file_encoder
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
	calculate_size_difference,
	calculate_space_saving,
	calculate_text_compression_metrics,
	calculate_total_code_bits,
)

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _sample(name: str) -> str:
	return str(_SAMPLES_DIR / name)


_TEXT_SOURCE = _sample("long_text.txt")


def _ensure_text_source() -> str:
	text = (
		"LZW metrics test input. "
		"This sentence is repeated. "
		"LZW metrics test input. "
		"This sentence is repeated. "
	)
	with open(_TEXT_SOURCE, "w") as file:
		file.write(text)
	return _TEXT_SOURCE


def _encode_text_to_lzw(tag: str) -> tuple[str, list[int], bytes]:
	"""Encode standard sample text and persist bitstream."""
	source = _ensure_text_source()
	codes = text_file_encoder(source)
	bitstream = convert_to_bitstream(codes)
	out = _sample(f"test_metrics_{tag}.lzw")
	write_bitstream_to_text_file(bitstream, out)
	return out, codes, bitstream


@test
def test_entropy_for_two_symbol_uniform_distribution_is_one_bit():
	"""P(0)=0.5, P(1)=0.5 => entropy should be exactly 1.0 bit."""
	values = np.array([0, 0, 1, 1], dtype=np.uint8)
	assert abs(calculate_entropy(values) - 1.0) < 1e-12


@test
def test_entropy_accepts_multidimensional_arrays():
	"""Entropy should use all symbols even when input is 2D."""
	values_2d = np.array([[1, 1], [2, 2]], dtype=np.uint8)
	assert abs(calculate_entropy(values_2d) - 1.0) < 1e-12


@test
def test_average_code_length_uses_symbol_count_when_provided():
	"""Average code length should be payload bits divided by source symbols."""
	codes = [65, 66, 67, 256]
	bitstream = convert_to_bitstream(codes)
	total_bits = calculate_total_code_bits(codes)
	source_symbols = 10
	got = calculate_average_code_length(
		bitstream, codes, symbol_count=source_symbols
	)
	assert abs(got - (total_bits / source_symbols)) < 1e-12


@test
def test_text_size_metrics_are_self_consistent():
	"""CR, CF, SS, and size-difference satisfy their algebraic relations."""
	out, _, _ = _encode_text_to_lzw("text_sizes")

	source = _ensure_text_source()
	cr = calculate_compression_ratio(source, out)
	cf = calculate_compression_factor(source, out)
	ss = calculate_space_saving(source, out)
	diff = calculate_size_difference(source, out)

	assert cr >= 0.0
	assert cf > 0.0
	assert abs((1.0 / cr) - cf) < 1e-9 if cr > 0 else cf > 0
	assert abs((1.0 - cr) - ss) < 1e-9
	assert isinstance(diff, int)


@test
def test_text_compression_metrics_payload_contains_expected_keys():
	"""Aggregate text metrics helper should provide all documented keys."""
	out, _, _ = _encode_text_to_lzw("text_payload")
	metrics = calculate_text_compression_metrics(_ensure_text_source(), out)

	for key in (
		"original_size",
		"compressed_size",
		"difference_bytes",
		"compression_ratio",
		"compression_factor",
		"space_saving_percent",
	):
		assert key in metrics


@test
def test_image_compression_metrics_payload_contains_expected_keys():
	"""Aggregate image metrics helper should provide all documented keys."""
	image = np.array(
		[[0, 32, 64], [96, 128, 160], [192, 224, 255]],
		dtype=np.uint8,
	)
	image_path = _sample("test_metrics_image.png")
	lzw_path = _sample("test_metrics_image.lzw")
	save_image_file(image, image_path)

	codes = encode_grayscale_array_lzw(image)
	bitstream = convert_to_bitstream(codes)
	h, w = image.shape
	write_bitstream_with_dimensions(bitstream, lzw_path, h, w)

	metrics = calculate_image_compression_metrics(
		image_path,
		lzw_path,
		image,
		codes,
		bitstream,
	)

	for key in (
		"original_size",
		"compressed_size",
		"difference_bytes",
		"entropy",
		"average_code_length",
		"compression_ratio",
		"compression_factor",
		"space_saving_percent",
	):
		assert key in metrics

	assert 0.0 <= metrics["entropy"] <= 8.0
	assert metrics["average_code_length"] >= 0.0
