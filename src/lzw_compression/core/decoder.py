import sys

import numpy as np
from numpy.typing import NDArray

from lzw_compression.core.bitstream import convert_bytes_to_codes
from lzw_compression.core.io import (
	open_bitstream_file,
	open_bitstream_file_with_dimensions,
)

# Maximum code value for 12-bit encoding (2^12 = 4096, indices 0-4095)
MAX_CODE = 4096


def text_file_decoder(file_path: str) -> list[int]:
	"""Decodes a bitstream file containing LZW encoded data back into a list of integer codes.

	Args:
	    file_path (str): The path to the bitstream file to be decoded.

	Returns:
	    list[int]: A list of integers representing the decoded LZW codes.

	"""
	try:
		# Open the bitstream file and read its content as bytes
		bitstream: bytes = open_bitstream_file(file_path)
		# Convert the byte stream back into a list of integer codes
		codes: list[int] = convert_bytes_to_codes(bitstream)
		return codes
	except Exception as e:
		print(
			f"An error occurred during decoding bitstream file '{file_path}': {e}",
		)
		sys.exit(1)


def codes_to_text(codes: list[int]) -> str:
	"""Decodes a list of LZW codes back to the original text.

	Args:
	    codes (list[int]): The list of LZW codes to decode.

	Returns:
	    str: The decoded text string.

	"""
	try:
		# Initialize reverse dictionary with ASCII characters
		dictionary: dict[int, str] = {i: chr(i) for i in range(256)}
		next_free_code: int = 256

		if not codes:
			return ""

		# Start with the first code (must be in ASCII range)
		previous_entry: str = dictionary[codes[0]]
		result: str = previous_entry

		# Process remaining codes
		for code in codes[1:]:
			if code in dictionary:
				entry: str = dictionary[code]
			elif code == next_free_code:
				# Handle the special case where code is the next code to be added
				entry = previous_entry + previous_entry[0]
			else:
				print(f"Invalid code: {code}")
				sys.exit(1)

			result += entry

			# Add new entry to dictionary with previous phrase + first char of current
			if next_free_code < MAX_CODE:  # Only add if within 12-bit limit
				dictionary[next_free_code] = previous_entry + entry[0]
				next_free_code += 1

			previous_entry = entry

		return result
	except Exception as e:
		print(f"An error occurred during decoding: {e}")
		sys.exit(1)


def codes_to_image_grayscale(
	codes: list[int], image_shape: tuple[int, int],
) -> np.ndarray:
	"""Decodes a list of LZW codes back to a grayscale image represented as a NumPy array.

	Args:
	    codes (list[int]): The list of LZW codes to decode.
	    image_shape (tuple[int, int]): The shape of the output image (height, width).

	Returns:
	    np.ndarray: The decoded grayscale image as a NumPy array.

	"""
	try:
		# Initialize reverse dictionary with ASCII characters
		dictionary: dict[int, str] = {i: chr(i) for i in range(256)}
		next_free_code: int = 256

		if not codes:
			return np.zeros(image_shape, dtype=np.uint8)

		# Start with the first code (must be in ASCII range)
		previous_entry: str = dictionary[codes[0]]
		result: str = previous_entry

		# Process remaining codes
		for code in codes[1:]:
			if code in dictionary:
				entry: str = dictionary[code]
			elif code == next_free_code:
				# Handle the special case where code is the next code to be added
				entry = previous_entry + previous_entry[0]
			else:
				print(f"Invalid code: {code}")
				sys.exit(1)

			result += entry

			# Add new entry to dictionary with previous phrase + first char of current
			if next_free_code < MAX_CODE:  # Only add if within 12-bit limit
				dictionary[next_free_code] = previous_entry + entry[0]
				next_free_code += 1

			previous_entry = entry

		# Convert the resulting string back to a NumPy array
		# and reshape it to the original image dimensions
		image_array = np.frombuffer(result.encode("latin-1"), dtype=np.uint8)
		return image_array.reshape(image_shape)
	except Exception as e:
		print(f"An error occurred during decoding image: {e}")
		sys.exit(1)


def image_file_decoder_grayscale(file_path: str) -> np.ndarray:
	"""Decodes a .lzw file containing LZW encoded grayscale image data.

	Args:
	    file_path (str): The path to the .lzw file to be decoded.

	Returns:
	    np.ndarray: The decoded grayscale image as a NumPy array.

	"""
	try:
		# Open the bitstream file with embedded dimensions
		bitstream, height, width = open_bitstream_file_with_dimensions(
			file_path,
		)
		image_shape = (height, width)

		# Convert the byte stream back into a list of integer codes
		codes: list[int] = convert_bytes_to_codes(bitstream)

		# Decode codes to image
		return codes_to_image_grayscale(codes, image_shape)
	except Exception as e:
		print(
			f"An error occurred during decoding image file '{file_path}': {e}",
		)
		sys.exit(1)


def codes_to_image_grayscale_differences(
	codes: list[int], image_shape: tuple[int, int],
) -> np.ndarray:
	"""Decodes LZW codes back to original image using 2D delta decoding.

	Args:
	    codes (list[int]): The list of LZW codes representing delta-encoded pixels.
	    image_shape (tuple[int, int]): The shape of the output image (height, width).

	Returns:
	    np.ndarray: The decoded grayscale image as a NumPy array.

	"""
	try:
		# Decode LZW codes to get offset delta values
		delta_pixels_offset = codes_to_image_grayscale(codes, image_shape)

		# Remove the +128 offset to get raw differences
		delta_pixels: NDArray[np.int16] = (
			delta_pixels_offset.astype(np.int16) - 128
		)

		# Initialize reconstructed image
		image: NDArray[np.uint8] = np.zeros(image_shape, dtype=np.uint8)

		# [0,0]: First pixel (original, not a difference)
		image[0, 0] = delta_pixels[0, 0]

		# [0,1:]: Row-wise differences in first row
		for col in range(1, image_shape[1]):
			image[0, col] = (image[0, col - 1] + delta_pixels[0, col]) % 256

		# [1:,0]: Column-wise differences in first column
		for row in range(1, image_shape[0]):
			image[row, 0] = (image[row - 1, 0] + delta_pixels[row, 0]) % 256

		# [1:,1:]: Row-wise differences in rest of image
		for row in range(1, image_shape[0]):
			for col in range(1, image_shape[1]):
				image[row, col] = (
					image[row, col - 1] + delta_pixels[row, col]
				) % 256

		return image.astype(np.uint8)
	except Exception as e:
		print(f"An error occurred during decoding delta image: {e}")
		sys.exit(1)


def image_file_decoder_grayscale_differences(file_path: str) -> np.ndarray:
	"""Decodes a .lzw file containing 2D delta-encoded LZW compressed image data.

	Args:
	    file_path (str): The path to the .lzw file to be decoded.

	Returns:
	    np.ndarray: The decoded grayscale image as a NumPy array.

	"""
	try:
		# Open the bitstream file with embedded dimensions
		bitstream, height, width = open_bitstream_file_with_dimensions(
			file_path,
		)
		image_shape = (height, width)

		# Convert the byte stream back into a list of integer codes
		codes: list[int] = convert_bytes_to_codes(bitstream)

		# Decode codes to 2D difference image, then reverse deltas to get original
		return codes_to_image_grayscale_differences(codes, image_shape)
	except Exception as e:
		print(
			f"An error occurred during decoding delta image file '{file_path}': {e}",
		)
		sys.exit(1)
