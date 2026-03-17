import sys

import numpy as np

from lzw_compression.core.io import open_image_file, open_text_file

# Maximum code value for 12-bit encoding (2^12 = 4096, indices 0-4095)
MAX_CODE = 4096


def _lzw_encode_symbol_stream(symbol_stream: list[str]) -> list[int]:
    """Encode a symbol stream using the project's 12-bit LZW variant."""
    dictionary: dict[str, int] = {chr(i): i for i in range(256)}
    next_free_code: int = 256
    result: list[int] = []

    current_string: str = ""
    for symbol in symbol_stream:
        combined_string: str = current_string + symbol

        if combined_string in dictionary:
            current_string = combined_string
        else:
            result.append(dictionary[current_string])
            if next_free_code < MAX_CODE:
                dictionary[combined_string] = next_free_code
                next_free_code += 1
            current_string = symbol

    if current_string:
        result.append(dictionary[current_string])

    return result


def _to_symbol_stream_from_uint8(values: np.ndarray) -> list[str]:
    """Convert uint8-compatible values to single-byte character symbols."""
    return [chr(int(value)) for value in values]


def text_file_encoder(file_path: str) -> list[int]:
    """Encodes a text file using the LZW compression algorithm.

    Args:
        file_path (str): The path to the text file to be encoded.

    Returns:
        list[int]: A list of integers representing the LZW encoded output.
    """
    try:
        content: str = open_text_file(file_path)
        return _lzw_encode_symbol_stream(list(content))
    except Exception as e:
        print(f"An error occurred during encoding text file '{file_path}': {e}")
        sys.exit(1)


def encode_grayscale_array_lzw(image_array: np.ndarray) -> list[int]:
    """Encode a grayscale image array with LZW.

    Args:
        image_array (np.ndarray): 2D grayscale image array.

    Returns:
        list[int]: LZW output codes.
    """
    try:
        pixel_values = image_array.flatten()
        symbols = _to_symbol_stream_from_uint8(pixel_values)
        return _lzw_encode_symbol_stream(symbols)
    except Exception as e:
        print(f"An error occurred during encoding grayscale image array: {e}")
        sys.exit(1)


def image_file_encoder_grayscale(file_path: str) -> list[int]:
    """Encodes a grayscale image file using the LZW compression algorithm.

    Args:
        file_path (str): The path to the grayscale image file to be encoded.

    Returns:
        list[int]: A list of integers representing the LZW encoded output.
    """
    try:
        image_array = open_image_file(file_path)
        return encode_grayscale_array_lzw(image_array)
    except Exception as e:
        print(f"An error occurred during encoding image file '{file_path}': {e}")
        sys.exit(1)


def image_array_encoder_grayscale(image_array: np.ndarray) -> list[int]:
    """Encodes a grayscale image array using the LZW compression algorithm.

    Args:
        image_array (np.ndarray): 2D grayscale image array.

    Returns:
        list[int]: A list of integers representing the LZW encoded output.
    """
    return encode_grayscale_array_lzw(image_array)


def compute_left_top_differences_2d(arr: np.ndarray) -> np.ndarray:
    """Compute 2D predictive deltas using left-neighbour and top-neighbour rules."""
    try:
        diff = np.zeros_like(arr, dtype=int)

        # First pixel: keep as is
        diff[0, 0] = arr[0, 0]

        # First row: row-wise differences (pixel - left neighbor)
        diff[0, 1:] = arr[0, 1:] - arr[0, :-1]

        # First column (except first row): column-wise differences (pixel - top neighbor)
        diff[1:, 0] = arr[1:, 0] - arr[:-1, 0]

        # Rest of image: row-wise differences (pixel - left neighbor)
        diff[1:, 1:] = arr[1:, 1:] - arr[1:, :-1]

        return diff
    except Exception as e:
        print(f"An error occurred while computing differences: {e}")
        sys.exit(1)


def image_file_compute_differences(arr: np.ndarray) -> np.ndarray:
    """Compute 2D differences: row-wise for all pixels, column-wise for first column.

    Args:
        arr (np.ndarray): 2D image array (height x width).

    Returns:
        np.ndarray: Array with differences:
            - [0,0]: Original pixel (no difference)
            - [0,1:]: Differences from left neighbor (row-wise)
            - [1:,0]: Differences from top neighbor (column-wise)
            - [1:,1:]: Differences from left neighbor (row-wise)
    """
    return compute_left_top_differences_2d(arr)


def encode_grayscale_array_lzw_with_differences(image_array: np.ndarray) -> list[int]:
    """Encode grayscale array by left/top 2D deltas followed by LZW."""
    try:
        diff_array = compute_left_top_differences_2d(image_array)

        # Offset differences to 0-255 range (to handle negative values)
        diff_array_offset: np.ndarray = ((diff_array.astype(int) + 128) % 256).astype(int)

        pixel_values = diff_array_offset.flatten()
        symbols = _to_symbol_stream_from_uint8(pixel_values)
        return _lzw_encode_symbol_stream(symbols)
    except Exception as e:
        print(f"An error occurred during encoding grayscale image array with differences: {e}")
        sys.exit(1)


def image_file_encoder_grayscale_differences(file_path: str) -> list[int]:
    """Encodes a grayscale image using 2D delta encoding + LZW compression.

    Computes differences:
    - Row-wise: Current pixel - left neighbor (all rows)
    - Column-wise: Current pixel - top neighbor (first column only)

    Args:
        file_path (str): The path to the grayscale image file.

    Returns:
        list[int]: List of LZW encoded codes from delta-encoded pixels.
    """
    try:
        image_array = open_image_file(file_path)
        return encode_grayscale_array_lzw_with_differences(image_array)
    except Exception as e:
        print(f"An error occurred during encoding image file with differences '{file_path}': {e}")
        sys.exit(1)


def image_array_encoder_grayscale_differences(image_array: np.ndarray) -> list[int]:
    """Encodes a grayscale image array using 2D delta encoding + LZW compression.

    Args:
        image_array (np.ndarray): 2D grayscale image array.

    Returns:
        list[int]: List of LZW encoded codes from delta-encoded pixels.
    """
    return encode_grayscale_array_lzw_with_differences(image_array)
