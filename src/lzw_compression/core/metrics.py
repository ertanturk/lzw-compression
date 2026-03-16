import sys
from pathlib import Path
from typing import Any

import numpy as np


def calculate_file_size(file_path: str) -> int:
    """Calculate the size of a file in bytes.

    Args:
        file_path (str): The path to the file.

    Returns:
        int: The size of the file in bytes.
    """
    try:
        return Path(file_path).stat().st_size
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while calculating file size: {e}")
        sys.exit(1)


def calculate_compression_ratio(original_file_path: str, compressed_file_path: str) -> float:
    """Calculate the compression ratio (CR).

    CR = size_of_compressed_file / size_of_original_file

    Args:
        original_file_path (str): Path to the original file.
        compressed_file_path (str): Path to the compressed file.

    Returns:
        float: The compression ratio.
    """
    try:
        original_size = calculate_file_size(original_file_path)
        compressed_size = calculate_file_size(compressed_file_path)

        if original_size == 0:
            return 0.0

        return compressed_size / original_size
    except Exception as e:
        print(f"An error occurred while calculating compression ratio: {e}")
        sys.exit(1)


def calculate_compression_factor(original_file_path: str, compressed_file_path: str) -> float:
    """Calculate the compression factor (CF).

    CF = size_of_original_file / size_of_compressed_file

    Args:
        original_file_path (str): Path to the original file.
        compressed_file_path (str): Path to the compressed file.

    Returns:
        float: The compression factor.
    """
    try:
        original_size = calculate_file_size(original_file_path)
        compressed_size = calculate_file_size(compressed_file_path)

        if compressed_size == 0:
            print("Compressed file size is 0, cannot calculate compression factor.")
            return 0.0

        return original_size / compressed_size
    except Exception as e:
        print(f"An error occurred while calculating compression factor: {e}")
        sys.exit(1)


def calculate_space_saving(original_file_path: str, compressed_file_path: str) -> float:
    """Calculate the space saving (SS) percentage.

    SS = (size_of_original_file - size_of_compressed_file) / size_of_original_file

    Args:
        original_file_path (str): Path to the original file.
        compressed_file_path (str): Path to the compressed file.

    Returns:
        float: The space saving as a decimal (0-1). Multiply by 100 for percentage.
    """
    try:
        original_size = calculate_file_size(original_file_path)
        compressed_size = calculate_file_size(compressed_file_path)

        if original_size == 0:
            return 0.0

        return (original_size - compressed_size) / original_size
    except Exception as e:
        print(f"An error occurred while calculating space saving: {e}")
        sys.exit(1)


def calculate_size_difference(original_file_path: str, compressed_file_path: str) -> int:
    """Calculate byte difference between original and compressed files.

    Difference = size_of_original_file - size_of_compressed_file

    Args:
        original_file_path (str): Path to the original file.
        compressed_file_path (str): Path to the compressed file.

    Returns:
        int: Difference in bytes (positive means saved space).
    """
    try:
        original_size = calculate_file_size(original_file_path)
        compressed_size = calculate_file_size(compressed_file_path)
        return original_size - compressed_size
    except Exception as e:
        print(f"An error occurred while calculating size difference: {e}")
        sys.exit(1)


def calculate_entropy(pixel_values: np.ndarray) -> float:
    """Calculate the entropy of pixel values (for images).

    Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of each pixel value.

    Args:
        pixel_values (np.ndarray): 1D array of pixel values (0-255).

    Returns:
        float: The entropy value in bits.
    """
    try:
        if len(pixel_values) == 0:
            return 0.0

        # Calculate frequency of each pixel value
        _, counts = np.unique(pixel_values, return_counts=True)
        probabilities = counts / len(pixel_values)

        # Calculate entropy: H = -Σ p(x) log2 p(x)
        entropy = float(np.sum(probabilities * np.log2(probabilities)) * -1.0)
        return entropy
    except Exception as e:
        print(f"An error occurred while calculating entropy: {e}")
        sys.exit(1)


def calculate_total_code_bits(codes: list[int], initial_width: int = 9, max_width: int = 12) -> int:
    """Calculate total payload bits used by variable-width LZW codes.

    Args:
        codes (list[int]): Encoded LZW codes.
        initial_width (int, optional): Initial code width. Defaults to 9.
        max_width (int, optional): Maximum code width. Defaults to 12.

    Returns:
        int: Number of data bits used by the encoded code stream.
    """
    try:
        total_bits = 0
        current_width = initial_width
        dictionary_size = 256

        for _ in codes:
            total_bits += current_width
            dictionary_size += 1
            if dictionary_size >= 2**current_width and current_width < max_width:
                current_width += 1

        return total_bits
    except Exception as e:
        print(f"An error occurred while calculating total code bits: {e}")
        sys.exit(1)


def calculate_average_code_length(
    bitstream: bytes,
    codes: list[int],
    symbol_count: int | None = None,
) -> float:
    """Calculate the average code length in bits.

    Average code length = (size_of_bitstream_in_bits) / (number_of_codes)

    Args:
        bitstream (bytes): The encoded bitstream.
        codes (list[int]): The list of LZW codes.

    Returns:
        float: The average code length in bits.
    """
    try:
        if len(codes) == 0:
            return 0.0

        total_bits = calculate_total_code_bits(codes)
        denominator = symbol_count if symbol_count is not None else len(codes)
        if denominator <= 0:
            return 0.0

        average_length = total_bits / denominator
        return average_length
    except Exception as e:
        print(f"An error occurred while calculating average code length: {e}")
        sys.exit(1)


def calculate_text_compression_metrics(
    original_file_path: str, compressed_file_path: str
) -> dict[str, Any]:
    """Calculate compression metrics for text files.

    Args:
        original_file_path (str): Path to the original text file.
        compressed_file_path (str): Path to the compressed .lzw file.

    Returns:
        dict[str, Any]: Dictionary containing CR, CF, and SS metrics.
    """
    try:
        original_size = calculate_file_size(original_file_path)
        compressed_size = calculate_file_size(compressed_file_path)

        compression_ratio = calculate_compression_ratio(original_file_path, compressed_file_path)
        compression_factor = calculate_compression_factor(original_file_path, compressed_file_path)
        space_saving = calculate_space_saving(original_file_path, compressed_file_path)
        size_difference = calculate_size_difference(original_file_path, compressed_file_path)

        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "difference_bytes": size_difference,
            "compression_ratio": round(compression_ratio, 4),
            "compression_factor": round(compression_factor, 4),
            "space_saving_percent": round(space_saving * 100, 2),
        }
    except Exception as e:
        print(f"An error occurred while calculating text compression metrics: {e}")
        sys.exit(1)


def calculate_image_compression_metrics(
    original_file_path: str,
    compressed_file_path: str,
    pixel_values: np.ndarray,
    codes: list[int],
    bitstream: bytes,
) -> dict[str, Any]:
    """Calculate compression metrics for image files.

    Args:
        original_file_path (str): Path to the original image file.
        compressed_file_path (str): Path to the compressed .lzw file.
        pixel_values (np.ndarray): Original pixel values (flattened).
        codes (list[int]): LZW codes generated from the image.
        bitstream (bytes): The encoded bitstream.

    Returns:
        dict[str, Any]: Dictionary containing entropy, avg code length, CR, CF, and SS.
    """
    try:
        original_size = calculate_file_size(original_file_path)
        compressed_size = calculate_file_size(compressed_file_path)

        entropy = calculate_entropy(pixel_values)
        avg_code_length = calculate_average_code_length(bitstream, codes)
        compression_ratio = calculate_compression_ratio(original_file_path, compressed_file_path)
        compression_factor = calculate_compression_factor(original_file_path, compressed_file_path)
        space_saving = calculate_space_saving(original_file_path, compressed_file_path)
        size_difference = calculate_size_difference(original_file_path, compressed_file_path)

        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "difference_bytes": size_difference,
            "entropy": round(entropy, 4),
            "average_code_length": round(avg_code_length, 4),
            "compression_ratio": round(compression_ratio, 4),
            "compression_factor": round(compression_factor, 4),
            "space_saving_percent": round(space_saving * 100, 2),
        }
    except Exception as e:
        print(f"An error occurred while calculating image compression metrics: {e}")
        sys.exit(1)


def print_text_compression_report(metrics: dict[str, Any]) -> None:
    """Print a formatted text compression report.

    Args:
        metrics (dict[str, Any]): Dictionary containing compression metrics.
    """
    print("\n" + "=" * 60)
    print("TEXT COMPRESSION METRICS")
    print("=" * 60)
    print(f"Original File Size:     {metrics['original_size']:,} bytes")
    print(f"Compressed File Size:   {metrics['compressed_size']:,} bytes")
    print("-" * 60)
    print(f"Compression Ratio (CR):  {metrics['compression_ratio']}")
    print(f"Compression Factor (CF): {metrics['compression_factor']}")
    print(f"Space Saving (SS):       {metrics['space_saving_percent']}%")
    print("=" * 60 + "\n")


def print_image_compression_report(metrics: dict[str, Any]) -> None:
    """Print a formatted image compression report.

    Args:
        metrics (dict[str, Any]): Dictionary containing compression metrics.
    """
    print("\n" + "=" * 60)
    print("IMAGE COMPRESSION METRICS")
    print("=" * 60)
    print(f"Original File Size:     {metrics['original_size']:,} bytes")
    print(f"Compressed File Size:   {metrics['compressed_size']:,} bytes")
    print("-" * 60)
    print(f"Entropy:                {metrics['entropy']} bits")
    print(f"Average Code Length:    {metrics['average_code_length']} bits")
    print("-" * 60)
    print(f"Compression Ratio (CR):  {metrics['compression_ratio']}")
    print(f"Compression Factor (CF): {metrics['compression_factor']}")
    print(f"Space Saving (SS):       {metrics['space_saving_percent']}%")
    print("=" * 60 + "\n")
