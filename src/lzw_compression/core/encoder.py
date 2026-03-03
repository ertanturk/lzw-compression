import sys

import numpy as np

from lzw_compression.core.io import open_image_file, open_text_file


def text_file_encoder(file_path: str) -> list[int]:
    """Encodes a text file using the LZW compression algorithm.

    Args:
        file_path (str): The path to the text file to be encoded.

    Returns:
        list[int]: A list of integers representing the LZW encoded output.
    """
    try:
        dictionary: dict[str, int] = {chr(i): i for i in range(256)}  # Initialize dictionary
        next_free_code: int = 256  # Next available code for new entries
        result: list[int] = []  # List to store the output codes

        # Open the text file and read its content
        content: str = open_text_file(file_path)
        current_string: str = ""  # Initialize the current string as empty

        # Iterate through each character in the content
        for symbol in content:
            combined_string: str = (
                current_string + symbol
            )  # Combine current string with the new symbol

            # Check if the combined string is in the dictionary
            if combined_string in dictionary:
                current_string = combined_string  # Update current string to the combined string
            else:
                result.append(dictionary[current_string])  # Output the code for the current string
                dictionary[combined_string] = (
                    next_free_code  # Add combined string to the dictionary
                )
                next_free_code += 1  # Increment the next free code
                current_string = symbol  # Start a new current string with the symbol
        if current_string:  # Output the code for the last current string if it's not empty
            result.append(dictionary[current_string])
        return result
    except Exception as e:
        print(f"An error occurred during encoding text file '{file_path}': {e}")
        sys.exit(1)


def image_file_encoder_grayscale(file_path: str) -> list[int]:
    """Encodes a grayscale image file using the LZW compression algorithm.

    Args:
        file_path (str): The path to the grayscale image file to be encoded.

    Returns:
        list[int]: A list of integers representing the LZW encoded output.
    """
    try:
        dictionary: dict[str, int] = {chr(i): i for i in range(256)}  # Initialize dictionary
        next_free_code: int = 256  # Next available code for new entries
        result: list[int] = []  # List to store the output codes

        # Open the image file and read its content as a NumPy array
        image_array = open_image_file(file_path)

        # Flatten the image array to a 1D list of pixel values
        pixel_values = image_array.flatten()

        current_string: str = ""  # Initialize the current string as empty

        # Iterate through each pixel value in the flattened array
        for pixel in pixel_values:
            symbol = chr(pixel)  # Convert pixel value to a character
            combined_string: str = (
                current_string + symbol
            )  # Combine current string with the new symbol

            # Check if the combined string is in the dictionary
            if combined_string in dictionary:
                current_string = combined_string  # Update current string to the combined string
            else:
                result.append(dictionary[current_string])  # Output the code for the current string
                dictionary[combined_string] = (
                    next_free_code  # Add combined string to the dictionary
                )
                next_free_code += 1  # Increment the next free code
                current_string = symbol  # Start a new current string with the symbol
        if current_string:  # Output the code for the last current string if it's not empty
            result.append(dictionary[current_string])
        return result
    except Exception as e:
        print(f"An error occurred during encoding image file '{file_path}': {e}")
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
        # Open the image file and read its content as a NumPy array
        image_array = open_image_file(file_path)

        # Compute 2D differences (row-wise and column-wise)
        diff_array = image_file_compute_differences(image_array)

        # Offset differences to 0-255 range (to handle negative values)
        # Differences range from -255 to +255, we shift by 128 to get 0-255
        diff_array_offset = ((diff_array.astype(int) + 128) % 256).astype(int)

        # Flatten the difference array to a 1D list of pixel values
        pixel_values = diff_array_offset.flatten()

        # Encode the differences using LZW
        dictionary: dict[str, int] = {chr(i): i for i in range(256)}  # Initialize dictionary
        next_free_code: int = 256  # Next available code for new entries
        result: list[int] = []  # List to store the output codes

        current_string: str = ""  # Initialize the current string as empty

        for pixel in pixel_values:
            symbol = chr(int(pixel))  # Convert pixel value to a character
            combined_string: str = (
                current_string + symbol
            )  # Combine current string with the new symbol

            if combined_string in dictionary:
                current_string = combined_string  # Update current string to the combined string
            else:
                result.append(dictionary[current_string])  # Output the code for the current string
                dictionary[combined_string] = (
                    next_free_code  # Add combined string to the dictionary
                )
                next_free_code += 1  # Increment the next free code
                current_string = symbol  # Start a new current string with the symbol
        if current_string:  # Output the code for the last current string if it's not empty
            result.append(dictionary[current_string])
        return result
    except Exception as e:
        print(f"An error occurred during encoding image file with differences '{file_path}': {e}")
        sys.exit(1)
