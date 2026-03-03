import sys

import numpy as np
from PIL import Image


def open_text_file(file_path: str) -> str:
    """Open and read a text file, returning its content as a string.

    Args:
        file_path (str): The path to the text file to be opened.

    Returns:
        str: The content of the file as a string.
    """

    try:
        with open(file_path) as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while opening the file '{file_path}': {e}")
        sys.exit(1)


def write_bitstream_to_text_file(bitstream: bytes, output_file_path: str) -> None:
    """Write a bitstream to a text file.

    Args:
        bitstream (bytes): The bitstream to be written to the file.
        output_file_path (str): The path to the output text file.
    """
    try:
        with open(output_file_path, "wb") as file:
            file.write(bitstream)
    except Exception as e:
        print(f"An error occurred while writing to the file '{output_file_path}': {e}")
        sys.exit(1)


def open_bitstream_file(file_path: str) -> bytes:
    """Open and read a bitstream file, returning its content as bytes.

    Args:
        file_path (str): The path to the bitstream file to be opened.
    Returns:
        bytes: The content of the file as bytes.
    """
    try:
        with open(file_path, "rb") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while opening the file '{file_path}': {e}")
        sys.exit(1)


def open_image_file(file_path: str) -> np.ndarray:
    """Open and read an image file, returning its content as a NumPy array.

    Args:
        file_path (str): The path to the image file to be opened.
    Returns:
        np.ndarray: The content of the image file as a NumPy array.
    """
    try:
        with Image.open(file_path) as img:
            return np.array(img)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while opening the file '{file_path}': {e}")
        sys.exit(1)


def open_color_image_file(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open a color image file and decompose it into RGB components.

    Reads a color image file and separates it into three grayscale arrays
    (red, green, and blue channels). If the image is already grayscale,
    returns the same array three times.

    Args:
        file_path (str): The path to the color image file to be opened.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing
            (red_channel, green_channel, blue_channel) as separate uint8 arrays.

    Raises:
        SystemExit: If the file is not found or another error occurs.
    """
    try:
        image_array = open_image_file(file_path)

        _grayscale_dims = 2
        _min_rgb_channels = 3

        # Handle grayscale (2D array)
        if len(image_array.shape) == _grayscale_dims:
            return image_array, image_array, image_array

        # Handle RGB/RGBA (3D array with 3+ channels)
        if len(image_array.shape) == _min_rgb_channels:
            if image_array.shape[2] >= _min_rgb_channels:
                red_channel = image_array[:, :, 0]
                green_channel = image_array[:, :, 1]
                blue_channel = image_array[:, :, 2]
                return red_channel, green_channel, blue_channel
            else:
                msg = f"Image has only {image_array.shape[2]} channel(s), RGB expected"
                raise ValueError(msg)

        msg = f"Unexpected image shape: {image_array.shape}"
        raise ValueError(msg)

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error processing image '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while opening the file '{file_path}': {e}")
        sys.exit(1)


def write_bitstream_with_dimensions(
    bitstream: bytes, output_file_path: str, height: int, width: int
) -> None:
    """Write a bitstream with embedded image dimensions to a file.

    Args:
        bitstream (bytes): The bitstream to be written to the file.
        output_file_path (str): The path to the output file.
        height (int): The image height.
        width (int): The image width.
    """
    try:
        with open(output_file_path, "wb") as file:
            # Write dimensions as 4-byte integers (little-endian)
            file.write(height.to_bytes(4, byteorder="little"))
            file.write(width.to_bytes(4, byteorder="little"))
            # Write the bitstream
            file.write(bitstream)
    except Exception as e:
        print(f"An error occurred while writing to the file '{output_file_path}': {e}")
        sys.exit(1)


def open_bitstream_file_with_dimensions(file_path: str) -> tuple[bytes, int, int]:
    """Open and read a bitstream file with embedded dimensions.

    Args:
        file_path (str): The path to the bitstream file to be opened.

    Returns:
        tuple[bytes, int, int]: A tuple containing (bitstream, height, width).
    """
    try:
        with open(file_path, "rb") as file:
            # Read dimensions (8 bytes total: 4 for height, 4 for width)
            height_bytes = file.read(4)
            width_bytes = file.read(4)
            height = int.from_bytes(height_bytes, byteorder="little")
            width = int.from_bytes(width_bytes, byteorder="little")
            # Read the remaining bitstream
            bitstream = file.read()
            return bitstream, height, width
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while opening the file '{file_path}': {e}")
        sys.exit(1)


def save_image_file(image_array: np.ndarray, output_file_path: str, format: str = "PNG") -> None:
    """Save a NumPy array as an image file.

    Args:
        image_array (np.ndarray): The image data as a NumPy array.
        output_file_path (str): The path where the image should be saved.
        format (str, optional): The image format (e.g., 'PNG', 'JPEG'). Defaults to 'PNG'.
    """
    try:
        img = Image.fromarray(image_array.astype(np.uint8))
        img.save(output_file_path, format=format)
    except Exception as e:
        print(f"An error occurred while saving the image '{output_file_path}': {e}")
        sys.exit(1)
