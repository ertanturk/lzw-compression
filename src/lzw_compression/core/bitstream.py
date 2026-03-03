import sys


def convert_to_bitstream(codes: list[int], initial_width: int = 9, max_width: int = 12) -> bytes:
    """Convert a list of variable-width codes into a bitstream.

    Args:
        codes (list[int]): List of integer codes to encode into the bitstream.
        initial_width (int, optional): Initial bit width for encoding codes. Defaults to 9.
        max_width (int, optional): Maximum bit width allowed during encoding. Defaults to 12.

    Returns:
        bytes: Encoded bitstream as bytes.
    """
    try:
        buffer = 0  # Buffer to hold bits before writing to output
        buffer_count = 0  # Number of bits currently in the buffer
        output = bytearray()  # Output byte array to hold the resulting bitstream
        current_width = initial_width  # Current bit width for encoding
        dictionary_size = 256  # Initial dictionary size (for single character codes)

        for code in codes:
            # Add the code to the buffer
            buffer = (buffer << current_width) | code  # Shift buffer and add new code
            buffer_count += current_width  # Update the count of bits in the buffer

            # Write out bytes from the buffer while it has enough bits
            while buffer_count >= 8:
                output.append(
                    (buffer >> (buffer_count - 8)) & 0xFF
                )  # Write the top 8 bits to output
                buffer_count -= 8  # Decrease the bit count by 8

            # Update the dictionary size and adjust bit width if necessary
            dictionary_size += 1
            if dictionary_size >= 2**current_width and current_width < max_width:
                current_width += 1
        # Write any remaining bits in the buffer to the output
        if buffer_count > 0:
            output.append(
                (buffer << (8 - buffer_count)) & 0xFF
            )  # Pad the remaining bits and write to output
        return bytes(output)  # Return the output as bytes
    except Exception as e:
        print(f"An error occurred during converting to bitstream: {e}")
        sys.exit(1)


def convert_bytes_to_codes(
    bitstream: bytes, initial_width: int = 9, max_width: int = 12
) -> list[int]:
    """Convert a bitstream back into a list of variable-width codes.

    Args:
        bitstream (bytes): The input bitstream to decode.
        initial_width (int, optional): Initial bit width for decoding codes. Defaults to 9.
        max_width (int, optional): Maximum bit width allowed during decoding. Defaults to 12.

    Returns:
        list[int]: List of integer codes decoded from the bitstream.
    """
    try:
        buffer = 0  # Buffer to hold bits before extracting codes
        buffer_count = 0  # Number of bits currently in the buffer
        codes: list[int] = []  # List to hold the resulting codes
        current_width = initial_width  # Current bit width for decoding
        dictionary_size = 256  # Initial dictionary size (for single character codes)

        for byte in bitstream:
            buffer = (buffer << 8) | byte  # Shift buffer and add new byte
            buffer_count += 8  # Update the count of bits in the buffer

            # Extract codes from the buffer while it has enough bits
            while buffer_count >= current_width:
                code = (buffer >> (buffer_count - current_width)) & (
                    (1 << current_width) - 1
                )  # Extract the top code
                codes.append(code)  # Add the code to the list
                buffer_count -= current_width  # Decrease the bit count by the current width

            # Update the dictionary size and adjust bit width if necessary
            dictionary_size += 1
            if dictionary_size >= 2**current_width and current_width < max_width:
                current_width += 1
        return codes  # Return the list of decoded codes
    except Exception as e:
        print(f"An error occurred during converting bytes to codes: {e}")
        sys.exit(1)
