import sys

_MAX_PADDING_BITS = 7


def convert_to_bitstream(
	codes: list[int],
	initial_width: int = 9,
	max_width: int = 12,
) -> bytes:
	"""Convert a list of variable-width codes into a bitstream.

	Args:
	    codes (list[int]): List of integer codes to encode into the bitstream.
	    initial_width (int, optional): Initial bit width for encoding codes. Defaults to 9.
	    max_width (int, optional): Maximum bit width allowed during encoding. Defaults to 12.

	Returns:
	    bytes: Encoded bitstream as bytes.

	"""
	try:
		current_width = initial_width  # Current bit width for encoding
		dictionary_size = (
			256  # Initial dictionary size (for single character codes)
		)
		code_bits: list[str] = []

		for code in codes:
			code_bits.append(f"{code:0{current_width}b}")

			# Update the dictionary size and adjust bit width if necessary
			dictionary_size += 1
			if (
				dictionary_size >= 2**current_width
				and current_width < max_width
			):
				current_width += 1

		# 1) int array -> binary string
		# 2) pad bits
		# 3) prepend padding info (1 byte)
		# 4) write bytes
		binary_data = "".join(code_bits)
		padding_bits = (8 - (len(binary_data) % 8)) % 8
		binary_data_padded = binary_data + ("0" * padding_bits)

		payload = [padding_bits]
		payload.extend(
			int(binary_data_padded[i : i + 8], 2)
			for i in range(0, len(binary_data_padded), 8)
		)
		return bytes(payload)
	except Exception as e:
		print(f"An error occurred during converting to bitstream: {e}")
		sys.exit(1)


def convert_bytes_to_codes(
	bitstream: bytes,
	initial_width: int = 9,
	max_width: int = 12,
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
		if not bitstream:
			return []

		padding_bits = bitstream[0]
		if padding_bits > _MAX_PADDING_BITS:
			print(f"Invalid padding header: {padding_bits}")
			sys.exit(1)

		binary_data = "".join(f"{byte:08b}" for byte in bitstream[1:])
		if padding_bits:
			if padding_bits > len(binary_data):
				print("Invalid bitstream: padding larger than payload")
				sys.exit(1)
			binary_data = binary_data[:-padding_bits]

		codes: list[int] = []  # List to hold the resulting codes
		current_width = initial_width  # Current bit width for decoding
		dictionary_size = (
			256  # Initial dictionary size (for single character codes)
		)

		bit_position = 0
		while bit_position + current_width <= len(binary_data):
			code = int(
				binary_data[
					bit_position : bit_position + current_width
				],
				2,
			)
			codes.append(code)
			bit_position += current_width

			dictionary_size += 1
			if (
				dictionary_size >= 2**current_width
				and current_width < max_width
			):
				current_width += 1

		return codes  # Return the list of decoded codes
	except Exception as e:
		print(f"An error occurred during converting bytes to codes: {e}")
		sys.exit(1)
