# LZW Compression

A professional Python implementation of the **Lempel-Ziv-Welch (LZW)** dictionary-based compression algorithm. Supports text and image compression with variable-width bitstream encoding and optional 2D preprocessing for improved compression ratios.

> **Course Assignment:** This project was developed as part of a university course.

## Features

- **Text & Image Compression** — LZW encoder/decoder for both plaintext and grayscale images
- **2D Encoding** — Optional row-wise and column-wise difference preprocessing for smoother images
- **RGB Channel Decomposition** — Extract and compress individual colour channels independently
- **Variable-Width Bitstream** — Efficient 9–12 bit encoding that grows with dictionary size
- **Compression Metrics** — Compression Ratio (CR), Compression Factor (CF), Space Saving (SS), entropy, average code length
- **Tkinter GUI** — User-friendly graphical interface for compression/decompression and metrics visualization
- **Professional Test Suite** — 23 comprehensive unit tests with 100% pass rate

## Quick Start

### Installation

```bash
git clone https://github.com/ertanturk/lzw-compression
cd lzw-compression
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"
```

### Launch GUI

```bash
lzw-gui
```

Or directly:

```bash
python -m lzw_compression.ui.app
```

### Command Line Usage

```python
from lzw_compression.core.encoder import text_file_encoder
from lzw_compression.core.io import write_bitstream_to_text_file
from lzw_compression.core.bitstream import convert_to_bitstream

# Compress text
codes = text_file_encoder("input.txt")
bitstream = convert_to_bitstream(codes)
write_bitstream_to_text_file(bitstream, "output.lzw")

# Decompress
from lzw_compression.core.decoder import codes_to_text
from lzw_compression.core.bitstream import convert_bytes_to_codes
from lzw_compression.core.io import open_bitstream_file

bitstream = open_bitstream_file("output.lzw")
codes = convert_bytes_to_codes(bitstream)
text = codes_to_text(codes)
```

## Architecture

```
src/lzw_compression/
├── core/
│   ├── encoder.py           # LZW encoding for text & grayscale images
│   ├── decoder.py           # LZW decoding
│   ├── bitstream.py         # Variable-width (9–12 bit) bitstream I/O
│   ├── io.py                # File operations, image I/O, channel decomposition
│   └── metrics.py           # Compression statistics (CR, CF, SS, entropy, etc.)
└── ui/
    └── app.py               # Tkinter GUI application
```

### Algorithm Overview

1. **Encoder** — Maintains a dictionary of string→code mappings (256–4096 entries). Reads input symbol by symbol, outputting dictionary codes and building new patterns.
2. **Bitstream** — Encodes variable-width codes (9–12 bits) into byte sequences; reverses the process on decode.
3. **2D Encoding** (optional) — Computes row-wise and column-wise pixel differences before LZW to improve compression on smooth images.
4. **Decoder** — Reconstructs the original dictionary on-the-fly from the code stream, reversing if needed.

## Usage Examples

### Text Compression

```python
from lzw_compression.core.encoder import text_file_encoder
codes = text_file_encoder("samples/short_text.csv")
print(codes)  # [65, 66, 82, 65, 67, 65, 68, 256, 258]
```

### Image Compression with Delta Encoding

```python
from lzw_compression.core.encoder import image_file_encoder_grayscale_differences
from lzw_compression.core.io import write_bitstream_with_dimensions
from lzw_compression.core.bitstream import convert_to_bitstream

codes = image_file_encoder_grayscale_differences("image.png")
bitstream = convert_to_bitstream(codes)
write_bitstream_with_dimensions(bitstream, "image.lzw", h=256, w=256)
```

### Channel Extraction

```python
from lzw_compression.core.io import open_color_image_file
red, green, blue = open_color_image_file("color_image.png")
# Each channel is a 2D numpy array (H x W)
```

## Requirements

- **Python 3.12+**
- **NumPy** — Array operations
- **Pillow** — Image I/O
- **tkinter** — GUI (included with Python on most systems; `apt install python3-tk` on Linux)

## Testing

All 23 tests pass:

```bash
python -m pyforge_test
```

Test coverage includes:

- Text encoding correctness (single char, repetition, patterns)
- Full encode–decode round-trips for text and images
- 2D encoding with varied pixel patterns
- Compression metrics validation (CR, CF, SS, entropy)
- Per-channel RGB encoding/decoding
- RGB channel decomposition accuracy

## Code Quality

- **Ruff linter** — Line length 100, PEP 8 compliance, auto-fixable
- **Type hints** — Full static type annotations for all functions
- **Error handling** — Graceful file I/O with descriptive error messages

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Ertan Tunç Türk  
Designed as a university course assignment.
