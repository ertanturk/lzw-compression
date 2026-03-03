# Copilot Instructions: LZW Compression

## Project Overview

LZW (Lempel-Ziv-Welch) compression library implementing a classic dictionary-based text compression algorithm. Python 3.12+ only. Early-stage: encoder is functional; decoder, bitstream utilities, and UI are planned.

## Architecture

### Core Components

- **`encoder.py`**: Main compression logic
  - `text_file_encoder(file_path)` → `list[int]`
  - Algorithm: Initializes dictionary with 256 ASCII chars (0-255), then iteratively builds longer patterns
  - Pattern: Check if `current_string + symbol` exists in dict; if yes, extend current_string; if no, output current code and add new string to dict
- **`io.py`**: File operations
  - `open_text_file(file_path)` → `str`
  - Simple wrapper with error handling (FileNotFoundError → exit 1)

- **`bitstream.py`, `decoder.py`**: Empty placeholder files (future implementation)

- **`ui/app.py`**: Empty placeholder (future UI)

### Data Flow

```
Text File → open_text_file() → string content
           → text_file_encoder() → list of integer codes
```

## Key Patterns & Conventions

### Error Handling

```python
try:
    # operation
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)  # Always exit with code 1 on file/processing errors
```

### Type Annotations

Always use explicit type hints:

```python
def function(param: str) -> list[int]:
```

### Dictionary Initialization

Encoder always initializes with ASCII table (0-255):

```python
dictionary: dict[str, int] = {chr(i): i for i in range(256)}
next_free_code: int = 256
```

## Testing & Development

### Test Framework

- Uses `pyforge_test` (custom test runner, imports via `pyforge_test.test`)
- Tests use dynamic module import pattern:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("module_name", "src/path/file.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

### Running Tests

```bash
pyforge  # Runs all tests in tests/ directory
```

### Sample Data

- `samples/short_text.csv`: "ABRACADABRA" (classic LZW example)
- `samples/short_text.txt`: "Lorem Ipsum is simply dummy text"
- Use these for algorithm verification

## Code Quality

### Linting Rules (Ruff)

- Line length: 100 characters
- Active rules: E, W, F, I, UP, B, C4, SIM, RUF, PL, ARG
- All rules auto-fixable
- Run: `ruff check --fix src/` or rely on editor integration

### Quote Style

- Use double quotes only (`"string"`, not `'string'`)
- Configured in ruff format settings

## External Dependencies

- **Pillow**: Image handling (imported but purpose unclear in current encoder—likely for planned image compression features)
- **numpy**: Likely for planned numerical/binary operations

## When Adding Features

### Decoder Implementation

- Mirror encoder's dictionary approach (reverse lookup)
- Handle code table reconstruction from integer stream
- Update `decoder.py` (currently empty)

### Bitstream Operations

- Implement efficient bit-level I/O in `bitstream.py`
- Consider variable-code-width encoding (256 codes → 9 bits initially, grows with dictionary)
- Likely integration point for Pillow/numpy

### UI Implementation

- Implement in `ui/app.py`
- Should expose `text_file_encoder()` and future decoder interface
- Consider cross-module imports from `core` package

## Important Notes

- Python 3.12+ requirement is strict (pyproject.toml `requires-python`)
- Empty decoder/bitstream files are scaffolding—prioritize before UI
- File paths in tests are relative (samples/ from project root)
