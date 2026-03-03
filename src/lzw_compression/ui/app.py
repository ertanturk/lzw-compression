"""LZW Compression GUI Application.

Tkinter-based graphical interface for the LZW compression library supporting
text and image files, RGB channel decomposition, grayscale / grayscale-differences
encoding methods, compression metrics, and decompression preview.
"""

import os
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# Make sure the package root is importable regardless of how app.py is launched
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from lzw_compression.core.bitstream import (  # noqa: E402
    convert_bytes_to_codes,
    convert_to_bitstream,
)
from lzw_compression.core.decoder import (  # noqa: E402
    codes_to_image_grayscale,
    codes_to_image_grayscale_differences,
    codes_to_text,
)
from lzw_compression.core.encoder import (  # noqa: E402
    image_file_encoder_grayscale,
    image_file_encoder_grayscale_differences,
    text_file_encoder,
)
from lzw_compression.core.io import (  # noqa: E402
    open_bitstream_file,
    open_bitstream_file_with_dimensions,
    open_color_image_file,
    open_image_file,
    open_text_file,
    save_image_file,
    write_bitstream_to_text_file,
    write_bitstream_with_dimensions,
)
from lzw_compression.core.metrics import (  # noqa: E402
    calculate_average_code_length,
    calculate_compression_factor,
    calculate_compression_ratio,
    calculate_entropy,
    calculate_file_size,
    calculate_space_saving,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS = {".txt", ".csv"}
IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg"}
MAX_PREVIEW_SIZE = 380
_BYTES_PER_KB = 1024
_MIN_RGB_CHANNELS = 3
_MAX_IMAGE_DIMENSION = 100_000


def _fit_image(pil_img: Image.Image, max_dim: int = MAX_PREVIEW_SIZE) -> Image.Image:
    """Return a copy of *pil_img* scaled to fit within *max_dim* x *max_dim*."""
    pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return pil_img


def _colorize_channel(grayscale: np.ndarray, channel_name: str) -> np.ndarray:
    """Create an RGB image that shows only the selected colour channel.

    Args:
        grayscale: 2D array of single-channel pixel intensities.
        channel_name: One of "Red", "Green", or "Blue".

    Returns:
        3D uint8 array (H x W x 3) with the selected channel populated
        and all other channels set to zero.
    """
    h, w = grayscale.shape[:2]
    rgb = np.zeros((h, w, _MIN_RGB_CHANNELS), dtype=np.uint8)
    index = {"Red": 0, "Green": 1, "Blue": 2}.get(channel_name)
    if index is not None:
        rgb[:, :, index] = grayscale
    return rgb


class LZWCompressionApp:
    """Tkinter GUI for LZW compression / decompression."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LZW Compression Tool")
        self.root.geometry("1100x820")
        self.root.minsize(960, 720)

        # ---- internal state ------------------------------------------------
        self._src_path: str | None = None
        self._is_text = False
        self._is_image = False
        self._image_array: np.ndarray | None = None
        self._channel_arrays: dict[str, np.ndarray] = {}
        self._active_channel: str = "Grayscale"
        self._active_array: np.ndarray | None = None
        self._text_content: str | None = None

        self._compressed_path: str | None = None
        self._codes: list[int] = []
        self._bitstream: bytes = b""

        self._tmpdir = tempfile.mkdtemp(prefix="lzw_")

        # ---- build widgets -------------------------------------------------
        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:  # noqa: PLR0915
        """Construct the entire widget tree."""
        # ===== Row 0 - File selection =======================================
        frm_file = ttk.LabelFrame(self.root, text="1 - File Selection", padding=8)
        frm_file.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        self._lbl_file = ttk.Label(frm_file, text="No file loaded", foreground="gray")
        self._lbl_file.pack(side="left", fill="x", expand=True)

        ttk.Button(frm_file, text="Open File...", command=self._on_open_file).pack(
            side="left",
            padx=4,
        )
        ttk.Button(
            frm_file,
            text="Open Compressed (.lzw)...",
            command=self._on_decompress,
        ).pack(side="left", padx=4)

        # ===== Row 1 - Method + Channel =====================================
        frm_opts = ttk.Frame(self.root)
        frm_opts.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)

        # -- method --
        frm_method = ttk.LabelFrame(frm_opts, text="2 - Encoding Method", padding=8)
        frm_method.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self._method_var = tk.StringVar(value="grayscale")
        ttk.Radiobutton(
            frm_method,
            text="Grayscale",
            variable=self._method_var,
            value="grayscale",
        ).pack(side="left", padx=6)
        ttk.Radiobutton(
            frm_method,
            text="Grayscale Differences",
            variable=self._method_var,
            value="differences",
        ).pack(side="left", padx=6)

        # -- channel selector --
        frm_chan = ttk.LabelFrame(frm_opts, text="3 - Color Channel", padding=8)
        frm_chan.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self._channel_var = tk.StringVar(value="Grayscale")
        self._channel_combo = ttk.Combobox(
            frm_chan,
            textvariable=self._channel_var,
            state="disabled",
            width=14,
            values=["Grayscale"],
        )
        self._channel_combo.pack(side="left", padx=6)
        self._channel_combo.bind("<<ComboboxSelected>>", self._on_channel_changed)

        # ===== Row 2 - Action buttons =======================================
        frm_actions = ttk.LabelFrame(self.root, text="4 - Actions", padding=8)
        frm_actions.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=4)

        ttk.Button(
            frm_actions,
            text="Compress & Save...",
            command=self._on_compress,
        ).pack(side="left", padx=5)
        ttk.Button(
            frm_actions,
            text="Decompress & Show",
            command=self._on_decompress,
        ).pack(side="left", padx=5)
        ttk.Button(frm_actions, text="Clear All", command=self._on_clear).pack(
            side="left",
            padx=5,
        )

        # ===== Row 3 - Preview + Metrics (main area) ========================
        frm_main = ttk.Frame(self.root)
        frm_main.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=6, pady=4)
        self.root.rowconfigure(3, weight=1)

        # -- left: preview notebook --
        frm_preview = ttk.LabelFrame(frm_main, text="Preview", padding=6)
        frm_preview.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self._notebook = ttk.Notebook(frm_preview)
        self._notebook.pack(fill="both", expand=True)

        # image tab
        self._frm_img_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._frm_img_tab, text="Image")
        self._lbl_image = ttk.Label(
            self._frm_img_tab,
            text="No image",
            anchor="center",
        )
        self._lbl_image.pack(fill="both", expand=True)

        # text tab
        self._frm_txt_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._frm_txt_tab, text="Text")
        self._txt_preview = tk.Text(
            self._frm_txt_tab,
            wrap="word",
            state="disabled",
            height=18,
            width=50,
        )
        scrollbar = ttk.Scrollbar(
            self._frm_txt_tab,
            command=self._txt_preview.yview,  # type: ignore[arg-type]
        )
        self._txt_preview.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._txt_preview.pack(side="left", fill="both", expand=True)

        # -- right: metrics panel --
        frm_metrics = ttk.LabelFrame(
            frm_main,
            text="Compression Metrics",
            padding=10,
        )
        frm_metrics.pack(side="right", fill="y", padx=(4, 0))

        labels_spec: list[tuple[str, str]] = [
            ("Original Size:", "orig_size"),
            ("Compressed Size:", "comp_size"),
            ("Compression Ratio (CR):", "cr"),
            ("Compression Factor (CF):", "cf"),
            ("Space Saving (SS):", "ss"),
            ("Entropy:", "entropy"),
            ("Avg Code Length:", "acl"),
            ("Number of Codes:", "ncodes"),
        ]
        self._metric_labels: dict[str, ttk.Label] = {}
        for idx, (caption, key) in enumerate(labels_spec):
            ttk.Label(
                frm_metrics,
                text=caption,
                font=("", 10, "bold"),
            ).grid(row=idx, column=0, sticky="w", pady=3)
            lbl = ttk.Label(frm_metrics, text="--", font=("", 10))
            lbl.grid(row=idx, column=1, sticky="w", padx=(10, 0), pady=3)
            self._metric_labels[key] = lbl

        # ===== Row 4 - Status bar ===========================================
        self._lbl_status = ttk.Label(
            self.root,
            text="Ready",
            relief="sunken",
            anchor="w",
            padding=(6, 2),
        )
        self._lbl_status.grid(row=4, column=0, columnspan=2, sticky="ew")

    # -----------------------------------------------------------------------
    # Helpers - display
    # -----------------------------------------------------------------------
    def _set_status(self, msg: str) -> None:
        self._lbl_status.config(text=msg)
        self.root.update_idletasks()

    def _show_image_preview(self, arr: np.ndarray) -> None:
        """Display an image (numpy array) in the Image tab."""
        pil = Image.fromarray(arr.astype(np.uint8))
        pil = _fit_image(pil, MAX_PREVIEW_SIZE)
        photo = ImageTk.PhotoImage(pil)
        self._lbl_image.config(image=photo, text="")  # type: ignore[arg-type]
        self._lbl_image.image = photo  # type: ignore[attr-defined]
        self._notebook.select(self._frm_img_tab)  # type: ignore[arg-type]

    def _show_text_preview(self, text: str) -> None:
        """Display text in the Text tab."""
        self._txt_preview.config(state="normal")
        self._txt_preview.delete("1.0", "end")
        self._txt_preview.insert("1.0", text)
        self._txt_preview.config(state="disabled")
        self._notebook.select(self._frm_txt_tab)  # type: ignore[arg-type]

    @staticmethod
    def _fmt_size(n: int | float) -> str:
        """Format byte count into human-readable string."""
        n = int(n)
        for unit in ("B", "KB", "MB", "GB"):
            if n < _BYTES_PER_KB:
                return f"{n:,.2f} {unit}" if unit != "B" else f"{n:,} {unit}"
            n /= _BYTES_PER_KB
        return f"{n:,.2f} TB"

    def _reset_metrics(self) -> None:
        for lbl in self._metric_labels.values():
            lbl.config(text="--")

    def _update_channel_combo(self, has_color: bool) -> None:
        """Populate the channel combo box depending on image type."""
        values = ["Grayscale", "Red", "Green", "Blue"] if has_color else ["Grayscale"]
        self._channel_combo.config(values=values, state="readonly")
        self._channel_var.set("Grayscale")
        self._on_channel_changed()

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def _on_open_file(self) -> None:
        """Let the user pick a text or image file, load and preview it."""
        path = filedialog.askopenfilename(
            title="Select a file to compress",
            filetypes=[
                ("All Supported", "*.txt *.csv *.png *.bmp *.jpg *.jpeg"),
                ("Text files", "*.txt *.csv"),
                ("Image files", "*.png *.bmp *.jpg *.jpeg"),
            ],
        )
        if not path:
            return

        try:
            self._on_clear(silent=True)
            self._src_path = path
            ext = Path(path).suffix.lower()
            self._is_text = ext in TEXT_EXTENSIONS
            self._is_image = ext in IMAGE_EXTENSIONS

            if self._is_text:
                self._text_content = open_text_file(path)
                self._show_text_preview(self._text_content)
                self._channel_combo.config(state="disabled")
                self._channel_var.set("Grayscale")

            elif self._is_image:
                self._image_array = open_image_file(path)

                # Decompose into colour channels
                red, green, blue = open_color_image_file(path)
                self._channel_arrays = {
                    "Red": red,
                    "Green": green,
                    "Blue": blue,
                }

                has_color = (
                    len(self._image_array.shape) == _MIN_RGB_CHANNELS
                    and self._image_array.shape[2] >= _MIN_RGB_CHANNELS
                )
                self._update_channel_combo(has_color)

                # Default preview shows the full original image
                self._show_image_preview(self._image_array)
            else:
                messagebox.showwarning(
                    "Unsupported",
                    f"File type '{ext}' is not supported.",
                )
                return

            name = Path(path).name
            size = os.path.getsize(path)
            self._lbl_file.config(
                text=f"{name}  ({self._fmt_size(size)})",
                foreground="black",
            )
            self._set_status(f"Loaded: {name}")
            self._reset_metrics()

        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load file:\n{exc}")

    def _on_channel_changed(self, _event: object = None) -> None:
        """User selected a different color channel - update the preview."""
        if not self._is_image or self._image_array is None:
            return

        channel = self._channel_var.get()
        self._active_channel = channel

        if channel in self._channel_arrays:
            arr = self._channel_arrays[channel]
            self._active_array = arr
            # Build a coloured RGB preview so the channel shows in its
            # actual colour instead of appearing as a dark grayscale image.
            coloured = _colorize_channel(arr, channel)
            self._show_image_preview(coloured)
        else:
            # "Grayscale" - convert or use original
            if len(self._image_array.shape) == _MIN_RGB_CHANNELS:
                gray = np.mean(
                    self._image_array[:, :, :_MIN_RGB_CHANNELS],
                    axis=2,
                ).astype(np.uint8)
            else:
                gray = self._image_array
            self._active_array = gray
            self._show_image_preview(gray)

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------
    def _on_compress(self) -> None:
        """Encode the loaded file and save the compressed .lzw output."""
        if self._src_path is None:
            messagebox.showwarning("Nothing loaded", "Please open a file first.")
            return

        method = self._method_var.get()

        try:
            self._set_status("Compressing...")

            # --- Text path ------------------------------------------------
            if self._is_text:
                codes = text_file_encoder(self._src_path)
                bitstream = convert_to_bitstream(codes)

                save_path = filedialog.asksaveasfilename(
                    title="Save compressed file",
                    defaultextension=".lzw",
                    filetypes=[("LZW Compressed", "*.lzw")],
                )
                if not save_path:
                    self._set_status("Compression cancelled")
                    return

                write_bitstream_to_text_file(bitstream, save_path)

                self._codes = codes
                self._bitstream = bitstream
                self._compressed_path = save_path
                self._display_metrics_text(save_path)
                self._set_status(
                    f"Saved compressed text -> {Path(save_path).name}",
                )
                messagebox.showinfo("Done", "Text compressed successfully.")
                return

            # --- Image path ------------------------------------------------
            if self._is_image:
                self._on_channel_changed()
                if self._active_array is None:
                    messagebox.showerror("Error", "No channel data available.")
                    return

                tmp_gray_path = os.path.join(
                    self._tmpdir,
                    "_active_channel.png",
                )
                save_image_file(self._active_array, tmp_gray_path)

                if method == "differences":
                    codes = image_file_encoder_grayscale_differences(
                        tmp_gray_path,
                    )
                else:
                    codes = image_file_encoder_grayscale(tmp_gray_path)

                bitstream = convert_to_bitstream(codes)

                save_path = filedialog.asksaveasfilename(
                    title="Save compressed image",
                    defaultextension=".lzw",
                    filetypes=[("LZW Compressed", "*.lzw")],
                )
                if not save_path:
                    self._set_status("Compression cancelled")
                    return

                h, w = self._active_array.shape[:2]
                write_bitstream_with_dimensions(bitstream, save_path, h, w)

                self._codes = codes
                self._bitstream = bitstream
                self._compressed_path = save_path
                self._display_metrics_image(tmp_gray_path, save_path)
                self._set_status(
                    f"Saved compressed image -> {Path(save_path).name}",
                )
                messagebox.showinfo("Done", "Image compressed successfully.")
                return

        except Exception as exc:
            messagebox.showerror("Compression Error", str(exc))
            self._set_status("Compression failed")

    # ------------------------------------------------------------------
    # Decompress
    # ------------------------------------------------------------------
    def _on_decompress(self) -> None:
        """Load a .lzw file and show the decompressed content + metrics."""
        path = filedialog.askopenfilename(
            title="Select a compressed .lzw file",
            filetypes=[("LZW Compressed", "*.lzw"), ("All Files", "*.*")],
        )
        if not path:
            return

        method = self._method_var.get()
        self._set_status("Decompressing...")

        try:
            # Try as image first (file has 8-byte dimension header)
            bitstream_raw, height, width = open_bitstream_file_with_dimensions(
                path,
            )

            if 0 < height < _MAX_IMAGE_DIMENSION and 0 < width < _MAX_IMAGE_DIMENSION:
                codes = convert_bytes_to_codes(bitstream_raw)

                if method == "differences":
                    decoded_img = codes_to_image_grayscale_differences(
                        codes,
                        (height, width),
                    )
                else:
                    decoded_img = codes_to_image_grayscale(
                        codes,
                        (height, width),
                    )

                self._show_image_preview(decoded_img)
                self._image_array = decoded_img
                self._active_array = decoded_img
                self._codes = codes
                self._bitstream = bitstream_raw
                self._compressed_path = path

                tmp_decoded = os.path.join(self._tmpdir, "_decoded.png")
                save_image_file(decoded_img, tmp_decoded)
                self._display_metrics_image(tmp_decoded, path)
                self._set_status(
                    f"Decompressed image ({height}x{width})",
                )
                return

        except Exception:
            pass  # fall through to text decompression

        try:
            bitstream_raw = open_bitstream_file(path)
            codes = convert_bytes_to_codes(bitstream_raw)
            text = codes_to_text(codes)

            self._show_text_preview(text)
            self._text_content = text
            self._codes = codes
            self._bitstream = bitstream_raw
            self._compressed_path = path

            tmp_txt = os.path.join(self._tmpdir, "_decoded.txt")
            with open(tmp_txt, "w") as f:
                f.write(text)
            self._display_metrics_text_from_sizes(
                original_size=len(text.encode("utf-8")),
                compressed_path=path,
            )
            self._set_status("Decompressed text file")

        except Exception as exc:
            messagebox.showerror("Decompression Error", str(exc))
            self._set_status("Decompression failed")

    # ------------------------------------------------------------------
    # Metrics display
    # ------------------------------------------------------------------
    def _display_metrics_text(self, compressed_path: str) -> None:
        """Compute and display metrics for text compression."""
        if self._src_path is None:
            return

        orig_size = calculate_file_size(self._src_path)
        comp_size = calculate_file_size(compressed_path)
        cr = calculate_compression_ratio(self._src_path, compressed_path)
        cf = calculate_compression_factor(self._src_path, compressed_path)
        ss = calculate_space_saving(self._src_path, compressed_path)

        raw_bytes = open_text_file(self._src_path).encode(
            "latin-1",
            errors="replace",
        )
        entropy = calculate_entropy(np.frombuffer(raw_bytes, dtype=np.uint8))
        acl = calculate_average_code_length(self._bitstream, self._codes)

        self._set_metric_values(
            orig_size=orig_size,
            comp_size=comp_size,
            cr=cr,
            cf=cf,
            ss=ss,
            entropy=entropy,
            acl=acl,
        )

    def _display_metrics_text_from_sizes(
        self,
        original_size: int,
        compressed_path: str,
    ) -> None:
        """Metrics when we only know original byte count (decompression)."""
        comp_size = calculate_file_size(compressed_path)
        cr = comp_size / original_size if original_size else 0.0
        cf = original_size / comp_size if comp_size else 0.0
        ss = (original_size - comp_size) / original_size if original_size else 0.0

        if self._text_content:
            raw = self._text_content.encode("latin-1", errors="replace")
            entropy = calculate_entropy(np.frombuffer(raw, dtype=np.uint8))
        else:
            entropy = 0.0
        acl = calculate_average_code_length(self._bitstream, self._codes)

        self._set_metric_values(
            orig_size=original_size,
            comp_size=comp_size,
            cr=cr,
            cf=cf,
            ss=ss,
            entropy=entropy,
            acl=acl,
        )

    def _display_metrics_image(
        self,
        original_path: str,
        compressed_path: str,
    ) -> None:
        """Compute and display metrics for image compression."""
        orig_size = calculate_file_size(original_path)
        comp_size = calculate_file_size(compressed_path)
        cr = calculate_compression_ratio(original_path, compressed_path)
        cf = calculate_compression_factor(original_path, compressed_path)
        ss = calculate_space_saving(original_path, compressed_path)

        if self._active_array is not None:
            entropy = calculate_entropy(self._active_array.flatten())
        else:
            entropy = 0.0
        acl = calculate_average_code_length(self._bitstream, self._codes)

        self._set_metric_values(
            orig_size=orig_size,
            comp_size=comp_size,
            cr=cr,
            cf=cf,
            ss=ss,
            entropy=entropy,
            acl=acl,
        )

    def _set_metric_values(  # noqa: PLR0913
        self,
        *,
        orig_size: int | float,
        comp_size: int | float,
        cr: float,
        cf: float,
        ss: float,
        entropy: float,
        acl: float,
    ) -> None:
        """Write pre-computed values into the metric labels."""
        self._metric_labels["orig_size"].config(text=self._fmt_size(orig_size))
        self._metric_labels["comp_size"].config(text=self._fmt_size(comp_size))
        self._metric_labels["cr"].config(text=f"{cr:.4f}")
        self._metric_labels["cf"].config(text=f"{cf:.2f}x")
        self._metric_labels["ss"].config(text=f"{ss * 100:.2f}%")
        self._metric_labels["entropy"].config(text=f"{entropy:.4f} bits")
        self._metric_labels["acl"].config(text=f"{acl:.4f} bits")
        self._metric_labels["ncodes"].config(text=f"{len(self._codes):,}")

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
    def _on_clear(self, silent: bool = False) -> None:
        """Reset all state and widgets."""
        self._src_path = None
        self._is_text = False
        self._is_image = False
        self._image_array = None
        self._channel_arrays = {}
        self._active_channel = "Grayscale"
        self._active_array = None
        self._text_content = None
        self._compressed_path = None
        self._codes = []
        self._bitstream = b""

        self._lbl_file.config(text="No file loaded", foreground="gray")
        self._lbl_image.config(image="", text="No image")  # type: ignore[arg-type]
        self._lbl_image.image = None  # type: ignore[attr-defined]
        self._txt_preview.config(state="normal")
        self._txt_preview.delete("1.0", "end")
        self._txt_preview.config(state="disabled")
        self._channel_combo.config(state="disabled", values=["Grayscale"])
        self._channel_var.set("Grayscale")
        self._reset_metrics()
        if not silent:
            self._set_status("Cleared")


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """Launch the LZW Compression GUI."""
    root = tk.Tk()
    LZWCompressionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
