from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from .install_types import InstallOptions
from .utils import split_languages


def _browse_dir(entry: ttk.Entry) -> None:
    selected = filedialog.askdirectory()
    if selected:
        entry.delete(0, tk.END)
        entry.insert(0, selected)


def run_minimal_gui(default_models_dir: Path) -> Optional[InstallOptions]:
    root = tk.Tk()
    root.title("Chart Analysis Installer")
    root.geometry("640x460")
    root.resizable(False, False)

    options = InstallOptions()

    purpose_var = tk.StringVar(value="user")
    scope_var = tk.StringVar(value="local")
    mode_var = tk.StringVar(value="gui")
    ocr_var = tk.StringVar(value="EasyOCR")
    languages_var = tk.StringVar(value="en,pt")
    predownload_var = tk.BooleanVar(value=False)
    tests_var = tk.BooleanVar(value=False)
    auto_python_var = tk.BooleanVar(value=False)
    profile_var = tk.StringVar(value="default")

    models_var = tk.StringVar(value=str(default_models_dir))
    easyocr_cache_var = tk.StringVar(value=str((Path.home() / ".EasyOCR").resolve()))
    paddle_cache_var = tk.StringVar(value=str((Path.home() / ".paddle").resolve()))

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)

    def _row(label: str, widget: tk.Widget) -> None:
        r = len(frame.grid_slaves(column=0))
        ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=5)
        widget.grid(row=r, column=1, sticky="ew", pady=5)

    frame.columnconfigure(1, weight=1)

    _row("Purpose", ttk.Combobox(frame, textvariable=purpose_var, values=["user", "developer"], state="readonly"))
    _row("Install scope", ttk.Combobox(frame, textvariable=scope_var, values=["local", "user", "global"], state="readonly"))
    _row("Interface", ttk.Combobox(frame, textvariable=mode_var, values=["gui", "cli"], state="readonly"))
    _row("OCR backend", ttk.Combobox(frame, textvariable=ocr_var, values=["EasyOCR", "Paddle"], state="readonly"))

    _row("OCR languages", ttk.Entry(frame, textvariable=languages_var))
    _row("Profile name", ttk.Entry(frame, textvariable=profile_var))

    models_row = ttk.Frame(frame)
    models_entry = ttk.Entry(models_row, textvariable=models_var)
    models_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(models_row, text="Browse", command=lambda: _browse_dir(models_entry)).pack(side=tk.LEFT, padx=(8, 0))
    _row("Models directory", models_row)

    easy_row = ttk.Entry(frame, textvariable=easyocr_cache_var)
    _row("EasyOCR cache dir", easy_row)

    paddle_row = ttk.Entry(frame, textvariable=paddle_cache_var)
    _row("Paddle cache dir", paddle_row)

    toggles = ttk.Frame(frame)
    toggles.grid(row=9, column=0, columnspan=2, sticky="w", pady=10)
    ttk.Checkbutton(toggles, text="Pre-download EasyOCR models", variable=predownload_var).pack(anchor="w")
    ttk.Checkbutton(toggles, text="Include test/dev tools", variable=tests_var).pack(anchor="w")
    ttk.Checkbutton(toggles, text="Attempt auto Python install workflow", variable=auto_python_var).pack(anchor="w")

    result: dict = {"options": None}

    def on_cancel() -> None:
        root.destroy()

    def on_install() -> None:
        try:
            langs = split_languages(languages_var.get())
            if not profile_var.get().strip():
                raise ValueError("Profile name is required")

            options.purpose = purpose_var.get()
            options.install_scope = scope_var.get()
            options.interface_mode = mode_var.get()
            options.ocr_backend = ocr_var.get()
            options.ocr_languages = langs
            options.predownload_ocr_models = predownload_var.get()
            options.include_test_tools = tests_var.get()
            options.auto_install_python = auto_python_var.get()
            options.profile_name = profile_var.get().strip()
            options.models_dir = Path(models_var.get()).expanduser()
            options.easyocr_model_storage_dir = Path(easyocr_cache_var.get()).expanduser()
            options.paddle_model_cache_dir = Path(paddle_cache_var.get()).expanduser()

            result["options"] = options
            root.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid options", str(exc))

    buttons = ttk.Frame(frame)
    buttons.grid(row=10, column=0, columnspan=2, sticky="e", pady=(8, 0))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(buttons, text="Install", command=on_install).pack(side=tk.RIGHT)

    root.mainloop()
    return result["options"]
