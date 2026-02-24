from __future__ import annotations
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_decimal_comma_stream(path: str | Path) -> np.ndarray:
    """
    Loads a text file containing numbers formatted with decimal commas,
    separated by arbitrary whitespace (tabs/spaces/newlines), into a 1D array.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    # Split on any whitespace, then convert decimal comma -> dot
    tokens = text.split()
    vals = np.array([float(tok.replace(",", ".")) for tok in tokens], dtype=float)
    return vals

fold_front = r"D:\Jelle_Tibe_Experimentele\ruwe_data\mod-mr\cf_2870-md_65-dbm_16-N_1-ds_600-mr_"
freq = ["1","3","4","5"]

fold_back = ""

folders = [fold_front + f + fold_back for f in freq ]
plt.figure()

for folder, f_label in zip(folders, freq):

    odmr_path = os.path.join(folder, "odmr.txt")
    sweep_path = os.path.join(folder, "sweep.txt")

    if not os.path.exists(odmr_path) or not os.path.exists(sweep_path):
        print(f"Missing files in {folder}")
        continue

    try:
        x = load_decimal_comma_stream(sweep_path)
        y = load_decimal_comma_stream(odmr_path)
    except ValueError as e:
        print(f"Skipping {folder}: {e}")
        continue

    if x.size != y.size:
        print(f"Length mismatch in {folder}")
        continue

    plt.plot(x, y, linewidth=1, label=f"mr_{f_label}")

plt.xlabel("Frequency (same units as SWEEP file)")
plt.ylabel("ODMR signal (same units as ODMR file)")
plt.title("ODMR vs SWEEP (All mr values)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.show()
