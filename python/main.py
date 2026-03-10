from __future__ import annotations
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# fit data between 0-1 linear converion factor
def normalize( data ):
    min_val = min( data )
    max_val = max( data )
    diff = max_val - min_val

    result = [ (d-min_val)/(diff) for d in data ]
    return result
    

    

def load_decimal_comma_stream(path: str | Path) -> np.ndarray:
    """
    Loads a text file containing numbers formatted with decimal dcommas,
    separated by arbitrary whitespace (tabs/spaces/newlines), into a 1D array.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    # Split on any whitespace, then convert decimal comma -> dot
    tokens = text.split()
    vals = np.array([float(tok.replace(",", ".")) for tok in tokens], dtype=float)
    return vals

to_plot = {
    ("../ruwe_data/mod-dbm-LP_30.0W/cf_2870-md_65-dbm_","-N_1-ds_600-mr_2"): ["8","16"],
    ("../ruwe_data/mod-mr/cf_2870-md_65-dbm_16-N_1-ds_600-mr_",""): ["1","3"],
}

fig, axes = plt.subplots(1,len(to_plot),figsize=(10,10))

axis_index = 0

for (prefix, suffix), labels in to_plot.items():
    folders = [Path(prefix + label + suffix) for label in labels]
    axis = axes[axis_index]
    axis.grid(True)
    axis.set_title(prefix + suffix)
    for folder, f_label in zip(folders, labels):

        odmr_path = os.path.join(folder, "odmr.txt")
        sweep_path = os.path.join(folder, "sweep.txt")

        if not os.path.exists(odmr_path) or not os.path.exists(sweep_path):
            print(f"Missing files in {folder}")
            continue

        try:
            x = load_decimal_comma_stream(sweep_path)
            x_n = normalize(x)
            y = load_decimal_comma_stream(odmr_path)
            y_n = normalize(y)
        except ValueError as e:
            print(f"Skipping {folder}: {e}")
            continue

        if x.size != y.size:
            print(f"Length mismatch in {folder}")
            continue

        axis.plot(x_n, y_n, linewidth=1, label=f"mr_{f_label}")
        axis.legend()

    axis_index += 1




plt.xlabel("Frequency (same units as SWEEP file)")
plt.ylabel("ODMR signal (same units as ODMR file)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.show()
