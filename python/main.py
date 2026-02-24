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

to_plot = {
    ("../ruwe_data/mod-mr/cf_2870-md_65-dbm_16-N_1-ds_600-mr_",""): ["1","1,1","1,2","1,3","1,4","1,5","1,6","1,7","1,8","1,9","2","2,1","2,2","2,3","2,4","2,5","2,6","2,7","2,8","2,9","3"],
}


for (prefix, suffix), labels in to_plot.items():
    folders = [Path(prefix + label + suffix) for label in labels]

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

        plt.plot(x_n, y_n, linewidth=1, label=f"mr_{f_label}") 

fold_front = "../ruwe_data/mod-mr/cf_2870-md_65-dbm_16-N_1-ds_600-mr_"
freq = ["1","1,1","1,2","1,3","1,4","1,5","1,6","1,7","1,8","1,9","2","2,1","2,2","2,3","2,4","2,5","2,6","2,7","2,8","2,9","3"]

folders = [fold_front + f for f in freq ]
plt.figure()

for folder, f_label in zip(folders, freq):

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

    plt.plot(x_n, y_n, linewidth=1, label=f"mr_{f_label}")

plt.xlabel("Frequency (same units as SWEEP file)")
plt.ylabel("ODMR signal (same units as ODMR file)")
plt.title("ODMR vs SWEEP (All mr values)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.show()
