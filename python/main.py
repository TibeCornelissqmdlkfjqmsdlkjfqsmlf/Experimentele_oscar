from __future__ import annotations
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# fit data between 0-1 linear converion factor
def normalize(data):
    data = np.asarray(data, dtype=float)

    max_val = data.max()

    if max_val == 0:
        return np.zeros_like(data)

    return data / max_val
def normalize_between( data, middle ):
    s = sum( data )
    av = s/ len(data)
    dist = av - middle
    result = [ d - dist for d in data ]

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

def load_data_set():

    to_plot = {
        ("../ruwe_data/mod-dbm-LP_30.0W/cf_2870-md_65-dbm_","-N_1-ds_600-mr_2"): ["8","16"],
        ("../ruwe_data/mod-mr/cf_2870-md_65-dbm_16-N_1-ds_600-mr_",""): ["1","3"],
    }
    datasets = []
    fig, axes = plt.subplots(1,len(to_plot),figsize=(10,10))

    axis_index = 0

    for (prefix, suffix), labels in to_plot.items():
        folders = [Path(prefix + label + suffix) for label in labels]
        #axis = axes[axis_index]
        #axis.grid(True)
        #axis.set_title(prefix + suffix)
        for folder, f_label in zip(folders, labels):

            odmr_path = folder / "odmr.txt"
            sweep_path = folder / "sweep.txt"

            if not odmr_path.exists() or not sweep_path.exists():
                print(f"Missing files in {folder}")
                continue

            try:
                x = load_decimal_comma_stream(str(sweep_path))
                y = load_decimal_comma_stream(str(odmr_path))
                y_n = normalize(y)
            except ValueError as e:
                print(f"Skipping {folder}: {e}")
                continue

            if x.size != y.size:
                print(f"Length mismatch in {folder}")
                continue

            #axis.plot(x, y_n, linewidth=1, label=f"mr_{f_label}")
            #axis.legend()
#
            #axis_index += 1


            datasets.append((f_label, x, y_n))

        return datasets

def plot_datasets(datasets):
    plt.figure()

    for label, x, y in datasets:
        plt.plot(x, y, linewidth=1, label=f"mr_{label}")

def filter_x_datasets(datasets, x_min, x_max ):
    filtered = []
    
    for label, x, y in datasets:
        mask = ( x >= x_min ) & ( x <= x_max )
        x_f = x[mask]
        y_f = y[mask]
        
        filtered.append((label, x_f,y_f))

    return filtered


def plot_f():
    to_plot = {
        ("../ruwe_data/mod-mr/cf_2870-md_65-dbm_16-N_1-ds_600-mr_",""): 
            [
                "1","3","2","4"
            ],
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
                y = load_decimal_comma_stream(odmr_path)
                y_n = normalize(y)
            except ValueError as e:
                print(f"Skipping {folder}: {e}")
                continue

            if x.size != y.size:
                print(f"Length mismatch in {folder}")
                continue

            plt.plot(x, y_n, linewidth=1, label=f"mr_{f_label}") 

    

def show():
    plt.xlabel("Frequency (same units as SWEEP file)")
    plt.ylabel("ODMR signal (same units as ODMR file)")
    plt.title("ODMR vs SWEEP (All mr values)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.show()

def main():
    datasets = load_data_set()
    
    datasets = filter_x_datasets(datasets=datasets,x_min=2860, x_max=2880)
    plot_datasets(datasets=datasets)

    show()


main()
