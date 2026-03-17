import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
from scipy import constants

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

# fit data between 0-1 linear converion factor
def normalize(data):
    data = np.asarray(data, dtype=float)

    max_val = data.max()

    if max_val == 0:
        return np.zeros_like(data)

    return data / max_val

def find_odmr_peaks(
    x: np.ndarray,
    y: np.ndarray,
    expected_peaks: int = 8,
    smooth: bool = True,
    window_length: int = 11,
    polyorder: int = 3,
    min_distance_fraction: float = 0.003,
    prominence: float | None = None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 5:
        raise ValueError("x and y are too short")

    y_n = normalize(y)

    if smooth:
        wl = min(window_length, len(y_n))
        if wl % 2 == 0:
            wl -= 1
        if wl <= polyorder:
            wl = polyorder + 2
            if wl % 2 == 0:
                wl += 1
        if wl >= len(y_n):
            wl = len(y_n) - 1 if len(y_n) % 2 == 0 else len(y_n)
        if wl < 3:
            y_smooth = y_n.copy()
        else:
            y_smooth = savgol_filter(y_n, wl, min(polyorder, wl - 1))
    else:
        y_smooth = y_n.copy()

    # Invert because dips in y become peaks in -y
    y_inv = -y_smooth

    min_distance = max(1, int(len(x) * min_distance_fraction))

    if prominence is None:
        # Heuristic: small fraction of signal range
        prominence = 0.02 * (np.max(y_smooth) - np.min(y_smooth))

    peak_indices, properties = find_peaks(
        y_inv,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peak_indices) == 0:
        return {
            "indices": np.array([], dtype=int),
            "x_peaks": np.array([], dtype=float),
            "y_peaks": np.array([], dtype=float),
            "y_smooth": y_smooth,
            "properties": properties,
        }

    # Keep the strongest expected_peaks by prominence
    if len(peak_indices) > expected_peaks:
        order = np.argsort(properties["prominences"])[::-1]
        keep = np.sort(order[:expected_peaks])

        peak_indices = peak_indices[keep]
        for key in properties:
            properties[key] = properties[key][keep]

    # Sort peaks by x-position
    sort_idx = np.argsort(peak_indices)
    peak_indices = peak_indices[sort_idx]
    for key in properties:
        properties[key] = properties[key][sort_idx]

    return {
        "indices": peak_indices,
        "x_peaks": x[peak_indices],
        "y_peaks": y_smooth[peak_indices],
        "y_smooth": y_smooth,
        "properties": properties,
    }


def cut_peak_window(x, y, peak_index, half_width=30):
    left = max(0, peak_index - half_width)
    right = min(len(x), peak_index + half_width + 1)

    return x[left:right], y[left:right], left, right

def voigt_model(x, c0, A, f0, sigma, gamma):
    return c0 - A * voigt_profile( x - f0, sigma, gamma )

def residuals(params, x, y):
    c0, A, f0, sigma, gamma = params
    return y - voigt_model(x, c0, A, f0, sigma, gamma)

def voigt_fit(relative_path_odmr, relative_path_sweep):
    x = load_decimal_comma_stream(str(relative_path_sweep))
    y = load_decimal_comma_stream(str(relative_path_odmr))

    result = find_odmr_peaks(x, y, expected_peaks=24, polyorder=6)

    #print("Detected peak positions:")
    #for i, (xp, yp) in enumerate(zip(result["x_peaks"], result["y_peaks"]), start=1):
    #    print(f"{i}: x = {xp:.6f}, y = {yp:.6f}")

    # choose one detected peak
    peak_number = 17
    peak_index = result["indices"][peak_number]
    f0_guess = x[peak_index]

    # local window around that peak
    x_cut, y_cut, left, right = cut_peak_window(
        x,
        result["y_smooth"],
        peak_index,
        half_width=15
    )

    # better initial guesses
    c0_guess = np.max(y_cut)
    A_guess = np.max(y_cut) - np.min(y_cut)
    sigma_guess = (x_cut[-1] - x_cut[0]) / 10.0
    gamma_guess = (x_cut[-1] - x_cut[0]) / 10.0

    p0 = [c0_guess, A_guess, f0_guess, sigma_guess, gamma_guess]

    result_least_squares = least_squares(
        residuals,
        p0,
        args=(x_cut, y_cut)
    )

    # fitted parameters
    c0_fit, A_fit, f0_fit, sigma_fit, gamma_fit = result_least_squares.x

    #print("\nFitted parameters:")
    #print(f"c0    = {c0_fit}")
    #print(f"A     = {A_fit}")
    #print(f"f0    = {f0_fit}")
    #print(f"sigma = {sigma_fit}")
    #print(f"gamma = {gamma_fit}")

    contrast , fwhm, sensitivity = sensitivity_from_fit(c0=c0_fit, A=A_fit, sigma=sigma_fit, gamma=gamma_fit, R = photon_rate(50, 480*10**-9))

    #print(f"sensitivity = {sensitivity_from_fit(c0=c0_fit, A=A_fit, sigma=sigma_fit, gamma=gamma_fit, R = photon_rate(50, 480*10**-9))}")

    # fitted curve
    #x_dense = np.linspace(x_cut.min(), x_cut.max(), 500)
    #y_dense = voigt_model(x_dense, *result_least_squares.x)
#
    #plt.figure(figsize=(10, 5))
    #plt.plot(x, result["y_smooth"], label="Smoothed ODMR")
    #plt.plot(result["x_peaks"], result["y_peaks"], "rx", label="Detected dips")
    #plt.axvline(f0_guess, linestyle="--", label="Chosen peak")
    #plt.grid(True)
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
#
    #plt.figure(figsize=(8, 5))
    #plt.plot(x_cut, y_cut, "o", label="Local data")
    #plt.plot(x_dense, y_dense, "-", label="Voigt fit")
    #plt.axvline(f0_fit, linestyle="--", label="Fitted f0")
    #plt.grid(True)
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    return contrast, fwhm, sensitivity

    

def contrast_from_voigt_profile(c0, A, sigma, gamma):
    dip_depth = A * voigt_profile(0.0, sigma, gamma )
    return dip_depth/c0
def FWHM_from_voit_profile(sigma, gamma):
    gamma_L = 2.0 * gamma
    gamma_G = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    return 0.5346 * gamma_L + np.sqrt(0.2166 * gamma_L**2 + gamma_G**2)

def photon_rate( power, wave_length ):
    return (power * wave_length )/ ( constants.h * constants.c )


def sensitivity_from_fit(c0, A, sigma, gamma, R):
    C = contrast_from_voigt_profile(c0, A, sigma, gamma)
    fwhm = FWHM_from_voit_profile(sigma, gamma)
    sens = fwhm / (C * np.sqrt(R))
    return C, fwhm, sens

def main():
    odmr_paths = []
    sweep_paths = []

    results = []

    base_paths = []
    base_paths.append((Path("../ruwe_data/mod-mr/cf_2870-md_65-dbm_16-N_1-ds_600-mr_1"),"Name"))

    for base_path, name in base_paths:
        odmr_path = Path( base_path / "odmr.txt")
        sweep_path = Path( base_path / "sweep.txt")
        odmr_paths.append( Path( base_path / "odmr.txt") )
        sweep_paths.append( Path( base_path / "sweep.txt") )

        results.append((voigt_fit( odmr_path, sweep_path ), name) )
        
    for (constrast, fwhm, sens), name in results:
        print(f"name plot : {name}")
        print( f"\tconstrats = {constrast}")
        print(f"\tfull width half maximum = {fwhm}")
        print(f"\tsens = { sens }")

main()
    