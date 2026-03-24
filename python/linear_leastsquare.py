import numpy as np
from scipy.optimize import curve_fit

from scipy.stats import linregress


def linear_model(x, a, b):
    return a * x + b

def linear_fit_scipy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    popt, pcov = curve_fit(linear_model, x, y)

    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

    return a, b, a_err, b_err

def linear_fit_numpy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Fit y = a*x + b
    a, b = np.polyfit(x, y, deg=1)

    return a, b

def find_linear_regions(x, y, window=5, r2_threshold=0.995):
    regions = []

    for i in range(len(x) - window + 1):
        xs = x[i:i+window]
        ys = y[i:i+window]

        slope, intercept, r_value, _, _ = linregress(xs, ys)
        r2 = r_value**2

        if r2 > r2_threshold:
            regions.append((i, i+window-1, r2))

    return regions


def find_linear_regions(x, y, window=10, r2_threshold=0.995):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    regions = []

    for i in range(len(x) - window + 1):
        xs = x[i:i + window]
        ys = y[i:i + window]

        result = linregress(xs, ys)
        r2 = result.rvalue ** 2

        if r2 >= r2_threshold:
            regions.append(
                {
                    "start_index": i,
                    "end_index": i + window - 1,
                    "x_start": xs[0],
                    "x_end": xs[-1],
                    "slope": result.slope,
                    "intercept": result.intercept,
                    "r2": r2
                }
            )

    return regions

def print_linear_regions(regions):
    if not regions:
        print("No linear regions found.")
        return

    for r in regions:
        print(
            f"[{r['x_start']} -> {r['x_end']}] "
            f"intercept={r['intercept']}"
            f"slope={r['slope']:.6f}, "
            f"R²={r['r2']:.6f}"
        )

def main():
    laser = [30,40,50]
    m_direct = [9.52,36.3,63.7]
    m_diamond = [2.54,10.24,17.88]
    
    pot = np.array([
        0,10,20,30,40,50,60,70,80,
        90,100,110,120,130,140,150,160,170,180,190,
        200,210,220,230,240,250,260,270,280,290,
        300,500,700,900,1000
    ], dtype=float)

    laser = np.array([
        64.5,63.7,62.2,59.9,56.7,52.6,48.7,43.5,39.0,
        34.5,30.5,26.9,23.6,20.6,18.7,15.7,13.77,11.89,10.27,8.78,
        7.44,6.27,5.02,4.07,3.09,2.78,1.4,0.73,0.27,0.052,
        0.035,0.02237,0.00865,0.00673,0.00606
    ], dtype=float)

    # === Global linear fit (for reference) ===
    a, b, a_err, b_err = linear_fit_scipy(pot, laser)

    print("Global fit:")
    print(f"I(x) = {a:.6f} x + {b:.6f}")
    print(f"uncertainties: da={a_err:.6f}, db={b_err:.6f}")

    # === Linear region detection ===
    regions = find_linear_regions(
        pot,
        laser,
        window=8,
        r2_threshold=0.998   # stricter because dataset is smooth
    )

    print("\nDetected linear regions:")
    print_linear_regions(regions)

    # === OPTIONAL: log-transform (physically meaningful) ===
    log_laser = np.log(laser)

    log_regions = find_linear_regions(
        pot,
        log_laser,
        window=6,
        r2_threshold=0.998
    )

    print("\nLinear regions in log-space (exponential behavior):")
    print_linear_regions(log_regions)



    return 


main()