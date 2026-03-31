import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-ys', required=True, help='Path to ground truth numpy file')
    parser.add_argument('--val-preds', required=True, help='Path to predictions numpy file')
    parser.add_argument('--out', default='sum_plot.png', help='Path to save the plot')
    parser.add_argument('--points-per-year', type=int, default=5840, help='Number of timesteps to average per year')
    args = parser.parse_args()

    # Load arrays: shape [T, H, W]
    ys = np.load(args.val_ys, mmap_mode="r")
    preds = np.load(args.val_preds, mmap_mode="r")
    if preds.ndim == 4 and preds.shape[1] == 1:
        preds = np.squeeze(preds, axis=1)
    if ys.ndim == 4 and ys.shape[1] == 1:
        ys = np.squeeze(ys, axis=1)

    # Safety check
    if ys.shape != preds.shape:
        raise ValueError(f"Shape mismatch: ys {ys.shape}, preds {preds.shape}")

    print("Summing spatial dimensions")
    # Sum spatial dimensions -> [T]
    # use float64 accumulator to avoid overflow/precision issues

    def spatial_sum_stream(arr, chunk=128):
        T = arr.shape[0]
        out = np.zeros(T, dtype=np.float64)
        for start in tqdm(range(0, T, chunk)):
            end = min(start + chunk, T)
            # nansum over spatial dims but only for the chunk
            out[start:end] = np.nansum(arr[start:end], axis=(1, 2))
        return out

    ys_sum = spatial_sum_stream(ys, chunk=128)
    preds_sum = spatial_sum_stream(preds, chunk=128)

    print("Aggregating yearly")
    # Aggregate into non-overlapping windows of points_per_year to get one value per year.
    ppy = 5840
    def aggregate_yearly(series, window):
        n = len(series)
        n_full = n // window
        if n_full == 0:
            # not enough points for a single year: return the mean of whatever exists
            return np.array([np.nanmean(series)])
        trimmed = series[: n_full * window]
        reshaped = trimmed.reshape(n_full, window)
        return np.nanmean(reshaped, axis=1)

    ys_year = aggregate_yearly(ys_sum, ppy)
    preds_year = aggregate_yearly(preds_sum, ppy)

    # Ensure we have a floating dtype compatible with numpy.linalg routines
    ys_year = ys_year.astype(np.float64, copy=False)
    preds_year = preds_year.astype(np.float64, copy=False)

    print("Creating plot")
    # Plot
    plt.figure(figsize=(12, 4))
    t = np.arange(len(ys_year))

    # main lines with opacity
    l1, = plt.plot(t, ys_year, label='Ground Truth (val_ys) — yearly mean', alpha=0.8, marker='o')
    l2, = plt.plot(t, preds_year, label='Predictions (val_preds) — yearly mean', alpha=0.8, marker='o')

    plt.xlabel('Timestep')
    plt.ylabel('Sum over H×W (yearly mean)')
    plt.xlim(start, end)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
