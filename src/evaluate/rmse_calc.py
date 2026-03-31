import numpy as np
import os
import sys

def calculate_average_rmse(file_path_real, file_path_preds):
    """
    Calculates the RMSE per timestep and returns the average over all timesteps.
    """
    # Verify files exist
    if not os.path.exists(file_path_real):
        print(f"Error: File not found: {file_path_real}")
        return None, []
    if not os.path.exists(file_path_preds):
        print(f"Error: File not found: {file_path_preds}")
        return None, []

    # Load the datasets (expected shape: [T, H, W])
    val_ys = np.load(file_path_real)
    val_preds = np.load(file_path_preds)

    if val_ys.shape != val_preds.shape:
        print(f"Error: Shape mismatch! {val_ys.shape} vs {val_preds.shape}")
        return None, []

    rmse_s = []

    for j in range(len(val_ys)):
        values = val_ys[j]
        prediction = val_preds[j]

        # Mask valid locations (non-NaN in both arrays)
        mask = ~np.isnan(values) & ~np.isnan(prediction)
        n_valid = mask.sum()

        if n_valid > 0:
            se = (values[mask] - prediction[mask]) ** 2
            rmse = np.sqrt(se.sum() / n_valid)
            rmse_s.append(rmse)
        
    if len(rmse_s) > 0:
        return np.mean(rmse_s), rmse_s
    return None, []

if __name__ == "__main__":
    # Check if correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python calculate_rmse.py <real_file.npy> <preds_file.npy>")
        sys.exit(1)

    real_input = sys.argv[1]
    preds_input = sys.argv[2]

    avg_rmse, all_rmses = calculate_average_rmse(real_input, preds_input)

    if avg_rmse is not None:
        print(f"Results for: {os.path.basename(preds_input)}")
        print(f"Mean RMSE across {len(all_rmses)} timesteps: {avg_rmse:.6f}")
    else:
        print("Failed to calculate RMSE.")