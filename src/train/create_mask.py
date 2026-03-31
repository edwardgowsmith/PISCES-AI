import numpy as np
import argparse
import os

def get_nonzero_mask(npy_file):
    # Load the .npy file
    data = np.load(npy_file)  # shape: [5840, 180, 360]
    
    # Take the first timestep
    first_step = data[0]  # shape: [180, 360]
    
    # Create a boolean mask of non-nan elements
    mask = ~np.isnan(first_step)  # True where non-nan
        
    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate non-zero mask from .npy file")
    parser.add_argument("npy_file", type=str, help="Path to the .npy file")
    parser.add_argument("out_file", type=str, help="Path to save the mask .npy file")
    args = parser.parse_args()
    
    mask = get_nonzero_mask(args.npy_file)
    
    print("Mask shape:", mask.shape)
    print("Number of True elements:", np.sum(mask))
    
    # Save the mask
    np.save(args.out_file, mask)
    print(f"Mask saved to {os.path.abspath(args.out_file)}")
