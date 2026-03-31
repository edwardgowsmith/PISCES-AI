#!/usr/bin/env python3
"""
Create a mask from a NetCDF file based on the open_ocean variable.
Maps True -> 0 and False -> 1, then saves to a .npy file.
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path


def create_mask_from_netcdf(netcdf_file: str, output_file: str = None) -> None:
    """
    Create a mask from a NetCDF file based on the open_ocean variable.
    
    Parameters
    ----------
    netcdf_file : str
        Path to the input NetCDF file
    output_file : str, optional
        Path to save the output .npy file. If None, uses the input filename
        with .npy extension instead of .nc
    """
    # Load the NetCDF file
    ds = xr.open_dataset(netcdf_file)
    
    if 'open_ocean' not in ds:
        raise ValueError(f"Variable 'open_ocean' not found in {netcdf_file}")
    
    # Get the open_ocean variable
    open_ocean = ds['open_ocean'].values
    
    # Create mask: 0 -> False, 1-5 -> True
    # This is equivalent to converting any non-zero value to True
    mask = (open_ocean > 0).astype(bool)
    
    # Determine output filename
    if output_file is None:
        input_path = Path(netcdf_file)
        output_file = str(input_path.with_suffix('.npy'))
    
    # Save to .npy file
    np.save(output_file, mask)
    print(f"Mask saved to {output_file}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a mask from a NetCDF file based on open_ocean variable"
    )
    parser.add_argument(
        "netcdf_file",
        type=str,
        help="Path to the input NetCDF file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the output .npy file (default: same name as input with .npy extension)"
    )
    
    args = parser.parse_args()
    create_mask_from_netcdf(args.netcdf_file, args.output)
