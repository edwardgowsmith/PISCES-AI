#!/usr/bin/env python3
"""
Save the first timestep from a [T, X, Y] .npy file as a [X, Y] .npy file.
"""
from pathlib import Path
import argparse
import numpy as np
import sys

def main():
    p = argparse.ArgumentParser(description="Extract first timestep from [T,X,Y] .npy -> [X,Y] .npy")
    p.add_argument("input", type=Path, help="Input .npy file (shape [T,X,Y])")
    p.add_argument("output", type=Path, help="Output .npy file (shape [X,Y])")
    p.add_argument("--index", type=int, default=0, help="Timestep index to extract (default: 0)")
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    if args.output.exists() and not args.force:
        print(f"Error: output exists ({args.output}); use --force to overwrite", file=sys.stderr)
        sys.exit(3)

    arr = np.load(str(args.input), mmap_mode="r")
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3:
        T = arr.shape[0]
        if not (0 <= args.index < T):
            print(f"Error: index {args.index} out of range [0, {T-1}]", file=sys.stderr)
            sys.exit(4)
        out = arr[args.index]
    elif arr.ndim == 4:
        # Handle 4D arrays (e.g., [T, C, X, Y]) by squeezing singleton dimensions
        T = arr.shape[0]
        if not (0 <= args.index < T):
            print(f"Error: index {args.index} out of range [0, {T-1}]", file=sys.stderr)
            sys.exit(4)
        out = arr[args.index]
        out = np.squeeze(out)
    else:
        print(f"Error: unsupported input shape {arr.shape}, expected 2, 3, or 4 dims", file=sys.stderr)
        sys.exit(5)

    # ensure 2D array
    if out.ndim != 2:
        print(f"Error: extracted slice has unexpected shape {out.shape}", file=sys.stderr)
        sys.exit(6)

    np.save(str(args.output), out.astype(np.float32))
    print(f"Wrote {args.output} (shape {out.shape})")


if __name__ == "__main__":
    main()