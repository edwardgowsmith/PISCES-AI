import argparse
import yaml
import pickle
import time
import torch
import numpy as np
from src.models.unet import UNet
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--local-config", required=True)
    p.add_argument("--checkpoint", required=True)

    p.add_argument("--inputs", nargs="+", required=True,
                   help="List of var=path.npy (one timestep each)")
    p.add_argument("--chl", required=True)
    p.add_argument("--dpco2", required=True)

    p.add_argument("--norm-pkl", required=True)
    p.add_argument("--mask-npy", required=True)

    p.add_argument("--co2-pkl", default=None,
                   help="Pickle containing co2_by_year dict")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--fixed-co2", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # --------------------
    # Config
    # --------------------
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.local_config) as f:
        cfg.update(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Load stats
    # --------------------
    with open(args.norm_pkl, "rb") as f:
        norm = pickle.load(f)

    in_mean = norm["in_mean"]
    in_std = norm["in_std"]
    chl_mean, chl_std = norm["chl_mean"], norm["chl_std"]
    dpco2_mean, dpco2_std = norm["dpco2_mean"], norm["dpco2_std"]

    # --------------------
    # Mask
    # --------------------
    mask = np.load(args.mask_npy).astype(bool)

    # --------------------
    # CO2
    # --------------------
    co2_by_year = None
    if cfg.get("co2_file"):
        co2_df = pd.read_csv(cfg["co2_file"], delim_whitespace=True)
        # keep per-year scalar fallback mapping
        co2_by_year = {int(row['Year']): float(row['atmCO2']) for _, row in co2_df.iterrows()}
        co2_values = np.array(list(co2_by_year.values()))
        co2_mean = np.mean(co2_values)
        co2_std = np.std(co2_values) + 1e-9
    # --------------------
    # Run preprocessing multiple times and collect timings
    # --------------------
    n_pre_runs = 10000
    load_inputs_ms_list = []
    stack_inputs_ms_list = []
    co2_norm_ms_list = []
    to_tensor_ms_list = []
    build_outputs_ms_list = []
    total_pre_ms_list = []

    last_x = None
    last_y = None

    # warm up CUDA
    torch.zeros(1).to(device)
    torch.cuda.synchronize()

    # Warm up file cache + get shape
    inputs_by_var_sample = {}
    for item in args.inputs:
        var, path = item.split("=")
        arr = np.load(path)
        inputs_by_var_sample[var] = arr
    _ = np.load(args.chl)
    _ = np.load(args.dpco2)

    # allocate once outside loop
    H, W = inputs_by_var_sample[list(inputs_by_var_sample.keys())[0]].shape
    C = len(inputs_by_var_sample)
    if co2_by_year is not None:
        C += 1  # add CO2 channel
    x_pinned = torch.empty((1, C, H, W), dtype=torch.float32, pin_memory=True)

    for _ in range(n_pre_runs):
        t0 = time.perf_counter()

        inputs_by_var = {}
        for item in args.inputs:
            var, path = item.split("=")
            arr = np.load(path)
            if arr.ndim != 2:
                raise ValueError(f"{var} must be (H,W), got {arr.shape}")
            inputs_by_var[var] = arr
        t_load_inputs = time.perf_counter()

        x_np = np.stack([inputs_by_var[v] for v in inputs_by_var.keys()], axis=0)  # (C,H,W)
        t_stack = time.perf_counter()

        H, W = x_np.shape[1:]

        # CO2 channel + normalization
        if co2_by_year is not None:
            co2_val = (co2_by_year[args.year] - co2_mean) / co2_std
            co2_map = np.full((1, H, W), co2_val, dtype=np.float32)
            x_np = np.concatenate([x_np, co2_map], axis=0)
            ext_mean = np.concatenate([in_mean, [0.0]])
            ext_std = np.concatenate([in_std, [1.0]])
            x_np = (x_np - ext_mean[:, None, None]) / ext_std[:, None, None]
        else:
            x_np = (x_np - in_mean[:, None, None]) / in_std[:, None, None]
        t_after_co2 = time.perf_counter()

        x_np = np.nan_to_num(x_np, nan=0.0)
        # inside loop
        x_pinned[0].copy_(torch.from_numpy(x_np))
        x_tensor = x_pinned.to(device, non_blocking=True)
        # x_tensor = torch.from_numpy(x_np).pin_memory().unsqueeze(0).to(device, dtype=torch.float32, non_blocking=True)
        # ensure async GPU copies finish before timing when using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_tensor = time.perf_counter()

        # Outputs (normalised, for completeness)
        chl = np.load(args.chl)
        dpco2 = np.load(args.dpco2)

        y_np = np.concatenate([
            np.where(mask, (chl - chl_mean) / chl_std, 0.0)[None],
            np.where(mask, (dpco2 - dpco2_mean) / dpco2_std, 0.0)[None],
        ], axis=0)
        y_tensor = torch.from_numpy(y_np).float().unsqueeze(0).to(device)
        # ensure y transfer complete before taking final preprocessing time
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # record timings
        load_inputs_ms_list.append((t_load_inputs - t0) * 1000.0)
        stack_inputs_ms_list.append((t_stack - t_load_inputs) * 1000.0)
        co2_norm_ms_list.append((t_after_co2 - t_stack) * 1000.0)
        to_tensor_ms_list.append((t_tensor - t_after_co2) * 1000.0)
        build_outputs_ms_list.append((t1 - t_tensor) * 1000.0)
        total_pre_ms_list.append((t1 - t0) * 1000.0)

        last_x = x_tensor
        last_y = y_tensor

    # use last produced tensors as inputs for model/run
    x = last_x
    y = last_y

    # compute mean/std for reporting
    import statistics
    load_inputs_ms = float(statistics.mean(load_inputs_ms_list))
    stack_inputs_ms = float(statistics.mean(stack_inputs_ms_list))
    co2_norm_ms = float(statistics.mean(co2_norm_ms_list))
    to_tensor_ms = float(statistics.mean(to_tensor_ms_list))
    build_outputs_ms = float(statistics.mean(build_outputs_ms_list))
    total_pre_ms = float(statistics.mean(total_pre_ms_list))

    # # Print individual "to tensor" times for each run
    # print("=== Individual 'to tensor' times (ms) ===")
    # for i, time_ms in enumerate(to_tensor_ms_list, 1):
    #     print(f"Run {i}: {time_ms:.2f} ms")
    # print()

    # standard deviations for preprocessing sub-steps
    load_inputs_std = float(np.std(load_inputs_ms_list))
    stack_inputs_std = float(np.std(stack_inputs_ms_list))
    co2_norm_std = float(np.std(co2_norm_ms_list))
    to_tensor_std = float(np.std(to_tensor_ms_list))
    build_outputs_std = float(np.std(build_outputs_ms_list))
    total_pre_std = float(np.std(total_pre_ms_list))

    # --------------------
    # Model
    # --------------------
    climatology_vars = cfg.get("climatology_vars", [])
    n_channels = len(cfg["input_vars"]) + len(climatology_vars)
    if co2_by_year is not None:
        n_channels += 1

    
    model = UNet(n_channels=n_channels, n_classes=2)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # --------------------
    # Warm-up
    # --------------------
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # --------------------
    # Forward timing: run multiple times and average
    # --------------------
    n_runs = 10
    run_times_ms = []
    pred = None
    with torch.no_grad():
        for i in range(n_runs):
            t_run_start = time.perf_counter()
            pred = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_run_end = time.perf_counter()
            run_times_ms.append((t_run_end - t_run_start) * 1000.0)
    avg_time = float(np.mean(run_times_ms))
    std_time = float(np.std(run_times_ms))
 
    # --------------------
    # Report
    # --------------------
    print("=== Single-timestep forward ===")
    print(f"Device        : {device}")
    print(f"Input shape   : {tuple(x.shape)}")
    print(f"Output shape  : {tuple(pred.shape)}")
    print()
    print(f"Preprocess    : {total_pre_ms:.2f} ms ± {total_pre_std:.2f} ms (n={n_pre_runs})")
    print(f"  load inputs : {load_inputs_ms:.2f} ms ± {load_inputs_std:.2f} ms")
    print(f"  stack       : {stack_inputs_ms:.2f} ms ± {stack_inputs_std:.2f} ms")
    print(f"  co2 & norm  : {co2_norm_ms:.2f} ms ± {co2_norm_std:.2f} ms")
    print(f"  to tensor   : {to_tensor_ms:.2f} ms ± {to_tensor_std:.2f} ms")
    print(f"  build y     : {build_outputs_ms:.2f} ms ± {build_outputs_std:.2f} ms")
    print(f"Forward pass  : mean {avg_time:.2f} ms ± {std_time:.2f} ms (n={n_runs})")


if __name__ == "__main__":
    main()
