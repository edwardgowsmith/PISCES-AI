import argparse
import yaml
import pickle
import time
import torch
import numpy as np
from scipy import ndimage
from pathlib import Path
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
    
    p.add_argument("--remap-grid", default="r360x180")

    p.add_argument("--norm-pkl", required=True)
    p.add_argument("--mask-npy", required=True)

    p.add_argument("--co2-pkl", default=None)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--fixed-co2", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()

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

    mask = np.load(args.mask_npy).astype(bool)

    # --------------------
    # CO2
    # --------------------
    co2_by_year = None
    if cfg.get("co2_file"):
        co2_df = pd.read_csv(cfg["co2_file"], delim_whitespace=True)
        co2_by_year = {int(row['Year']): float(row['atmCO2']) for _, row in co2_df.iterrows()}
        co2_values = np.array(list(co2_by_year.values()))
        co2_mean = np.mean(co2_values)
        co2_std = np.std(co2_values) + 1e-9

    # --------------------
    # Preprocessing timing
    # --------------------
    n_pre_runs = 10
    remap_input_ms_list = []
    load_inputs_ms_list = []
    stack_inputs_ms_list = []
    co2_norm_ms_list = []
    to_tensor_ms_list = []
    build_outputs_ms_list = []
    total_pre_ms_list = []

    target_h, target_w = 180, 360

    last_x = None
    last_y = None

    torch.zeros(1).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    inputs_by_var_sample = {}
    for item in args.inputs:
        var, path = item.split("=")
        arr = np.load(path)
        inputs_by_var_sample[var] = arr

    sample_arr = next(iter(inputs_by_var_sample.values()))
    zoom_factors = (target_h / sample_arr.shape[0], target_w / sample_arr.shape[1])
    sample_remapped = ndimage.zoom(sample_arr, zoom_factors, order=0)
    H, W = sample_remapped.shape

    chl = np.load(args.chl)
    dpco2 = np.load(args.dpco2)

    y_np = np.concatenate([
        np.where(mask, (chl - chl_mean) / chl_std, 0.0)[None],
        np.where(mask, (dpco2 - dpco2_mean) / dpco2_std, 0.0)[None],
    ], axis=0)
    y_tensor = torch.from_numpy(y_np).float().unsqueeze(0).to(device)

    for _ in range(n_pre_runs):
        t0 = time.perf_counter()
        
        # Remap inputs
        t_remap_start = time.perf_counter()
        remapped_inputs = {}
        
        for item in args.inputs:
            var, path = item.split("=")
            arr = np.load(path)
            zoom_factors = (target_h / arr.shape[0], target_w / arr.shape[1])
            remapped_arr = ndimage.zoom(arr, zoom_factors, order=0)
            remapped_inputs[var] = remapped_arr
        
        t_remap_end = time.perf_counter()

        inputs_by_var = remapped_inputs
        t_load_inputs = time.perf_counter()

        x_np = np.stack([inputs_by_var[v] for v in inputs_by_var.keys()], axis=0)
        t_stack = time.perf_counter()

        H, W = x_np.shape[1:]

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

        x_np = np.nan_to_num(x_np, nan=0.0).astype(np.float32)
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_tensor = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        remap_input_ms_list.append((t_remap_end - t_remap_start) * 1000.0)
        load_inputs_ms_list.append((t_load_inputs - t_remap_end) * 1000.0)
        stack_inputs_ms_list.append((t_stack - t_load_inputs) * 1000.0)
        co2_norm_ms_list.append((t_after_co2 - t_stack) * 1000.0)
        to_tensor_ms_list.append((t_tensor - t_after_co2) * 1000.0)
        build_outputs_ms_list.append((t1 - t_tensor) * 1000.0)
        total_pre_ms_list.append((t1 - t0) * 1000.0)

        last_x = x_tensor
        last_y = y_tensor

    x = last_x
    y = last_y

    import statistics
    remap_input_ms = float(statistics.mean(remap_input_ms_list))
    load_inputs_ms = float(statistics.mean(load_inputs_ms_list))
    stack_inputs_ms = float(statistics.mean(stack_inputs_ms_list))
    co2_norm_ms = float(statistics.mean(co2_norm_ms_list))
    to_tensor_ms = float(statistics.mean(to_tensor_ms_list))
    build_outputs_ms = float(statistics.mean(build_outputs_ms_list))
    total_pre_ms = float(statistics.mean(total_pre_ms_list))

    remap_input_std = float(np.std(remap_input_ms_list))
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

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Forward timing
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
    # GPU -> CPU transfer timing
    # --------------------
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu_cpu_start = time.perf_counter()

    pred_cpu = pred.detach().cpu()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu_cpu_end = time.perf_counter()

    gpu_to_cpu_time_ms = (t_gpu_cpu_end - t_gpu_cpu_start) * 1000.0

    pred_np = pred_cpu.squeeze(0).numpy()

    # --------------------
    # Remap outputs
    # --------------------
    t_remap_output_start = time.perf_counter()

    original_chl = np.load(args.chl)
    original_h, original_w = original_chl.shape

    pred_remapped = np.zeros((pred_np.shape[0], original_h, original_w), dtype=pred_np.dtype)
    for c in range(pred_np.shape[0]):
        zoom_factors = (original_h / pred_np.shape[1], original_w / pred_np.shape[2])
        pred_remapped[c] = ndimage.zoom(pred_np[c], zoom_factors, order=0)

    t_remap_output_end = time.perf_counter()
    remap_output_time_ms = (t_remap_output_end - t_remap_output_start) * 1000.0

    # --------------------
    # Report
    # --------------------
    print("=== Single-timestep forward ===")
    print(f"Device        : {device}")
    print(f"Input shape   : {tuple(x.shape)}")
    print(f"Output shape  : {tuple(pred.shape)}")
    print()
    print(f"Preprocess    : {total_pre_ms:.2f} ms ± {total_pre_std:.2f} ms (n={n_pre_runs})")
    print(f"  remap       : {remap_input_ms:.2f} ms ± {remap_input_std:.2f} ms")
    print(f"  load inputs : {load_inputs_ms:.2f} ms ± {load_inputs_std:.2f} ms")
    print(f"  stack       : {stack_inputs_ms:.2f} ms ± {stack_inputs_std:.2f} ms")
    print(f"  co2 & norm  : {co2_norm_ms:.2f} ms ± {co2_norm_std:.2f} ms")
    print(f"  to tensor   : {to_tensor_ms:.2f} ms ± {to_tensor_std:.2f} ms")
    print(f"  build y     : {build_outputs_ms:.2f} ms ± {build_outputs_std:.2f} ms")
    print(f"Forward pass  : mean {avg_time:.2f} ms ± {std_time:.2f} ms (n={n_runs})")
    print(f"GPU -> CPU    : {gpu_to_cpu_time_ms:.2f} ms")
    print(f"Output remap  : {remap_output_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
