import argparse
import os
import yaml
import torch
import pickle
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from src.models.unet import PositiveUNet
from src.utils.datasets import OceanChlDpco2MultifileFreqCO2DatasetLog
import csv
import numpy as np
import random
from torch.cuda.amp import autocast
import gc

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-config", type=str, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--average-steps", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=None)
    parser.add_argument("--eval-data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint to load (overrides best/latest search)")
    parser.add_argument("--add-on-path", type=str, default=None)
    parser.add_argument("--fixed-co2", type=float, default=None)
    cli_args = parser.parse_args()

    # Load configs and merge
    with open(cli_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(cli_args.local_config, "r") as f:
        local_config = yaml.safe_load(f)
    config.update(local_config)

    # Override with CLI
    config.update({
        "start_year": cli_args.start_year,
        "end_year": cli_args.end_year,
        "average_steps": cli_args.average_steps,
        "frequency": cli_args.frequency,
        "eval_data_dir": cli_args.eval_data_dir if cli_args.eval_data_dir is not None else config.get("data_dir", None),
        "add_on_path": cli_args.add_on_path
    })

    args = argparse.Namespace(**config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    years = list(range(args.start_year, args.end_year + 1))

    # Set climatology vars if present
    climatology_vars = getattr(args, "climatology_vars", [])

    # Prepare RNG generator for DataLoader
    g = torch.Generator()
    seed = getattr(args, "seed", 0)
    g.manual_seed(seed)

    # Load mask
    mask = torch.from_numpy(np.load(args.mask_file))  # (H, W)

    out_dir = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{datetime.now()} - Loading normalisation values from: {out_dir}/normalisation_vals.pkl")

    with open(f"{out_dir}/normalisation_vals.pkl", "rb") as f:
        normalisation_vals = pickle.load(f)
        
    in_mean = normalisation_vals["in_mean"]
    in_std = normalisation_vals["in_std"]
    chl_mean = normalisation_vals["chl_mean"]
    chl_std = normalisation_vals["chl_std"]
    dpco2_mean = normalisation_vals["dpco2_mean"]
    dpco2_std = normalisation_vals["dpco2_std"]

    print(f"{datetime.now()} - Loading evaluation dataset from: {args.eval_data_dir}")
    # Build dev Multifile dataset (will provide in/out mean/std)
    dev_ds = OceanChlDpco2MultifileFreqCO2DatasetLog(
        data_dir=args.eval_data_dir,
        input_vars=args.input_vars,
        co2_file=args.co2_file,
        climatology_vars=climatology_vars,
        years=years,
        average_steps=args.average_steps,
        in_mean=in_mean,
        in_std=in_std,
        chl_mean=chl_mean,
        chl_std=chl_std,
        dpco2_mean=dpco2_mean,
        dpco2_std=dpco2_std,
        frequency=args.frequency,
        mask=mask,
        fixed_co2=cli_args.fixed_co2,
        no_freq=True
    )

    # DataLoader
    loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=16, worker_init_fn=seed_worker, generator=g)

    # Model
    if not args.co2_file:
        num_preds = len(args.input_vars) + len(climatology_vars)
    else:
        num_preds = len(args.input_vars) + len(climatology_vars) + 1
        
    if hasattr(args, "positive_output") and args.positive_output:
        # allow configuring the minimum output value for the positive channel via config
        min_value = float(getattr(args, "positive_min_value", 0.0))
        # convert to normalized model space for the CHL channel so that after
        # denormalisation (pred * chl_std + chl_mean) the floor is `min_value`.
        chl_mean_f = float(chl_mean)
        chl_std_f = float(chl_std) if float(chl_std) != 0.0 else 1.0
        min_value_norm = (np.log10(min_value) - chl_mean_f) / chl_std_f
        print(f"{datetime.now()} - Loading Positive U-Net with {num_preds} channels (min_value={min_value}, min_value_norm={min_value_norm})")
        model = PositiveUNet(n_channels=num_preds, n_classes=2, min_value=min_value_norm)
    else:
        print(f"{datetime.now()} - Loading U-Net with {num_preds} channels")
        model = UNet(n_channels=num_preds, n_classes=2)

    # Load checkpoint: prefer explicit CLI path, then best-model.pth, then latest checkpoint-step-*.pth
    model_dir = os.path.join(args.save_dir, args.prediction, args.wandb_name)
    if cli_args.checkpoint:
        ckpt_path = cli_args.checkpoint
    else:
        best_path = os.path.join(model_dir, "best-model.pth")
        if os.path.exists(best_path):
            ckpt_path = best_path
        else:
            # find latest checkpoint-step-*.pth
            import glob
            cands = glob.glob(os.path.join(model_dir, "checkpoint-step-*.pth"))
            if cands:
                ckpt_path = max(cands, key=os.path.getmtime)
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_dir} and no --checkpoint provided")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    rmse_per_year = {year: 0.0 for year in years}
    n_examples_per_year = {year: 0 for year in years}

    # Determine per-year sample counts from the dev dataset if available.
    # OceanChlCflxDpco2MultifileFreqCO2Dataset provides cumulative_lengths (len = n_years+1)
    # which we can use to map a global sample index to a year even when lengths vary.
    if hasattr(dev_ds, "cumulative_lengths"):
        cumulative_lengths = np.array(dev_ds.cumulative_lengths, dtype=np.int64)
        # cumulative_lengths is [0, len_year0, len_year0+len_year1, ...]
    else:
        # fallback to constant examples per year (legacy behaviour)
        if args.average_steps and args.average_steps > 1:
            examples_per_year = int(5840 / args.average_steps)
        else:
            examples_per_year = 5840
        cumulative_lengths = None

    # Prepare scalars for denormalisation
    chl_mean_f = float(np.array(chl_mean))
    chl_std_f = float(np.array(chl_std))
    dpco2_mean_f = float(np.array(dpco2_mean))
    dpco2_std_f = float(np.array(dpco2_std))

    # Prepare mask template on device
    mask_exp_template = mask[None, None, :, :].to(device)

    # --- START: stream preds to disk with memmap to avoid OOM ---
    total_dev = len(dev_ds)
    H, W = mask.shape

    # compute add_on_path early so filenames match final outputs
    if hasattr(args, "add_on_path"):
        add_on_path = args.add_on_path
    elif args.average_steps == 1:
        add_on_path = f"dev-end-{args.start_year}-{args.end_year}"
    else:
        add_on_path = f"dev-end-{args.start_year}-{args.end_year}-{args.average_steps}-average"

    print("Add on path for filenames:", add_on_path)

    os.makedirs(model_dir, exist_ok=True)
    temp_pred_file = os.path.join(model_dir, f"val-preds-temp-{add_on_path}.npy")
    temp_ys_file = os.path.join(model_dir, f"val-ys-temp-{add_on_path}.npy")

    # create separate memmaps for each output variable with shape (N, H, W)
    temp_pred_chl = temp_pred_file.replace("-temp-", "-chl-temp-")
    temp_pred_dpco2 = temp_pred_file.replace("-temp-", "-dpco2-temp-")
    temp_ys_chl = temp_ys_file.replace("-temp-", "-chl-temp-")
    temp_ys_dpco2 = temp_ys_file.replace("-temp-", "-dpco2-temp-")

    preds_chl_mm = np.lib.format.open_memmap(temp_pred_chl, mode="w+", dtype=np.float32, shape=(total_dev, H, W))
    preds_dpco2_mm = np.lib.format.open_memmap(temp_pred_dpco2, mode="w+", dtype=np.float32, shape=(total_dev, H, W))
    ys_chl_mm = np.lib.format.open_memmap(temp_ys_chl, mode="w+", dtype=np.float32, shape=(total_dev, H, W))
    ys_dpco2_mm = np.lib.format.open_memmap(temp_ys_dpco2, mode="w+", dtype=np.float32, shape=(total_dev, H, W))

    per_example_rmses = []
    val_loss = 0.0
    write_idx = 0
    mask_np = mask.numpy()

    def _np_stats(a):
        return {
            "shape": a.shape,
            "dtype": str(a.dtype),
            "min": float(np.nanmin(a)) if np.isfinite(np.nanmin(a)) else float("nan"),
            "max": float(np.nanmax(a)) if np.isfinite(np.nanmax(a)) else float("nan"),
            "mean": float(np.nanmean(a)) if np.isfinite(np.nanmean(a)) else float("nan"),
            "std": float(np.nanstd(a)) if np.isfinite(np.nanstd(a)) else float("nan"),
            "n_nan": int(np.isnan(a).sum()),
        }

    # after you instantiate dev_ds / loader and load model:
    print("DEBUG: dataset attributes (if present):")
    if hasattr(dev_ds, "input_vars"):
        print(" input_vars:", dev_ds.input_vars)
    if hasattr(dev_ds, "climatology_vars"):
        print(" climatology_vars:", dev_ds.climatology_vars)
    if hasattr(dev_ds, "in_mean"):
        print(" in_mean:", getattr(dev_ds, "in_mean", None))
        print(" in_std:", getattr(dev_ds, "in_std", None))
    if hasattr(dev_ds, "chl_mean"):
        print(" chl_mean/std:", getattr(dev_ds, "chl_mean", None), getattr(dev_ds, "chl_std", None))
    if hasattr(dev_ds, "dpco2_mean"):
        print(" dpco2_mean/std:", getattr(dev_ds, "dpco2_mean", None), getattr(dev_ds, "dpco2_std", None))
    if hasattr(dev_ds, "co2_mean"):
        print(" co2_mean/std:", getattr(dev_ds, "co2_mean", None), getattr(dev_ds, "co2_std", None))
    print("model device & eval:", next(model.parameters()).device, model.training)
    
    per_example_rmses_chl = []
    per_example_rmses_dpco2 = []
    with torch.no_grad():
        for x_val, y_val in tqdm(loader, desc="Evaluating"):
            x_val, y_val = x_val.to(device).float(), y_val.to(device).float()

            with autocast():
                pred = model(x_val)  # (B, 1, H, W)

                # masked MSE like in training
                exp_mask = mask_exp_template.expand_as(pred)
                masked_sq = ((pred - y_val)[exp_mask] ** 2)
                loss_val = masked_sq.mean() if masked_sq.numel() > 0 else torch.tensor(0.0, device=device)
                val_loss += float(loss_val)

                # Denormalise
                pred_denorm = torch.empty_like(pred)
                pred_denorm[:, 0] = 10**(pred[:, 0] * chl_std_f + chl_mean_f)
                pred_denorm[:, 1] = pred[:, 1] * dpco2_std_f + dpco2_mean_f

                y_denorm = torch.empty_like(y_val)
                y_denorm[:, 0] = 10**(y_val[:, 0] * chl_std_f + chl_mean_f)
                y_denorm[:, 1] = y_val[:, 1] * dpco2_std_f + dpco2_mean_f

            # Expand mask to full batch for denormed tensors
            exp_mask = mask_exp_template.expand_as(pred_denorm)

            # Prepare numpy arrays for writing
            pred_masked = pred_denorm.masked_fill(~exp_mask, float("nan"))
            y_masked = y_denorm.masked_fill(~exp_mask, float("nan"))

            pred_np = pred_masked.cpu().numpy().astype(np.float32)   # (B,3,H,W)
            y_np = y_masked.cpu().numpy().astype(np.float32)

            b = pred_np.shape[0]
            # write per-variable slices (B,H,W)
            preds_chl_mm[write_idx:write_idx + b] = pred_np[:, 0]
            preds_dpco2_mm[write_idx:write_idx + b] = pred_np[:, 1]
            ys_chl_mm[write_idx:write_idx + b] = y_np[:, 0]
            ys_dpco2_mm[write_idx:write_idx + b] = y_np[:, 1]

            

            for i in range(b):
                # expand mask
                m = mask_np.astype(bool)

                # CHL
                valid_chl = m & np.isfinite(y_np[i, 0])
                diff_chl = pred_np[i, 0][valid_chl] - y_np[i, 0][valid_chl]
                if diff_chl.size:
                    per_example_rmses_chl.append(np.sqrt(np.mean(diff_chl ** 2)))

                # DPCO2
                valid_dpco2 = m & np.isfinite(y_np[i, 1])
                diff_dpco2 = pred_np[i, 1][valid_dpco2] - y_np[i, 1][valid_dpco2]
                if diff_dpco2.size:
                    per_example_rmses_dpco2.append(np.sqrt(np.mean(diff_dpco2 ** 2)))

            write_idx += b

    avg_rmse_chl = np.mean(per_example_rmses_chl) if per_example_rmses_chl else float('nan')
    avg_rmse_dpco2 = np.mean(per_example_rmses_dpco2) if per_example_rmses_dpco2 else float('nan')

    print("Evaluation results for", add_on_path)
    print(f"RMSE CHL: {avg_rmse_chl:.12f}, DPCO2: {avg_rmse_dpco2:.12f}")

    # flush memmaps
    preds_chl_mm.flush()
    preds_dpco2_mm.flush()
    ys_chl_mm.flush()
    ys_dpco2_mm.flush()

    # finalize metrics
    val_loss = val_loss / len(loader) if len(loader) > 0 else float("nan")
    # --- END streaming preds to disk ---

    # move temp per-variable memmaps to final filenames (atomic replace)
    final_preds_chl = os.path.join(model_dir, f"val-preds-chl-{add_on_path}.npy")
    final_preds_dpco2 = os.path.join(model_dir, f"val-preds-dpco2-{add_on_path}.npy")
    final_ys_chl = os.path.join(model_dir, f"val-ys-chl-{add_on_path}.npy")
    final_ys_dpco2 = os.path.join(model_dir, f"val-ys-dpco2-{add_on_path}.npy")

    try:
        os.replace(temp_pred_chl, final_preds_chl)
        os.replace(temp_pred_dpco2, final_preds_dpco2)
        os.replace(temp_ys_chl, final_ys_chl)
        os.replace(temp_ys_dpco2, final_ys_dpco2)
    except Exception:
        import shutil
        shutil.copy(temp_pred_chl, final_preds_chl)
        shutil.copy(temp_pred_dpco2, final_preds_dpco2)
        shutil.copy(temp_ys_chl, final_ys_chl)
        shutil.copy(temp_ys_dpco2, final_ys_dpco2)
        os.remove(temp_pred_chl)
        os.remove(temp_pred_dpco2)
        os.remove(temp_ys_chl)
        os.remove(temp_ys_dpco2)

    # cleanup
    del preds_chl_mm, preds_dpco2_mm, ys_chl_mm, ys_dpco2_mm
    torch.cuda.empty_cache()
    gc.collect()  

if __name__ == "__main__":
    main()
