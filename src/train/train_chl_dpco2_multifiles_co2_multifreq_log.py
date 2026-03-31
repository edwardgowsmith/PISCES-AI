import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import pickle
from src.models.unet import UNet, TunableUNet, MultiTaskTunableUNet, DualEncoderTunableUNet, PositiveUNet
from src.utils.datasets import *
from datetime import datetime
import random
import time
import threading
import queue
import gc
import shutil

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (at the cost of speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def tensor_stats_no_nan(t: torch.Tensor):
    """Return (min, max, n_nan) for tensor, ignoring NaNs. Returns float('nan') for min/max if no valid values."""
    t = t.detach()
    n_nan = int(torch.isnan(t).sum().item())
    valid_mask = ~torch.isnan(t)
    if valid_mask.any():
        valid = t[valid_mask]
        return float(valid.min().item()), float(valid.max().item()), n_nan
    else:
        return float("nan"), float("nan"), n_nan

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--local-config", type=str, required=True, help="Path to local YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.local_config) as f:
        local_config = yaml.safe_load(f)

    config.update(local_config)

    return argparse.Namespace(**config)

def train_model(args):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Loading data")

    # Set climatology vars
    if hasattr(args, "climatology_vars"):
        climatology_vars = args.climatology_vars
    else:
        climatology_vars = []

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    mask = torch.from_numpy(np.load(args.mask_file))  # shape (H, W)

    # Support multiple averaging frequencies (average_steps) by creating one dataset per freq
    averaging = getattr(args, "averaging", [1])
    if isinstance(averaging, int):
        averaging = [averaging]

    no_freq = getattr(args, "no_freq", False)

    train_dss = []
    lengths = []
    # Build the first dataset to compute normalization stats, then reuse those
    # stats for all subsequent training datasets to ensure consistent normalisation.
    for data_dir in args.training["data_dirs"]:
        path = data_dir['path']
        years = list(range(data_dir['years'][0], data_dir['years'][1] + 1))
        averaging = data_dir.get('averaging', [1])
        frequency = data_dir.get('frequency', None)

        for average in averaging:
            if len(train_dss) == 0:
                ds = OceanChlDpco2MultifileFreqCO2DatasetLog(
                    data_dir=path,
                    input_vars=args.input_vars,
                    climatology_vars=climatology_vars,
                    years=years,
                    co2_file=args.co2_file,
                    average_steps=average,
                    frequency=frequency,
                    mask=mask,
                    chl_scalar=args.chl_scalar if hasattr(args, "chl_scalar") else None,
                    no_freq=no_freq
                )
                # extract normalization stats from first dataset
                in_mean, in_std = ds.in_mean, ds.in_std
                chl_mean, chl_std = ds.chl_mean, ds.chl_std
                dpco2_mean, dpco2_std = ds.dpco2_mean, ds.dpco2_std
                co2_mean, co2_std = ds.co2_mean, ds.co2_std
                train_dss.append(ds)
            else:
                ds = OceanChlDpco2MultifileFreqCO2DatasetLog(
                    data_dir=path,
                    input_vars=args.input_vars,
                    climatology_vars=climatology_vars,
                    years=years,
                    co2_file=args.co2_file,
                    average_steps=average,
                    frequency=frequency,
                    in_mean=in_mean,
                    in_std=in_std,
                    chl_mean=chl_mean,
                    chl_std=chl_std,
                    dpco2_mean=dpco2_mean,
                    dpco2_std=dpco2_std,
                    mask=mask,
                    chl_scalar=args.chl_scalar if hasattr(args, "chl_scalar") else None,
                    no_freq=no_freq
                )
                train_dss.append(ds)
            lengths.append(len(train_dss[-1]))  # weight by average steps

    normalisation_vals = {
        "in_mean": in_mean, "in_std": in_std,
        "chl_mean": chl_mean, "chl_std": chl_std,
        "dpco2_mean": dpco2_mean, "dpco2_std": dpco2_std
    }
    print(normalisation_vals)

    out_dir = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}")
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/normalisation_vals.pkl", "wb") as f:
        pickle.dump(normalisation_vals, f)

    # Load evaluation datasets
    dev_dss = []
    for data_dir in args.evaluation["data_dirs"]:
        path = data_dir['path']
        years = list(range(data_dir['years'][0], data_dir['years'][1] + 1))
        averaging = data_dir.get('averaging', [1])
        frequency = data_dir.get('frequency', None)

        for average in averaging:
            dev_dss.append((data_dir["add_on_path"], OceanChlDpco2MultifileFreqCO2DatasetLog(
                data_dir=path,
                input_vars=args.input_vars,
                climatology_vars=climatology_vars,
                years=years,
                co2_file=args.co2_file,
                average_steps=average,
                frequency=frequency,
                in_mean=in_mean,
                in_std=in_std,
                chl_mean=chl_mean,
                chl_std=chl_std,
                dpco2_mean=dpco2_mean,
                dpco2_std=dpco2_std,
                mask=mask,
                chl_scalar=args.chl_scalar if hasattr(args, "chl_scalar") else None,
                no_freq=no_freq
            )))

    # concat train datasets and create a weighted sampler so frequencies are sampled proportional to avg steps
    train_ds = ConcatDataset(train_dss)

    train_num_workers = 16
    eval_num_workers = 16

    dataset_weighting = getattr(args, "dataset_weighting", False)

    if dataset_weighting:
        print(f"{datetime.now()} - Using weighted sampling for training datasets")
        weights = []
        for i, ds in enumerate(train_dss):
            w = 1/lengths[i]
            weights.extend([w] * len(ds))

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=train_num_workers, worker_init_fn=seed_worker, generator=g)

    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=train_num_workers, worker_init_fn=seed_worker, generator=g)

    # per-frequency dev loaders used during evaluation (created once, outside training loop)
    dev_loaders = [
        DataLoader(dev_ds_i, batch_size=args.batch_size, shuffle=False,
                   num_workers=eval_num_workers, worker_init_fn=seed_worker, generator=g)
        for (_, dev_ds_i) in dev_dss
    ]

    # number of input channels: core vars + climatology vars + CO2 map + frequency scalar
    if not args.co2_file:
        num_preds = len(args.input_vars) + len(climatology_vars)
    elif no_freq:
        num_preds = len(args.input_vars) + len(climatology_vars) + 1
    else:  
        num_preds = len(args.input_vars) + len(climatology_vars) + 2

    
        


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

    # Load model from checkpoint if exists
    if hasattr(args, "continue_from") and args.continue_from:
        model.load_state_dict(torch.load(args.continue_from))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mask = mask.to(device)
    mask_exp = mask.unsqueeze(0).unsqueeze(0)

    # Load the learning rate, default to 1e-4
    if hasattr(args, "learning_rate"):
        learning_rate = float(args.learning_rate)
    else:
        learning_rate = 1e-4

    # Load the optimizer, default to Adam without weight decay    
    if hasattr(args, "weight_decay"):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=float(args.weight_decay))
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if hasattr(args, "linear_lr_scheduler") and args.linear_lr_scheduler:
        print(f"{datetime.now()} - Loading linear learning rate scheduler")
        # Apply linear lr scheduler
        def lr_lambda(current_step):
            warmup_steps = int(0.05 * args.train_steps)
            if current_step < warmup_steps:
                return current_step / warmup_steps  # linear warmup
            else:
                decay_steps = args.train_steps - warmup_steps
                progress = (current_step - warmup_steps) / max(decay_steps, 1)
                return max(0.0, 1.0 - progress)  # linear decay to 0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def masked_mse(pred, target, mask):
        """
        Compute MSE over valid mask & finite values only.
        mask may be (H,W) or (1,1,H,W) or similar; this function will move/expand it to match pred.
        Returns torch.nan if no valid pixels are available.
        """
        # prepare mask: allow (H,W) or (1,1,H,W)
        mask_in = mask
        if mask_in.dim() == 2:
            mask_in = mask_in.unsqueeze(0).unsqueeze(0)
        elif mask_in.dim() == 3:
            mask_in = mask_in.unsqueeze(1)
        # ensure mask on same device as preds
        mask_in = mask_in.to(pred.device)
        exp_mask = mask_in.expand_as(pred)

        # valid pixels: mask true and both pred and target finite
        valid = exp_mask & (~torch.isnan(target)) & (~torch.isinf(target)) & (~torch.isnan(pred)) & (~torch.isinf(pred))
        if valid.sum() == 0:
            # no valid pixels -> signal with NaN so caller can handle/skip
            return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
        diffs = (pred - target)[valid]
        return (diffs ** 2).mean()

    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, name=args.wandb_name)
    best_loss = float("inf")
    best_model_path = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}", "best-model.pth")

    # Eval history and best-eval record
    eval_history = []  # list of dicts: {"step": int, "overall_val_loss": float, "per_avg": [{...}, ...]}
    best_eval = None

    train_iter = iter(train_loader)
    steps_per_epoch = len(train_loader)
    model.train()

    if hasattr(args, "start_step") and args.start_step:
        start_step = args.start_step
    else:
        start_step = 0

    scaler = GradScaler()

    # Save per-frequency dev labels to .npy memmaps (memory-efficient)
    mask_new = mask.to("cpu")

    # for dev_idx, (add_on_path, dev_ds_i) in enumerate(dev_dss):
    #     total_dev = len(dev_ds_i)
    #     avg = dev_ds_i.average_steps
    #     H, W = mask_new.shape
    #     fname_chl = os.path.join(out_dir, f"val-ys-chl-{add_on_path}-avg{avg}.npy")
    #     fname_dpco2 = os.path.join(out_dir, f"val-ys-dpco2-{add_on_path}-avg{avg}.npy")

    #     chl_mm = np.lib.format.open_memmap(fname_chl, mode="w+", dtype=np.float32, shape=(total_dev, H, W))
    #     dpco2_mm = np.lib.format.open_memmap(fname_dpco2, mode="w+", dtype=np.float32, shape=(total_dev, H, W))

    #     write_idx = 0
    #     for _, y_val in tqdm(dev_loaders[dev_idx], desc=f"Saving dev set labels avg={avg}"):
    #         # y_val on CPU: (B, 2, H, W) => CHL (0), DPCO2 (1)
    #         chl = 10**(y_val[:, 0] * chl_std + chl_mean)
    #         dpco2 = y_val[:, 1] * dpco2_std + dpco2_mean

    #         # mask invalid locations -> nan
    #         chl = chl.masked_fill(~mask_new, float("nan"))
    #         dpco2 = dpco2.masked_fill(~mask_new, float("nan"))

    #         chl_np = chl.numpy().astype(np.float32)
    #         dpco2_np = dpco2.numpy().astype(np.float32)

    #         b = chl_np.shape[0]
    #         chl_mm[write_idx:write_idx + b] = chl_np
    #         dpco2_mm[write_idx:write_idx + b] = dpco2_np
    #         write_idx += b

    #     # flush and free
    #     chl_mm.flush()
    #     dpco2_mm.flush()
    #     del chl_mm, dpco2_mm
    #     torch.cuda.empty_cache()
    #     gc.collect()

    mask_exp = mask_exp.to(device)

    print(f"{datetime.now()} - Starting Training")
    for step in tqdm(range(start_step+1, args.train_steps + 1)):
        
        if step % args.eval_every == 0 or (hasattr(args, "eval_at_start") and step == 1):
            model.eval()

            # Evaluate each dev dataset separately and report per-frequency scores
            all_val_losses = []
            all_avg_rmses_chl = []
            all_avg_rmses_dpco2 = []
            per_dataset_metrics = []  # Track RMSEs per dataset

            for dev_idx, (add_on_path, dev_ds_i) in enumerate(dev_dss):
                avg = dev_ds_i.average_steps
                print(f"{datetime.now()} - Evaluating dev set {dev_idx}: {add_on_path}, avg={avg}")
                total_dev = len(dev_ds_i)
                H, W = mask_new.shape
                step_tag = f"step{step}-avg{avg}"
                
                print("Opening mem-mapped files")
                temp_pred_chl = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}", f"val-preds-chl-{add_on_path}-{step_tag}.npy")
                temp_pred_dpco2 = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}", f"val-preds-dpco2-{add_on_path}-{step_tag}.npy")

                preds_chl_mm = np.lib.format.open_memmap(temp_pred_chl, mode="w+", dtype=np.float32, shape=(total_dev, H, W))
                preds_dpco2_mm = np.lib.format.open_memmap(temp_pred_dpco2, mode="w+", dtype=np.float32, shape=(total_dev, H, W))

                per_example_rmses_chl = []
                per_example_rmses_dpco2 = []
                val_loss = 0.0
                write_idx = 0

                # use pre-created per-frequency dev loader
                dev_loader_i = dev_loaders[dev_idx]

                print("Starting eval loop")
                with torch.no_grad():
                    for x_val, y_val in tqdm(dev_loader_i, desc=f"[Eval @ Step {step} | avg={avg}]"):
                        x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)

                        with torch.cuda.amp.autocast():
                            pred_val = model(x_val)
                            loss_val = masked_mse(pred_val, y_val, mask)
                            val_loss += loss_val.item()

                            # denormalise (model/y: CHL=0, DPCO2=1)
                            pred_chl = (10**(pred_val[:, 0] * chl_std + chl_mean)).detach()
                            pred_dpco2 = (pred_val[:, 1] * dpco2_std + dpco2_mean).detach()
                            y_chl = (10**(y_val[:, 0] * chl_std + chl_mean)).detach()
                            y_dpco2 = (y_val[:, 1] * dpco2_std + dpco2_mean).detach()

                        pred_chl_np = pred_chl.cpu().numpy()
                        pred_dpco2_np = pred_dpco2.cpu().numpy()
                        y_chl_np = y_chl.cpu().numpy()
                        y_dpco2_np = y_dpco2.cpu().numpy()

                        b = pred_chl_np.shape[0]
                        preds_chl_mm[write_idx:write_idx + b] = pred_chl_np
                        preds_dpco2_mm[write_idx:write_idx + b] = pred_dpco2_np
                        write_idx += b

                        mask_np = mask_new.numpy()
                        for i in range(b):
                            m = mask_np.astype(bool)
                            # keep locations that are valid in the TARGET only (drop target NaNs)
                            # CHL
                            valid_chl = m & np.isfinite(y_chl_np[i])
                            diff_chl = pred_chl_np[i][valid_chl] - y_chl_np[i][valid_chl]
                            if diff_chl.size:
                                per_example_rmses_chl.append(np.sqrt(np.mean(diff_chl ** 2)))

                            # DPCO2
                            valid_dpco2 = m & np.isfinite(y_dpco2_np[i])
                            diff_dpco2 = pred_dpco2_np[i][valid_dpco2] - y_dpco2_np[i][valid_dpco2]
                            if diff_dpco2.size:
                                per_example_rmses_dpco2.append(np.sqrt(np.mean(diff_dpco2 ** 2)))

                # average val loss across batches for this dev set (keep same behavior as before)
                val_loss /= max(1, len(dev_loader_i))
                avg_rmse_chl = np.mean(per_example_rmses_chl) if per_example_rmses_chl else float("nan")
                avg_rmse_dpco2 = np.mean(per_example_rmses_dpco2) if per_example_rmses_dpco2 else float("nan")

                # flush memmaps
                preds_chl_mm.flush()
                preds_dpco2_mm.flush()

                # move/cleanup: save per-frequency final preds only if overall improvement (use mean over dev sets)
                all_val_losses.append(val_loss)
                all_avg_rmses_chl.append(avg_rmse_chl)
                all_avg_rmses_dpco2.append(avg_rmse_dpco2)
                
                # Track per-dataset metrics
                per_dataset_metrics.append({
                    "dataset_path": add_on_path,
                    "average_steps": avg,
                    "rmse_chl": avg_rmse_chl,
                    "rmse_dpco2": avg_rmse_dpco2,
                    "val_loss": val_loss
                })

                final_pred_chl = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}", f"val-preds-chl-{add_on_path}-avg{avg}.npy")
                final_pred_dpco2 = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}", f"val-preds-dpco2-{add_on_path}-avg{avg}.npy")

                # tentatively keep per-dev files; if not best we'll remove them below
                os.replace(temp_pred_chl, final_pred_chl)
                os.replace(temp_pred_dpco2, final_pred_dpco2)

                if step == 1:
                    # copy the final preds created at the first eval to "-start" suffixed files
                    try:
                        start_pred_chl = final_pred_chl.replace(".npy", "-start.npy")
                        start_pred_dpco2 = final_pred_dpco2.replace(".npy", "-start.npy")
                        shutil.copyfile(final_pred_chl, start_pred_chl)
                        shutil.copyfile(final_pred_dpco2, start_pred_dpco2)
                        print(f"{datetime.now()} - Saved start dev preds to {start_pred_chl}, {start_pred_dpco2}")
                    except Exception as e:
                        print(f"{datetime.now()} - Warning: failed to save start preds: {e}")

                print(f"{datetime.now()} - Saved dev preds for avg={avg} to {final_pred_chl}, {final_pred_dpco2}")

                # cleanup per-dev memmaps
                del preds_chl_mm, preds_dpco2_mm, per_example_rmses_chl, per_example_rmses_dpco2
                torch.cuda.empty_cache()
                gc.collect()

                # log per-dev metrics to wandb
                wandb.log(
                    {
                        f"eval_rmse_chl_{add_on_path}_avg_{avg}": avg_rmse_chl,
                        f"eval_rmse_dpco2_{add_on_path}_avg_{avg}": avg_rmse_dpco2,
                        f"eval_loss_avg_{add_on_path}_{avg}": val_loss,
                        "epoch": step / steps_per_epoch,
                        "steps": step,
                    },
                    step=step,
                )

            # compute an overall val loss (mean across dev sets) and keep best model based on that
            overall_val_loss = float(np.mean(all_val_losses)) if all_val_losses else float("inf")
            overall_avg_rmse_chl = float(np.mean(all_avg_rmses_chl)) if all_avg_rmses_chl else float("nan")
            overall_avg_rmse_dpco2 = float(np.mean(all_avg_rmses_dpco2)) if all_avg_rmses_dpco2 else float("nan")

            eval_history.append({
                "step": int(step),
                "overall_val_loss": float(overall_val_loss),
                "overall_rmse_chl": float(overall_avg_rmse_chl),
                "overall_rmse_dpco2": float(overall_avg_rmse_dpco2),
                "per_datasets": per_dataset_metrics,
            })

            # move or remove the temp files depending on whether this eval is the new best
            if overall_val_loss < best_loss:
                best_loss = overall_val_loss
                print(f"New best overall val loss: {best_loss:.10f} — saving model to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)
                # move temp files -> final
                if 'temp_paths_this_eval' in locals():
                    for tmp, final in temp_paths_this_eval:
                        try:
                            os.replace(tmp, final)
                        except Exception as e:
                            print(f"Warning: failed to move {tmp} -> {final}: {e}")
                    del temp_paths_this_eval
                # save best-eval snapshot (per-frequency too)
                best_eval = eval_history[-1].copy()
            else:
                # remove temp files for this eval (not the best)
                if 'temp_paths_this_eval' in locals():
                    for tmp, final in temp_paths_this_eval:
                        try:
                            if os.path.exists(tmp):
                                os.remove(tmp)
                        except Exception as e:
                            print(f"Warning: failed to remove temp file {tmp}: {e}")
                    del temp_paths_this_eval

            # log overall aggregated metrics
            wandb.log(
                {
                    "eval_rmse_chl": overall_avg_rmse_chl,
                    "eval_rmse_dpco2": overall_avg_rmse_dpco2,
                    "eval_loss": overall_val_loss,
                    "epoch": step / steps_per_epoch,
                    "steps": step,
                },
                step=step,
            )

            model.train()
            
        # -------------------- Data Loading --------------------
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # check mask selects at least one pixel
        expanded_mask = mask_exp.expand(x.shape[0], 3, mask.shape[0], mask.shape[1]) if 'mask_exp' in globals() else mask.unsqueeze(0).unsqueeze(0).expand(x.shape[0],1,*mask.shape)
        if expanded_mask.sum() == 0:
            raise RuntimeError("Mask expanded selects zero pixels (mask empty)")

        if torch.isinf(y).any() or torch.isinf(x).any():
            print("Skipping example due to infinities")
            continue

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # -------------------- Forward Pass --------------------
        # forward_start = time.time()
        with autocast():
            pred = model(x)
            # DEBUG: detect NaNs in model output
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print("Model output contains NaN/Inf")
                torch.save({"x": x.cpu(), "pred": pred.cpu()}, f"{out_dir}/bad_pred.pt")
                raise RuntimeError("Model produced NaN/Inf output")
            loss = masked_mse(pred, y, mask_exp)

            if not torch.isfinite(loss):
                print(f"Non-finite loss at step {step}: {loss}")
                xmin, xmax, xn = tensor_stats_no_nan(x)
                ymin, ymax, yn = tensor_stats_no_nan(y)
                print(f"  x min/max/n_nan: {xmin}/{xmax}/{xn}")
                print(f"  y min/max/n_nan: {ymin}/{ymax}/{yn}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(args, "linear_lr_scheduler") and args.linear_lr_scheduler:
            scheduler.step()


        if step % args.log_every == 0:
            wandb.log({"train_loss": loss.item(), "epoch": step / steps_per_epoch, "steps": step}, step=step)

        

    print(f"{datetime.now()} - Finished Training!")
    # Print best evaluation metrics found during training
    if best_eval is not None:
        print("Best evaluation (by overall_val_loss):")
        print(f"  step: {best_eval['step']}")
        print(f"  overall_val_loss: {best_eval['overall_val_loss']:.10f}")
        print(f"  overall eval_rmse_chl: {best_eval['overall_rmse_chl']:.12f}")
        print(f"  overall eval_rmse_dpco2: {best_eval['overall_rmse_dpco2']:.12f}")
        print("  Per-dataset scores:")
        for per_ds in best_eval.get("per_datasets", []):
            print(f"    {per_ds['dataset_path']}, avg={per_ds['average_steps']}: rmse_chl={per_ds['rmse_chl']:.12f}, rmse_dpco2={per_ds['rmse_dpco2']:.12f}")
    else:
        print("No evaluations were recorded during training.")
    
    # Only save the model once at the end of training, plus the best RMSE
    checkpoint_path = os.path.join(f"{args.save_dir}/{args.prediction}/{args.wandb_name}", f"checkpoint-step-{args.train_steps}.pth")
    torch.save(model.state_dict(), checkpoint_path)    

if __name__ == "__main__":
    start_time = time.time()

    args = load_config()
    set_seed(args.seed)
    train_model(args)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal training time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


