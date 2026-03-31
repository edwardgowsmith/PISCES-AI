import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from datetime import datetime
import pandas as pd
import os

class OceanChlDpco2MultifileFreqCO2DatasetLog(Dataset):
    """Stacks input_vars (including climatology_vars) as input, plus CO2 and a frequency scalar.
    Outputs CHL, DPCO2.
    """
    def __init__(self, data_dir, years,
                 input_vars=["tos", "sos", "zos"],
                 climatology_vars=None,
                 in_mean=None, in_std=None,
                 chl_mean=None, chl_std=None,
                 dpco2_mean=None, dpco2_std=None,
                 co2_file=None,
                 mask=None, average_steps=1,
                 frequency=None, chl_scalar=None,
                 fixed_co2=None, no_freq=False):

        self.data_dir = data_dir
        self.years = years
        self.input_vars = input_vars.copy()
        self.climatology_vars = climatology_vars or []
        self.co2_file = co2_file
        self.average_steps = average_steps
        self.frequency = frequency
        self.fixed_co2 = fixed_co2
        self.no_freq = no_freq
        print("Frequency:", frequency)

        print(f"{datetime.now()} - Starting to load .npy files for years: {years}")
        self.inputs = {}

        # === Handle CO2 file ===
        if self.co2_file:
            print(f"{datetime.now()} - Loading CO2 reference (for mean/std) from: {self.co2_file}")
            co2_df = pd.read_csv(self.co2_file, delim_whitespace=True)
            # keep per-year scalar fallback mapping
            self.co2_by_year = {int(row['Year']): float(row['atmCO2']) for _, row in co2_df.iterrows()}
            co2_values = np.array(list(self.co2_by_year.values()))
            self.co2_mean = np.mean(co2_values)
            self.co2_std = np.std(co2_values) + 1e-9
            print(f"{datetime.now()} - Loaded CO2 reference. mean={self.co2_mean:.4f}, std={self.co2_std:.4f}")
            # look for per-year CO2 .npy files named "co2_{year}.npy" in data_dir
            co2_arrays = []
            co2_per_year = {}
            found_any = False
            for y in years:
                p = os.path.join(data_dir, f"co2_{y}.npy")
                if os.path.exists(p):
                    found_any = True
                    arr = np.load(p, mmap_mode="r")
                    # apply averaging using same frequency/average logic
                    arr_avg = self._average_array(arr, frequency=None)
                    # co2_arrays.append(arr_avg)

                    mean_val = float(np.nanmean(arr_avg))
                    co2_per_year[int(y)] = mean_val
                else:
                    co2_arrays.append(None)  # placeholder for missing year
            if found_any:
                # If some years are missing, leave None for those years and fall back to scalar there
                self.co2_arrays = None
                self.co2_by_year = co2_per_year
                print(f"Loaded per-year CO2, first year: {list(co2_per_year.keys())[0]} mean={list(co2_per_year.values())[0]:.4f}")
                print(f"{datetime.now()} - Found per-year CO2 .npy files for some years; using per-timestep CO2 where available.")
            else:
                self.co2_arrays = None
        else:
            self.co2_by_year = {}
            self.co2_mean, self.co2_std = 0.0, 1.0
            self.co2_arrays = None

        print("Co2 by year:", self.co2_by_year)
        # Load dynamic vars (apply averaging)
        for var in self.input_vars:
            if var in self.climatology_vars:
                continue
            loaded = [np.load(f"{data_dir}/{var}_{year}.npy", mmap_mode="r") for year in years]
            self.inputs[var] = [self._average_array(arr) for arr in loaded]
            print(f"{datetime.now()} - Loaded variable: {var}")
            if self.average_steps != 1:
                print(f"{datetime.now()} - Averaged variable {var} over {self.average_steps} steps")

        # Load climatology vars (apply averaging)
        for var in self.climatology_vars:
            if var not in self.input_vars:
                self.input_vars.append(var)
            path = os.path.join(data_dir, f"{var}.npy")
            static_data = self._average_array(np.load(path, mmap_mode="r"))
            self.inputs[var] = [static_data for _ in years]
            print(f"{datetime.now()} - Loaded and averaged climatology variable: {var} over {self.average_steps} steps")

        # Load outputs (apply averaging)
        loaded_chl = [np.load(f"{data_dir}/chl_{year}.npy", mmap_mode="r") for year in years]
        loaded_dpco2 = [np.load(f"{data_dir}/dpco2_{year}.npy", mmap_mode="r") for year in years]

        if chl_scalar:
            print(f"{datetime.now()} - Applying CHL scalar of {chl_scalar} to outputs")
            loaded_chl = [arr * float(chl_scalar) for arr in loaded_chl]

        print(f"{datetime.now()} - Loading CHL and DPCO2")
        self.chl = [self._average_array(arr) for arr in loaded_chl]
        self.dpco2 = [self._average_array(arr) for arr in loaded_dpco2]

        print(f"{datetime.now()} - Loaded CHL and DPCO2 files")
        if self.average_steps != 1:
            print(f"{datetime.now()} - Averaged outputs over {self.average_steps} steps")

        # Time helpers (use averaged arrays)
        first_var = self.input_vars[0]
        self.lengths = [arr.shape[0] for arr in self.inputs[first_var]]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

        # --- Trim/expand climatology vars to match per-year lengths ---
        for var in list(self.climatology_vars or []):
            raw = self.inputs[var][0]
            T = raw.shape[0]
            
            # Check if all lengths are the same and match the raw data length
            if len(set(self.lengths)) == 1 and self.lengths[0] == T:
                continue
            
            per_year = []
            for L in self.lengths:
                if T < L:
                    reps = int(np.ceil(L / T))
                    expanded = np.concatenate([raw] * reps, axis=0)[:L]
                    per_year.append(expanded)
                else:
                    per_year.append(raw[:L].copy())

            self.inputs[var] = per_year
            print(f"{datetime.now()} - Trimmed/expanded climatology '{var}' to per-year lengths {[a.shape[0] for a in per_year]}")

        # Static mask (from first averaged timestep)
        self.mask = mask if mask is not None else torch.tensor(~np.isnan(self.inputs[first_var][0][0]), dtype=torch.bool)

        for var in self.input_vars:
            print(f"{var} shape first year: {self.inputs[var][0].shape}")
            
        # Input stats (computed on averaged first-year data if not provided)
        if in_mean is None or in_std is None:
            print(f"{datetime.now()} - Computing input stats using only the first year: {years[0]}")
            stacked = np.stack([self.inputs[var][0] for var in self.input_vars], axis=1)
            self.in_mean = np.nanmean(stacked, axis=(0, 2, 3))
            self.in_std = np.nanstd(stacked, axis=(0, 2, 3)) + 1e-9
        else:
            self.in_mean = in_mean
            self.in_std = in_std

        # CHL stats (computed on log-transformed data)
        if chl_mean is None or chl_std is None:
            chl_log = self._log_chl(self.chl[0])
            self.chl_mean = np.nanmean(chl_log)
            self.chl_std = np.nanstd(chl_log) + 1e-9
        else:
            self.chl_mean = chl_mean
            self.chl_std = chl_std

        # DPCO2 stats
        if dpco2_mean is None or dpco2_std is None:
            self.dpco2_mean = np.nanmean(self.dpco2[0])
            self.dpco2_std = np.nanstd(self.dpco2[0]) + 1e-9
        else:
            self.dpco2_mean = dpco2_mean
            self.dpco2_std = dpco2_std

        # Frequency scalar for extra channel
        if not self.no_freq:
            if not frequency:
                self.freq_scalar = self.average_steps / 5840.0
            else:
                print(f"{datetime.now()} - Using provided frequency: {frequency} timesteps")
                self.freq_scalar = frequency / 5840.0

    def _average_array(self, arr, frequency=None):
        arr = np.squeeze(arr)
        if not frequency:
            average_steps = self.average_steps
        else:
            average_steps = frequency
            
        if average_steps == 1:
            return arr
        T, *spatial_dims = arr.shape
        new_len = T // average_steps
        if new_len == 0:
            raise ValueError(f"average_steps={self.average_steps} is larger than available timesteps ({T})")
        trimmed = arr[:new_len * average_steps].reshape(new_len, average_steps, *spatial_dims)
        return np.nanmean(trimmed, axis=1)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _get_file_index(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        return file_idx, local_idx

    def _log_chl(self, chl):
        return np.log10(chl)

    def __getitem__(self, idx):
        file_idx, local_idx = self._get_file_index(idx)

        # === Base inputs (shape: #vars, H, W)
        x = np.stack([self.inputs[var][file_idx][local_idx] for var in self.input_vars], axis=0)

        # === CO2 channel: prefer per-year per-timestep .npy if available, otherwise scalar by year
        if self.co2_by_year:
            if self.fixed_co2 is not None:
                co2_val = (self.fixed_co2 - self.co2_mean) / self.co2_std
                H, W = x.shape[1], x.shape[2]
                co2_map = np.full((1, H, W), co2_val, dtype=np.float32)
            elif self.co2_arrays is not None and self.co2_arrays[file_idx] is not None:
                # per-year array exists: index its local timestep, normalize per global mean/std, broadcast as channel
                co2_frame = self.co2_arrays[file_idx][local_idx]  # shape (H,W) expected
                co2_norm = (co2_frame - self.co2_mean) / self.co2_std
                co2_map = np.expand_dims(co2_norm.astype(np.float32), axis=0)
            else:
                year = self.years[file_idx]
                co2_val = (self.co2_by_year[year] - self.co2_mean) / self.co2_std
                H, W = x.shape[1], x.shape[2]
                co2_map = np.full((1, H, W), co2_val, dtype=np.float32)
            x = np.concatenate([x, co2_map], axis=0)

        # === Normalize inputs (core channels and CO2 if present) ===
        if self.co2_by_year:
            extended_mean = np.concatenate([self.in_mean, [0.0]])
            extended_std = np.concatenate([self.in_std, [1.0]])
            x = (x - extended_mean[:, None, None]) / extended_std[:, None, None]
        else:
            x = (x - self.in_mean[:, None, None]) / self.in_std[:, None, None]

        x = np.nan_to_num(x, nan=0.0)

        if not self.no_freq:
            # === Add frequency channel (not normalized) ===
            freq_channel = np.full_like(x[0:1], fill_value=self.freq_scalar)
            x = np.concatenate([x, freq_channel], axis=0)

        # === Output ===
        chl = self.chl[file_idx][local_idx]
        dpco2 = self.dpco2[file_idx][local_idx]
        
        chl = self._log_chl(chl)

        chl = chl[None, ...] if chl.ndim == 2 else chl
        dpco2 = dpco2[None, ...] if dpco2.ndim == 2 else dpco2

        chl = np.where(self.mask, (chl - self.chl_mean) / self.chl_std, 0)
        dpco2 = np.where(self.mask, (dpco2 - self.dpco2_mean) / self.dpco2_std, 0)

        y = np.concatenate([chl, dpco2], axis=0)  # (3, H, W)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
