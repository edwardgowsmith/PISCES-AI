# PISCES-AI

This repository contains the code for training a U-Net based emulator of the PISCES biogeochemical model.

## Set-Up
The code has been run with Python 3.11, with the dependencies in `requirements.txt`.

## Scripts
The `Scripts` folder contains an example script (`forced-1848-1999+ssp585-2015-2089+mercator-2010-2016+landschutzer-1982-2004.slurm`) for training a model on a SLURM cluster.
Modify the $PYTHONPATH variable to point to the `src` directory. 
This script takes a .yaml config file.
For it to run, you need to specify:
- $MASK_FILE, a .npy file of [X, Y] with 1 corresponding to the ocean, and 0 corresponding to the land. Scripts to generate this can be found in `src/train`.
- $ROOT, as the location of the directory on your system
- $DATA_ROOT, the folder containing all of the data (see below for details on the data)
- $WANDB_PROJECT, the name of the Weights & Biases project
- $WANDB_NAME, the name of the Weights & Biases run
- $WANDB_KEY, the Weights & Biases API key

## Data
The training script expects the data in the form of one .npy file per variable per year, of the shape [T, X, Y] where T is the number of timesteps, X and Y are the grid dimensions.
For example, in `${DATA_ROOT}/ocean_data_1848_2020_v2/`, we have the files:
- tos_1848.npy
- sos_1848.npy
- chl_1848.npy
- dpco2_1848.npy
For the year 1848, and equivalently for every other year.




