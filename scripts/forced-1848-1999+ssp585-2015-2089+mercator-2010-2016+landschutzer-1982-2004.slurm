#!/bin/bash
#SBATCH --job-name=forced-1848-1999+ssp585-2015-2089+mercator-2010-2016+landschutzer-1982-2004
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=forced-1848-1999+ssp585-2015-2089+mercator-2010-2016+landschutzer-1982-2004.log
#SBATCH --error=forced-1848-1999+ssp585-2015-2089+mercator-2010-2016+landschutzer-1982-2004.log

source activate pytorch

export PYTHONPATH=$PYTHON_PATH
export PYTHONUNBUFFERED=1

python -u ../src/train/train_chl_dpco2_multifiles_co2_multifreq_log.py --config configs/forced-1848-1999+ssp585-2015-2089+mercator-2010-2016+landschutzer-1982-2004.yaml --local-config ../local.yaml
