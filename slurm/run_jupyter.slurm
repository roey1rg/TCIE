#! /bin/sh
#SBATCH --job-name=llava
#SBATCH --output="/home/dcor/roeyron/out.txt"
#SBATCH --error="/home/dcor/roeyron/err.txt"
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="geforce_rtx_3090"



/bin/bash


cd /home/dcor/roeyron/TCIE



conda activate patchscopes
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "(Roey) Python:"
echo $(which python) 
echo $PYTHONPATH

# python -m debugpy \
# --listen 0.0.0.0:5700 \
# --wait-for-client \
# llava.py

jupyter lab --no-browser --port=5700 --ip=0.0.0.0