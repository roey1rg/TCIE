#! /bin/sh
#SBATCH --job-name=cie_08
#SBATCH --output="/home/dcor/roeyron/tmp/cie_08.out"
#SBATCH --error="/home/dcor/roeyron/tmp/cie_08.err"
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="geforce_rtx_3090"

/bin/bash


cd /home/dcor/roeyron/interpretability_multi_hop/patchscopes/code


conda activate patchscopes
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "(Roey) Python:"
echo $(which python) 
echo $PYTHONPATH

python -m debugpy \
--listen 0.0.0.0:6010 \
--wait-for-client \
nlp_project/text_conditioned_image_embedding.py 16 8

# python nlp_project/text_conditioned_image_embedding.py 16 8