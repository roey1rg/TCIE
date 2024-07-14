#! /bin/sh
#SBATCH --job-name=llava
#SBATCH --output="/home/joberant/NLP_2324/maorlavi/out.txt"
#SBATCH --error="/home/joberant/NLP_2324/maorlavi/err.txt"
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="geforce_rtx_3090"



# Maor TODO: replace this entire with your part from .bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/dcor/roeyron/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/dcor/roeyron/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/dcor/roeyron/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/dcor/roeyron/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


cd /home/joberant/NLP_2324/maorlavi/TCIE



conda activate patchscopes
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo $(which python) 
echo $PYTHONPATH

# python -m debugpy \
# --listen 0.0.0.0:5700 \
# --wait-for-client \
# llava.py

jupyter lab --no-browser --port=5711 --ip=0.0.0.0