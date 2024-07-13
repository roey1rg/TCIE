#! /bin/sh
#SBATCH --job-name=llava
#SBATCH --output="/home/joberant/NLP_2324/galetayt/out.txt"
#SBATCH --error="/home/joberant/NLP_2324/galetayt/err.txt"
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="geforce_rtx_3090"



# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/joberant/NLP_2324/galetayt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/joberant/NLP_2324/galetayt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/joberant/NLP_2324/galetayt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/joberant/NLP_2324/galetayt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


cd /home/joberant/NLP_2324/galetayt/TCIE



conda activate patchscopes
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "(Gal) Python:"
echo $(which python)
echo $PYTHONPATH

# python -m debugpy \
# --listen 0.0.0.0:5700 \
# --wait-for-client \
# llava.py

jupyter lab --no-browser --port=5800 --ip=0.0.0.0