import os

SLURM_TEMPLATE = """#! /bin/sh
#SBATCH --job-name=JOBNAME
#SBATCH --output="OUTPUT_PATH_STD"
#SBATCH --error="OUTPUT_PATH_ERR"
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


cd /home/dcor/roeyron/TCIE


conda activate patchscopes
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "(Roey) Python:"
echo $(which python)
echo $PYTHONPATH

# python extract_embeddings_celeba.py --output_dir /home/dcor/roeyron/TCIE/results/celeba_conditioned_embeddings_large --n_parts N_PARTS --part CURRENT_PART
python extract_embeddings_qard.py --output_dir '/home/dcor/roeyron/TCIE/results/qard_v4_embeddings_wo_answer_shortly' --n_parts N_PARTS --part CURRENT_PART
"""


def push_to_slurm(n_parts, current_part, experiment_name):
    slurm_content = SLURM_TEMPLATE
    slurm_dump_path = '/home/dcor/roeyron/tmp'
    run_id = f'{experiment_name}_{current_part:02d}'
    slurm_content = slurm_content.replace('JOBNAME', run_id)
    slurm_content = slurm_content.replace('OUTPUT_PATH_STD', os.path.join(slurm_dump_path, f'{run_id}.out'))
    slurm_content = slurm_content.replace('OUTPUT_PATH_ERR', os.path.join(slurm_dump_path, f'{run_id}.err'))
    slurm_content = slurm_content.replace('N_PARTS', str(n_parts))
    slurm_content = slurm_content.replace('CURRENT_PART', str(current_part))
    slurm_file_path = os.path.join(slurm_dump_path, f'cie_{run_id}.slurm')
    with open(slurm_file_path, 'w') as f:
        f.write(slurm_content)
    print(slurm_file_path)
    print(slurm_content)
    print(os.system(f'sbatch {slurm_file_path}'))


def push_many():
    experiment_name = 'celebaLarge'
    n_parts = 10
    for part in range(n_parts):
        push_to_slurm(n_parts, part, experiment_name)


if __name__ == "__main__":
    push_many()
