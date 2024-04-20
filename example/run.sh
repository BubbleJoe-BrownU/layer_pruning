!/bin/bash

#SBATCH --job-name=environment_debug
#SBATCH --partition=A10compute

#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=3
#SBATCH --chdir=/home/jiayu_zheng/intern_projects/layer_pruning/example
#SBATCH -o /home/jiayu_zheng/intern_projects/layer_pruning/example/logs/slurm_log_%j_%N.out
#SBATCH -e /home/jiayu_zheng/intern_projects/layer_pruning/example/logs/slurm_log_%j_%N.err

conda activate dl_default
python example.py