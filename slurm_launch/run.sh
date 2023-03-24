#!/bin/bash

#SBATCH --job-name=ft_rm_b
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=all
#SBATCH --mail-user=tongshq@shanghaitech.edu.cn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=10-00:00:00
#SBATCH --exclude=sist_gpu[38,63-66]

torchrun --nproc_per_node=4 main.py --mode train --config configs/default_config.py --workdir test