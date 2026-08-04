#!/bin/bash
#SBATCH --partition=train
#SBATCH --nodelist=YOUR_NODE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=train
#SBATCH --time=2-00:00:00
#SBATCH --mem=150GB
#SBATCH --output=YOUR_OUTPUT_DIR/%j.out
#SBATCH --error=YOUR_OUTPUT_DIR/%j.err

eval "$(conda shell.bash hook)"
conda activate empanada

srun --label python train.py -c train_config.yaml
