#!/bin/bash
#SBATCH --partition=train
#SBATCH --nodelist=YOUR_NODE
#SBATCH --nodes=1
#SBATCH --gres=gpu:YOUR_GPU_COUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=YOUR_CPU_COUNT
#SBATCH --job-name=train
#SBATCH --time=2-00:00:00
#SBATCH --mem=150GB
#SBATCH --output=YOUR_OUTPUT_DIR/%j.out
#SBATCH --error=YOUR_OUTPUT_DIR/%j.err

eval "$(conda shell.bash hook)"
conda activate YOUR_ENV

cd YOUR_WORKING_DIR

torchrun --standalone --nproc_per_node=YOUR_GPU_COUNT multigpu_train.py -c train_config.yaml
