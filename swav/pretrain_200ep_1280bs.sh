# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the swav directory of this source tree.
#


#!/bin/bash
#SBATCH --partition=YOUR_PARTITION 
#SBATCH --nodelist=YOUR_NODES
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pretrain
#SBATCH --time=7-00:00:00
#SBATCH --mem=300GB

dist_url="tcp://YOUR_NODE_NAME:40000" # <-- any one node from your nodelist
DATASET_PATH="YOUR_DATASET_PATH"
EXPERIMENT_PATH="YOUR_EXPERIMENT_PATH"
mkdir -p $EXPERIMENT_PATH

eval "$(conda shell.bash hook)"
conda activate YOUR_ENVIRONMENT

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python YOUR_FILE_PATH/main_swav.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 3840 \
--epoch_queue_starts 15 \
--epochs 200 \
--batch_size 160 \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--dist_url $dist_url \
--arch resnet50 \
--use_fp16 true \
--sync_bn pytorch \
--syncbn_process_group_size 4 \
--dump_path $EXPERIMENT_PATH
