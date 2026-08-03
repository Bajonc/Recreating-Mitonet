'''
Adapted from https://github.com/volume-em/empanada-napari/blob/main/empanada_napari/train.py

Changed to be independant from the GUI and specialised for MitoNet recreation hardcoding parameters and architectures to match the paper
'''

import os
import time
import platform
import numpy as np
from tqdm import tqdm
from glob import glob

import argparse
import yaml

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb

from empanada.losses import PanopticLoss
from empanada import metrics
from empanada.data.single_class_instance_dataset import SingleClassInstanceDataset
from empanada.models.panoptic_deeplab import PanopticDeepLabPR
from empanada.inference.engines import PanopticDeepLabEngine
from empanada.data.utils.transforms import FactorPad


def main(config):
    if not os.path.isdir(config['TRAIN']['model_dir']):
        os.mkdir(config['TRAIN']['model_dir'])

    return main_worker(config)

def main_worker(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb_config = config.get('WANDB', {})
    wandb.init(
        project=wandb_config.get('project', 'empanada-mitonet'),
        entity=wandb_config.get('entity', None),
        name=config.get('model_name', f"panoptic_deeplab_{int(time.time())}"),
        config=config,
        reinit=True
    )

    # model used in the paper is PanopticDeepLab
    model = PanopticDeepLabPR(
        encoder="resnet50",
        num_classes=1,
        stage4_stride=16,
        decoder_channels=256,
        low_level_channels_project=[32],
        atrous_rates=[2, 4, 6],
        aspp_channels=256,
        aspp_dropout=0.5,
        ins_decoder=True,
        ins_ratio=0.5,
        low_level_stages=[1],
        num_fc=3,
        train_num_points=1024,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        subdivision_steps=2,
        subdivision_num_points=8192
    )

    if config['TRAIN']['encoder_pretraining_path'] is not None:
        state = torch.load(config['TRAIN']['encoder_pretraining_path'], weights_only=False)
        state_dict = state['state_dict']

        for k in list(state_dict.keys()):
            if not k.startswith('fc'):
                state_dict['encoder.' + k] = state_dict[k]
                del state_dict[k]

        msg = model.load_state_dict(state['state_dict'], strict=False)
        norms = {}
        if state.get('norms') is not None:
          norms['mean'] = state['norms'][0]
          norms['std'] = state['norms'][1]
        else:
          # Norms calculated from the CEM1.5M dataset
          norms['mean'] = 0.574
          norms['std'] = 0.176
    else:
        raise Exception("Pretrained weights path is None.")


