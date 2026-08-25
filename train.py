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
from empanada import models
from empanada.inference.engines import PanopticDeepLabEngine
from empanada.data.utils.transforms import FactorPad
from empanada.models.encoders.convnext import LayerNorm as CustomLayerNorm

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
    arch = config['MODEL']['arch']
    model = models.__dict__[arch](**config['MODEL'])

    # load pretrained weights and convert them 
    pretraining_path = config['TRAIN']['encoder_pretraining_path']
    pretraining_norms = config['TRAIN']['pretraining_norms']

    if pretraining_path is None or pretraining_path == 'null':
        pretraining = False
        norms = pretraining_norms

    else:
        pretraining = True
        state, state_dict = load_encoder_weights(pretraining_path, config['MODEL']['encoder'])

        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded backbone from checkpoint '{}' with msg {}".format(pretraining_path , msg))

        norms = {}
        if state.get('norms') is not None:
          norms['mean'] = state['norms'][0]
          norms['std'] = state['norms'][1]
        else:  
            norms = pretraining_norms
         #   norms = {'mean': 0.57571, 'std': 0.12765}
         
    finetune_layer = (config['TRAIN']['finetune_layer'] if pretraining else 'all') 

    for pname, param in model.named_parameters():
        if 'encoder' in pname:
            param.requires_grad = False

    if finetune_layer == 'none':
        pass

    elif finetune_layer == 'all':
        for pname, param in model.named_parameters():
            if 'encoder' in pname:
                param.requires_grad = True

    else:
        valid_layers = ['layer1', 'layer2', 'layer3', 'layer4']

        assert finetune_layer in valid_layers

        for layer_name in valid_layers[valid_layers.index(finetune_layer):]:
            for pname, param in model.named_parameters():
                if f'encoder.{layer_name}' in pname:
                    param.requires_grad = True

    model = model.to(device)
    cudnn.benchmark = True

    config['aug_string'] = []
    dataset_augs = []
    for aug_params in config['TRAIN']['augmentations']:
        aug_name = aug_params['aug']
        config['aug_string'].append(aug_params['aug'])
        del aug_params['aug']
        dataset_augs.append(A.__dict__[aug_name](**aug_params))

    tfs = A.Compose([
        *dataset_augs,
        A.Normalize(**norms),
        ToTensorV2()
    ])

    train_dataset = SingleClassInstanceDataset(data_dir=config['TRAIN']['train_dir'], transforms = tfs)
    train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['TRAIN']['batch_size'], 
        shuffle=False,
        num_workers=config['TRAIN']['workers'], 
        pin_memory=torch.cuda.is_available(), 
        sampler=train_sampler,
        drop_last=True
    )

    if config['EVAL']['eval_dir'] is not None:

        eval_tfs = A.Compose([
            FactorPad(128),
            A.Normalize(**norms),
            ToTensorV2()
        ])
        eval_dataset = SingleClassInstanceDataset(data_dir=config['EVAL']['eval_dir'], transforms=eval_tfs)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 pin_memory=torch.cuda.is_available(),
                                 num_workers=config['TRAIN']['workers'])
    else:
        eval_loader = None

    # Loss as specified in the paper
    criterion = PanopticLoss(
        ce_weight=1,
        mse_weight=200,
        l1_weight=0.01,
        top_k_percent=0.2,
        pr_weight=1
    )

    optimizer = configure_optimizer(model, weight_decay=0.1)
    schedule_params = config['TRAIN']['schedule_params']

    if 'steps_per_epoch' in schedule_params:
        n_steps = schedule_params['steps_per_epoch']

        if n_steps != len(train_loader):
            schedule_params['steps_per_epoch'] = len(train_loader)
            print(f'Steps per epoch adjusted from {n_steps} to {len(train_loader)}')

    scheduler = lr_scheduler.OneCycleLR(optimizer, **schedule_params)
    scaler = GradScaler()

    if 'epochs' in config['TRAIN']['schedule_params']:
        epochs = config['TRAIN']['schedule_params']['epochs']
    else:
        raise Exception('Number of training epochs not defined!')

    # Step tracking for W&B logging

    global_step = 0

    for epoch in range(epochs):

        global_step = train(train_loader, model, criterion, optimizer,
                            scheduler, scaler, epoch, config, global_step)

        is_val_epoch = (epoch + 1) % config['EVAL']['epochs_per_eval'] == 0
        is_last_epoch = (epoch + 1) % epochs == 0

        if eval_loader is not None and (is_val_epoch or is_last_epoch):
            validate(eval_loader, model, criterion, epoch, config, step =global_step)

        save_now = (epoch + 1) % config['TRAIN']['save_freq'] == 0
        if save_now:
               save_path = os.path.join(config['TRAIN']['model_dir'], f"{config['model_name']}-{epoch + 1}_checkpoint.pth.tar")
               torch.save({
                   'arch': "PanopticDeepLabPR",
                   'state_dict': model.state_dict(),
                   'norms' : norms
               }, save_path)

    wandb.finish()
    return config

def train(
        train_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        scaler,
        epoch,
        config,
        global_step
):
    batch_time = ProgressAverageMeter('Time', ':6.3f')
    data_time = ProgressAverageMeter('Data', ':6.3f')
    loss_meters = None

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time],
        prefix=f"Epoch: [{epoch}]"
    )

    class_names = ['background', 'mitochondrion']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metric_dict = {}
    for metric_params in config['TRAIN']['metrics']:
        reg_name = metric_params['name']
        metric_name = metric_params['metric']
        metric_params = {k: v for k, v in metric_params.items() if k not in ['name', 'metric']}
        metric_dict[reg_name] = metrics.__dict__[metric_name](metrics.EMAMeter, **metric_params)

    meters = metrics.ComposeMetrics(metric_dict, class_names)
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = batch['image']
        target = {k: v for k, v in batch.items() if k not in ['image', 'fname']}

        images = images.to(device, non_blocking = True)
        target = {k: tensor.to(device, non_blocking=True) for k, tensor in target.items()}

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                output = model(images)
                loss, aux_loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images)
            loss, aux_loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if loss_meters is None:
            loss_meters = {'total_loss': ProgressEMAMeter('total_loss', ':.4e')}
            loss_meters['total_loss'].update(loss.item())
            for k, v in aux_loss.items():
                loss_meters[k] = ProgressEMAMeter(k, ':.4e')
                loss_meters[k].update(v)
                progress.meters.append(loss_meters[k])
        else:
            loss_meters['total_loss'].update(loss.item())
            for k, v in aux_loss.items():
                loss_meters[k].update(v)

        with torch.no_grad():
            meters.evaluate(output, target)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['TRAIN']['print_freq'] == 0:
            progress.display(i)

        wandb_train_dict = {
            f"train/loss_{k}": meter.avg
            for k, meter in loss_meters.items() if k != 'total_loss'
        }
        wandb_train_dict["train/total_loss"] = loss_meters['total_loss'].avg
        wandb_train_dict["train/learning_rate"] = scheduler.get_last_lr()[0]
        wandb_train_dict["epoch"] = epoch

        wandb.log(wandb_train_dict, step=global_step)
        global_step += 1

    print('\n')
    print(f'Epoch {epoch} training metrics:')
    meters.display()

    return global_step


def validate(
        eval_loader,
        model,
        criterion,
        epoch,
        config,
        step
):
    class_names = ['background', 'mitochondrion']

    metric_dict = {}
    for metric_params in config['EVAL']['metrics']:
        reg_name = metric_params['name']
        metric_name = metric_params['metric']
        metric_params = {k: v for k, v in metric_params.items() if k not in ['name', 'metric']}
        metric_dict[reg_name] = metrics.__dict__[metric_name](metrics.AverageMeter, **metric_params)

    meters = metrics.ComposeMetrics(metric_dict, class_names)
    batch_time = ProgressAverageMeter('Time', ':6.3f')
    loss_meters = None

    progress = ProgressMeter(
        len(eval_loader),
        [batch_time],
        prefix='Validation: '
    )

    engine = PanopticDeepLabEngine(model, **config['EVAL']['engine_params'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    for i, batch in enumerate(eval_loader):
        end = time.time()
        images = batch['image']
        target = {k: v for k, v in batch.items() if k not in ['image', 'fname']}

        images = images.to(device, non_blocking=True)
        target = {k: tensor.to(device, non_blocking=True) for k, tensor in target.items()}

        output = engine.infer(images)
        semantic = engine._harden_seg(output['sem'])
        output['pan_seg'] = engine.postprocess(
            semantic, output['ctr_hmp'], output['offsets']
        )
        target['pan_seg'] = engine.postprocess(
            target['sem'].unsqueeze(1), target['ctr_hmp'], target['offsets']
        )

        loss, aux_loss = criterion(output, target)

        if loss_meters is None:
            loss_meters = {}
            for k, v in aux_loss.items():
                loss_meters[k] = ProgressAverageMeter(k, ':.4e')
                loss_meters[k].update(v)
                progress.meters.append(loss_meters[k])
        else:
            for k, v in aux_loss.items():
                loss_meters[k].update(v)

        with torch.no_grad():
            meters.evaluate(output, target)

        batch_time.update(time.time() - end)

        if i % config['TRAIN']['print_freq'] == 0:
            progress.display(i)

    print('\n')
    print(f'Validation results:')
    meters.display()

    wandb_val_dict = {"epoch": epoch}

    if loss_meters is not None:
        wandb_val_dict.update({
            f"val/loss_{k}": meter.avg
            for k, meter in loss_meters.items() if k != 'total_loss'
        })

        wandb_val_dict["val/total_loss"] = loss_meters['total_loss'].avg

    if hasattr(meters, 'meters'):
        for name, meter in meters.meters.items():
            if hasattr(meter, 'avg'):
                wandb_val_dict[f"val_metrics/{name}"] = meter.avg

    wandb.log(wandb_val_dict, step=step)



def configure_optimizer(model, weight_decay=0.1, **kwargs):
    decay = set()
    no_decay = set()

    blacklist = (torch.nn.BatchNorm2d,torch.nn.LayerNorm, CustomLayerNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = f"{mn}.{pn}" if mn else pn

            if full_name.endswith('bias'):
                no_decay.add(full_name)
            elif full_name.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(full_name)
            else:
                decay.add(full_name)

    param_dict = {pn: p for pn, p in model.named_parameters()}

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (len(inter_params) == 0), "Overlapping decay and no decay"
    assert (len(param_dict.keys() - union_params) == 0), "Missing decay parameters"

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    return optim.AdamW(param_groups, **kwargs)

def load_encoder_weights(pretraining_path: str, encoder: str) -> dict:
    '''Loads encoder weights depending on the encoder type'''
    
    state = torch.load(pretraining_path, weights_only = False)
    state_dict = state.get('state_dict', state)

    if 'resnet' in encoder:
        
        for k in list(state_dict.keys()):
            clean_k = k.replace('module.','')
    
            if clean_k.startswith('fc'):
                del state_dict[k]
                continue

            if clean_k == 'conv1.weight':
                state_dict[k] = state_dict[k].mean(dim=1, keepdim=True)

            state_dict['encoder.' + clean_k] = state_dict[k]
            del state_dict[k]

    elif 'convnext' in encoder:

        for k in list(state_dict.keys()):
            clean_k = k.replace('module.','')
    
            if clean_k.startswith('fc'):
                del state_dict[k]
                continue
    
            if clean_k == 'downsample_layers.0.0.weight':
                state_dict[k] = state_dict[k].mean(dim=1, keepdim=True)
                
            state_dict['encoder.' + 'model.' + clean_k] = state_dict[k]
            del state_dict[k]

    return state, state_dict

def parse_args():
    parser = argparse.ArgumentParser(description="YAML config")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to yaml config file"
    )
    return parser.parse_args()


class ProgressAverageMeter(metrics.AverageMeter):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        super().__init__()

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressEMAMeter(metrics.EMAMeter):
    def __init__(self, name, fmt=':f', momentum=0.98):
        self.name = name
        self.fmt = fmt
        super().__init__(momentum)

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
