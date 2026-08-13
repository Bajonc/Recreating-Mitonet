import os
import time
import yaml
import torch
from torch.utils.data import DataLoader
import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2

from empanada.models.panoptic_deeplab import PanopticDeepLabPR
from empanada.inference.engines import PanopticDeepLabEngine
from empanada.data.single_class_instance_dataset import SingleClassInstanceDataset
from empanada.data.utils.transforms import FactorPad
import empanada.metrics as metrics

from my_train import ProgressMeter, ProgressAverageMeter

cem_norms = {'mean': 0.574, 'std': 0.176}


def load_model(model_path, device, fallback_norms = cem_norms):

    weights = torch.load(model_path, map_location = device)

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

    state_dict = weights.get('state_dict', weights)
    norms = weights.get('norms', fallback_norms)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, norms

def get_eval_loader(norms, config):

    eval_tfs = A.Compose([
        FactorPad(128),
        A.Normalize(**norms),
        ToTensorV2()
    ])

    eval_dataset = SingleClassInstanceDataset(data_dir=config['EVAL']['eval_dir'], transforms=eval_tfs)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 pin_memory=torch.cuda.is_available(),
                                 num_workers=4)
    return eval_loader

def validate(model, eval_loader, config, device):
    class_names = {1: 'mitochondrion'}

    metric_dict = {}
    for metric_params in config['EVAL']['metrics']:
        reg_name = metric_params['name']
        metric_name = metric_params['metric']
        params = {k: v for k, v in metric_params.items() if k not in ['name', 'metric']}
        metric_dict[reg_name] = getattr(metrics, metric_name)(metrics.AverageMeter, **params)

    meters = metrics.ComposeMetrics(metric_dict, class_names, reset_on_print=False)
    engine = PanopticDeepLabEngine(model, **config['EVAL']['engine_params'])

    batch_time = ProgressAverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time],
        prefix='Evaluating: '
    )

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            end = time.time()
            images = batch['image'].to(device, non_blocking=True)
            target = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ['image', 'fname']}

            output = engine.infer(images)
            semantic = engine._harden_seg(output['sem'])

            output['pan_seg'] = engine.postprocess(
                semantic, output['ctr_hmp'], output['offsets']
            )
            target['pan_seg'] = engine.postprocess(
                target['sem'].unsqueeze(1), target['ctr_hmp'], target['offsets']
            )

            meters.evaluate(output, target)

            batch_time.update(time.time() - end)
            if i % config['EVAL'].get('print_freq', 50) == 0:
                progress.display(i)

    results = {}
    for metric_name, metric_obj in meters.metrics_dict.items():

        if config['EVAL']['global']:
            global_scores = metric_obj.calculate_global()
            results[f"{metric_name}"] = float(global_scores)
        else:
            avg_scores = metric_obj.average()
            for label, score in avg_scores.items():
                class_label = class_names[label]
                results[f"{class_label}_{metric_name}"] = float(score)
    
    return results

def run_benchmark(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary_results = []

    # evaluate original model 
    Mitonet = torch.load(config["ORIGINAL_MODEL"], map_location = device)
    Mitonet.to(device)
    Mitonet.eval() 

    Mitonet_norms = {'mean': 0.57571, 'std': 0.12765}

    eval_loader = get_eval_loader(Mitonet_norms, config)


    scores = validate(Mitonet, eval_loader, config, device)

    scores['model'] = 'MitoNet'
    summary_results.append(scores)

    for model_path in config['MODELS']:
        model_name = os.path.basename(model_path)

        if not os.path.exists(model_path):
            print(f"--> File not found at path: {model_path}. Skipping.")
            continue

        try:
            model, norms = load_model(model_path, device, fallback_norms=cem_norms)
            eval_loader = get_eval_loader(norms, config)

            scores = validate(model, eval_loader, config, device)

            scores['model'] = model_name
            summary_results.append(scores)

        except Exception as e:
            print(f"error evaluating {model_name}")

    
    spaces = 47  
    header = f"{'Model Checkpoint':<45} |"
    for metric in config['EVAL']['metrics']:
        header += f" {metric['name']:<12} |"
        spaces += 15  
    
    print("\n" + "=" * spaces)
    print(header)
    print("=" * spaces)
    
    for res in summary_results:
        output = f"{res['model'].replace('-120_checkpoint.pth.tar', ''):<45} |"
        for metric, result in res.items():
            if metric != 'model':
                output += f" {result:<12.4f} |"
        print(output)
    print("=" * spaces)
    



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

if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_benchmark(config)
