import os
import time
import yaml
import torch
from torch.utils.data import DataLoader
import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2

from empanada import models
from empanada.inference.engines import PanopticDeepLabEngine
from empanada.data.single_class_instance_dataset import SingleClassInstanceDataset
from empanada.data.utils.transforms import FactorPad
import empanada.metrics as metrics

from train import ProgressMeter, ProgressAverageMeter, sliding_window_infer

cem_norms = {'mean': 0.574, 'std': 0.176}


def load_model(model_path, config, device, fallback_norms=cem_norms):
    weights = torch.load(model_path, map_location=device)

    arch = config['MODELS_ARCHITECTURE']['arch']
    model = models.__dict__[arch](**config['MODELS_ARCHITECTURE'])

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


def validate(model, eval_loader, config, device, model_name):
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

    n_batches = len(eval_loader)
    progress = ProgressMeter(n_batches, [batch_time], prefix=f'Evaluating:  {model_name} ')

    loader_iter = eval_loader 



    with torch.no_grad():
        for i, batch in enumerate(loader_iter):
            end = time.time()
            images = batch['image'].to(device, non_blocking=True)
            target = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ['image', 'fname']}

           
            if config['EVAL']['patch_based'] and model_name != 'MitoNet':
                patch_size = config['MODELS_ARCHITECTURE']['decoder_channels']
                overlap = config['EVAL']['sliding_window_inference_overlap']
                output = sliding_window_infer(
                    images, engine,
                    patch_size=patch_size,
                    overlap=overlap,
                    batch_chunk=config['EVAL'].get('patch_chunk_size', 16),
                )
            else:
                output = engine.infer(images)

            semantic = engine._harden_seg(output['sem'])

            output['pan_seg'] = engine.postprocess(
                semantic, output['ctr_hmp'], output['offsets'], is_target= False
            )
            target['pan_seg'] = engine.postprocess(
                target['sem'].unsqueeze(1), target['ctr_hmp'], target['offsets'], is_target = True
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


def run_benchmark(config, patch_mode="sliding"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary_results = []
    
    # evaluate original model 
    if config["ORIGINAL_MODEL"]:
        Mitonet = torch.load(config["ORIGINAL_MODEL"], map_location = device)
        Mitonet.to(device)
        Mitonet.eval() 

        Mitonet_norms = {'mean': 0.57571, 'std': 0.12765}

        eval_loader = get_eval_loader(Mitonet_norms, config)


        scores = validate(Mitonet, eval_loader, config, device, model_name="MitoNet")

        scores['model'] = 'MitoNet'
        summary_results.append(scores)

    
    for model_path in config['MODELS']:
        model_name = os.path.basename(model_path)

        if not os.path.exists(model_path):
            print(f"--> File not found at path: {model_path}. Skipping.")
            continue

        try:
            model, norms = load_model(model_path, config, device, fallback_norms=cem_norms)
            eval_loader = get_eval_loader(norms, config)

            scores = validate(
                model, eval_loader, config, device, model_name
            )

            scores['model'] = model_name
            summary_results.append(scores)

        except Exception as e:
            print(f"error evaluating {model_name} with the following message {e}")

    spaces = 47
    header = f"{'Model Checkpoint':<45} |"
    for metric in config['EVAL']['metrics']:
        header += f" {metric['name']:<12} |"
        spaces += 15

    print("\n" + "=" * spaces)
    print(header)
    print("=" * spaces)

    for res in summary_results:
        output = f"{res['model'].replace('_checkpoint.pth.tar', 'ep'):<45} |"
        for metric, result in res.items():
            if metric != 'model':
                output += f" {result:<12.3f} |"
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
