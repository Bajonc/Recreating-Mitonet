# Recreating-MitoNet
The goal of this project is to provide an easily modifiable and understandable pipeline for recreating MitoNet, a model for mitochondria panoptic segmentation from this paper: https://doi.org/10.1016/j.cels.2022.12.006. Any mentions of 'the paper' in this code refer to this work. The training is divided into self-supervised pretraining and training on a labeled dataset.

### My contribution
Most of the code has been taken from the following repositories:
- https://github.com/volume-em/empanada-napari
- https://github.com/facebookresearch/swav

I contributed by fixing bugs, adapting the code to match the specific task at hand, adding benchmark scripts and adapting encoders that were taken from:
- https://github.com/facebookresearch/ConvNeXt
- https://github.com/microsoft/swin-transformer
   

### Pretraining
MitoNet uses SwAV to pretrain on 1.5M unlabeled electron microscopy mitochondria images, the code from this part required minor changes

### Training
Done on labeled electron microscopy mitochondria images, the code from this part required major changes

### Tutorial
To recreate MitoNet:
- Switch to environment fulfilling SwAV requirements
- Fill out the shell script in the SwAV directory and run it
- Switch to environment fulfilling MitoNet training requirements 
- Fill out the train_config.yaml file
- Fill out the train.sh script and run it

To benchmark:
- Switch to the same environment as MitoNet
- Fill out the benchmark.yaml file
- Run the following:
```bash
python benchmark.py -c benchmark.yaml
```

Different encoders:
| Family | Variants |
|---|---|
| ResNet | resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2 |
| Swin | swin_t, swin_s, swin_b, swin_l |
| SwinV2 | swinv2_t, swinv2_s, swinv2_b, swinv2_l |
| ConvNeXt | convnext_tiny, convnext_small, convnext_base, convnext_large |
- All encoders use the PanopticDeepLab architecture, changing the decoder_channels, aspp_channels and crop sizes in the training transformations may be necessary when training with Swin encoders. 
- Pretraining on different encoders than ResNet50 is not supported


## Models

Results were benchmarked with global set to True and thing_area set to 150 in benchmark.yaml

| encoder | pretrain | semantic_iou | f1@50 | f1@75 | AP@50 | AP@75 | PQ | MSA | #params | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50 | CEM1.5M | 0.771 | 0.757 | 0.669 | 0.609 | 0.502 | 0.662 | 0.455 | 25M | [HuggingFace](https://huggingface.co/Bajonc/resnet50_cem1.5M_recreated/resolve/main/resnet50_cem1.5M_recreated_pretraining.pth.tar) |
| resnet50 | ImageNet | | | | | | | | 25M | [HuggingFace](https://huggingface.co/Bajonc/resnet50_imagenet_pretraining/resolve/main/resnet50_ImageNet_pretrain.pth.tar) |
