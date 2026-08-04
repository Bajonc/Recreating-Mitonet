# Recreating-Mitonet
The goal of this project is to provide an easily modifiable and understandable pipeline for recreating MitoNet, a model for mitochondria panoptic segmentation from this paper: https://doi.org/10.1016/j.cels.2022.12.006. Any mentions of 'the paper' in this code refer to this work. The training is divided into self-supervised pretraining and training on a labeled dataset.

### My contribution
Most of the code has been taken from the following repositories:
- https://github.com/volume-em/empanada-napari
- https://github.com/facebookresearch/swav

I contributed by fixing bugs and adapting the code to match the specific task at hand 

### Pretraining
MitoNet uses SwaV to pretrain on 1.5M unlabeled electron microscopy mitochondria images, the code from this part required minor changes

### Training
Done on labeled electron microscopy mitochondria images, the code from this part required major changes

### Tutorial
- fill in a pretraining script of your choosing and run it
- put the path to your pretrained weights into the config file
- fill in the training script and config file
- run the training script

I recommend using some sort of environment manager as the version and package requirements can differ wildly between training and pretraining 
