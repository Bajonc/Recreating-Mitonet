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
To recreate mitonet:
- Switch to environment fulfilling swav requirements
- Fill out the shell script in the swav directory and run it
- Switch to environment fulfilling mitonet training requirements 
- Replace the capitalized values in the train_config.yaml The most important part is the path to the final checkpoint produced by the pretraining script.
- Replace the capitalized values in the train.sh script and run it

To benchmark:
- Download original mitonet weights to benchmark against
- Switch to environment fulfilling mitonet training requirements
- Fill in the benchmark.yaml file with the paths to the models you wish to benchmark and the downloaded weights
- Run the benchmark.py file

To experiment:
- Currently the code supports different ResNet sizes and different ConvNeXt sizes that can be changed under model architecture in the training script.
- Pretraining of different encoders than ResNet is currently not supported 

It is recommended to use a package manager as the requirements for pretraining and training differ
