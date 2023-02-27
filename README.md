# The Trade-off between Universality and Label Efficiency of Representations from Contrastive Learning

This repository contains the Pytorch implementation of our method in the paper 
```
The Trade-off between Universality and Label Efficiency of Representations from Contrastive Learning
Zhenmei Shi*, Jiefeng Chen*, Kunyang Li, Jayaram Raghuram, Xi Wu, Yingyu Liang, Somesh Jha
```

This paper is published as an Spotlight at ICLR 2023. 

## Requirements

It is tested under Ubuntu Linux 20.04 and Python 3.9 environment and requires some packages to be installed.

Pytorch >= 1.12.1 (guide is [here](https://pytorch.org/get-started/locally/))

Install other used packages:

`pip install -r requirements.txt`

`pip install git+https://github.com/openai/CLIP.git`


## Run Experiments
You may modify the config file to run your own experiments. Here we give some examples. 

### Training

An example to train a MoCo v2 model. Just need to change the config file `moco_cifar_pretrain.yaml`.

`python main.py --data_dir ./data/ --log_dir ./logs/ --config-file ./configs/moco_cifar_pretrain.yaml --ckpt_dir ./checkpoints/ --download --hide_progress --save_interval 10`

An example to continue training from a checkpoint.

`python main.py --data_dir ./data/ --log_dir ./logs/ --config-file ./configs/moco_cifar_pretrain.yaml --ckpt_dir ./checkpoints/ --download --hide_progress --save_interval 10 --start_epoch {number} --save_dir checkpoints/{ckp_dir}`

### Evaluation

#### Linear Probing

`python linear_eval.py --config-file ./configs/moco_cifar_eval_sgd.yaml --data_dir ./data/ --log_dir ./logs/ --ckpt_dir checkpoints/ --eval_from ./checkpoints/{model_ckpt} --percent 1.0 --hide_progress`

#### Finetune

`python finetune_eval.py --config-file ./configs/moco_cifar_finetune_contrastive_eval.yaml --data_dir ./data/ --log_dir ./logs/ --ckpt_dir checkpoints/ --eval_from ./checkpoints/{model_ckpt} --percent 1.0 --hide_progress`

#### Finetune + Contrastive Regularization

`python finetune_contrastive_eval.py --config-file ./configs/moco_cifar_finetune_contrastive_eval.yaml --data_dir ./data/ --log_dir ./logs/ --ckpt_dir checkpoints/ --eval_from ./checkpoints/{model_ckpt} --percent 1.0 --hide_progress`

### CLIP

First, save train, test and augmentation features (Please download the imagenet to the folder `./data/imagenet/`):

`python get_model_feature.py --config-file ./configs/clip_castrate_imagenet.yaml --data_dir ./data/ --log_dir ./logs/ --ckpt_dir checkpoints/ --hide_progress --start_epoch 0 --end_epoch 5`

#### Linear Probing for CLIP

`python linear_eval.py --config-file ./configs/clip_castrate:ViT-L-14_imagenet_feature_eval_sgd.yaml --data_dir ./data/ --log_dir ./logs/ --ckpt_dir checkpoints/ --percent 1.0 --hide_progress`

#### Finetune for CLIP

`python finetune_eval.py --config-file ./configs/clip_castrate:ViT-L-14_imagenet_feature_simclr_mlp_eval_sgd.yaml --data_dir ./data/ --log_dir ./logs/ --ckpt_dir checkpoints/ --percent 1.0 --hide_progress`

#### Finetune + Contrastive Regularization for CLIP

`python finetune_contrastive_eval.py --config-file ./configs/clip_castrate:ViT-L-14_imagenet_feature_simclr_mlp_eval_sgd.yaml --data_dir ./data/ --log_dir ./logs/ --ckpt_dir checkpoints/ --percent 1.0 --hide_progress`
