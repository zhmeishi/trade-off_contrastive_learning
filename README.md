# The Trade-off between Universality and Label Efficiency of Representations from Contrastive Learning

This repository is the official Pytorch implementation of our method in the paper

```
The Trade-off between Universality and Label Efficiency of Representations from Contrastive Learning

Zhenmei Shi*, Jiefeng Chen*, Kunyang Li, Jayaram Raghuram, Xi Wu, Yingyu Liang, Somesh Jha
```

This paper is published as a Spotlight at ICLR 2023 ([OpenReview link](https://openreview.net/forum?id=rvsbw2YthH_)).

## Requirements

It is tested under Ubuntu Linux 20.04 and Python 3.9 environment and requires some packages to be installed.

Pytorch >= 1.12.1 (guide is [here](https://pytorch.org/get-started/locally/))

Install other used packages:

```
pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git
```

## Prepare Datasets

Download some datasets under your datasets folder `/my/data/folder` following the link below:

[ImageNet](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh), [ImageNet32](https://patrykchrabaszcz.github.io/Imagenet32/), [GTSRB](https://drive.google.com/file/d/1f37CPYd9YYMHuRk6JM7Oy-nFqUvZLFR2/view?usp=sharing), [Fer2013](https://drive.google.com/drive/folders/1f0YDhph4amlXtDRdiMpDLn07ni0SI6Aj?usp=sharing), [FaceScrub](https://drive.google.com/drive/folders/1f0YDhph4amlXtDRdiMpDLn07ni0SI6Aj?usp=sharing)

ImageNet-Bird
```
n01514668, n01530575, n01534433, n01560419, n01592084, n01614925, n01795545, n01798484, n01807496, n01819313, n01828970, n01843065, n01855032, n02002556, n02007558, n02011460, n02017213, n02025239, n02033041, n02056570, n01514859, n01531178, n01537544, n01580077, n01601694, n01616318, n01796340, n01806143, n01817953, n01820546, n01829413, n01843383, n01855672, n02002724, n02009229, n02012849, n02018207, n02027492, n02037110, n02058221, n01518878, n01532829, n01558993, n01582220, n01608432, n01622779, n01797886, n01806567, n01818515, n01824575, n01833805, n01847000, n01860187, n02006656, n02009912, n02013706, n02018795, n02028035, n02051845
```

ImageNet-Vehicle
```
n02701002, n02797295, n02835271, n03100240, n03345487, n03393912, n03444034, n03478589, n03594945, n03670208, n03777568, n03791053, n03796401, n03895866, n03977966, n04065272, n04252077, n04285008, n04335435, n04461696, n04467665, n04509417
n02704792, n02814533, n02930766, n03272562, n03384352, n03417042, n03445924, n03538406, n03599486, n03770679, n03785016, n03792782, n03868242, n03930630, n04037443, n04204347, n04252225, n04310018, n04389033, n04465501, n04482393
```

ImageNet-Cat/Ball/Shop/Clothing/Fruit
```
n02123045, n02123394, n02124075, n02127052, n02128757, n02129165, n02130308, n02791270, n02802426, n02927161, n03089624, n03445777, n03942813, n04118538, n04254680, n04443257, n04540053, n06874185, n02123159, n02123597, n02125311, n02128385, n02128925, n02129604, n02776631, n02799071, n02871525, n03032252, n03134739, n03461385, n04023962, n04200800, n04409515, n04462240, n06794110, n02669723, n02807133, n02869837, n03026506, n03127747, n03450230, n03623198, n03724870, n03775071, n03877472, n04209133, n04259630, n04584207, n02730930, n02817516, n02892767, n03124170, n03379051, n03594734, n03710637, n03763968, n03787032, n04162706, n04254777, n04532106, n07742313, n07745940, n07747607, n07749582, n07753113, n07753275, n07753592, n07754684, n07760859, n07768694, n11879895, n12144580, n12267677, n12620546, n12768682, n13133613

```

Other datasets will be downloaded automatically by setting `download=True`.

## Run Experiments

You may modify the config file to run your own experiments. Here we give some examples.

### Training

Here is an example of training a MoCo v2 model. You just need to change the config file `moco_cifar_pretrain.yaml`.

```
python main.py --data_dir /my/data/folder --log_dir ./logs/ --config-file ./configs/moco_cifar_pretrain.yaml --ckpt_dir ./checkpoints/ --download --hide_progress --save_interval 10
```

An example to continue training from a checkpoint.

```
python main.py --data_dir /my/data/folder --log_dir ./logs/ --config-file ./configs/moco_cifar_pretrain.yaml --ckpt_dir ./checkpoints/ --download --hide_progress --save_interval 10 --start_epoch {number} --save_dir checkpoints/{ckp_dir}
```

### Evaluation

#### Linear Probing

```
python linear_eval.py --config-file ./configs/moco_cifar_eval_sgd.yaml --data_dir /my/data/folder --log_dir ./logs/ --ckpt_dir checkpoints/ --eval_from ./checkpoints/{model_ckpt} --percent 1.0 --hide_progress
```

#### Finetune

```
python finetune_eval.py --config-file ./configs/moco_cifar_finetune_contrastive_eval.yaml --data_dir /my/data/folder --log_dir ./logs/ --ckpt_dir checkpoints/ --eval_from ./checkpoints/{model_ckpt} --percent 1.0 --hide_progress
```

#### Finetune + Contrastive Regularization

```
python finetune_contrastive_eval.py --config-file ./configs/moco_cifar_finetune_contrastive_eval.yaml --data_dir /my/data/folder --log_dir ./logs/ --ckpt_dir checkpoints/ --eval_from ./checkpoints/{model_ckpt} --percent 1.0 --hide_progress
```

### CLIP

First, save train, test, and augmentation features (Please download the ImageNet to the folder `/my/data/folder/imagenet/`):

```
python get_model_feature.py --config-file ./configs/clip_castrate_imagenet.yaml --data_dir /my/data/folder --log_dir ./logs/ --ckpt_dir checkpoints/ --hide_progress --start_epoch 0 --end_epoch 5
```

#### Linear Probing for CLIP

```
python linear_eval.py --config-file ./configs/clip_castrate:ViT-L-14_imagenet_feature_eval_sgd.yaml --data_dir /my/data/folder --log_dir ./logs/ --ckpt_dir checkpoints/ --percent 1.0 --hide_progress
```

#### Finetune for CLIP

```
python finetune_eval.py --config-file ./configs/clip_castrate:ViT-L-14_imagenet_feature_simclr_mlp_eval_sgd.yaml --data_dir /my/data/folder --log_dir ./logs/ --ckpt_dir checkpoints/ --percent 1.0 --hide_progress
```

#### Finetune + Contrastive Regularization for CLIP

```
python finetune_contrastive_eval.py --config-file ./configs/clip_castrate:ViT-L-14_imagenet_feature_simclr_mlp_eval_sgd.yaml --data_dir /my/data/folder --log_dir ./logs/ --ckpt_dir checkpoints/ --percent 1.0 --hide_progress
```

## Citation

Please cite our work if you use the codebase:

```
@inproceedings{
shi2023the,
title={The Trade-off between Universality and Label Efficiency of Representations from Contrastive Learning},
author={Zhenmei Shi and Jiefeng Chen and Kunyang Li and Jayaram Raghuram and Xi Wu and Yingyu Liang and Somesh Jha},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=rvsbw2YthH_}
}
```

## License

Please refer to the [LICENSE](LICENSE).
