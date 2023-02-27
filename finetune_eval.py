import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from augmentations import get_aug
from models import get_eval_model
from tools import get_args, AverageMeter
from datasets import get_dataset, subsample
import copy
from linear_eval import eval_model, get_eval_optimizer
import math

# Finetune all layer
def get_eval_dataset(args, get_reg=False):
    # if args.dataset.aug_num == 1:
    #     train_dataset = get_dataset( 
    #         transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), 
    #         train=True, 
    #         **args.dataset_kwargs
    #     )
    # else:
    assert(args.dataset.aug_num == -1 or args.dataset.aug_num >= 1)
    train_dataset = get_dataset( 
        transform=get_aug(train=True, train_classifier=True, **args.aug_kwargs), # Augmentation for Finetune
        train=True, 
        **args.dataset_kwargs
    )
    if args.dataset.aug_num >= 1 and "feature" in args.dataset.name:
        train_dataset.set_aug_num(args.dataset.aug_num)
    elif args.dataset.aug_num == -1:
        pass
    else:
        raise NotImplementedError
    
    sub_size = int(len(train_dataset)*args.percent)
    if get_reg:
        train_dataset, train_reg_dataset = subsample(train_dataset, sub_size, get_reg=get_reg)
        extend_ratio = int(len(train_reg_dataset)/len(train_dataset)) + 1
        extend_ratio = 2**math.floor(math.log(extend_ratio, 2))
        assert(extend_ratio > 1)
        train_reg_loader = torch.utils.data.DataLoader(
            dataset= train_reg_dataset,
            shuffle=True,
            batch_size=args.eval.batch_size * (extend_ratio - 1)//2, # Note here
            **args.dataloader_kwargs
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.eval.batch_size//2,
            shuffle=True,
            **args.dataloader_kwargs
        )
    else:
        extend_ratio = 2
        train_dataset = subsample(train_dataset, sub_size)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.eval.batch_size,
            shuffle=True,
            **args.dataloader_kwargs
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size * extend_ratio // 2, # Note here
        shuffle=False,
        **args.dataloader_kwargs
    )

    num_classes = len(np.unique(np.concatenate((train_loader.dataset.targets, test_loader.dataset.targets), axis=0)))
    if get_reg:
        print("train size:", len(train_dataset), "regularizer size:", len(train_reg_dataset))
        return train_loader, train_reg_loader, test_loader, num_classes
    else:
        return train_loader, test_loader, num_classes

def get_save_folder(args):
    if args.eval_from is not None:
        save_folder = os.path.dirname(args.eval_from)
    else:
        save_folder = os.path.join(args.ckpt_dir, args.name)
        os.makedirs(save_folder, exist_ok=True)
    return save_folder

def main(args):
    if args.dataset.aug_num == 1:
        print("Finetune with weight_decay")
    else:
        print("Finetune with weight_decay + Data Augmentation")
    model = get_eval_model(args, backbone_only=True)
    train_loader, test_loader, num_classes = get_eval_dataset(args)

    if hasattr(model, 'output_dim'):
        in_features = model.output_dim
    else:
        in_features = model.feature_dim
    classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True).to(args.device)
    
    # print(msg)
    model = model.to(args.device)
    net = torch.nn.Sequential(model, classifier)
    net = torch.nn.DataParallel(net)

    # args.eval.optimizer.name = "adam"
    # args.eval.base_lr /= 60
    # args.eval.optimizer.weight_decay *= 100

    optimizer, lr_scheduler = get_eval_optimizer(args, net, train_loader, constant_predictor_lr=True)

    train_loss_meter = AverageMeter(name='Train Loss')
    train_acc_meter = AverageMeter(name='Train Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        net.train()
        train_loss_meter.reset()
        train_acc_meter.reset()
        
        for idx, (images, labels) in enumerate(train_loader):
            if type(images) is list:
                images = images[0]
            net.zero_grad()
            preds = net(images.to(args.device))
            loss = F.cross_entropy(preds, labels.to(args.device)) # reduction is mean

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss_meter.update(loss.item(), n=images.shape[0])
            correct = (preds.argmax(dim=1) == labels.to(args.device)).sum().item()
            train_acc_meter.update(correct/preds.shape[0], n=preds.shape[0])
        
        epoch_dict = {"epoch":epoch, "train loss": train_loss_meter.avg, "train acc": train_acc_meter.avg}
        global_progress.set_postfix(epoch_dict)
        if "feature" in args.dataset.name:
            train_loader.dataset.step()

    # Finish training, start testing
    print(f'Train Accuracy = {train_acc_meter.avg:.2%}')
    test_acc_meter, test_acc5_meter, test_loss_meter = eval_model(args, net, test_loader)    

    save_folder = get_save_folder(args)
    save_finetune_path = os.path.join(save_folder, \
        f"finetune_{int(args.percent*100)}%_{args.seed}_checkpoint.pth")
    torch.save({
                'test_loss': test_loss_meter.avg,
                'train_acc': train_acc_meter.avg,
                'test_acc': test_acc_meter.avg,
                'state_dict': model.state_dict()
            }, save_finetune_path)

 
    return  0.0, train_loss_meter.avg, train_acc_meter.avg, \
        test_loss_meter.avg, test_acc_meter.avg, test_acc5_meter.avg
    #  "contrastive_loss", "train_loss" , "train_acc", "test_loss", "test_acc", "test_acc5"

if __name__ == "__main__":
    args=get_args()
    main(args=args)