import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from models import get_eval_model
from tools import get_args, AverageMeter, Logger, file_exist_check
from datetime import datetime
import copy
from linear_eval import eval_model, get_eval_optimizer
from finetune_eval import get_eval_dataset, get_save_folder

# Finetune + Contrastive Regularization

def main(args):
    print("Finetune with Contrastive + SGD + weight_decay", f"theta:{args.eval.theta}")
    model = get_eval_model(args, backbone_only=False)
    train_loader, test_loader, num_classes = get_eval_dataset(args)

    # constant learning rate for classifier
    if hasattr(model, 'output_dim'):
        in_features = model.output_dim
    else:
        in_features = model.feature_dim
    classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True).to(args.device)

    model = model.to(args.device)
    net = torch.nn.Sequential(model, classifier)
    net = torch.nn.DataParallel(net)
    
    optimizer, lr_scheduler = get_eval_optimizer(args, net, train_loader, constant_predictor_lr=True)

    contrastive_loss_meter = AverageMeter(name='Contrastive Loss')
    train_loss_meter = AverageMeter(name='Train Loss')
    train_acc_meter = AverageMeter(name='Train Accuracy')

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Training')
    for epoch in global_progress:
        net.train()
        contrastive_loss_meter.reset()
        train_loss_meter.reset()
        train_acc_meter.reset()
        for idx, ((images1, images2), labels) in enumerate(train_loader):
            net.zero_grad()
            data_dict = model.forward(images1.to(args.device, non_blocking=True), images2.to(args.device, non_blocking=True))
            preds = classifier(data_dict["feature"][0])

            contrasrive_loss = data_dict['loss'].mean() # ddp
            pred_loss = F.cross_entropy(preds, labels.to(args.device)) # reduction is mean
            loss = pred_loss + args.eval.theta * contrasrive_loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            
            contrastive_loss_meter.update(contrasrive_loss.item(), n=images1.shape[0])
            train_loss_meter.update(pred_loss.item(), n=images1.shape[0])
            correct = (preds.argmax(dim=1) == labels.to(args.device)).sum().item()
            train_acc_meter.update(correct/preds.shape[0], n=preds.shape[0])

        epoch_dict = {"epoch":epoch, "train loss": train_loss_meter.avg, \
            "contrastive loss": contrastive_loss_meter.avg,"train acc": train_acc_meter.avg}
        global_progress.set_postfix(epoch_dict)
        if "feature" in args.dataset.name:
            train_loader.dataset.step()

    # Finish training, start testing
    print(f'Train Accuracy = {train_acc_meter.avg:.2%}')
    eval_net = torch.nn.Sequential(model.backbone, classifier)
    eval_net = torch.nn.DataParallel(eval_net)
    test_acc_meter, test_acc5_meter, test_loss_meter = eval_model(args, eval_net, test_loader)    

    save_folder = get_save_folder(args)
    save_finetune_path = os.path.join(save_folder, \
        f"finetune_contrastive_{int(args.percent*100)}%_{args.seed}_checkpoint.pth")
    torch.save({
                'test_loss': test_loss_meter.avg,
                'train_acc': train_acc_meter.avg,
                'test_acc': test_acc_meter.avg,
                'state_dict': model.state_dict(),
            }, save_finetune_path)

    return  contrastive_loss_meter.avg, train_loss_meter.avg, train_acc_meter.avg, \
        test_loss_meter.avg, test_acc_meter.avg, test_acc5_meter.avg
    #  "contrastive_loss", "train_loss" , "train_acc", "test_loss", "test_acc", "test_acc5"

if __name__ == "__main__":
    args = get_args()
    main(args=args)
