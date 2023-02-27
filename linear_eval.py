import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from tools import get_args, AverageMeter, accuracy
from optimizers import get_optimizer, LR_Scheduler
from sklearn.model_selection import StratifiedShuffleSplit
from get_model_feature import get_feature

# Linear Probing
def eval_model(args, classifier, dataloader):
    classifier.eval()
    acc_meter = AverageMeter(name='Accuracy')
    acc5_meter = AverageMeter(name='Accuracy top 5')
    loss_meter = AverageMeter(name='Loss')
    acc_meter.reset()
    acc5_meter.reset()
    loss_meter.reset()
    for idx, (images, labels) in enumerate(dataloader):
        with torch.no_grad():
            features = images.to(args.device)
            output = classifier(features)
            preds = output.argmax(dim=1)
            loss = F.cross_entropy(output, labels.to(args.device))

            loss_meter.update(loss.item(), n=features.shape[0])
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0], n=preds.shape[0])
            acc5_meter.update(accuracy(output.cpu(), labels, topk=(5,))[0].item(), n=preds.shape[0])

    print(f'Test Accuracy = {acc_meter.avg:.2%}')
    print(f'Test Top 5 Accuracy = {acc5_meter.avg:.2%}')
    return acc_meter, acc5_meter, loss_meter

def get_eval_optimizer(args, net, train_loader, constant_predictor_lr=False):
    optimizer = get_optimizer(
        args.eval.optimizer.name, net, 
        lr=args.eval.base_lr*args.eval.batch_size/256, 
        momentum=args.eval.optimizer.momentum, 
        weight_decay=args.eval.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/256, 
        args.eval.num_epochs, args.eval.base_lr*args.eval.batch_size/256, args.eval.final_lr*args.eval.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=constant_predictor_lr # see the end of section 4.2 predictor
    )
    return optimizer, lr_scheduler
    
def main(args):
    args.linear=True
    print("Linear Probing")
    # Extract pretrained train feature
    train_features, train_feature_labels, test_features, test_feature_labels = get_feature(args)
    
    if args.save_feature_only:
        print(train_features.shape)
        save_path = os.path.join(os.path.dirname(args.eval_from),\
            os.path.basename(args.eval_from).split(".")[0] + "_" + args.dataset.name + "_features.pt")
        print(save_path)
        torch.save({"features":train_features, "feature_labels":train_feature_labels}, save_path)
        return

    test_dataset = torch.utils.data.TensorDataset(test_features, test_feature_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval.batch_size, shuffle=False, num_workers=4)

    num_classes = len(np.unique(np.concatenate((train_feature_labels, test_feature_labels), axis=0)))

    # subsample
    if args.percent == 1.0:
        select_features = train_features
        select_feature_labels = train_feature_labels
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_classes, train_size=args.percent, random_state=0)
        train_index, test_index = next(iter(sss.split(np.zeros(train_feature_labels.shape[0]), train_feature_labels.cpu().numpy())))
        select_features = train_features[train_index, :]
        select_feature_labels = train_feature_labels[train_index]

    sub_dataset = torch.utils.data.TensorDataset(select_features, select_feature_labels)
    sub_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=args.eval.batch_size, shuffle=True, num_workers=4)

    classifier = nn.Linear(in_features=train_features[0].shape[0], out_features=num_classes, bias=True).to(args.device)
    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.DataParallel(classifier)

    optimizer, lr_scheduler = get_eval_optimizer(args, classifier, sub_loader)

    train_loss_meter = AverageMeter(name='Train Loss')
    train_acc_meter = AverageMeter(name='Train Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        train_loss_meter.reset()
        train_acc_meter.reset()
        classifier.train()
        local_progress = tqdm(sub_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        
        for idx, (features, labels) in enumerate(local_progress):
            classifier.zero_grad()
            preds = classifier(features)
            loss = F.cross_entropy(preds, labels.to(args.device)) # reduction is mean

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_loss_meter.update(loss.item(), n=features.shape[0])
            correct = (preds.argmax(dim=1) == labels.to(args.device)).sum().item()
            train_acc_meter.update(correct/preds.shape[0], n=preds.shape[0])

        epoch_dict = {"epoch":epoch, "train loss": train_loss_meter.avg, "train acc": train_acc_meter.avg}
        global_progress.set_postfix(epoch_dict)
    
    # Finish training, start testing
    print(f'Train Accuracy = {train_acc_meter.avg:.2%}')
    
    test_acc_meter, test_acc5_meter, test_loss_meter = eval_model(args, classifier, test_loader)   
    return  0.0, train_loss_meter.avg, train_acc_meter.avg, \
        test_loss_meter.avg, test_acc_meter.avg, test_acc5_meter.avg
    #  "contrastive_loss", "train_loss" , "train_acc", "test_loss", "test_acc", "test_acc5"


if __name__ == "__main__":
    main(args=get_args())
