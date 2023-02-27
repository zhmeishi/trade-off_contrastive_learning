import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from augmentations import get_aug
from models import get_model
from tools import get_args, AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from datetime import datetime
import shutil
import copy


def train_early_stop(device, args):
    
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args=args).to(device)
    model = torch.nn.DataParallel(model)
    best_model = copy.deepcopy(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    if args.start_epoch == 0:
        fail_cnt = 0
        best_training_loss = np.inf
    else:
        save_dict = torch.load(os.path.join(args.save_dir, f"checkpoint_{args.start_epoch}.pth"), map_location=device)
        msg = model.load_state_dict(save_dict['state_dict'], strict=True)
        msg = best_model.load_state_dict(save_dict['best_state_dict'], strict=True)
        optimizer.load_state_dict(save_dict['optimizer'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        fail_cnt = save_dict['fail_cnt']
        best_training_loss = save_dict['best_training_loss']
    
    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    train_loss_meter = AverageMeter(name="training loss")
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        if epoch < args.start_epoch:
            epoch_dict = {"epoch": epoch, "loss": train_loss_meter.avg}
            global_progress.set_postfix(epoch_dict)
            continue

        model.train()
        train_loss_meter.reset()
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):

            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            data_dict['loss'] = data_dict['loss'].mean()
            
            train_loss_meter.update(data_dict['loss'].item(), n=images1.shape[0])
            local_progress.set_postfix(data_dict)
        
        epoch_dict = {"epoch":epoch, "loss": train_loss_meter.avg}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
        if train_loss_meter.avg <= best_training_loss:
            best_training_loss = train_loss_meter.avg
            fail_cnt = 0
            best_model = copy.deepcopy(model)
        else:
            fail_cnt += 1
            if fail_cnt >= args.train.patience_epochs:
                print(f"Early stop trainnig at epoch {epoch+1}!")
                break

        if (epoch+1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, f"checkpoint_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_state_dict': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'fail_cnt': fail_cnt,
                'best_training_loss': best_training_loss
            }, model_path)
    
    save_model(args, epoch, best_model)


def train(device, args):
    
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args=args).to(device)
    if hasattr(args.model, 'init_path'):
        print("load model from", args.model.init_path)
        save_dict = torch.load(os.path.join(args.model.init_path), map_location=device)
        msg = model.load_state_dict(save_dict['state_dict'], strict=True)
    model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    if args.start_epoch != 0:
        if hasattr(args.model, 'init_path') and args.model.init_path is not None:
            raise NotImplementedError
        save_dict = torch.load(os.path.join(args.save_dir, f"checkpoint_{args.start_epoch}.pth"), map_location=device)
        msg = model.load_state_dict(save_dict['state_dict'], strict=True)
        optimizer.load_state_dict(save_dict['optimizer'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    train_loss_meter = AverageMeter(name="training loss")
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    epoch = 0
    for epoch in global_progress:
        if epoch < args.start_epoch:
            epoch_dict = {"epoch": epoch, "loss": train_loss_meter.avg}
            global_progress.set_postfix(epoch_dict)
            continue
        
        model.train()
        train_loss_meter.reset()
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            if hasattr(args.model, 'init_path') and epoch < 2:
                pass
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            data_dict['loss'] = data_dict['loss'].mean()
            
            train_loss_meter.update(data_dict['loss'].item(), n=images1.shape[0])
            local_progress.set_postfix(data_dict)
        
        epoch_dict = {"epoch":epoch, "loss": train_loss_meter.avg}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
        if (epoch+1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, f"checkpoint_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, model_path)
    
    save_model(args, epoch, model)

def save_model(args, epoch, model):
    # Save final checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "checkpoint.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

    os.system(f"rm -f {os.path.join(args.save_dir, 'checkpoint_*.pth')}")


if __name__ == "__main__":
    args = get_args()

    if args.save_dir is None:
        curr_time = datetime.now().strftime('%m%d%H%M%S')
        args.save_dir = os.path.join(args.ckpt_dir, f"{args.name}_{curr_time}")
    else:
        curr_time = args.save_dir.split('_')[-1].strip('/')

    args.log_dir = os.path.join(args.log_dir, 'in-progress_'+curr_time+'_'+args.name)

    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    shutil.copy2(args.config_file, args.log_dir)

    if args.train.early_stop:
        print("Use training early stopping!")
        train_early_stop(device=args.device, args=args)
    else:
        train(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')

