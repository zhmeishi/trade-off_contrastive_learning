import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from augmentations import get_aug
from models import get_eval_model
from tools import get_args, set_deterministic
from datasets import get_dataset, load_feature

def extract_save_feature(args, model, data_loader, split="train"):
    save_model_name = args.model.name.replace("/","-")
    save_folder = os.path.join(os.path.dirname(args.data_dir), f"{save_model_name}_{args.dataset.name}_feature", split)
    final_save_folder = os.path.join(os.path.dirname(args.data_dir), f"{save_model_name}_{args.dataset.name}_feature")
    final_save_path = os.path.join(final_save_folder, f"{save_model_name}_{args.dataset.name}_feature_{split}.pt")

    if not (os.path.exists(save_folder) or os.path.exists(final_save_path)):
        os.mkdir(save_folder)
        model.eval()
        local_progress = tqdm(data_loader, desc='Extract features', disable=False)
        for idx, (images, labels) in enumerate(local_progress):
            with torch.no_grad():
                if "aug" in split:
                    feature = model(images[0].to(args.device))
                else:
                    feature = model(images.to(args.device))
            
            save_path = os.path.join(save_folder, f"{save_model_name}_{args.dataset.name}_feature_{split}_{idx}.pt")
            torch.save({"features":feature.cpu(), "feature_labels":labels.cpu()}, save_path)
    

        feature_list = os.listdir(save_folder)
        features = []
        feature_labels = []
        for idx in range(len(feature_list)):
            load_path = os.path.join(save_folder, f"{save_model_name}_{args.dataset.name}_feature_{split}_{idx}.pt")
            dic = torch.load(load_path, map_location='cpu')
            features.append(dic["features"])
            feature_labels.append(dic["feature_labels"])
            
        features = torch.cat(features, dim=0)
        feature_labels = torch.cat(feature_labels, dim=0).long()

        
        torch.save({"features":features, "feature_labels":feature_labels}, final_save_path)
        print(features.shape)
        
        os.system(f"rm {save_folder} -r")
    return 

# return a tensor feature
def extract_feature(args, model, data_loader):
    model.eval()
    features = []
    feature_labels = []
    local_progress = tqdm(data_loader, desc='Extract features', disable=False)
    for idx, (images, labels) in enumerate(local_progress):
        with torch.no_grad():
            feature = model(images.to(args.device))
        features.append(feature)
        feature_labels.append(labels)
        
    features = torch.cat(features, dim=0).cpu()
    feature_labels = torch.cat(feature_labels, dim=0).cpu().long()

    print(features.shape)
    return features, feature_labels

# return a tensor feature
def get_feature(args, save=False):
    print(args.dataset.name)
    if "feature" in args.dataset.name:
        # if args.linear:
        #     train_features, train_feature_labels =  load_feature(args.data_dir, args.dataset.name, "train")
        # else:
        #     # idx = args.seed % 100
        train_features, train_feature_labels =  load_feature(args.data_dir, args.dataset.name, f"aug_0") 
        test_features, test_feature_labels =  load_feature(args.data_dir, args.dataset.name, "test")
        return torch.Tensor(train_features), torch.Tensor(train_feature_labels).long(), \
            torch.Tensor(test_features), torch.Tensor(test_feature_labels).long()
    model = get_eval_model(args, backbone_only=True)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform= get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=True, 
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform= get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )
    if save:
        save_model_name = args.model.name.replace("/","-")
        data_folder = os.path.join(os.path.dirname(args.data_dir), f"{save_model_name}_{args.dataset.name}_feature")
        os.makedirs(data_folder, exist_ok=True)
        extract_save_feature(args, model, train_loader, "train")
        extract_save_feature(args, model, test_loader, "test")
    else: # For Linear Probing Only
        train_features, train_feature_labels =  extract_feature(args, model, train_loader)
        test_features, test_feature_labels =  extract_feature(args, model, test_loader)
        return train_features, train_feature_labels, test_features, test_feature_labels

def get_aug_feature(args):
    print(args.dataset.name)
    model = get_eval_model(args, backbone_only=True)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform= get_aug(train=True, train_classifier=False, **args.aug_kwargs), 
            train=True, 
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )

    for epoch in range(args.start_epoch, args.end_epoch):
        print("Epoch:", epoch)
        set_deterministic(epoch)
        extract_save_feature(args, model, train_loader, f"aug_{epoch}")

if __name__ == "__main__":
    print("Extract Feature")
    args=get_args()
    if args.start_epoch == 0:
        get_feature(args=args, save=True)
    get_aug_feature(args)