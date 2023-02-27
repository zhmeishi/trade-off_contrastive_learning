from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from .barlowtwins import BarlowTwins
from .nnclr import NNCLR
from .moco import MoCo
from .moco_v3 import MoCo_ViT, MoCo_ResNet
import torchvision.models as torchvision_models
import torch
import torch.nn as nn
from .backbones import get_backbone, MoCoBase
from functools import partial
import clip
from .vits import vit_small, vit_base, vit_conv_small, vit_conv_base
import os


def get_model(args):    
    model_cfg = args.model
    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'barlowtwins':
        model = BarlowTwins(get_backbone(model_cfg.backbone), args.device, model_cfg.lambda_param)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'moco':
        model = MoCo(dim=model_cfg.moco_dim, K=model_cfg.moco_k, m=model_cfg.moco_m, T=model_cfg.moco_t, \
            arch=model_cfg.backbone, bn_splits=model_cfg.bn_splits, device=args.device)
    elif 'moco_v3' in model_cfg.name: # Use pretrain model only
        # https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
        # 'vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'
        arch = model_cfg.name.split(":")[1]
        linear_keyword = "head" # for vit only
        if 'vit' in arch:
            model = eval(f"{arch}()")
            save_dict = torch.load(os.path.join(args.eval_from), map_location="cpu")
            state_dict = save_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
            model.head = torch.nn.Identity()
            model.eval()
        else:
            raise NotImplementedError
    elif model_cfg.name == 'nnclr':
        model = NNCLR(get_backbone(model_cfg.backbone), proj_layers = model_cfg.proj_layers, device=args.device)
    elif 'clip' in model_cfg.name: # Use pretrain model only
        # https://github.com/openai/CLIP
        # clip: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
        load_model = model_cfg.name.split(":")[1]
        model, _ = clip.load(load_model, args.device)
        model.eval()
        print("model.visual.input_resolution", model.visual.input_resolution)
        return CLIP(model, castrate = "castrate" in model_cfg.name)
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model

def get_eval_model(args, backbone_only=False):
    if "clip" in args.model.name or "moco_v3" in args.model.name:
        print(args.model.name)
        return get_model(args) 
    
    if args.eval_from is None:
        print("Random init model!!")
        if backbone_only:
            print(args.model.backbone)
            return get_backbone(args.model.backbone) 
        else:
            print(args.model.name, ":", args.model.backbone)
            return get_model(args)

    print(args.eval_from)
    save_dict = torch.load(args.eval_from, map_location='cpu')
    if backbone_only:
        if args.model.name=="moco" and args.model.backbone == "resnet18":
            model = MoCoBase(feature_dim=args.model.moco_dim, arch=args.model.backbone, bn_splits=args.model.bn_splits)
        else:
            model = get_backbone(args.model.backbone) 
        try:
            if args.model.name=="moco":
                msg = model.load_state_dict({k.replace("module.", "")[len('encoder_q.'):]:v for k, v in save_dict['state_dict'].items() \
                    if k.startswith('encoder_q.') or k.startswith('module.encoder_q.')}, strict=True)
            else:
                msg = model.load_state_dict({k.replace("module.", "")[len('backbone.'):]:v for k, v in save_dict['state_dict'].items() \
                    if k.startswith('backbone.') or k.startswith('module.backbone.')}, strict=True)
        except:
            msg = model.load_state_dict(save_dict['state_dict'], strict=True)
    else:
        model = get_model(args=args)
        msg = model.load_state_dict({k.replace("module.", ""):v for k, v in save_dict['state_dict'].items()}, strict=False)
    return model


class CLIP(nn.Module):
    def __init__(self, model, castrate=False):
        super().__init__()
        self.model = model
        if castrate:
            self.output_dim = 1024
            self.model.visual.proj = None
        else:
            self.output_dim = 768
    
    def forward(self, x):
        return self.model.encode_image(x).float()
