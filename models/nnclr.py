""" NNCLR Model """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
from .memory_bank import MemoryBankModule
from .ntx_ent_loss import NTXentLoss
from .backbones import projection_MLP, prediction_MLP


class NNCLR(nn.Module):
    """Implementation of the NNCLR[0] architecture
    Recommended loss: :py:class:`lightly.loss.ntx_ent_loss.NTXentLoss`
    Recommended module: :py:class:`lightly.models.modules.nn_memory_bank.NNmemoryBankModule`
    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548
    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim: 
            Dimension of the hidden layer of the projection head.
        pred_hidden_dim:
            Dimension of the hidden layer of the predicion head.
        out_dim:
            Dimension of the output (after the projection head).
        num_mlp_layers:
            Number of linear layers for MLP.
    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>> 
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 512,
                 proj_hidden_dim: int = 2048,
                 pred_hidden_dim: int = 4096,
                 out_dim: int = 256,
                 proj_layers: int = 2,
                 device: str='cuda'):

        super(NNCLR, self).__init__()

        self.backbone = backbone
        self.feature_dim = backbone.output_dim
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = projection_MLP(
            num_ftrs,
            proj_hidden_dim,
            out_dim,
            num_layers = proj_layers
        )
        
        self.prediction_mlp = prediction_MLP(
            out_dim,
            pred_hidden_dim,
            out_dim,
        )

        
        self.memory_bank = NNMemoryBankModule(size=4096)
        self.memory_bank.to(device)
        self.criterion = NTXentLoss()

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None):
        """Embeds and projects the input images.
        Extracts features with the backbone and applies the projection
        head to the output space. If both x0 and x1 are not None, both will be
        passed through the backbone and projection head. If x1 is None, only
        x0 will be forwarded.
        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).
        Returns:
            The output projection of x0 and (if x1 is not None) the output
            projection of x1. If return_features is True, the output for each x
            is a tuple (out, f) where f are the features before the projection
            head.
        Examples:
            >>> # single input, single output
            >>> out = model(x) 
            >>> 
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)
        """
        
        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_mlp(f0)
        p0 = self.prediction_mlp(z0)

        # forward pass of second input x1
        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_mlp(f1)
        p1 = self.prediction_mlp(z1)

        # return both outputs
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return {'loss': loss, "feature": (f0, f1)}

class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.
    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548
    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>>
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
    """
    def __init__(self, size: int = 2 ** 16):
        super(NNMemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank
        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it
        """

        output, bank = \
            super(NNMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = \
            torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = \
            torch.index_select(bank, dim=0, index=index_nearest_neighbours)

        return nearest_neighbours