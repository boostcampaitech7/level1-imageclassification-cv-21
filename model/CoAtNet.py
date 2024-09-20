# model/CoAtNet.py
from typing import Optional

import torch
import torch.nn as nn
import timm

class CoAtNet(nn.Module):
    """
    CoAtNet model with optional additional layers or customizations.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to None.
        pretrained (bool, optional): Use pre-trained weights. Defaults to True.
    """

    def __init__(self, num_classes: Optional[int] = 500, pretrained: bool = False, **kwargs):
        super(CoAtNet, self).__init__()

        # Load pre-trained CoAtNet model
        self.model = timm.create_model('coatnet_bn_0_rw_224.sw_in1k', pretrained=pretrained, **kwargs)
        print(self.model.head)
        print(dir(self.model))

        # If num_classes is provided, replace the last layer
        if num_classes:
            self.model.head = nn.Linear(64, num_classes)  # Change to correct num_features for CoAtNet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CoAtNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the CoAtNet model.
        """
        return self.model(x)
