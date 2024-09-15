from typing import Optional
from torchvision import models
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    """
    ResNet18 model with optional additional layers or customizations.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to None.
        pretrained (bool, optional): Use pre-trained weights. Defaults to True.
    """

    def __init__(self, num_classes: Optional[int] = 500, pretrained: bool = False):
        super(ResNet18, self).__init__()

        # Load pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=pretrained)

        # If num_classes is provided, replace the last layer
        if num_classes:
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet18 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the ResNet18 model.
        """
        return self.model(x)
