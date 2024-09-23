# 필요한 패키지 임포트
from typing import Optional

import torch
import torch.nn as nn
import timm

class EfficientNet(nn.Module):
    """
    추가 레이어나 사용자 정의 구현이 가능한 EfficientNet 모델.
    """
    def __init__(self, num_classes: Optional[int] = 500, pretrained: bool = False, **kwargs):
        super(EfficientNet, self).__init__()
        # 사전 학습된 EfficientNet 모델 로드
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        EfficientNet 모델을 통한 포워드 패스.
        """
        return self.model(x)
