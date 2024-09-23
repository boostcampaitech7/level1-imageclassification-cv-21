# 필요한 패키지 임포트
from typing import Optional

import torch
import torch.nn as nn
import timm
from timm.utils import freeze

class ViT(nn.Module):
    """
    추가 레이어나 사용자 정의 구현이 가능한 Vision Transformer (ViT) 모델.

    인자:
        num_classes (int, optional): 출력 클래스 수. 기본값은 None이다.
        pretrained (bool, optional): 사전 학습된 가중치를 사용할지 여부. 기본값은 False이다.
    """

    def __init__(self, num_classes: Optional[int] = 500, pretrained: bool = True, **kwargs):
        super(ViT, self).__init__()

        # 사전 학습된 ViT 모델 로드
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, **kwargs)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, 500)

        # 'head'를 제외한 서브모듈 얼리기
        submodules = [n for n, _ in self.model.named_children()]
        freeze(self.model, submodules[:submodules.index('head')])
        print(f"Non-head requires grad?: {self.model.blocks[0].attn.qkv.weight.requires_grad}")
        print(f"Non-head requires grad?: {self.model.head.weight.requires_grad}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ViT 모델을 통한 포워드 패스.

        인자:
            x (torch.Tensor): 입력 텐서.

        반환:
            torch.Tensor: ViT 모델의 출력.
        """
        return self.model(x)