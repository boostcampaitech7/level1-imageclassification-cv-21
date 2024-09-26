import torch
import torch.nn as nn
from PIL import Image
import timm

class EVACLIP(nn.Module):
    """
    EVA-CLIP-18B 기반 스케치 이미지 분류 모델.

    인자:
        num_classes (int): 출력 클래스 수. 기본값은 100.
    """

    def __init__(self, num_classes: int = 500, pretrained: bool = True, drop_path_rate=0.0, **kwargs):
        super(EVACLIP, self).__init__()
        self.num_classes = num_classes
        self.drop_path_rate = drop_path_rate
        self.cross_attention = nn.MultiheadAttention(embed_dim=1000, num_heads=8)

        # EVA-CLIP-18B 모델 불러오기
        self.model = timm.create_model("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", pretrained=pretrained)

        # CLIP 모델의 이미지 인코더만 사용
        self.model = self.model.eval()

        # 분류 레이어 추가 (CLIP의 이미지 출력 크기에 맞게 조정)
        self.fc = nn.Linear(1000, num_classes)
        self.set_attn_only_finetune()
        self.enhance_cross_attention_finetune()

        # 설정 확인
        for name, param in self.model.named_parameters():
           print(f"{name}: requires_grad = {param.requires_grad}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        EVA-CLIP-18B 모델을 통한 포워드 패스.

        인자:
            x (torch.Tensor): 입력 텐서 (이미지).

        반환:
            torch.Tensor: 분류 결과.
        """
        # 이미지 인코더로 특징 추출
        image_features = self.model(x)

        # 분류 레이어 통과
        logits = self.fc(image_features)
        return logits
    
    def set_attn_only_finetune(self):
        """
        모델의 attention 파라미터만 학습하도록 설정.

        인자:
            model (nn.Module): 입력 모델
        """
        # 전체 파라미터 학습 비활성화
        for name_p, p in self.model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        # 헤드 또는 마지막 분류 레이어 학습 활성화
        try:
            self.model.head.weight.requires_grad = True
            self.model.head.bias.requires_grad = True
        except AttributeError:
            try:
                self.model.fc.weight.requires_grad = True
                self.model.fc.bias.requires_grad = True
            except AttributeError:
                print("No head or fc layer found.")
        
        # Position embedding 학습 활성화
        try:
            self.model.pos_embed.requires_grad = True
        except AttributeError:
            print('No position embedding found.')
        
        # Patch embedding 학습 비활성화
        try:
            for p in self.model.patch_embed.parameters():
                p.requires_grad = False
        except AttributeError:
            print('No patch embed found.')

        # CLS Token 학습 활성화
        try:
            self.model.cls_token.requires_grad = True
            print("CLS token requires_grad is set to True")
        except AttributeError:
            print("No CLS token found.")
    
    def enhance_cross_attention_finetune(self):
        """
            Cross-Attention 메커니즘을 강화하고 fine-tuning을 위해 설정
        """ 
        # Cross-Attention 층의 파라미터 학습 활성화
        for param in self.cross_attention.parameters():
            param.requires_grad = True    