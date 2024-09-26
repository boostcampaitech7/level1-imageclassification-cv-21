from typing import Optional
import torch
import torch.nn as nn
import timm

class rViT(nn.Module):
    """
    Vision Transformer(ViT) 모델 - 'vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k'.
    이미지 분류 작업에 사용.
    
    인자:
        num_classes (int, optional): 출력 클래스 수. 기본값은 1000이다.
        pretrained (bool, optional): 사전 학습된 가중치를 사용할지 여부. 기본값은 False이다.
    """

    def __init__(self, num_classes: Optional[int] = 500, pretrained: bool = False, **kwargs):
        super(rViT, self).__init__()

        # 사전 학습된 ViT 모델 로드
        self.model = timm.create_model(
            'vit_betwixt_patch16_reg4_gap_384', 
            pretrained=pretrained, 
            num_classes=num_classes, 
            **kwargs
        )
        # 사전 학습된 DeiT3 모델 로드
        self.model = timm.create_model('deit3_large_patch16_224.fb_in22k_ft_in1k', pretrained=pretrained, num_classes=num_classes, **kwargs)
        self.set_attn_only_finetune()

        # 설정 확인
        # for name, param in self.model.named_parameters():
        #    print(f"{name}: requires_grad = {param.requires_grad}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vision Transformer 모델을 통한 포워드 패스.
        인자:
            x (torch.Tensor): 입력 텐서.

        반환:
            torch.Tensor: ViT 모델의 출력.
        """
        return self.model(x)
    
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