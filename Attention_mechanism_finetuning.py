def enhance_attention_finetune(self):
        """Attention 메커니즘을 강화하고 fine-tuning을 위해 설정"""
    for name, param in self.model.named_parameters():
        if 'attn' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # Cross-Attention 층의 파라미터 학습 활성화
    for param in self.cross_attention.parameters():
        param.requires_grad = True
        
    # 헤드 또는 마지막 분류 레이어 학습 활성화
    self.model.head.weight.requires_grad = True
    self.model.head.bias.requires_grad = True
        
    # Position embedding 학습 활성화
    self.model.pos_embed.requires_grad = True
    print("Position embedding is set to True")
    
    # CLS Token 학습 활성화
    self.model.cls_token.requires_grad = True
    print("CLS token requires_grad is set to True")