import timm
import torch.nn as nn

class CoatNetModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super(CoatNetModel, self).__init__()
        # timm 라이브러리에서 CoAtNet 불러오기
        self.model = timm.create_model('coatnet_0', pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
    
        # 예측 성능을 평가하는 메트릭 계산
        acc = (logits.argmax(dim=-1) == y).float().mean()
    
        # 메트릭 리포트
        self.log("val_loss", loss)
        self.log("val_acc", acc)
    
        # Ray Tune을 위한 리포트
        tune.report(val_loss=loss.item(), val_acc=acc.item())

