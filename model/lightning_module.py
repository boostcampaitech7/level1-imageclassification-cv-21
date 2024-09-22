# 필요한 import 수행
import lightning as pl
import torch

from timm.scheduler import create_scheduler_v2
from timm.scheduler.cosine_lr import CosineLRScheduler

from .model_factory import create_model
from config import ModelConfig



# 라이트닝 모듈 정의
class LightningModule(pl.LightningModule):
    def __init__(self, hparams, config: ModelConfig = None):
        """
        라이트닝 모듈 초기화.

        Args:
            hparams (dict): 모델 하이퍼 파라미터.
        """
        super().__init__()

        model_hparams = vars(config) if config else {}
        hparams = {**hparams, **model_hparams}
        self.save_hyperparameters(hparams)
        self.model = create_model(**model_hparams)

    def forward(self, x):
        """
        모델의 순전파 정의.

        Args:
            x (torch.Tensor): 입력 텐서.

        Returns:
            torch.Tensor: 출력 텐서.
        """
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        """
        모델의 훈련 스텝 정의.

        Args:
            train_batch (tuple): 입력, 출력 텐서 배치.
            batch_idx (int): 배치 인덱스.

        Returns:
            dict: 손실값을 포함하는 딕셔너리.
        """
        x, y = train_batch
        output = self.forward(x)
        loss = torch.nn.CrossEntropyLoss(LabelSmoothing=0.1)(output, y)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        모델의 검증 스텝 정의.

        Args:
            val_batch (tuple): 입력, 출력 텐서 배치.
            batch_idx (int): 배치 인덱스.

        Returns:
            None
        """
        x, y = val_batch
        output = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == y).sum().item() / len(x)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        모델의 예측 스텝 정의.

        Args:
            test_batch (tuple): 입력 텐서 배치.
            batch_idx (int): 배치 인덱스.

        Returns:
            list: 예측된 클래스 인덱스 목록.
        """
        x = test_batch
        output = self.forward(x)
        _, predicted = torch.max(output, 1)
        return predicted

    def configure_optimizers(self):
        """
        모델의 최적화 함수 정의.

        Returns:
            torch.optim.Adam: 모델의 Adam 최적화 함수.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # if self.hparams.sched=='cosine':
        #     lr_scheduler = CosineLRScheduler(optimizer, t_initial=20, warmup_t=5, warmup_lr_init=1e-6)
        # elif self.hparams.sched=='step':
        #     lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #         optimizer, step_size=self.trainer.estimated_stepping_batches * 2, gamma=0.1
        #     )
        
        
        lr_scheduler, _ = create_scheduler_v2(
            optimizer, 
            sched=self.hparams.sched, 
            num_epochs=self.trainer.max_epochs, 
            warmup_epochs=self.hparams.warmup_epochs, 
            warmup_lr=self.hparams.warmup_lr,
            )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, lr_scheduler, metric):
        lr_scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value