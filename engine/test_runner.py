from lightning import Trainer

from .callbacks import PredictionCallback
from dataset import get_genuine_valid_loader, get_test_loader
from model import LightningModule


def run_test(config):
    """
    모델 테스팅을 수행합니다.

    Args:
        config: 모델 및 실험 설정이 포함된 configuration 객체
        ckpt_dir: 체크포인트 디렉토리
    """
    ckpt_dir = config.experiment.checkpoint_path
    # 검증 데이터 로더(진) 생성
    valid_loader = get_genuine_valid_loader(config, batch_size=64)
    # 테스팅 데이터 로더 생성
    test_loader = get_test_loader(config)

    # 테스팅을 위한 PyTorch Lightning Trainer 및 콜백 정의
    pred_callback = PredictionCallback(
        f"{config.dataset.data_path}/test.csv", ckpt_dir, config.model.model_name
    )
    trainer_test = Trainer(
        devices=1,
        callbacks=[pred_callback],
        logger=False,
        enable_progress_bar=True,
    )

    # 체크포인트에서 모델 로드
    best_model = LightningModule.load_from_checkpoint(
        f"{ckpt_dir}/checkpoint.ckpt", config=config.model
    )

    trainer_test.validate(best_model, dataloaders=valid_loader)

    # 로드된 모델 테스팅 수행
    trainer_test.test(best_model, dataloaders=test_loader)
