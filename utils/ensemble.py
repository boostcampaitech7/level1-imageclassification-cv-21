import os
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from lightning import LightningModule, Trainer

from dataset import get_test_loader

class EnsemblePredictor:
    def __init__(self, config):
        """
        Args:
            config: 모델 및 실험 설정이 포함된 configuration 객체
            ckpt_dir (str): 체크포인트 디렉토리
            device (str): 연산에 사용할 디바이스
        """
        self.config = config
        self.method = config.experiment.ensemble_method
        self.ckpt_dir = config.experiment.ensemble_path
        self.test_loader = get_test_loader(config)

    def load_models(self):
        # 체크포인트에서 모델 로드
        models = []
        for ckpt_file in os.listdir(self.ckpt_dir):
            if ckpt_file.endswith(".ckpt"):
                model = LightningModule.load_from_checkpoint(os.path.join(self.ckpt_dir, ckpt_file), config=self.config.model)
                models.append(model)
        return models

# '''
#     def get_train_valid_loader(self):
#         train_data = pd.read_csv(self.config.dataset.data_path + "/train.csv")
#         train_input = train_data.drop(self.config.dataset.target_name, axis=1)
#         train_target = train_data[self.config.dataset.target_name]

#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         valid_loaders = []

#         for _, valid_index in skf.split(train_input):
#             valid_input_fold = train_input.iloc[valid_index]
#             valid_target_fold = train_target.iloc[valid_index]

#             # 데이터 로더 생성
#             valid_dataset = MyDataset(valid_input_fold, valid_target_fold)
            
#             valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
            
#             valid_loaders.append(valid_loader)

#         return valid_loaders

#     def ensemble_validate_kfold(self, models, valid_loaders):
#         # Ensemble Validate: Perform k-fold validation for ensemble model
#         accuracies = []
#         for i in range(len(valid_loaders)):
#             valid_loader = valid_loaders[i]
#             targets = np.array([y for _, y in valid_loader.dataset])
#             predictions = ensemble_predict(models, valid_loader)
#             accuracy = self.evaluate(predictions, targets)
#             accuracies.append(accuracy)

#         return np.mean(accuracies)
# '''

    # 앙상블 예측
    def ensemble_predict(models, dataloader):
        # 예측값을 저장할 배열을 초기화 합니다.
        predictions = []
        for model in models:
            trainer = Trainer(devices=1)
            predictions.append(trainer.predict(model, dataloaders=dataloader))
        # 예측값을 합산하여 앙상블 합니다.
        prediction_output = np.array(predictions).mean(axis=0)
        print(f"prediction output shape is {prediction_output.shape}")
        # 최종 예측값을 반환합니다.
        return prediction_output

# '''
#     def uniform_soup(self, predictions):
#         # Uniform Soup: Combine predictions by taking the average of all models' predictions
#         return uniform_soup_model

#     def greedy_soup(self, predictions, targets):
#         # Greedy Soup: Combine predictions by iteratively adding models that improve validation performance
#         ensemble = []
#         best_accuracy = 0.0
#         for i in range(predictions.shape[0]):
#             new_ensemble = ensemble + [predictions[i]]
#             accuracy = self.evaluate(np.mean(new_ensemble, axis=0), targets)
#             if accuracy > best_accuracy:
#                 ensemble = new_ensemble
#                 best_accuracy = accuracy
#         return np.mean(ensemble, axis=0)
#         return greedy_soup_model
# '''

    # Ensemble predictions using the specified method
    def ensemble(self):
        models = self.load_models()
        if self.method == 'uniform_soup':
            model = self.uniform_soup(models)
        elif self.method == 'greedy_soup':
            model = self.greedy_soup(models)
        elif self.method == 'ensemble_predict':
            pass
        else:
            raise ValueError("Invalid ensemble method")

# '''
#         검증 정확도 출력(아직 미구현)
#         predictions = []
#         for model in models:
#             predictions_model_fold = []
#             for i in range(len(self.get_train_valid_loader())):
#                 valid_loader = self.get_train_valid_loader()[i]
#                 predictions_model_fold.append(self.predict(model, valid_loader))
#             predictions.append(predictions_model_fold)
#         predictions = np.array(predictions)
# '''

        if self.method == 'ensemble_predict':
            predictions = self.ensemble_predict(models, self.test_loader)
        else:
            predictions = model.predict(self.test_loader)
        
        self.save_to_csv(predictions=predictions)

# '''
#     def evaluate(self, predictions, targets):
#         # Evaluate the accuracy of the predictions
#         return np.mean(np.argmax(predictions, axis=1) == targets)
# '''

    def save_to_csv(self, predictions):
        # 예측 결과를 csv 파일로 저장
        test_info = pd.read_csv(self.config.dataset.data_path + "/test.csv")
        test_info["target"] = np.argmax(predictions, axis=1)
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        file_name = os.path.join(
            self.ckpt_dir, f"ensemble_predictions_{current_time}.csv"
        )
        test_info.to_csv(file_name, index=False, lineterminator="\n")
        print(f"Output csv file successfully saved in {file_name}!!")

    # Run the ensemble predictor using the specified method
    def run(self):
        self.ensemble()
