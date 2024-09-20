import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class EVA_CLIP_SketchClassifier(nn.Module):
    """
    EVA-CLIP-18B 기반 스케치 이미지 분류 모델.

    인자:
        num_classes (int): 출력 클래스 수. 기본값은 100.
    """

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(EVA_CLIP_SketchClassifier, self).__init__()

        # EVA-CLIP-18B 모델 불러오기 (HuggingFace의 CLIPModel 사용)
        self.model = CLIPModel.from_pretrained("BAAI/EVA-CLIP-18B", use_auth_token=pretrained)

        # CLIP 모델의 이미지 인코더만 사용
        self.image_encoder = self.model.vision_model

        # 분류 레이어 추가 (CLIP의 이미지 출력 크기에 맞게 조정)
        self.fc = nn.Linear(self.image_encoder.config.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        EVA-CLIP-18B 모델을 통한 포워드 패스.

        인자:
            x (torch.Tensor): 입력 텐서 (이미지).

        반환:
            torch.Tensor: 분류 결과.
        """
        # 이미지 인코더로 특징 추출
        image_features = self.image_encoder(x).pooler_output

        # 분류 레이어 통과
        logits = self.fc(image_features)
        return logits

# 데이터 로딩 및 전처리 코드
def preprocess_image(image_path: str, processor: CLIPProcessor, device: torch.device) -> torch.Tensor:
    """
    이미지 전처리 함수. EVA-CLIP에 맞게 이미지를 처리하고 텐서로 변환.

    인자:
        image_path (str): 이미지 파일 경로.
        processor (CLIPProcessor): CLIPProcessor 객체.
        device (torch.device): 사용할 디바이스 (CPU 또는 GPU).

    반환:
        torch.Tensor: 전처리된 이미지 텐서.
    """
    # 이미지 로드 (PIL 형식으로)
    image = Image.open(image_path)

    # 이미지 전처리 (CLIPProcessor 사용)
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)


import pandas as pd
import torch
from tqdm import tqdm

# 데이터셋 로드와 전처리 함수는 이미 정의된 것으로 가정
# from dataset_module import preprocess_image, model

# 가정: 모델은 이미 학습된 상태이고, processor와 model 객체가 존재한다고 가정
# processor = CLIPProcessor.from_pretrained("BAAI/EVA-CLIP-18B")
# model = EVA_CLIP_SketchClassifier(num_classes=100).to(device)

# 테스트 데이터 로드
test_df = pd.read_csv("data/test.csv")  # 테스트 파일 경로

# 결과를 저장할 리스트
results = []

# 모델을 평가 모드로 설정
model.eval()

# 디바이스 설정 (GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 추론 과정
with torch.no_grad():
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = f"data/test/{row['image_path']}"  # 이미지 파일 경로
        
        # 이미지 전처리
        image_tensor = preprocess_image(image_path, processor, device)

        # 모델 예측
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()  # 예측된 클래스
        
        # 결과를 저장
        results.append({
            "ID": row["ID"],
            "image_path": row["image_path"],
            "target": predicted_class
        })

# 결과를 DataFrame으로 변환
submission_df = pd.DataFrame(results)

# CSV 파일로 저장
submission_df.to_csv("submission.csv", index=False)

# 사용 예시
if __name__ == "__main__":
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EVA-CLIP 전처리기 및 모델 초기화
    processor = CLIPProcessor.from_pretrained("BAAI/EVA-CLIP-18B")
    model = EVA_CLIP_SketchClassifier(num_classes=100).to(device)

    # 예시 이미지 로드 및 전처리
    image_path = "data/test/0.JPEG"
    image_tensor = preprocess_image(image_path, processor, device)

    # 모델 추론
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        print(output)