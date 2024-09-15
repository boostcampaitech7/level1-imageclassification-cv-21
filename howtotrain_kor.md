# 시작하기
===============

이 프로젝트는 Python 환경에 특정 종속성을 설치해야 합니다. 다음 단계를 따라 환경을 설정하고 `train.py` 파일을 실행합니다.

## 1단계: 필요한 종속성 설치
먼저, pip를 사용하여 필요한 종속성을 설치합니다.

```pip install -U lightning "ray[data,train,tune,serve]" wandb```
이 명령어는 PyTorch Lightning, Ray 및 Weights & Biases 등 필요한 종속성을 설치합니다.

## 2단계: Weights & Biases 설정
Weights & Biases 계정을 설정하고 SDK를 설치합니다. Weights & Biases 설정 방법에 대한 자세한 내용은 [여기](https://docs.wandb.ai/ko/quickstart)를 참조하세요.

## 3단계: train.py 파일 실행
종속성이 설치되면 Python을 사용하여 `train.py` 파일을 실행할 수 있습니다.

### 기본 설정
인수를 지정하지 않으면 스크립트는 기본 설정을 사용합니다.

```python train.py```
이 명령어는 기본 설정으로 모델을 훈련합니다.

### 사용자 정의 설정
대신 인수를 지정하여 기본 설정을 재정의할 수도 있습니다:

```python train.py --model_name <모델_이름> --num_gpus <GPU_개수> --smoke_test```
- --model_name: 훈련할 모델의 이름을 지정합니다. 모델 정의를 확인하여 사용할 수 있는 모델 목록을 확인할 수 있습니다. 기본 모델은 ResNet18입니다.
- --num_gpus: 훈련에 사용할 GPU 개수를 지정합니다. 멀티-GPU 환경에서 훈련할 때 사용합니다. 기본: 1
- --smoke_test: (선택 사항) 훈련 스크립트가 올바르게 작동하는지 확인하기 위해 빠른 스모크 테스트를 실행하려면 이 플래그를 추가합니다. 스모크 테스트는 작은 배치 크기와 제한된 에포크 수로 훈련 스크립트를 실행합니다.

예시:
```python train.py --model_name resnet50 --num_gpus 2```
이 명령어는 ResNet-50 모델을 2개의 GPU로 훈련합니다.

`train.py` 파일이 있는 디렉토리에서 이 명령어를 실행하십시오.

## 문제 해결
설치 또는 `train.py` 파일 실행 중 문제가 발생하면 다음을 확인하세요.

- 최신 버전의 conda가 설치되어 있는지 확인합니다.
- requirements.yaml 파일이 올바른 위치에 있는지 확인합니다.
- conda 환경이 올바르게 활성화되었는지 확인합니다.

## 기여
이 프로젝트에 기여하고 싶다면 저장소를 포크하고 풀 요청을 제출합니다.

## 문의
질문이나 프로젝트에 대한 도움이 필요하면 [연락처 정보를 입력하세요]을 통해 연락해 주세요.