
# 1. Introduction  
<br/>
<p align="center">

   ![Banner](https://github.com/user-attachments/assets/a8a7edbe-04e2-461c-bbdd-50c31ba80ff9)
  
<br/>

## 🎨 AI Palette  
”심장을 바쳐라”  
### 🔅 Members  

김보현|김성주|김한얼|윤남규|정수현|허민석
:-:|:-:|:-:|:-:|:-:|:-:
<img src="https://github.com/user-attachments/assets/afedd001-af1e-4526-8a26-7c349a257ac2" height="180"/>|<img src="https://github.com/user-attachments/assets/ded33cfe-53d2-4220-b609-c4e5f25db61f" height="180" width="120"/>|<img src="https://github.com/user-attachments/assets/c4f6ca39-0528-4fa2-8587-fdeceb4405b4" height="180" width="120"/>|<img src="https://github.com/user-attachments/assets/94a7ddff-1da8-460c-8bd0-98e12b29f53f" height="180" width="120"/>|<img src="https://github.com/user-attachments/assets/f357c358-4099-464f-9e4a-ace9340f4ea0" height="180" width="120"/>|<img src="https://github.com/user-attachments/assets/ded33cfe-53d2-4220-b609-c4e5f25db61f" height="180" width="120"/>
[Github](https://github.com/boyamie)|[Github](https://github.com/kimmaru)|[Github](https://github.com/Haneol-Kijm)|[Github](https://github.com/Namgyu-Youn)|[Github](https://github.com/suhyun6363)|[Github](https://github.com/minseokheo)


### 🔅 Contribution  
- `김보현` &nbsp; Model search, Dataset curation, Hyperparameter tuning, Induce efficient augmentation
- `김성주` &nbsp; Data feature analysis, Enhanced evaluation accuracy, Data augmentation, Hyperparameter tuning
- `김한얼` &nbsp; Pipeline construction, Code refactorization, Team schedule management, Team workload management, model search
- `윤남규` &nbsp; Data preprocessing management, Model search, Data augmentation  
- `정수현` &nbsp; EDA, data generate, model search, feature engineering
- `허민석` &nbsp; Data curation, Dataset generation, Model search, Code refactoriation

[image1]: ./_img/김한얼.png
[image2]: ./_img/김보현.png
[image3]: ./_img/김성주.jpg
[image4]: ./_img/윤남규.png
[image5]: ./_img/정수현.png
[image6]: ./_img/허민석.jpg

<br/>

# 2. Project Outline  

![Competition Info](https://github.com/user-attachments/assets/bad4743f-73d4-4b83-a2de-d863ef264aa3)


- Task : Image Classification
- Date : 2024.09.10 - 2024.09.26
- Description : Sketch Image를 입력 받아서, 어떤 대상을 묘사하는지 추측해 500개의 class로 분류했습니다.
- Image Resolution : Auto-augmetation, Label smoothing
- Train : Kaggle ImageNet-Sketch([ImageNet-Sketch](https://www.kaggle.com/datasets/wanghaohan/imagenetsketch)) + Upstage dataset
- Test : Kaggle ImaeNet-Sketch + Upstage dataset ( size : 15,000)

### 🏆 Final Score  

![final_score](https://github.com/user-attachments/assets/c7ed5fb2-56eb-452d-bf28-db9f11725562)


# 3. Solution
![process](https://github.com/user-attachments/assets/ba89917f-66de-46f5-bf99-861cd670691d)

### KEY POINT
- Open source library인 timm의 모델들을 위주로 학습을 진행했습니다. (About timm : https://github.com/huggingface/pytorch-image-models)
- Image 내부에 object의 갯수가 복수인 경우, 유사한 class가 있는 경우가 많아 이를 해결하는 것이 가장 중요했습니다.
- Firsthand pre-processing을 시도한 train dataset으로 학습을 진행한 경우, 약 7%의 성능 저하가 있었습니다. 이를 통해서 pure dataset은 generalization에 어렵다는 점을 배웠습니다.
- Mixup, cutmix 등의 적절한 augmentation 기법을 수행한 경우, 약 1.5%의 유의미한 성능을 관찰할 수 있었습니다.
- WanDB, Tmux을 활용하여 작업 시 효율성과 편의성을 높였습니다.

&nbsp; &nbsp; → 주요 논점을 해결하는 방법론을 제시하고 실험결과를 공유하며 토론을 반복했습니다   

[process]: ./_img/process.png
<br/>

### Checklist

1. Data Preprocessing
- Augmentation : Auto, Trival, Mixup, Cutmix
- Label smoothing
- Firsthand data preprocessing (**Data preprossing** branch에서 관련 내용을 확인할 수 있습니다.)

2. Used models
- CNN based models : ResNet-18, EfficentNet, ConvNeXt
- Transformer based models : DeiT, EVA-CLIP, SwinV2

3. Ensemble method
- Median : 성능이 준수한 모델들의 prediction value들의 평균값을 구해서, 이를 새로운 prediction value로 이용
- Voting : 성능이 준수한 모델들의 값이 일치한 경우를 prediction value로 이용
- Uniform soup (not applied) : 단일 모델에 대한 앙상블 기법입니다. 본 프로젝트에서는 다양한 모델들의 앙상블 기법만 시도되었습니다.

### Evaluation

| Method                    | Accuracy |
| ------------------------- | ------- |
| rViT + pre-trained        | 88.9    |
| SwinV2                    | 87.4    |
| CoatNet + pre-tratined    | 88.7    |
| DeiT + Voting ensemble    | 90.0    |
| DeiT Large + TTA Ensemble | 90.3    |

<br/>

# 4. Descriptons about main branch
1. config : 모델 학습에 필요한 설정 파일들을 관리합니다. 다양한 모델의 정보를 포함하고 있으며, 변경이 가능합니다.
2. dataset : Data loading 및 Data pre-processing 과정을 처리합니다. 
3. engine : Training process 관련 코드를 포함하고 있으며, training 및 tuning을 수행합니다.
4. utils : 프로젝트 내에서 이용되는 utility function들을 포함합니다.
5. **train.py** : Model training을 수행하는 main script file 이며, 전체 pipeline을 담당합니다.
<br/>

# 5. How to use
- 본 대회의 dataset은 외부로 유출이 금지되어 있기 때문에, External data을 이용해야 합니다.
- External dataset은 '[ImageNet-Sketch](https://www.kaggle.com/datasets/wanghaohan/imagenetsketch)'를 권장합니다
- 사용자는 전체 코드를 내려받은 후, 옵션을 지정하여 개별 라이브러리의 모델을 활용할 수 있습니다
`git clone https://github.com/Namgyu-Youn/Booscamp_AI_tech7_level1.git`
