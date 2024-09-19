from torchvision import transforms

def get_transforms(mode='basic'):
    common_transforms = [
        transforms.Resize((224, 224)),  # 이미지를 224x224 크기로 리사이즈
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ]
    
    if mode=='train':
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            final_transforms = [
                  transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                  transforms.RandomRotation(15),  # 최대 15도 회전
                  transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기 및 대비 조정
                  ] + common_transforms
    else:
        # 검증/테스트용 변환: 공통 변환만 적용
        final_transforms = common_transforms
    transform = transforms.Compose(final_transforms)
    return transform
