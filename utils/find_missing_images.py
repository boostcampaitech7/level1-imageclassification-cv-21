import os
import pandas as pd

def find_missing_images(train_csv_path, processed_folder_path, output_csv_path):
    """
    train.csv의 image_path를 참고하여 processed_final 폴더에서 누락된 이미지를 찾고, 
    각 클래스별로 몇 개의 이미지가 빠졌는지 계산하여 CSV 파일로 저장합니다.
    
    Args:
        train_csv_path (str): train.csv 파일 경로.
        processed_folder_path (str): 전처리된 이미지들이 저장된 상위 폴더 경로.
        output_csv_path (str): 결과를 저장할 CSV 파일 경로.
    """
    # train.csv 파일에서 image_path와 class_name을 불러옴
    train_df = pd.read_csv(train_csv_path)
    
    # 클래스별 이미지 개수를 세기 위한 딕셔너리
    class_image_count = train_df['class_name'].value_counts().to_dict()
    
    # 클래스별로 누락된 이미지 개수를 저장할 딕셔너리
    missing_images_count = {class_name: 0 for class_name in class_image_count.keys()}

    # 클래스별로 실제 이미지 파일 수를 세기 위한 딕셔너리
    processed_image_count = {class_name: 0 for class_name in class_image_count.keys()}
    
    # processed_final 폴더 내 클래스별로 이미지 파일 개수를 세고, 누락된 이미지 확인
    for class_name, count in class_image_count.items():
        class_folder = os.path.join(processed_folder_path, class_name)
        if not os.path.isdir(class_folder) or class_name == "PaxHeader":
            # 클래스 폴더 자체가 없거나 PaxHeader 폴더인 경우, 모든 이미지가 누락된 것으로 간주
            missing_images_count[class_name] = count
            continue

        # 해당 클래스 폴더 내 모든 이미지 파일 리스트
        processed_files = [
            f for f in os.listdir(class_folder)
            if not f.startswith("._") and f.lower().endswith(('.jpeg', '.jpg', '.png'))
        ]
        processed_image_count[class_name] = len(processed_files)

        # train.csv에 기록된 해당 클래스의 이미지 파일 리스트
        train_files = train_df[train_df['class_name'] == class_name]['image_path'].tolist()
        train_files = [os.path.basename(f) for f in train_files]  # 파일명만 추출

        # 누락된 이미지 파일 확인
        missing_files = [file for file in train_files if file not in processed_files]
        missing_images_count[class_name] = len(missing_files)
    
    # 결과를 데이터프레임으로 생성
    result_df = pd.DataFrame({
        'class_name': list(class_image_count.keys()),
        'expected_count': list(class_image_count.values()),
        'processed_count': [processed_image_count[class_name] for class_name in class_image_count.keys()],
        'missing_count': [missing_images_count[class_name] for class_name in class_image_count.keys()]
    })

    # 결과를 CSV로 저장
    result_df.to_csv(output_csv_path, index=False)
    print(f"클래스별 이미지 누락 정보가 {output_csv_path}에 저장되었습니다.")

if __name__ == "__main__":
    # 실제 경로 설정
    train_csv_path = "/data/ephemeral/home/data/train.csv"  # train.csv 파일 경로
    processed_folder_path = "/data/ephemeral/home/data/EDA_data"  # processed_final 폴더 경로
    output_csv_path = "/data/ephemeral/home/data/missing_images_report.csv"  # 결과를 저장할 CSV 파일 경로

    # 함수 호출
    find_missing_images(train_csv_path, processed_folder_path, output_csv_path)
