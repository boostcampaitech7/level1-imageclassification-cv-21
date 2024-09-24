import os
import pandas as pd
import re

def create_processed_csv(train_csv_path, folder_path, output_csv_path, sorted_csv_path):
    """
    train.csv의 정보를 바탕으로 processed_final 폴더 내의 파일들을 processed_final.csv로 생성하고, 정렬합니다.
    
    Args:
        train_csv_path (str): train.csv 파일 경로 (class_name, target 정보 포함).
        folder_path (str): 이미지 파일들이 저장된 상위 폴더 경로.
        output_csv_path (str): 생성할 CSV 파일 경로.
        sorted_csv_path (str): 정렬된 결과를 저장할 CSV 파일 경로.
    """
    # train.csv 파일에서 class_name과 target 정보를 불러옴
    train_df = pd.read_csv(train_csv_path)
    class_to_target = {row['class_name']: row['target'] for _, row in train_df.iterrows()}
    
    # 결과를 담을 리스트
    data = []

    # 클래스별로 폴더 순회
    for class_name, target in class_to_target.items():
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        
        # 해당 클래스 폴더 내 모든 이미지 파일 리스트
        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(class_name, image_file)
            data.append([class_name, image_path, target])
    
    # 데이터프레임 생성 및 CSV로 저장
    df = pd.DataFrame(data, columns=["class_name", "image_path", "target"])
    df.to_csv(output_csv_path, index=False)
    print(f"CSV 파일이 생성되었습니다: {output_csv_path}")

    # CSV 파일 정렬
    # sketch_ 뒤에 있는 숫자 추출 함수
    def extract_sketch_number(image_path):
        match = re.search(r'sketch_(\d+)', image_path)
        return int(match.group(1)) if match else -1

    # image_path 기준으로 정렬을 위해 새로운 컬럼 생성
    df['sketch_number'] = df['image_path'].apply(extract_sketch_number)

    # target 기준으로 정렬하고, 그 다음 sketch_number 기준으로 정렬
    df_sorted = df.sort_values(by=['target', 'sketch_number'])

    # 불필요한 임시 컬럼 삭제
    df_sorted = df_sorted.drop(columns=['sketch_number'])

    # 정렬된 데이터를 새로운 CSV 파일로 저장
    df_sorted.to_csv(sorted_csv_path, index=False)
    print(f"정렬된 CSV 파일이 생성되었습니다: {sorted_csv_path}")

if __name__ == "__main__":
    # 실제 경로 설정
    train_csv_path = "/data/ephemeral/home/data/train.csv"  # train.csv 파일 경로
    folder_path = "/data/ephemeral/home/data/processed_final"  # processed_final 폴더 경로
    output_csv_path = "/data/ephemeral/home/data/processed_final.csv"  # 생성할 CSV 파일 경로
    sorted_csv_path = "/data/ephemeral/home/data/processed_final_sorted.csv"  # 정렬된 결과를 저장할 CSV 파일 경로

    # 함수 호출
    create_processed_csv(train_csv_path, folder_path, output_csv_path, sorted_csv_path)
