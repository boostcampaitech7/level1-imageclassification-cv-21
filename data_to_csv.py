import os
import argparse
from pathlib import Path
import csv

def create_class_to_index(dataset_path):
    paxheader_path = Path(dataset_path) / 'PaxHeader'
    class_to_index = {}
    
    if paxheader_path.exists() and paxheader_path.is_dir():
        # PaxHeader 폴더 내의 파일 이름을 클래스 이름으로 사용
        class_names = sorted([f.name for f in paxheader_path.iterdir() if f.is_file()])
        class_to_index = {name.lower(): idx for idx, name in enumerate(class_names)}
    else:
        print(f"Warning: PaxHeader directory not found at {paxheader_path}")
        # PaxHeader가 없는 경우 기존 방식으로 폴백
        dataset_dir = Path(dataset_path)
        class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name != 'PaxHeader'])
        class_to_index = {cls.name.lower(): idx for idx, cls in enumerate(class_dirs)}
    
    return class_to_index

def process_dataset(dataset_path, csv_file):
    dataset_dir = Path(dataset_path)
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name != 'PaxHeader'])
    
    # PaxHeader를 이용하여 class_to_index 생성
    class_to_index = create_class_to_index(dataset_path)
    
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class_Name', 'Image', 'Target'])  # CSV 헤더 작성
        for class_dir in class_dirs:
            # 각 클래스 폴더 내의 모든 이미지 파일 순회
            image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            for image_file in image_files:
                # 파일 이름이 '._'로 시작하는 경우 무시
                if image_file.name.startswith("._"):
                    continue
                
                # 클래스 이름을 소문자로 변환하여 매칭
                class_index = class_to_index.get(class_dir.name.lower(), -1)
                
                # 이미지 경로를 'class_name/image_name' 형식으로 생성
                image_path = f"{class_dir.name}/{image_file.name}"
                
                # 이미지 정보를 CSV 파일에 기록
                writer.writerow([class_dir.name, image_path, class_index])
            print([class_dir.name, image_path, class_index])                

        # 매칭되지 않은 클래스 출력
        unmatched_classes = set(class_dir.name.lower() for class_dir in class_dirs) - set(class_to_index.keys())
        if unmatched_classes:
            print("Warning: The following classes were not found in PaxHeader:")
            for cls in unmatched_classes:
                print(f"  - {cls}")

def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Image Information Logging with Class Index")
    parser.add_argument('--dataset', required=True, help="Path to the dataset folder (e.g., /data/ephemeral/home/data/train/)")
    parser.add_argument('--csv', required=True, help="Path to the CSV file to log results")
    args = parser.parse_args()
    
    dataset_path = args.dataset
    csv_file = args.csv
    
    # CSV 파일 경로의 상위 디렉토리 생성
    csv_dir = os.path.dirname(csv_file)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    # 데이터셋 전체에 대해 처리
    process_dataset(dataset_path, csv_file)

if __name__ == "__main__":
    main()
