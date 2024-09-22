import os
import argparse
from pathlib import Path
import csv

def process_dataset(dataset_path, csv_file):
    # /data/ephemeral/home/data 하위의 각 클래스 폴더를 순회
    dataset_dir = Path(dataset_path)
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    
    # 클래스 이름과 인덱스 매핑 생성
    class_to_index = {cls.name: idx for idx, cls in enumerate(class_dirs)}

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class_Name', 'Image', 'Target'])  # CSV 헤더 작성

        for class_dir in class_dirs:
            print(f"Processing class: {class_dir.name}")
            
            # 각 클래스 폴더 내의 모든 이미지 파일 순회
            image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            for image_file in image_files:
                # 파일 이름이 '._'로 시작하는 경우 무시
                if image_file.name.startswith("._"):
                    continue

                print(f"Processing image: {image_file}")

                # 이미지 정보를 CSV 파일에 기록
                writer.writerow([class_dir.name, image_file.name, class_to_index[class_dir.name]])

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