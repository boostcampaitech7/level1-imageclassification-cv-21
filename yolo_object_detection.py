import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import csv
from ultralytics import YOLO

# YOLOv5 모델 로드
model = YOLO('yolov8s.pt')

def detect_objects_yolo(image_path, model, conf_threshold=0.5):
    # YOLOv8 모델로 객체 감지 수행
    results = model(image_path)
    
    # 결과에서 bounding box와 confidence score 추출
    detections = results[0].boxes.data.cpu().numpy()  # 첫 번째 결과에서 데이터 추출
    
    # 감지된 객체를 Pandas DataFrame으로 변환
    detection_df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

    # confidence score가 threshold 이상인 객체만 필터링
    filtered_detections = detection_df[detection_df['confidence'] > conf_threshold]
    
    return filtered_detections

def crop_objects_yolo(image_path, detections):
    # 이미지 로드 (OpenCV를 사용하여 이미지 RGB로 변환)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cropped_images = []
    for idx, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cropped = img_rgb[y1:y2, x1:x2]
        cropped_images.append(cropped)

    return cropped_images

def save_cropped_images(cropped_images, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(cropped_images):
        output_path = os.path.join(output_dir, f'{image_name}_object_{i}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def log_detection_info(csv_file, class_name, image_name, num_objects):
    # CSV 파일에 감지 결과를 기록
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_name, image_name, num_objects])

def process_dataset(dataset_path, output_path, model, csv_file):
    # /data/ephemeral/home/data 하위의 각 클래스 폴더를 순회
    dataset_dir = Path(dataset_path)
    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]

    for class_dir in class_dirs:
        print(f"Processing class: {class_dir.name}")
        
        # 각 클래스 폴더 내의 모든 이미지 파일 순회
        image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        for image_file in image_files:
            # 파일 이름이 '._'로 시작하는 경우 무시
            if image_file.name.startswith("._"):
                continue

            print(f"Processing image: {image_file}")

            # 객체 감지
            detections = detect_objects_yolo(str(image_file), model)
            num_objects = len(detections)

            # 객체 크롭
            cropped_images = crop_objects_yolo(str(image_file), detections)

            # 크롭된 이미지 저장 경로 설정 (클래스별로 저장)
            class_output_dir = os.path.join(output_path, class_dir.name)
            save_cropped_images(cropped_images, class_output_dir, image_file.stem)

            # 감지 결과를 CSV 파일에 기록
            log_detection_info(csv_file, class_dir.name, image_file.name, num_objects)

def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="YOLOv5 Object Detection for Dataset")
    parser.add_argument('--dataset', required=True, help="Path to the dataset folder (e.g., /data/ephemeral/home/data/train/)")
    parser.add_argument('--output', required=True, help="Directory to save cropped images (e.g., /data/ephemeral/home/data/output/)")
    parser.add_argument('--csv', required=True, help="Path to the CSV file to log detection results")
    args = parser.parse_args()

    dataset_path = args.dataset
    output_path = args.output
    csv_file = args.csv

    # CSV 파일 경로의 상위 디렉토리 생성
    csv_dir = os.path.dirname(csv_file)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # CSV 파일의 헤더 작성
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class_Name', 'Image', 'Num_Objects'])

    # 데이터셋 전체에 대해 처리
    process_dataset(dataset_path, output_path, model, csv_file)

if __name__ == "__main__":
    main()
