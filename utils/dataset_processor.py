# dataset_processor.py

import os
import shutil
import pandas as pd
import re

def process_dataset(dataset_path, csv_path, output_folder, output_csv_path, sorted_csv_path):
    """
    데이터셋을 처리하여 전처리된 이미지 파일로 대체하고, CSV 파일을 생성합니다.
    이후 생성된 CSV 파일을 정렬하여 새로운 파일로 저장합니다.
    
    Args:
        dataset_path (str): 데이터셋의 경로 (클래스 폴더가 포함된 경로).
        csv_path (str): 기존 train.csv 파일 경로.
        output_folder (str): 전처리된 파일을 저장할 폴더 경로.
        output_csv_path (str): 생성할 CSV 파일 경로.
        sorted_csv_path (str): 정렬된 CSV 파일 경로.
    """
    # 기존 train.csv 파일에서 클래스 이름과 라벨을 불러옴
    train_df = pd.read_csv(csv_path)
    class_to_label = {row['class_name']: row['target'] for _, row in train_df.iterrows()}

    # 새로운 process 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"폴더가 생성되었습니다: {output_folder}")
    
    # 결과를 담을 리스트와 중복 방지를 위한 세트
    data = []
    added_images = set()

    # 클래스 폴더 순회
    for class_name in class_to_label.keys():
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder) or class_name == "PaxHeader":
            continue

        # 각 클래스 폴더의 파일 리스트
        files = os.listdir(class_folder)
        
        # 전처리된 파일들을 찾기 위한 딕셔너리
        processed_files = {}

        # 전처리된 파일들을 딕셔너리에 저장
        for file_name in files:
            if file_name.startswith("._") or not file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue
            base_name, ext = os.path.splitext(file_name)
            # 전처리된 파일인지 확인
            if "_ed" in base_name or "_fixed" in base_name:
                # "sketch_숫자" 형식의 original_base 추출
                original_base = '_'.join(base_name.split('_')[:2])
                if original_base not in processed_files:
                    processed_files[original_base] = []
                processed_files[original_base].append(file_name)

        # 새로운 클래스 폴더를 process 폴더 내에 생성
        new_class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(new_class_folder):
            os.makedirs(new_class_folder)

        # 각 파일을 확인하여 전처리된 파일로 대체하여 새로운 폴더에 복사
        for file_name in files:
            if file_name.startswith("._") or not file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue
            base_name, ext = os.path.splitext(file_name)
            
            # 원본 파일인 경우
            if base_name.startswith("sketch_") and not ("_ed" in base_name or "_fixed" in base_name):
                original_base = '_'.join(base_name.split('_')[:2])
                # 전처리된 파일이 존재하는 경우
                if original_base in processed_files:
                    for processed_file in processed_files[original_base]:
                        src_file = os.path.join(class_folder, processed_file)
                        dest_file = os.path.join(new_class_folder, processed_file)
                        if dest_file not in added_images:  # 중복 방지
                            shutil.copy(src_file, dest_file)
                            data.append([class_name, os.path.join(class_name, processed_file), class_to_label[class_name]])
                            added_images.add(dest_file)
                else:
                    # 전처리된 파일이 없는 경우, 원본 파일을 복사
                    src_file = os.path.join(class_folder, file_name)
                    dest_file = os.path.join(new_class_folder, file_name)
                    if dest_file not in added_images:  # 중복 방지
                        shutil.copy(src_file, dest_file)
                        data.append([class_name, os.path.join(class_name, file_name), class_to_label[class_name]])
                        added_images.add(dest_file)

    # 데이터프레임 생성 및 CSV로 저장
    df = pd.DataFrame(data, columns=["class_name", "image_path", "target"])
    df.to_csv(output_csv_path, index=False)
    print(f"CSV 파일이 생성되었습니다: {output_csv_path}")

    # CSV 파일 정렬
    df = pd.read_csv(output_csv_path)

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
    # 실제 데이터셋 경로와 기존 train.csv 파일 경로, 출력 폴더, CSV 파일 이름을 설정하세요.
    dataset_path = "/data/ephemeral/home/data/processed2"  # 데이터셋 경로 (예: "/home/user/dataset/processed2")
    csv_path = "/data/ephemeral/home/data/train.csv"  # 기존 train.csv 파일 경로 (예: "/home/user/dataset/train.csv")
    output_folder = "/data/ephemeral/home/data/output"  # 결과를 저장할 폴더 (예: "/home/user/dataset/processed2/process")
    output_csv_path = "/data/ephemeral/home/data/output.csv"  # 전처리된 결과를 저장할 CSV 파일 경로
    sorted_csv_path = "/data/ephemeral/home/data/sorted_output.csv"  # 정렬된 결과를 저장할 CSV 파일 경로
    
    # 데이터셋 처리 및 CSV 생성/정렬 함수 호출
    process_dataset(dataset_path, csv_path, output_folder, output_csv_path, sorted_csv_path)


