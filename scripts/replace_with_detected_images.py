import os
import shutil
import argparse

def replace_with_detected_images(train_dir, output_dir, processed_dir):
    # 이미지 파일 확장자 목록
    valid_image_extensions = {'.jpg', '.jpeg', '.png'}

    # processed 디렉토리가 존재하지 않으면 생성
    os.makedirs(processed_dir, exist_ok=True)

    # 클래스별 폴더를 순회
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        class_processed_path = os.path.join(processed_dir, class_name)

        # 클래스 폴더가 train과 output 모두에 존재할 경우에만 진행
        if os.path.isdir(class_train_path) and os.path.isdir(class_output_path):
            print(f"Processing class: {class_name}")

            # processed 디렉토리 내 클래스 폴더 생성
            os.makedirs(class_processed_path, exist_ok=True)

            # train 클래스 폴더의 이미지 파일 순회
            for train_image_file in os.listdir(class_train_path):
                train_image_path = os.path.join(class_train_path, train_image_file)
                train_image_name, train_image_extension = os.path.splitext(train_image_file)  # 이미지 이름 및 확장자 추출

                # 이미지 파일인지 확인 (확장자가 이미지 확장자 목록에 포함된 경우)
                if train_image_extension.lower() in valid_image_extensions:
                    # 감지된 객체 이미지 파일 확인
                    detected_images = [
                        output_image_file for output_image_file in os.listdir(class_output_path)
                        if output_image_file.startswith(train_image_name) and '_object_' in output_image_file
                    ]

                    if detected_images:
                        # 감지된 이미지를 processed 이미지 위치에 복사하고 확장자를 .JPEG로 변경
                        for output_image_file in detected_images:
                            output_image_path = os.path.join(class_output_path, output_image_file)
                            new_image_name = f"{os.path.splitext(output_image_file)[0]}.JPEG"
                            new_image_path = os.path.join(class_processed_path, new_image_name)

                            shutil.copy(output_image_path, new_image_path)
                            print(f"Copied detected image to: {new_image_path}")
                    else:
                        # 감지된 이미지가 없는 경우 원본 이미지를 그대로 복사하여 processed에 저장
                        new_image_path = os.path.join(class_processed_path, train_image_file)
                        shutil.copy(train_image_path, new_image_path)
                        print(f"Copied original image to: {new_image_path}")

def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Replace train images with detected images and create a processed dataset.")
    parser.add_argument('--train_dir', required=True, help="Path to the train dataset directory.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory with detected images.")
    parser.add_argument('--processed_dir', required=True, help="Path to the directory where processed images will be stored.")
    args = parser.parse_args()

    # 인자로 받은 경로 설정
    train_dir = args.train_dir
    output_dir = args.output_dir
    processed_dir = args.processed_dir

    # 함수 실행
    replace_with_detected_images(train_dir, output_dir, processed_dir)

if __name__ == "__main__":
    main()

