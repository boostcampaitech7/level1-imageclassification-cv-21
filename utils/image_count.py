import os
import pandas as pd

def get_image_files_from_folder(folder_path):
    """
    폴더 내 모든 이미지 파일의 상대 경로 리스트를 반환합니다.
    
    Args:
        folder_path (str): 이미지 파일들이 저장된 상위 폴더 경로.
        
    Returns:
        list: 이미지 파일의 상대 경로 리스트.
    """
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                image_files.append(relative_path)
    return image_files

def find_mismatch_images(folder_path, csv_path, output_csv_path):
    """
    폴더와 CSV 파일 간의 이미지 파일 경로 차이를 찾아 출력하고, 불일치 정보를 CSV 파일로 저장합니다.
    
    Args:
        folder_path (str): 이미지 파일들이 저장된 상위 폴더 경로.
        csv_path (str): 이미지 파일 경로 정보를 담고 있는 CSV 파일 경로.
        output_csv_path (str): 불일치 정보를 저장할 CSV 파일 경로.
    """
    # 폴더 내 이미지 파일 목록
    folder_image_files = get_image_files_from_folder(folder_path)
    print(f"폴더 내 총 이미지 파일 수: {len(folder_image_files)}")
    
    # CSV 파일 내 이미지 파일 목록
    df = pd.read_csv(csv_path)
    csv_image_files = df['image_path'].tolist()
    print(f"CSV 파일 내 총 이미지 파일 수: {len(csv_image_files)}")
    
    # 폴더와 CSV의 차이점 비교
    folder_set = set(folder_image_files)
    csv_set = set(csv_image_files)
    
    # 경로 및 파일명 불일치 목록
    in_folder_not_in_csv = folder_set - csv_set
    in_csv_not_in_folder = csv_set - folder_set
    
    # 경로 및 파일명 불일치 세부 확인
    mismatch_info = []
    for file in in_folder_not_in_csv:
        folder_parts = file.split(os.sep)
        for csv_file in csv_set:
            csv_parts = csv_file.split('/')
            if len(folder_parts) == len(csv_parts):
                mismatched_parts = [(folder_part, csv_part) for folder_part, csv_part in zip(folder_parts, csv_parts) if folder_part != csv_part]
                if mismatched_parts:
                    mismatch_info.append({
                        'folder_image': file,
                        'csv_image': csv_file,
                        'mismatched_parts': mismatched_parts
                    })
    
    # 불일치 세부 정보 CSV로 저장
    if mismatch_info:
        print("\n경로 및 파일명 불일치 세부 정보를 CSV로 저장 중...")
        mismatch_df = pd.DataFrame(mismatch_info)
        mismatch_df.to_csv(output_csv_path, index=False)
        print(f"불일치 세부 정보가 {output_csv_path}에 저장되었습니다.")
    else:
        print("\n경로 및 파일명 불일치 없음.")

if __name__ == "__main__":
    # 실제 폴더 경로와 CSV 파일 경로를 설정하세요.
    output_folder_path = "/data/ephemeral/home/data/processed_final"  # output 폴더 경로
    csv_file_path = "/data/ephemeral/home/data/processed_final.csv"  # output.csv 파일 경로
    output_csv_path = "/data/ephemeral/home/data/mismatch_info.csv"  # 불일치 정보 저장할 CSV 경로
    
    # 경로 및 파일명 불일치 확인 함수 호출
    find_mismatch_images(output_folder_path, csv_file_path, output_csv_path)

