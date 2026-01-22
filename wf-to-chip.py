import pandas as pd
import numpy as np
import cv2
import os

def process_single_wafer():
    # 설정
    input_path = 'models/ft_100.pkl'
    output_dir = 'chip1step'
    target_size = (32, 32) # 웨이퍼 리사이즈 크기
    chip_grid_size = (32, 32) # 각 칩 내부의 TSV 격자 크기 (새로 생성되는 행렬)

    # 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 데이터 로드
    print(f"Loading {input_path}...")
    try:
        df = pd.read_pickle(input_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 테스트용: 첫 번째 웨이퍼만 선택
    if len(df) == 0:
        print("Empty dataframe.")
        return

    # 첫 번째 행 가져오기
    row_idx = 0
    row = df.iloc[row_idx]
    
    # 웨이퍼 맵 가져오기
    original_map = row['waferMap']
    
    # [수정] 메타데이터 추출 (부모 웨이퍼 정보)
    # failureType의 경우, 이전에 확인한대로 리스트가 중첩되어 있을 수 있으므로 단순화 처리 필요할 수 있음
    # 여기서는 있는 그대로 저장하되, 나중에 읽을 때 주의
    failure_type = row['failureType'] 
    lot_name = row['lotName']
    die_size = row['dieSize']
    train_test_label = row['trianTestLabel']
    
    # Wafer ID 대신 Lot Name 사용
    print(f"Processing Lot: {lot_name} (Type: {failure_type})...")

    # 1. 32x32로 리사이즈 (Nearest Neighbor로 값 변질 방지)
    # 데이터는 0, 1, 2 정수이므로 float 변환 후 resize하고 다시 정수로 변환 필요할 수 있음
    # cv2.resize는 기본적으로 float 입력을 선호하나 정수도 처리 가능. 
    # 하지만 interpolation 결과의 안전성을 위해 float -> resize -> round/cast 권장
    resized_map = cv2.resize(original_map.astype('float32'), target_size, interpolation=cv2.INTER_NEAREST)
    resized_map = resized_map.astype(int)

    # 2. 각 픽셀(다이) 순회
    saved_count = 0
    rows, cols = resized_map.shape
    
    last_saved_file_path = None # 검증용

    for y in range(rows):
        for x in range(cols):
            die_status = resized_map[y, x]
            
            # 0(배경)은 버리고, 1(정상)과 2(불량)만 처리
            if die_status == 1 or die_status == 2:
                # 3. 칩 데이터 딕셔너리 생성 [원본 컬럼명 그대로 사용]
                chip_data = {
                    "tsv_matrix": np.zeros(chip_grid_size, dtype=int),
                    "lotName": lot_name,           # parent_lot_name -> lotName
                    "failureType": failure_type,   # parent_failure_type -> failureType
                    "dieSize": die_size,           # parent_die_size -> dieSize
                    "trianTestLabel": train_test_label, # parent_train_test_label -> trianTestLabel
                    "tsv_coordinate": (x, y),
                    "die_status": int(die_status) # 1 or 2
                }
                
                # 파일명 생성 규칙: [lotName]_[x좌표]_[y좌표]_[상태]
                file_name = f"{lot_name}_{x}_{y}_{die_status}.npy"
                file_path = os.path.join(output_dir, file_name)
                
                # 저장 (딕셔너리 객체 저장)
                np.save(file_path, chip_data)
                saved_count += 1
                last_saved_file_path = file_path

    print(f"[{lot_name}] Processing Done.")
    print(f"  - Original Shape: {original_map.shape}")
    print(f"  - Resized Shape : {resized_map.shape}")
    print(f"  - Chips Created : {saved_count} files in '{output_dir}/'")

if __name__ == "__main__":
    process_single_wafer()
