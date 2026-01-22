import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    pkl_path = 'models/failure100.pkl'
    
    # 1. 데이터 로드
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    print(f"Loading {pkl_path}...")
    df = pd.read_pickle(pkl_path)
    
    # 2. 예시로 보여줄 불량 유형 선택 (예: 'LOC')
    target_type = 'LOC'
    print(f"Selecting a sample for failure type: '{target_type}'")
    
    # 해당 유형의 데이터 필터링
    sample_df = df[df['failureType'] == target_type]
    
    if len(sample_df) == 0:
        print(f"No samples found for {target_type}. Using the first available sample instead.")
        sample_row = df.iloc[0]
    else:
        sample_row = sample_df.iloc[0] # 첫 번째 샘플 선택
    
    # 데이터 추출
    wafer_map = sample_row['waferMap']
    failure_type = sample_row['failureType']
    die_size = sample_row['dieSize']
    lot_name = sample_row['lotName']
    
    # 3. 정보 출력
    print("\n" + "="*50)
    print(" [ Selected Wafer Sample Info ]")
    print("="*50)
    print(f" Failure Type : {failure_type}")
    print(f" Die Size     : {die_size}")
    print(f" Lot Name     : {lot_name}")
    print(f" Map Shape    : {wafer_map.shape}")
    print("="*50)
    
    # 4. 데이터 구조 확인 (시각화 대신 Raw Data 출력)
    print("\n[ Raw Data Structure Analysis ]")
    print(f"Data Type: {type(wafer_map)}")
    if hasattr(wafer_map, 'dtype'):
        print(f"Numpy Dtype: {wafer_map.dtype}")
    
    # 고유값 확인 (어떤 값으로 구성되어 있는지)
    unique_vals = np.unique(wafer_map)
    print(f"Unique Values in Map: {unique_vals}")
    print("-" * 30)
    
    # 데이터 일부 출력 (전체는 너무 클 수 있으므로)
    print("Raw Data Content (Center 10x10 slice):")
    cy, cx = wafer_map.shape[0] // 2, wafer_map.shape[1] // 2
    slice_size = 5
    # 중앙 부분 슬라이싱 (범위 체크)
    sy = max(0, cy - slice_size)
    ey = min(wafer_map.shape[0], cy + slice_size)
    sx = max(0, cx - slice_size)
    ex = min(wafer_map.shape[1], cx + slice_size)
    
    subset = wafer_map[sy:ey, sx:ex]
    print(subset)
    
    print("-" * 30)
    print("Full Matrix Representation (Text):")
    # 전체 행렬을 보고 싶다면 아래와 같이 출력 (너무 크면 생략될 수 있음)
    import sys
    np.set_printoptions(threshold=sys.maxsize, linewidth=150) # 전체 출력 설정
    print(wafer_map) # 전체 맵 출력

if __name__ == "__main__":
    main()
