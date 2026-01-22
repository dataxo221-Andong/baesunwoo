import pandas as pd
import numpy as np
import os

def extract_matrices():
    input_path = 'models/ft_100.pkl'
    output_dir = 'matrix'
    
    # matrix 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    print(f"Loading {input_path}...")
    if not os.path.exists(input_path):
         print(f"Error: File {input_path} does not exist.")
         return

    try:
        df = pd.read_pickle(input_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    # 타겟 불량 유형 정의
    target_labels = ['CENTER', 'DONUT', 'EDGE-LOC', 'EDGE-RING', 'LOC', 'NEAR-FULL', 'RANDOM', 'SCRATCH', 'NONE']
    
    # [수정] failureType 컬럼 데이터 전처리 (리스트/배열 -> 문자열 변환)
    # ValueError: The truth value of an empty array is ambiguous 오류 해결
    if len(df) > 0:
        sample_val = df['failureType'].iloc[0]
        print(f"Sample failureType raw value: {sample_val} (Type: {type(sample_val)})")
        
        # 리스트나 배열 형태라면 문자열로 풀기
        def clean_label(x):
            if isinstance(x, (list, np.ndarray)):
                if len(x) > 0:
                    return clean_label(x[0]) # 재귀적으로 내부 값 찾기
                else:
                    return "" # 빈 리스트
            return str(x)
            
        df['failureType'] = df['failureType'].apply(clean_label).str.upper()
        print("Converted 'failureType' column to strings and UPPERCASE.")
        print(f"Sample after conversion: {df['failureType'].iloc[0]}")
        print(f"Unique values found: {df['failureType'].unique()}")

    total_saved = 0
    
    print("\nStarting extraction (10 samples per type)...")
    
    for label in target_labels:
        # 해당 불량 유형 데이터 필터링
        # 데이터의 failureType이 리스트인지 문자열인지 확인 필요하나, k-means01.py 수정본을 보면 문자열 처리됨.
        # 만약 원본이 중첩 리스트라면 처리가 필요할 수 있으나, ft_100.pkl은 전처리된 파일일 가능성이 높음.
        # 안전을 위해 문자열 변환이나 체크 로직 없이 pandas 필터링 시도, 안되면 이슈 발생.
        # 이전 문맥상 df['failureType']이 이미 문자열이거나 처리 가능한 상태임.
        
        subset = df[df['failureType'] == label]
        
        # 데이터가 없다면 대소문자 문제일 수 있으니 확인
        if len(subset) == 0:
            print(f"Warning: No samples found for label '{label}'. Checking raw values...")
            # 혹시 모르니 unique 값 출력 (디버깅용)
            # print(df['failureType'].unique())
            continue
            
        # 상위 10개 추출
        samples = subset.iloc[:10]
        
        print(f"[{label}] Found: {len(subset)} -> Extracting: {len(samples)}")
        
        for i, (index, row) in enumerate(samples.iterrows()):
            wafer_map = row['waferMap']
            
            # 파일명 생성: 유형_번호.npy (01 ~ 10)
            file_base = f"{label}_{i+1:02d}"
            npy_path = os.path.join(output_dir, f"{file_base}.npy")
            txt_path = os.path.join(output_dir, f"{file_base}.txt")
            
            # 1. Numpy 바이너리로 저장 (데이터 보존용)
            np.save(npy_path, wafer_map)
            
            # 2. 텍스트 파일로 저장 (눈으로 확인용 - 정수형 포맷)
            # 웨이퍼 맵은 보통 0, 1, 2 정수로 되어있으므로 %d 사용
            np.savetxt(txt_path, wafer_map, fmt='%d', delimiter=' ')
            
            total_saved += 1

    print("\n" + "="*50)
    print(f"Extraction Completed!")
    print(f"Total files created: {total_saved} sets (.npy + .txt) in '{output_dir}' folder.")
    print("="*50)

if __name__ == "__main__":
    extract_matrices()
