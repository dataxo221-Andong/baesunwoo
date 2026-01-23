import os
import glob
import numpy as np
import pandas as pd

def generate_dataset_pkl():
    import datetime
    
    # 설정
    input_dir = 'chipfin'
    output_dir = 'models'
    
    now_str = datetime.datetime.now().strftime("%y%m%d_%H%M")
    output_filename = f'chip_dataset_{now_str}.pkl'
    
    # 1. 경로 확인
    if not os.path.exists(input_dir):
        print(f"[Error] Input directory '{input_dir}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[Info] Created directory '{output_dir}'.")

    # 2. 파일 리스트 가져오기
    file_list = glob.glob(os.path.join(input_dir, "*.npy"))
    total_files = len(file_list)
    print(f"[Info] Found {total_files} chip files in '{input_dir}'.")
    
    if total_files == 0:
        print("[Info] No files to process.")
        return

    # 3. 데이터 로드 및 통합
    data_list = []
    print("[Info] Loading data and building dataset...")
    
    for idx, fpath in enumerate(file_list):
        try:
            # .npy 파일 로드 (딕셔너리)
            chip_data = np.load(fpath, allow_pickle=True).item()
            data_list.append(chip_data)
            
            # 진행 상황 표시 (100개 단위)
            if (idx + 1) % 100 == 0:
                print(f"  - Processed {idx + 1}/{total_files} files...")
                
        except Exception as e:
            print(f"[Warning] Failed to load {fpath}: {e}")

    # 4. DataFrame 변환
    print("[Info] Converting to Pandas DataFrame...")
    df = pd.DataFrame(data_list)
    
    # 5. Pickle 파일로 저장
    output_path = os.path.join(output_dir, output_filename)
    print(f"[Info] Saving dataset to '{output_path}'...")
    
    try:
        df.to_pickle(output_path)
        print(f"[Success] Dataset saved successfully!")
        print(f"  - Total Data Points: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - File Path: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"[Error] Failed to save pickle file: {e}")

if __name__ == "__main__":
    generate_dataset_pkl()
