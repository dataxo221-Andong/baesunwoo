import pandas as pd
import numpy as np
import os

def visualize_matrix(mat):
    """행렬 데이터를 시각화하여 출력"""
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        print(" [!] Not a 2D numpy array.")
        return
        
    h, w = mat.shape
    print(f"  - Shape: {mat.shape}")
    print("  - Visualization:")
    print("   " + "".join([str(i%10) for i in range(w)]))
    print("   " + "-" * w)
    for r in range(h):
        row_str = ""
        for c in range(w):
            val = mat[r, c]
            char = "■" if val > 0 else "."
            row_str += char
        print(f"{r:2d}|{row_str}|")
    print("   " + "-" * w)

def analyze_pkl_dataset():
    # 분석할 파일 경로 (사용자가 지정한 파일)
    target_path = 'models/chip_dataset_260123_1017.pkl'
    
    # 파일 존재 확인
    if not os.path.exists(target_path):
        print(f"[Error] File not found: {target_path}")
        # models 폴더 내의 다른 pkl 파일 추천
        if os.path.exists('models'):
            print("Accessible files in 'models/':")
            for f in os.listdir('models'):
                if f.endswith('.pkl'):
                    print(f" - {f}")
        return

    print(f"Loading dataset from {target_path}...")
    try:
        df = pd.read_pickle(target_path)
    except Exception as e:
        print(f"[Error] Failed to load pickle: {e}")
        return

    # 1. 기본 구조 분석
    print("\n" + "="*60)
    print(" [ 1. Dataset Overview ]")
    print("="*60)
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 60)
    df.info()

    # 2. 컬럼별 데이터 분석
    print("\n" + "="*60)
    print(" [ 2. Column Analysis ]")
    print("="*60)
    
    # failureType 분포
    if 'failureType' in df.columns:
        print("\n[failureType Distribution]")
        print(df['failureType'].value_counts())
    
    # die_status 분포
    if 'die_status' in df.columns:
        print("\n[die_status Distribution]")
        print(df['die_status'].value_counts())

    # lotName 분포 (상위 10개)
    if 'lotName' in df.columns:
        print("\n[Top 10 Lot Names]")
        print(df['lotName'].value_counts().head(10))

    # 3. 샘플 데이터 확인
    print("\n" + "="*60)
    print(" [ 3. Sample Data Inspection ]")
    print("="*60)
    
    if len(df) > 0:
        # 랜덤 샘플 1개 추출
        sample = df.sample(1).iloc[0]
        
        print(f"Sample Index: {sample.name}")
        for col in df.columns:
            val = sample[col]
            # 행렬 데이터는 시각화
            if col == 'tsv_matrix' or (isinstance(val, np.ndarray) and val.ndim == 2):
                print(f"[{col}]")
                visualize_matrix(val)
            else:
                print(f"[{col}]: {val}")
    else:
        print("Dataframe is empty.")

if __name__ == "__main__":
    analyze_pkl_dataset()
