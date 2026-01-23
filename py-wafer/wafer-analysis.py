import pandas as pd
import numpy as np
import os

def load_data(file_path):
    print(f"Loading {file_path}...")
    try:
        df = pd.read_pickle(file_path)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_failure_type(val):
    """
    failureType이 중첩 리스트/배열로 되어있는 경우 단일 문자열로 변환
    """
    temp = val
    # 중첩 구조 해제
    while isinstance(temp, (list, np.ndarray)):
        if len(temp) > 0:
            if isinstance(temp, list):
                temp = temp[0]
            else: # np.ndarray
                temp = temp.item() if temp.size == 1 else temp[0]
        else:
            return 'None'
    
    return str(temp).strip()

def main():
    pkl_path = 'models/ft_100.pkl'
    
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return

    df = load_data(pkl_path)
    if df is None:
        return

    # failureType 데이터 정제 (검색을 위해)
    print("Preprocessing failure types...")
    # 원본 데이터 보호를 위해 복사본이나 새로운 컬럼 사용 권장
    # 여기서는 검색용 임시 컬럼 생성
    df['clean_failureType'] = df['failureType'].apply(clean_failure_type)

    # 사용 가능한 Failure Types 출력
    unique_types = df['clean_failureType'].unique()
    print("\n[Available Failure Types]")
    print(", ".join(sorted(unique_types)))
    print("-" * 50)

    while True:
        target_type = input("\n찾고 싶은 Failure Type을 입력하세요 (종료: q): ").strip()
        
        if target_type.lower() == 'q':
            print("종료합니다.")
            break
            
        # 검색
        matched_indices = df[df['clean_failureType'] == target_type].index.tolist()
        
        if matched_indices:
            print(f"\nFound {len(matched_indices)} wafers with Failure Type '{target_type}':")
            print(f"Indices: {matched_indices}")
        else:
            print(f"\n'{target_type}'에 해당하는 웨이퍼를 찾을 수 없습니다. (대소문자 확인 필요)")

if __name__ == "__main__":
    main()
