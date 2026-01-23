import pandas as pd
import os
import numpy as np

# 경로 설정 (k-means01.py 참고)
pkl_path = 'models/failure100.pkl'

def review_failure_types():
    if not os.path.exists(pkl_path):
        print(f"[오류]: '{pkl_path}' 파일이 없습니다. 경로를 확인해주세요.")
        return

    print(f"[진행]: '{pkl_path}' 파일 로드 중...")
    try:
        df = pd.read_pickle(pkl_path)
        print(f"[완료]: 데이터 로드 성공 (총 {len(df)}건)")

        # k-means01.py의 전처리 로직 참고
        if 'waferIndex' in df.columns:
            df = df.drop(['waferIndex'], axis=1)
            print("  - 'waferIndex' 컬럼 제거됨")

        if 'failureType' not in df.columns:
            print("[오류]: 'failureType' 컬럼이 데이터에 없습니다.")
            print("  - 컬럼 목록:", df.columns)
            return

        # 데이터 타입 확인 및 처리
        sample_val = df['failureType'].iloc[0]
        print(f"\n[정보]: 'failureType' 데이터 샘플 (첫번째): {sample_val} (Type: {type(sample_val)})")

        # 만약 리스트 형태라면 k-means01.py 처럼 처리 시도 (현재는 문자열로 확인됨)
        # 필요시 주석 해제하여 사용
        # df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if isinstance(x, (list, np.ndarray)) and len(x)>0 else x)

        print("\n[결과]: Failure Type 별 데이터 분포:")
        print("="*40)
        value_counts = df['failureType'].value_counts()
        print(value_counts)
        print("="*40)

        print(f"\n[요약]: 총 {len(value_counts)}개의 분류 유형이 존재합니다.")
        print(f"  - 분류 목록: {value_counts.index.tolist()}")

    except Exception as e:
        print(f"[오류]: 데이터 분석 중 예외 발생: {e}")

if __name__ == "__main__":
    review_failure_types()
