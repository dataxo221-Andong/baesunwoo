import numpy as np
import os

def analyze_chip_file():
    # 분석할 파일 경로
    file_path = 'chip1step/lot36838X1Y17D2.npy'

    if not os.path.exists(file_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
        print("경로가 정확한지, 혹은 wf-to-chip.py를 실행하여 파일을 생성했는지 확인해주세요.")
        return

    print(f"Loading {file_path}...")
    try:
        # 딕셔너리 형태로 저장했으므로 item()을 호출하여 가져옵니다.
        # allow_pickle=True는 객체(딕셔너리) 로딩을 위해 필수입니다.
        chip_data = np.load(file_path, allow_pickle=True).item()
        
        print("\n" + "="*50)
        print(" [ Chip Data Structure ]")
        print("="*50)
        
        # 키 목록 확인
        print(f"Keys: {list(chip_data.keys())}")
        print("-" * 50)
        
        # 각 항목 상세 출력
        for key, value in chip_data.items():
            if key == 'matrix':
                print(f"[{key}]")
                print(f"  - Shape: {value.shape}")
                print(f"  - Dtype: {value.dtype}")
                # 내용이 모두 0인지 확인 (초기화 상태 점검)
                print(f"  - Sum: {np.sum(value)}")
                print(f"  - Preview:\n{value}")
            else:
                print(f"[{key}]: {value}")
                
    except Exception as e:
        print(f"[오류] 파일 로드 중 에러 발생: {e}")

if __name__ == "__main__":
    analyze_chip_file()
