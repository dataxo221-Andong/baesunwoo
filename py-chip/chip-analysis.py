import numpy as np
import os

def analyze_chip_file():
    # 분석할 파일 경로
    file_path = 'chipfin/lot17137X15Y12D2.npy'

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
        # 시각화 함수
        def visualize_matrix(mat):
            h, w = mat.shape
            print(f"  - Shape: {mat.shape}, Sum: {np.sum(mat)}")
            print("  - Visualization:")
            print("   " + "".join([str(i%10) for i in range(w)]))
            print("   " + "-" * w)
            for r in range(h):
                row_str = ""
                for c in range(w):
                    val = mat[r, c]
                    # 값이 0이면 공백/점, 1이면 특수문자
                    char = "■" if val > 0 else "."
                    row_str += char
                print(f"{r:2d}|{row_str}|")
            print("   " + "-" * w)

        # 각 항목 상세 출력
        for key, value in chip_data.items():
            # 행렬 데이터인 경우 시각화
            if isinstance(value, np.ndarray) and value.ndim == 2:
                print(f"[{key}]")
                visualize_matrix(value)
            else:
                print(f"[{key}]: {value}")
                
    except Exception as e:
        print(f"[오류] 파일 로드 중 에러 발생: {e}")

if __name__ == "__main__":
    analyze_chip_file()
