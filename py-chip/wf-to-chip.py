import pandas as pd
import numpy as np
import cv2
import os

def process_all_wafers():
    # 설정
    input_path = 'models/ft_100.pkl'
    output_dir = 'chip1step'
    target_size = (32, 32) # 웨이퍼 리사이즈 크기
    chip_grid_size = (32, 32) # 각 칩 내부의 TSV 격자 크기
    
    # [설정] 샘플링 설정
    NUM_SAMPLES_PER_CHUNK = 10   # 각 100개 단위(청크)에서 앞에서부터 몇 개를 처리할지 (N)
    CHUNK_SIZE = 100            # 건너뛸 단위 (100)

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

    if len(df) == 0:
        print("Empty dataframe.")
        return

    if len(df) == 0:
        print("Empty dataframe.")
        return

    # 타겟 인덱스 리스트 생성 (0~4, 100~104, 200~204, ...)
    target_indices = []
    total_wafers = len(df)
    
    for start in range(0, total_wafers, CHUNK_SIZE):
        end = min(start + NUM_SAMPLES_PER_CHUNK, total_wafers)
        # 100개 단위 시작점에서 N개만큼 인덱스 추가
        chunk_indices = list(range(start, end))
        target_indices.extend(chunk_indices)

    print(f"Processing {len(target_indices)} wafers based on sampling rule (first {NUM_SAMPLES_PER_CHUNK} of every {CHUNK_SIZE})...")
    # print(f"Indices: {target_indices}") # 디버깅용
    
    total_chips = 0

    # 타겟 인덱스에 대해서만 반복 처리
    for row_idx in target_indices:
        row = df.iloc[row_idx]
        
        # 웨이퍼 맵 가져오기
        original_map = row['waferMap']
        
        # 메타데이터 추출 및 정제
        failure_type = row['failureType'] 
        lot_name = row['lotName']
        die_size = row['dieSize']
        train_test_label = row['trianTestLabel']
        
        # failureType 정제
        if isinstance(failure_type, (list, np.ndarray)):
            if len(failure_type) == 0:
                failure_type = 'None'
            else:
                failure_type = str(failure_type[0][0]) if (isinstance(failure_type[0], (list, np.ndarray))) else str(failure_type[0])
        
        failure_type = str(failure_type).strip()
        
        # 1. 32x32로 리사이즈
        resized_map = cv2.resize(original_map.astype('float32'), target_size, interpolation=cv2.INTER_NEAREST)
        resized_map = resized_map.astype(int)

        # 2. 각 픽셀(다이) 순회
        saved_count = 0
        rows, cols = resized_map.shape
        
        # [수정] 상태별 수집 카운터 및 목표 개수 설정 (각 25개)
        count_normal = 0    # 양품(1)
        count_failure = 0   # 불량(2)
        TARGET_PER_TYPE = 25 
        
        # 좌표 리스트 생성 후 셔플 (랜덤 샘플링)
        coords = [(y, x) for y in range(rows) for x in range(cols)]
        import random
        random.shuffle(coords)

        for y, x in coords:
            # 목표 달성 시 조기 종료
            if count_normal >= TARGET_PER_TYPE and count_failure >= TARGET_PER_TYPE:
                break

            die_status = resized_map[y, x]
            is_target = False
            
            if die_status == 1:
                if count_normal < TARGET_PER_TYPE:
                    count_normal += 1
                    is_target = True
            elif die_status == 2:
                if count_failure < TARGET_PER_TYPE:
                    count_failure += 1
                    is_target = True
            
            # 타겟으로 선정된 경우에만 저장
            if is_target:
                chip_data = {
                    "chip_uid": f"{lot_name}X{x}Y{y}D{die_status}",
                    "tsv_matrix": np.zeros(chip_grid_size, dtype=int),
                    "lotName": lot_name,
                    "failureType": failure_type,
                    "dieSize": die_size,
                    "trianTestLabel": train_test_label,
                    "tsv_coordinate": (x, y),
                    "die_status": int(die_status) # 1 or 2
                }
                
                # 파일명 생성 규칙: [lotName]_[x좌표]_[y좌표]_[상태]
                file_name = f"{lot_name}X{x}Y{y}D{die_status}.npy"
                file_path = os.path.join(output_dir, file_name)
                
                # 저장 (딕셔너리 객체 저장)
                np.save(file_path, chip_data)
                saved_count += 1
        
        total_chips += saved_count
        # 진행 상황 출력
        print(f"[{row_idx+1}/{len(df)}] Lot: {lot_name} ({failure_type}) -> {saved_count} chips (Normal:{count_normal}, Fail:{count_failure})")

    print(f"\nAll Done. Total Generated Chips: {total_chips} in '{output_dir}/'")

if __name__ == "__main__":
    process_all_wafers()
