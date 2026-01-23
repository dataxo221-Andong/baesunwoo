import os
import glob
import numpy as np
import cv2
import random

# 설정
INPUT_DIR = 'chip1step'
GRID_SIZE = (32, 32)

# ==========================================
# 1. 패턴 생성 함수 (오류 수정 및 최적화)
# ==========================================

def gen_none(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    
    # [Fix] 튜플 연산 에러 수정 (shape[1]-1 -> w-1)
    num_noise = random.randint(0, 3)
    for _ in range(num_noise):
        rx, ry = random.randint(0, w-1), random.randint(0, h-1)
        matrix[ry, rx] = 1
    return matrix

def gen_center(shape):
    """Center: 중앙 밀집 (내부를 꽉 채우도록 수정)"""
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cx = (w // 2) + random.randint(-2, 2)
    cy = (h // 2) + random.randint(-2, 2)
    radius = random.uniform(6, 10)
    
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = (x_idx - cx)**2 + (y_idx - cy)**2 <= radius**2
    
    # [수정] 0.8~0.95 -> 0.95~1.0 (거의 구멍 없이 꽉 차게)
    # 32x32에서는 구멍이 조금만 있어도 모양이 망가짐
    prob_mask = np.random.rand(h, w) < random.uniform(0.95, 1.0)
    matrix[mask & prob_mask] = 1
    return matrix

def gen_donut(shape):
    """Donut: 도넛 형태 (고리가 끊어지지 않게 수정)"""
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cx, cy = w // 2, h // 2
    r_in = random.uniform(4, 6)
    r_out = random.uniform(11, 14)
    
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_sq = (x_idx - cx)**2 + (y_idx - cy)**2
    mask = (dist_sq >= r_in**2) & (dist_sq <= r_out**2)
    
    # [수정] 0.7~0.9 -> 0.9~1.0 (고리가 선명하게 이어지도록)
    prob_mask = np.random.rand(h, w) < random.uniform(0.9, 1.0)
    matrix[mask & prob_mask] = 1
    return matrix

def gen_edge_ring(shape):
    """Edge-Ring: 테두리 띠 (너무 자주 끊기지 않게 수정)"""
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    thickness = random.randint(2, 4)
    
    matrix[:thickness, :] = 1
    matrix[-thickness:, :] = 1
    matrix[:, :thickness] = 1
    matrix[:, -thickness:] = 1
    
    # [수정] 15% -> 5% 미만 (가끔만 끊기게) 또는 아예 제거
    # 링은 연결성이 생명이므로 노이즈를 최소화
    gap_prob = np.random.rand(h, w)
    matrix[gap_prob < 0.03] = 0 
    return matrix

def gen_edge_loc(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    side = random.choice(['top', 'bottom', 'left', 'right'])
    cluster_r = random.uniform(5, 9)
    if side == 'top': cx, cy = random.randint(5, w-5), 0
    elif side == 'bottom': cx, cy = random.randint(5, w-5), h
    elif side == 'left': cx, cy = 0, random.randint(5, h-5)
    else: cx, cy = w, random.randint(5, h-5)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = (x_idx - cx)**2 + (y_idx - cy)**2 <= cluster_r**2
    matrix[mask] = 1
    return matrix

def gen_loc(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cx = random.randint(8, w-8)
    cy = random.randint(8, h-8)
    radius = random.uniform(3, 6)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = (x_idx - cx)**2 + (y_idx - cy)**2 <= radius**2
    matrix[mask] = 1
    return matrix

def gen_scratch(shape):
    """Scratch: 선형 패턴 (유지하되 노이즈 살짝 감소)"""
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    num_lines = random.randint(1, 2)
    for _ in range(num_lines):
        p1 = (random.randint(0, w), random.randint(0, h))
        p2 = (random.randint(0, w), random.randint(0, h))
        thickness = random.randint(1, 2)
        cv2.line(matrix, p1, p2, 1, thickness)
    
    # [수정] 10% -> 5% (선이 너무 점선처럼 되지 않게)
    noise = np.random.rand(h, w)
    matrix[noise < 0.05] = 0
    return matrix

def gen_random(shape):
    h, w = shape
    density = random.uniform(0.1, 0.3)
    # [Fix] 모집단을 [0, 1]로 설정하여 ValueError 해결
    matrix = np.random.choice([0, 1], size=shape, p=[1-density, density])
    return matrix.astype(np.uint8)

def gen_near_full(shape):
    h, w = shape
    density = random.uniform(0.7, 0.9)
    # [Fix] 모집단을 [0, 1]로 설정하여 ValueError 해결
    matrix = np.random.choice([0, 1], size=shape, p=[1-density, density])
    return matrix.astype(np.uint8)

# 매핑 테이블
PATTERN_MAPPING = {
    'Center': gen_center, 'Donut': gen_donut, 'Edge-Ring': gen_edge_ring,
    'Edge-Loc': gen_edge_loc, 'Loc': gen_loc, 'Scratch': gen_scratch,
    'Random': gen_random, 'Near-full': gen_near_full,
    'none': gen_none, 'None': gen_none,
}

# ==========================================
# 3. 메인 로직 (안전한 덮어쓰기)
# ==========================================

def insert_tsv_data():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    # [수정] 출력용 폴더 생성 (chipfin)
    output_dir = 'chipfin'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = glob.glob(os.path.join(INPUT_DIR, "*.npy"))
    print(f"Found {len(file_list)} chips. Start injecting patterns...")
    
    count = 0
    
    for fpath in file_list:
        try:
            # 1. 원본 데이터 전체 로드 (기존 키: waferIndex, failureType, die_status 등 보존됨)
            chip_data = np.load(fpath, allow_pickle=True).item()
            
            # 2. 라벨 파싱 (문자열 깨짐 방지)
            f_type = chip_data.get('failureType', 'None')
            
            # 무한 중첩 대응
            temp = chip_data.get('failureType', 'None')
            while isinstance(temp, (list, np.ndarray)):
                if len(temp) > 0:
                    temp = temp[0] if isinstance(temp, list) else temp.item()
                else:
                    temp = 'None'
                    break
            f_type = str(temp).strip()

            # 3. 패턴 생성
            target_matrix = None
            status = chip_data.get('die_status', 0)
            
            if status == 1: # 양품
                target_matrix = gen_none(GRID_SIZE)
            elif status == 2: # 불량
                generator = PATTERN_MAPPING.get(f_type)
                if generator:
                    target_matrix = generator(GRID_SIZE)
                else:
                    # 매핑 안 된 라벨은 Random 처리
                    target_matrix = gen_random(GRID_SIZE)
            
            # 4. 데이터 주입 및 [수정] 새로운 경로에 저장
            if target_matrix is not None:
                # [핵심] 기존 딕셔너리에 'tsv_matrix' 키만 새로 추가/갱신
                chip_data['tsv_matrix'] = target_matrix
                
                # 파일명 유지
                file_name = os.path.basename(fpath)
                save_path = os.path.join(output_dir, file_name)
                
                # [수정] chipfin 폴더에 저장
                np.save(save_path, chip_data)
                count += 1
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    print(f"Successfully processed {count} chips. Saved to '{output_dir}/'.")

if __name__ == "__main__":
    insert_tsv_data()