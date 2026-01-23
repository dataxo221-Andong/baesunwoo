import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import datetime
import random
from torchvision import models

# ==========================================
# 1. 모델 설정 (사용자 환경에 맞게 수정 필요)
# ==========================================

class WaferClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(WaferClassifier, self).__init__()
        # ResNet18을 불러오고 미리 학습된 가중치를 사용합니다.
        # weights=models.ResNet18_Weights.IMAGENET1K_V1 또는 pretrained=True (버전에 따름)
        try:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            # 구버전 torch 호환성
            self.model = models.resnet18(pretrained=True)

        # 입력 채널을 3 (RGB)에서 1 (GrayScale)로 변경
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 최종 분류 레이어를 우리의 클래스 수에 맞게 변경 (9개 클래스)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# WM-811K 데이터셋 라벨 (학습 시 사용된 라벨 순서와 대소문자가 정확해야 함)
LABELS = ['CENTER', 'DONUT', 'EDGE-LOC', 'EDGE-RING', 'LOC', 'NEAR-FULL', 'RANDOM', 'SCRATCH', 'NONE']

# ==========================================
# 2. 유틸리티 함수
# ==========================================

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaferClassifier() # [주의] 실제 모델 클래스로 변경 필요
    
    if not os.path.exists(model_path):
        print(f"[Warning] 모델 파일이 없습니다: {model_path}")
        return None

    try:
        # 1. 전체 모델 로드 시도
        model = torch.load(model_path, map_location=device)
        print("[Info] 전체 모델 로드 성공")
    except:
        try:
            # 2. state_dict 로드 시도
            ckpt = torch.load(model_path, map_location=device)
            # 만약 ckpt가 딕셔너리고 'state_dict' 키를 가진다면
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            print("[Info] State dict 로드 성공")
        except Exception as e:
            print(f"[Error] 모델 로드 실패: {e}")
            print("  -> 모델 클래스 정의가 pth 파일과 일치하지 않을 수 있습니다.")
            return None
            
    model.to(device)
    model.eval()
    return model

def predict_failure_type(model, image):
    """
    이미지를 모델에 넣어 Failure Type을 예측합니다.
    """
    if model is None:
        return "Unknown"

    # 전처리: 224x224 리사이즈, 정규화, 텐서 변환
    # (실제 학습 시 사용한 전처리와 동일해야 함)
    img_resized = cv2.resize(image, (224, 224))
    
    # 만약 모델이 Grayscale 입력을 받는다면 채널 변환 필요
    # 모델 정의에서 1채널(Grayscale) 입력을 받도록 설정했으므로 그에 맞춤
    if len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    img_float = img_resized.astype(np.float32) / 255.0
    # Grayscale이면 (H, W) -> (1, H, W) 로 차원 추가
    # Transpose 대신 차원 확장 사용
    tensor = torch.tensor(img_float).unsqueeze(0).unsqueeze(0).float() # (1, 1, H, W)
    
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_idx = torch.max(outputs, 1)
        label_idx = predicted_idx.item()
        
    if 0 <= label_idx < len(LABELS):
        return LABELS[label_idx]
    else:
        return "Unknown"

# ==========================================
# 3. 메인 로직: 이미지 -> 칩 변환
# ==========================================

def process_wafer_image(image_path, output_dir='chip_upload'):
    """
    업로드된 웨이퍼 이미지를 받아 분류 후 칩 데이터로 분할 저장합니다.
    """
    # 설정
    model_path = os.path.join(os.path.dirname(__file__), '../models/wafer_classifier.pth') # 상대 경로
    target_size = (32, 32)
    chip_grid_size = (32, 32)
    
    # 1. 이미지 로드
    if not os.path.exists(image_path):
        print(f"[Error] 이미지를 찾을 수 없습니다: {image_path}")
        return

    original_img = cv2.imread(image_path)
    if original_img is None:
        print("[Error] 이미지 파일을 읽을 수 없습니다.")
        return

    print(f"[Info] 이미지 로드됨: {image_path} {original_img.shape}")

    # 2. Failure Type 분류
    model = load_model(model_path)
    failure_type = predict_failure_type(model, original_img)
    print(f"[Result] 예측된 불량 유형: {failure_type}")

    # 3. 웨이퍼 맵 변환 (이미지 -> 0,1,2 맵)
    # 이미지를 Grayscale로 변환
    gray_map = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
    
    # 32x32로 리사이즈 (Nearest Neighbor로 데이터 변형 방지)
    resized_map = cv2.resize(gray_map.astype('float32'), target_size, interpolation=cv2.INTER_NEAREST)
    resized_map = resized_map.astype(int)
    
    # [중요] 픽셀 값을 0(배경), 1(정상), 2(불량)으로 매핑
    # 업로드된 이미지가 이미 0,1,2 값을 가지고 있다면 그대로 쓰지만,
    # 시각화된 이미지(RGB)라면 픽셀 밝기 구간으로 나눔 (단순 예시)
    # 0~80: 배경(0), 81~170: 정상(1), 171~255: 불량(2)
    # 실제로는 색상값 분석이 필요할 수 있음.
    quantized_map = np.zeros_like(resized_map)
    quantized_map[resized_map > 80] = 1
    quantized_map[resized_map > 170] = 2
    # 배경(0)은 0으로 유지
    
    # 4. 칩 저장 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 5. 메타데이터 생성 (업로드된 파일이라 정보가 없으므로 생성)
    lot_name = f"Upload_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    die_size = float(target_size[0] * target_size[1]) # 임의값
    train_test_label = [] # 정보 없음

    # 6. wf-to-chip 로직 수행 (칩 분할 저장)
    rows, cols = quantized_map.shape
    saved_count = 0
    
    # 상태별 수집 목표 (Upload된건 보통 테스트용이니 25개 제한 없이 다 뽑거나, 제한을 둠)
    # 여기서는 wf-to-chip 처럼 25개 제한을 둡시다.
    count_normal = 0
    count_failure = 0
    TARGET_PER_TYPE = 25
    
    coords = [(y, x) for y in range(rows) for x in range(cols)]
    random.shuffle(coords)
    
    for y, x in coords:
        if count_normal >= TARGET_PER_TYPE and count_failure >= TARGET_PER_TYPE:
            break
            
        die_status = quantized_map[y, x]
        is_target = False
        
        if die_status == 1:
            if count_normal < TARGET_PER_TYPE:
                count_normal += 1
                is_target = True
        elif die_status == 2:
            if count_failure < TARGET_PER_TYPE:
                count_failure += 1
                is_target = True
                
        if is_target:
            chip_data = {
                "chip_uid": f"{lot_name}X{x}Y{y}D{die_status}",
                "tsv_matrix": np.zeros(chip_grid_size, dtype=int),
                "lotName": lot_name,
                "failureType": failure_type, # [핵심] 예측된 라벨 사용
                "dieSize": die_size,
                "trianTestLabel": train_test_label,
                "tsv_coordinate": (x, y),
                "die_status": int(die_status)
            }
            
            file_name = f"{lot_name}X{x}Y{y}D{die_status}.npy"
            file_path = os.path.join(output_dir, file_name)
            np.save(file_path, chip_data)
            saved_count += 1

    print(f"[Done] '{lot_name}' 처리 완료. {saved_count}개 칩 생성됨 (Normal:{count_normal}, Fail:{count_failure})")
    print(f"       저장 경로: {output_dir}")
    return failure_type, saved_count

if __name__ == "__main__":
    # 테스트용 코드
    # sample.png가 있다고 가정하고 실행
    print("--- Wafer Image Processing Test ---")
    # process_wafer_image("sample_wafer.png")
