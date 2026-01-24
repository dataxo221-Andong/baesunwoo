import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import datetime
import random
import base64
from torchvision import models
import torch.nn.functional as F

# ==========================================
# 1. 모델 설정
# ==========================================

class WaferClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(WaferClassifier, self).__init__()
        try:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            self.model = models.resnet18(pretrained=True)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

LABELS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
# None 관련 변형들 (안전한 처리를 위한 매핑용)
NONE_VARIANTS = {'none', 'None', 'NONE', '', '[]'}

# ==========================================
# 2. 유틸리티 함수
# ==========================================

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaferClassifier()
    
    if not os.path.exists(model_path):
        print(f"[Warning] 모델 파일이 없습니다: {model_path}")
        return None

    try:
        loaded_obj = torch.load(model_path, map_location=device)
        if isinstance(loaded_obj, dict):
            state_dict = loaded_obj['state_dict'] if 'state_dict' in loaded_obj else loaded_obj
            first_key = next(iter(state_dict.keys()))
            if not first_key.startswith('model.') and hasattr(model, 'model'):
                model.model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
        else:
            model = loaded_obj
    except Exception as e:
        print(f"[Error] 모델 로드 실패: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

def predict_failure_type(model, image):
    """
    이미지를 모델에 넣어 Failure Type과 Confidence(확신도)를 예측합니다.
    """
    if model is None:
        return "Unknown", 0.0

    img_resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_NEAREST)
    
    if len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    img_float = img_resized.astype(np.float32) / 255.0
    tensor = torch.tensor(img_float).unsqueeze(0).unsqueeze(0).float() # (1, 1, H, W)
    
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        # Softmax로 확률 계산
        probs = F.softmax(outputs, dim=1)
        max_prob, predicted_idx = torch.max(probs, 1)
        label_idx = predicted_idx.item()
        confidence = max_prob.item() * 100.0 # 퍼센트 변환
        
    predicted_label = "Unknown"
    if 0 <= label_idx < len(LABELS):
        predicted_label = LABELS[label_idx]

    # None 변형 처리 (표준화)
    if predicted_label in NONE_VARIANTS:
        predicted_label = "None"
        
    return predicted_label, confidence

def determine_grade(density):
    # 임의의 등급 기준 (필요시 조정)
    if density < 0.05: return "A"
    elif density < 0.10: return "B"
    elif density < 0.15: return "C"
    elif density < 0.20: return "D"
    else: return "F"

def numpy_to_base64(img_array):
    """OpenCV 이미지를 Base64 문자열로 변환"""
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# ==========================================
# 3. 메인 로직: 이미지 -> 웨이퍼 정보 + 칩 데이터 생성
# ==========================================

def process_wafer_image(image_path, output_dir='chip_upload'):
    """
    업로드된 웨이퍼 이미지를 받아 웨이퍼 메타데이터와 칩 데이터를 생성합니다.
    """
    # 설정
    model_path = os.path.join(os.path.dirname(__file__), '../models/wafer_classifier.pth')
    target_size = (32, 32)
    chip_grid_size = (32, 32)
    die_size = float(target_size[0] * target_size[1]) # Die Size Calculation
    train_test_label = [] # Default label list
    
    if not os.path.exists(image_path):
        print(f"[Error] 이미지를 찾을 수 없습니다: {image_path}")
        return None, None

    original_img = cv2.imread(image_path)
    if original_img is None:
        print("[Error] 이미지 파일을 읽을 수 없습니다.")
        return None, None

    # 1. Failure Type & Confidence 예측
    model = load_model(model_path)
    failure_type, confidence = predict_failure_type(model, original_img)
    print(f"[Result] 예측: {failure_type} ({confidence:.2f}%)")

    # 2. 웨이퍼 맵 변환 및 통계 계산
    gray_map = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
    resized_map = cv2.resize(gray_map.astype('float32'), target_size, interpolation=cv2.INTER_NEAREST)
    resized_map = resized_map.astype(int)
    
    # 픽셀 매핑: 0(배경), 1(정상), 2(불량)
    quantized_map = np.zeros_like(resized_map)
    quantized_map[resized_map > 80] = 1
    quantized_map[resized_map > 170] = 2
    
    # 통계 계산
    rows, cols = quantized_map.shape
    total_pixels = rows * cols # 32x32 = 1024
    
    # 실제 칩 영역으로 간주되는 픽셀 수 (1 또는 2)
    # 배경(0)을 제외한 것을 전체 칩 개수로 볼지 여부는 정책에 따라 결정
    # 여기서는 0이 아닌 것을 유효 칩으로 가정
    valid_mask = quantized_map > 0
    die_count = np.sum(valid_mask)
    defect_count = np.sum(quantized_map == 2)
    
    defect_density = 0.0
    if die_count > 0:
        defect_density = defect_count / die_count
        
    total_grade = determine_grade(defect_density)
    
    # Lot Name 생성
    now = datetime.datetime.now()
    import string
    chars = string.ascii_uppercase + string.digits
    random_serial = ''.join(random.choices(chars, k=6))
    lot_name = f"{now.strftime('%y%m%d')}{random_serial}{now.strftime('%H%M')}"
    
    # Wafer Map 이미지 Base64 인코딩 (시각화된 quantized map 또는 원본 resized)
    # 여기서는 시각화를 위해 원본(리사이즈된)이나 quantized map을 사용
    # quantized map을 시각화(0:검정, 1:회색, 2:흰색)하여 저장
    vis_map = np.zeros((rows, cols), dtype=np.uint8)
    vis_map[quantized_map == 1] = 127
    vis_map[quantized_map == 2] = 255
    wafer_map_b64 = numpy_to_base64(vis_map)

    # ==========================================
    # (A) Wafer Data 구조 생성 (부모 DB용)
    # ==========================================
    wafer_data = {
        "lotName": lot_name,
        "waferMap": wafer_map_b64,
        "waferSize": f"{original_img.shape[1]}x{original_img.shape[0]}", # Width x Height
        "failureType": failure_type,
        "confidence": float(confidence),
        "dieCount": int(die_count),
        "defectCount": int(defect_count),
        "defectDensity": float(defect_density),
        "totalGrade": total_grade,
        "created_at": now.isoformat()
    }

    # ==========================================
    # (B) Chip Data 파일 생성 (자식 DB용)
    # ==========================================
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 칩 추출 및 저장 (샘플링)
    # wf_insert_chip.py의 로직 복원: Normal/Fail 각각 25개씩 추출 시도
    
    coords = [(y, x) for y in range(rows) for x in range(cols) if quantized_map[y, x] > 0]
    random.shuffle(coords)
    
    saved_count = 0
    count_normal = 0
    count_failure = 0
    TARGET_PER_TYPE = 25
    
    chip_files_created = []

    for y, x in coords:
        if count_normal >= TARGET_PER_TYPE and count_failure >= TARGET_PER_TYPE:
            break
            
        tsv_status = quantized_map[y, x] # 1(정상) or 2(불량)
        is_target = False
        
        if tsv_status == 1:
            if count_normal < TARGET_PER_TYPE:
                count_normal += 1
                is_target = True
        elif tsv_status == 2:
            if count_failure < TARGET_PER_TYPE:
                count_failure += 1
                is_target = True

        if is_target:
            # 칩 데이터 구조 (메모 반영 + 기존 항목 복원)
            chip_info = {
                "chip_uid": f"{lot_name}X{x}Y{y}D{tsv_status}",
                "lotName": lot_name,            
                "tsv_matrix": np.zeros(chip_grid_size, dtype=int), # Placeholder
                "tsv_coordinate": (x, y),
                "tsv_status": int(tsv_status),
                "dieSize": die_size,
                "trainTestLabel": train_test_label,
                "created_at": now.isoformat()
            }
            
            file_name = f"{lot_name}X{x}Y{y}D{tsv_status}.npy"
            file_path = os.path.join(output_dir, file_name)
            np.save(file_path, chip_info)
            chip_files_created.append(file_name)
            saved_count += 1

    print(f"[Done] '{lot_name}' 웨이퍼 처리 완료.")
    print(f"       웨이퍼 등급: {total_grade} (불량률 {defect_density*100:.1f}%)")
    print(f"       생성된 칩 파일: {saved_count}개 (Normal:{count_normal}, Fail:{count_failure})")
    
    return wafer_data, chip_files_created

if __name__ == "__main__":
    print("--- Wafer & Chip Data Generation (Normalized) ---")
    # 테스트 시 주석 해제
    # process_wafer_image("sample_wafer.png")
