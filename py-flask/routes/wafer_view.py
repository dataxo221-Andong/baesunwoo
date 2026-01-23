from flask import Blueprint, render_template, request, jsonify, current_app
import os
import sys

# 상위 모듈 import를 위한 경로 설정 (wf_insert_chip 사용)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # hbm-project
py_chip_path = os.path.join(project_root, 'py-chip')
if py_chip_path not in sys.path:
    sys.path.append(py_chip_path)

# 이제 파일명이 wf_insert_chip.py 이므로 바로 import 가능
from wf_insert_chip import process_wafer_image, load_model, predict_failure_type

bp = Blueprint('wafer', __name__, url_prefix='/wafer')

@bp.route('/classify', methods=['GET', 'POST'])
def classify():
    """
    웨이퍼 이미지를 업로드하고 불량 패턴을 분류하는 페이지
    """
    if request.method == 'POST':
        if 'wafer_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['wafer_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            # 1. 파일 임시 저장
            upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            
            # 2. 모델 로드 및 예측 (wf_insert_chip 모듈 활용)
            try:
                # 모델 경로 (상대 경로 주의)
                model_path = os.path.join(project_root, 'models', 'wafer_classifier.pth')
                model = load_model(model_path)
                
                # 이미지 읽기 (OpenCV)
                import cv2
                img = cv2.imread(file_path)
                
                prediction = predict_failure_type(model, img)
                
                return jsonify({
                    'status': 'success',
                    'prediction': prediction,
                    'image_url': f"/static/uploads/{file.filename}"
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})

    return render_template('wafer_classify.html')

@bp.route('/info', methods=['GET', 'POST'])
def info():
    """
    웨이퍼 이미지를 업로드하고 상세 정보를 시각화하는 페이지
    """
    # [TODO] 상세 정보 로직 구현 (단순 분류 외에 픽셀 분포 등 분석)
    return render_template('wafer_info.html')
