import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# 1. 데이터 로드 (절대 경로 변환으로 오류 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(current_dir, '../models/ft_100.pkl')

if not os.path.exists(pkl_path):
    print(f"[Error] File not found: {pkl_path}")
    exit(1)

df = pd.read_pickle(pkl_path)

# 보고 싶은 데이터의 인덱스
idx = 61

# 2. 데이터 시각화
wafer_array = df.iloc[idx]['waferMap']
label = df.iloc[idx]['failureType']

plt.figure(figsize=(5, 5))
plt.imshow(wafer_array, cmap='inferno')
# 타이틀, 축, 컬러바 제거 -> 순수 이미지만 저장
plt.axis('off')

# 3. 이미지 저장
# 저장 경로: py-flask/static/image
save_dir = os.path.join(current_dir, '../py-flask/static/image')
os.makedirs(save_dir, exist_ok=True)

# [수정] f-string 적용
file_name = f'wafer_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
save_path = os.path.join(save_dir, file_name)

# 여백 없이 꽉 채워서 저장
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
print(f"Image saved to: {save_path}")

plt.show()