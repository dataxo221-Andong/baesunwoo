import matplotlib.pyplot as plt
import pandas as pd

# 데이터프레임 로드
df = pd.read_pickle('models/LSWMD.pkl')

# 보고 싶은 데이터의 인덱스 (예: 100번째 데이터)
idx = 460

# 1. 데이터 꺼내기 (숫자 행렬)
wafer_array = df.iloc[idx]['waferMap']
label = df.iloc[idx]['failureType']

# 2. 그림 그리기
plt.figure(figsize=(5, 5))
plt.imshow(wafer_array, cmap='inferno') # cmap은 색상 테마 (viridis, gray 등 변경 가능)
plt.title(f"Label: {label}")
plt.colorbar()

# 웨이퍼 시각화를 이미지로 저장
plt.savefig('image/wafer_image.png')
plt.show()